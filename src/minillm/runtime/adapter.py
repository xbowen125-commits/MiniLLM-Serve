"""Legacy decode-state view backed by a single RuntimeRequestContext."""

import time
from collections.abc import Mapping

from ..contracts import DecodeStatus, ExecutionPhase, FinishReason
from ..engine.request import RequestStatus
from .context import RuntimeRequestContext


class ContextStateViews(Mapping):
    """Live legacy mapping over the coordinator's context registry."""

    def __init__(self, contexts):
        self.contexts = contexts

    def __getitem__(self, key):
        return DecodeStateAdapter(self.contexts[key])

    def __iter__(self):
        return iter(self.contexts)

    def __len__(self):
        return len(self.contexts)


class DecodeStateAdapter:
    """Expose legacy observations without storing lifecycle or token copies."""

    def __init__(self, context: RuntimeRequestContext) -> None:
        self.context = context
        context.observations.setdefault("phase_history", [ExecutionPhase.WAITING])
        context.observations.setdefault("phase_updated_time", context.request.created_time)
        context.observations.setdefault("no_progress_steps", 0)
        context.observations.setdefault("max_no_progress_steps", 10)
        context.observations.setdefault("finish_reason", None)
        context.observations.setdefault("error", None)

    def __getattr__(self, name):
        request = self.context.request
        if name in {"request_id", "created_time", "started_time", "finished_time", "generated_tokens"}:
            return getattr(request, name)
        if name == "max_tokens":
            return request.generation_config.max_tokens
        if name == "num_generated_tokens":
            return len(request.generated_tokens)
        if name in {"decode_steps", "first_token_time", "last_token_time", "prefill_tokens", "decode_tokens"}:
            return getattr(self.context.decode_state, name)
        if name in self.context.observations:
            return self.context.observations[name]
        raise AttributeError(name)

    @property
    def status(self):
        status = self.context.request.status
        return DecodeStatus.ACTIVE if status in {RequestStatus.WAITING, RequestStatus.RUNNING} else DecodeStatus(status.value)

    @property
    def is_terminal(self):
        return self.context.request.is_terminal

    @property
    def phase(self):
        return self.context.decode_state.execution_phase

    def can_prefill(self):
        return not self.is_terminal and self.phase is ExecutionPhase.WAITING

    def can_decode(self):
        return not self.is_terminal and self.phase in {ExecutionPhase.PREFILL, ExecutionPhase.DECODE}

    def start(self, *, now=None):
        if self.context.request.status is RequestStatus.WAITING:
            self.context.start(now=now)

    def transition_phase(self, phase, *, now=None):
        previous = self.phase
        self.context.decode_state.transition_phase(phase)
        if previous is not phase:
            self.context.observations["phase_history"].append(phase)
            self.context.observations["phase_updated_time"] = time.time() if now is None else now

    def record_step(self):
        self.context.decode_state.decode_steps += 1

    def append_token(self, token_id, *, now=None):
        """Append once, or observe a token already written by BatchExecutor."""
        timestamp = time.time() if now is None else now
        if self.phase is ExecutionPhase.PREFILL:
            self.transition_phase(ExecutionPhase.DECODE, now=timestamp)
        request = self.context.request
        if len(request.generated_tokens) == self.decode_tokens:
            request.append_token(token_id)
        elif len(request.generated_tokens) != self.decode_tokens + 1 or request.generated_tokens[-1] != token_id:
            raise RuntimeError("token observation does not match request token source")
        self.context.decode_state.record_token(now=timestamp)
        self.context.observations["no_progress_steps"] = 0
        if len(request.generated_tokens) >= self.max_tokens and not request.is_terminal:
            self.finish(FinishReason.LENGTH, now=timestamp)

    def _end(self, status, reason, *, now=None):
        request = self.context.request
        if not request.is_terminal:
            if status is RequestStatus.FINISHED:
                self.start(now=now)
            getattr(request, {RequestStatus.FINISHED: "mark_finished", RequestStatus.FAILED: "mark_failed", RequestStatus.CANCELLED: "mark_cancelled"}[status])(now=now)
        elif request.status is not status:
            raise RuntimeError("conflicting request terminal outcome")
        self.context.observations["finish_reason"] = reason
        self.transition_phase(ExecutionPhase.FINISHED, now=request.finished_time)

    def finish(self, reason=FinishReason.STOP, *, now=None):
        self._end(RequestStatus.FINISHED, reason, now=now)

    def fail(self, error, *, now=None):
        self._end(RequestStatus.FAILED, FinishReason.ERROR, now=now)
        self.context.observations["error"] = error

    def cancel(self, *, now=None):
        self._end(RequestStatus.CANCELLED, FinishReason.CANCELLED, now=now)

    def record_no_progress(self, *, now=None):
        self.context.observations["no_progress_steps"] += 1
        if self.no_progress_steps >= self.max_no_progress_steps:
            self.fail("decode made no progress", now=now)
