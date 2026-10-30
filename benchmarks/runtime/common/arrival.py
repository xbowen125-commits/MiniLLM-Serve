"""Shared immutable workload and virtual arrival gating, not a scheduler."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    prompt_tokens: int | None
    generation_tokens: int
    arrival_time: float

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be nonempty")
        if self.prompt_tokens is not None and (type(self.prompt_tokens) is not int or not 1 <= self.prompt_tokens <= 4096):
            raise ValueError("prompt_tokens must be positive or None for legacy unknown length")
        if type(self.generation_tokens) is not int or not 1 <= self.generation_tokens <= 1000:
            raise ValueError("generation_tokens must be in 1..1000")
        if isinstance(self.arrival_time, bool) or not math.isfinite(self.arrival_time) or self.arrival_time < 0:
            raise ValueError("arrival_time must be finite and nonnegative")

    @property
    def max_tokens(self):
        return self.generation_tokens

    @property
    def prompt_length(self):
        return self.prompt_tokens

    @property
    def generation_length(self):
        return self.generation_tokens


def normalize_requests(requests):
    """Adapt older requests without inventing missing prompt lengths."""
    workload = tuple(r if isinstance(r, WorkloadRequest) else WorkloadRequest(
        r.request_id, getattr(r, "prompt_length", None), r.max_tokens, r.arrival_time) for r in requests)
    if len({r.request_id for r in workload}) != len(workload):
        raise ValueError("duplicate workload IDs")
    return tuple(sorted(workload, key=lambda r: r.arrival_time))


def pop_due(pending: list[WorkloadRequest], now: float) -> list[WorkloadRequest]:
    """Remove only arrived items from a sorted pending list at a cycle boundary."""
    due = []
    while pending and pending[0].arrival_time <= now:
        due.append(pending.pop(0))
    return due
