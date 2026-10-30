"""Single serial owner for the experimental HTTP runtime.

Only this thread touches the tokenizer, facade and batch worker. The event loop
passes admitted RequestState objects and cancellation Events; completion Futures
are resolved only after synchronous GPU cleanup AND result acknowledgement.
Transport metadata is bounded by admission, and is not a second lifecycle store.
"""

from concurrent.futures import Future
from dataclasses import dataclass, replace
import logging
import queue
import threading
import time

from ..engine import GenerationResult, RequestState, RequestStatus
from ..contracts import FinishReason
from .facade import RuntimeFacade

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HTTPCompletion:
    result: GenerationResult | None
    outcome: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """Immutable transport copy; never retains a Context or KV reference.

    decoded_text is the full untrimmed generated prefix, decoded on the owner
    thread so tokenizer access remains serialized. The HTTP adapter decides
    which text is safe to send (Unicode/stop prefixes/append-only validation).
    """
    token_id: int
    ordinal: int
    decoded_text: str


class StreamHandle:
    """Bounded nonblocking output, plus out-of-band confirmed completion."""

    def __init__(self, capacity, cancel_event):
        self.events: queue.Queue[TokenEvent] = queue.Queue(maxsize=capacity)
        self.completion: Future[HTTPCompletion] = Future()
        self.cancel_event = cancel_event
        self._failure = None
        self._lock = threading.Lock()

    @property
    def failure(self):
        with self._lock:
            return self._failure

    def fail(self, code, message):
        with self._lock:
            if self._failure is None:
                self._failure = (code, message)
        self.cancel_event.set()

    def cancel(self):
        self.cancel_event.set()


@dataclass
class _Delivery:
    state: RequestState
    messages: list
    cancel: threading.Event
    deadline: float
    future: Future[HTTPCompletion]
    timed_out: bool = False
    stream: StreamHandle | None = None
    emitted_tokens: int = 0


class ContinuousHTTPOwner:
    """Own one model and worker, with cancellation only between facade steps.

    Backend release failures keep the request/future/lease outstanding and are
    retried at safe boundaries. A permanently refusing backend therefore prevents
    idle/shutdown; this is intentional, rather than falsely reporting cleanup.
    """

    def __init__(self, engine, settings, *, worker_factory=None):
        self.engine = engine
        self.settings = settings
        self.worker_factory = worker_factory
        self.ready = Future()
        self.stopped = Future()
        self._queue = queue.Queue(maxsize=settings.max_concurrent_generations)
        self._closing = threading.Event()
        self._status_lock = threading.Lock()
        self._status = {"ready": False, "pending": 0, "allocated_slots": 0}
        self._stream_overflows = 0
        self.thread = threading.Thread(target=self._run, name="minillm-continuous-owner", daemon=True)

    def start(self):
        self.thread.start()
        return self.ready

    def submit(self, state, messages, cancel_event, *, deadline=None, _stream=None):
        if self._closing.is_set() or not self.thread.is_alive():
            raise RuntimeError("continuous owner is not accepting requests")
        future = _stream.completion if _stream is not None else Future()
        self._queue.put_nowait(_Delivery(state, messages, cancel_event,
            deadline if deadline is not None else time.monotonic() + self.settings.inference_timeout_seconds,
            future, stream=_stream))
        return future

    def submit_stream(self, state, messages, cancel_event, *, deadline=None):
        handle = StreamHandle(self.settings.continuous_output_capacity, cancel_event)
        self.submit(state, messages, cancel_event, deadline=deadline, _stream=handle)
        return handle

    def close(self):
        self._closing.set()
        return self.stopped

    def status(self):
        with self._status_lock:
            return dict(self._status)

    def _publish(self, runtime, worker, pending):
        slots = worker.pool.snapshot()
        with self._status_lock:
            self._status = {"ready": not self._closing.is_set(), "pending": len(pending),
                **slots, **runtime.retention(),
                "runtime_active_requests": runtime.snapshot().active_requests,
                "pending_commands": self._queue.qsize(),
                "stream_overflows": self._stream_overflows,
                "peak_allocated_slots": max(self._status.get("peak_allocated_slots", 0), slots["allocated_slots"])}

    def _run(self):
        runtime = worker = None
        pending = {}
        try:
            self.engine.load()
            if self.worker_factory is None:
                from ..workers import BatchedTransformersWorker, LoadedModel
                worker = BatchedTransformersWorker(LoadedModel(self.engine.model,
                    self.engine.tokenizer, str(self.engine.device)),
                    slot_capacity=self.settings.continuous_batch_size)
            else:
                worker = self.worker_factory(self.engine)
            if self.settings.max_concurrent_generations > worker.pool.capacity:
                raise ValueError("admission exceeds Worker slot capacity")
            runtime = RuntimeFacade(worker, execution_mode="continuous",
                max_batch_size=self.settings.continuous_batch_size, owns_worker=True)
            self._publish(runtime, worker, pending)
            self.ready.set_result(None)
            while True:
                # Drain bounded transport submissions before building a plan.
                while True:
                    try:
                        delivery = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    state = delivery.state
                    pending[state.request_id] = delivery
                    try:
                        prompt = self.engine.build_prompt(delivery.messages)
                        count = len(self.engine.tokenizer.encode(prompt, add_special_tokens=True))
                        if count >= self.settings.max_model_len:
                            raise ValueError(f"prompt length {count} >= max_model_len={self.settings.max_model_len}")
                        config = replace(state.generation_config,
                            max_tokens=min(state.generation_config.max_tokens,
                                           self.settings.max_model_len - count))
                        runtime.submit(prompt, config, request_state=state)
                    except Exception as exc:
                        # No facade ownership/resources exist if submit failed.
                        if not state.is_terminal:
                            state.mark_failed()
                        pending.pop(state.request_id)
                        delivery.future.set_result(HTTPCompletion(None, "failed", str(exc)))
                for rid, delivery in tuple(pending.items()):
                    if time.monotonic() >= delivery.deadline and not delivery.state.is_terminal:
                        delivery.timed_out = True
                        delivery.cancel.set()
                    if self._closing.is_set():
                        delivery.cancel.set()
                    if delivery.cancel.is_set():
                        try:
                            runtime.cancel(rid)
                        except Exception:
                            logger.exception("continuous cancellation/cleanup pending request_id=%s", rid)
                if pending:
                    try:
                        runtime.step()
                    except Exception:
                        # Executor records affected requests FAILED, preserving
                        # errors. Other requests continue; cleanup is retried.
                        logger.exception("continuous step failed")
                # Never block a shared GPU owner on a consumer. Completion is
                # separate from the token queue, including on the last token.
                for rid, delivery in tuple(pending.items()):
                    handle = delivery.stream
                    if handle is None or handle.cancel_event.is_set():
                        continue
                    tokens = runtime.get_result(rid).generated_tokens
                    try:
                        for index in range(delivery.emitted_tokens, len(tokens)):
                            text = self.engine.tokenizer.decode(list(tokens[:index + 1]), skip_special_tokens=True)
                            handle.events.put_nowait(TokenEvent(tokens[index], index + 1, text))
                            delivery.emitted_tokens = index + 1
                    except queue.Full:
                        self._stream_overflows += 1
                        handle.fail("stream_buffer_overflow", "Continuous output buffer is full; request cancelled")
                    except Exception as exc:
                        handle.fail("stream_decode_error", str(exc))
                    if handle.failure is not None:
                        try:
                            runtime.cancel(rid)
                        except Exception:
                            logger.exception("stream cancellation cleanup pending request_id=%s", rid)
                for rid, delivery in tuple(pending.items()):
                    if not delivery.state.is_terminal:
                        continue
                    # A long synchronous forward may finish after the deadline;
                    # it cannot be interrupted, but must not become a 200.
                    if time.monotonic() >= delivery.deadline:
                        delivery.timed_out = True
                    result = runtime.get_result(rid)
                    try:
                        if delivery.stream is not None and delivery.stream.failure is not None:
                            completion = HTTPCompletion(None, "failed", delivery.stream.failure[1])
                        elif delivery.timed_out:
                            completion = HTTPCompletion(None, "inference_timeout")
                        elif result.status is RequestStatus.FINISHED:
                            text = self.engine.tokenizer.decode(list(result.generated_tokens), skip_special_tokens=True)
                            positions = [text.find(s) for s in delivery.state.generation_config.stop or [] if s in text]
                            if positions:
                                text = text[:min(positions)]
                            completion = HTTPCompletion(GenerationResult(text,
                                "stop" if result.finish_reason is FinishReason.STOP else "length",
                                result.metrics.prefill_tokens, len(result.generated_tokens)), "completed")
                        else:
                            outcome = ("inference_timeout" if delivery.timed_out else
                                       "cancelled" if result.status is RequestStatus.CANCELLED else "failed")
                            completion = HTTPCompletion(None, outcome, result.error)
                    except Exception as exc:
                        completion = HTTPCompletion(None, "failed", str(exc))
                    try:
                        # Copy first; acknowledge releases/forgets only when safe.
                        runtime.acknowledge(rid)
                    except Exception:
                        logger.exception("continuous cleanup unconfirmed request_id=%s", rid)
                        continue
                    if time.monotonic() >= delivery.deadline:
                        completion = HTTPCompletion(None, "inference_timeout")
                    pending.pop(rid)
                    self._publish(runtime, worker, pending)
                    delivery.future.set_result(completion)
                self._publish(runtime, worker, pending)
                if self._closing.is_set() and not pending and self._queue.empty():
                    break
                if not pending:
                    # No per-request producers or model execution on event loop.
                    time.sleep(0.002)
                elif all(d.state.is_terminal for d in pending.values()):
                    time.sleep(0.02)  # bounded-rate cleanup retries
            runtime.close()
            self.stopped.set_result(None)
        except BaseException as exc:
            logger.exception("continuous owner failed")
            if not self.ready.done():
                self.ready.set_exception(exc)
                if worker is not None:
                    try:
                        worker.close()
                    except Exception:
                        logger.exception("Worker cleanup failed after startup failure")
            # Never acknowledge pending requests or release leases on fatal
            # owner failure: resource safety is unknown. Expose failure to ops.
            with self._status_lock:
                self._status.update(ready=False, fatal_error=str(exc))
            for delivery in pending.values():
                if not delivery.future.done():
                    delivery.future.set_exception(RuntimeError(f"owner failed; cleanup unconfirmed: {exc}"))
            while not self._queue.empty():
                delivery = self._queue.get_nowait()
                delivery.future.set_exception(RuntimeError(f"owner failed before handoff: {exc}"))
            self.stopped.set_exception(exc)
        finally:
            with self._status_lock:
                self._status["ready"] = False
