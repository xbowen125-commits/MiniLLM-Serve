"""FIFO request scheduling for the local inference engine.

The scheduler deliberately manages admission only.  The caller owns the model
worker and must release its lease *after* that worker has stopped.  This keeps a
timed-out CUDA request from freeing a concurrency slot while a kernel is still
running.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Literal

from ..engine import GenerationConfig, RequestState
from ..request_queue import RequestQueue
from .phase_policy import PhaseAwareBatchPolicy, PhaseSchedulingPolicy
from .snapshot import SchedulerSnapshot
from .fair_policy import FairPhaseBatchPolicy
from .priority import PhasePriority


class SchedulerError(RuntimeError):
    """Base class for request admission errors."""


class SchedulerQueueFull(SchedulerError):
    """Raised when all execution and waiting slots are occupied."""


class SchedulerQueueTimeout(SchedulerError):
    """Raised when a request waits too long for an execution slot."""


class SchedulerClosed(SchedulerError):
    """Raised when the service is shutting down."""


RequestOutcome = Literal[
    "completed",
    "failed",
    "inference_timeout",
    "cancelled",
]


@dataclass(slots=True)
class _Waiter:
    sequence: int
    enqueued_at: float
    future: asyncio.Future["SchedulerLease"]
    request_state: RequestState


class SchedulerLease:
    """One admitted inference request and its request-local cancellation flag."""

    def __init__(
        self,
        scheduler: "RequestScheduler",
        request_id: int,
        queue_wait_seconds: float,
        request_state: RequestState,
    ) -> None:
        self._scheduler = scheduler
        self.request_id = request_id
        self.queue_wait_seconds = queue_wait_seconds
        self.request_state = request_state
        self.cancel_event = threading.Event()
        self._released = False

    async def release(self, outcome: RequestOutcome) -> None:
        """Release the execution slot exactly once."""
        if self._released:
            return
        await self._scheduler._release(self, outcome)
        # Marking before the await would permanently leak this lease if the
        # releasing task were cancelled while waiting for the scheduler lock.
        self._released = True


class RequestScheduler:
    """A small, fair, in-process scheduler for one model instance."""

    def __init__(
        self,
        *,
        max_running: int,
        queue_capacity: int,
        queue_timeout_seconds: float,
        inference_timeout_seconds: float,
        max_batch_size: int = 4,
        phase_policy: PhaseAwareBatchPolicy | None = None,
    ) -> None:
        if max_running < 1:
            raise ValueError("max_running must be at least 1")
        if queue_capacity < 0:
            raise ValueError("queue_capacity must be non-negative")
        if queue_timeout_seconds <= 0:
            raise ValueError("queue_timeout_seconds must be positive")
        if inference_timeout_seconds <= 0:
            raise ValueError("inference_timeout_seconds must be positive")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")

        self.max_running = max_running
        self.queue_capacity = queue_capacity
        self.queue_timeout_seconds = queue_timeout_seconds
        self.inference_timeout_seconds = inference_timeout_seconds
        self.max_batch_size = max_batch_size
        self.phase_policy = phase_policy

        self._lock = asyncio.Lock()
        self._waiters = RequestQueue(queue_capacity)
        self._waiting_metadata: dict[str, _Waiter] = {}
        self._running: dict[int, SchedulerLease] = {}
        self._next_sequence = 1
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()

        self._accepted = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._queue_timeouts = 0
        self._inference_timeouts = 0
        self._cancelled = 0
        self._queue_wait_samples = 0
        self._queue_wait_total = 0.0
        self._queue_wait_max = 0.0

    async def acquire(
        self,
        generation_config: GenerationConfig | None = None,
    ) -> SchedulerLease:
        """Admit immediately or wait in strict FIFO order."""
        waiter: _Waiter | None = None

        async with self._lock:
            if self._closed:
                self._rejected += 1
                raise SchedulerClosed("request scheduler is shutting down")

            now = time.monotonic()
            if len(self._running) < self.max_running and not self._waiters:
                sequence, request_state = self._new_request_locked(generation_config)
                return self._grant_locked(sequence, request_state, now, now)

            if self._waiters.is_full:
                self._rejected += 1
                raise SchedulerQueueFull("inference request queue is full")

            future: asyncio.Future[SchedulerLease] = asyncio.get_running_loop().create_future()
            sequence, request_state = self._new_request_locked(generation_config)
            waiter = _Waiter(sequence, now, future, request_state)
            if not self._waiters.put(request_state):
                self._rejected += 1
                raise SchedulerQueueFull("inference request queue is full")
            self._waiting_metadata[request_state.request_id] = waiter

        try:
            return await asyncio.wait_for(
                asyncio.shield(waiter.future),
                timeout=self.queue_timeout_seconds,
            )
        except TimeoutError as exc:
            async with self._lock:
                # A release at the timeout boundary may already have admitted us.
                if waiter.future.done() and not waiter.future.cancelled():
                    return waiter.future.result()
                self._remove_waiter_locked(waiter)
                self._cancel_request(waiter.request_state)
                self._queue_timeouts += 1
                waiter.future.cancel()
            raise SchedulerQueueTimeout(
                "timed out while waiting for an inference slot"
            ) from exc
        except asyncio.CancelledError:
            admitted: SchedulerLease | None = None
            async with self._lock:
                if waiter.future.done() and not waiter.future.cancelled():
                    admitted = waiter.future.result()
                else:
                    self._remove_waiter_locked(waiter)
                    self._cancel_request(waiter.request_state)
                    waiter.future.cancel()
                    self._cancelled += 1
            if admitted is not None:
                admitted.cancel_event.set()
                await asyncio.shield(admitted.release("cancelled"))
            raise

    async def snapshot(self) -> dict[str, int | float | bool]:
        """Return prompt-free local scheduling metrics."""
        async with self._lock:
            average = (
                self._queue_wait_total / self._queue_wait_samples
                if self._queue_wait_samples
                else 0.0
            )
            return {
                "running_requests": len(self._running),
                "queued_requests": len(self._waiters),
                "max_concurrent_requests": self.max_running,
                "queue_capacity": self.queue_capacity,
                "accepted_requests": self._accepted,
                "completed_requests": self._completed,
                "failed_requests": self._failed,
                "rejected_requests": self._rejected,
                "queue_timeout_requests": self._queue_timeouts,
                "inference_timeout_requests": self._inference_timeouts,
                "cancelled_requests": self._cancelled,
                "queue_wait_samples": self._queue_wait_samples,
                "total_queue_wait_ms": round(self._queue_wait_total * 1000, 3),
                "average_queue_wait_ms": round(average * 1000, 3),
                "max_queue_wait_ms": round(self._queue_wait_max * 1000, 3),
                "shutting_down": self._closed,
            }

    async def close(self) -> None:
        """Reject queued work, cancel running work, and wait for real cleanup."""
        async with self._lock:
            if not self._closed:
                self._closed = True
                waiting = self._waiters.drain()
                self._cancelled += len(waiting)
                for request_state in waiting:
                    waiter = self._waiting_metadata.pop(request_state.request_id)
                    self._cancel_request(request_state)
                    if not waiter.future.done():
                        waiter.future.set_exception(
                            SchedulerClosed("request scheduler is shutting down")
                        )
                for lease in self._running.values():
                    lease.cancel_event.set()
            if not self._running:
                self._idle.set()

        # There is intentionally no forced slot release here.  The request
        # handlers release leases only after their worker thread has exited.
        await self._idle.wait()

    def _new_request_locked(
        self,
        generation_config: GenerationConfig | None,
    ) -> tuple[int, RequestState]:
        request_id = self._next_sequence
        self._next_sequence += 1
        request_state = RequestState(
            request_id=str(request_id),
            generation_config=generation_config or GenerationConfig(),
        )
        return request_id, request_state

    def _grant_locked(
        self,
        request_id: int,
        request_state: RequestState,
        enqueued_at: float,
        now: float,
    ) -> SchedulerLease:
        wait_seconds = max(0.0, now - enqueued_at)
        request_state.mark_running()
        lease = SchedulerLease(self, request_id, wait_seconds, request_state)
        self._running[request_id] = lease
        self._accepted += 1
        self._idle.clear()
        if wait_seconds > 0:
            self._queue_wait_samples += 1
            self._queue_wait_total += wait_seconds
            self._queue_wait_max = max(self._queue_wait_max, wait_seconds)
        return lease

    def _remove_waiter_locked(self, target: _Waiter) -> None:
        self._waiters.remove(target.request_state)
        self._waiting_metadata.pop(target.request_state.request_id, None)

    async def schedule_batch(self) -> list[RequestState]:
        """Admit one FIFO batch without exceeding the execution limit."""
        async with self._lock:
            if self._closed:
                return []
            return self._schedule_batch_locked(time.monotonic())

    async def phase_snapshot(self, *, now: float | None = None,
                             starvation_threshold_seconds: float = 60.0) -> SchedulerSnapshot:
        """Observe only currently owned waiting and running requests."""
        async with self._lock:
            requests = [waiter.request_state for waiter in self._waiting_metadata.values()]
            waiting_requests = tuple(requests)
            requests.extend(lease.request_state for lease in self._running.values())
            return SchedulerSnapshot.from_requests(
                requests,
                self.phase_policy or PhaseSchedulingPolicy(),
                active_batch_size=len(self._running),
                now=now,
                starvation_threshold_seconds=starvation_threshold_seconds,
                waiting_requests=waiting_requests,
            )

    def _schedule_batch_locked(self, now: float) -> list[RequestState]:
        available_slots = self.max_running - len(self._running)
        if available_slots <= 0:
            return []

        limit = min(self.max_batch_size, available_slots)
        if self.phase_policy is None:
            states = self._waiters.get_batch(limit)
        else:
            states = self.phase_policy.select_batch(
                [waiter.request_state for waiter in self._waiting_metadata.values()],
                limit,
            )
            for state in states:
                self._waiters.remove(state)
        scheduled: list[RequestState] = []
        for request_state in states:
            waiter = self._waiting_metadata.pop(request_state.request_id)
            if waiter.future.done():
                self._cancel_request(request_state)
                continue
            next_lease = self._grant_locked(
                waiter.sequence,
                request_state,
                waiter.enqueued_at,
                now,
            )
            waiter.future.set_result(next_lease)
            scheduled.append(request_state)
        return scheduled

    @staticmethod
    def _cancel_request(request_state: RequestState) -> None:
        if not request_state.is_terminal:
            request_state.mark_cancelled()

    @staticmethod
    def _finish_request(request_state: RequestState, outcome: RequestOutcome) -> None:
        if request_state.is_terminal:
            return
        if outcome == "completed":
            request_state.mark_finished()
        elif outcome in {"failed", "inference_timeout"}:
            request_state.mark_failed()
        elif outcome == "cancelled":
            request_state.mark_cancelled()

    async def _release(
        self,
        lease: SchedulerLease,
        outcome: RequestOutcome,
    ) -> None:
        async with self._lock:
            if self._running.pop(lease.request_id, None) is None:
                return

            self._finish_request(lease.request_state, outcome)

            if outcome == "completed":
                self._completed += 1
            elif outcome == "failed":
                self._failed += 1
            elif outcome == "inference_timeout":
                self._inference_timeouts += 1
            elif outcome == "cancelled":
                self._cancelled += 1

            if not self._closed:
                self._schedule_batch_locked(time.monotonic())

            if not self._running:
                self._idle.set()
