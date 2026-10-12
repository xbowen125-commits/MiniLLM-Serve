"""Offline bounded admission; independent from the online RequestScheduler."""

from ..contracts import ExecutionPhase
from ..runtime.context import RuntimeRequestContext
from .batch import ContinuousBatch
from .plan import ContinuousBatchPlan


class BatchScheduler:
    """Keep decoding requests active and admit oldest waiting contexts."""

    def __init__(self, *, max_active: int = 4, max_waiting: int = 64) -> None:
        if max_active < 1 or max_waiting < 0:
            raise ValueError("invalid continuous batch capacity")
        self.max_active = max_active
        self.max_waiting = max_waiting

    def add(self, batch: ContinuousBatch, context: RuntimeRequestContext) -> None:
        if len(batch.waiting_requests) >= self.max_waiting:
            raise RuntimeError("continuous waiting queue is full")
        if context.decode_state.execution_phase is not ExecutionPhase.WAITING:
            raise ValueError("new contexts must be in WAITING phase")
        batch.add(context)

    def decode_requests(self, batch: ContinuousBatch) -> tuple[RuntimeRequestContext, ...]:
        batch.retire()
        return tuple(c for c in batch.active_requests.values()
                     if c.decode_state.execution_phase is ExecutionPhase.DECODE)

    def build_plan(self, batch: ContinuousBatch) -> ContinuousBatchPlan:
        """Read-only selection; prefill admission is deferred until after decode."""
        decode = tuple(c for c in batch.active_requests.values()
                       if not c.request.is_terminal and
                       c.decode_state.execution_phase is ExecutionPhase.DECODE)
        waiting = sorted((c for c in batch.waiting_requests.values()
                          if not c.request.is_terminal), key=lambda c: c.request.created_time)
        return ContinuousBatchPlan(tuple(waiting[:self.max_active]), decode,
                                   tuple(batch.active_requests))

    def admit_prefill(self, batch: ContinuousBatch) -> tuple[RuntimeRequestContext, ...]:
        batch.retire()
        slots = max(0, self.max_active - len(batch.active_requests))
        waiting = sorted(batch.waiting_requests.values(), key=lambda c: c.request.created_time)
        admitted = waiting[:slots]
        for context in admitted:
            batch.active_requests[context.request_id] = batch.waiting_requests.pop(context.request_id)
        return tuple(admitted)
