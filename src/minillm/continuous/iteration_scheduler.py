"""Orca-inspired offline iteration selection, not selective tensor batching.

    Reference: https://www.usenix.org/conference/osdi22/presentation/yu
"""

from ..contracts import ExecutionPhase
from .batch import ContinuousBatch
from .plan import ContinuousBatchPlan
from .scheduler import BatchScheduler
from .state import IterationState


class IterationScheduler(BatchScheduler):
    """Rebuild a bounded plan each iteration, preserving active decode order.

    Compatible with the existing BatchExecutor through build_plan/add/max_active.
    Prefill uses capacity available at selection time; a slot freed during decode
    is admitted on the next iteration. Plans do not move waiting contexts or
    execute workers. Terminal retirement is the only batch mutation here.
    The scheduler instance belongs to one batch and is driven serially.
    """

    def __init__(self, *, max_batch_size: int = 4, max_waiting: int = 64):
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if type(max_waiting) is not int or max_waiting < 0:
            raise ValueError("max_waiting must be a nonnegative integer")
        super().__init__(max_active=max_batch_size, max_waiting=max_waiting)
        self.last_state = IterationState(0)
        self._observed_finished: set[str] = set()

    def build_plan(self, batch: ContinuousBatch) -> ContinuousBatchPlan:
        batch.retire()
        active = tuple(batch.active_requests.values())
        if len(active) > self.max_active:
            raise RuntimeError("active requests exceed iteration batch capacity")
        if any(c.decode_state.execution_phase is not ExecutionPhase.DECODE for c in active):
            raise RuntimeError("active requests must be in DECODE phase")
        waiting = sorted((c for c in batch.waiting_requests.values()
                          if not c.request.is_terminal), key=lambda c: c.request.created_time)
        if any(c.decode_state.execution_phase is not ExecutionPhase.WAITING for c in waiting):
            raise RuntimeError("waiting requests must be in WAITING phase")
        prefill = tuple(waiting[:self.max_active - len(active)])
        finished = tuple(r for r in batch.finished_requests if r not in self._observed_finished)
        plan = ContinuousBatchPlan(prefill, active, tuple(batch.active_requests))
        self.last_state = IterationState(self.last_state.iteration_id + 1,
            tuple(c.request_id for c in active), tuple(c.request_id for c in prefill), finished)
        self._observed_finished.update(finished)
        return plan
