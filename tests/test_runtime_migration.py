"""Coordinator contexts are authoritative; legacy views never copy state."""

from dataclasses import fields

from minillm.batching import BatchExecutor, BatchStepResult, DecodeCoordinator
from minillm.batching.state import DecodeState as LegacyDecodeState
from minillm.contracts import DecodeStatus, ExecutionPhase, DecodeStepResult, PrefillStepResult
from minillm.engine import RequestState, RequestStatus
from minillm.runtime import DecodeState, RuntimeRequestContext
from minillm.runtime.adapter import DecodeStateAdapter
from minillm.scheduler.phase_policy import PhaseSchedulingPolicy


class Worker:
    def execute_step(self, requests):
        return [BatchStepResult(r.request_id, token_id=7, finished=True) for r in requests]


def test_coordinator_uses_context_and_single_token_source():
    coordinator = DecodeCoordinator(BatchExecutor(worker=Worker()))
    request = RequestState(request_id="A")
    batch = coordinator.executor.create_batch([request], max_batch_size=1)
    coordinator.step(batch)
    context = coordinator.contexts["A"]
    assert isinstance(context, RuntimeRequestContext)
    assert context.request is request
    view = coordinator.get_state("A")
    assert isinstance(view, DecodeStateAdapter)
    assert view.generated_tokens is request.generated_tokens
    assert request.generated_tokens == [7]
    assert context.decode_state.decode_tokens == 1
    assert context.request.status is RequestStatus.FINISHED
    assert view.status is DecodeStatus.FINISHED


def test_phase_and_lifecycle_are_separate_and_external_cancel_is_visible():
    request = RequestState(request_id="A")
    context = RuntimeRequestContext(request)
    view = DecodeStateAdapter(context)
    view.transition_phase(ExecutionPhase.PREFILL)
    assert request.status is RequestStatus.WAITING
    request.mark_cancelled(now=5)
    assert view.status is DecodeStatus.CANCELLED
    assert view.finished_time == 5
    assert context.decode_state.execution_phase is ExecutionPhase.PREFILL
    assert not {"status", "generated_tokens", "started_time", "finished_time"}.intersection(f.name for f in fields(DecodeState))


def test_explicit_execution_results_write_tokens_once():
    coordinator = DecodeCoordinator(BatchExecutor())
    request = RequestState(request_id="A")
    batch = coordinator.executor.create_batch([request], max_batch_size=1)
    coordinator.apply_prefill_results(batch, [PrefillStepResult("A", 12, True, False)])
    coordinator.apply_decode_results(batch, [DecodeStepResult("A", token_id=9)])
    context = coordinator.contexts["A"]
    assert context.request.generated_tokens == [9]
    assert context.decode_state.prefill_tokens == 12
    assert context.decode_state.decode_tokens == 1
    assert context.decode_state.last_token_time == context.decode_state.first_token_time


def test_old_export_and_live_phase_mapping_remain_compatible():
    assert LegacyDecodeState("legacy", max_tokens=5).status is DecodeStatus.ACTIVE
    coordinator = DecodeCoordinator(BatchExecutor())
    policy = PhaseSchedulingPolicy(coordinator.decode_states)
    request = RequestState(request_id="A")
    batch = coordinator.executor.create_batch([request], max_batch_size=1)
    coordinator.apply_prefill_results(batch, [PrefillStepResult("A", 2, True, False)])
    assert policy.phase_for(request) is ExecutionPhase.DECODE
