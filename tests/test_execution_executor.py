"""Execution contract and manager integration tests."""
import pytest
from minillm.batching import BatchExecutor, DecodeCoordinator, DecodeState, DecodeStatus
from minillm.engine import RequestState
from minillm.execution import ExecutionPhase, PrefillStepResult, DecodeStepResult
from minillm.execution.executor import PrefillExecutor, DecodeExecutor, ExecutionManager, ExecutionValidationError


class FakeWorker:
    def __init__(self):
        self.prefill_results = None
        self.decode_results = None
        self.calls = 0

    def prefill(self, requests):
        self.calls += 1
        if self.prefill_results is not None:
            return self.prefill_results
        return [PrefillStepResult(r.request_id, 3, True, False) for r in requests]

    def decode(self, requests):
        self.calls += 1
        if self.decode_results is not None:
            return self.decode_results
        return [DecodeStepResult(r.request_id, token_id=7) for r in requests]


def inputs():
    requests = [RequestState(request_id=name) for name in ("A", "B", "C")]
    states = {r.request_id: DecodeState(r.request_id, max_tokens=5) for r in requests}
    return requests, states


def test_prefill_executor_returns_complete_results():
    requests, states = inputs()
    results = PrefillExecutor(FakeWorker()).execute(requests, states)
    assert [r.request_id for r in results] == ["A", "B", "C"]


@pytest.mark.parametrize("ids", [("A", "B"), ("A", "A", "C"), ("A", "B", "unknown")])
def test_prefill_rejects_invalid_results(ids):
    requests, states = inputs()
    worker = FakeWorker()
    worker.prefill_results = [PrefillStepResult(name, 3, True, False) for name in ids]
    with pytest.raises(ExecutionValidationError):
        PrefillExecutor(worker).execute(requests, states)


def test_decode_rejects_waiting_before_worker_call():
    requests, states = inputs()
    worker = FakeWorker()
    with pytest.raises(ExecutionValidationError, match="phase"):
        DecodeExecutor(worker).execute(requests, states)
    assert worker.calls == 0


def test_decode_executor_returns_token_without_mutating_request():
    requests, states = inputs()
    for state in states.values():
        state.transition_phase(ExecutionPhase.PREFILL)
        state.transition_phase(ExecutionPhase.DECODE)
    results = DecodeExecutor(FakeWorker()).execute(requests, states)
    assert [r.token_id for r in results] == [7, 7, 7]
    assert all(r.generated_tokens == [] for r in requests)


def test_manager_coordinator_isolation_and_metrics():
    worker = FakeWorker()
    manager = ExecutionManager(worker)
    coordinator = DecodeCoordinator(BatchExecutor(), execution_manager=manager)
    first = RequestState(request_id="A")
    second = RequestState(request_id="B")
    batch = coordinator.executor.create_batch([first, second], max_batch_size=2)
    coordinator.execute_prefill(batch)
    worker.decode_results = [DecodeStepResult("A", error=ValueError("invalid input")), DecodeStepResult("B", token_id=7)]
    coordinator.execute_decode(batch)
    assert coordinator.get_state("A").status is DecodeStatus.FAILED
    assert coordinator.get_state("B").status is DecodeStatus.ACTIVE
    assert second.generated_tokens == [7]
    assert batch.requests == [second]
    assert manager.metrics.prefill_requests == 2
    assert manager.metrics.prefill_tokens == 6
    assert manager.metrics.decode_requests == 2
    assert manager.metrics.decode_tokens == 1
    assert manager.metrics.failed_requests == 1


def test_invalid_results_do_not_update_metrics():
    worker = FakeWorker()
    manager = ExecutionManager(worker)
    requests, states = inputs()
    worker.prefill_results = []
    with pytest.raises(ExecutionValidationError):
        manager.execute_prefill(requests, states)
    assert manager.metrics.prefill_requests == 0
    assert all(state.phase is ExecutionPhase.WAITING for state in states.values())


def test_step_routes_prefill_then_decode_through_manager():
    worker = FakeWorker()
    manager = ExecutionManager(worker)
    coordinator = DecodeCoordinator(BatchExecutor(), execution_manager=manager)
    current = RequestState(request_id="A")
    batch = coordinator.executor.create_batch([current], max_batch_size=1)
    coordinator.step(batch)
    assert coordinator.get_state("A").phase is ExecutionPhase.DECODE
    coordinator.step(batch)
    assert current.generated_tokens == [7]
    assert manager.metrics.prefill_requests == 1
    assert manager.metrics.decode_tokens == 1
