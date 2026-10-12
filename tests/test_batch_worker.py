"""Tests for model-independent batch worker results."""

import pytest

from minillm.batching import BatchExecutor, BatchStepResult, BatchWorkerProtocol
from minillm.engine import RequestState, RequestStatus


class FakeBatchWorker:
    def __init__(self, results: list[BatchStepResult]) -> None:
        self.results = results
        self.seen_request_ids: list[str] = []

    def execute_step(self, requests):
        self.seen_request_ids = [request.request_id for request in requests]
        return self.results


def request(request_id: str, created_time: float) -> RequestState:
    return RequestState(request_id=request_id, created_time=created_time)


def test_fake_batch_worker_satisfies_protocol():
    worker = FakeBatchWorker([])

    assert isinstance(worker, BatchWorkerProtocol)


def test_executor_consumes_tokens_and_finish_reason():
    first = request("first", 1.0)
    worker = FakeBatchWorker(
        [
            BatchStepResult(
                request_id="first",
                token_id=42,
                finished=True,
                finish_reason="stop",
            )
        ]
    )
    executor = BatchExecutor(worker=worker)
    batch = executor.create_batch([first], max_batch_size=1)

    completed = executor.execute_step(batch)

    assert worker.seen_request_ids == ["first"]
    assert completed == [first]
    assert first.generated_tokens == [42]
    assert first.status is RequestStatus.FINISHED
    assert batch.step_results[0].finish_reason == "stop"


def test_requests_can_return_different_completion_states():
    first = request("first", 1.0)
    second = request("second", 2.0)
    worker = FakeBatchWorker(
        [
            BatchStepResult("first", token_id=7, finished=True, finish_reason="length"),
            BatchStepResult("second", token_id=9),
        ]
    )
    executor = BatchExecutor(worker=worker)
    batch = executor.create_batch([first, second], max_batch_size=2)

    completed = executor.execute_step(batch)

    assert completed == [first]
    assert first.status is RequestStatus.FINISHED
    assert second.status is RequestStatus.RUNNING
    assert first.generated_tokens == [7]
    assert second.generated_tokens == [9]
    assert batch.requests == [second]
    assert batch.is_finished is False


def test_worker_error_fails_only_affected_request():
    failed = request("failed", 1.0)
    active = request("active", 2.0)
    worker = FakeBatchWorker(
        [
            BatchStepResult("failed", error="model step failed"),
            BatchStepResult("active", token_id=3),
        ]
    )
    executor = BatchExecutor(worker=worker)
    batch = executor.create_batch([failed, active], max_batch_size=2)

    completed = executor.execute_step(batch)

    assert completed == [failed]
    assert failed.status is RequestStatus.FAILED
    assert active.status is RequestStatus.RUNNING
    assert batch.step_results[0].error == "model step failed"


def test_executor_rejects_unknown_and_duplicate_results():
    current = request("known", 1.0)
    unknown = BatchExecutor(worker=FakeBatchWorker([BatchStepResult("unknown")]))
    unknown_batch = unknown.create_batch([current], max_batch_size=1)

    with pytest.raises(ValueError, match="unknown request_id"):
        unknown.execute_step(unknown_batch)

    another = request("known", 1.0)
    duplicate = BatchExecutor(
        worker=FakeBatchWorker(
            [BatchStepResult("known"), BatchStepResult("known")]
        )
    )
    duplicate_batch = duplicate.create_batch([another], max_batch_size=1)

    with pytest.raises(ValueError, match="duplicate result"):
        duplicate.execute_step(duplicate_batch)


def test_batch_step_result_validation():
    with pytest.raises(ValueError, match="request_id"):
        BatchStepResult("")
    with pytest.raises(ValueError, match="finish_reason"):
        BatchStepResult("request", finish_reason="stop")
