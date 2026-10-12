"""Tests for the model-independent batching architecture."""

import pytest

from minillm.batching import BatchExecutor, BatchPolicy, BatchState, FIFOBatchPolicy
from minillm.engine import RequestState, RequestStatus


def request(request_id: str, created_time: float) -> RequestState:
    return RequestState(request_id=request_id, created_time=created_time)


def test_batch_adds_requests_and_rejects_duplicates():
    batch = BatchState(batch_id="batch-1")
    first = request("first", 1.0)
    second = request("second", 2.0)

    batch.add_request(first)
    batch.add_request(second)

    assert batch.requests == [first, second]
    assert batch.all_requests == (first, second)
    with pytest.raises(ValueError, match="duplicate"):
        batch.add_request(first)


def test_batch_removes_completed_requests_only():
    batch = BatchState()
    finished = request("finished", 1.0)
    active = request("active", 2.0)
    batch.add_request(finished)
    batch.add_request(active)
    batch.start(now=3.0)
    finished.mark_finished(now=4.0)

    removed = batch.remove_completed()

    assert removed == [finished]
    assert batch.requests == [active]
    assert batch.completed_requests == [finished]


def test_fifo_policy_orders_by_creation_time_and_limits_size():
    policy = FIFOBatchPolicy()
    newest = request("newest", 30.0)
    oldest = request("oldest", 10.0)
    middle = request("middle", 20.0)

    selected = policy.select([newest, oldest, middle], max_batch_size=2)

    assert selected == [oldest, middle]
    assert isinstance(policy, BatchPolicy)


def test_policy_excludes_terminal_requests():
    policy = FIFOBatchPolicy()
    terminal = request("terminal", 1.0)
    terminal.mark_cancelled(now=2.0)
    waiting = request("waiting", 3.0)

    assert policy.select([terminal, waiting], max_batch_size=2) == [waiting]


def test_executor_lifecycle_runs_steps_and_finishes_batch():
    seen_steps: list[list[str]] = []

    def finish_requests(requests):
        seen_steps.append([request.request_id for request in requests])
        for current in requests:
            current.append_token(7)
            current.mark_finished()

    first = request("first", 1.0)
    second = request("second", 2.0)
    executor = BatchExecutor(step=finish_requests)
    batch = executor.create_batch([second, first], max_batch_size=2)

    completed = executor.execute_step(batch)

    assert seen_steps == [["first", "second"]]
    assert completed == [first, second]
    assert all(item.status is RequestStatus.FINISHED for item in completed)
    assert all(item.generated_tokens == [7] for item in completed)
    assert batch.requests == []
    assert batch.step_count == 1
    assert batch.is_finished is True
    assert batch.started_time is not None
    assert batch.finished_time is not None


def test_executor_default_step_keeps_requests_active():
    current = request("request", 1.0)
    batch = BatchExecutor().create_batch([current], max_batch_size=1)

    assert BatchExecutor().execute_step(batch) == []
    assert current.status is RequestStatus.RUNNING
    assert batch.step_count == 1
    assert batch.is_finished is False


def test_executor_rejects_step_after_batch_finishes():
    current = request("request", 1.0)

    def finish(requests):
        requests[0].mark_finished()

    executor = BatchExecutor(step=finish)
    batch = executor.create_batch([current], max_batch_size=1)
    executor.execute_step(batch)

    with pytest.raises(RuntimeError, match="finished batch"):
        executor.execute_step(batch)
