"""Fake-worker tests for persistent active contexts and incremental admission."""

import pytest

from minillm.continuous import BatchExecutor, BatchScheduler
from minillm.contracts import PrefillStepResult, DecodeStepResult
from minillm.engine import GenerationConfig, RequestState, RequestStatus
from minillm.runtime import RuntimeRequestContext


class FakeWorker:
    def __init__(self):
        self.calls = []

    def prefill(self, requests):
        self.calls.append(("prefill", tuple(r.request_id for r in requests)))
        return [PrefillStepResult(r.request_id, 3, False, False) for r in requests]

    def decode(self, requests):
        self.calls.append(("decode", tuple(r.request_id for r in requests)))
        return [DecodeStepResult(r.request_id, token_id=7) for r in requests]


def context(name, tokens=3):
    return RuntimeRequestContext(RequestState(request_id=name,
        generation_config=GenerationConfig(max_tokens=tokens)))


def test_multiple_requests_remain_active():
    executor = BatchExecutor(FakeWorker())
    first, second = context("A"), context("B")
    executor.add(first)
    executor.add(second)
    result = executor.step()
    assert result.prefilled == ("A", "B")
    assert executor.batch.active_requests == {"A": first, "B": second}
    executor.step()
    assert first.request.generated_tokens == [7]
    assert second.request.generated_tokens == [7]


def test_one_request_finishes_without_stopping_other():
    executor = BatchExecutor(FakeWorker())
    first, second = context("A", 1), context("B", 3)
    executor.add(first)
    executor.add(second)
    executor.step()
    result = executor.step()
    assert result.retired == ("A",)
    assert executor.batch.finished_requests["A"] is first
    assert executor.batch.active_requests["B"] is second
    assert second.request.status is RequestStatus.RUNNING
    executor.step()
    assert second.request.generated_tokens == [7, 7]


def test_new_request_joins_existing_batch_in_correct_step_order():
    worker = FakeWorker()
    executor = BatchExecutor(worker)
    first, second = context("A"), context("B")
    executor.add(first)
    executor.step()
    executor.add(second)
    step = executor.step()
    assert step.decoded == ("A",)
    assert step.prefilled == ("B",)
    executor.step()
    assert worker.calls == [("prefill", ("A",)), ("decode", ("A",)),
                            ("prefill", ("B",)), ("decode", ("A", "B"))]


def test_capacity_preserves_waiting_and_close_cancels_remaining():
    executor = BatchExecutor(FakeWorker(), scheduler=BatchScheduler(max_active=1, max_waiting=2))
    first, second = context("A", 1), context("B")
    executor.add(first)
    executor.add(second)
    executor.step()
    assert list(executor.batch.waiting_requests) == ["B"]
    executor.step()
    assert list(executor.batch.active_requests) == ["B"]
    executor.close()
    executor.close()
    assert second.request.status is RequestStatus.CANCELLED
    assert executor.batch.active_requests == {}
    assert executor.batch.waiting_requests == {}
    with pytest.raises(RuntimeError):
        executor.step()
