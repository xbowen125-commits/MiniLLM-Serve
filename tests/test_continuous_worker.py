"""Logical batch protocol, result mapping, and per-request error isolation."""

import pytest

from minillm.continuous import BatchExecutor, ContinuousBatchWorkerProtocol, ContinuousBatchStepResult
from minillm.contracts import ExecutionPhase, FinishReason
from minillm.engine import RequestStatus
from test_continuous_batch import context


class FakeBatchWorker:
    def __init__(self, *, fail=None, invalid=None):
        self.fail = fail
        self.invalid = invalid
        self.calls = []

    def prefill_batch(self, requests):
        self.calls.append(("prefill", tuple(c.request_id for c in requests)))
        return [ContinuousBatchStepResult(c.request_id, ExecutionPhase.PREFILL)
                for c in reversed(requests)]

    def decode_batch(self, requests):
        self.calls.append(("decode", tuple(c.request_id for c in requests)))
        results = [ContinuousBatchStepResult(
            c.request_id, ExecutionPhase.DECODE,
            token_id=10 if c.request_id == "A" else 20,
            error=ValueError("request failed") if c.request_id == self.fail else None,
            finished=c.request_id == "A", finish_reason=FinishReason.STOP if c.request_id == "A" else None,
        ) for c in reversed(requests)]
        if self.invalid == "duplicate":
            return [results[0], results[0]]
        if self.invalid == "missing":
            return results[:1]
        if self.invalid == "unknown":
            return [ContinuousBatchStepResult("unknown", ExecutionPhase.DECODE), results[0]]
        return results


def setup(worker):
    executor = BatchExecutor(worker)
    first, second = context("A", 5), context("B", 5)
    executor.add(first)
    executor.add(second)
    return executor, first, second


def test_batch_worker_protocol_and_multi_prefill():
    worker = FakeBatchWorker()
    assert isinstance(worker, ContinuousBatchWorkerProtocol)
    executor, first, second = setup(worker)
    result = executor.step()
    assert result.prefilled == ("A", "B")
    assert len(executor.batch.active_requests) == 2
    assert first.request.status is second.request.status is RequestStatus.RUNNING
    assert worker.calls == [("prefill", ("A", "B"))]


def test_multi_decode_maps_reordered_results_and_retires_finished():
    executor, first, second = setup(FakeBatchWorker())
    executor.step()
    result = executor.step()
    assert first.request.generated_tokens == [10]
    assert second.request.generated_tokens == [20]
    assert result.retired == ("A",)
    assert list(executor.batch.active_requests) == ["B"]
    assert result.results[0].request_id == "A"
    assert result.results[0].token_id == 10
    assert result.results[0].finish_reason is FinishReason.STOP


def test_request_error_does_not_fail_other_context():
    executor, first, second = setup(FakeBatchWorker(fail="A"))
    executor.step()
    result = executor.step()
    assert first.request.status is RequestStatus.FAILED
    assert second.request.status is RequestStatus.RUNNING
    assert first.request.generated_tokens == []
    assert second.request.generated_tokens == [20]
    assert result.results[0].error == "request failed"
    assert result.retired == ("A",)


@pytest.mark.parametrize("invalid", ["duplicate", "missing", "unknown"])
def test_invalid_result_set_is_rejected_before_state_application(invalid):
    executor, first, second = setup(FakeBatchWorker(invalid=invalid))
    executor.step()
    with pytest.raises(RuntimeError):
        executor.step()
    assert first.request.generated_tokens == second.request.generated_tokens == []
    assert first.request.status is second.request.status is RequestStatus.RUNNING
