"""Offline paging integration; no models, downloads, or GPU execution."""

import pytest

from minillm.continuous import BatchExecutor, ContinuousBatchStepResult, PagedContinuousBatchWorker
from minillm.contracts import ExecutionPhase
from minillm.engine import RequestStatus
from minillm.memory import PagedKVCacheManager
from test_continuous_batch import context


class FakeWorker:
    def __init__(self, fail=None, malformed=False):
        self.fail = fail
        self.malformed = malformed
        self.released = []

    def prefill_batch(self, requests):
        results = [ContinuousBatchStepResult(c.request_id, ExecutionPhase.PREFILL)
                   for c in reversed(requests)]
        return results[:1] if self.malformed else results

    def decode_batch(self, requests):
        return [ContinuousBatchStepResult(c.request_id, ExecutionPhase.DECODE,
                token_id=7 if c.request_id == "A" else 8,
                error=ValueError("failed") if c.request_id == self.fail else None)
                for c in reversed(requests)]

    def release_request(self, request_id):
        self.released.append(request_id)


def setup(*, capacity=64, max_tokens=5, backend=None):
    backend = backend or FakeWorker()
    worker = PagedContinuousBatchWorker(backend,
        PagedKVCacheManager(capacity, tokens_per_block=2, bytes_per_token=4),
        prompt_token_count=lambda c: 2)
    executor = BatchExecutor(worker)
    a, b = context("A", max_tokens), context("B", max_tokens)
    executor.add(a)
    executor.add(b)
    return worker, executor, a, b


def test_prefill_distinct_pages_and_decode_growth():
    worker, executor, a, b = setup()
    executor.step()
    first = worker.get_block_table("A")
    second = worker.get_block_table("B")
    assert len(first) == len(second) == 1
    assert set(first.values()).isdisjoint(second.values())
    executor.step()
    assert worker.snapshot().request_tokens == {"A": 3, "B": 3}
    assert len(worker.get_block_table("A")) == len(worker.get_block_table("B")) == 2
    assert set(worker.get_block_table("A").values()).isdisjoint(worker.get_block_table("B").values())
    assert a.request.generated_tokens == [7]
    assert b.request.generated_tokens == [8]
    worker.close()


def test_release_is_idempotent_and_isolated():
    worker, executor, _, _ = setup()
    executor.step()
    worker.release_request("A")
    worker.release_request("A")
    with pytest.raises(KeyError):
        worker.get_block_table("A")
    assert len(worker.get_block_table("B")) == 1
    assert worker.worker.released == ["A"]
    executor.close()
    worker.close()
    assert worker.snapshot().memory.allocated_bytes == 0
    assert worker.snapshot().request_tokens == {}


def test_length_completion_releases_without_next_step():
    worker, executor, a, b = setup(max_tokens=1)
    executor.step()
    executor.step()
    assert a.request.status is b.request.status is RequestStatus.FINISHED
    assert worker.snapshot().memory.allocated_bytes == 0
    assert set(worker.worker.released) == {"A", "B"}


def test_request_error_isolation():
    worker, executor, a, b = setup(backend=FakeWorker(fail="A"))
    executor.step()
    executor.step()
    assert a.request.status is RequestStatus.FAILED
    assert b.request.status is RequestStatus.RUNNING
    assert worker.snapshot().request_tokens == {"B": 3}
    worker.close()


def test_prefill_capacity_failure_does_not_leak():
    worker, executor, a, b = setup(capacity=8)
    executor.step()
    assert a.request.status is RequestStatus.RUNNING
    assert b.request.status is RequestStatus.FAILED
    assert worker.snapshot().request_tokens == {"A": 2}
    worker.close()


def test_decode_capacity_failure_releases_only_failed_request():
    worker, executor, a, b = setup(capacity=24)
    executor.step()
    executor.step()
    assert a.request.status is RequestStatus.RUNNING
    assert b.request.status is RequestStatus.FAILED
    assert worker.snapshot().request_tokens == {"A": 3}
    assert b.request.generated_tokens == []
    worker.close()


def test_bad_result_mapping_releases_entire_prefill_batch():
    worker, executor, _, _ = setup(backend=FakeWorker(malformed=True))
    with pytest.raises(RuntimeError, match="count mismatch"):
        executor.step()
    assert worker.snapshot().request_tokens == {}
    assert worker.snapshot().memory.allocated_bytes == 0


def test_cancelled_context_reclaimed_and_closed_worker_rejects_calls():
    worker, executor, a, b = setup()
    executor.step()
    a.request.mark_cancelled()
    executor.step()
    assert worker.snapshot().request_tokens == {"B": 3}
    executor.close()
    worker.close()
    assert worker.snapshot().memory.allocated_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        worker.decode_batch([b])
