"""Tests for bounded multi-step batch decode coordination."""

from minillm.batching import (
    BatchExecutor,
    BatchStepResult,
    DecodeCoordinator,
    DecodeStatus,
    FinishReason,
)
from minillm.engine import RequestState, RequestStatus


class CountingBatchWorker:
    def __init__(
        self,
        finish_after: dict[str, int],
        *,
        fail_on: dict[str, int] | None = None,
    ) -> None:
        self.finish_after = finish_after
        self.fail_on = fail_on or {}
        self.counts: dict[str, int] = {}
        self.calls: list[list[str]] = []

    def execute_step(self, requests):
        self.calls.append([request.request_id for request in requests])
        results = []
        for request in requests:
            count = self.counts.get(request.request_id, 0) + 1
            self.counts[request.request_id] = count
            if self.fail_on.get(request.request_id) == count:
                results.append(
                    BatchStepResult(request.request_id, error="isolated decode error")
                )
                continue
            finished = count >= self.finish_after[request.request_id]
            results.append(
                BatchStepResult(
                    request.request_id,
                    token_id=count,
                    finished=finished,
                    finish_reason="stop" if finished else None,
                )
            )
        return results


def request(request_id: str, created_time: float) -> RequestState:
    return RequestState(request_id=request_id, created_time=created_time)


def coordinator_for(worker: CountingBatchWorker) -> DecodeCoordinator:
    return DecodeCoordinator(BatchExecutor(worker=worker))


def test_multi_step_decode_accumulates_tokens_until_finish():
    worker = CountingBatchWorker({"request": 3})
    coordinator = coordinator_for(worker)
    current = request("request", 1.0)
    batch = coordinator.executor.create_batch([current], max_batch_size=1)

    coordinator.run(batch, max_steps=3)

    assert current.generated_tokens == [1, 2, 3]
    assert current.status is RequestStatus.FINISHED
    assert batch.step_count == 3
    assert batch.is_finished is True


def test_early_finish_does_not_stop_other_requests():
    worker = CountingBatchWorker({"early": 1, "long": 3})
    coordinator = coordinator_for(worker)
    early = request("early", 1.0)
    long = request("long", 2.0)
    batch = coordinator.executor.create_batch([early, long], max_batch_size=2)

    coordinator.run(batch, max_steps=3)

    assert worker.calls == [["early", "long"], ["long"], ["long"]]
    assert early.generated_tokens == [1]
    assert long.generated_tokens == [1, 2, 3]
    assert [item.request_id for item in batch.completed_requests] == ["early", "long"]


def test_request_error_is_isolated_from_remaining_decode():
    worker = CountingBatchWorker(
        {"failed": 10, "healthy": 2},
        fail_on={"failed": 1},
    )
    coordinator = coordinator_for(worker)
    failed = request("failed", 1.0)
    healthy = request("healthy", 2.0)
    batch = coordinator.executor.create_batch([failed, healthy], max_batch_size=2)

    coordinator.run(batch, max_steps=2)

    assert failed.status is RequestStatus.FAILED
    assert healthy.status is RequestStatus.FINISHED
    assert healthy.generated_tokens == [1, 2]
    assert worker.calls == [["failed", "healthy"], ["healthy"]]


def test_cancelled_request_is_removed_before_later_steps():
    worker = CountingBatchWorker({"cancelled": 10, "remaining": 2})
    coordinator = coordinator_for(worker)
    cancelled = request("cancelled", 1.0)
    remaining = request("remaining", 2.0)
    batch = coordinator.executor.create_batch(
        [cancelled, remaining],
        max_batch_size=2,
    )

    coordinator.step(batch)
    assert coordinator.cancel_request(batch, "cancelled", now=5.0) is True
    coordinator.run(batch, max_steps=1)

    assert cancelled.status is RequestStatus.CANCELLED
    assert cancelled.finished_time == 5.0
    assert coordinator.get_state("cancelled").status is DecodeStatus.CANCELLED
    assert coordinator.get_state("cancelled").finish_reason is FinishReason.CANCELLED
    assert remaining.status is RequestStatus.FINISHED
    assert worker.calls == [["cancelled", "remaining"], ["remaining"]]
    assert coordinator.cancel_request(batch, "missing") is False
