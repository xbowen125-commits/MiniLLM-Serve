"""Request-level decode runtime lifecycle tests."""

import pytest

from minillm.batching import (
    BatchExecutor,
    BatchStepResult,
    DecodeCoordinator,
    DecodeState,
    DecodeStatus,
    FinishReason,
)
from minillm.engine import GenerationConfig, RequestState, RequestStatus


class LimitAwareWorker:
    def __init__(self, *, error_request_id: str | None = None) -> None:
        self.error_request_id = error_request_id
        self.counts: dict[str, int] = {}
        self.calls: list[list[str]] = []

    def execute_step(self, requests):
        self.calls.append([request.request_id for request in requests])
        results = []
        for request in requests:
            if request.request_id == self.error_request_id:
                results.append(BatchStepResult(request.request_id, error="decode failed"))
                continue
            count = self.counts.get(request.request_id, 0) + 1
            self.counts[request.request_id] = count
            results.append(BatchStepResult(request.request_id, token_id=count))
        return results


def request(request_id: str, max_tokens: int) -> RequestState:
    return RequestState(
        request_id=request_id,
        generation_config=GenerationConfig(max_tokens=max_tokens),
    )


def test_decode_state_lifecycle_and_invalid_transitions():
    state = DecodeState(request_id="request", max_tokens=4, created_time=1.0)
    state.start(now=2.0)
    state.append_token(7, now=3.0)
    state.finish(FinishReason.STOP, now=4.0)

    assert state.status is DecodeStatus.FINISHED
    assert state.finish_reason is FinishReason.STOP
    assert state.started_time == 2.0
    assert state.first_token_time == 3.0
    assert state.finished_time == 4.0

    with pytest.raises(RuntimeError, match="cannot append"):
        state.append_token(8)
    with pytest.raises(RuntimeError, match="invalid decode transition"):
        state.cancel()


def test_decode_state_finishes_at_max_tokens():
    state = DecodeState(request_id="request", max_tokens=3)
    state.start(now=1.0)

    state.append_token(10, now=2.0)
    state.append_token(11, now=3.0)
    state.append_token(12, now=4.0)

    assert state.generated_tokens == [10, 11, 12]
    assert state.num_generated_tokens == 3
    assert state.status is DecodeStatus.FINISHED
    assert state.finish_reason is FinishReason.LENGTH
    assert state.finished_time == 4.0


def test_coordinator_enforces_different_request_token_limits():
    worker = LimitAwareWorker()
    coordinator = DecodeCoordinator(BatchExecutor(worker=worker))
    short = request("short", 2)
    long = request("long", 5)
    batch = coordinator.executor.create_batch([short, long], max_batch_size=2)

    coordinator.step(batch)
    coordinator.step(batch)
    coordinator.step(batch)

    short_state = coordinator.get_state("short")
    long_state = coordinator.get_state("long")
    assert short_state.status is DecodeStatus.FINISHED
    assert short_state.finish_reason is FinishReason.LENGTH
    assert short.status is RequestStatus.FINISHED
    assert short not in batch.requests
    assert long_state.status is DecodeStatus.ACTIVE
    assert long_state.generated_tokens == [1, 2, 3]
    assert long.status is RequestStatus.RUNNING
    assert worker.calls == [["short", "long"], ["short", "long"], ["long"]]


def test_coordinator_isolates_request_error_in_decode_state():
    worker = LimitAwareWorker(error_request_id="failed")
    coordinator = DecodeCoordinator(BatchExecutor(worker=worker))
    failed = request("failed", 5)
    healthy = request("healthy", 5)
    batch = coordinator.executor.create_batch([failed, healthy], max_batch_size=2)

    coordinator.step(batch)

    failed_state = coordinator.get_state("failed")
    healthy_state = coordinator.get_state("healthy")
    assert failed_state.status is DecodeStatus.FAILED
    assert failed_state.finish_reason is FinishReason.ERROR
    assert failed_state.error == "decode failed"
    assert failed not in batch.requests
    assert healthy_state.status is DecodeStatus.ACTIVE
    assert healthy_state.generated_tokens == [1]
    assert healthy in batch.requests
