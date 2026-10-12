"""Per-request decode progress monitoring tests."""

from minillm.batching import (
    BatchExecutor,
    BatchStepResult,
    DecodeCoordinator,
    DecodeStatus,
    FinishReason,
)
from minillm.engine import GenerationConfig, RequestState


class NoProgressWorker:
    def execute_step(self, requests):
        return [BatchStepResult(request.request_id) for request in requests]


class StructuredFinishWorker:
    def execute_step(self, requests):
        return [
            BatchStepResult(
                requests[0].request_id,
                token_id=2,
                finished=True,
                finish_reason=FinishReason.STOP,
            )
        ]


def request(request_id: str) -> RequestState:
    return RequestState(
        request_id=request_id,
        generation_config=GenerationConfig(max_tokens=20),
    )


def test_request_fails_after_ten_steps_without_progress():
    coordinator = DecodeCoordinator(BatchExecutor(worker=NoProgressWorker()))
    stalled = request("stalled")
    batch = coordinator.executor.create_batch([stalled], max_batch_size=1)

    for _ in range(10):
        coordinator.step(batch)

    state = coordinator.get_state("stalled")
    assert state.status is DecodeStatus.FAILED
    assert state.finish_reason is FinishReason.ERROR
    assert state.error == "decode made no progress"
    assert state.no_progress_steps == 10
    assert state.decode_steps == 10
    assert batch.requests == []
    assert batch.is_finished is True


def test_structured_stop_reason_is_saved():
    coordinator = DecodeCoordinator(BatchExecutor(worker=StructuredFinishWorker()))
    current = request("request")
    batch = coordinator.executor.create_batch([current], max_batch_size=1)

    coordinator.step(batch)

    state = coordinator.get_state("request")
    assert state.status is DecodeStatus.FINISHED
    assert state.finish_reason is FinishReason.STOP
    assert state.generated_tokens == [2]
