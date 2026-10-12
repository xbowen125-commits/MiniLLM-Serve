"""Request-level and worker-level batch error separation tests."""

import pytest

from minillm.batching import (
    BatchExecutor,
    BatchStepResult,
    BatchWorkerError,
    DecodeCoordinator,
    DecodeStatus,
    RequestExecutionError,
)
from minillm.engine import GenerationConfig, RequestState


def request(request_id: str) -> RequestState:
    return RequestState(
        request_id=request_id,
        generation_config=GenerationConfig(max_tokens=5),
    )


class IsolatedRequestErrorWorker:
    def execute_step(self, requests):
        return [
            BatchStepResult(
                requests[0].request_id,
                error=RequestExecutionError("invalid request input"),
            ),
            BatchStepResult(requests[1].request_id, token_id=7),
        ]


class FailingBatchWorker:
    def execute_step(self, requests):
        raise BatchWorkerError("model forward failed")


def test_request_execution_error_is_isolated():
    coordinator = DecodeCoordinator(
        BatchExecutor(worker=IsolatedRequestErrorWorker())
    )
    failed = request("failed")
    healthy = request("healthy")
    batch = coordinator.executor.create_batch([failed, healthy], max_batch_size=2)

    coordinator.step(batch)

    assert coordinator.get_state("failed").status is DecodeStatus.FAILED
    assert coordinator.get_state("failed").error == "invalid request input"
    assert coordinator.get_state("healthy").status is DecodeStatus.ACTIVE
    assert coordinator.get_state("healthy").generated_tokens == [7]
    assert batch.requests == [healthy]


def test_batch_worker_error_fails_the_whole_step():
    coordinator = DecodeCoordinator(BatchExecutor(worker=FailingBatchWorker()))
    first = request("first")
    second = request("second")
    batch = coordinator.executor.create_batch([first, second], max_batch_size=2)

    with pytest.raises(BatchWorkerError, match="model forward failed"):
        coordinator.step(batch)

    assert coordinator.get_state("first").status is DecodeStatus.ACTIVE
    assert coordinator.get_state("second").status is DecodeStatus.ACTIVE
    assert batch.requests == [first, second]
