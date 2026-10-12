"""Phase-separated execution result, protocol, and coordinator tests."""

import pytest

from minillm.batching import (
    BatchExecutor,
    DecodeCoordinator,
    DecodeState,
    DecodeStatus,
    FinishReason,
)
from minillm.engine import GenerationConfig, RequestState
from minillm.execution import (
    DecodeStepResult,
    ExecutionPhase,
    ExecutionWorkerProtocol,
    PrefillStepResult,
)


class FakeExecutionWorker:
    def prefill(self, requests):
        return [
            PrefillStepResult(
                request_id=request.request_id,
                prompt_tokens=3,
                kv_cache_created=True,
                first_token_ready=False,
            )
            for request in requests
        ]

    def decode(self, requests):
        return [
            DecodeStepResult(request_id=request.request_id, token_id=7)
            for request in requests
        ]


def request(request_id: str) -> RequestState:
    return RequestState(
        request_id=request_id,
        generation_config=GenerationConfig(max_tokens=5),
    )


def test_prefill_step_result_creation():
    result = PrefillStepResult(
        request_id="request",
        prompt_tokens=12,
        kv_cache_created=True,
        first_token_ready=True,
    )

    assert result.prompt_tokens == 12
    assert result.kv_cache_created is True
    assert result.first_token_ready is True
    assert result.error is None


def test_decode_step_result_creation():
    result = DecodeStepResult(
        request_id="request",
        token_id=9,
        finished=True,
        finish_reason=FinishReason.STOP,
    )

    assert result.token_id == 9
    assert result.finished is True
    assert result.finish_reason is FinishReason.STOP


def test_fake_worker_satisfies_execution_protocol():
    assert isinstance(FakeExecutionWorker(), ExecutionWorkerProtocol)


def test_waiting_request_allows_prefill_but_not_decode():
    waiting = DecodeState(request_id="waiting", max_tokens=5)
    assert waiting.can_prefill() is True
    assert waiting.can_decode() is False

    coordinator = DecodeCoordinator(BatchExecutor())
    current = request("request")
    batch = coordinator.executor.create_batch([current], max_batch_size=1)
    with pytest.raises(RuntimeError, match="cannot decode from waiting"):
        coordinator.apply_decode_results(
            batch,
            [DecodeStepResult("request", token_id=1)],
        )


def test_successful_prefill_advances_request_to_decode():
    coordinator = DecodeCoordinator(BatchExecutor())
    current = request("request")
    batch = coordinator.executor.create_batch([current], max_batch_size=1)

    coordinator.apply_prefill_results(
        batch,
        [PrefillStepResult("request", 4, True, False)],
    )

    state = coordinator.get_state("request")
    assert state.phase is ExecutionPhase.DECODE
    assert state.can_decode() is True
    assert state.phase_history == [
        ExecutionPhase.WAITING,
        ExecutionPhase.PREFILL,
        ExecutionPhase.DECODE,
    ]


def test_decode_result_appends_token():
    coordinator = DecodeCoordinator(BatchExecutor())
    current = request("request")
    batch = coordinator.executor.create_batch([current], max_batch_size=1)
    coordinator.apply_prefill_results(
        batch,
        [PrefillStepResult("request", 4, True, False)],
    )

    coordinator.apply_decode_results(
        batch,
        [DecodeStepResult("request", token_id=11)],
    )

    assert coordinator.get_state("request").generated_tokens == [11]
    assert current.generated_tokens == [11]


def test_decode_error_is_isolated_from_other_request():
    coordinator = DecodeCoordinator(BatchExecutor())
    failed = request("failed")
    healthy = request("healthy")
    batch = coordinator.executor.create_batch([failed, healthy], max_batch_size=2)
    coordinator.apply_prefill_results(
        batch,
        [
            PrefillStepResult("failed", 4, True, False),
            PrefillStepResult("healthy", 4, True, False),
        ],
    )

    coordinator.apply_decode_results(
        batch,
        [
            DecodeStepResult("failed", error=ValueError("decode failed")),
            DecodeStepResult("healthy", token_id=3),
        ],
    )

    assert coordinator.get_state("failed").status is DecodeStatus.FAILED
    assert coordinator.get_state("healthy").status is DecodeStatus.ACTIVE
    assert coordinator.get_state("healthy").generated_tokens == [3]
    assert batch.requests == [healthy]
