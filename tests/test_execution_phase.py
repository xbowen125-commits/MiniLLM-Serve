"""ExecutionPhase enum and worker-result compatibility tests."""

from minillm.batching import BatchStepResult
from minillm.execution import ExecutionPhase


def test_execution_phase_members():
    assert list(ExecutionPhase) == [
        ExecutionPhase.WAITING,
        ExecutionPhase.PREFILL,
        ExecutionPhase.DECODE,
        ExecutionPhase.FINISHED,
    ]
    assert [phase.value for phase in ExecutionPhase] == [
        "waiting",
        "prefill",
        "decode",
        "finished",
    ]


def test_batch_step_result_phase_is_optional_and_backward_compatible():
    legacy = BatchStepResult(request_id="legacy", token_id=1)
    prefill = BatchStepResult(
        request_id="prefill",
        phase=ExecutionPhase.PREFILL,
    )
    decode = BatchStepResult(
        request_id="decode",
        token_id=2,
        phase=ExecutionPhase.DECODE,
    )

    assert legacy.phase is None
    assert prefill.phase is ExecutionPhase.PREFILL
    assert decode.phase is ExecutionPhase.DECODE
