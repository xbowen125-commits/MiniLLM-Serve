"""DecodeState execution-phase transition tests."""

import pytest

from minillm.batching import DecodeState, FinishReason
from minillm.execution import ExecutionPhase


def test_legal_phase_lifecycle_and_history():
    state = DecodeState(request_id="request", max_tokens=4, created_time=1.0)

    state.transition_phase(ExecutionPhase.PREFILL, now=2.0)
    state.transition_phase(ExecutionPhase.DECODE, now=3.0)
    state.finish(FinishReason.STOP, now=4.0)

    assert state.phase is ExecutionPhase.FINISHED
    assert state.phase_history == [
        ExecutionPhase.WAITING,
        ExecutionPhase.PREFILL,
        ExecutionPhase.DECODE,
        ExecutionPhase.FINISHED,
    ]
    assert state.phase_updated_time == 4.0


def test_waiting_cannot_skip_prefill_and_enter_decode():
    state = DecodeState(request_id="request", max_tokens=4)

    with pytest.raises(RuntimeError, match="waiting -> decode"):
        state.transition_phase(ExecutionPhase.DECODE)


def test_first_token_moves_prefill_request_to_decode():
    state = DecodeState(request_id="request", max_tokens=4)
    state.start(now=1.0)
    state.transition_phase(ExecutionPhase.PREFILL, now=2.0)

    state.append_token(7, now=3.0)

    assert state.phase is ExecutionPhase.DECODE
    assert state.phase_history == [
        ExecutionPhase.WAITING,
        ExecutionPhase.PREFILL,
        ExecutionPhase.DECODE,
    ]


def test_repeated_phase_metadata_is_idempotent():
    state = DecodeState(request_id="request", max_tokens=4)
    state.transition_phase(ExecutionPhase.PREFILL, now=2.0)

    state.transition_phase(ExecutionPhase.PREFILL, now=3.0)

    assert state.phase_history == [ExecutionPhase.WAITING, ExecutionPhase.PREFILL]
    assert state.phase_updated_time == 2.0
