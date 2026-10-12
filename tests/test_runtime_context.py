"""Single-owner lifecycle and independent decode observations."""

import subprocess
import sys
from dataclasses import fields

import pytest

from minillm.contracts import ExecutionPhase
from minillm.engine import RequestState, RequestStatus
from minillm.runtime import DecodeState, RuntimeRequestContext


def test_context_composes_request_and_execution_state():
    request = RequestState(request_id="A", prompt="hello")
    decode = DecodeState()
    context = RuntimeRequestContext(request, decode)
    assert context.request_state is request
    assert context.decode_state is decode
    assert context.metrics.request_state is request
    assert context.metrics.decode_state is decode
    assert context.request_id == "A"


def test_lifecycle_has_one_owner():
    request = RequestState(request_id="A")
    context = RuntimeRequestContext(request)
    context.start(now=1)
    assert context.status is RequestStatus.RUNNING
    context.finish(now=3)
    assert request.status is RequestStatus.FINISHED
    assert request.finished_time == 3
    assert context.metrics.total_latency_ms == 2000
    assert "status" not in {f.name for f in fields(DecodeState)}
    assert not hasattr(context.decode_state, "generated_tokens")
    with pytest.raises(RuntimeError):
        context.append_token(7)


def test_decode_observations_and_token_storage_are_separate():
    request = RequestState(request_id="A")
    context = RuntimeRequestContext(request)
    context.start(now=1)
    context.decode_state.transition_phase(ExecutionPhase.PREFILL)
    context.decode_state.record_prefill(12)
    context.decode_state.transition_phase(ExecutionPhase.DECODE)
    context.decode_state.record_step()
    context.append_token(7, now=2)
    context.decode_state.record_step()
    context.append_token(9, now=3)
    assert request.generated_tokens == [7, 9]
    assert request.status is RequestStatus.RUNNING
    assert context.decode_state.prefill_tokens == 12
    assert context.decode_state.decode_tokens == 2
    assert context.decode_state.decode_steps == 2
    assert context.decode_state.first_token_time == 2
    assert context.decode_state.last_token_time == 3
    assert context.metrics.first_token_latency_ms == 1000


def test_contracts_do_not_import_runtime():
    result = subprocess.run(
        [sys.executable, "-B", "-c", "import sys; import minillm.contracts; assert not any(n.startswith('minillm.runtime') for n in sys.modules)"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
