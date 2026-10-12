"""Offline facade lifecycle and authoritative context tests."""

import pytest

from minillm.runtime import RuntimeFacade, RuntimeRequestContext
from minillm.contracts import PrefillStepResult, DecodeStepResult, ExecutionPhase
from minillm.engine import GenerationConfig, RequestStatus


class Worker:
    def __init__(self):
        self.calls = []

    def prefill(self, requests):
        self.calls.append(("prefill", tuple(r.request_id for r in requests)))
        return [PrefillStepResult(r.request_id, 3, True, False) for r in requests]

    def decode(self, requests):
        self.calls.append(("decode", tuple(r.request_id for r in requests)))
        return [DecodeStepResult(r.request_id, token_id=7) for r in requests]


def test_submit_creates_unique_context():
    runtime = RuntimeFacade(Worker())
    context = runtime.submit("hello", request_id="A")
    assert isinstance(context, RuntimeRequestContext)
    assert context.request.prompt == "hello"
    assert context.status is RequestStatus.WAITING
    assert runtime._coordinator.contexts["A"] is context
    with pytest.raises(ValueError, match="duplicate"):
        runtime.submit("another", request_id="A")


def test_step_updates_context_and_finishes_at_token_limit():
    worker = Worker()
    runtime = RuntimeFacade(worker)
    context = runtime.submit("hello", GenerationConfig(max_tokens=1), request_id="A")
    assert runtime.step() == ("A",)
    assert context.decode_state.execution_phase is ExecutionPhase.DECODE
    assert runtime.step() == ("A",)
    assert context.status is RequestStatus.FINISHED
    assert context.request.generated_tokens == [7]
    assert runtime.step() == ()
    assert worker.calls == [("prefill", ("A",)), ("decode", ("A",))]


def test_cancel_waiting_and_running_lifecycle():
    runtime = RuntimeFacade(Worker())
    waiting = runtime.submit("hello", request_id="A")
    assert runtime.cancel("A") is True
    assert waiting.status is RequestStatus.CANCELLED
    assert runtime.cancel("A") is False
    running = runtime.submit("hello", request_id="B")
    runtime.step()
    assert runtime.cancel("B") is True
    assert running.status is RequestStatus.CANCELLED
    assert runtime.step() == ()


def test_snapshot_and_result_metrics_are_prompt_free():
    runtime = RuntimeFacade(Worker())
    runtime.submit("private prompt", GenerationConfig(max_tokens=1), request_id="A")
    assert runtime.snapshot().active_requests == 1
    runtime.step()
    runtime.step()
    snapshot = runtime.snapshot()
    result = runtime.get_result("A")
    assert snapshot.finished_requests == 1
    assert snapshot.total_generated_tokens == 1
    assert result.generated_tokens == (7,)
    assert result.metrics.prefill_tokens == 3
    assert result.metrics.decode_tokens == 1
    assert result.metrics.decode_steps == 1
    assert result.metrics.first_token_latency_ms is not None
    assert "private prompt" not in repr(result)


def test_close_discards_contexts_but_preserves_results():
    runtime = RuntimeFacade(Worker())
    runtime.submit("hello", request_id="A")
    runtime.close()
    runtime.close()
    assert runtime._coordinator.contexts == {}
    assert runtime.get_result("A").status is RequestStatus.CANCELLED
    assert runtime.snapshot().active_requests == 0
    assert runtime.snapshot().cancelled_requests == 1
    with pytest.raises(RuntimeError, match="closed"):
        runtime.submit("new")
    with pytest.raises(RuntimeError, match="closed"):
        runtime.step()


def test_mixed_phases_are_selected_as_bounded_homogeneous_steps():
    worker = Worker()
    runtime = RuntimeFacade(worker, max_batch_size=1)
    runtime.submit("A", GenerationConfig(max_tokens=1), request_id="A")
    runtime.submit("B", GenerationConfig(max_tokens=1), request_id="B")
    for _ in range(4):
        runtime.step()
    assert worker.calls == [
        ("prefill", ("A",)), ("decode", ("A",)),
        ("prefill", ("B",)), ("decode", ("B",)),
    ]
