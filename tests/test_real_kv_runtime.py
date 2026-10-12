"""Opt-in real-model runtime tests; local model cache only, no downloads."""

import gc
import os
import weakref

import pytest
import torch

from minillm.contracts import ExecutionPhase
from minillm.engine import GenerationConfig, RequestStatus
from minillm.runtime import RuntimeFacade
from minillm.workers import TransformersModelLoader, TransformersWorker

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def local_model():
    model_name = os.environ.get("MINILLM_RUNTIME_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = os.environ.get("MINILLM_RUNTIME_TEST_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        pytest.fail("MINILLM_RUNTIME_TEST_DEVICE must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA was explicitly requested but is unavailable")
    loader = TransformersModelLoader(
        model_name, device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
    )
    try:
        loaded = loader.load()
    except OSError as error:
        pytest.fail(
            "Real runtime tests require a complete local model/tokenizer cache. "
            "Set MINILLM_RUNTIME_TEST_MODEL to a cached model ID or local directory. "
            f"No download was attempted. Loader error type: {type(error).__name__}"
        )
    yield loaded
    del loaded
    gc.collect()


@pytest.fixture
def runtime_backend(local_model):
    worker = TransformersWorker(local_model)
    runtime = RuntimeFacade(worker, max_batch_size=2)
    try:
        yield runtime, worker
    finally:
        # Facade owns contexts; caller owns the worker's retained sessions.
        runtime.close()
        worker.close()


def submit(runtime, request_id="A", prompt="Continue the number sequence: 1, 2, 3,"):
    return runtime.submit(prompt, GenerationConfig(max_tokens=8, temperature=0), request_id=request_id)


def test_real_prefill_creates_cache_and_advances_phase(runtime_backend):
    runtime, _ = runtime_backend
    context = submit(runtime)
    assert context.decode_state.execution_phase is ExecutionPhase.WAITING
    runtime.step()
    assert context.decode_state.execution_phase is ExecutionPhase.DECODE
    assert context.request.status is RequestStatus.RUNNING
    assert context.request.kv_cache_ref is not None
    assert context.decode_state.prefill_tokens > 0
    assert context.request.kv_cache_ref.get_seq_length() == context.decode_state.prefill_tokens
    assert context.request.generated_tokens == []


def test_real_decode_reuses_cache_and_forwards_single_token(runtime_backend, local_model):
    runtime, _ = runtime_backend
    context = submit(runtime)
    runtime.step()
    cache = context.request.kv_cache_ref
    prompt_length = cache.get_seq_length()
    calls = []

    def observe(_module, _args, kwargs):
        calls.append((kwargs["input_ids"].shape[1],
                      kwargs["past_key_values"] is cache,
                      kwargs["attention_mask"].shape[1],
                      kwargs["position_ids"].tolist()))

    hook = local_model.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        runtime.step()
        # The first token comes from prefill logits, not a redundant forward.
        assert calls == []
        assert len(context.request.generated_tokens) == 1
        assert context.request.status is RequestStatus.RUNNING
        runtime.step()
    finally:
        hook.remove()
    assert calls == [(1, True, prompt_length + 1, [[prompt_length]])]
    assert len(context.request.generated_tokens) == 2
    assert context.decode_state.decode_tokens == 2
    assert context.request.kv_cache_ref is cache
    assert cache.get_seq_length() == prompt_length + 1


def test_real_requests_have_independent_kv_caches(runtime_backend):
    runtime, _ = runtime_backend
    first = submit(runtime, "A")
    second = submit(runtime, "B", "Continue the alphabet: A, B, C,")
    runtime.step()
    first_cache = first.request.kv_cache_ref
    second_cache = second.request.kv_cache_ref
    assert first_cache is not None and second_cache is not None
    assert first_cache is not second_cache
    runtime.step()
    assert len(first.request.generated_tokens) == 1
    assert len(second.request.generated_tokens) == 1
    assert first.request.generated_tokens is not second.request.generated_tokens


def test_real_release_request_drops_cache_reference(runtime_backend):
    runtime, worker = runtime_backend
    context = submit(runtime)
    runtime.step()
    cache_ref = weakref.ref(context.request.kv_cache_ref)
    worker.release_request(context.request_id)
    gc.collect()
    assert context.request.kv_cache_ref is None
    assert cache_ref() is None
    result = worker.decode([context])[0]
    assert result.error is not None
    assert context.request.kv_cache_ref is None
