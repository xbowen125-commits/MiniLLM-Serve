"""Experimental backend contract tests with CPU tensors and fake model objects."""

from types import SimpleNamespace

import torch

from minillm.contracts import FinishReason
from minillm.engine import GenerationConfig, RequestState
from minillm.execution import ExecutionWorkerProtocol
from minillm.runtime import RuntimeRequestContext, RuntimeFacade
from minillm.workers import LoadedModel, TransformersWorker


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, prompt, return_tensors):
        if prompt == "invalid":
            raise ValueError("tokenizer failed")
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


class FakeModel:
    config = SimpleNamespace()

    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, **kwargs):
        if self.fail:
            raise RuntimeError("forward failed")
        self.calls.append(kwargs)
        cache = object()
        logits = torch.tensor([[[0.0, 1.0, 5.0]]])
        return SimpleNamespace(logits=logits, past_key_values=cache)


def make_worker():
    model = FakeModel()
    worker = TransformersWorker(LoadedModel(model, FakeTokenizer()), cache_factory=lambda config: object())
    return worker, model


def make_context(request_id="A", prompt="hello", max_tokens=4):
    return RuntimeRequestContext(RequestState(
        request_id=request_id, prompt=prompt,
        generation_config=GenerationConfig(max_tokens=max_tokens, temperature=0),
    ))


def test_protocol_and_prefill_context_result():
    worker, model = make_worker()
    context = make_context()
    assert isinstance(worker, ExecutionWorkerProtocol)
    result = worker.prefill([context])[0]
    assert result.error is None
    assert result.prompt_tokens == 3
    assert result.kv_cache_created and result.first_token_ready
    assert context.request.kv_cache_ref is not None
    assert context.request.generated_tokens == []
    assert model.calls[0]["input_ids"].shape == (1, 3)
    worker.close()
    assert context.request.kv_cache_ref is None


def test_decode_consumes_prefill_logits_then_updates_cache_with_one_token():
    worker, model = make_worker()
    context = make_context()
    worker.prefill([context])
    cache = context.request.kv_cache_ref
    first = worker.decode([context])[0]
    assert first.token_id == 2
    assert len(model.calls) == 1
    second = worker.decode([context])[0]
    assert second.error is None
    assert model.calls[1]["past_key_values"] is cache
    assert model.calls[1]["input_ids"].tolist() == [[2]]
    assert model.calls[1]["attention_mask"].shape == (1, 4)
    assert model.calls[1]["position_ids"].tolist() == [[3]]
    assert context.request.kv_cache_ref is not cache
    worker.close()


def test_error_isolation_and_missing_prefill():
    worker, _ = make_worker()
    bad, good = make_context("bad", "invalid"), make_context("good")
    results = worker.prefill([bad, good])
    assert isinstance(results[0].error, ValueError)
    assert results[1].error is None
    assert worker.decode([bad])[0].error is not None
    assert worker.decode([good])[0].token_id == 2
    worker.close()


def test_forward_failure_releases_cache():
    worker, model = make_worker()
    context = make_context()
    worker.prefill([context])
    worker.decode([context])
    model.fail = True
    result = worker.decode([context])[0]
    assert isinstance(result.error, RuntimeError)
    assert context.request.kv_cache_ref is None


def test_runtime_facade_adapter_and_terminal_cleanup():
    worker, model = make_worker()
    runtime = RuntimeFacade(worker)
    context = runtime.submit("hello", GenerationConfig(max_tokens=2, temperature=0), request_id="A")
    runtime.step()
    runtime.step()
    runtime.step()
    result = runtime.get_result("A")
    assert result.generated_tokens == (2, 2)
    assert result.finish_reason is FinishReason.LENGTH
    assert context.request.kv_cache_ref is None
    runtime.close()
    worker.close()


def test_independent_caches_and_cancel_cleanup():
    worker, _ = make_worker()
    first, second = make_context("A"), make_context("B")
    worker.prefill([first, second])
    assert first.request.kv_cache_ref is not second.request.kv_cache_ref
    first.cancel()
    worker.decode([second])
    assert first.request.kv_cache_ref is None
    assert second.request.kv_cache_ref is not None
    worker.close()


def test_loader_is_explicit_and_eos_returns_stop():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = 2

    class Loader:
        calls = 0

        def load(self):
            self.calls += 1
            return LoadedModel(model, tokenizer)

    loader = Loader()
    worker = TransformersWorker(loader=loader, cache_factory=lambda config: object())
    context = make_context()
    assert loader.calls == 0
    assert worker.prefill([context])[0].error is not None
    assert loader.calls == 0
    worker.load()
    assert loader.calls == 1
    assert worker.prefill([context])[0].error is None
    result = worker.decode([context])[0]
    assert result.finished is True
    assert result.finish_reason is FinishReason.STOP
    assert context.request.kv_cache_ref is None
