"""CPU tensor batch adapter checks using fake cache/model/tokenizer only."""

from types import SimpleNamespace

import pytest
import torch

from minillm.continuous import BatchExecutor, ContinuousBatchWorkerProtocol
from minillm.engine import GenerationConfig, RequestState, RequestStatus
from minillm.runtime import RuntimeRequestContext
from minillm.workers import LoadedModel, TransformersBatchWorker


class FakeCache:
    def __init__(self):
        self.layers = []

    def update(self, keys, values, index):
        if index == len(self.layers):
            self.layers.append(SimpleNamespace(keys=keys, values=values))
        else:
            layer = self.layers[index]
            layer.keys = torch.cat([layer.keys, keys], dim=-2)
            layer.values = torch.cat([layer.values, values], dim=-2)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=True):
        if text == "bad":
            raise ValueError("invalid prompt")
        return {"short": [1, 2], "long": [1, 2, 3]}[text]

    def decode(self, tokens, skip_special_tokens=True):
        return str(tokens)


class FakeModel:
    config = SimpleNamespace()

    def __init__(self):
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        keys = ids.float()[:, None, :, None]
        cache = kwargs["past_key_values"]
        cache.update(keys, keys.clone(), 0)
        logits = torch.zeros(ids.shape[0], ids.shape[1], 10)
        logits.scatter_(2, ((ids + 1) % 10).unsqueeze(-1), 10)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def backend():
    model = FakeModel()
    worker = TransformersBatchWorker(LoadedModel(model, FakeTokenizer()), cache_factory=lambda _: FakeCache())
    return worker, model


def context(name, prompt):
    return RuntimeRequestContext(RequestState(request_id=name, prompt=prompt,
        generation_config=GenerationConfig(temperature=0, max_tokens=5)))


def test_two_request_prefill_single_forward_independent_cache_rows():
    worker, model = backend()
    assert isinstance(worker, ContinuousBatchWorkerProtocol)
    executor = BatchExecutor(worker)
    first, second = context("A", "short"), context("B", "long")
    executor.add(first)
    executor.add(second)
    executor.step()
    assert len(model.calls) == 1
    assert model.calls[0]["input_ids"].tolist() == [[0, 1, 2], [1, 2, 3]]
    assert model.calls[0]["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    first_cache, second_cache = worker.kv.get_batch_kv(["A", "B"])
    assert first_cache is not second_cache
    assert first_cache.layers[0].keys.data_ptr() != second_cache.layers[0].keys.data_ptr()
    assert first.request.kv_cache_ref is first_cache
    worker.close()
    executor.close()


def test_decode_batch_one_forward_updates_caches_and_preserves_mapping():
    worker, model = backend()
    executor = BatchExecutor(worker)
    first, second = context("A", "short"), context("B", "long")
    executor.add(first)
    executor.add(second)
    executor.step()
    original = worker.kv.get_batch_kv(["A", "B"])
    executor.step()  # First tokens from prefill logits.
    assert len(model.calls) == 1
    assert first.request.generated_tokens == [3]
    assert second.request.generated_tokens == [4]
    executor.step()
    assert len(model.calls) == 2
    assert model.calls[1]["input_ids"].tolist() == [[3], [4]]
    assert model.calls[1]["attention_mask"].tolist() == [[0, 1, 1, 1], [1, 1, 1, 1]]
    assert model.calls[1]["position_ids"].tolist() == [[2], [3]]
    updated = worker.kv.get_batch_kv(["B", "A"])
    assert updated[0] is second.request.kv_cache_ref
    assert updated[1] is first.request.kv_cache_ref
    assert first.request.kv_cache_ref is not original[0]
    assert first.request.generated_tokens == [3, 4]
    assert second.request.generated_tokens == [4, 5]
    executor.close()
    worker.close()


def test_invalid_prompt_isolated_from_good_request():
    worker, model = backend()
    executor = BatchExecutor(worker)
    bad, good = context("bad", "bad"), context("good", "long")
    executor.add(bad)
    executor.add(good)
    executor.step()
    assert bad.request.status is RequestStatus.FAILED
    assert good.request.status is RequestStatus.RUNNING
    assert model.calls[0]["input_ids"].shape[0] == 1
    executor.step()
    assert good.request.generated_tokens == [4]
    executor.close()
    worker.close()


def test_cpu_and_batch_limit_are_enforced():
    worker, _ = backend()
    with pytest.raises(ValueError, match="CPU"):
        TransformersBatchWorker(LoadedModel(FakeModel(), FakeTokenizer(), "cuda"))
    with pytest.raises(ValueError, match="1 or 2"):
        worker.prefill_batch([context(str(i), "short") for i in range(3)])


def test_decode_input_error_isolated_and_releases_affected_cache():
    worker, model = backend()
    executor = BatchExecutor(worker)
    bad, good = context("A", "short"), context("B", "long")
    executor.add(bad)
    executor.add(good)
    executor.step()
    executor.step()
    bad.request.generated_tokens.clear()  # Simulate malformed runtime input.
    executor.step()
    assert bad.request.status is RequestStatus.FAILED
    assert bad.request.kv_cache_ref is None
    assert good.request.status is RequestStatus.RUNNING
    assert good.request.generated_tokens == [4, 5]
    assert model.calls[-1]["input_ids"].shape == (1, 1)
    executor.close()
    worker.close()
