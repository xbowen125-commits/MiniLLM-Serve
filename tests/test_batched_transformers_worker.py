"""Small CPU tensors exercise batched backend and real-cache slot ownership."""

import pytest
import torch

from minillm.continuous import BatchExecutor, ContinuousBatchWorkerProtocol, IterationScheduler
from minillm.memory.slot_kv_pool import SlotKVPool
from minillm.workers import BatchedTransformersWorker, LoadedModel
from test_transformers_batch_worker import FakeCache, FakeModel, FakeTokenizer, context


def test_batch_forward_cache_isolation_and_release():
    model = FakeModel()
    worker = BatchedTransformersWorker(LoadedModel(model, FakeTokenizer()),
                                      cache_factory=lambda _: FakeCache())
    assert isinstance(worker, ContinuousBatchWorkerProtocol)
    executor = BatchExecutor(worker, scheduler=IterationScheduler(max_batch_size=2))
    a, b = context("a", "short"), context("b", "long")
    try:
        executor.add(a)
        executor.add(b)
        executor.step()
        assert len(model.calls) == 1 and model.calls[0]["input_ids"].shape[0] == 2
        assert worker.pool.slot_id("a") != worker.pool.slot_id("b")
        assert worker.pool.lookup("a").layers[0].keys.data_ptr() != worker.pool.lookup("b").layers[0].keys.data_ptr()
        executor.step()
        executor.step()
        assert len(model.calls) == 2
        assert model.calls[1]["input_ids"].tolist() == [[3], [4]]
        assert a.request.generated_tokens == [3, 4]
        worker.release_request("a")
        assert a.request.kv_cache_ref is None
        with pytest.raises(KeyError):
            worker.pool.lookup("a")
        assert worker.pool.lookup("b") is b.request.kv_cache_ref
    finally:
        executor.close()
        worker.close()
    assert worker.pool.snapshot()["allocated_slots"] == 0


def test_slot_capacity_reuse_and_update():
    pool = SlotKVPool(1)
    first, second = object(), object()
    slot = pool.allocate("a", first)
    with pytest.raises(MemoryError):
        pool.allocate("b", second)
    pool.update_request_cache("a", second)
    assert pool.lookup("a") is second
    pool.release("a")
    pool.release("a")
    assert pool.allocate("b", first) == slot


def test_cuda_unavailable_has_no_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        BatchedTransformersWorker(LoadedModel(FakeModel(), FakeTokenizer(), "cuda"))


def test_fake_greedy_single_and_batch_tokens_match():
    def run(prompts):
        worker = BatchedTransformersWorker(LoadedModel(FakeModel(), FakeTokenizer()),
                                          cache_factory=lambda _: FakeCache())
        executor = BatchExecutor(worker, scheduler=IterationScheduler(max_batch_size=2))
        contexts = [context(str(i), prompt) for i, prompt in enumerate(prompts)]
        try:
            for c in contexts:
                executor.add(c)
            for _ in range(6):
                executor.step()
            return [c.request.generated_tokens for c in contexts]
        finally:
            executor.close()
            worker.close()
    assert run(["short", "long"]) == [run(["short"])[0], run(["long"])[0]]
