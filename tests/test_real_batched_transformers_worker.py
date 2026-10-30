"""Opt-in real GPU correctness test, local cache only; never downloads."""

import os
import pytest
import torch

from minillm.continuous import BatchExecutor, IterationScheduler
from minillm.engine import GenerationConfig, RequestState
from minillm.runtime import RuntimeRequestContext
from minillm.workers import BatchedTransformersWorker, TransformersModelLoader


@pytest.mark.slow
def test_real_gpu_greedy_batch_one_matches_two():
    if not torch.cuda.is_available():
        pytest.skip("real GPU validation unavailable: CUDA required")
    loaded = TransformersModelLoader(os.environ.get("MINILLM_BATCH_TEST_MODEL",
        "Qwen/Qwen2.5-0.5B-Instruct"), device="cuda", dtype=torch.float32,
        local_files_only=True).load()
    prompts = ["The capital of France is", "Explain the purpose of a computer in one sentence:"]

    def run(texts):
        worker = BatchedTransformersWorker(loaded, slot_capacity=2)
        executor = BatchExecutor(worker, scheduler=IterationScheduler(max_batch_size=2))
        contexts = [RuntimeRequestContext(RequestState(request_id=str(i), prompt=text,
            generation_config=GenerationConfig(max_tokens=6, temperature=0))) for i, text in enumerate(texts)]
        try:
            for c in contexts:
                executor.add(c)
            executor.step()
            if len(contexts) == 2:
                assert worker.pool.slot_id("0") != worker.pool.slot_id("1")
                assert contexts[0].request.kv_cache_ref is not contexts[1].request.kv_cache_ref
            for _ in range(6):
                executor.step()
            assert all(c.request.is_terminal for c in contexts)
            return [tuple(c.request.generated_tokens) for c in contexts]
        finally:
            executor.close()
            worker.close()
            assert worker.pool.snapshot()["allocated_slots"] == 0
    assert run(prompts) == [run([p])[0] for p in prompts]
