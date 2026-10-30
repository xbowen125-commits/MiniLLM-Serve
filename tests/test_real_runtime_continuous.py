"""Opt-in real CUDA Facade integration, local cached Qwen only."""

import pytest
import torch

from minillm.runtime import RuntimeFacade
from minillm.engine import GenerationConfig
from minillm.workers import BatchedTransformersWorker, TransformersModelLoader


@pytest.mark.slow
def test_real_facade_continuous_greedy_lifecycle():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable: not GPU validation')
    loaded = TransformersModelLoader('Qwen/Qwen2.5-0.5B-Instruct', device='cuda',
        dtype=torch.float32, local_files_only=True).load()
    prompts = ['The capital of France is', 'Explain what a computer does:', 'The largest ocean is']

    def run(indices):
        worker = BatchedTransformersWorker(loaded, slot_capacity=2)
        runtime = RuntimeFacade(worker, execution_mode='continuous', max_batch_size=2)
        contexts = []
        try:
            for index in indices:
                if len(indices) > 1 and index == 2:
                    continue
                contexts.append(runtime.submit(prompts[index], GenerationConfig(max_tokens=4, temperature=0),
                                               request_id=str(index)))
            runtime.step()
            assert all(runtime.get_result(c.request_id).metrics.prefill_tokens > 0 for c in contexts)
            assert all(c.decode_state.first_token_time is None for c in contexts)
            if len(indices) > 1:
                assert worker.pool.slot_id('0') != worker.pool.slot_id('1')
                runtime.cancel('1')
                assert worker.pool.snapshot()['allocated_slots'] == 1
                contexts.append(runtime.submit(prompts[2], GenerationConfig(max_tokens=4, temperature=0), request_id='2'))
            for _ in range(10):
                runtime.step()
                if runtime.snapshot().active_requests == 0:
                    break
            assert runtime.snapshot().active_requests == 0
            assert runtime.snapshot().failed_requests == 0
            assert all(c.request.status.name == ('CANCELLED' if c.request_id == '1' else 'FINISHED') for c in contexts)
            assert worker.pool.snapshot()['allocated_slots'] == 0
            return {c.request_id: tuple(c.request.generated_tokens) for c in contexts if c.request_id != '1'}
        finally:
            runtime.close(); runtime.close(); worker.close()
            assert worker.pool.snapshot()['allocated_slots'] == 0
    batched = run([0, 1, 2])
    assert batched['0'] == run([0])['0']
    assert batched['2'] == run([2])['2']
