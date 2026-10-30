"""Real ASGI + cached Qwen + CUDA; not a network-server or load benchmark."""

import asyncio
import json

import pytest
import torch

from minillm.config import Settings
from minillm.runtime.http_owner import ContinuousHTTPOwner
from minillm.server import create_app
from minillm.workers import TransformersModelLoader
from test_http_continuous import asgi_request, wait_idle


@pytest.mark.slow
def test_real_cuda_asgi_concurrency_cancel_shutdown():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this explicit integration test")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    settings = Settings(model_name=model_name, execution_mode="continuous", device="cuda",
        dtype="float32", max_concurrent_generations=2, inference_timeout_seconds=60)

    def owner_factory(engine, config):
        def load():
            # No network fallback; missing local files are a real test failure.
            loaded = TransformersModelLoader(model_name, device="cuda", dtype=torch.float32,
                local_files_only=True).load()
            engine.model = loaded.model
            engine.tokenizer = loaded.tokenizer
            engine.device = "cuda"
            engine.dtype = torch.float32
            if engine.tokenizer.pad_token_id is None:
                engine.tokenizer.pad_token_id = engine.tokenizer.eos_token_id
        engine.load = load
        return ContinuousHTTPOwner(engine, config)

    app = create_app(settings, continuous_owner_factory=owner_factory)
    evidence = {}

    async def wait_active():
        for _ in range(5000):
            if app.state.continuous_owner.status()["allocated_slots"]:
                return
            await asyncio.sleep(.001)
        pytest.fail("real request never acquired a KV slot")

    async def run():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        try:
            prompts = ["用中文解释什么是KV缓存。", "Count from one to ten."]
            results = await asyncio.gather(*(asgi_request(app, prompt=p, max_tokens=n)
                for p, n in zip(prompts, (4, 6))))
            assert [r[0] for r in results] == [200, 200]
            await wait_idle(app)
            assert app.state.continuous_owner.status()["peak_allocated_slots"] == 2
            # Independent legacy greedy reference, after owner is fully idle.
            reference = [app.state.engine.chat([{"role": "user", "content": p}], max_tokens=n,
                temperature=0) for p, n in zip(prompts, (4, 6))]
            for response, expected in zip(results, reference):
                assert response[1]["choices"][0]["message"]["content"] == expected.text
                assert response[1]["usage"]["completion_tokens"] == expected.completion_tokens
                assert response[1]["usage"]["prompt_tokens"] == expected.prompt_tokens
            disconnected = asyncio.Event()
            task = asyncio.create_task(asgi_request(app, prompt=prompts[0], max_tokens=128,
                disconnect=disconnected))
            await wait_active()
            disconnected.set()
            assert (await task)[0] == 503
            await wait_idle(app)
            active = asyncio.create_task(asgi_request(app, prompt=prompts[1], max_tokens=128))
            await wait_active()
            await lifespan.__aexit__(None, None, None)
            assert (await active)[0] == 503
            status = await wait_idle(app)
            evidence.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                cuda=torch.version.cuda, completed=status["completed_requests"],
                cancelled=status["cancelled_requests"], running=status["running_requests"],
                queued=status["queued_requests"], runtime=app.state.continuous_owner.status())
            assert status["completed_requests"] == 2 and status["cancelled_requests"] == 2
        finally:
            await lifespan.__aexit__(None, None, None)
        app.state.continuous_owner.thread.join(2)
        assert not app.state.continuous_owner.thread.is_alive()
    asyncio.run(run())
    print("ASGI_CUDA_EVIDENCE=" + json.dumps(evidence, ensure_ascii=False))
