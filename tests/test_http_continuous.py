"""Actual ASGI calls with a CPU fake tensor backend; never download a model."""

import asyncio
import json
import threading
import time
from dataclasses import replace

import pytest

from minillm.config import Settings
from minillm.engine import GenerationConfig, RequestState, RequestStatus
from minillm.runtime.http_owner import ContinuousHTTPOwner
from minillm.runtime import RuntimeFacade
from minillm.server import create_app
from minillm.workers import BatchedTransformersWorker, LoadedModel
from test_transformers_batch_worker import FakeModel, FakeTokenizer, FakeCache


class Tokenizer(FakeTokenizer):
    chat_template = "fake"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return messages[-1]["content"]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join({3: "你", 4: "好", 5: "停", 6: "止"}.get(t, str(t)) for t in tokens if t != self.eos_token_id)


def application(*, worker_class=BatchedTransformersWorker, model=None, **options):
    settings = Settings(model_name="fake", execution_mode="continuous",
        max_concurrent_generations=options.pop("max_concurrent_generations", 2), **options)
    evidence = {}

    def factory(engine, config):
        def load():
            engine.model = model or FakeModel()
            engine.tokenizer = Tokenizer()
        engine.load = load

        def worker_factory(eng):
            worker = worker_class(LoadedModel(eng.model, eng.tokenizer),
                slot_capacity=2, cache_factory=lambda _: FakeCache())
            evidence["worker"] = worker
            return worker
        owner = ContinuousHTTPOwner(engine, config, worker_factory=worker_factory)
        evidence["owner"] = owner
        return owner
    return create_app(settings, continuous_owner_factory=factory), evidence


async def asgi_request(app, *, prompt="short", max_tokens=3, disconnect=None, **options):
    """Exercise routing, validation, handler and response serialization via ASGI."""
    body = json.dumps({"model": app.state.settings.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, **options}).encode()
    first = True
    never = asyncio.Event()
    messages = []

    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": body, "more_body": False}
        await (disconnect or never).wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions", "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8000)}, receive, send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    content = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], json.loads(content), dict(start["headers"])


async def wait_idle(app):
    for _ in range(500):
        status = await app.state.request_scheduler.snapshot()
        if status["running_requests"] == status["queued_requests"] == 0 and not app.state.cleanup_tasks:
            break
        await asyncio.sleep(.01)
    else:
        raise AssertionError("leases/background finalizers did not settle")
    runtime = app.state.continuous_owner.status()
    assert runtime["pending"] == runtime["allocated_slots"] == runtime["contexts"] == 0
    assert runtime["runtime_active_requests"] == 0
    assert runtime["claims"] == runtime["released_ids"] == runtime["finished_ids"] == 0
    return status


def test_defaults_and_capacity_validation():
    assert Settings().execution_mode == "legacy"
    with pytest.raises(ValueError):
        Settings(execution_mode="wrong")
    with pytest.raises(ValueError):
        Settings(execution_mode="continuous", max_concurrent_generations=3)
    with pytest.raises(ValueError):
        Settings(execution_mode="continuous", use_kv_cache=False)


def test_context_handoff_and_acknowledgement_contract():
    worker = BatchedTransformersWorker(LoadedModel(FakeModel(), Tokenizer()),
        slot_capacity=2, cache_factory=lambda _: FakeCache())
    runtime = RuntimeFacade(worker, execution_mode="continuous", max_batch_size=2)
    state = RequestState(request_id="same", generation_config=GenerationConfig(max_tokens=1, temperature=0))
    try:
        context = runtime.submit("short", state.generation_config, request_state=state)
        assert context.request is state
        with pytest.raises(RuntimeError, match="active"):
            runtime.acknowledge("same")
        runtime.step()
        assert state.started_time is not None and context.decode_state.first_token_time is None
        runtime.step()
        result = runtime.get_result("same")
        assert result.generated_tokens == (3,)
        runtime.acknowledge("same")
        runtime.acknowledge("same")
        assert all(n == 0 for n in runtime.retention().values())
        assert worker.pool.snapshot()["allocated_slots"] == 0
        assert result.generated_tokens == (3,)  # immutable copy survives forget
    finally:
        runtime.close()
        worker.close()


def test_fatal_owner_propagates_without_false_idle_or_shutdown_deadlock():
    app, evidence = application()
    async def run():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        owner = evidence["owner"]
        publish = owner._publish
        def failure(runtime, worker, pending):
            if pending:
                raise RuntimeError("fatal owner diagnostic failure")
            return publish(runtime, worker, pending)
        owner._publish = failure
        result = await asyncio.wait_for(asgi_request(app, max_tokens=100), 5)
        assert result[0] == 503 and result[1]["error"]["code"] == "cleanup_unconfirmed"
        assert (await app.state.request_scheduler.snapshot())["running_requests"] == 1
        assert owner.status()["ready"] is False
        with pytest.raises(RuntimeError, match="fatal owner"):
            await asyncio.wait_for(lifespan.__aexit__(None, None, None), 2)
        owner.thread.join(2)
        assert not owner.thread.is_alive()
        # Deliberate fatal test injection leaves ownership unconfirmed. Only
        # test teardown cleans its fake backend; production never reports idle.
        evidence["worker"].close()
    asyncio.run(run())


def test_startup_failure_never_falls_back():
    class Bad(BatchedTransformersWorker):
        max_batch_size = 1
    app, evidence = application(worker_class=Bad)
    async def run():
        with pytest.raises(ValueError, match="capacity mismatch"):
            async with app.router.lifespan_context(app):
                pytest.fail("startup must not succeed")
    asyncio.run(run())
    evidence["owner"].thread.join(2)
    assert not evidence["owner"].thread.is_alive()


def test_asgi_concurrent_json_stop_unicode_and_id_handoff():
    states = []
    class Observe(BatchedTransformersWorker):
        def prefill_batch(self, contexts):
            states.extend(c.request for c in contexts)
            return super().prefill_batch(contexts)
    app, evidence = application(worker_class=Observe)
    async def run():
        async with app.router.lifespan_context(app):
            a, b = await asyncio.gather(asgi_request(app, max_tokens=2),
                asgi_request(app, prompt="long", max_tokens=4, stop=["停止"]))
            assert a[0] == b[0] == 200
            assert a[1]["choices"][0]["message"]["content"] == "你好"
            assert b[1]["choices"][0]["message"]["content"] == "好"
            assert a[1]["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
            assert b[1]["usage"]["completion_tokens"] == 3
            assert b[1]["choices"][0]["finish_reason"] == "stop"
            assert a[1]["id"].startswith("chatcmpl-") and a[1]["id"] != b[1]["id"]
            assert {s.request_id for s in states} == {a[2][b"x-minillm-request-id"].decode(), b[2][b"x-minillm-request-id"].decode()}
            assert all(s.status is RequestStatus.FINISHED and s.kv_cache_ref is None for s in states)
            await wait_idle(app)
        assert not evidence["owner"].thread.is_alive() or evidence["owner"].stopped.done()
    asyncio.run(run())


def test_validation_before_admission():
    app, _ = application()
    async def run():
        async with app.router.lifespan_context(app):
            assert (await asgi_request(app, max_tokens=0))[0] == 422
            assert (await asgi_request(app, model="other"))[0] == 404
            assert (await app.state.request_scheduler.snapshot())["accepted_requests"] == 0
    asyncio.run(run())


def test_input_error_isolation_and_model_length_limit():
    app, _ = application(max_model_len=5)
    async def run():
        async with app.router.lifespan_context(app):
            bad, good = await asyncio.gather(asgi_request(app, prompt="bad"),
                asgi_request(app, prompt="long", max_tokens=20))
            assert bad[0] == 500 and good[0] == 200
            assert good[1]["usage"]["completion_tokens"] == 2
            assert good[1]["choices"][0]["finish_reason"] == "length"
            status = await wait_idle(app)
            assert status["completed_requests"] == status["failed_requests"] == 1
    asyncio.run(run())


def test_eos_usage_and_nonstream_text_adaptation():
    app, evidence = application()
    async def run():
        async with app.router.lifespan_context(app):
            evidence["worker"].loaded.tokenizer.eos_token_id = 4
            response = await asgi_request(app, max_tokens=10)
            assert response[0] == 200
            assert response[1]["choices"][0]["message"]["content"] == "你"
            assert response[1]["choices"][0]["finish_reason"] == "stop"
            assert response[1]["usage"]["completion_tokens"] == 2  # EOS counts
            await wait_idle(app)
    asyncio.run(run())


def test_waiting_disconnect_releases_queue_without_owner_handoff():
    entered = threading.Event()
    permit = threading.Event()
    class Block(FakeModel):
        def forward(self, **kwargs):
            entered.set()
            assert permit.wait(5)
            return super().forward(**kwargs)
    app, _ = application(model=Block(), max_concurrent_generations=1)
    async def run():
        async with app.router.lifespan_context(app):
            active = asyncio.create_task(asgi_request(app))
            while not entered.is_set():
                await asyncio.sleep(.001)
            disconnected = asyncio.Event()
            queued = asyncio.create_task(asgi_request(app, disconnect=disconnected))
            while not (await app.state.request_scheduler.snapshot())["queued_requests"]:
                await asyncio.sleep(.001)
            disconnected.set()
            with pytest.raises(asyncio.CancelledError):
                await queued
            status = await app.state.request_scheduler.snapshot()
            assert status["queued_requests"] == 0 and status["running_requests"] == 1
            permit.set()
            assert (await active)[0] == 200
            status = await wait_idle(app)
            assert status["cancelled_requests"] == 1
    try:
        asyncio.run(run())
    finally:
        permit.set()


@pytest.mark.parametrize("cancel_kind", ["disconnect", "task", "finalizer", "timeout", "shutdown"])
def test_cancellation_keeps_lease_until_safe_cleanup(cancel_kind):
    entered = threading.Event()
    proceed = threading.Event()
    class Block(FakeModel):
        def forward(self, **kwargs):
            entered.set()
            assert proceed.wait(5)
            return super().forward(**kwargs)
    app, evidence = application(model=Block(), inference_timeout_seconds=.03 if cancel_kind == "timeout" else 5)
    async def run():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        disconnected = asyncio.Event()
        task = asyncio.create_task(asgi_request(app, disconnect=disconnected, max_tokens=20))
        while not entered.is_set():
            await asyncio.sleep(.001)
        shutdown = None
        if cancel_kind == "disconnect":
            disconnected.set()
        elif cancel_kind == "task":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif cancel_kind == "finalizer":
            cleanup = next(iter(app.state.cleanup_tasks))
            cleanup.cancel()
            await asyncio.sleep(.001)
            cleanup.cancel()  # repeated cancellation cannot abandon the lease
        elif cancel_kind == "shutdown":
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.sleep(.06)
        assert (await app.state.request_scheduler.snapshot())["running_requests"] == 1
        assert not evidence["owner"].stopped.done()
        proceed.set()
        if cancel_kind != "task":
            response = await asyncio.wait_for(task, 5)
            assert response[0] == (504 if cancel_kind == "timeout" else 503)
        status = await wait_idle(app)
        assert status["inference_timeout_requests" if cancel_kind == "timeout" else "cancelled_requests"] == 1
        if shutdown:
            await shutdown
        else:
            await lifespan.__aexit__(None, None, None)
        await asyncio.wrap_future(evidence["owner"].close())  # idempotent
    try:
        asyncio.run(run())
    finally:
        proceed.set()


@pytest.mark.parametrize("violation", ["missing", "duplicate", "unknown", "forward"])
def test_failed_batches_release_slots_and_service_recovers(violation):
    class Bad(BatchedTransformersWorker):
        broken = True
        def prefill_batch(self, contexts):
            result = super().prefill_batch(contexts)
            if not self.broken:
                return result
            if violation == "forward":
                raise RuntimeError("batch forward failed")
            if violation == "missing":
                return []
            if violation == "duplicate":
                return result + result
            return [replace(result[0], request_id="unknown", token_result=None)]
    app, evidence = application(worker_class=Bad)
    async def run():
        async with app.router.lifespan_context(app):
            results = await asyncio.gather(asgi_request(app), asgi_request(app, prompt="long"))
            assert all(r[0] == 500 for r in results)
            status = await wait_idle(app)
            assert status["failed_requests"] == 2
            evidence["worker"].broken = False
            assert (await asgi_request(app))[0] == 200
            await wait_idle(app)
    asyncio.run(run())


def test_cleanup_retry_retains_lease_and_bounded_histories():
    attempted = threading.Event()
    permit = threading.Event()
    class Refuse(BatchedTransformersWorker):
        def release_request(self, rid):
            if not permit.is_set():
                attempted.set()
                raise RuntimeError("backend cleanup temporarily unavailable")
            return super().release_request(rid)
    app, evidence = application(worker_class=Refuse)
    async def run():
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(asgi_request(app, max_tokens=1))
            while not attempted.is_set():
                await asyncio.sleep(.001)
            assert not task.done()
            assert (await app.state.request_scheduler.snapshot())["running_requests"] == 1
            permit.set()
            assert (await task)[0] == 500  # original cleanup failure is not hidden
            await wait_idle(app)
            for _ in range(100):
                assert (await asgi_request(app, max_tokens=1))[0] == 200
                await wait_idle(app)
    try:
        asyncio.run(run())
    finally:
        permit.set()


def test_queue_timeout_full_and_delayed_arrival():
    entered = threading.Event()
    permit = threading.Event()
    class Slow(FakeModel):
        def forward(self, **kwargs):
            entered.set()
            assert permit.wait(5)
            return super().forward(**kwargs)
    app, _ = application(model=Slow(), max_concurrent_generations=1,
        request_queue_capacity=1, queue_timeout_seconds=.05)
    async def run():
        async with app.router.lifespan_context(app):
            a = asyncio.create_task(asgi_request(app))
            while not entered.is_set():
                await asyncio.sleep(.001)
            b = asyncio.create_task(asgi_request(app))
            await asyncio.sleep(.01)
            assert (await asgi_request(app))[0] == 429
            second = await b
            assert second[0] == 503 and second[1]["error"]["code"] == "queue_timeout"
            permit.set()
            assert (await a)[0] == 200
            assert (await asgi_request(app, prompt="long"))[0] == 200
            status = await wait_idle(app)
            assert status["queue_timeout_requests"] == 1
    try:
        asyncio.run(run())
    finally:
        permit.set()
