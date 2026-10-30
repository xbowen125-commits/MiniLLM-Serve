"""Real ASGI + controlled fake tensors: deterministic bounded transport tests."""

import asyncio
import json
import threading
import time

import pytest

from minillm.workers import BatchedTransformersWorker
from minillm.runtime.http_owner import HTTPCompletion, StreamHandle
from minillm.server import _ContinuousCleanupTask, _ContinuousStreamSession
from minillm.scheduler import RequestScheduler
from test_http_continuous import application, asgi_request, wait_idle


async def stream_asgi(app, *, prompt="short", max_tokens=3, send_hook=None,
                      disconnect=None, **options):
    payload = json.dumps({"model": "fake", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True, **options}).encode()
    first, received = True, []
    never = asyncio.Event()
    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": payload, "more_body": False}
        await (disconnect or never).wait()
        return {"type": "http.disconnect"}
    async def send(message):
        if send_hook:
            await send_hook(message)
        received.append(message)
    await app({"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions", "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8000)}, receive, send)
    start = next(m for m in received if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in received)
    events = [data.decode("utf-8") for data in body.split(b"\n\n") if data]
    events = [e[6:] for e in events if e.startswith("data: ")]
    return start, [e if e == "[DONE]" else json.loads(e) for e in events]


def assert_success(events):
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert len({c["id"] for c in chunks}) == 1
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}
    return "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)


def assert_error(events, code):
    assert events[-1] == "[DONE]"
    assert events[-2]["error"]["code"] == code
    assert all(c["choices"][0]["finish_reason"] is None
        for c in events if isinstance(c, dict) and "choices" in c)


class Paced(BatchedTransformersWorker):
    def decode_batch(self, contexts):
        time.sleep(.003)
        return super().decode_batch(contexts)


def test_finalizer_cancel_before_first_execution_is_intent_not_abandonment():
    async def run():
        class Lease:
            calls = 0
            async def release(self, outcome):
                assert outcome == "cancelled"
                self.calls += 1
        lease = Lease()
        handle = StreamHandle(1, threading.Event())
        session = _ContinuousStreamSession(None, handle, lease)
        task = _ContinuousCleanupTask(session, name="prestart-cancel-test")
        assert task.cancel() is False
        assert task.cancel() is False
        assert handle.cancel_event.is_set() and not task.cancelled()
        handle.completion.set_result(HTTPCompletion(None, "cancelled"))
        await asyncio.sleep(0)
        assert lease.calls == 0  # consumer still owns its output
        session.consumer_done.set()
        await task
        assert lease.calls == 1
        assert task.cancel() is False
    asyncio.run(run())


def test_repeated_finalization_releases_actual_scheduler_lease_once():
    async def run():
        scheduler = RequestScheduler(max_running=1, queue_capacity=1,
            queue_timeout_seconds=1, inference_timeout_seconds=1)
        lease = await scheduler.acquire()
        lease.request_state.mark_finished()
        handle = StreamHandle(1, lease.cancel_event)
        handle.completion.set_result(HTTPCompletion(None, "completed"))
        session = _ContinuousStreamSession(None, handle, lease)
        session.consumer_done.set()
        await asyncio.gather(session.finalize(), session.finalize())
        await session.finalize()
        status = await scheduler.snapshot()
        assert status["completed_requests"] == 1 and status["running_requests"] == 0
        await scheduler.close()
    asyncio.run(run())


def test_asgi_sse_ids_unicode_stop_eos_and_concurrency():
    app, evidence = application(worker_class=Paced, continuous_output_capacity=64)
    async def run():
        async with app.router.lifespan_context(app):
            a, b = await asyncio.gather(stream_asgi(app, max_tokens=3),
                stream_asgi(app, prompt="long", max_tokens=8, stop=["停止"]))
            assert a[0]["status"] == b[0]["status"] == 200
            assert b"text/event-stream" in dict(a[0]["headers"])[b"content-type"]
            assert assert_success(a[1]) == "你好停"
            assert assert_success(b[1]) == "好"
            assert b[1][-2]["choices"][0]["finish_reason"] == "stop"
            assert "停" not in "".join(c["choices"][0]["delta"].get("content", "") for c in b[1][:-1])
            assert a[1][0]["id"] != b[1][0]["id"]
            evidence["worker"].loaded.tokenizer.eos_token_id = 4
            eos = await stream_asgi(app, max_tokens=8)
            assert assert_success(eos[1]) == "你"
            assert eos[1][-2]["choices"][0]["finish_reason"] == "stop"
            await wait_idle(app)
            assert not app.state.stream_sessions
    asyncio.run(run())


@pytest.mark.parametrize("generation_length", [2, 40])
def test_bounded_overflow_cancels_only_slow_request_and_holds_transport_lease(generation_length):
    app, _ = application(worker_class=Paced, continuous_output_capacity=1)
    async def run():
        async with app.router.lifespan_context(app):
            entered, permit = asyncio.Event(), asyncio.Event()
            async def block(message):
                if message.get("body") and not entered.is_set():
                    entered.set()
                    await permit.wait()
            slow = asyncio.create_task(stream_asgi(app, max_tokens=generation_length, send_hook=block))
            await entered.wait()
            for _ in range(500):
                if app.state.continuous_owner.status().get("stream_overflows"):
                    break
                await asyncio.sleep(.002)
            else:
                pytest.fail("bounded output did not overflow")
            assert (await asgi_request(app, prompt="long", max_tokens=2))[0] == 200
            assert (await app.state.request_scheduler.snapshot())["running_requests"] == 1
            assert app.state.continuous_owner.status()["allocated_slots"] == 0
            permit.set()
            events = (await slow)[1]
            assert_error(events, "stream_buffer_overflow")
            status = await wait_idle(app)
            assert status["failed_requests"] == 1 and status["completed_requests"] == 1
    asyncio.run(run())


def test_runtime_forgets_only_copied_results_while_transport_still_consumes():
    app, _ = application(worker_class=Paced, continuous_output_capacity=64)
    async def run():
        async with app.router.lifespan_context(app):
            entered, permit = asyncio.Event(), asyncio.Event()
            async def block_finish(message):
                if b'"finish_reason":"length"' in message.get("body", b""):
                    entered.set()
                    await permit.wait()
            task = asyncio.create_task(stream_asgi(app, max_tokens=2, send_hook=block_finish))
            await entered.wait()
            status = app.state.continuous_owner.status()
            assert status["contexts"] == status["claims"] == status["allocated_slots"] == 0
            assert (await app.state.request_scheduler.snapshot())["running_requests"] == 1
            assert len(app.state.stream_sessions) == 1
            permit.set()
            assert assert_success((await task)[1]) == "你好"
            await wait_idle(app)
            assert not app.state.stream_sessions
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["disconnect", "task", "finalizer", "shutdown"])
def test_sse_cancellation_safe_boundaries_and_survivor(kind):
    app, _ = application(worker_class=Paced, continuous_output_capacity=64)
    async def run():
        async with app.router.lifespan_context(app):
            disconnect, entered, permit = asyncio.Event(), asyncio.Event(), asyncio.Event()
            async def block(message):
                if message.get("body") and b'"content"' in message["body"] and not entered.is_set():
                    entered.set()
                    await permit.wait()
            stream = asyncio.create_task(stream_asgi(app, max_tokens=100, disconnect=disconnect, send_hook=block))
            await entered.wait()
            if kind == "disconnect":
                disconnect.set()
            elif kind == "task":
                stream.cancel()
            elif kind == "finalizer":
                finalizer = next(iter(app.state.cleanup_tasks))
                finalizer.cancel()
                await asyncio.sleep(.001)
                finalizer.cancel()
            else:
                await app.state.begin_shutdown()
                await app.state.begin_shutdown()
            permit.set()
            if kind == "task":
                with pytest.raises(asyncio.CancelledError):
                    await stream
            else:
                result = await stream
                if kind in {"finalizer", "shutdown"}:
                    assert_error(result[1], "request_cancelled")
            status = await wait_idle(app)
            assert status["cancelled_requests"] == 1
            if kind != "shutdown":
                survivor = await stream_asgi(app, prompt="long", max_tokens=2)
                assert assert_success(survivor[1]) == "好停"
                await wait_idle(app)
    asyncio.run(run())


def test_sse_worker_error_after_headers_and_recovery():
    class Bad(Paced):
        calls = 0
        def decode_batch(self, contexts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected decode failure")
            return super().decode_batch(contexts)
    app, _ = application(worker_class=Bad)
    async def run():
        async with app.router.lifespan_context(app):
            result = await stream_asgi(app, max_tokens=8)
            assert result[0]["status"] == 200  # headers were already sent
            assert_error(result[1], "generation_failed")
            await wait_idle(app)
            assert assert_success((await stream_asgi(app, max_tokens=2))[1]) == "你好"
            await wait_idle(app)
    asyncio.run(run())


def test_sse_inference_timeout_is_error_not_finish():
    app, _ = application(worker_class=Paced, inference_timeout_seconds=.05)
    async def run():
        async with app.router.lifespan_context(app):
            assert_error((await stream_asgi(app, max_tokens=100))[1], "inference_timeout")
            status = await wait_idle(app)
            assert status["inference_timeout_requests"] == 1
    asyncio.run(run())


def test_sse_cleanup_failure_cannot_report_completion_or_idle():
    attempted, permit = threading.Event(), threading.Event()
    class Refuse(Paced):
        def release_request(self, rid):
            if not permit.is_set():
                attempted.set()
                raise RuntimeError("cleanup refusal")
            return super().release_request(rid)
    app, _ = application(worker_class=Refuse)
    async def run():
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(stream_asgi(app, max_tokens=2))
            while not attempted.is_set():
                await asyncio.sleep(.002)
            assert not task.done()
            assert (await app.state.request_scheduler.snapshot())["running_requests"] == 1
            permit.set()
            assert_error((await task)[1], "generation_failed")
            await wait_idle(app)
    try:
        asyncio.run(run())
    finally:
        permit.set()


def test_sse_send_timeout_cleans_up_and_forgets_consumers():
    app, _ = application(worker_class=Paced, continuous_send_timeout_seconds=.03,
        continuous_output_capacity=1)
    async def run():
        async with app.router.lifespan_context(app):
            async def blocked(message):
                if message.get("body"):
                    await asyncio.Event().wait()
            with pytest.raises(TimeoutError):
                await stream_asgi(app, send_hook=blocked, max_tokens=100)
            status = await wait_idle(app)
            assert status["failed_requests"] == 1
            assert not app.state.stream_sessions
    asyncio.run(run())


def test_disconnect_during_headers_before_body_starts_cancels_owner():
    app, _ = application(worker_class=Paced, continuous_output_capacity=64)
    async def run():
        async with app.router.lifespan_context(app):
            entered = asyncio.Event()
            async def blocked_headers(message):
                if message["type"] == "http.response.start":
                    entered.set()
                    await asyncio.Event().wait()
            task = asyncio.create_task(stream_asgi(app, send_hook=blocked_headers, max_tokens=100))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            status = await wait_idle(app)
            assert status["cancelled_requests"] == 1
            assert not app.state.stream_sessions
    asyncio.run(run())


def test_many_consumed_sse_results_are_forgotten():
    app, _ = application(worker_class=Paced)
    async def run():
        async with app.router.lifespan_context(app):
            for _ in range(100):
                assert assert_success((await stream_asgi(app, max_tokens=1))[1]) == "你"
                await wait_idle(app)
                assert not app.state.stream_sessions
    asyncio.run(run())


def test_fatal_owner_sse_reports_unconfirmed_cleanup_without_false_idle():
    app, evidence = application(worker_class=Paced)
    async def run():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        owner = evidence["owner"]
        publish = owner._publish
        def fatal(runtime, worker, pending):
            if pending:
                raise RuntimeError("fatal SSE owner failure")
            publish(runtime, worker, pending)
        owner._publish = fatal
        result = await stream_asgi(app, max_tokens=100)
        assert_error(result[1], "cleanup_unconfirmed")
        status = await app.state.request_scheduler.snapshot()
        assert status["running_requests"] == 1  # deliberately NOT acknowledged
        with pytest.raises(RuntimeError, match="fatal SSE"):
            await asyncio.wait_for(lifespan.__aexit__(None, None, None), 2)
        owner.thread.join(2)
        assert not owner.thread.is_alive()
        evidence["worker"].close()  # isolated fake test teardown, not production recovery
    asyncio.run(run())
