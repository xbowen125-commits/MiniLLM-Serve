"""server 层单元测试：直接调用路由，不加载真实模型。"""

import asyncio
import json
import threading
import time

import pytest
from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from minillm.config import Settings
from minillm.engine import GenerationChunk, GenerationResult
from minillm.schemas import ChatCompletionRequest
from minillm.server import create_app

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(autouse=True)
def run_thread_calls_inline(monkeypatch):
    """当前 Python 3.14 沙箱的线程池会挂起；API 单测只替换调度层。"""

    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    async def inline_shield(awaitable):
        return await awaitable

    async def inline_wait_for(awaitable, timeout):
        return await awaitable

    async def inline_sleep(delay):
        time.sleep(min(delay, 0.001))

    class BackgroundTask:
        def __init__(self, awaitable):
            self._awaitable = awaitable
            self._result = None
            self._error = None
            self._callbacks = []
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def _run(self):
            iterator = self._awaitable.__await__()
            try:
                yielded = next(iterator)
            except StopIteration as exc:
                self._result = exc.value
            except BaseException as exc:
                self._error = exc
            else:
                self._error = AssertionError(f"background coroutine unexpectedly blocked: {yielded!r}")
            for callback in self._callbacks:
                callback(self)

        def __await__(self):
            self._thread.join()
            if self._error is not None:
                raise self._error
            if False:
                yield None
            return self._result

        def add_done_callback(self, callback):
            if self._thread.is_alive():
                self._callbacks.append(callback)
            else:
                callback(self)

        def done(self):
            return not self._thread.is_alive()

        def exception(self):
            if threading.current_thread() is not self._thread:
                self._thread.join()
            return self._error

    monkeypatch.setattr(asyncio, "to_thread", inline)
    monkeypatch.setattr(asyncio, "shield", inline_shield)
    monkeypatch.setattr(asyncio, "wait_for", inline_wait_for)
    monkeypatch.setattr(asyncio, "sleep", inline_sleep)
    monkeypatch.setattr(
        asyncio,
        "create_task",
        lambda awaitable, **_kwargs: BackgroundTask(awaitable),
    )


class FakeEngine:
    model_name = MODEL
    is_loaded = True

    def __init__(self, result: GenerationResult | None = None, error: Exception | None = None):
        self.result = result or GenerationResult("你好，世界！", "stop", 8, 6)
        self.error = error
        self.calls: list[dict] = []

    def _record(self, messages, max_tokens, temperature, top_p, top_k, stop=None, seed=None, **extra):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "stop": stop,
                "seed": seed,
                **extra,
            }
        )

    def chat(
        self,
        messages,
        max_tokens,
        temperature,
        top_p,
        top_k,
        stop=None,
        seed=None,
        cancel_event=None,
    ):
        self._record(
            messages,
            max_tokens,
            temperature,
            top_p,
            top_k,
            stop,
            seed,
            cancel_event=cancel_event,
        )
        if self.error:
            raise self.error
        return self.result

    def stream_chat(
        self,
        messages,
        max_tokens,
        temperature,
        top_p,
        top_k,
        stop=None,
        seed=None,
        cancel_event=None,
    ):
        self._record(
            messages, max_tokens, temperature, top_p, top_k, stop, seed, cancel_event=cancel_event
        )
        if self.error:
            raise self.error
        yield GenerationChunk("你", None, 8, 1)
        yield GenerationChunk("好", None, 8, 2)
        yield GenerationChunk("", "stop", 8, 2)


def make_app(engine: FakeEngine, **settings_overrides):
    app = create_app(Settings(model_name=engine.model_name, **settings_overrides))
    app.state.engine = engine
    return app


def route_endpoint(app, path: str, method: str):
    return next(route.endpoint for route in app.routes if route.path == path and method in route.methods)


def make_request(app, path="/v1/chat/completions"):
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
            "app": app,
        }
    )


async def call_completion(app, payload):
    endpoint = route_endpoint(app, "/v1/chat/completions", "POST")
    req = ChatCompletionRequest.model_validate(payload)
    return await endpoint(req, make_request(app), Response())


def run_immediate(awaitable):
    """驱动不应挂起等待 I/O 的测试协程，绕过当前沙箱的 asyncio.run 问题。"""
    iterator = awaitable.__await__()
    try:
        yielded = next(iterator)
    except StopIteration as exc:
        return exc.value
    raise AssertionError(f"测试协程意外挂起: {yielded!r}")


def consume_stream_immediate(response):
    parts = []
    while True:
        try:
            parts.append(run_immediate(anext(response.body_iterator)))
        except StopAsyncIteration:
            return b"".join(parts)


def parse_sse(body: bytes):
    return [line[6:] for line in body.decode("utf-8").splitlines() if line.startswith("data: ")]


def test_chat_completions_non_stream_backwards_compatible():
    response = run_immediate(
        call_completion(
            make_app(FakeEngine()), {"model": MODEL, "messages": [{"role": "user", "content": "你好"}]}
        )
    )
    body = response.model_dump()
    assert body["object"] == "chat.completion"
    assert body["model"] == MODEL
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "你好，世界！"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14}


def test_chat_completions_params_forwarded():
    engine = FakeEngine()
    run_immediate(
        call_completion(
            make_app(engine),
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32,
                "temperature": 0.0,
                "top_p": 0.8,
                "top_k": 12,
                "stop": "END",
                "seed": 42,
            },
        )
    )
    call = engine.calls[0]
    assert call["max_tokens"] == 32
    assert call["temperature"] == 0.0
    assert call["top_p"] == 0.8
    assert call["top_k"] == 12
    assert call["stop"] == ["END"]
    assert call["seed"] == 42


def test_stream_sse_headers_chunk_shape_order_and_done():
    response = run_immediate(
        call_completion(
            make_app(FakeEngine()),
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    )
    raw = consume_stream_immediate(response)
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"

    events = parse_sse(raw)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    contents = [
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if "content" in chunk["choices"][0]["delta"]
    ]
    assert contents == ["你", "好"]
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_generation_error_is_an_sse_error_then_done():
    response = run_immediate(
        call_completion(
            make_app(FakeEngine(error=RuntimeError("boom"))),
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    )
    events = parse_sse(consume_stream_immediate(response))
    assert json.loads(events[-2])["error"]["type"] == "internal_error"
    assert "boom" in json.loads(events[-2])["error"]["message"]
    assert events[-1] == "[DONE]"


def test_stream_inference_timeout_is_sse_error_then_done():
    class TimeoutEngine(FakeEngine):
        def stream_chat(self, *args, cancel_event=None, **kwargs):
            while not cancel_event.is_set():
                time.sleep(0.001)
            if False:
                yield

    app = make_app(TimeoutEngine(), inference_timeout_seconds=0.005)
    response = run_immediate(
        call_completion(
            app,
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    )
    events = parse_sse(consume_stream_immediate(response))
    assert json.loads(events[-2])["error"]["code"] == "inference_timeout"
    assert events[-1] == "[DONE]"
    assert run_immediate(app.state.request_scheduler.snapshot())["running_requests"] == 0


def test_stream_close_sets_cancel_event():
    class SlowEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.cancel_observed = threading.Event()

        def stream_chat(self, *args, cancel_event=None, **kwargs):
            try:
                while not cancel_event.is_set():
                    yield GenerationChunk("x", None, 1, 1)
                    time.sleep(0.01)
            finally:
                self.cancel_observed.set()

    engine = SlowEngine()
    app = make_app(engine)
    response = run_immediate(
        call_completion(
            app,
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    )
    stream = response.body_iterator
    run_immediate(anext(stream))  # assistant role chunk
    run_immediate(stream.aclose())  # 模拟客户端断开，触发生成器 finally
    assert engine.cancel_observed.wait(1)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        status = run_immediate(app.state.request_scheduler.snapshot())
        if status["cancelled_requests"] == 1:
            break
        time.sleep(0.001)
    assert status["cancelled_requests"] == 1
    assert status["running_requests"] == 0


def test_cancelled_stream_body_cannot_skip_independent_finalizer():
    class BlockingEngine(FakeEngine):
        def stream_chat(self, *args, cancel_event=None, **kwargs):
            while not cancel_event.is_set():
                yield GenerationChunk("x", None, 1, 1)
                time.sleep(0.002)

    app = make_app(BlockingEngine())
    response = run_immediate(
        call_completion(
            app,
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    )
    stream = response.body_iterator
    run_immediate(anext(stream))
    with pytest.raises(asyncio.CancelledError):
        run_immediate(stream.athrow(asyncio.CancelledError()))

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        status = run_immediate(app.state.request_scheduler.snapshot())
        if status["running_requests"] == 0:
            break
        time.sleep(0.001)
    assert status["running_requests"] == 0
    assert status["queued_requests"] == 0
    assert status["cancelled_requests"] == 1


def test_model_not_found():
    response = run_immediate(
        call_completion(
            make_app(FakeEngine()),
            {"model": "some-other-model", "messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert response.status_code == 404
    assert json.loads(response.body)["error"]["type"] == "model_not_found"


def test_non_stream_internal_error_shape():
    response = run_immediate(
        call_completion(
            make_app(FakeEngine(error=RuntimeError("boom"))),
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"]["type"] == "internal_error"
    assert "boom" in body["error"]["message"]


def test_invalid_request_and_error_handler():
    app = make_app(FakeEngine())
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({"model": MODEL, "messages": [{"role": "user"}]})

    handler = app.exception_handlers[RequestValidationError]
    exc = RequestValidationError(
        [{"type": "missing", "loc": ("body", "messages", 0, "content"), "msg": "Field required", "input": {}}]
    )
    response = run_immediate(handler(make_request(app), exc))
    assert response.status_code == 422
    body = json.loads(response.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "validation_error"


def test_health():
    app = make_app(FakeEngine())
    endpoint = route_endpoint(app, "/health", "GET")
    result = endpoint(make_request(app, "/health"))
    assert result["status"] == "ok"
    assert result["model"] == MODEL
    assert result["loaded"] is True
    assert result["use_kv_cache"] is True
    assert result["versions"]["transformers"]


def test_queue_full_returns_429_before_generation():
    engine = FakeEngine()
    app = make_app(engine, request_queue_capacity=0)
    scheduler = app.state.request_scheduler
    occupied = run_immediate(scheduler.acquire())
    response = run_immediate(
        call_completion(
            app,
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert response.status_code == 429
    assert json.loads(response.body)["error"]["code"] == "queue_full"
    assert engine.calls == []
    run_immediate(occupied.release("completed"))


def test_non_stream_inference_timeout_waits_for_worker_cleanup(monkeypatch):
    class CancelAwareEngine(FakeEngine):
        cancel_seen = False

        def chat(self, *args, cancel_event=None, **kwargs):
            while not cancel_event.is_set():
                time.sleep(0.001)
            self.cancel_seen = True
            return self.result

    async def timeout(awaitable, timeout):
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", timeout)
    engine = CancelAwareEngine()
    app = make_app(engine, inference_timeout_seconds=0.01)
    response = run_immediate(
        call_completion(
            app,
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert response.status_code == 504
    assert json.loads(response.body)["error"]["code"] == "inference_timeout"
    assert engine.cancel_seen is True
    status = run_immediate(app.state.request_scheduler.snapshot())
    assert status["running_requests"] == 0
    assert status["inference_timeout_requests"] == 1


def test_scheduler_status_has_no_request_content():
    app = make_app(FakeEngine())
    endpoint = route_endpoint(app, "/scheduler/status", "GET")
    status = run_immediate(endpoint(make_request(app, "/scheduler/status")))
    assert status["running_requests"] == 0
    assert status["queued_requests"] == 0
    assert "prompt" not in json.dumps(status).lower()
