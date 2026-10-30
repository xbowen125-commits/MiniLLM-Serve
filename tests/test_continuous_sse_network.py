"""Real Uvicorn/TCP lifecycle checks (CPU fake backend unless marked slow).

These are socket tests, not TestClient or direct-ASGI calls. Small responses fit
in the operating-system send buffer: the slow-reader test proves isolation, not
that TCP backpressure necessarily reaches the runtime's bounded output queue.
Deterministic output-queue overflow belongs to the ASGI/unit tests.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import socket
import threading
import time

import pytest
import uvicorn

from minillm.config import Settings
from minillm.continuous_server import ContinuousServer
from minillm.engine import GenerationChunk
from minillm.runtime.http_owner import ContinuousHTTPOwner
from minillm.server import create_app
from minillm.workers import BatchedTransformersWorker, TransformersModelLoader, LoadedModel
from test_http_continuous import application, Tokenizer
from test_transformers_batch_worker import FakeModel


class TCPServer:
    """Bounded local test server with its own actual Uvicorn event loop."""

    def __init__(self, app):
        self.app = app
        self.loop = None
        self.failure = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(128)
        self.port = self.socket.getsockname()[1]
        server_class = ContinuousServer if app.state.continuous_owner is not None else uvicorn.Server
        self.server = server_class(uvicorn.Config(app, host="127.0.0.1",
            port=self.port, log_level="error", access_log=False,
            timeout_graceful_shutdown=8, lifespan="on"))
        self.thread = threading.Thread(target=self._run, name="test-uvicorn-tcp", daemon=True)

    def _run(self):
        async def serve():
            self.loop = asyncio.get_running_loop()
            await self.server.serve(sockets=[self.socket])
        try:
            asyncio.run(serve())
        except BaseException as exc:
            self.failure = exc

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 90
        while not self.server.started:
            if self.failure is not None or not self.thread.is_alive():
                raise RuntimeError("Uvicorn failed to start") from self.failure
            if time.monotonic() >= deadline:
                raise TimeoutError("Uvicorn startup exceeded 90 seconds")
            time.sleep(.01)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.thread.is_alive():
            # This is the same exit flag Uvicorn's signal handler sets. The
            # continuous launcher itself must stop admission before draining.
            self.server.should_exit = True
            self.thread.join(15)
        self.socket.close()
        assert not self.thread.is_alive(), "Uvicorn thread did not stop"
        if self.failure is not None:
            raise self.failure
        owner = self.app.state.continuous_owner
        if owner is not None:
            owner.thread.join(2)
            assert not owner.thread.is_alive(), "runtime owner survived shutdown"
            assert owner.stopped.done() and owner.stopped.exception() is None
        assert not self.app.state.cleanup_tasks, "finalizers survived shutdown"
        assert not self.app.state.stream_sessions, "consumers survived shutdown"

    def open_stream(self, *, prompt="short", max_tokens=8, **options):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        payload = {"model": self.app.state.settings.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True, **options}
        connection.request("POST", "/v1/chat/completions", body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200, response.read().decode()
        assert response.getheader("Content-Type").startswith("text/event-stream")
        assert response.getheader("X-MiniLLM-Request-ID")
        return connection, response

    def status(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("GET", "/scheduler/status")
            response = connection.getresponse()
            assert response.status == 200
            return json.loads(response.read())
        finally:
            connection.close()

    def settle(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = self.status()
            runtime = status.get("continuous", {})
            if status["running_requests"] == status["queued_requests"] == 0 and not runtime.get("pending", 0):
                async def finalizers():
                    return len(self.app.state.cleanup_tasks)
                if asyncio.run_coroutine_threadsafe(finalizers(), self.loop).result(5) == 0:
                    break
            time.sleep(.01)
        else:
            raise AssertionError(f"TCP request resources did not settle: {status}")
        for key in ("pending", "allocated_slots", "contexts", "claims", "released_ids",
                    "finished_ids", "runtime_active_requests", "pending_commands", "stream_consumers"):
            assert runtime.get(key, 0) == 0, (key, runtime)
        return status

    def stopped_status(self):
        self.thread.join(15)
        assert not self.thread.is_alive(), "server failed to drain during real Uvicorn shutdown"
        status = asyncio.run(self.app.state.request_scheduler.snapshot())
        owner = self.app.state.continuous_owner
        status["continuous"] = owner.status() if owner is not None else {}
        if owner is not None:
            status["continuous"].update(stream_consumers=len(self.app.state.stream_sessions),
                pending_finalizers=len(self.app.state.cleanup_tasks), owner_thread_alive=owner.thread.is_alive())
        assert status["running_requests"] == status["queued_requests"] == 0
        for key in ("pending", "allocated_slots", "contexts", "claims", "released_ids",
                    "finished_ids", "runtime_active_requests", "pending_commands", "stream_consumers"):
            assert status["continuous"].get(key, 0) == 0, status
        assert not self.app.state.cleanup_tasks
        assert not self.app.state.stream_sessions
        return status


def read_event(response):
    """Read one SSE event through HTTP chunk framing, decoding UTF-8 strictly."""
    data = []
    while True:
        line = response.readline()
        if not line:
            if data:
                break
            raise AssertionError("TCP stream ended without [DONE]")
        if line in (b"\n", b"\r\n"):
            if data:
                break
            continue
        if line.startswith(b"data: "):
            data.append(line[6:].rstrip(b"\r\n"))
    text = b"\n".join(data).decode("utf-8", errors="strict")
    return text if text == "[DONE]" else json.loads(text)


def read_all(response, *, delay=0):
    events = []
    for _ in range(5000):
        event = read_event(response)
        events.append(event)
        if event == "[DONE]":
            return events
        if delay:
            time.sleep(delay)
    raise AssertionError("SSE event limit exceeded")


def stream_request(server, **options):
    connection, response = server.open_stream(**options)
    try:
        return read_all(response)
    finally:
        response.close()
        connection.close()


def assert_normal(events, *, finish="length"):
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert chunks and all("error" not in event for event in chunks), events
    assert {chunk["object"] for chunk in chunks} == {"chat.completion.chunk"}
    ids = {chunk["id"] for chunk in chunks}
    assert len(ids) == 1 and next(iter(ids)).startswith("chatcmpl-")
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[-1]["choices"][0]["finish_reason"] == finish
    assert all(chunk["choices"][0]["finish_reason"] is None for chunk in chunks[:-1])
    return "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)


class PacedWorker(BatchedTransformersWorker):
    """CPU-only pacing exposes real transport disconnects, not GPU timing."""

    def decode_batch(self, contexts):
        time.sleep(.01)
        return super().decode_batch(contexts)


def test_tcp_sse_normal_unicode_stop_and_concurrent_requests():
    app, _ = application(worker_class=PacedWorker, continuous_output_capacity=64)
    with TCPServer(app) as server:
        assert assert_normal(stream_request(server, max_tokens=2)) == "你好"
        assert assert_normal(stream_request(server, prompt="long", max_tokens=8,
            stop=["停止"]), finish="stop") == "好"
        with ThreadPoolExecutor(max_workers=2) as clients:
            one = clients.submit(stream_request, server, max_tokens=3)
            two = clients.submit(stream_request, server, prompt="long", max_tokens=5)
            first, second = one.result(15), two.result(15)
        assert assert_normal(first) == "你好停"
        assert assert_normal(second) == "好停止78"
        assert first[0]["id"] != second[0]["id"]
        status = server.settle()
        assert status["completed_requests"] == 4
        print("TCP_CPU_NORMAL_EVIDENCE=" + json.dumps(status, ensure_ascii=False))


def test_tcp_partial_disconnect_does_not_cancel_survivor():
    app, _ = application(worker_class=PacedWorker, continuous_output_capacity=64)
    with TCPServer(app) as server, ThreadPoolExecutor(max_workers=1) as client:
        survivor = client.submit(stream_request, server, prompt="long", max_tokens=18)
        connection, response = server.open_stream(max_tokens=128)
        for _ in range(30):
            event = read_event(response)
            if isinstance(event, dict) and event.get("choices", [{}])[0].get("delta", {}).get("content"):
                break
        else:
            pytest.fail("no partial token text before disconnect")
        if connection.sock is not None:
            connection.sock.shutdown(socket.SHUT_RDWR)
        response.close()
        connection.close()
        assert_normal(survivor.result(15))
        status = server.settle()
        assert status["completed_requests"] == 1
        assert status["cancelled_requests"] == 1
        print("TCP_CPU_DISCONNECT_EVIDENCE=" + json.dumps(status, ensure_ascii=False))


def test_tcp_slow_reader_keeps_peer_progressing():
    app, _ = application(worker_class=PacedWorker, continuous_output_capacity=64)
    with TCPServer(app) as server, ThreadPoolExecutor(max_workers=1) as client:
        connection, response = server.open_stream(max_tokens=16)
        peer = client.submit(stream_request, server, prompt="long", max_tokens=6)
        # Intentionally leave the socket unread while a second client completes.
        # This bounded test does not claim to saturate the OS TCP send buffer.
        assert_normal(peer.result(15))
        assert_normal(read_all(response, delay=.01))
        response.close()
        connection.close()
        status = server.settle()
        assert status["completed_requests"] == 2


def test_tcp_error_after_headers_is_not_normal_finish_and_recovers():
    class OneShotFailure(PacedWorker):
        decode_calls = 0
        broken = True

        def decode_batch(self, contexts):
            self.decode_calls += 1
            if self.broken and self.decode_calls == 3:
                self.broken = False
                raise RuntimeError("injected network batch decode failure")
            return super().decode_batch(contexts)

    app, _ = application(worker_class=OneShotFailure, continuous_output_capacity=64)
    with TCPServer(app) as server:
        events = stream_request(server, max_tokens=16)
        assert events[-1] == "[DONE]"
        assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert any(isinstance(event, dict) and "error" in event for event in events)
        assert not any(isinstance(event, dict) and event.get("choices", [{}])[0].get("finish_reason")
            for event in events)
        assert_normal(stream_request(server, max_tokens=2))
        status = server.settle()
        assert status["failed_requests"] == status["completed_requests"] == 1


def test_tcp_shutdown_active_request_repeated_cancel_and_finalization():
    app, evidence = application(worker_class=PacedWorker, continuous_output_capacity=64)
    with TCPServer(app) as server:
        connection, response = server.open_stream(max_tokens=256)
        initial = [read_event(response), read_event(response)]
        assert initial[0]["choices"][0]["delta"] == {"role": "assistant"}
        # Real Uvicorn shutdown while the TCP response is still active. No
        # direct scheduler cancellation substitutes for the launcher's hook.
        server.server.should_exit = True
        events = initial + read_all(response)
        response.close()
        connection.close()
        assert events[-1] == "[DONE]"
        assert any(isinstance(event, dict) and "error" in event for event in events)
        assert not any(isinstance(event, dict) and event.get("choices", [{}])[0].get("finish_reason")
            for event in events)
        status = server.stopped_status()
        assert status["cancelled_requests"] == 1 and status["shutting_down"] is True
        # Idempotent application stop-admission and owner finalization after
        # their actual server-driven completion must not double-count requests.
        asyncio.run(app.state.begin_shutdown())
        asyncio.run(app.state.begin_shutdown())
        print("TCP_CPU_SHUTDOWN_EVIDENCE=" + json.dumps(status, ensure_ascii=False))
    assert evidence["owner"].close().result(2) is None


def test_tcp_completed_stream_history_is_forgotten_after_consumption():
    app, _ = application(worker_class=PacedWorker, continuous_output_capacity=64)
    with TCPServer(app) as server:
        for _ in range(30):
            assert assert_normal(stream_request(server, max_tokens=1)) == "你"
            server.settle()
        status = server.settle()
        assert status["completed_requests"] == 30


def test_tcp_repeated_active_cancellation_and_finalization_are_idempotent():
    app, _ = application(worker_class=PacedWorker, continuous_output_capacity=64)
    with TCPServer(app) as server:
        connection, response = server.open_stream(max_tokens=128)
        initial = [read_event(response), read_event(response)]
        # The public shutdown hook submits cancellation intent, not direct KV
        # release; repeat it while the request is active and again after drain.
        for _ in range(2):
            asyncio.run_coroutine_threadsafe(app.state.begin_shutdown(), server.loop).result(5)
        events = initial + read_all(response)
        response.close()
        connection.close()
        assert events[-1] == "[DONE]"
        assert any(isinstance(event, dict) and "error" in event for event in events)
        before = server.settle()
        asyncio.run_coroutine_threadsafe(app.state.begin_shutdown(), server.loop).result(5)
        after = server.settle()
        assert before["cancelled_requests"] == after["cancelled_requests"] == 1
        assert before["completed_requests"] == after["completed_requests"] == 0


def test_tcp_legacy_sse_transport_regression():
    app = create_app(Settings(model_name="fake", execution_mode="legacy"))
    engine = app.state.engine

    def load():
        engine.model = FakeModel()
        engine.tokenizer = Tokenizer()

    def stream(messages, **kwargs):
        yield GenerationChunk("你", None, 2, 1)
        yield GenerationChunk("好", None, 2, 2)
        yield GenerationChunk("", "length", 2, 2)

    engine.load = load
    engine.stream_chat = stream
    with TCPServer(app) as server:
        assert assert_normal(stream_request(server, max_tokens=2)) == "你好"
        status = server.settle()
        assert status["completed_requests"] == 1
        assert app.state.continuous_owner is None


@pytest.mark.slow
def test_real_cuda_tcp_sse_concurrent_disconnect_and_shutdown():
    """Cached Qwen + actual CUDA forward + actual loopback HTTP sockets."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("this explicit network integration test requires CUDA")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    settings = Settings(model_name=model_name, execution_mode="continuous", device="cuda",
        dtype="float32", max_concurrent_generations=2, continuous_output_capacity=128,
        inference_timeout_seconds=60)
    backend = {}

    class InjectedFailureWorker(BatchedTransformersWorker):
        fail_next = threading.Event()
        def decode_batch(self, requests):
            if self.fail_next.is_set():
                self.fail_next.clear()
                raise RuntimeError("injected CUDA worker step failure (not a hardware fault)")
            return super().decode_batch(requests)

    def owner_factory(engine, config):
        def load():
            loaded = TransformersModelLoader(model_name, device="cuda", dtype=torch.float32,
                local_files_only=True).load()
            engine.model, engine.tokenizer = loaded.model, loaded.tokenizer
            engine.device, engine.dtype = "cuda", torch.float32
            if engine.tokenizer.pad_token_id is None:
                engine.tokenizer.pad_token_id = engine.tokenizer.eos_token_id
        engine.load = load
        def worker_factory(eng):
            worker = InjectedFailureWorker(LoadedModel(eng.model, eng.tokenizer, "cuda"), slot_capacity=2)
            backend["worker"] = worker
            return worker
        return ContinuousHTTPOwner(engine, config, worker_factory=worker_factory)

    app = create_app(settings, continuous_owner_factory=owner_factory)
    evidence = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transport": "Uvicorn real TCP HTTP, not ASGI shortcut"}
    with TCPServer(app) as server:
        prompts = ["用中文解释什么是KV缓存。", "Count from one to ten."]
        with ThreadPoolExecutor(max_workers=2) as clients:
            first = clients.submit(stream_request, server, prompt=prompts[0], max_tokens=8)
            second = clients.submit(stream_request, server, prompt=prompts[1], max_tokens=10)
            observed = [assert_normal(first.result(30)), assert_normal(second.result(30))]
        evidence["after_concurrent"] = server.settle()
        # Reference model uses the unchanged legacy algorithm only after the
        # shared continuous owner is idle; no concurrent model access occurs.
        reference = [app.state.engine.chat([{"role": "user", "content": prompt}],
            max_tokens=count, temperature=0).text for prompt, count in zip(prompts, (8, 10))]
        assert observed == reference
        # Actual legacy SSE generation over TCP on the same loaded model, only
        # while the continuous owner is idle; no model copies or concurrent use.
        legacy = create_app(Settings(model_name=model_name, execution_mode="legacy", device="cuda", dtype="float32"))
        def legacy_load():
            target, source = legacy.state.engine, app.state.engine
            target.model, target.tokenizer = source.model, source.tokenizer
            target.device, target.dtype = source.device, source.dtype
        legacy.state.engine.load = legacy_load
        with TCPServer(legacy) as legacy_server:
            legacy_text = assert_normal(stream_request(legacy_server, prompt=prompts[0], max_tokens=8))
            assert legacy_text == observed[0]
            evidence["legacy_tcp_completed"] = legacy_server.settle()["completed_requests"]
        # Actual slow TCP reader with a concurrent CUDA peer. These small
        # payloads do not prove OS send-buffer saturation or output overflow.
        with ThreadPoolExecutor(max_workers=1) as clients:
            connection, response = server.open_stream(prompt=prompts[0], max_tokens=16)
            peer = clients.submit(stream_request, server, prompt=prompts[1], max_tokens=8)
            assert_normal(peer.result(30))
            assert_normal(read_all(response, delay=.01))
            response.close(); connection.close()
        evidence["after_slow_reader"] = server.settle()
        with ThreadPoolExecutor(max_workers=1) as clients:
            survivor = clients.submit(stream_request, server, prompt=prompts[1], max_tokens=24)
            connection, response = server.open_stream(prompt=prompts[0], max_tokens=256)
            for _ in range(100):
                event = read_event(response)
                if isinstance(event, dict) and event.get("choices", [{}])[0].get("delta", {}).get("content"):
                    break
            else:
                pytest.fail("CUDA SSE never emitted partial text")
            if connection.sock is not None:
                connection.sock.shutdown(socket.SHUT_RDWR)
            response.close()
            connection.close()
            assert_normal(survivor.result(30))
        evidence["after_disconnect"] = server.settle()
        assert evidence["after_disconnect"]["cancelled_requests"] == 1
        connection, response = server.open_stream(prompt=prompts[0], max_tokens=256)
        initial = [read_event(response), read_event(response)]
        backend["worker"].fail_next.set()
        failed = initial + read_all(response)
        response.close(); connection.close()
        assert failed[-1] == "[DONE]" and failed[-2]["error"]["code"] == "generation_failed"
        assert not any(isinstance(e, dict) and e.get("choices", [{}])[0].get("finish_reason") for e in failed)
        assert_normal(stream_request(server, prompt=prompts[1], max_tokens=8))
        evidence["after_injected_failure_recovery"] = server.settle()
        assert evidence["after_injected_failure_recovery"]["failed_requests"] == 1
        connection, response = server.open_stream(prompt=prompts[0], max_tokens=256)
        initial = [read_event(response), read_event(response)]
        server.server.should_exit = True
        events = initial + read_all(response)
        response.close()
        connection.close()
        assert events[-1] == "[DONE]" and any(isinstance(event, dict) and "error" in event for event in events)
        evidence["after_shutdown"] = server.stopped_status()
        assert evidence["after_shutdown"]["cancelled_requests"] == 2
    evidence["owner_thread_alive"] = app.state.continuous_owner.thread.is_alive()
    evidence["uvicorn_thread_alive"] = server.thread.is_alive()
    evidence["pending_finalizers"] = len(app.state.cleanup_tasks)
    print("TCP_CUDA_SSE_EVIDENCE=" + json.dumps(evidence, ensure_ascii=False))
