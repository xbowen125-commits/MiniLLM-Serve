"""FastAPI application with FIFO admission control for local model inference."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Settings
from .engine import GenerationChunk, GenerationConfig, GenerationEngine, GenerationResult
from .scheduler import (
    RequestOutcome,
    RequestScheduler,
    SchedulerClosed,
    SchedulerLease,
    SchedulerQueueFull,
    SchedulerQueueTimeout,
)
from .schemas import (
    APIError,
    APIErrorResponse,
    ChatCompletionRequest,
    make_chunk,
    make_response,
    make_response_id,
)

logger = logging.getLogger("minillm.server")


def _generation_config(req: ChatCompletionRequest) -> GenerationConfig:
    """Translate the public API request into the engine's internal options."""
    return GenerationConfig(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        stop=req.stop,
        stream=req.stream,
        seed=req.seed,
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _gpu_status(eng: GenerationEngine) -> dict:
    device = str(getattr(eng, "device", "unknown"))
    if not device.startswith("cuda"):
        return {"available": False, "device": device}
    try:
        import torch

        index = torch.cuda.current_device()
        return {
            "available": True,
            "device": device,
            "name": torch.cuda.get_device_name(index),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(index) / 1024**2, 3),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(index) / 1024**2, 3),
            "max_memory_allocated_mb": round(
                torch.cuda.max_memory_allocated(index) / 1024**2,
                3,
            ),
        }
    except Exception as exc:
        return {"available": False, "device": device, "error": str(exc)}


def _lease_headers(lease: SchedulerLease) -> dict[str, str]:
    return {
        "X-MiniLLM-Request-ID": str(lease.request_id),
        "X-MiniLLM-Queue-Wait-Ms": f"{lease.queue_wait_seconds * 1000:.3f}",
    }


def _error_response(status: int, message: str, error_type: str, code: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=APIErrorResponse(error=APIError(message=message, type=error_type, code=code)).model_dump(),
    )


def _sse_data(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {data}\n\n".encode("utf-8")


def _chunk_payload(chunk) -> dict:
    payload = chunk.model_dump()
    delta = payload["choices"][0]["delta"]
    payload["choices"][0]["delta"] = {key: value for key, value in delta.items() if value is not None}
    return payload


async def _finalize_nonstream_worker(
    worker: asyncio.Task[GenerationResult],
    lease: SchedulerLease,
    outcome: RequestOutcome,
) -> None:
    """Settle a worker and release its lease outside the request task."""
    try:
        await asyncio.shield(worker)
    except BaseException:
        pass
    await lease.release(outcome)


class _StreamSession:
    """Own a streaming worker independently from the cancellable HTTP body task."""

    def __init__(self, req: ChatCompletionRequest, eng: GenerationEngine, lease: SchedulerLease):
        self.req = req
        self.eng = eng
        self.lease = lease
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=8)
        self.cleanup_done = asyncio.Event()
        self.cancel_reason: str | None = None
        self.worker_error: Exception | None = None
        self.released_outcome: RequestOutcome | None = None
        self._reason_lock = threading.Lock()
        self.producer = threading.Thread(
            target=self._produce,
            name=f"minillm-stream-{lease.request_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.producer.start()

    def cancel(self, reason: str) -> None:
        with self._reason_lock:
            if self.cancel_reason is None:
                self.cancel_reason = reason
        self.lease.cancel_event.set()

    def _put_message(self, item: tuple[str, object]) -> bool:
        while not self.lease.cancel_event.is_set():
            try:
                self.messages.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            chunks = self.eng.stream_chat(
                self.req.messages,
                cancel_event=self.lease.cancel_event,
                config=self.lease.request_state.generation_config,
            )
            for chunk in chunks:
                if not self._put_message(("chunk", chunk)):
                    return
        except Exception as exc:
            self.worker_error = exc
            logger.exception("stream generation failed")
            self._put_message(("error", exc))
        finally:
            self._put_message(("done", None))

    async def finalize(self, inference_timeout_seconds: float) -> None:
        """Release the lease after the physical producer thread has exited.

        This task is created by the endpoint before StreamingResponse starts, so
        Starlette cancelling its body task on disconnect cannot cancel cleanup.
        """
        deadline = time.monotonic() + inference_timeout_seconds
        try:
            while self.producer.is_alive():
                if time.monotonic() >= deadline and self.cancel_reason is None:
                    self.cancel("inference_timeout")
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            # Event-loop shutdown is the last-resort cancellation path.  Clear
            # this task's cancellation and still retain the lease until the
            # cooperative worker has really stopped.
            self.cancel("cancelled")
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            while self.producer.is_alive():
                await asyncio.sleep(0.05)

        self.producer.join()
        if self.cancel_reason == "inference_timeout":
            outcome: RequestOutcome = "inference_timeout"
        elif self.cancel_reason is not None or self.lease.cancel_event.is_set():
            outcome = "cancelled"
        elif self.worker_error is not None:
            outcome = "failed"
        else:
            outcome = "completed"
        try:
            await self.lease.release(outcome)
            self.released_outcome = outcome
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            await self.lease.release(outcome)
            self.released_outcome = outcome
        finally:
            self.cleanup_done.set()


async def _stream_chat_events(session: _StreamSession):
    """Deliver SSE; worker/lease cleanup is owned by the independent finalizer."""
    req = session.req
    response_id = make_response_id()
    created = int(time.time())
    stream_error: APIErrorResponse | None = None
    saw_done = False

    try:
        yield _sse_data(_chunk_payload(make_chunk(response_id, created, req.model, role="assistant")))
        while True:
            try:
                kind, value = await asyncio.to_thread(session.messages.get, True, 0.1)
            except queue.Empty:
                if session.cleanup_done.is_set():
                    break
                continue

            if kind == "chunk":
                chunk = value
                assert isinstance(chunk, GenerationChunk)
                if chunk.text:
                    yield _sse_data(
                        _chunk_payload(make_chunk(response_id, created, req.model, content=chunk.text))
                    )
                if chunk.finish_reason is not None:
                    yield _sse_data(
                        _chunk_payload(
                            make_chunk(
                                response_id,
                                created,
                                req.model,
                                finish_reason=chunk.finish_reason,
                            )
                        )
                    )
            elif kind == "error":
                stream_error = APIErrorResponse(
                    error=APIError(message=f"生成失败: {value}", type="internal_error", code=None)
                )
            elif kind == "done":
                saw_done = True
                break
    except (asyncio.CancelledError, GeneratorExit):
        logger.info("stream client disconnected request_id=%d", session.lease.request_id)
        session.cancel("client_disconnect")
        raise
    finally:
        if not saw_done and not session.cleanup_done.is_set():
            session.cancel("client_disconnect")

    while not session.cleanup_done.is_set():
        await asyncio.sleep(0.01)
    if session.released_outcome == "inference_timeout":
        stream_error = APIErrorResponse(
            error=APIError(
                message="推理执行超时",
                type="timeout_error",
                code="inference_timeout",
            )
        )
    if stream_error is not None:
        yield _sse_data(stream_error.model_dump())
    yield _sse_data("[DONE]")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    engine = GenerationEngine(settings)
    scheduler = RequestScheduler(
        max_running=settings.max_concurrent_generations,
        queue_capacity=settings.request_queue_capacity,
        queue_timeout_seconds=settings.queue_timeout_seconds,
        inference_timeout_seconds=settings.inference_timeout_seconds,
    )
    cleanup_tasks: set[asyncio.Task[None]] = set()

    def track_cleanup(task: asyncio.Task[None]) -> asyncio.Task[None]:
        cleanup_tasks.add(task)

        def completed(done: asyncio.Task[None]) -> None:
            cleanup_tasks.discard(done)
            try:
                error = done.exception()
            except asyncio.CancelledError:
                logger.error("background cleanup task was cancelled")
            else:
                if error is not None:
                    logger.error(
                        "background cleanup task failed",
                        exc_info=(type(error), error, error.__traceback__),
                    )

        task.add_done_callback(completed)
        return task

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("loading model %s ...", settings.model_name)
        engine.load()
        logger.info("model ready (device=%s)", engine.device)
        try:
            yield
        finally:
            logger.info("stopping request scheduler")
            await scheduler.close()
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    app = FastAPI(
        title="MiniLLM-Serve",
        version="0.1.0",
        description="OpenAI-compatible local inference (Mission 07: FIFO request scheduling)",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.request_scheduler = scheduler
    app.state.cleanup_tasks = cleanup_tasks

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(422, str(exc), "invalid_request_error", "validation_error")

    @app.get("/health")
    def health(request: Request) -> dict:
        current_settings: Settings = request.app.state.settings
        current_engine: GenerationEngine = request.app.state.engine
        return {
            "status": "ok",
            "model": current_settings.model_name,
            "loaded": current_engine.is_loaded,
            "device": str(getattr(current_engine, "device", "unknown")),
            "dtype": str(getattr(current_engine, "dtype", "unknown")),
            "use_kv_cache": current_settings.use_kv_cache,
            "versions": {
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "accelerate": _package_version("accelerate"),
            },
        }

    @app.get("/scheduler/status")
    async def scheduler_status(request: Request) -> dict:
        current: RequestScheduler = request.app.state.request_scheduler
        snapshot = await current.snapshot()
        snapshot["gpu"] = _gpu_status(request.app.state.engine)
        return snapshot

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        request: Request,
        response: Response,
    ) -> Response:
        eng: GenerationEngine = request.app.state.engine
        current_scheduler: RequestScheduler = request.app.state.request_scheduler

        if not eng.is_loaded:
            return _error_response(503, "模型尚未加载完成，请稍后重试", "not_ready")
        if req.model != eng.model_name:
            return _error_response(
                404,
                f"未加载模型 {req.model}（当前服务模型: {eng.model_name}）",
                "model_not_found",
            )

        try:
            lease = await current_scheduler.acquire(_generation_config(req))
        except SchedulerQueueFull as exc:
            return _error_response(429, str(exc), "rate_limit_error", "queue_full")
        except SchedulerQueueTimeout as exc:
            return _error_response(503, str(exc), "timeout_error", "queue_timeout")
        except SchedulerClosed as exc:
            return _error_response(503, str(exc), "service_unavailable", "scheduler_closed")

        if req.stream:
            session = _StreamSession(req, eng, lease)
            session.start()
            track_cleanup(
                asyncio.create_task(
                    session.finalize(settings.inference_timeout_seconds),
                    name=f"minillm-stream-finalizer-{lease.request_id}",
                )
            )
            headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            headers.update(_lease_headers(lease))
            return StreamingResponse(
                _stream_chat_events(session),
                media_type="text/event-stream",
                headers=headers,
            )

        started = time.perf_counter()
        worker = asyncio.create_task(
            asyncio.to_thread(
                eng.chat,
                req.messages,
                cancel_event=lease.cancel_event,
                config=lease.request_state.generation_config,
            )
        )
        try:
            result: GenerationResult = await asyncio.wait_for(
                asyncio.shield(worker),
                timeout=settings.inference_timeout_seconds,
            )
        except TimeoutError:
            lease.cancel_event.set()
            cleanup = track_cleanup(
                asyncio.create_task(
                    _finalize_nonstream_worker(worker, lease, "inference_timeout"),
                    name=f"minillm-worker-finalizer-{lease.request_id}",
                )
            )
            await asyncio.shield(cleanup)
            return _error_response(504, "推理执行超时", "timeout_error", "inference_timeout")
        except asyncio.CancelledError:
            lease.cancel_event.set()
            track_cleanup(
                asyncio.create_task(
                    _finalize_nonstream_worker(worker, lease, "cancelled"),
                    name=f"minillm-worker-finalizer-{lease.request_id}",
                )
            )
            raise
        except Exception as exc:
            cleanup = track_cleanup(
                asyncio.create_task(
                    _finalize_nonstream_worker(worker, lease, "failed"),
                    name=f"minillm-worker-finalizer-{lease.request_id}",
                )
            )
            await asyncio.shield(cleanup)
            logger.exception("generation failed")
            return _error_response(500, f"生成失败: {exc}", "internal_error")

        cleanup = track_cleanup(
            asyncio.create_task(
                _finalize_nonstream_worker(worker, lease, "completed"),
                name=f"minillm-worker-finalizer-{lease.request_id}",
            )
        )
        await asyncio.shield(cleanup)
        response.headers.update(_lease_headers(lease))
        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "chat_completions ok request_id=%d model=%s prompt_tokens=%d completion_tokens=%d "
            "queue_wait_ms=%.1f latency_ms=%.1f finish=%s",
            lease.request_id,
            req.model,
            result.prompt_tokens,
            result.completion_tokens,
            lease.queue_wait_seconds * 1000,
            latency_ms,
            result.finish_reason,
        )
        return make_response(
            req.model,
            result.text,
            result.finish_reason,
            result.prompt_tokens,
            result.completion_tokens,
        )

    return app


settings = Settings()
app = create_app(settings)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
