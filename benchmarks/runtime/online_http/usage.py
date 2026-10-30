"""Opt-in benchmark SSE metadata, never standard chat content.

Read existing RequestState counters only. Disconnected clients cannot receive a
final event; terminal observer journals remain the out-of-band evidence there.
"""

from contextvars import ContextVar
import json
import threading
import weakref

_enabled = ContextVar("benchmark_usage", default=False)


def usage_payload(state, *, response_id, lease_id, prompt_tokens=None):
    known = state is not None and state.is_terminal
    count = len(state.generated_tokens) if known else None
    return {"object": "minillm.benchmark.usage", "id": response_id,
        "request_id": str(lease_id), "engine_request_id": state.request_id if state is not None else None,
        "status": state.status.value if known else "unknown", "count_source": "terminal_request_state",
        "includes_eos": True, "includes_stop_and_hidden_tokens": True,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": count,
                  "total_tokens": prompt_tokens + count if prompt_tokens is not None and count is not None else None}}


class UsageOptIn:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        enabled = (b"x-minillm-benchmark-usage", b"1") in scope.get("headers", [])
        token = _enabled.set(enabled)
        try:
            await self.app(scope, receive, send)
        finally:
            _enabled.reset(token)


def install(app):
    """Benchmark host only; the ordinary application and schema stay unchanged."""
    import minillm.server as server
    from minillm.engine.generation import GenerationRunner
    lock = threading.Lock()
    watched = weakref.WeakValueDictionary()
    init = server._StreamSession.__init__
    stream = GenerationRunner.stream

    def session_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if _enabled.get():
            with lock:
                watched[id(self.lease.cancel_event)] = self

    def generate(self, **kwargs):
        key = id(kwargs.get("cancel_event"))
        with lock:
            session = watched.pop(key, None)
            if session is not None:
                session._benchmark_request = kwargs["request"]
        yield from stream(self, **kwargs)

    def wrap(original, continuous):
        async def events(session):
            response_id = None
            prompt_tokens = None
            enabled = _enabled.get()
            try:
                async for frame in original(session):
                    if enabled and frame.startswith(b"data: {"):
                        data = json.loads(frame[6:])
                        response_id = data.get("id", response_id)
                    if enabled and frame == b"data: [DONE]\n\n":
                        if continuous:
                            state = session.lease.request_state
                            try:
                                completion = session.handle.completion.result()
                            except Exception:
                                state = None  # Cleanup/execution is unconfirmed.
                            else:
                                if completion.result is not None:
                                    prompt_tokens = completion.result.prompt_tokens
                        else:
                            state = getattr(session, "_benchmark_request", None)
                        if state is not None and prompt_tokens is None and state.prompt:
                            prompt_tokens = len(session.eng.tokenizer.encode(state.prompt)) if not continuous else None
                        payload = usage_payload(state, response_id=response_id,
                            lease_id=session.lease.request_id, prompt_tokens=prompt_tokens)
                        yield b"event: minillm.usage\n" + server._sse_data(payload)
                    yield frame
            finally:
                if not continuous:
                    with lock:
                        watched.pop(id(session.lease.cancel_event), None)
        return events

    server._StreamSession.__init__ = session_init
    GenerationRunner.stream = generate
    server._stream_chat_events = wrap(server._stream_chat_events, False)
    server._continuous_chat_events = wrap(server._continuous_chat_events, True)
    app.add_middleware(UsageOptIn)
