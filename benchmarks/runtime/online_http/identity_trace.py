"""Explicitly installed correctness-only boundary observers, never a scheduler.

Metadata contains identities, not prompts or generated text. ContextVars cross
HTTP child tasks; actual Delivery objects bridge the owner thread. No identity
is recovered from timestamps. Failed observations invalidate the proof, not work.
"""
from contextvars import ContextVar
import json
import uuid

_http = ContextVar("identity_trace_http", default=None)


class HTTPIdentityBoundary:
    def __init__(self, app, observer):
        self.app, self.observer = app, observer

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") != "/v1/chat/completions":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        link = {"http_exchange_id": uuid.uuid4().hex,
                "planned_request_id": headers.get(b"x-minillm-planned-id", b"").decode("ascii", errors="replace") or None,
                "client_record_id": headers.get(b"x-minillm-client-id", b"").decode("ascii", errors="replace") or None}
        token = _http.set(link)
        self.observer.event("http_begin", **link)

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                self.observer.event("disconnect", **link)
            return message

        async def observed_send(message):
            # Observe only after the transport accepts this message. No content
            # is retained; parse one original frame to bind its real response ID.
            await send(message)
            if message["type"] == "http.response.start":
                h = dict(message.get("headers", []))
                self.observer.event("http_headers", **link, status=message["status"],
                    lease_id=h.get(b"x-minillm-request-id", b"").decode() or None)
            elif message["type"] == "http.response.body":
                def response_identity():
                    body = message.get("body", b"")
                    if body.startswith(b"data: {"):
                        payload = json.loads(body.split(b"\n", 1)[0][6:])
                        if payload.get("object") == "chat.completion.chunk":
                            self.observer.event("response_identity", **link, response_id=payload.get("id"))
                self.observer.observe(response_identity)
        try:
            return await self.app(scope, observed_receive, observed_send)
        finally:
            self.observer.event("http_end", **link)
            _http.reset(token)


class IdentityObserver:
    VERSION = 1

    def __init__(self, trace):
        self.trace = trace
        self.requests = {}  # bounded diagnostic identity links; no lifecycle ownership

    def observe(self, callback):
        try:
            return callback()
        except Exception as exc:
            self.trace.errors.append("identity observer: " + type(exc).__name__ + ":" + str(exc))
            self.event("observation_error", error_type=type(exc).__name__)

    def event(self, boundary, **fields):
        self.trace.emit("identity_" + boundary, boundary_version=self.VERSION, **fields)

    def fields(self, state):
        return {**self.requests.get(id(state), {}), "request_id": state.request_id,
                "request_object_identity": id(state)}

    def install(self):
        import minillm.server as server
        from minillm.scheduler import RequestScheduler, SchedulerLease
        from minillm.request_queue import RequestQueue
        from minillm.runtime.http_owner import ContinuousHTTPOwner
        from minillm.runtime.facade import RuntimeFacade
        observer, patch = self, self.trace.patch

        def create(original):
            def call(*args, **kwargs):
                app = original(*args, **kwargs)
                app.add_middleware(HTTPIdentityBoundary, observer=observer)
                return app
            return call
        patch(server, "create_app", create)

        def new_request(original):
            def call(scheduler, *args, **kwargs):
                sequence, state = original(scheduler, *args, **kwargs)
                def record():
                    observer.requests[id(state)] = dict(_http.get() or {})
                    observer.event("request_created", **observer.fields(state), sequence=sequence,
                        scheduler_identity=id(scheduler), queue_identity=id(scheduler._waiters))
                observer.observe(record)
                return sequence, state
            return call
        patch(RequestScheduler, "_new_request_locked", new_request)

        def grant(original):
            def call(scheduler, *args, **kwargs):
                lease = original(scheduler, *args, **kwargs)
                def record():
                    observer.event("admission_accept", **observer.fields(lease.request_state),
                        lease_id=str(lease.request_id), scheduler_identity=id(scheduler))
                    def cancel_set(previous):
                        def set_event():
                            result = previous()
                            observer.event("cancel_intent", **observer.fields(lease.request_state),
                                lease_id=str(lease.request_id), source="actual cancellation Event.set")
                            return result
                        return set_event
                    patch(lease.cancel_event, "set", cancel_set)
                observer.observe(record)
                return lease
            return call
        patch(RequestScheduler, "_grant_locked", grant)

        def acquire(original):
            async def call(scheduler, *args, **kwargs):
                aid = uuid.uuid4().hex
                observer.event("admission_begin", **(_http.get() or {}), admission_id=aid)
                try:
                    return await original(scheduler, *args, **kwargs)
                except BaseException as exc:
                    observer.event("admission_reject", **(_http.get() or {}), admission_id=aid,
                        error_type=type(exc).__name__)
                    raise
            return call
        patch(RequestScheduler, "acquire", acquire)

        def queue_call(name):
            def wrap(original):
                def call(queue, *args, **kwargs):
                    before = observer.observe(lambda: list(queue._items))
                    result = original(queue, *args, **kwargs)
                    def record():
                        after = list(queue._items)
                        old, new = {id(r): r for r in before}, {id(r): r for r in after}
                        for oid in [key for key in old if key not in new]:
                            observer.event("queue_leave", **observer.fields(old[oid]), queue_identity=id(queue), operation=name)
                        for oid in [key for key in new if key not in old]:
                            observer.event("queue_enter", **observer.fields(new[oid]), queue_identity=id(queue), operation=name)
                    observer.observe(record)
                    return result
                return call
            return wrap
        for name in ("put", "remove", "get_batch"):
            patch(RequestQueue, name, queue_call(name))

        def owner_init(original):
            def call(owner, *args, **kwargs):
                original(owner, *args, **kwargs)
                def put(previous):
                    def call(delivery):
                        command_id = uuid.uuid4().hex
                        fields = {**observer.fields(delivery.state), "command_id": command_id,
                            "delivery_identity": id(delivery), "owner_identity": id(owner),
                            "future_identity": id(delivery.future)}
                        # Queue's existing synchronization remains authoritative.
                        # Save metadata on observer only, never on request state.
                        commands[id(delivery)] = fields
                        observer.event("command_enqueue", **fields)
                        try:
                            result = previous(delivery)
                        except BaseException as exc:
                            commands.pop(id(delivery), None)
                            observer.event("command_reject", **fields, error_type=type(exc).__name__)
                            raise
                        def completed(future):
                            def record():
                                error, outcome = None, None
                                try:
                                    outcome = future.result().outcome
                                except BaseException as exc:
                                    error = type(exc).__name__
                                observer.event("command_finish", **fields, terminal_result_id=command_id,
                                    outcome=outcome, error_type=error, status=delivery.state.status.value,
                                    has_cache_ref=delivery.state.kv_cache_ref is not None,
                                    owner_status=owner.status())
                            observer.observe(record)
                        delivery.future.add_done_callback(completed)
                        return result
                    return call
                def get(previous):
                    def call():
                        delivery = previous()
                        fields = commands.pop(id(delivery), None)
                        observer.event("command_start", **{**(fields or {}), "request_id": delivery.state.request_id})
                        return delivery
                    return call
                commands = {}
                # Installation itself is observed; failure cannot change owner construction.
                observer.observe(lambda: patch(owner._queue, "put_nowait", put))
                observer.observe(lambda: patch(owner._queue, "get_nowait", get))
            return call
        patch(ContinuousHTTPOwner, "__init__", owner_init)

        def submit(original):
            def call(runtime, *args, **kwargs):
                context = original(runtime, *args, **kwargs)
                observer.observe(lambda: observer.event("runtime_handoff", **observer.fields(context.request),
                    runtime_context_id=context.request_id, context_object_identity=id(context),
                    runtime_identity=id(runtime)))
                return context
            return call
        patch(RuntimeFacade, "submit", submit)

        def release(original):
            async def call(lease, outcome):
                result = await original(lease, outcome)
                observer.event("finalizer_complete", **observer.fields(lease.request_state),
                    lease_id=str(lease.request_id), released=lease._released, outcome=outcome,
                    has_cache_ref=lease.request_state.kv_cache_ref is not None)
                return result
            return call
        patch(SchedulerLease, "release", release)

        def close(original):
            def call(owner):
                observer.event("owner_shutdown", owner_identity=id(owner), status=owner.status())
                future = original(owner)
                def stopped(done):
                    observer.event("owner_stopped", owner_identity=id(owner),
                        confirmed=done.exception() is None, status=owner.status())
                future.add_done_callback(stopped)
                return future
            return call
        patch(ContinuousHTTPOwner, "close", close)
        return self
