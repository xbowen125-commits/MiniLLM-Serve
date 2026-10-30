"""Deterministic lifespan ordering checks; no model, sockets or owner thread."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from minillm.config import Settings
from minillm.server import create_app


class Owner:
    def __init__(self):
        self.stopped = Future()
        self.calls = []
        self.thread = SimpleNamespace(join=lambda: self.calls.append("join"))

    def start(self):
        ready = Future()
        ready.set_result(None)
        return ready

    def close(self):
        self.calls.append("close")
        if not self.stopped.done():
            self.stopped.set_result(None)
        return self.stopped


def application():
    owner = Owner()
    app = create_app(Settings(model_name="fake", device="cpu", dtype="float32",
        execution_mode="continuous"), continuous_owner_factory=lambda *_: owner)
    return app, owner


def run(coroutine):
    async def bounded():
        async with asyncio.timeout(5):
            await coroutine
    asyncio.run(bounded())


@pytest.mark.parametrize("order", ["owner_first", "drain_first", "together"])
def test_shutdown_completion_order(monkeypatch, order):
    async def scenario():
        app, owner = application()
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        started, release, settled, selected = (asyncio.Event() for _ in range(4))
        tasks = {}

        async def close():
            started.set()
            try:
                await release.wait()
                owner.calls.append("drained")
            finally:
                settled.set()

        async def ordered_wait(futures, **kwargs):
            drain, stopped = futures
            tasks.update(drain=drain, stopped=stopped)
            assert kwargs == {"return_when": asyncio.FIRST_COMPLETED}
            await started.wait()
            if order != "owner_first":
                release.set()
                await drain
            if order != "drain_first":
                owner.stopped.set_result(None)
                await stopped
            selected.set()
            done = {drain} if order == "drain_first" else {stopped}
            if order == "together":
                done.add(drain)
            return done, set(futures) - done

        monkeypatch.setattr(app.state.request_scheduler, "close", close)
        monkeypatch.setattr(asyncio, "wait", ordered_wait)
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        try:
            await selected.wait()
            release.set()
            await shutdown
            assert tasks["drain"].done() and not tasks["drain"].cancelled()
            assert settled.is_set()
            assert owner.calls == ["drained", "close", "join"]
        finally:
            release.set()
            await asyncio.gather(shutdown, *tasks.values(), return_exceptions=True)
    run(scenario())


@pytest.mark.parametrize("both", [False, True])
def test_real_drain_error_is_propagated(monkeypatch, caplog, both):
    async def scenario():
        app, owner = application()
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        error = RuntimeError("scheduler drain failed")

        async def close():
            raise error

        async def ordered_wait(futures, **kwargs):
            drain, stopped = futures
            await asyncio.gather(drain, return_exceptions=True)
            if both:
                owner.stopped.set_result(None)
                await stopped
            return ({drain, stopped} if both else {drain}), set()

        monkeypatch.setattr(app.state.request_scheduler, "close", close)
        monkeypatch.setattr(asyncio, "wait", ordered_wait)
        try:
            with pytest.raises(RuntimeError) as caught:
                await lifespan.__aexit__(None, None, None)
            assert caught.value is error
            assert not owner.calls
            assert not owner.stopped.cancelled()
            assert not any(record.getMessage() == "secondary scheduler drain failure during shutdown"
                for record in caplog.records)
        finally:
            if not owner.stopped.done():
                owner.stopped.set_result(None)
    run(scenario())


def test_owner_cleanup_error_retains_real_lease(monkeypatch, caplog):
    async def scenario():
        app, owner = application()
        scheduler = app.state.request_scheduler
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        lease = await scheduler.acquire()
        started, settled = asyncio.Event(), asyncio.Event()
        original_close = scheduler.close
        tasks = {}
        error = RuntimeError("owner cleanup unconfirmed")

        async def close():
            started.set()
            try:
                await original_close()
            finally:
                settled.set()

        async def ordered_wait(futures, **kwargs):
            drain, stopped = futures
            tasks["drain"] = drain
            await started.wait()
            owner.stopped.set_exception(error)
            await asyncio.gather(stopped, return_exceptions=True)
            return {stopped}, {drain}

        monkeypatch.setattr(scheduler, "close", close)
        monkeypatch.setattr(asyncio, "wait", ordered_wait)
        try:
            with pytest.raises(RuntimeError) as caught:
                await lifespan.__aexit__(None, None, None)
            assert caught.value is error
            assert (await scheduler.snapshot())["running_requests"] == 1
            assert not lease._released and not scheduler._idle.is_set()
            assert settled.is_set() and tasks["drain"].cancelled()
            assert not owner.calls
            assert not any(record.getMessage() == "secondary scheduler drain failure during shutdown"
                for record in caplog.records)
        finally:
            await lease.release("cancelled")  # test-owned fake request only
    run(scenario())


def test_normal_owner_stop_still_waits_for_lease(monkeypatch):
    async def scenario():
        app, owner = application()
        scheduler = app.state.request_scheduler
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        lease = await scheduler.acquire()
        selected = asyncio.Event()
        tasks = {}

        async def ordered_wait(futures, **kwargs):
            drain, stopped = futures
            tasks["drain"] = drain
            owner.stopped.set_result(None)
            await stopped
            selected.set()
            return {stopped}, {drain}

        monkeypatch.setattr(asyncio, "wait", ordered_wait)
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        try:
            await selected.wait()
            assert not shutdown.done()
            assert (await scheduler.snapshot())["running_requests"] == 1
            assert not lease._released
            await lease.release("cancelled")
            await shutdown
            assert not tasks["drain"].cancelled()
            assert (await scheduler.snapshot())["running_requests"] == 0
            assert owner.calls == ["close", "join"]
        finally:
            await lease.release("cancelled")
            await asyncio.gather(shutdown, return_exceptions=True)
    run(scenario())


@pytest.mark.parametrize("timeout", [False, True])
def test_external_cancellation_settles_auxiliary_drain(monkeypatch, timeout):
    async def scenario():
        app, owner = application()
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        started, settled = asyncio.Event(), asyncio.Event()
        tasks = {}

        async def close():
            tasks["drain"] = asyncio.current_task()
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                settled.set()

        monkeypatch.setattr(app.state.request_scheduler, "close", close)
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        try:
            await started.wait()
            if timeout:
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(None) as deadline:
                        deadline.reschedule(asyncio.get_running_loop().time() - 1)
                        await shutdown
            else:
                shutdown.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await shutdown
            assert settled.is_set()
            assert tasks["drain"].done() and tasks["drain"].cancelled()
            assert not owner.stopped.cancelled()
            assert not owner.calls  # external cancellation cannot claim clean close
        finally:
            if not tasks["drain"].done():
                tasks["drain"].cancel()
            owner.stopped.set_result(None)
            await asyncio.gather(shutdown, tasks["drain"], return_exceptions=True)
    run(scenario())


@pytest.mark.parametrize("trigger", ["cancel", "timeout"])
@pytest.mark.parametrize("owner_failure", [False, True])
def test_late_shutdown_cancellation_preserves_owner_future(monkeypatch, trigger, owner_failure):
    async def scenario():
        close_requested, drained, completion_observed = (asyncio.Event() for _ in range(3))

        class DelayedOwner(Owner):
            def close(self):
                self.calls.append("close")
                close_requested.set()
                return self.stopped

        owner = DelayedOwner()
        app = create_app(Settings(model_name="fake", device="cpu", dtype="float32",
            execution_mode="continuous"), continuous_owner_factory=lambda *_: owner)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        unhandled = []
        loop.set_exception_handler(lambda _, context: unhandled.append(context))
        wrap_future = asyncio.wrap_future
        wrappers = []

        def observe_wrapper(future, **kwargs):
            wrapper = wrap_future(future, **kwargs)
            if future is owner.stopped:
                wrappers.append(wrapper)
                # Schedule the barrier after callbacks already installed by
                # lifespan have had an opportunity to observe completion.
                wrapper.add_done_callback(lambda _: loop.call_soon(completion_observed.set))
            return wrapper

        async def close_scheduler():
            owner.calls.append("drained")
            drained.set()

        monkeypatch.setattr(asyncio, "wrap_future", observe_wrapper)
        monkeypatch.setattr(app.state.request_scheduler, "close", close_scheduler)
        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        try:
            await close_requested.wait()
            assert drained.is_set() and not owner.stopped.done()
            assert owner.calls == ["drained", "close"]
            if trigger == "timeout":
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(None) as deadline:
                        deadline.reschedule(loop.time() - 1)
                        await shutdown
            else:
                shutdown.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await shutdown
            assert shutdown.cancelled()
            assert not owner.stopped.cancelled()
            assert not owner.stopped.done()
            if owner_failure:
                error = RuntimeError("late owner cleanup failure")
                owner.stopped.set_exception(error)
                assert owner.stopped.exception() is error
            else:
                owner.stopped.set_result(None)
                assert owner.stopped.result() is None
            await completion_observed.wait()
            assert all(w.done() and not w.cancelled() for w in wrappers)
            assert not unhandled
            assert not (asyncio.all_tasks() - {asyncio.current_task()})
            assert owner.calls == ["drained", "close"]  # no false join or second close
        finally:
            if not owner.stopped.done():
                owner.stopped.set_result(None)
                await completion_observed.wait()
            await asyncio.gather(shutdown, return_exceptions=True)
            loop.set_exception_handler(previous_handler)
    run(scenario())


def test_simultaneous_failures_preserve_secondary_drain_diagnostic(monkeypatch, caplog):
    async def scenario():
        app, owner = application()
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        owner_error = RuntimeError("owner cleanup failed")
        drain_error = RuntimeError("scheduler drain independently failed")

        async def close():
            raise drain_error

        async def ordered_wait(futures, **kwargs):
            drain, stopped = futures
            await asyncio.gather(drain, return_exceptions=True)
            owner.stopped.set_exception(owner_error)
            await asyncio.gather(stopped, return_exceptions=True)
            return {drain, stopped}, set()

        monkeypatch.setattr(app.state.request_scheduler, "close", close)
        monkeypatch.setattr(asyncio, "wait", ordered_wait)
        with pytest.raises(RuntimeError) as caught:
            await lifespan.__aexit__(None, None, None)
        assert caught.value is owner_error
        records = [record for record in caplog.records
            if record.getMessage() == "secondary scheduler drain failure during shutdown"]
        assert len(records) == 1
        assert records[0].exc_info[1] is drain_error
        assert not owner.calls
    run(scenario())
