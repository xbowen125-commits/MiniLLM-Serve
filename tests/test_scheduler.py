"""Mission 07 scheduler tests without a model or a live HTTP server."""

import asyncio

import pytest

from minillm.scheduler import (
    RequestScheduler,
    SchedulerClosed,
    SchedulerQueueFull,
    SchedulerQueueTimeout,
)


class FakeFuture:
    """Minimal Future used because this sandbox cannot start an asyncio loop."""

    def __init__(self):
        self._done = False
        self._cancelled = False
        self._result = None
        self._exception = None

    def __await__(self):
        if not self._done:
            yield self
        return self.result()

    def done(self):
        return self._done

    def cancelled(self):
        return self._cancelled

    def cancel(self):
        self._done = True
        self._cancelled = True

    def set_result(self, value):
        self._done = True
        self._result = value

    def set_exception(self, exception):
        self._done = True
        self._exception = exception

    def result(self):
        if self._cancelled:
            raise asyncio.CancelledError
        if self._exception is not None:
            raise self._exception
        return self._result


class FakeLoop:
    def create_future(self):
        return FakeFuture()


class FakeEvent:
    def __init__(self):
        self._set = True
        self._waiters = []

    def set(self):
        self._set = True
        for waiter in self._waiters:
            waiter.set_result(True)
        self._waiters.clear()

    def clear(self):
        self._set = False

    async def wait(self):
        if self._set:
            return True
        waiter = FakeFuture()
        self._waiters.append(waiter)
        return await waiter


@pytest.fixture
def scheduler_asyncio(monkeypatch):
    async def wait_for(awaitable, timeout):
        return await awaitable

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr(asyncio, "shield", lambda awaitable: awaitable)
    monkeypatch.setattr(asyncio, "wait_for", wait_for)


def run_immediate(awaitable):
    iterator = awaitable.__await__()
    try:
        yielded = next(iterator)
    except StopIteration as exc:
        return exc.value
    raise AssertionError(f"coroutine unexpectedly blocked: {yielded!r}")


def start_blocked(awaitable):
    iterator = awaitable.__await__()
    yielded = next(iterator)
    assert isinstance(yielded, FakeFuture)
    return iterator


def finish(iterator):
    try:
        next(iterator)
    except StopIteration as exc:
        return exc.value
    raise AssertionError("coroutine remained blocked")


def make_scheduler(**overrides):
    values = {
        "max_running": 1,
        "queue_capacity": 2,
        "queue_timeout_seconds": 10,
        "inference_timeout_seconds": 20,
    }
    values.update(overrides)
    return RequestScheduler(**values)


def test_fifo_order_and_concurrency_limit(scheduler_asyncio):
    scheduler = make_scheduler()
    first = run_immediate(scheduler.acquire())
    second_wait = start_blocked(scheduler.acquire())
    third_wait = start_blocked(scheduler.acquire())

    snapshot = run_immediate(scheduler.snapshot())
    assert snapshot["running_requests"] == 1
    assert snapshot["queued_requests"] == 2

    run_immediate(first.release("completed"))
    second = finish(second_wait)
    assert run_immediate(scheduler.snapshot())["queued_requests"] == 1

    run_immediate(second.release("completed"))
    third = finish(third_wait)
    assert second.request_id < third.request_id
    run_immediate(third.release("completed"))
    snapshot = run_immediate(scheduler.snapshot())
    assert snapshot["running_requests"] == 0
    assert snapshot["completed_requests"] == 3


def test_queue_full_rejects_immediately(scheduler_asyncio):
    scheduler = make_scheduler(queue_capacity=1)
    running = run_immediate(scheduler.acquire())
    waiting = start_blocked(scheduler.acquire())
    with pytest.raises(SchedulerQueueFull):
        run_immediate(scheduler.acquire())
    assert run_immediate(scheduler.snapshot())["rejected_requests"] == 1
    run_immediate(running.release("completed"))
    next_lease = finish(waiting)
    run_immediate(next_lease.release("completed"))


def test_queue_timeout_removes_waiter(monkeypatch, scheduler_asyncio):
    async def timeout(_awaitable, timeout):
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", timeout)
    scheduler = make_scheduler()
    running = run_immediate(scheduler.acquire())
    with pytest.raises(SchedulerQueueTimeout):
        run_immediate(scheduler.acquire())
    snapshot = run_immediate(scheduler.snapshot())
    assert snapshot["queued_requests"] == 0
    assert snapshot["queue_timeout_requests"] == 1
    run_immediate(running.release("completed"))


def test_waiting_request_cancellation_releases_queue_place(scheduler_asyncio):
    scheduler = make_scheduler(queue_capacity=1)
    running = run_immediate(scheduler.acquire())
    waiting = start_blocked(scheduler.acquire())
    with pytest.raises(asyncio.CancelledError):
        waiting.throw(asyncio.CancelledError())
    snapshot = run_immediate(scheduler.snapshot())
    assert snapshot["queued_requests"] == 0
    assert snapshot["cancelled_requests"] == 1
    run_immediate(running.release("completed"))


def test_running_outcomes_and_metrics(scheduler_asyncio):
    scheduler = make_scheduler()
    for outcome in ("inference_timeout", "cancelled", "failed"):
        lease = run_immediate(scheduler.acquire())
        lease.cancel_event.set()
        run_immediate(lease.release(outcome))
    snapshot = run_immediate(scheduler.snapshot())
    assert snapshot["inference_timeout_requests"] == 1
    assert snapshot["cancelled_requests"] == 1
    assert snapshot["failed_requests"] == 1


def test_cancelled_release_can_be_retried_without_leaking_slot(scheduler_asyncio):
    scheduler = make_scheduler()
    lease = run_immediate(scheduler.acquire())
    original_release = scheduler._release

    async def interrupted_release(_lease, _outcome):
        raise asyncio.CancelledError

    scheduler._release = interrupted_release
    with pytest.raises(asyncio.CancelledError):
        run_immediate(lease.release("cancelled"))
    assert lease._released is False
    assert run_immediate(scheduler.snapshot())["running_requests"] == 1

    scheduler._release = original_release
    run_immediate(lease.release("cancelled"))
    assert run_immediate(scheduler.snapshot())["running_requests"] == 0


def test_shutdown_cancels_waiting_and_running_then_waits_for_release(scheduler_asyncio):
    scheduler = make_scheduler()
    scheduler._idle = FakeEvent()
    running = run_immediate(scheduler.acquire())
    waiting = start_blocked(scheduler.acquire())
    closing = start_blocked(scheduler.close())

    assert running.cancel_event.is_set()
    with pytest.raises(SchedulerClosed):
        finish(waiting)
    assert run_immediate(scheduler.snapshot())["shutting_down"] is True

    run_immediate(running.release("cancelled"))
    finish(closing)
    with pytest.raises(SchedulerClosed):
        run_immediate(scheduler.acquire())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_running": 0},
        {"queue_capacity": -1},
        {"queue_timeout_seconds": 0},
        {"inference_timeout_seconds": 0},
    ],
)
def test_invalid_scheduler_configuration(kwargs):
    with pytest.raises(ValueError):
        make_scheduler(**kwargs)
