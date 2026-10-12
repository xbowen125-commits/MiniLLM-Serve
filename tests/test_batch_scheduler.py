"""Foundational batch selection tests without batched model execution."""

import pytest

from minillm.engine import RequestState, RequestStatus
from minillm.request_queue import RequestQueue
from minillm.scheduler import RequestScheduler
from test_scheduler import finish, make_scheduler, run_immediate, scheduler_asyncio, start_blocked


def make_request(request_id: str) -> RequestState:
    return RequestState(request_id=request_id)


def test_get_batch_preserves_fifo_order():
    queue = RequestQueue(capacity=4)
    requests = [make_request(str(index)) for index in range(4)]
    for request in requests:
        queue.put(request)

    assert queue.get_batch(3) == requests[:3]


def test_get_batch_respects_max_size_and_keeps_remainder():
    queue = RequestQueue(capacity=4)
    requests = [make_request(str(index)) for index in range(4)]
    for request in requests:
        queue.put(request)

    assert queue.get_batch(2) == requests[:2]
    assert len(queue) == 2
    assert queue.get_batch(4) == requests[2:]


def test_get_batch_from_empty_queue():
    assert RequestQueue(capacity=4).get_batch(4) == []


def test_get_batch_validates_size():
    queue = RequestQueue(capacity=4)
    with pytest.raises(ValueError, match="max_size"):
        queue.get_batch(0)


def test_schedule_batch_admits_fifo_requests_up_to_batch_size(scheduler_asyncio):
    scheduler = make_scheduler(max_running=1, queue_capacity=4, max_batch_size=2)
    running = run_immediate(scheduler.acquire())
    second_wait = start_blocked(scheduler.acquire())
    third_wait = start_blocked(scheduler.acquire())
    fourth_wait = start_blocked(scheduler.acquire())

    # Simulate capacity becoming available to exercise the future batch
    # coordinator entry point without adding batched model execution.
    scheduler.max_running = 3
    scheduled = run_immediate(scheduler.schedule_batch())

    assert [state.request_id for state in scheduled] == ["2", "3"]
    assert all(state.status is RequestStatus.RUNNING for state in scheduled)
    assert run_immediate(scheduler.snapshot())["queued_requests"] == 1

    second = finish(second_wait)
    third = finish(third_wait)
    run_immediate(running.release("completed"))
    fourth = finish(fourth_wait)
    run_immediate(second.release("completed"))
    run_immediate(third.release("completed"))
    run_immediate(fourth.release("completed"))


def test_schedule_batch_returns_empty_without_waiting_requests(scheduler_asyncio):
    scheduler = make_scheduler(max_batch_size=4)

    assert run_immediate(scheduler.schedule_batch()) == []


def test_invalid_max_batch_size():
    with pytest.raises(ValueError, match="max_batch_size"):
        RequestScheduler(
            max_running=1,
            queue_capacity=1,
            queue_timeout_seconds=1,
            inference_timeout_seconds=1,
            max_batch_size=0,
        )
