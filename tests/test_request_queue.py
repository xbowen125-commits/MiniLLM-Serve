"""Unit tests for the scheduler's bounded FIFO queue."""

import pytest

from minillm.engine import RequestState
from minillm.request_queue import RequestQueue


def request(request_id: str) -> RequestState:
    return RequestState(request_id=request_id)


def test_request_queue_preserves_fifo_order():
    request_queue = RequestQueue(capacity=3)
    first = request("first")
    second = request("second")
    third = request("third")

    assert request_queue.put(first) is True
    assert request_queue.put(second) is True
    assert request_queue.put(third) is True

    assert request_queue.pop() is first
    assert request_queue.pop() is second
    assert request_queue.pop() is third


def test_request_queue_enforces_capacity_without_mutating_contents():
    request_queue = RequestQueue(capacity=1)
    first = request("first")

    assert request_queue.put(first) is True
    assert request_queue.is_full is True
    assert request_queue.put(request("second")) is False
    assert len(request_queue) == 1
    assert request_queue.pop() is first


def test_request_queue_remove_preserves_remaining_order():
    request_queue = RequestQueue(capacity=3)
    first = request("first")
    second = request("second")
    third = request("third")
    request_queue.put(first)
    request_queue.put(second)
    request_queue.put(third)

    assert request_queue.remove(second) is True
    assert request_queue.remove(request("missing")) is False
    assert request_queue.drain() == [first, third]


def test_request_queue_drain_empties_queue():
    request_queue = RequestQueue(capacity=2)
    first = request("first")
    second = request("second")
    request_queue.put(first)
    request_queue.put(second)

    assert request_queue.drain() == [first, second]
    assert len(request_queue) == 0
    assert bool(request_queue) is False


def test_zero_capacity_queue_always_rejects_items():
    request_queue = RequestQueue(capacity=0)

    assert request_queue.is_full is True
    assert request_queue.put(request("first")) is False


def test_request_queue_validates_capacity_and_empty_pop():
    with pytest.raises(ValueError):
        RequestQueue(capacity=-1)

    with pytest.raises(IndexError):
        RequestQueue(capacity=1).pop()


def test_request_queue_rejects_non_waiting_state():
    running = request("running")
    running.mark_running()

    with pytest.raises(ValueError, match="only waiting requests"):
        RequestQueue(capacity=1).put(running)
