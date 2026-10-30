"""Offline per-iteration selection using the existing executor and fake worker."""

import pytest

from minillm.continuous import BatchExecutor, ContinuousBatch, IterationScheduler, IterationState
from test_continuous_batch import FakeWorker, context


def test_decode_priority_and_dynamic_arrival():
    scheduler = IterationScheduler(max_batch_size=2)
    executor = BatchExecutor(FakeWorker(), scheduler=scheduler)
    a, b, c = context("A", 3), context("B", 2), context("C", 2)
    executor.add(a)
    executor.step()
    executor.add(b)
    executor.add(c)
    plan = scheduler.build_plan(executor.batch)
    assert plan.decode_requests == (a,)
    assert plan.prefill_requests == (b,)
    assert plan.active_request_ids == ("A",)
    assert list(executor.batch.waiting_requests) == ["B", "C"]
    step = executor.execute_plan(plan)
    assert step.decoded == ("A",)
    assert step.prefilled == ("B",)
    assert executor.coordinator.contexts["A"] is a
    assert a.request.generated_tokens == [7]


def test_finished_request_exits_and_new_request_enters_next_iteration():
    scheduler = IterationScheduler(max_batch_size=1)
    executor = BatchExecutor(FakeWorker(), scheduler=scheduler)
    a, b = context("A", 1), context("B", 2)
    executor.add(a)
    executor.add(b)
    executor.step()
    step = executor.step()
    assert step.decoded == ("A",) and step.prefilled == ()
    assert executor.batch.active_requests == {}
    step = executor.step()
    assert step.prefilled == ("B",)
    assert scheduler.last_state == IterationState(3, (), ("B",), ("A",))
    executor.step()
    assert scheduler.last_state.finished_request_ids == ()


def test_terminal_cleanup_and_stable_fifo():
    scheduler = IterationScheduler(max_batch_size=2)
    batch = ContinuousBatch()
    a, b, c = context("A"), context("B"), context("C")
    a.request.created_time = b.request.created_time = 1
    for item in (a, b, c):
        scheduler.add(batch, item)
    c.request.mark_cancelled()
    plan = scheduler.build_plan(batch)
    assert plan.prefill_requests == (a, b)
    assert scheduler.last_state.finished_request_ids == ("C",)
    assert "C" not in batch.waiting_requests


def test_batch_limit_across_multiple_steps():
    scheduler = IterationScheduler(max_batch_size=2)
    executor = BatchExecutor(FakeWorker(), scheduler=scheduler)
    requests = [context(str(i), i + 1) for i in range(5)]
    for item in requests:
        executor.add(item)
    for _ in range(20):
        executor.step()
        state = scheduler.last_state
        assert len(state.decode_request_ids) + len(state.prefill_request_ids) <= 2
        assert len(executor.batch.active_requests) <= 2
        if all(c.request.is_terminal for c in requests):
            break
    assert all(c.request.is_terminal for c in requests)
    assert [len(c.request.generated_tokens) for c in requests] == [1, 2, 3, 4, 5]


def test_empty_iterations():
    scheduler = IterationScheduler()
    batch = ContinuousBatch()
    assert scheduler.build_plan(batch).decode_requests == ()
    assert scheduler.last_state == IterationState(1)
    scheduler.build_plan(batch)
    assert scheduler.last_state == IterationState(2)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_capacity(size):
    with pytest.raises(ValueError):
        IterationScheduler(max_batch_size=size)
