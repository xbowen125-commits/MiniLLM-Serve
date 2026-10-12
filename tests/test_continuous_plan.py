"""Stable plans preserve shared contexts and multi-cycle execution state."""

import pytest

from minillm.continuous import BatchExecutor, BatchScheduler
from minillm.contracts import ExecutionPhase
from test_continuous_batch import FakeWorker, context


def test_plan_contains_decode_and_prefill_without_mutating_batch():
    executor = BatchExecutor(FakeWorker())
    first, second = context("A"), context("B")
    executor.add(first)
    executor.step()
    executor.add(second)
    plan = executor.scheduler.build_plan(executor.batch)
    assert plan.decode_requests == (first,)
    assert plan.prefill_requests == (second,)
    assert plan.active_request_ids == ("A",)
    assert list(executor.batch.waiting_requests) == ["B"]
    result = executor.execute_plan(plan)
    assert [r.phase for r in result.results] == [ExecutionPhase.DECODE, ExecutionPhase.PREFILL]
    assert result.results[0].token_result.token_id == 7


def test_plan_order_is_stable():
    executor = BatchExecutor(FakeWorker())
    first, second = context("A"), context("B")
    first.request.created_time = second.request.created_time = 1
    executor.add(first)
    executor.add(second)
    a = executor.scheduler.build_plan(executor.batch)
    b = executor.scheduler.build_plan(executor.batch)
    assert a == b
    assert a.prefill_requests == (first, second)


def test_finished_request_removed_and_new_request_prefilled_same_cycle():
    executor = BatchExecutor(FakeWorker(), scheduler=BatchScheduler(max_active=1))
    first, second = context("A", 1), context("B", 2)
    executor.add(first)
    executor.add(second)
    executor.step()
    result = executor.step()
    assert result.retired == ("A",)
    assert result.prefilled == ("B",)
    assert result.results[0].finished is True
    assert executor.batch.active_requests == {"B": second}
    executor.step()
    assert second.request.generated_tokens == [7]
    executor.step()
    assert executor.batch.active_requests == {}
    assert executor.batch.finished_requests == {"A": first, "B": second}


def test_stale_plan_rejected_before_worker_call():
    worker = FakeWorker()
    executor = BatchExecutor(worker)
    current = context("A")
    executor.add(current)
    plan = executor.scheduler.build_plan(executor.batch)
    executor.execute_plan(plan)
    count = len(worker.calls)
    with pytest.raises(RuntimeError):
        executor.execute_plan(plan)
    assert len(worker.calls) == count
