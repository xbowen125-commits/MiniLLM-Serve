"""Deterministic phase priority, aging, and admission tests."""

import pytest

from minillm.engine import RequestState
from minillm.execution import ExecutionPhase
from minillm.scheduler import FairPhaseBatchPolicy, PhasePriority
from minillm.scheduler.snapshot import SchedulerSnapshot
from test_phase_scheduler import runtime_state
from test_scheduler import make_scheduler, run_immediate, start_blocked, finish, scheduler_asyncio


def policy(aging_factor=1.0):
    return FairPhaseBatchPolicy(
        {"decode": runtime_state("decode", ExecutionPhase.DECODE)},
        aging_factor=aging_factor, clock=lambda: 100.0,
    )


def test_phase_base_priority():
    fair = policy()
    decode = RequestState(request_id="decode", created_time=100)
    waiting = RequestState(request_id="waiting", created_time=100)
    assert fair.priority.score(decode) == 100
    assert fair.priority.score(waiting) == 10
    assert fair.select_batch([waiting, decode], 1) == [decode]


def test_aging_can_overcome_decode_priority():
    fair = policy()
    decode = RequestState(request_id="decode", created_time=99)
    waiting = RequestState(request_id="waiting", created_time=0)
    assert fair.priority.score(waiting) > fair.priority.score(decode)
    assert fair.select_batch([decode, waiting], 1) == [waiting]


def test_same_phase_fifo_and_stable_ties():
    older = RequestState(request_id="old", created_time=1)
    newer = RequestState(request_id="new", created_time=2)
    tied = RequestState(request_id="tie", created_time=2)
    states = {r.request_id: runtime_state(r.request_id, ExecutionPhase.DECODE)
              for r in [older, newer, tied]}
    fair = FairPhaseBatchPolicy(states, aging_factor=0, clock=lambda: 100)
    assert fair.select_batch([newer, tied, older], 3) == [older, newer, tied]


def test_starvation_snapshot_excludes_finished():
    fair = policy()
    old = RequestState(request_id="old", created_time=0)
    recent = RequestState(request_id="recent", created_time=99)
    finished = RequestState(request_id="finished", created_time=-100)
    finished.mark_cancelled()
    snapshot = SchedulerSnapshot.from_requests([old, recent, finished], fair, now=100)
    assert snapshot.oldest_wait_ms == 100000
    assert snapshot.starvation_candidates == ("old",)


def test_batch_size_and_terminal_exclusion():
    fair = policy()
    requests = [RequestState(request_id=str(i), created_time=i) for i in range(10)]
    assert len(fair.select_batch(requests, 4)) == 4
    requests[0].mark_cancelled()
    assert requests[0] not in fair.select_batch(requests, 10)
    with pytest.raises(ValueError):
        fair.select_batch(requests, 0)


@pytest.mark.parametrize("fair_enabled", [False, True])
def test_optional_fair_policy_and_default_fifo(fair_enabled, scheduler_asyncio):
    fair = FairPhaseBatchPolicy(clock=lambda: 1000) if fair_enabled else None
    scheduler = make_scheduler(phase_policy=fair)
    first = run_immediate(scheduler.acquire())
    second_wait = start_blocked(scheduler.acquire())
    third_wait = start_blocked(scheduler.acquire())
    run_immediate(first.release("completed"))
    second = finish(second_wait)
    assert second.request_id == 2
    run_immediate(second.release("completed"))
    third = finish(third_wait)
    run_immediate(third.release("completed"))
    assert run_immediate(scheduler.snapshot())["running_requests"] == 0


def test_invalid_aging_factor():
    with pytest.raises(ValueError):
        policy(-1)
