"""Phase selection remains an optional runtime admission policy."""

import pytest

from minillm.batching import DecodeState
from minillm.engine import RequestState
from minillm.execution import ExecutionPhase
from minillm.scheduler.phase_policy import PhaseAwareBatchPolicy, PhaseSchedulingPolicy
from minillm.scheduler.snapshot import SchedulerSnapshot
from test_scheduler import finish, make_scheduler, run_immediate, scheduler_asyncio, start_blocked


def runtime_state(request_id, phase):
    state = DecodeState(request_id, max_tokens=8)
    if phase in {ExecutionPhase.PREFILL, ExecutionPhase.DECODE}:
        state.transition_phase(ExecutionPhase.PREFILL)
    if phase is ExecutionPhase.DECODE:
        state.transition_phase(ExecutionPhase.DECODE)
    if phase is ExecutionPhase.FINISHED:
        state.finish()
    return state


def test_classify_phases_without_modifying_requests():
    requests = [RequestState(request_id=name) for name in ("A", "B", "C")]
    policy = PhaseSchedulingPolicy({
        "B": runtime_state("B", ExecutionPhase.PREFILL),
        "C": runtime_state("C", ExecutionPhase.DECODE),
    })
    groups = policy.classify(requests)
    assert groups["waiting"] == [requests[0]]
    assert groups["prefill"] == [requests[1]]
    assert groups["decode"] == [requests[2]]
    assert groups["finished"] == []
    assert not hasattr(requests[0], "phase")


def test_decode_priority_and_stage_fifo():
    prefill = RequestState(request_id="prefill", created_time=0)
    newer = RequestState(request_id="newer", created_time=2)
    older = RequestState(request_id="older", created_time=1)
    waiting = RequestState(request_id="waiting", created_time=-1)
    policy = PhaseAwareBatchPolicy({
        "prefill": runtime_state("prefill", ExecutionPhase.PREFILL),
        "newer": runtime_state("newer", ExecutionPhase.DECODE),
        "older": runtime_state("older", ExecutionPhase.DECODE),
    })
    assert policy.select_batch([prefill, newer, waiting, older], 1) == [older]
    assert policy.select_batch([prefill, newer, waiting, older], 4) == [older, newer, prefill, waiting]


def test_batch_limit_and_finished_exclusion():
    requests = [RequestState(request_id=str(i), created_time=i) for i in range(10)]
    states = {r.request_id: runtime_state(r.request_id, ExecutionPhase.DECODE) for r in requests}
    policy = PhaseAwareBatchPolicy(states)
    assert policy.select_batch(list(reversed(requests)), 4) == requests[:4]
    requests[0].mark_cancelled()
    assert policy.select_batch(requests, 1) == [requests[1]]
    with pytest.raises(ValueError):
        policy.select_batch(requests, 0)


def test_snapshot_counts_phase_views():
    requests = [RequestState(request_id=str(i)) for i in range(4)]
    policy = PhaseSchedulingPolicy({
        "1": runtime_state("1", ExecutionPhase.PREFILL),
        "2": runtime_state("2", ExecutionPhase.DECODE),
        "3": runtime_state("3", ExecutionPhase.FINISHED),
    })
    snapshot = SchedulerSnapshot.from_requests(requests, policy, active_batch_size=2)
    assert (snapshot.waiting_count, snapshot.prefill_count, snapshot.decode_count, snapshot.finished_count) == (1, 1, 1, 1)
    assert snapshot.active_batch_size == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_scheduler_opt_in_preserves_lease_cleanup(enabled, scheduler_asyncio):
    states = {"3": runtime_state("3", ExecutionPhase.DECODE)}
    policy = PhaseAwareBatchPolicy(states) if enabled else None
    scheduler = make_scheduler(phase_policy=policy)
    running = run_immediate(scheduler.acquire())
    second_wait = start_blocked(scheduler.acquire())
    third_wait = start_blocked(scheduler.acquire())
    snapshot = run_immediate(scheduler.phase_snapshot())
    assert snapshot.active_batch_size == 1
    run_immediate(running.release("completed"))
    chosen = finish(third_wait if enabled else second_wait)
    assert chosen.request_id == (3 if enabled else 2)
    run_immediate(chosen.release("completed"))
    remaining = finish(second_wait if enabled else third_wait)
    run_immediate(remaining.release("completed"))
    assert run_immediate(scheduler.snapshot())["running_requests"] == 0
    assert run_immediate(scheduler.snapshot())["queued_requests"] == 0
