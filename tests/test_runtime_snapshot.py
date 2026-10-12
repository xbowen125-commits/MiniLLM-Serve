"""Prompt-free runtime observability snapshot tests."""

from minillm.batching import DecodeState, FinishReason, RuntimeSnapshot


def test_runtime_snapshot_counts_states_tokens_and_steps():
    active = DecodeState(request_id="active", max_tokens=5)
    active.start(now=1.0)
    active.record_step()
    active.append_token(10, now=2.0)

    finished = DecodeState(request_id="finished", max_tokens=5)
    finished.start(now=1.0)
    finished.record_step()
    finished.record_step()
    finished.append_token(20, now=2.0)
    finished.append_token(21, now=3.0)
    finished.finish(FinishReason.STOP, now=4.0)

    snapshot = RuntimeSnapshot.from_states(
        [active, finished],
        snapshot_time=10.0,
    )

    assert snapshot.active_requests == 1
    assert snapshot.finished_requests == 1
    assert snapshot.failed_requests == 0
    assert snapshot.cancelled_requests == 0
    assert snapshot.total_generated_tokens == 3
    assert snapshot.total_decode_steps == 3
    assert snapshot.snapshot_time == 10.0
    assert not hasattr(snapshot, "prompt")
