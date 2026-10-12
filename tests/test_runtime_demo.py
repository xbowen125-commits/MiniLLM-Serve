"""Terminal demo behavior without models or GPU access."""

import json

from demo.runtime_demo import run_demo


def test_demo_lifecycle_tokens_and_cleanup():
    lines = []
    result = run_demo(lines.append)
    events = [json.loads(line) for line in lines]
    assert result["facade"] == {"status": "FINISHED", "generated_tokens": 2}
    assert result["continuous"] == {
        "batch_A": {"status": "FINISHED", "generated_tokens": 3},
        "batch_B": {"status": "FINISHED", "generated_tokens": 2},
    }
    for request_id in ("facade_A", "batch_A", "batch_B"):
        assert [e["phase"] for e in events if e["event"] == "phase_history"
                and e["request_id"] == request_id] == ["WAITING", "PREFILL", "DECODE", "FINISHED"]
    assert result["kv_allocated_bytes"] == result["kv_registered_requests"] == 0
    assert result["runtime_snapshot"]["active_requests"] == 0
    assert result["runtime_snapshot"]["finished_requests"] == 2
    assert sorted(result["released"]) == ["batch_A", "batch_B"]


def test_demo_incremental_admission_and_isolated_pages():
    lines = []
    result = run_demo(lines.append)
    assert result["steps"][1]["decoded"] == ["batch_A"]
    assert result["steps"][1]["prefilled"] == ["batch_B"]
    snapshots = [json.loads(line) for line in lines if json.loads(line)["event"] == "kv_snapshot"]
    first, second = snapshots[:2]
    assert first["block_count"] == 1
    assert second["request_tokens"] == {"batch_A": 3, "batch_B": 2}
    assert len(second["block_tables"]["batch_A"]) == 2
    assert set(second["block_tables"]["batch_A"].values()).isdisjoint(
        second["block_tables"]["batch_B"].values())
