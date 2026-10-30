"""JSONL round trip and final completion visibility."""

import json
from benchmarks.runtime.common.trace import export_trace
from benchmarks.runtime.orca.runner import run_benchmark
from benchmarks.runtime.orca.scenario import heterogeneous_workload
from benchmarks.runtime.orca.report import save_reports


def test_trace_export(tmp_path):
    result = run_benchmark(heterogeneous_workload())
    path = export_trace(result["iteration_trace"], tmp_path / "trace.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == result["metrics"]["iteration_count"]
    assert records[-1]["finished_request_ids"] == list(result["iteration_trace"][-1]["completed_this_iteration"])
    assert records[-1]["finished_request_ids"]
    assert records[0]["timestamp"] == result["iteration_trace"][0]["start_s"]
    assert all({"iteration_id", "timestamp", "decode_request_ids", "prefill_request_ids", "finished_request_ids"} <= r.keys() for r in records)
    save_reports([result], tmp_path / "reports")
    assert (tmp_path / "reports/case_1_iterations.jsonl").exists()
    assert result["environment"]["python"]
