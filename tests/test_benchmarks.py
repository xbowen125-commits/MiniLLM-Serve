"""Benchmark math and protocol parsing tests; no model or server required."""

import json
from pathlib import Path

import pytest

from benchmarks.benchmark_http import request_once, summarize_case
from benchmarks.benchmark_utils import distribution, percentile, write_csv, write_json, write_markdown
from benchmarks.stress_http import count_error_codes


class FakeHeaders(dict):
    def items(self):
        return super().items()


class FakeResponse:
    def __init__(self, lines, *, status=200, headers=None):
        self.lines = lines
        self.status = status
        self.headers = FakeHeaders(headers or {})

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return b"".join(self.lines)


def test_percentiles_use_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([], 99) is None
    result = distribution([1, None, 3])
    assert result["count"] == 2
    assert result["mean"] == 2


def test_stream_metrics_count_tokens_not_sse_chunks(monkeypatch):
    events = [
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "hello "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "world"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    lines = [f"data: {json.dumps(event)}\n\n".encode() for event in events] + [b"data: [DONE]\n\n"]
    response = FakeResponse(
        lines,
        headers={"X-MiniLLM-Request-ID": "7", "X-MiniLLM-Queue-Wait-Ms": "12.5"},
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)
    result = request_once(
        base_url="http://test",
        model="test",
        prompt="prompt",
        max_tokens=8,
        stream=True,
        seed=1,
        timeout=2,
        token_count=lambda text: len(text.split()),
        request_index=0,
    )
    assert result["success"] is True
    assert result["completion_tokens"] == 2
    assert result["completion_tokens"] != len(events)
    assert result["ttft_ms"] is not None
    assert result["tpot_ms"] is not None
    assert result["queue_wait_ms"] == 12.5
    assert result["request_id"] == "7"


def test_non_stream_uses_server_token_usage(monkeypatch):
    payload = {
        "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse([json.dumps(payload).encode()]),
    )
    result = request_once(
        base_url="http://test",
        model="test",
        prompt="prompt",
        max_tokens=8,
        stream=False,
        seed=1,
        timeout=2,
        token_count=None,
        request_index=0,
    )
    assert result["completion_tokens"] == 3
    assert result["prompt_tokens"] == 11
    assert result["ttft_ms"] is None
    assert result["tpot_ms"] is None


def test_case_summary_and_report_exports():
    runs = [
        {
            "success": True,
            "error_code": None,
            "status": 200,
            "e2e_ms": value,
            "ttft_ms": value / 2,
            "tpot_ms": 2.0,
            "queue_wait_ms": 1.0,
            "tokens_per_second": 10.0,
        }
        for value in (10.0, 20.0, 30.0)
    ]
    summary = summarize_case(
        runs,
        wall_seconds=1,
        stream=True,
        concurrency=2,
        target_prompt_tokens=32,
        max_tokens=16,
        kv_cache_state="on",
    )
    assert summary["request_throughput_rps"] == 3
    assert summary["e2e_p50_ms"] == 20
    assert summary["e2e_p95_ms"] == pytest.approx(29)

    output_dir = Path("benchmarks/results/.test-reports")
    json_path = output_dir / "result.json"
    csv_path = output_dir / "result.csv"
    md_path = output_dir / "result.md"
    try:
        write_json(json_path, {"summary": summary})
        write_csv(csv_path, [summary])
        write_markdown(
            md_path,
            title="Report",
            metadata={"model": "mock"},
            rows=[summary],
            columns=("stream", "e2e_p50_ms"),
            notes=("Mock data only.",),
        )
        assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["successful"] == 3
        assert "e2e_p50_ms" in csv_path.read_text(encoding="utf-8-sig")
        assert "# Report" in md_path.read_text(encoding="utf-8")
    finally:
        for path in (json_path, csv_path, md_path):
            path.unlink(missing_ok=True)
        output_dir.rmdir()


def test_stress_error_counts_include_disconnect_and_isolation_phases():
    records = [
        {"success": True, "status": 200, "error_code": None},
        {"success": False, "status": 503, "error_code": "queue_timeout"},
        {"success": False, "status": 503, "error_code": "queue_timeout"},
        {"success": False, "status": 429, "error_code": "queue_full"},
    ]
    assert count_error_codes(records) == {"queue_timeout": 2, "queue_full": 1}
