"""Generate evidence-based Markdown from saved real-worker benchmark JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
METRIC_NAMES = ("ttft_ms", "decode_latency_ms", "total_latency_ms", "tokens_per_second")


def parse_benchmark(data: dict) -> dict:
    """Validate the report contract without substituting missing measurements."""
    if data.get("schema_version") != 1 or data.get("scope") != "real-transformers-worker":
        raise ValueError("expected schema v1 real-transformers-worker results")
    for key in ("model_name", "device", "dtype", "timestamp"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"missing experiment metadata: {key}")
    config = data.get("configuration", {})
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must contain measured results")
    if config.get("repeats") != len(runs):
        raise ValueError("configured repeats do not match measured run count")
    if type(config.get("max_tokens")) is not int or config["max_tokens"] < 1:
        raise ValueError("max_tokens must be a positive integer")
    seen = set()
    for run in runs:
        index = run.get("run_index")
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("invalid or duplicate run_index")
        seen.add(index)
        for key in ("prompt_length", "generated_tokens"):
            if type(run.get(key)) is not int or run[key] < 0:
                raise ValueError(f"invalid {key}")
        metrics = run.get("metrics", {})
        if metrics.get("generated_tokens") != run["generated_tokens"]:
            raise ValueError("generated token counts disagree")
        for key in METRIC_NAMES:
            if key not in metrics:
                raise ValueError(f"missing metric: {key}")
            value = metrics[key]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"invalid metric: {key}")
    return data


def load_benchmark(path: Path | str) -> dict:
    return parse_benchmark(json.loads(Path(path).read_text(encoding="utf-8")))


def aggregate_metrics(data: dict) -> dict:
    parse_benchmark(data)
    aggregated = {}
    for name in METRIC_NAMES:
        values = [run["metrics"][name] for run in data["runs"] if run["metrics"][name] is not None]
        aggregated[name] = {"average": mean(values) if values else None, "samples": len(values)}
    return aggregated


def display(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def escape(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def generate_report(data: dict, *, source_name: str = "real_worker_cpu.json",
                    source_sha256: str | None = None) -> str:
    aggregated = aggregate_metrics(data)
    config = data["configuration"]
    lines = ["# Real TransformersWorker Benchmark Report", "", "## Experiment Setup", "",
             f"Source artifact: `{escape(source_name)}` (local JSON; Git-ignored).",
             f"Recorded run timestamp: {escape(data['timestamp'])}", ""]
    if source_sha256:
        lines.extend([f"Source SHA-256: `{source_sha256}`", ""])
    lines.extend(["| Parameter | Recorded value |", "| --- | --- |",
                  f"| Model | {escape(data['model_name'])} |",
                  f"| Device | {escape(data['device'])} |",
                  f"| dtype | {escape(data['dtype'])} |",
                  f"| max_tokens | {config['max_tokens']} |",
                  f"| Measured repeats | {config['repeats']} |",
                  f"| Warmups | {escape(config.get('warmups', 'N/A'))} |"])
    for key, value in data.get("versions", {}).items():
        lines.append(f"| {escape(key)} version | {escape(value)} |")
    lines.extend(["", "## Metrics", "",
                  "Arithmetic means are calculated from the recorded per-run values.",
                  "Unavailable values are excluded with explicit sample counts; they are not zero.", "",
                  "| Metric | Average | Unit | Valid samples |", "| --- | ---: | --- | ---: |"])
    names = ("TTFT", "Decode latency", "Total latency", "Tokens/s")
    for key, name in zip(METRIC_NAMES, names):
        result = aggregated[key]
        lines.append(f"| {name} | {display(result['average'])} | "
                     f"{'tokens/s' if key == 'tokens_per_second' else 'ms'} | {result['samples']} |")
    lines.extend(["", "TTFT runs from prefill start to the first observed token. Decode latency sums",
                  "all decode steps, including first-token sampling. Total latency includes prefill",
                  "and loop overhead. Tokens/s divides generated tokens by total latency; the table",
                  "reports the arithmetic mean of per-run ratios, not an aggregate throughput ratio.",
                  "Model loading and warmup samples are excluded by the benchmark producer.", "",
                  "## Run Summary", "",
                  "| Run | Prompt tokens | Generated tokens | TTFT (ms) | Decode (ms) | Total (ms) | Tokens/s | Finish |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for run in data["runs"]:
        m = run["metrics"]
        lines.append(f"| {run['run_index']} | {run['prompt_length']} | {run['generated_tokens']} | "
                     f"{display(m['ttft_ms'])} | {display(m['decode_latency_ms'])} | "
                     f"{display(m['total_latency_ms'])} | {display(m['tokens_per_second'])} | "
                     f"{escape(run.get('finish_reason', 'N/A'))} |")
    lines.extend(["", "## Limitations", "",
                  "- This is a single-request sequential worker experiment, with no tensor batching.",
                  f"- Recorded device is `{escape(data['device'])}`; this report is not GPU validation.",
                  "- For CPU execution, CUDA memory measurements are N/A; no memory values are inferred.",
                  "- Tokens/s is per-request generation rate, not production request throughput.",
                  "- A small number of repeats does not establish stable tail latency or capacity.",
                  "- CPU thread count, hardware details, and prompt text are absent from the source",
                  "  artifact, limiting exact reproduction. No commit SHA is recorded.",
                  "- No HTTP, concurrency, Continuous Batching, or PagedAttention is measured.",
                  "- This document summarizes an existing artifact and does not rerun inference.", "",
                  "Report reproduction from the project root:", "", "```powershell",
                  ".\\.venv\\Scripts\\python.exe -m benchmarks.runtime.real_model.report_generator",
                  "```"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "results" / "real_worker_cpu.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "real_worker_cpu_report.md")
    args = parser.parse_args()
    data = load_benchmark(args.input)
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    report = generate_report(data, source_name=args.input.name, source_sha256=digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Generated {args.output.name} from {len(data['runs'])} recorded runs")


if __name__ == "__main__":
    main()
