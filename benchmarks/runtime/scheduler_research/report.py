"""Export traceable JSON and Markdown for offline scheduler research."""

import argparse
import json
from pathlib import Path

from .runner import SchedulerResearchRunner
from .scenario import scenarios


def generate_report(results):
    lines = ["# Scheduler Research: Virtual Admission Simulation", "",
             "No GPU, model, HTTP, or online runtime changes. Uses existing RequestQueue,",
             "PhaseAwareBatchPolicy and FairPhaseBatchPolicy in a single-slot non-preemptive simulation.",
             "", "## Definitions", "",
             "Queue delay = selection time - arrival; P50/P95/P99 use linear interpolation over completed waits.",
             "TTFT = queue delay + first_token_fraction × synthetic service time; total latency = wait + service.",
             "These are request-arrival-relative synthetic metrics, not measured model first-token timings.",
             "Starvation count = requests whose wait strictly exceeds the configured threshold; not permanent starvation.",
             "Fairness = Jain index on inverse slowdown (service / total latency); 1 means equal slowdown,",
             "not necessarily low latency. Length normalization differs from equal token service shares.",
             "Phase priority: DECODE > PREFILL > WAITING, with FIFO within each phase.",
             "Fair aging: weights 100/50/10 plus creation age × aging_factor; FIFO breaks ties.",
             "", "## Results (seconds)", "",
             "| Scenario | Policy | Avg queue | P50 | P95 | P99 | Avg TTFT | Avg total | Starvation | Fairness |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for result in results:
        for policy, row in result["results"].items():
            m = row["metrics"]
            values = [m[k] for k in ("average_queue_delay", "p50_wait_time", "p95_wait_time",
                      "p99_wait_time", "average_ttft", "average_total_latency")]
            lines.append(f"| {result['scenario']['name']} | {policy} | " +
                         " | ".join(f"{v:.6f}" for v in values) +
                         f" | {m['starvation_count']} | {m['fairness_score']:.6f} |")
    lines.extend(["", "## Experiment configuration", "", "```json",
        json.dumps([r["scenario"] for r in results], indent=2), "```", "",
        "## Limitations", "",
        "Phase labels are fixed; selected requests run to completion. This does not model continuous decode",
        "or real prefill/decode costs. Finite workloads finish under every policy, so no starvation-freedom proof is made.",
        "Fair aging is not assumed superior: one common age bonus changes cross-phase ranking but not same-phase FIFO.",
        "Arrival offsets and service costs are deterministic experimental inputs, not historical model measurements.",
        "Per-request selection traces and full configuration are preserved in the companion JSON.", "",
        "## Reproduce (PowerShell, project root)", "", "```powershell",
        ".\\.venv\\Scripts\\python.exe -m benchmarks.runtime.scheduler_research.report", "```", ""])
    return "\n".join(lines)


def save_reports(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scheduler_research.json"
    markdown_path = output_dir / "scheduler_research.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(generate_report(results), encoding="utf-8")
    return json_path, markdown_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/runtime/results/scheduler_research"))
    args = parser.parse_args()
    runner = SchedulerResearchRunner()
    for path in save_reports([runner.run(s) for s in scenarios()], args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
