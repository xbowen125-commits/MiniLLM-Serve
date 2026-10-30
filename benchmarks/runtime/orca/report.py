"""JSON and Markdown for Orca-inspired logical iteration experiments."""

import argparse
import json
from pathlib import Path

from .runner import run_benchmark
from .scenario import scenarios
from ..common.trace import export_trace


def generate_report(results):
    lines = ["# Orca-inspired Iteration Scheduling Benchmark", "",
        "## Experimental Scope", "",
        "Scope: existing offline Continuous BatchExecutor and IterationScheduler, with an unchanged Fake Worker.",
        "No model, GPU, selective tensor batching or production Orca implementation is exercised.", "",
        "## Methodology", "",
        "Every iteration retains active Decode requests and uses remaining capacity for FIFO Prefill admission.",
        "A slot freed by decode is available to admission on the next iteration. Arrivals enter at cycle boundaries.",
        "Virtual phase cost = base_s + per_request_s * group size; these are assumptions, not measurements.",
        "Prompt length is synthetic workload metadata: it does not change the reused Fake Worker's fixed per-request prefill cost.",
        "Generation length controls the actual number of fake decode tokens; arrival order is stable for timestamp ties.",
        "TTFT and total latency start at request arrival, including queue time. Throughput includes idle gaps.",
        "Batch utilization is group size / configured capacity, weighted by busy phase durations; idle gaps excluded.",
        "Decode fairness is the Jain index of tokens / eligible active-decode iterations, not admission fairness.", "",
        "Average batch size is the arithmetic mean of decode plus prefill selections per iteration, not a time-weighted phase size.", "",
        "Identical iteration counts or average batch sizes do not imply identical arrival scheduling; compare per-iteration ID traces.", "",
        "## Results", "",
        "| Case | Requests | Tokens | Avg TTFT (s) | Avg total (s) | Sim tokens/s | Utilization | Decode fairness | Avg batch | Iterations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for i, result in enumerate(results, 1):
        m = result["metrics"]
        lines.append(f"| {i} | {m['completed_requests']} | {m['generated_tokens']} | "
            f"{m['average_ttft_s']:.6f} | {m['average_total_latency_s']:.6f} | "
            f"{m['simulated_tokens_per_second']:.6f} | {m['batch_utilization']:.2%} | {m['decode_fairness']:.6f} | "
            f"{m['average_batch_size']:.6f} | {m['iteration_count']} |")
    lines.extend(["", "## Recorded configuration", "", "```json",
        json.dumps([r["scenario"] for r in results], indent=2), "```", "",
        "## Analysis", "",
        "Long requests remain active while short requests enter available prefill slots; iteration traces expose turnover.",
        "Low utilization may reflect a long request outliving its short peers, not inefficient real GPU kernels.",
        "A higher average batch size does not by itself prove lower latency; admission delay and assumed phase costs matter.",
        "## Limitations", "",
        "JSON records every iteration ID, planned decode/prefill IDs, arrivals, virtual start/end,",
        "newly observed terminal IDs, actual retirements, and remaining active/waiting requests.",
        "finished_request_ids is a selection-time observation; completed_this_iteration records execution outcomes.",
        "The final retirement need not appear in another selection-time snapshot because no extra iteration is run.",
        "All eligible decode requests receive one token per iteration, so fairness normally equals 1;",
        "this does not prove low short-request latency or starvation freedom. Per-request TTFT exposes queueing.",
        "A finite heterogeneous workload is not a GPU performance study. No KV pressure, real prompt costs,",
        "network, CUDA kernels, PagedAttention, chunked prefill or prefix cache are modeled.",
        "No claim of outperforming vLLM or reproducing Orca's published speedups is made.", "",
        "## Reproduce (PowerShell, project root)", "", "```powershell",
        ".\\.venv\\Scripts\\python.exe -m benchmarks.runtime.orca.report", "```", ""])
    return "\n".join(lines)


def save_reports(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_dir / "orca_iteration.json", output_dir / "orca_iteration.md"
    paths[0].write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    paths[1].write_text(generate_report(results), encoding="utf-8")
    for index, result in enumerate(results, 1):
        export_trace(result["iteration_trace"], output_dir / f"case_{index}_iterations.jsonl")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/runtime/results/orca-validity"))
    args = parser.parse_args()
    for path in save_reports([run_benchmark(s) for s in scenarios()], args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
