"""Reproducible JSON and Markdown export for virtual batch comparisons."""

import argparse
import json
from pathlib import Path

from .runner import run_comparison
from .scenario import scenarios


def generate_report(results):
    lines = ["# Continuous Batch Runtime Simulation", "",
             "Uses the existing offline BatchExecutor and a Fake Worker. No GPU or real model.",
             "", "## Timing model and definitions", "",
             "Each phase group costs base_s + per_request_s × group size; costs are configured assumptions.",
             "Decode runs before new prefill admission. Arrivals are admitted at cycle boundaries, not mid-step.",
             "Sequential uses one active slot; continuous uses the configured active limit. Both retain FIFO admission.",
             "TTFT = first token completion - arrival; queue = prefill start - arrival; total = finish - arrival.",
             "Throughput = generated tokens / simulated elapsed time (first arrival to last completion, including idle gaps).",
             "Active batch size is the decode-call group size, averaged over decode calls, not time-weighted occupancy.",
             "", "## Results", "",
             "| Scenario | Policy | Avg TTFT (s) | Avg queue (s) | Avg total (s) | Tokens | Sim tokens/s | Avg/max decode batch |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for result in results:
        for policy, values in result["results"].items():
            m = values["metrics"]
            lines.append(f"| {result['scenario']['name']} | {policy} | {m['average_ttft_s']:.6f} | "
                f"{m['average_queue_delay_s']:.6f} | {m['average_total_latency_s']:.6f} | "
                f"{m['generated_tokens']} | {m['simulated_tokens_per_second']:.6f} | "
                f"{m['average_active_decode_batch_size']:.3f}/{m['max_active_decode_batch_size']} |")
    lines.extend(["", "## Recorded experiment configuration", "", "```json",
                  json.dumps([r["scenario"] for r in results], indent=2), "```", "",
                  "## Limitations", "",
                  "This is logical multi-request execution, not GPU tensor batching or measured production throughput.",
                  "No KV memory, real prefill cost, sampling, GPU efficiency, or network latency is measured.",
                  "Batching may improve simulated throughput under shared-base costs, but is not assumed superior.",
                  "Runtime wall-clock timestamps are not used for these virtual-clock metrics.", "",
                  "## Reproduce (PowerShell, project root)", "", "```powershell",
                  ".\\.venv\\Scripts\\python.exe -m benchmarks.runtime.continuous_batch.report", "```", ""])
    return "\n".join(lines)


def save_reports(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "continuous_batch.json"
    markdown_path = output_dir / "continuous_batch.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(generate_report(results), encoding="utf-8")
    return json_path, markdown_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/runtime/results/continuous_batch"))
    args = parser.parse_args()
    for path in save_reports([run_comparison(s) for s in scenarios()], args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
