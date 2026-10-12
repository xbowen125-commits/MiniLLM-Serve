"""Run the pure-Python capacity experiment and export JSON and Markdown."""

import argparse
import json
from pathlib import Path

from .experiment import run_experiment
from .scenario import MemoryScenario


def generate_report(result: dict) -> str:
    lines = ["# Paged vs Contiguous KV Capacity Simulation", "",
             "## Experiment setup", "",
             f"Timestamp (UTC): {result['timestamp']}",
             "Scope: pure Python bookkeeping; no model, CUDA, or GPU measurement.",
             "", "```json", json.dumps(result["configuration"], indent=2), "```", "",
             "All requests remain resident until measurement, then are released.",
             "Contiguous reserves max_tokens per request; paged grows in fixed-size blocks.",
             "", "## Results", "",
             "| Strategy | Requested tokens | Allocated tokens | Wasted tokens | Block count | Utilization |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for policy, metrics in result["metrics"].items():
        lines.append(f"| {policy} | {metrics['requested_tokens']} | {metrics['allocated_tokens']} | "
                     f"{metrics['wasted_tokens']} | {metrics['block_count']} | {metrics['utilization']:.2%} |")
    lines.extend(["", "## Metric definitions and limitations", "",
                  "Requested tokens are occupied logical KV positions (total sequence length, not just output tokens).",
                  "Allocated tokens are reserved capacity; wasted = allocated - requested.",
                  "Utilization = requested / allocated. Contiguous block count means one allocation per request;",
                  "paged block count means fixed-size pages, so block counts are not equivalent allocation sizes.",
                  "Default lengths are multiples of 16; zero paged waste under that setup is expected, not universal.",
                  "No tensor memory, allocator overhead, external fragmentation, throughput, attention kernel,",
                  "or production benefit is measured. Paging can waste more than contiguous for some block sizes.",
                  f"Allocated token capacity after cleanup: {result['cleanup_allocated_tokens']}",
                  "", "## Reproduce (PowerShell, project root)", "",
                  "```powershell", ".\\.venv\\Scripts\\python.exe -m benchmarks.runtime.paged_memory.report `",
                  f"  --requests {result['configuration']['request_count']} --seed {result['configuration']['seed']} `",
                  f"  --max-tokens {result['configuration']['max_tokens']} --block-size {result['configuration']['tokens_per_block']} `",
                  f"  --growth-step {result['configuration']['growth_step']} `",
                  "  --lengths " + " ".join(map(str, result['configuration']['lengths'])),
                  "```", ""])
    return "\n".join(lines)


def save_report(result: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paged_memory.json"
    markdown_path = output_dir / "paged_memory.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(generate_report(result), encoding="utf-8")
    return json_path, markdown_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--growth-step", type=int, default=16)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/runtime/results/paged_memory"))
    args = parser.parse_args()
    result = run_experiment(MemoryScenario(args.requests, args.seed, tuple(args.lengths),
                            args.max_tokens, args.block_size, args.growth_step))
    for path in save_report(result, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
