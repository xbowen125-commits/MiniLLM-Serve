"""Config-driven virtual-clock experiments with JSON and Markdown artifacts."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from minillm.contracts import ExecutionPhase
from .scheduler_compare import SimulatedRequest
from .stress_scenarios import StressScenario

ROOT = Path(__file__).resolve().parent


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    build_scenario(config)
    return config


def build_scenario(config):
    if config.get("scenario") not in {"decode_flood", "mixed_length", "continuous_arrival"}:
        raise ValueError("unsupported scenario")
    groups = config.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a non-empty list")
    requests = []
    for group in groups:
        count = group["count"]
        if type(count) is not int or not 1 <= count <= 1000 or len(requests) + count > 1000:
            raise ValueError("workload must contain 1..1000 requests")
        interval = group["arrival_interval"]
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("arrival interval must be finite and non-negative")
        if not group["phases"] or not group["service_seconds"]:
            raise ValueError("phase and service cycles must not be empty")
        for i in range(count):
            requests.append(SimulatedRequest(
                f"{group['prefix']}-{i}",
                ExecutionPhase(group["phases"][i % len(group["phases"])]),
                group["arrival_start"] + i * interval,
                group["service_seconds"][i % len(group["service_seconds"])],
            ))
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("duplicate workload request IDs")
    return StressScenario(config["scenario"], tuple(requests), config["start_time"],
                          config["starvation_threshold"], config["aging_factor"])


def run_experiment(config):
    # Normalize and detach the complete configuration saved in every record.
    config = json.loads(json.dumps(config, allow_nan=False))
    scenario = build_scenario(config)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()
    return [
        {"schema_version": 1, "scope": "virtual-clock-admission-simulation",
         "scenario": scenario.name, "policy": policy, "workload_config": config,
         "config_sha256": digest, "timestamp": timestamp, "delay_unit": "seconds",
         "metrics": asdict(result)}
        for policy, result in scenario.run().items()
    ]


def save_results(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def generate_report(records):
    lines = ["# Runtime Benchmark v2 Report", "", "## Experiment setup", "",
             "Deterministic virtual-clock admission simulation; one non-preemptive slot.",
             "Each row represents one finite workload run. No real model or GPU executes.",
             "Complete configurations and UTC run timestamps are recorded in JSON.", "",
             "Reproduce from the project root:", "", "```powershell",
             ".\\.venv\\Scripts\\python.exe -m benchmarks.runtime.report_generator", "```", "",
             "## Runtime architecture", "",
             "Configuration → StressScenario → SchedulerComparisonRunner → FIFO / FairPhaseBatchPolicy.",
             "This admission simulator uses request and phase contracts; it does not drive RuntimeFacade",
             "or model execution. The separate FakeWorker benchmark exercises RuntimeFacade.", "",
             "## Metrics definition", "",
             "Queue delay = selection minus arrival, in virtual seconds. P50/P95 use linear interpolation.",
             "Starvation count = unique delays strictly above the configured threshold.",
             "Completed/s = completions / simulated duration (including idle gaps).",
             "Service distribution counts whole-request selections, not token steps.", "",
             "## Scenario description", ""]
    seen = set()
    for record in records:
        if record["scenario"] in seen:
            continue
        seen.add(record["scenario"])
        config = record["workload_config"]
        lines.extend([f"### {record['scenario']}", "",
                      f"Run timestamp (UTC): {record['timestamp']}", "",
                      f"Config SHA-256: `{record['config_sha256']}`", "",
                      "```json", json.dumps(config, indent=2), "```", ""])
    lines.extend(["## Results summary", "",
                  "| Scenario | Policy | Completed | Avg queue (s) | Max queue (s) | P50 (s) | P95 (s) | Starvation | Completed/s |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for record in records:
        m = record["metrics"]
        lines.append(f"| {record['scenario']} | {record['policy']} | {m['completed_requests']} | "
                     f"{m['average_queue_delay']:.6f} | {m['max_queue_delay']:.6f} | "
                     f"{m['p50_queue_delay']:.6f} | {m['p95_queue_delay']:.6f} | "
                     f"{m['starvation_count']} | {m['completed_per_sim_second']:.6f} |")
    lines.extend(["", "## Limitations", "",
                  "All durations are synthetic, not measured serving latency. Phases are fixed labels.",
                  "No continuous batching, GPU work, token scheduling, cache pressure, or HTTP is modeled.",
                  "Fair may worsen long-tail waiting. Identical repeated inputs reproduce logical metrics",
                  "rather than independent performance samples. No universal fairness improvement is claimed.",
                  "Timestamps vary between runs; configuration hashes and metrics should match.",
                  "Small finite samples do not establish production capacity or permanent starvation."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "runtime_benchmark_report.md")
    args = parser.parse_args()
    configs = sorted(args.config_dir.glob("*.json"))
    if not configs:
        parser.error("no experiment configurations found")
    records = []
    for path in configs:
        result = run_experiment(load_config(path))
        result_path = args.results_dir / f"{path.stem}.json"
        save_results(result, result_path)
        records.extend(json.loads(result_path.read_text(encoding="utf-8")))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(generate_report(records), encoding="utf-8")
    print(f"Generated {len(records)} policy results and {args.report.name}")


if __name__ == "__main__":
    main()
