"""Read-only Markdown rendering for the frozen online HTTP protocol.

The runner owns exclusive artifact creation. This module never edits raw rows,
merges historical result directories, filters trials, or infers missing metrics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


def _value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else "N/A"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_value(value) for value in row) + " |" for row in rows)
    return lines


def _json(value: Any) -> list[str]:
    return ["```json", json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), "```"]


def _artifact(value: Any) -> str:
    """Link only relative artifact paths; do not publish local absolute paths."""
    if not value:
        return "N/A"
    text = str(value)
    windows = PureWindowsPath(text)
    posix = PurePosixPath(text.replace("\\", "/"))
    if windows.is_absolute() or posix.is_absolute() or ".." in posix.parts:
        return f"`{windows.name if windows.is_absolute() else posix.name}` (local artifact)"
    relative = posix.as_posix()
    # Runner-generated filenames are simple relative paths. A custom unsafe
    # Markdown target is rendered as text rather than interpreted as a link.
    if any(char in relative for char in "[]()<>\n\r"):
        return _value(relative)
    return f"[{relative}]({relative})"


def render_report(experiment: Mapping[str, Any]) -> str:
    """Render every trial, including invalid/failed trials, without file writes.

    Trial summary populations and timing formulas come from ``metrics``. No
    pooled speedup or best-trial selection is performed here. Unknown actual
    SSE token counts, internal execution times and exact phase peaks stay N/A.
    """
    trials = list(experiment.get("trials", []))
    evidence = experiment.get("evidence_kind")
    lines = ["# Frozen Online HTTP Benchmark Report", "", "## Experimental scope", ""]
    if evidence == "cpu_fake":
        lines.extend([
            "**CPU/Fake HTTP protocol validation only. These values are not real model or GPU performance evidence.**",
            "No throughput improvement, latency improvement or production readiness is inferred.", "",
        ])
    else:
        lines.extend([
            "Client-observed, bounded HTTP measurements for the exact recorded configuration only.",
            "Real GPU evidence requires successful explicit GPU execution and verified server provenance;",
            "a configuration label alone is not proof that such an experiment ran.", "",
        ])
    provenance = {key: experiment.get(key) for key in
                  ("config_sha256", "trace_sha256", "evidence_kind", "git", "planned_order")}
    planned = experiment.get("planned_order")
    planned_count = len(planned) if isinstance(planned, (list, tuple)) else None
    unrun_count = max(0, planned_count - len(trials)) if planned_count is not None else None
    attempted_keys = {(trial.get("pair_index"), trial.get("mode")) for trial in trials}
    unrun = [item for item in planned or []
             if isinstance(item, (list, tuple)) and len(item) == 2 and tuple(item) not in attempted_keys]
    lines.extend(["## Frozen configuration and provenance", "", *_json(provenance), "",
                  "## Experiment completion and final validity", "",
                  *_table(["Trial accounting", "Count"], [
                      ["Planned trials", planned_count], ["Executed trial records", len(trials)],
                      ["Unrun trials", unrun_count]]), "",
                  f"Experiment aborted: **{_value(experiment.get('aborted'))}**.", "",
                  f"Source unchanged during experiment: **{_value(experiment.get('source_unchanged'))}**.", "",
                  "Unrun planned entries (pair index, mode):", "", *_json(unrun), "",
                  "Final pair validation:", "", *_json(experiment.get("pair_validation")), "",
                  "## Formal admission gate", "", *_json(experiment.get("admission")), "",
                  "Raw summary values below are diagnostic observations, including invalid trials.",
                  "Only a passing complete validity gate can populate formal_statistics; CPU/Fake is never formal GPU evidence.", "",
                  "The root summary and appended final-verdict files are authoritative after pair/source validation.",
                  "Each immutable per-trial summary has trial_local_only validity scope; it is not a final paired verdict.", "",
                  "All executed trial records appear below, including failures and invalid comparisons; unrun trials are not invented.",
                  "Warmup rows are retained separately and never pooled into formal samples.", "",
                  "## Trial results", ""])
    rows = []
    for trial in trials:
        summary = trial.get("summary") or {}
        rows.append([
            trial.get("trial_id"), trial.get("pair_index"), trial.get("mode"),
            trial.get("comparison_valid"), summary.get("planned_requests"),
            summary.get("completed_requests"), summary.get("failed_requests"),
            summary.get("completed_generated_output_tokens_per_s"),
            summary.get("completed_visible_output_tokens_per_s"),
            summary.get("completed_requests_per_s"), summary.get("slo_goodput_requests_per_s"),
        ])
    lines.extend(_table([
        "Trial", "Pair", "Mode", "Comparison valid", "Planned", "Completed", "Failed",
        "Actual generated tokens/s", "Visible text tokens/s proxy", "Completed requests/s", "SLO goodput requests/s",
    ], rows))
    if not trials:
        lines.extend(["", "No executed trial records are present; no experiment result is claimed."])
    for trial in trials:
        summary = trial.get("summary") or {}
        distributions = summary.get("distributions") or {}
        lines.extend(["", f"### Trial {_value(trial.get('trial_id'))}: {_value(trial.get('mode'))}", "",
                      f"Comparison valid: **{_value(trial.get('comparison_valid'))}**.", "",
                      "Issues (retained, not filtered):"])
        issues = trial.get("issues") or []
        lines.extend(["", *[f"- {_value(issue)}" for issue in issues]] if issues else ["", "None recorded."])
        lines.extend(["", f"Raw request journal: {_artifact(trial.get('raw_jsonl'))}.", "",
                      "Machine-readable validity verdict:", "",
                      *_json({key: (trial.get("validity_gate") or {}).get(key)
                              for key in ("status", "reason_codes", "eligibility")}), "",
                      "#### Timing distributions (milliseconds)", ""])
        distribution_rows = []
        for metric in ("completed_ttft_ms", "completed_tpot_ms", "completed_e2e_ms", "all_attempted_e2e_ms"):
            values = distributions.get(metric) or {}
            distribution_rows.append([metric, values.get("count"), values.get("mean"),
                values.get("median", values.get("p50")), values.get("sample_stddev"),
                values.get("min"), values.get("max"), values.get("p95"), values.get("p99")])
        lines.extend(_table(["Population / metric", "N", "Mean", "P50", "Sample SD", "Min", "Max", "P95", "P99"],
                            distribution_rows))
        lines.extend(["", "#### Outcome counts, rates and frozen SLO", "",
                      *_json({key: summary.get(key) for key in (
                          "outcome_counts", "rates", "slo_thresholds_ms", "slo_eligible_requests",
                          "token_accounting", "metric_validation_error_requests")}), "",
                      "#### Phase durations (seconds)", "", *_json(trial.get("phase_durations_s", {})), "",
                      "#### Resource observations", "", *_json(trial.get("resources", {})), "",
                      "Process-lifetime peaks include startup and warmup. Measured polling maxima are lower bounds,",
                      "not exact measurement-interval GPU/KV peaks. Missing legacy slot telemetry is N/A, not zero.", "",
                      "#### Server provenance", "", *_json(trial.get("server_metadata", {}))])
    lines.extend(["", "## Metric boundaries", "",
        "- TTFT is send-start to the first nonempty, tokenizer-valid client-visible content delta; it is not internal first-token latency.",
        "- Visible TPOT proxy = (last content arrival - first content arrival) / (retokenized visible output tokens - 1), for N > 1.",
        "  Role, finish and [DONE] events do not contribute to this numerator; chunks are never counted as tokens.",
        "- E2E ends at received [DONE] for a valid SSE stream, the complete JSON body, or observed failure/timeout/cancellation.",
        "- Formal rate window begins at replay origin and ends at the last terminal client observation, including arrival gaps.",
        "  Startup/model loading, warmup and post-request drain are timed separately and excluded from that window.",
        "- Actual completed generated tokens/s requires authoritative server usage. SSE without usage remains N/A.",
        "  Visible-text tokens/s is a separate retokenization proxy, never substituted for actual generation throughput.",
        "- All planned requests remain in failure-rate denominators. Successful and all-attempted latency populations are labelled separately.",
        "- SLO goodput counts only successful, length-valid requests meeting every applicable predeclared threshold.",
        "- Runtime internal per-request TTFT/TPOT/E2E: N/A; the current HTTP interface does not export those observations.", "",
        "## Frozen workload configuration", "", *_json(experiment.get("config", {})), "",
        "## Analysis and limitations", "",
        "Fresh server processes replay the same nominal trace in alternating AB/BA pair order (legacy/continuous, then continuous/legacy).",
        "Actual client send offsets and lateness are retained; identical server-ingress times are not claimed.",
        "An invalid trial or mismatched pair is retained for diagnosis, not silently removed or promoted into a performance conclusion.",
        "No cross-mode speedup is inferred by this report. CPU/Fake validation does not establish GPU correctness or GPU performance.",
        "Per-process GPU peaks include warmup; periodic device telemetry is not a continuous profiler and cannot prove causal explanations.",
        "Fixed prompts, model revisions and AB/BA order do not fix laptop clocks, power, temperature or competing GPU work.",
        "The current SSE API does not provide authoritative generated counts or internal token-arrival timestamps; these remain explicit gaps.",
        "No historical benchmark is overwritten or incorporated implicitly. See the protocol README for limits and reproduction commands.", "",
    ])
    return "\n".join(lines)
