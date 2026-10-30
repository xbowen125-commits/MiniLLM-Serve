"""Rendering tests use synthetic rows, never real GPU performance samples."""

from copy import deepcopy

from benchmarks.runtime.online_http.metrics import summarize
from benchmarks.runtime.online_http.report import render_report


def _experiment():
    summary = summarize([
        {"request_id": "completed", "sent_at_s": 0.0, "first_content_at_s": .2,
         "last_content_at_s": .4, "terminal_at_s": .7,
         "visible_output_tokens": 3, "generated_output_tokens": None,
         "outcome": "completed", "success": True},
        {"request_id": "failed", "sent_at_s": .1, "terminal_at_s": .5,
         "outcome": "error", "success": False},
    ], window_s=.7, slo={"ttft_ms": 1000, "tpot_ms": 1000, "e2e_ms": 1000})
    return {
        "evidence_kind": "cpu_fake", "config_sha256": "a" * 64,
        "trace_sha256": "b" * 64,
        "git": {"sha": "c" * 40, "dirty": True, "source_tree_sha256": "d" * 64},
        "config": {"slo": {"ttft_ms": 1000, "tpot_ms": 1000, "e2e_ms": 1000}},
        "planned_order": [[0, "legacy"], [0, "continuous"]],
        "trials": [{
            "trial_id": "trial-000-legacy", "pair_index": 0, "mode": "legacy",
            "summary": summary, "comparison_valid": False,
            "issues": ["injected failure is retained", "arrival lateness | budget exceeded"],
            "phase_durations_s": {"startup": .1, "warmup": .2, "measured": .7,
                                  "drain": .1, "shutdown": .1},
            "resources": {"before": {}, "after": {"running_requests": 0},
                "peak_kv_slots_process_lifetime": None,
                "gpu_peak_allocated_mb_process_lifetime": None,
                "measured_sample_peak_kv_slots": None,
                "measured_sample_peak_gpu_allocated_mb": None},
            "server_metadata": {"pid": 1234, "device": "cpu", "evidence_kind": "cpu_fake"},
            "raw_jsonl": "trial-000-legacy/requests.jsonl",
        }],
    }


def test_report_retains_invalid_trial_and_failures_without_gpu_claims():
    experiment = _experiment()
    original = deepcopy(experiment)
    text = render_report(experiment)
    assert "CPU/Fake HTTP protocol validation only" in text
    assert "not real model or GPU performance evidence" in text
    assert "Comparison valid: **false**" in text
    assert "injected failure is retained" in text
    assert "arrival lateness \\| budget exceeded" in text
    assert '"error": 1' in text
    assert "| trial-000-legacy | 0 | legacy | false | 2 | 1 | 1 | N/A |" in text
    assert experiment == original


def test_report_preserves_missing_counts_and_distinct_metric_scopes():
    text = render_report(_experiment())
    assert "Actual generated tokens/s" in text
    assert "Visible text tokens/s proxy" in text
    assert "completed_ttft_ms" in text
    assert "all_attempted_e2e_ms" in text
    assert "Sample SD" in text
    assert "P95" in text and "P99" in text
    assert "Runtime internal per-request TTFT/TPOT/E2E: N/A" in text
    assert "SSE without usage remains N/A" in text
    assert "Process-lifetime peaks include startup and warmup" in text
    assert "Missing legacy slot telemetry is N/A, not zero" in text


def test_report_records_full_configuration_provenance_and_raw_link():
    experiment = _experiment()
    text = render_report(experiment)
    assert experiment["config_sha256"] in text
    assert experiment["trace_sha256"] in text
    assert experiment["git"]["sha"] in text
    assert experiment["git"]["source_tree_sha256"] in text
    assert '"dirty": true' in text
    assert '"e2e_ms": 1000' in text
    assert "[trial-000-legacy/requests.jsonl](trial-000-legacy/requests.jsonl)" in text
    assert "#### Phase durations (seconds)" in text
    assert '"measured": 0.7' in text


def test_report_never_filters_issues_or_selects_best_trials():
    experiment = _experiment()
    second = deepcopy(experiment["trials"][0])
    second.update(trial_id="trial-001-continuous", mode="continuous",
                  comparison_valid=True, issues=[])
    experiment["trials"].append(second)
    text = render_report(experiment)
    assert "### Trial trial-000-legacy: legacy" in text
    assert "### Trial trial-001-continuous: continuous" in text
    assert "Comparison valid: **false**" in text and "Comparison valid: **true**" in text
    assert "No cross-mode speedup is inferred" in text


def test_report_without_execution_does_not_invent_trial_results():
    experiment = _experiment()
    experiment["trials"] = []
    text = render_report(experiment)
    assert "No executed trial records are present" in text
    assert "no experiment result is claimed" in text
    assert "### Trial" not in text


def test_report_avoids_absolute_local_artifact_links():
    experiment = _experiment()
    experiment["trials"][0]["raw_jsonl"] = "C:/private-user/experiment/requests.jsonl"
    text = render_report(experiment)
    assert "private-user" not in text
    assert "`requests.jsonl` (local artifact)" in text


def test_gpu_label_alone_does_not_become_successful_gpu_execution_claim():
    experiment = _experiment()
    experiment.update(evidence_kind="real_gpu", trials=[])
    text = render_report(experiment)
    assert "configuration label alone is not proof" in text
    assert "No executed trial records are present" in text


def test_report_exposes_aborted_run_unrun_trials_and_final_pair_verdict():
    experiment = _experiment()
    experiment.update(
        aborted="previous server may still own resources; no further process launched",
        source_unchanged=False,
        pair_validation=[{"pair_index": 0, "valid": False, "issues": ["pair incomplete"]}],
    )
    text = render_report(experiment)
    assert "| Planned trials | 2 |" in text
    assert "| Executed trial records | 1 |" in text
    assert "| Unrun trials | 1 |" in text
    assert "previous server may still own resources" in text
    assert "Source unchanged during experiment: **false**" in text
    assert "pair incomplete" in text
    assert "Unrun planned entries" in text
    assert "final-verdict files are authoritative" in text
    assert "trial_local_only validity scope" in text
    assert "All configured trials appear below" not in text


def test_report_complete_trial_accounting_does_not_report_false_unrun_entries():
    experiment = _experiment()
    other = deepcopy(experiment["trials"][0])
    other.update(trial_id="trial-001-continuous", mode="continuous")
    experiment["trials"].append(other)
    experiment["source_unchanged"] = True
    experiment["pair_validation"] = [{"pair_index": 0, "valid": False, "issues": []}]
    text = render_report(experiment)
    assert "| Executed trial records | 2 |" in text
    assert "| Unrun trials | 0 |" in text
    assert "Source unchanged during experiment: **true**" in text
    assert "Unrun planned entries (pair index, mode):\n\n```json\n[]\n```" in text


def test_missing_plan_or_source_verdict_is_unknown_not_a_clean_run():
    experiment = _experiment()
    experiment.pop("planned_order")
    text = render_report(experiment)
    assert "| Planned trials | N/A |" in text
    assert "| Unrun trials | N/A |" in text
    assert "Source unchanged during experiment: **N/A**" in text
