"""Configuration, serialization, report rendering, and reproducibility tests."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from benchmarks.runtime.report_generator import ROOT, load_config, run_experiment, save_results, generate_report


@pytest.fixture
def artifact_dir():
    root = ROOT / "results"
    root.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="test-report-", dir=root) as directory:
        yield Path(directory)


def test_config_loading_supports_all_scenarios():
    for name in ("decode_flood", "mixed_length", "continuous_arrival"):
        assert load_config(ROOT / "configs" / f"{name}.json")["scenario"] == name


def test_json_serialization_and_report(artifact_dir):
    records = run_experiment(load_config(ROOT / "configs" / "mixed_length.json"))
    path = artifact_dir / "results" / "mixed.json"
    save_results(records, path)
    loaded = json.loads(path.read_text())
    assert len(loaded) == 2
    assert loaded[0]["timestamp"]
    assert loaded[0]["workload_config"]["scenario"] == "mixed_length"
    report = generate_report(loaded)
    for title in ("Experiment setup", "Runtime architecture", "Metrics definition",
                  "Scenario description", "Results summary", "Limitations"):
        assert title in report
    assert "FairPhaseBatchPolicy" in report
    assert "37.083333" in report


def test_reproducible_metrics_and_config_hash():
    config = load_config(ROOT / "configs" / "decode_flood.json")
    first = run_experiment(config)
    second = run_experiment(config)
    for a, b in zip(first, second):
        assert a["metrics"] == b["metrics"]
        assert a["config_sha256"] == b["config_sha256"]
    changed = json.loads(json.dumps(config))
    changed["aging_factor"] = 2
    assert run_experiment(changed)[0]["config_sha256"] != first[0]["config_sha256"]


def test_config_rejects_unbounded_workload(artifact_dir):
    config = load_config(ROOT / "configs" / "decode_flood.json")
    config["groups"][0]["count"] = 1001
    path = artifact_dir / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)
