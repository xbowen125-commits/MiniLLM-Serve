"""Deterministic simulated capacity accounting, not GPU performance."""

import json
import pytest

from benchmarks.runtime.paged_memory.scenario import MemoryScenario
from benchmarks.runtime.paged_memory.experiment import run_experiment
from benchmarks.runtime.paged_memory.report import generate_report, save_report


def test_seeded_workload():
    scenario = MemoryScenario()
    assert scenario.generate_lengths() == scenario.generate_lengths()
    assert len(scenario.generate_lengths()) == 30
    assert set(scenario.generate_lengths()) <= set(scenario.lengths)


def test_exact_capacity_and_tail_waste():
    result = run_experiment(MemoryScenario(request_count=2, lengths=(17,), max_tokens=64))
    assert result["metrics"]["contiguous"] == dict(requested_tokens=34, allocated_tokens=128,
        wasted_tokens=94, block_count=2, utilization=34 / 128)
    assert result["metrics"]["paged"] == dict(requested_tokens=34, allocated_tokens=64,
        wasted_tokens=30, block_count=4, utilization=34 / 64)
    assert result["cleanup_allocated_tokens"] == 0


def test_reproducibility_and_growth_step_independence():
    first = run_experiment(MemoryScenario(growth_step=1))
    second = run_experiment(MemoryScenario(growth_step=31))
    assert first["request_lengths"] == second["request_lengths"]
    assert first["metrics"] == second["metrics"]
    for policy in ("contiguous", "paged"):
        assert first["metrics"][policy]["allocated_tokens"] == sum(
            row[policy]["allocated_tokens"] for row in first["requests"])


def test_paging_is_not_assumed_better():
    result = run_experiment(MemoryScenario(request_count=1, lengths=(17,), max_tokens=17))
    assert result["metrics"]["paged"]["wasted_tokens"] > result["metrics"]["contiguous"]["wasted_tokens"]


def test_json_and_markdown_export(tmp_path):
    result = run_experiment(MemoryScenario())
    json_path, markdown_path = save_report(result, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"] == result["metrics"]
    assert markdown_path.read_text(encoding="utf-8") == generate_report(result)
    assert "no model, CUDA, or GPU measurement" in generate_report(result)


@pytest.mark.parametrize("kwargs", [{"request_count": 0}, {"lengths": ()},
    {"lengths": (1025,)}, {"tokens_per_block": 0}, {"growth_step": -1}, {"seed": True}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MemoryScenario(**kwargs)
