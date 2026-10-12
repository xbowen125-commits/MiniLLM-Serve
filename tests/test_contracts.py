"""Shared contracts retain identity and resolve types without runtime imports."""

import ast
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

from minillm.contracts import DecodeStatus, ExecutionPhase, FinishReason, DecodeStepResult
from minillm.execution.phase import ExecutionPhase as LegacyPhase
from minillm.execution.result import DecodeStepResult as LegacyResult
from minillm.batching.state import DecodeStatus as LegacyStatus, FinishReason as LegacyReason
from minillm.execution.executor import PrefillExecutor


def test_compatibility_exports_preserve_object_identity():
    assert LegacyPhase is ExecutionPhase
    assert LegacyResult is DecodeStepResult
    assert LegacyStatus is DecodeStatus
    assert LegacyReason is FinishReason


def test_contract_and_executor_type_hints_resolve():
    assert get_type_hints(DecodeStepResult)["finish_reason"] == FinishReason | None
    assert "states" in get_type_hints(PrefillExecutor.execute)


def test_contract_import_does_not_load_implementation_layers():
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys; import minillm.contracts; "
         "assert not any(n.startswith(('minillm.batching', 'minillm.execution', 'minillm.engine')) for n in sys.modules)"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_contract_or_execution_imports_of_batching():
    root = Path(__file__).resolve().parents[1] / "src" / "minillm"
    for package in ("contracts", "execution"):
        for path in (root / package).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert "batching" not in module.split("."), path
                    if package == "contracts":
                        assert "execution" not in module.split("."), path
