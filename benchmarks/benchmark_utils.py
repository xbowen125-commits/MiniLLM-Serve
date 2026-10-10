"""Shared statistics and report writers for Mission 08 benchmarks."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Linear-interpolated percentile (same definition for every report)."""
    if not values:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values if value is not None]
    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples) if samples else None,
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "min": min(samples) if samples else None,
        "max": max(samples) if samples else None,
    }


def runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        metadata["torch"] = None
    try:
        import transformers

        metadata["transformers"] = transformers.__version__
    except Exception:
        metadata["transformers"] = None
    return metadata


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def markdown_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No results._"

    def render(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(row.get(column)) for column in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def write_markdown(
    path: Path,
    *,
    title: str,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    columns: Sequence[str],
    notes: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_lines = "\n".join(
        f"- {key}: `{value if value is not None else 'N/A'}`" for key, value in metadata.items()
    )
    note_lines = "\n".join(f"- {note}" for note in notes)
    content = (
        f"# {title}\n\n"
        f"## Environment\n\n{metadata_lines}\n\n"
        f"## Results\n\n{markdown_table(rows, columns)}\n\n"
        f"## Interpretation notes\n\n{note_lines}\n"
    )
    path.write_text(content, encoding="utf-8")
