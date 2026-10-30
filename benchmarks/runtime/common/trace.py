"""Prompt-free per-iteration JSONL export with explicit virtual timestamps."""

import json
from pathlib import Path


def export_trace(iterations, path):
    """One record per iteration; finished IDs are actual retirements this cycle.

    Scheduler selection-time terminal observations remain separately labelled.
    timestamp is virtual iteration start in seconds, not UTC wall-clock time.
    """
    lines = []
    for iteration in iterations:
        record = {**iteration, "timestamp": iteration["start_s"], "timestamp_unit": "virtual_seconds",
                  "scheduler_finished_request_ids": iteration["finished_request_ids"],
                  "finished_request_ids": iteration["completed_this_iteration"]}
        lines.append(json.dumps(record, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
