"""Fixed arrival traces shared byte-for-byte by legacy and continuous modes."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from .config import ProtocolConfig, canonical_json


def build_trace(config: ProtocolConfig) -> list[dict[str, Any]]:
    """Copy the frozen arrival trace, never fabricate verified token lengths.

    The runner releases each request at its recorded arrival offset. Client
    concurrency may delay actual dispatch; scheduled and actual offsets must be
    recorded separately. Identical traces do not imply identical server
    admission times or identical execution schedules.
    """
    return config.as_dict()["requests"]


def trace_sha256(trace: Sequence[dict[str, Any]]) -> str:
    """Identify the exact shared contents, arrival offsets and input ordering."""
    return hashlib.sha256(canonical_json(list(trace)).encode("utf-8")).hexdigest()


def execution_order(pairs: int) -> list[tuple[int, str]]:
    """Return pair-indexed AB/BA order: LC, CL, LC, CL, ...

    Every pair runs both modes using the same trace. Each mode is a separate
    server process/session in the runner, rather than a mode switched while a
    model remains loaded. Pair index is zero based in raw records.
    """
    if type(pairs) is not int or not 1 <= pairs <= 20:
        raise ValueError("pairs must be an integer in [1, 20]")
    return [(pair, mode) for pair in range(pairs)
            for mode in (("legacy", "continuous") if pair % 2 == 0
                         else ("continuous", "legacy"))]
