"""Compare maximum reservation with incremental paging; no model execution."""

from dataclasses import asdict
from datetime import datetime, timezone

from minillm.memory import PagedKVCacheManager
from .scenario import MemoryScenario


def _metrics(requested, allocated, blocks):
    return {"requested_tokens": requested, "allocated_tokens": allocated,
            "wasted_tokens": allocated - requested, "block_count": blocks,
            "utilization": requested / allocated if allocated else 0.0}


def run_experiment(scenario: MemoryScenario) -> dict:
    """Measure capacity with every request resident at its final length.

    Contiguous uses one synthetic allocation per request reserving max_tokens.
    Paged uses the existing manager with one simulated byte per token. Capacity
    covers both strategies, so this experiment does not measure admission/OOM,
    allocator fragmentation, elapsed time, or actual GPU bytes.
    """
    lengths = scenario.generate_lengths()
    page = scenario.tokens_per_block
    capacity = scenario.request_count * max(scenario.max_tokens,
                    ((scenario.max_tokens + page - 1) // page) * page)
    manager = PagedKVCacheManager(capacity, tokens_per_block=page, bytes_per_token=1)
    rows = []
    try:
        for index, length in enumerate(lengths):
            request_id = f"request_{index}"
            manager.allocate_request(request_id)
            remaining = length
            while remaining:
                added = min(remaining, scenario.growth_step)
                manager.append_tokens(request_id, added)
                remaining -= added
            blocks = len(manager.get_block_table(request_id))
            rows.append({"request_id": request_id,
                         "contiguous": _metrics(length, scenario.max_tokens, 1),
                         "paged": _metrics(length, blocks * page, blocks)})
        snapshot = manager.snapshot()
        requested = sum(snapshot.request_tokens.values())
        results = {
            "contiguous": _metrics(requested, len(lengths) * scenario.max_tokens, len(lengths)),
            "paged": _metrics(requested, snapshot.memory.allocated_bytes,
                               snapshot.memory.allocated_blocks),
        }
    finally:
        for index in range(len(lengths)):
            manager.release_request(f"request_{index}")
    return {"schema_version": 1, "scope": "simulated-kv-capacity",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(scenario), "request_lengths": list(lengths),
            "metrics": results, "requests": rows,
            "cleanup_allocated_tokens": manager.snapshot().memory.allocated_bytes}
