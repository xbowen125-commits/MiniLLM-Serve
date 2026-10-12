"""Immutable snapshots of simulated KV capacity and allocation counters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KVMemorySnapshot:
    capacity_bytes: int
    allocated_bytes: int
    free_bytes: int
    peak_allocated_bytes: int
    active_requests: int
    allocated_blocks: int
    allocation_count: int
    released_blocks: int
    released_bytes: int
    rejected_allocations: int
