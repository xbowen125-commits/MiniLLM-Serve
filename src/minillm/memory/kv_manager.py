"""Bounded, thread-safe Python bookkeeping for simulated KV allocations."""

import threading

from .block import KVBlock
from .metrics import KVMemorySnapshot


class KVCapacityExceeded(MemoryError):
    """A simulated allocation exceeds the manager's available capacity."""


class KVMemoryManager:
    """Track variable-size logical blocks, not GPU pages or tensor allocations.

    Each allocate call creates one block. A request may own multiple blocks.
    release drops all its blocks and is idempotent for already-released/unknown
    IDs. get_blocks returns only active blocks in allocation order. No retired
    block history is retained internally; aggregate counters survive release.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("capacity_bytes must be a non-negative integer")
        self.capacity_bytes = capacity_bytes
        self._requests: dict[str, list[KVBlock]] = {}
        self._next_id = 1
        self._allocated = 0
        self._peak = 0
        self._allocation_count = 0
        self._released_blocks = 0
        self._released_bytes = 0
        self._rejected = 0
        self._lock = threading.Lock()

    @staticmethod
    def _validate_id(request_id):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")

    def allocate(self, request_id: str, size_bytes: int) -> KVBlock:
        """Atomically allocate one logical block or reject without partial changes."""
        return self.allocate_blocks(request_id, size_bytes, 1)[0]

    def allocate_blocks(self, request_id: str, size_bytes: int, count: int) -> tuple[KVBlock, ...]:
        """Reserve multiple equally sized blocks atomically for a store growth."""
        self._validate_id(request_id)
        if type(size_bytes) is not int or size_bytes <= 0:
            raise ValueError("size_bytes must be a positive integer")
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        with self._lock:
            total_bytes = size_bytes * count
            if total_bytes > self.capacity_bytes - self._allocated:
                self._rejected += 1
                raise KVCapacityExceeded("simulated KV memory capacity exceeded")
            blocks = tuple(KVBlock(self._next_id + index, request_id, size_bytes)
                           for index in range(count))
            self._requests.setdefault(request_id, []).extend(blocks)
            self._next_id += count
            self._allocated += total_bytes
            self._peak = max(self._peak, self._allocated)
            self._allocation_count += count
            return blocks

    def release(self, request_id: str) -> None:
        """Release every block owned by the request and restore logical capacity."""
        self._validate_id(request_id)
        with self._lock:
            blocks = self._requests.pop(request_id, ())
            released = sum(block.size_bytes for block in blocks)
            for block in blocks:
                block._mark_released()
            self._allocated -= released
            self._released_blocks += len(blocks)
            self._released_bytes += released

    def get_blocks(self, request_id: str) -> tuple[KVBlock, ...]:
        self._validate_id(request_id)
        with self._lock:
            return tuple(self._requests.get(request_id, ()))

    def snapshot(self) -> KVMemorySnapshot:
        with self._lock:
            return KVMemorySnapshot(
                self.capacity_bytes, self._allocated, self.capacity_bytes - self._allocated,
                self._peak, len(self._requests), sum(map(len, self._requests.values())),
                self._allocation_count, self._released_blocks, self._released_bytes, self._rejected,
            )
