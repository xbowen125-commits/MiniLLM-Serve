"""Logical-to-physical block ID mapping for simulated request KV sequences."""

import threading
from types import MappingProxyType
from typing import Mapping


class BlockTable:
    """Map zero-based request-local logical IDs to opaque physical IDs.

    append_block returns the next logical ID. get_blocks returns physical IDs
    in logical order. Each physical ID has one request owner; sharing/COW is not
    supported. This table does not allocate memory or verify manager ownership.
    Removing a request removes mappings only, not KVMemoryManager allocations.
    """

    def __init__(self) -> None:
        self._requests: dict[str, list[int]] = {}
        self._owners: dict[int, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _validate_id(request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")

    def append_block(self, request_id: str, physical_block_id: int) -> int:
        """Append an independently owned block and return its logical index."""
        self._validate_id(request_id)
        if type(physical_block_id) is not int or physical_block_id < 0:
            raise ValueError("physical_block_id must be a non-negative integer")
        with self._lock:
            if physical_block_id in self._owners:
                raise ValueError("physical block already mapped")
            blocks = self._requests.setdefault(request_id, [])
            logical_block_id = len(blocks)
            blocks.append(physical_block_id)
            self._owners[physical_block_id] = request_id
            return logical_block_id

    def get_blocks(self, request_id: str) -> tuple[int, ...]:
        """Return physical IDs ordered by logical block index."""
        self._validate_id(request_id)
        with self._lock:
            if request_id not in self._requests:
                raise KeyError(f"unknown block-table request_id: {request_id}")
            return tuple(self._requests[request_id])

    def remove_request(self, request_id: str) -> None:
        self._validate_id(request_id)
        with self._lock:
            if request_id not in self._requests:
                raise KeyError(f"unknown block-table request_id: {request_id}")
            for physical_id in self._requests.pop(request_id):
                del self._owners[physical_id]

    def snapshot(self) -> Mapping[str, Mapping[int, int]]:
        """Return a detached immutable request → logical ID → physical ID view."""
        with self._lock:
            return MappingProxyType({
                request_id: MappingProxyType(dict(enumerate(blocks)))
                for request_id, blocks in self._requests.items()
            })
