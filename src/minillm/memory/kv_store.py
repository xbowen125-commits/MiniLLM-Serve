"""Coordinated simulated token capacity, block allocation, and block mappings."""

from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Mapping

from .block import KVBlock
from .block_table import BlockTable
from .kv_manager import KVMemoryManager
from .metrics import KVMemorySnapshot


@dataclass(frozen=True, slots=True)
class KVCacheStoreSnapshot:
    memory: KVMemorySnapshot
    block_table: Mapping[str, Mapping[int, int]]
    request_tokens: Mapping[str, int]
    tokens_per_block: int
    bytes_per_token: int


class KVCacheStore:
    """Own the manager and table as a single simulated capacity ledger.

    Block count is ceil(tokens / tokens_per_block); every allocated block reserves
    tokens_per_block * bytes_per_token bytes, including unused tail space.
    Token counts are logical occupancy only, not tensors or attention pages.
    The underlying objects are private so callers cannot mutate only one ledger.
    """

    def __init__(self, capacity_bytes: int, *, tokens_per_block: int = 16,
                 bytes_per_token: int = 128):
        for value in (tokens_per_block, bytes_per_token):
            if type(value) is not int or value < 1:
                raise ValueError("block/token sizes must be positive integers")
        self.tokens_per_block = tokens_per_block
        self.bytes_per_token = bytes_per_token
        self._memory = KVMemoryManager(capacity_bytes)
        self._table = BlockTable()
        self._tokens: dict[str, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _validate(request_id, tokens):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if type(tokens) is not int or tokens < 0:
            raise ValueError("token count must be a non-negative integer")

    def _block_count(self, tokens):
        return (tokens + self.tokens_per_block - 1) // self.tokens_per_block

    def _grow(self, request_id, old_tokens, new_tokens):
        count = self._block_count(new_tokens) - self._block_count(old_tokens)
        if count:
            blocks = self._memory.allocate_blocks(
                request_id, self.tokens_per_block * self.bytes_per_token, count,
            )
            for block in blocks:
                self._table.append_block(request_id, block.block_id)
        # Commit logical occupancy only after capacity was reserved successfully.
        self._tokens[request_id] = new_tokens

    def allocate_request(self, request_id: str, num_tokens: int) -> tuple[KVBlock, ...]:
        self._validate(request_id, num_tokens)
        with self._lock:
            if request_id in self._tokens:
                raise ValueError("request already allocated")
            self._grow(request_id, 0, num_tokens)
            return self._memory.get_blocks(request_id)

    def append_tokens(self, request_id: str, token_count: int) -> tuple[KVBlock, ...]:
        self._validate(request_id, token_count)
        with self._lock:
            if request_id not in self._tokens:
                raise KeyError(f"unknown KV store request_id: {request_id}")
            old = self._tokens[request_id]
            self._grow(request_id, old, old + token_count)
            return self._memory.get_blocks(request_id)

    def get_request_blocks(self, request_id: str) -> tuple[KVBlock, ...]:
        self._validate(request_id, 0)
        with self._lock:
            if request_id not in self._tokens:
                raise KeyError(f"unknown KV store request_id: {request_id}")
            blocks = self._memory.get_blocks(request_id)
            expected = self._table.get_blocks(request_id) if blocks else ()
            if tuple(b.block_id for b in blocks) != expected:
                raise RuntimeError("KV allocation and block table are inconsistent")
            return blocks

    def release_request(self, request_id: str) -> None:
        """Release mappings and capacity together; repeated release is harmless."""
        self._validate(request_id, 0)
        with self._lock:
            if request_id not in self._tokens:
                return
            if self._block_count(self._tokens[request_id]):
                self._table.remove_request(request_id)
            self._memory.release(request_id)
            del self._tokens[request_id]

    def snapshot(self) -> KVCacheStoreSnapshot:
        with self._lock:
            return KVCacheStoreSnapshot(
                self._memory.snapshot(), self._table.snapshot(),
                MappingProxyType(dict(self._tokens)), self.tokens_per_block,
                self.bytes_per_token,
            )
