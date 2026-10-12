"""Simulated paged KV request interface over the existing capacity ledger."""

from types import MappingProxyType
from typing import Mapping

from .block import KVBlock
from .kv_store import KVCacheStore, KVCacheStoreSnapshot


class PagedKVCacheManager:
    """Grow request-local logical pages as tokens cross fixed block boundaries.

    Every page reserves tokens_per_block * bytes_per_token simulated bytes.
    KVCacheStore owns the real bookkeeping: its BlockTable maps logical page IDs
    to KVMemoryManager's synthetic block IDs. This wrapper creates no second
    mapping or token counter. It neither stores K/V tensors nor performs paged
    attention, CUDA allocation, page reuse, sharing, or copy-on-write.
    """

    def __init__(self, capacity_bytes: int, *, tokens_per_block: int = 16,
                 bytes_per_token: int = 128) -> None:
        self._store = KVCacheStore(capacity_bytes,
                                   tokens_per_block=tokens_per_block,
                                   bytes_per_token=bytes_per_token)

    def allocate_request(self, request_id: str) -> None:
        """Register an empty sequence; no simulated bytes are reserved yet."""
        self._store.allocate_request(request_id, 0)

    def append_tokens(self, request_id: str, token_count: int) -> tuple[KVBlock, ...]:
        """Grow pages atomically and return all current blocks in logical order."""
        return self._store.append_tokens(request_id, token_count)

    def get_block_table(self, request_id: str) -> Mapping[int, int]:
        """Return an immutable detached logical-page → physical-block ID view.

        An allocated empty request returns an empty mapping. Unknown/released
        requests raise KeyError. Store lookup also verifies table consistency.
        """
        blocks = self._store.get_request_blocks(request_id)
        return MappingProxyType({index: block.block_id for index, block in enumerate(blocks)})

    def release_request(self, request_id: str) -> None:
        """Remove token tracking, mappings, and capacity; repeated release is safe."""
        self._store.release_request(request_id)

    def snapshot(self) -> KVCacheStoreSnapshot:
        """Expose the store's consistent simulated capacity and mapping snapshot."""
        return self._store.snapshot()
