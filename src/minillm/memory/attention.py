"""Read-only logical block lookup; no attention arithmetic or model execution."""

from .paged_cache import PagedKVCacheManager


class PagedAttentionSimulator:
    """Resolve a request's simulated physical block IDs in logical order.

    The supplied cache manager remains the only mapping owner. Every lookup
    reads its current table, so growth and release are immediately reflected.
    Returned lists are detached; editing them cannot change cache mappings.
    This is address lookup simulation, not a PagedAttention implementation.
    """

    def __init__(self, cache_manager: PagedKVCacheManager) -> None:
        self._cache_manager = cache_manager

    def get_attention_blocks(self, request_id: str) -> list[int]:
        """Return ordered IDs, [] for an empty request, or KeyError if unknown."""
        table = self._cache_manager.get_block_table(request_id)
        return [table[logical_id] for logical_id in sorted(table)]
