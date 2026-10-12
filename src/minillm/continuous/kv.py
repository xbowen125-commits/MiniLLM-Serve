"""Request-local cache reference mapping for offline logical batches."""

from __future__ import annotations

from typing import Any, Sequence


class BatchKVAdapter:
    """Own one distinct cache reference per request and preserve batch order.

    Caches remain opaque objects: this adapter neither stacks tensors nor changes
    their contents. It rejects reuse of the same cache object across requests.
    Callers must supply independently allocated cache internals; hidden shared
    tensor storage cannot be detected by this model-independent mapping.

    release_request drops only this adapter's reference. External batch lists,
    contexts, and workers must release their own references separately. No forced
    GPU cleanup or empty_cache is performed. Calls are synchronous; a future
    runtime coordinator must serialize mutations with model execution.
    """

    def __init__(self) -> None:
        self._caches: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._caches)

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")

    def _require_request(self, request_id: str) -> None:
        self._validate_request_id(request_id)
        if request_id not in self._caches:
            raise KeyError(f"unknown cache request_id: {request_id}")

    def _validate_cache(self, request_id: str, cache: Any) -> None:
        if cache is None:
            raise ValueError("cache must not be None")
        if any(other_id != request_id and other is cache
               for other_id, other in self._caches.items()):
            raise ValueError("the same cache object cannot belong to multiple requests")

    def add_request_cache(self, request_id: str, cache: Any) -> None:
        """Register a new request; existing IDs cannot be silently overwritten."""
        self._validate_request_id(request_id)
        if request_id in self._caches:
            raise ValueError(f"cache already registered: {request_id}")
        self._validate_cache(request_id, cache)
        self._caches[request_id] = cache

    def get_batch_kv(self, request_ids: Sequence[str]) -> list[Any]:
        """Return original cache objects in the supplied order, without merging."""
        ids = tuple(request_ids)
        for request_id in ids:
            self._require_request(request_id)
        if len(set(ids)) != len(ids):
            raise ValueError("batch request_ids must be unique")
        return [self._caches[request_id] for request_id in ids]

    def update_request_cache(self, request_id: str, cache: Any) -> None:
        """Replace one request's cache, or retain its in-place mutated object."""
        self._require_request(request_id)
        self._validate_cache(request_id, cache)
        self._caches[request_id] = cache

    def release_request(self, request_id: str) -> None:
        """Remove a completed/failed/cancelled request's owned cache reference."""
        self._require_request(request_id)
        del self._caches[request_id]
