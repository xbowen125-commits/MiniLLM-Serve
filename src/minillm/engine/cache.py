"""Request-local KV Cache creation and lifecycle management."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


@dataclass
class RequestKVCache:
    """Mutable cache handle owned by exactly one generation request."""

    enabled: bool
    value: Any = None

    def update(self, value: Any) -> None:
        if self.enabled:
            self.value = value

    def release(self) -> None:
        self.value = None


class KVCacheManager:
    """Creates a fresh Transformers cache and releases it on every exit path."""

    def create(self, model_config: Any) -> Any:
        from transformers import DynamicCache

        return DynamicCache(config=model_config)

    @contextmanager
    def request_cache(
        self,
        *,
        enabled: bool,
        factory: Callable[[], Any],
    ) -> Iterator[RequestKVCache]:
        cache = RequestKVCache(enabled=enabled, value=factory() if enabled else None)
        try:
            yield cache
        finally:
            cache.release()
