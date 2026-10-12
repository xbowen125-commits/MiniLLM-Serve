"""Bounded FIFO queue used by the in-process request scheduler."""

from __future__ import annotations

from collections import deque

from .engine import RequestState, RequestStatus


class RequestQueue:
    """A small bounded FIFO container for waiting request state.

    Synchronization intentionally remains the scheduler's responsibility so
    queue ordering and admission state can be updated under the same lock.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self._items: deque[RequestState] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    @property
    def is_full(self) -> bool:
        return len(self._items) >= self.capacity

    def put(self, item: RequestState) -> bool:
        """Append a waiting request, returning ``False`` when full."""
        if item.status is not RequestStatus.WAITING:
            raise ValueError("only waiting requests can enter the request queue")
        if self.is_full:
            return False
        self._items.append(item)
        return True

    def pop(self) -> RequestState:
        """Remove and return the oldest item."""
        return self._items.popleft()

    def remove(self, item: RequestState) -> bool:
        """Remove one matching item while preserving the remaining order."""
        try:
            self._items.remove(item)
        except ValueError:
            return False
        return True

    def get_batch(self, max_size: int) -> list[RequestState]:
        """Remove and return up to ``max_size`` waiting requests in FIFO order."""
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        batch: list[RequestState] = []
        while self._items and len(batch) < max_size:
            request = self._items.popleft()
            if request.status is RequestStatus.WAITING:
                batch.append(request)
        return batch

    def drain(self) -> list[RequestState]:
        """Remove all items and return them in FIFO order."""
        if not self._items:
            return []
        return self.get_batch(len(self._items))
