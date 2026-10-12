"""State container for a group of independently owned requests."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..engine import RequestState, RequestStatus

if TYPE_CHECKING:
    from .result import BatchStepResult


@dataclass
class BatchState:
    """Track active and completed requests for one logical batch."""

    batch_id: str = field(default_factory=lambda: f"batch_{uuid.uuid4().hex}")
    requests: list[RequestState] = field(default_factory=list)
    completed_requests: list[RequestState] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)
    started_time: float | None = None
    finished_time: float | None = None
    step_count: int = 0
    step_results: list[BatchStepResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def is_finished(self) -> bool:
        return self.finished_time is not None

    def add_request(self, request: RequestState) -> None:
        """Add one non-terminal request, rejecting duplicate request IDs."""
        if request.is_terminal:
            raise ValueError("terminal requests cannot enter a batch")
        if any(existing.request_id == request.request_id for existing in self.all_requests):
            raise ValueError(f"duplicate request_id in batch: {request.request_id}")
        if self.is_finished:
            raise RuntimeError("cannot add requests to a finished batch")
        self.requests.append(request)

    @property
    def all_requests(self) -> tuple[RequestState, ...]:
        return tuple(self.requests + self.completed_requests)

    def start(self, *, now: float | None = None) -> None:
        """Start every waiting request while preserving already-running ones."""
        if self.started_time is not None:
            return
        timestamp = time.time() if now is None else now
        self.started_time = timestamp
        for request in self.requests:
            if request.status is RequestStatus.WAITING:
                request.mark_running(now=timestamp)

    def remove_completed(self) -> list[RequestState]:
        """Move terminal requests out of the active request list."""
        removed = [request for request in self.requests if request.is_terminal]
        if not removed:
            return []
        self.requests = [request for request in self.requests if not request.is_terminal]
        self.completed_requests.extend(removed)
        return removed

    def finish_if_idle(self, *, now: float | None = None) -> bool:
        """Finish a started batch once it has no active requests."""
        if self.started_time is None or self.requests or self.is_finished:
            return False
        self.finished_time = time.time() if now is None else now
        return True
