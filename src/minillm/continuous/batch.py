"""Persistent context groups for an offline continuous decode simulation."""

from dataclasses import dataclass, field

from ..runtime.context import RuntimeRequestContext


@dataclass
class ContinuousBatch:
    """Retain the same context object as it moves between runtime groups."""

    active_requests: dict[str, RuntimeRequestContext] = field(default_factory=dict)
    waiting_requests: dict[str, RuntimeRequestContext] = field(default_factory=dict)
    finished_requests: dict[str, RuntimeRequestContext] = field(default_factory=dict)

    def add(self, context: RuntimeRequestContext) -> None:
        request_id = context.request_id
        if any(request_id in group for group in (
            self.active_requests, self.waiting_requests, self.finished_requests,
        )):
            raise ValueError("duplicate request_id")
        if context.request.is_terminal:
            raise ValueError("terminal request cannot enter a continuous batch")
        self.waiting_requests[request_id] = context

    def retire(self) -> tuple[str, ...]:
        """Move terminal contexts to the finished group, regardless of outcome."""
        retired = []
        for group in (self.active_requests, self.waiting_requests):
            for request_id, context in tuple(group.items()):
                if context.request.is_terminal:
                    self.finished_requests[request_id] = group.pop(request_id)
                    retired.append(request_id)
        return tuple(retired)
