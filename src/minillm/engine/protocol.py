"""Structural worker contract consumed by the token generation loop."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WorkerProtocol(Protocol):
    """Narrow execution interface implemented by local or future workers."""

    device: str

    def forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        position_ids: Any,
        past_key_values: Any,
        use_cache: bool,
    ) -> Any:
        ...
