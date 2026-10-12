"""Allocation metadata for simulated KV memory, without tensor storage."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KVBlock:
    """Manager-owned block metadata; users cannot edit size or allocation state.

    A retained handle observes allocated=False after its manager releases it.
    block_id is unique within a manager, and IDs are not reused on release.
    """

    block_id: int
    request_id: str
    size_bytes: int
    allocated: bool = True

    def _mark_released(self) -> None:
        """Internal manager operation; no physical memory is freed."""
        object.__setattr__(self, "allocated", False)
