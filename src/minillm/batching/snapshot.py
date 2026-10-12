"""Prompt-free observability snapshot for the offline decode runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from .state import DecodeState, DecodeStatus


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    active_requests: int
    finished_requests: int
    failed_requests: int
    cancelled_requests: int
    total_generated_tokens: int
    total_decode_steps: int
    snapshot_time: float

    @classmethod
    def from_states(
        cls,
        states: Iterable[DecodeState],
        *,
        snapshot_time: float | None = None,
    ) -> "RuntimeSnapshot":
        materialized = tuple(states)
        return cls(
            active_requests=sum(
                state.status is DecodeStatus.ACTIVE for state in materialized
            ),
            finished_requests=sum(
                state.status is DecodeStatus.FINISHED for state in materialized
            ),
            failed_requests=sum(
                state.status is DecodeStatus.FAILED for state in materialized
            ),
            cancelled_requests=sum(
                state.status is DecodeStatus.CANCELLED for state in materialized
            ),
            total_generated_tokens=sum(
                state.num_generated_tokens for state in materialized
            ),
            total_decode_steps=sum(state.decode_steps for state in materialized),
            snapshot_time=time.time() if snapshot_time is None else snapshot_time,
        )
