"""Immutable observation of one offline iteration scheduling decision."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IterationState:
    """Selected IDs and newly observed terminal IDs, not execution outcomes.

    Iteration IDs start at one. FINISHED IDs include failed/cancelled requests
    because all terminal outcomes leave the active set. No prompts are retained.
    """

    iteration_id: int
    decode_request_ids: tuple[str, ...] = ()
    prefill_request_ids: tuple[str, ...] = ()
    finished_request_ids: tuple[str, ...] = ()
