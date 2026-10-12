"""Stable request selections for one offline continuous execution cycle."""

from dataclasses import dataclass

from ..runtime.context import RuntimeRequestContext


@dataclass(frozen=True, slots=True)
class ContinuousBatchPlan:
    """Decode selections and ordered prospective prefill admissions.

    Prefill candidates are subject to available capacity after decode retires
    requests. active_request_ids records the active set when this plan was built.
    Context references are shared with the batch, never copied.
    """

    prefill_requests: tuple[RuntimeRequestContext, ...]
    decode_requests: tuple[RuntimeRequestContext, ...]
    active_request_ids: tuple[str, ...]
