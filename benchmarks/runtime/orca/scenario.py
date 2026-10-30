"""Heterogeneous request lengths for offline iteration scheduling research."""

from dataclasses import dataclass

from ..continuous_batch.scenario import Scenario, SimRequest
from ..common.arrival import WorkloadRequest, normalize_requests


@dataclass(frozen=True)
class OrcaRequest(SimRequest):
    """Synthetic prompt length and output budget; no actual tokenization."""

    prompt_length: int = 32

    def __post_init__(self):
        super().__post_init__()
        if type(self.prompt_length) is not int or not 1 <= self.prompt_length <= 4096:
            raise ValueError("prompt_length must be in 1..4096")

    @property
    def generation_length(self) -> int:
        return self.max_tokens


def workload_metadata(scenario: Scenario) -> list[dict]:
    """Stable arrival rank, with original input order breaking timestamp ties.

    Legacy SimRequest remains supported: absent prompt length is reported as
    None, not inferred from the placeholder string sent to the Fake Worker.
    """
    ordered = normalize_requests(scenario.requests)
    return [{"request_id": r.request_id, "arrival_order": index,
             "arrival_time_s": r.arrival_time,
             "prompt_length": getattr(r, "prompt_length", None),
             "generation_length": r.max_tokens} for index, r in enumerate(ordered)]


def heterogeneous_workload(*, long_tokens: int = 32, short_tokens: int = 2,
                           short_count: int = 6, arrival_interval: float = 0.04,
                           max_batch_size: int = 4) -> Scenario:
    """One long request followed by bounded short-request arrivals."""
    if type(short_count) is not int or not 1 <= short_count <= 999:
        raise ValueError("short_count must be in 1..999")
    return Scenario("heterogeneous_arrival", (
        WorkloadRequest("long", 256, long_tokens, 0.0),
        *(WorkloadRequest(f"short_{i}", 32, short_tokens, (i + 1) * arrival_interval)
          for i in range(short_count)),
    ), max_active=max_batch_size)


def scenarios() -> tuple[Scenario, ...]:
    return (heterogeneous_workload(arrival_interval=0), heterogeneous_workload(),
            heterogeneous_workload(long_tokens=64, short_tokens=4, arrival_interval=0.08),
            Scenario("mixed_length", tuple(OrcaRequest(f"mixed_{i}", 0, output, prompt)
                for i, (prompt, output) in enumerate(((16, 2), (256, 32), (64, 8),
                    (128, 16), (32, 4), (512, 64), (48, 6), (96, 12))))))
