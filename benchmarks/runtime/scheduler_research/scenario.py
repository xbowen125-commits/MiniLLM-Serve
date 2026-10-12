"""Bounded mixed-phase admission workloads with synthetic service costs."""

from dataclasses import dataclass
import math

from minillm.contracts import ExecutionPhase
from ..scheduler_compare import SimulatedRequest


@dataclass(frozen=True)
class ResearchScenario:
    name: str
    requests: tuple[SimulatedRequest, ...]
    start_time: float = 0.0
    aging_factor: float = 5.0
    starvation_threshold: float = 10.0
    first_token_fraction: float = 0.1

    def __post_init__(self):
        if not self.name or not 1 <= len(self.requests) <= 1000:
            raise ValueError("scenario needs 1..1000 requests")
        if len({r.request_id for r in self.requests}) != len(self.requests):
            raise ValueError("duplicate request IDs")
        for value in (self.start_time, self.aging_factor, self.starvation_threshold):
            if not math.isfinite(value) or value < 0:
                raise ValueError("timing/aging values must be finite and nonnegative")
        if not math.isfinite(self.first_token_fraction) or not 0 < self.first_token_fraction <= 1:
            raise ValueError("first_token_fraction must be in (0, 1]")
        if any(r.phase not in (ExecutionPhase.WAITING, ExecutionPhase.PREFILL, ExecutionPhase.DECODE)
               for r in self.requests):
            raise ValueError("requests must have an active execution phase")


def scenarios() -> tuple[ResearchScenario, ...]:
    return (
        ResearchScenario("decode_flood", (
            SimulatedRequest("decode_0", ExecutionPhase.DECODE, 0, 5),
            SimulatedRequest("waiting", ExecutionPhase.WAITING, 1, 1),
            SimulatedRequest("prefill", ExecutionPhase.PREFILL, 2, 2),
            *(SimulatedRequest(f"decode_{i}", ExecutionPhase.DECODE, i * 3, 5)
              for i in range(1, 9)),
        )),
        ResearchScenario("mixed_length", tuple(SimulatedRequest(
            f"r{i}", (ExecutionPhase.WAITING, ExecutionPhase.PREFILL, ExecutionPhase.DECODE)[i % 3],
            0, length) for i, length in enumerate((1, 8, 2, 16, 3, 1, 12, 4, 2)))),
        ResearchScenario("continuous_arrival", tuple(SimulatedRequest(
            f"r{i}", (ExecutionPhase.WAITING, ExecutionPhase.PREFILL, ExecutionPhase.DECODE)[i % 3],
            i * 2, 3 + i % 4) for i in range(18))),
    )
