"""Bounded deterministic workloads and explicit virtual worker costs."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SimRequest:
    request_id: str
    arrival_time: float
    max_tokens: int

    def __post_init__(self):
        if not self.request_id or not math.isfinite(self.arrival_time) or self.arrival_time < 0:
            raise ValueError("invalid request ID or arrival time")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 1000:
            raise ValueError("max_tokens must be in 1..1000")


@dataclass(frozen=True)
class Scenario:
    name: str
    requests: tuple[SimRequest, ...]
    max_active: int = 4
    prefill_base_s: float = 0.02
    prefill_per_request_s: float = 0.01
    decode_base_s: float = 0.01
    decode_per_request_s: float = 0.005

    def __post_init__(self):
        if not self.name or not 1 <= len(self.requests) <= 1000:
            raise ValueError("scenario requires 1..1000 requests")
        if len({r.request_id for r in self.requests}) != len(self.requests):
            raise ValueError("duplicate request IDs")
        if type(self.max_active) is not int or self.max_active < 1:
            raise ValueError("max_active must be positive")
        for cost in (self.prefill_base_s, self.prefill_per_request_s,
                     self.decode_base_s, self.decode_per_request_s):
            if isinstance(cost, bool) or not math.isfinite(cost) or cost < 0:
                raise ValueError("costs must be finite and nonnegative")
        if self.prefill_base_s + self.prefill_per_request_s <= 0 or self.decode_base_s + self.decode_per_request_s <= 0:
            raise ValueError("each phase needs a positive cost")


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("short_requests", tuple(SimRequest(f"r{i}", 0, 2) for i in range(8))),
        Scenario("mixed_length", tuple(SimRequest(f"r{i}", 0, n)
                                       for i, n in enumerate((2, 16, 4, 32, 2, 8, 16, 4)))),
        Scenario("continuous_arrival", tuple(SimRequest(f"r{i}", i * 0.04, 8) for i in range(8))),
    )
