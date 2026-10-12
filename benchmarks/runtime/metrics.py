"""Runtime observations; these are not model or HTTP performance metrics."""

from dataclasses import asdict, dataclass

from minillm.runtime import RuntimeRequestContext


def queue_percentile(values, percentile):
    """Linear interpolation over sorted virtual-clock queue delays."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    request_id: str
    status: str
    queue_delay_ms: float | None
    ttft_ms: float | None
    total_latency_ms: float | None
    decode_tokens: int
    decode_steps: int


def elapsed_ms(start, end):
    if start is None or end is None or end < start:
        return None
    return (end - start) * 1000


def measure_request(context: RuntimeRequestContext) -> RequestMetrics:
    request = context.request
    decode = context.decode_state
    return RequestMetrics(
        request.request_id, request.status.value,
        elapsed_ms(request.created_time, request.started_time),
        elapsed_ms(request.started_time, decode.first_token_time),
        elapsed_ms(request.created_time, request.finished_time),
        decode.decode_tokens, decode.decode_steps,
    )


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    cycle: int
    phase: str
    request_ids: tuple[str, ...]
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    scenario: dict
    scope: str
    worker: str
    requests: tuple[RequestMetrics, ...]
    scheduler_decisions: tuple[SchedulerDecision, ...]
    wall_time_ms: float
    step_budget_exhausted: bool
    final_active_requests: int

    def to_dict(self) -> dict:
        return asdict(self)
