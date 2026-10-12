"""Counters for validated runtime execution, independent of benchmarks."""

from dataclasses import dataclass


@dataclass
class ExecutionMetrics:
    """Request counts represent step participations, not unique requests."""

    prefill_requests: int = 0
    prefill_tokens: int = 0
    decode_requests: int = 0
    decode_tokens: int = 0
    failed_requests: int = 0

    def record_prefill(self, prompt_tokens: int) -> None:
        self.prefill_requests += 1
        self.prefill_tokens += prompt_tokens

    def record_decode(self, token_id: int | None) -> None:
        self.decode_requests += 1
        self.decode_tokens += int(token_id is not None)

    def record_failure(self, count: int = 1) -> None:
        self.failed_requests += count
