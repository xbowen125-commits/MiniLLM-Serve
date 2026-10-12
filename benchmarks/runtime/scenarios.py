"""Bounded, reproducible workload configurations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str = "single-request"
    token_limits: tuple[int, ...] = (4,)
    prompt_words: int = 8
    max_batch_size: int = 4
    max_steps: int = 1000

    def __post_init__(self):
        if not self.name or not 1 <= len(self.token_limits) <= 1000:
            raise ValueError("scenario requires a name and 1..1000 requests")
        if any(not 1 <= limit <= 1000 for limit in self.token_limits):
            raise ValueError("token limits must be 1..1000")
        if not 1 <= self.prompt_words <= 10000:
            raise ValueError("prompt_words must be 1..10000")
        if not 1 <= self.max_batch_size <= 1000 or not 1 <= self.max_steps <= 10000:
            raise ValueError("batch size or step budget outside allowed range")


def single_request() -> Scenario:
    return Scenario()


def mixed_lengths() -> Scenario:
    return Scenario(name="mixed-lengths", token_limits=(2, 5, 3, 8), max_batch_size=2)
