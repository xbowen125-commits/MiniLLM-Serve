"""Seeded, bounded token-length workloads for simulated KV capacity."""

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class MemoryScenario:
    request_count: int = 30
    seed: int = 42
    lengths: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
    max_tokens: int = 1024
    tokens_per_block: int = 16
    growth_step: int = 16

    def __post_init__(self):
        for name in ("request_count", "max_tokens", "tokens_per_block", "growth_step"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not self.lengths or any(type(n) is not int or not 0 < n <= self.max_tokens for n in self.lengths):
            raise ValueError("lengths must be positive and within max_tokens")

    def generate_lengths(self) -> tuple[int, ...]:
        rng = random.Random(self.seed)
        return tuple(rng.choice(self.lengths) for _ in range(self.request_count))
