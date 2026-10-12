"""Internal generation options shared by Engine and GenerationRunner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GenerationConfig:
    """Validated-at-the-boundary options for one generation request."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = -1
    top_p: float = 1.0
    stop: Optional[list[str]] = None
    stream: bool = False
    seed: Optional[int] = None
