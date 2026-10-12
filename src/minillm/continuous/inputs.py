"""CPU tensor inputs for logical prefill and decode batches, without forwards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from ..contracts import ExecutionPhase
from ..runtime.context import RuntimeRequestContext


class PromptTokenizer(Protocol):
    """Minimal tokenizer adapter; no model loading or remote access required."""

    pad_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = True) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class BatchModelInput:
    """Rows map exactly to request_ids; tensors are torch.long on CPU.

    Decode attention_mask covers only the new-token input, not past KV tokens.
    A future cache-aware backend must extend it and calculate position IDs.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    request_ids: tuple[str, ...]
    phase: ExecutionPhase


class BatchInputBuilder:
    """Build inputs without mutating context, tokenizer, or cache state.

    Left padding is the default for causal LM prefill; right padding is also
    available for input-shape experiments. Mask values are based on lengths,
    not token equality, so a real token equal to pad_token_id remains visible.
    """

    def __init__(self, tokenizer: PromptTokenizer, *, pad_token_id: int | None = None,
                 padding_side: str = "left") -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id
        if type(self.pad_token_id) is not int or self.pad_token_id < 0:
            raise ValueError("provide a non-negative pad_token_id explicitly if tokenizer has none")
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be left or right")
        self.padding_side = padding_side

    @staticmethod
    def _validate_requests(requests):
        contexts = tuple(requests)
        if not contexts:
            raise ValueError("empty batch is not supported")
        ids = [c.request_id for c in contexts]
        if any(not request_id for request_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("batch requires non-empty unique request IDs")
        if any(c.request.is_terminal for c in contexts):
            raise ValueError("terminal requests cannot enter model inputs")
        return contexts

    @staticmethod
    def _validate_tokens(tokens):
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("token input must contain non-negative integer token IDs")

    def build_prefill(self, requests: Sequence[RuntimeRequestContext]) -> BatchModelInput:
        contexts = self._validate_requests(requests)
        if any(c.decode_state.execution_phase not in {ExecutionPhase.WAITING, ExecutionPhase.PREFILL}
               for c in contexts):
            raise ValueError("prefill requires waiting or prefill contexts")
        sequences = [list(self.tokenizer.encode(c.request.prompt, add_special_tokens=True))
                     for c in contexts]
        for tokens in sequences:
            self._validate_tokens(tokens)
        width = max(map(len, sequences))
        ids = torch.full((len(contexts), width), self.pad_token_id, dtype=torch.long, device="cpu")
        mask = torch.zeros_like(ids)
        for row, tokens in enumerate(sequences):
            start = width - len(tokens) if self.padding_side == "left" else 0
            ids[row, start:start + len(tokens)] = torch.tensor(tokens, dtype=torch.long, device="cpu")
            mask[row, start:start + len(tokens)] = 1
        return BatchModelInput(ids, mask, tuple(c.request_id for c in contexts), ExecutionPhase.PREFILL)

    def build_decode(self, requests: Sequence[RuntimeRequestContext]) -> BatchModelInput:
        contexts = self._validate_requests(requests)
        if any(c.decode_state.execution_phase is not ExecutionPhase.DECODE for c in contexts):
            raise ValueError("decode requires DECODE phase")
        tokens = []
        for context in contexts:
            if not context.request.generated_tokens:
                raise ValueError("decode requires an already-generated latest token")
            token = context.request.generated_tokens[-1]
            self._validate_tokens([token])
            tokens.append([token])
        ids = torch.tensor(tokens, dtype=torch.long, device="cpu")
        return BatchModelInput(ids, torch.ones_like(ids), tuple(c.request_id for c in contexts), ExecutionPhase.DECODE)
