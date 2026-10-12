"""Model ownership and forward execution for one local inference worker."""

from __future__ import annotations

import logging
from typing import Any

from ..config import Settings

logger = logging.getLogger("minillm.engine.worker")


class ModelWorker:
    """Owns the tokenizer/model and is the only component that calls the model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: str = "cpu"
        self.dtype: Any = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load the tokenizer and model using the existing single-worker policy."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = self.settings.resolve_device()
        self.dtype = self.settings.resolve_dtype()

        logger.info(
            "loading %s (device=%s, dtype=%s)",
            self.settings.model_name,
            self.device,
            self.dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_name,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        logger.info("model loaded")

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache: bool,
    ):
        if self.model is None:
            raise RuntimeError("engine 未加载，请先调用 load()")
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
