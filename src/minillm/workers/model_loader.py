"""Explicit model loading boundary with offline loading as the default."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str = "cpu"


class ModelLoaderProtocol(Protocol):
    def load(self) -> LoadedModel: ...


@dataclass
class TransformersModelLoader:
    """Load a local/cached causal LM only when load() is explicitly called."""

    model_name: str
    device: str = "cpu"
    dtype: Any = None
    local_files_only: bool = True

    def load(self) -> LoadedModel:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=self.local_files_only,
            trust_remote_code=False,
        )
        options = {"local_files_only": self.local_files_only, "trust_remote_code": False}
        if self.dtype is not None:
            options["dtype"] = self.dtype
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **options)
        model.to(self.device)
        model.eval()
        return LoadedModel(model, tokenizer, self.device)
