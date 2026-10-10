"""服务配置：全部支持通过环境变量覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false、1/0、yes/no 或 on/off")


@dataclass
class Settings:
    model_name: str = field(default_factory=lambda: _env("MINILLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    device: str = field(default_factory=lambda: _env("MINILLM_DEVICE", "auto"))
    dtype: str = field(default_factory=lambda: _env("MINILLM_DTYPE", "auto"))
    host: str = field(default_factory=lambda: _env("MINILLM_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("MINILLM_PORT", "8000")))
    max_model_len: int = field(default_factory=lambda: int(_env("MINILLM_MAX_MODEL_LEN", "2048")))
    use_kv_cache: bool = field(default_factory=lambda: _env_bool("MINILLM_USE_KV_CACHE", True))
    max_concurrent_generations: int = field(
        default_factory=lambda: int(_env("MINILLM_MAX_CONCURRENT_GENERATIONS", "1"))
    )
    request_queue_capacity: int = field(
        default_factory=lambda: int(_env("MINILLM_REQUEST_QUEUE_CAPACITY", "8"))
    )
    queue_timeout_seconds: float = field(
        default_factory=lambda: float(_env("MINILLM_QUEUE_TIMEOUT_SECONDS", "30"))
    )
    inference_timeout_seconds: float = field(
        default_factory=lambda: float(_env("MINILLM_INFERENCE_TIMEOUT_SECONDS", "120"))
    )

    def __post_init__(self) -> None:
        if self.max_concurrent_generations < 1:
            raise ValueError("MINILLM_MAX_CONCURRENT_GENERATIONS 必须 >= 1")
        if self.request_queue_capacity < 0:
            raise ValueError("MINILLM_REQUEST_QUEUE_CAPACITY 必须 >= 0")
        if self.queue_timeout_seconds <= 0:
            raise ValueError("MINILLM_QUEUE_TIMEOUT_SECONDS 必须 > 0")
        if self.inference_timeout_seconds <= 0:
            raise ValueError("MINILLM_INFERENCE_TIMEOUT_SECONDS 必须 > 0")

    def resolve_device(self) -> str:
        """cuda / cpu / auto -> 具体设备名。"""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def resolve_dtype(self):
        """auto -> CUDA 用 fp16、CPU 用 fp32。"""
        import torch

        if self.dtype != "auto":
            return getattr(torch, self.dtype)
        if self.resolve_device() == "cuda":
            return torch.float16
        return torch.float32
