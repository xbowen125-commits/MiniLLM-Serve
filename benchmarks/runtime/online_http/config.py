"""Validated, immutable protocol configuration; no model or GPU is loaded here.

The canonical JSON is the experiment contract. Its SHA-256 includes the arrival
trace, frozen SLO thresholds, sampling parameters and both modes' shared limits.
Changing any of those requires a different result directory/configuration hash.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically and reject NaN/infinity."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float,
            *, exclusive_minimum: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value > maximum or (value <= minimum if exclusive_minimum else value < minimum):
        raise ValueError(f"{name} is outside the permitted range")
    return float(value)


def _text(value: Any, name: str, max_length: int = 1024) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"{name} must be a nonempty string (max {max_length} characters)")
    return value


_DEFAULTS: dict[str, Any] = {
    "protocol_version": 1,
    "name": "online-http-protocol",
    "stream": True,
    "max_client_concurrency": 2,
    "max_requests": 32,
    "max_model_len": 2048,
    "admission_limit": 2,
    "queue_capacity": 8,
    "queue_timeout_s": 30.0,
    "inference_timeout_s": 120.0,
    "request_timeout_s": 180.0,
    "drain_timeout_s": 60.0,
    "startup_timeout_s": 180.0,
    "poll_interval_s": 0.05,
    "warmup_requests": 2,
    "pairs": 2,
    "seed": 42,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "stop": None,
    "output_capacity": 8,
    "send_timeout_s": 10.0,
    "arrival_tolerance_ms": 25.0,
    "max_run_seconds": 600.0,
    "telemetry_interval_s": 0.05,
}
_REQUIRED = {"evidence_kind", "model", "model_revision", "tokenizer_revision",
             "dtype", "device", "slo", "requests"}


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    """An immutable experiment contract, including predeclared SLO thresholds.

    Returned dictionaries are independent copies rather than references to an
    internal mutable request list. Real GPU protocols require immutable model
    and tokenizer commit revisions, never a moving branch such as ``main``.
    """

    _canonical_json: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtocolConfig":
        if not isinstance(value, Mapping):
            raise ValueError("configuration must be a JSON object")
        missing = _REQUIRED - set(value)
        unknown = set(value) - set(_DEFAULTS) - _REQUIRED - {"authoritative_sse_usage", "warmup_strategy"}
        if missing or unknown:
            raise ValueError(f"configuration fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
        # JSON round-trip both copies caller data and rejects unsupported values.
        data = json.loads(canonical_json({**_DEFAULTS, **dict(value)}))
        _integer(data["protocol_version"], "protocol_version", 1, 1)
        _text(data["name"], "name", 128)
        if data["evidence_kind"] not in {"cpu_fake", "real_gpu"}:
            raise ValueError("evidence_kind must be cpu_fake or real_gpu")
        _text(data["model"], "model", 256)
        for key in ("model_revision", "tokenizer_revision"):
            revision = _text(data[key], key, 256)
            if data["evidence_kind"] == "real_gpu" and not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(f"{key} must be an immutable full 40-character commit SHA")
        if data["dtype"] not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        if data["device"] not in {"cpu", "cuda"}:
            raise ValueError("device must be explicit cpu or cuda, not auto")
        if data["evidence_kind"] == "cpu_fake" and data["device"] != "cpu":
            raise ValueError("cpu_fake evidence requires device=cpu")
        if data["evidence_kind"] == "real_gpu" and data["device"] != "cuda":
            raise ValueError("real_gpu evidence requires device=cuda")
        if type(data["stream"]) is not bool:
            raise ValueError("stream must be a boolean")
        if "authoritative_sse_usage" in data and type(data["authoritative_sse_usage"]) is not bool:
            raise ValueError("authoritative_sse_usage must be boolean")
        if "warmup_strategy" in data and data["warmup_strategy"] not in {"sequential", "sequential_and_pairs"}:
            raise ValueError("invalid warmup_strategy")
        for key, low, high in (
            ("max_client_concurrency", 1, 16), ("max_requests", 1, 256),
            ("max_model_len", 2, 32768), ("admission_limit", 1, 2),
            ("queue_capacity", 0, 256), ("warmup_requests", 0, 32),
            ("pairs", 1, 20), ("output_capacity", 1, 4096),
            ("seed", 0, 2**63 - 1), ("top_k", -1, 100000),
        ):
            _integer(data[key], key, low, high)
        for key in ("queue_timeout_s", "inference_timeout_s", "request_timeout_s",
                    "drain_timeout_s", "startup_timeout_s", "send_timeout_s"):
            data[key] = _number(data[key], key, 0, 600, exclusive_minimum=True)
        data["poll_interval_s"] = _number(data["poll_interval_s"], "poll_interval_s", .01, 60)
        data["arrival_tolerance_ms"] = _number(data["arrival_tolerance_ms"],
                                                "arrival_tolerance_ms", 0, 60000)
        data["max_run_seconds"] = _number(data["max_run_seconds"], "max_run_seconds", 0, 3600,
                                           exclusive_minimum=True)
        data["telemetry_interval_s"] = _number(data["telemetry_interval_s"],
                                               "telemetry_interval_s", .01, 60)
        data["temperature"] = _number(data["temperature"], "temperature", 0, 2)
        data["top_p"] = _number(data["top_p"], "top_p", 0, 1)
        stop = data["stop"]
        if stop is not None:
            if not isinstance(stop, list) or not 1 <= len(stop) <= 4:
                raise ValueError("stop must be null or a list of 1 to 4 nonempty strings")
            for item in stop:
                _text(item, "stop item", 4096)
        slo = data["slo"]
        if not isinstance(slo, dict) or set(slo) != {"ttft_ms", "e2e_ms", "tpot_ms"}:
            raise ValueError("slo must predeclare ttft_ms, e2e_ms and tpot_ms")
        for key in slo:
            slo[key] = _number(slo[key], f"slo.{key}", 0, 3600000, exclusive_minimum=True)
        requests = data["requests"]
        if not isinstance(requests, list) or not 1 <= len(requests) <= data["max_requests"]:
            raise ValueError("requests must be nonempty and bounded by max_requests")
        seen: set[str] = set()
        previous_arrival = -1.0
        allowed = {"request_id", "arrival_s", "messages", "max_tokens",
                   "expected_prompt_tokens", "expected_visible_output_tokens"}
        for index, item in enumerate(requests):
            if not isinstance(item, dict) or not set(item) <= allowed:
                raise ValueError(f"requests[{index}] has invalid fields")
            if not {"request_id", "arrival_s", "messages", "max_tokens"} <= set(item):
                raise ValueError(f"requests[{index}] lacks required request fields")
            request_id = _text(item["request_id"], "request_id", 128)
            if request_id in seen:
                raise ValueError("request_id values must be unique")
            seen.add(request_id)
            arrival = _number(item["arrival_s"], "arrival_s", 0, 3600)
            if arrival < previous_arrival:
                raise ValueError("requests must be in nondecreasing arrival_s order; ties preserve input order")
            previous_arrival = item["arrival_s"] = arrival
            _integer(item["max_tokens"], "max_tokens", 1, min(4096, data["max_model_len"] - 1))
            for key in ("expected_prompt_tokens", "expected_visible_output_tokens"):
                if key in item and item[key] is not None:
                    _integer(item[key], key, 0, 1000000)
            messages = item["messages"]
            if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
                raise ValueError("messages must be a nonempty list (max 128)")
            for message in messages:
                if not isinstance(message, dict) or set(message) != {"role", "content"}:
                    raise ValueError("each message must contain only role and content")
                if message["role"] not in {"system", "user", "assistant"}:
                    raise ValueError("message role is invalid")
                if not isinstance(message["content"], str) or len(message["content"]) > 65536:
                    raise ValueError("message content must be text (max 65536 characters)")
        # Establish a finite whole-session ceiling before the runner starts.
        if previous_arrival + data["request_timeout_s"] > data["max_run_seconds"]:
            raise ValueError("arrival horizon plus request timeout must fit within max_run_seconds")
        return cls(canonical_json(data))

    @property
    def data(self) -> dict[str, Any]:
        return self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible copy, including explicit defaults."""
        return json.loads(self._canonical_json)

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()

    @property
    def sha256(self) -> str:
        """Convenience alias for callers recording configuration provenance."""
        return self.config_sha256


def load_config(path: str | Path) -> ProtocolConfig:
    """Read and validate an experiment contract without starting a server."""
    with Path(path).open(encoding="utf-8") as source:
        return ProtocolConfig.from_dict(json.load(source))
