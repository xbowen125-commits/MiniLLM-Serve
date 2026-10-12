"""Engine runtime timing formulas and cache tensor-storage accounting."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RealWorkerMetrics:
    ttft_ms: float | None
    decode_latency_ms: float
    total_latency_ms: float
    generated_tokens: int
    tokens_per_second: float | None

    def to_dict(self):
        return asdict(self)


def calculate_metrics(*, start: float, first_token: float | None,
                      end: float, decode_seconds: float,
                      generated_tokens: int) -> RealWorkerMetrics:
    if end < start or decode_seconds < 0 or generated_tokens < 0:
        raise ValueError("invalid benchmark timing or token count")
    duration = end - start
    return RealWorkerMetrics(
        None if first_token is None else (first_token - start) * 1000,
        decode_seconds * 1000, duration * 1000, generated_tokens,
        generated_tokens / duration if duration > 0 else None,
    )


def cache_storage_bytes(cache) -> int | None:
    """Count unique live K/V tensor storages for modern Transformers caches.

    Returns None for unsupported layouts rather than pretending no cache exists.
    This is storage size, distinct from allocator deltas/reserved GPU memory.
    """
    layers = getattr(cache, "layers", None)
    if layers is None:
        return None
    storages = {}
    for layer in layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if tensor is None:
                continue
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())
