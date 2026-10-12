"""Worker protocol and GenerationRunner substitution tests."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from minillm.engine import GenerationConfig, RequestState, RequestStatus, WorkerProtocol  # noqa: E402
from minillm.engine.cache import KVCacheManager  # noqa: E402
from minillm.engine.generation import GenerationRunner  # noqa: E402


class FakeWorker:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        logits = torch.zeros((1, kwargs["input_ids"].shape[1], 10))
        logits[:, -1, 9] = 10
        return SimpleNamespace(logits=logits, past_key_values=kwargs["past_key_values"])


class FakeTokenizer:
    eos_token_id = 9

    def decode(self, _token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return ""


def test_fake_worker_satisfies_protocol():
    assert isinstance(FakeWorker(), WorkerProtocol)


def test_generation_runner_accepts_protocol_worker():
    worker = FakeWorker()
    runner = GenerationRunner(worker, KVCacheManager())
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.ones_like(input_ids)
    request = RequestState(
        request_id="test-request",
        prompt="prompt",
        generation_config=GenerationConfig(max_tokens=1, temperature=0, stream=True),
    )

    chunks = list(
        runner.stream(
            request=request,
            tokenizer=FakeTokenizer(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_model_len=64,
            cancel_event=None,
            metrics=None,
            use_kv_cache=False,
            cache_factory=lambda: pytest.fail("cache must stay disabled"),
        )
    )

    assert len(worker.calls) == 1
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].completion_tokens == 1
    assert request.status is RequestStatus.FINISHED
    assert request.generated_tokens == [9]
    assert request.kv_cache_ref is None
