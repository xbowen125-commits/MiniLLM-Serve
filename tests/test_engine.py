"""采样、Streaming 与 KV Cache 测试；不下载真实模型。"""

import gc
import threading
import weakref
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from minillm.config import Settings  # noqa: E402
from minillm.engine import GenerationEngine, GenerationMetrics, sample  # noqa: E402


def test_sample_greedy_argmax():
    logits = torch.tensor([[0.1, 0.2, 0.7, 0.0]])
    assert int(sample(logits, temperature=0.0).item()) == 2


def test_sample_greedy_ignores_top_p():
    logits = torch.tensor([[0.1, 0.2, 0.7, 0.0]])
    assert int(sample(logits, temperature=0.0, top_p=0.5).item()) == 2


def test_sample_top_k_restricts_vocab():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    torch.manual_seed(0)
    for _ in range(30):
        assert int(sample(logits, temperature=1.0, top_k=2).item()) in (0, 1)


def test_sample_top_p_filters_tail():
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    torch.manual_seed(0)
    for _ in range(30):
        assert int(sample(logits, temperature=1.0, top_p=0.4).item()) in (0, 1)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 9
    chat_template = "fake"

    def __init__(self, prompt_len=2):
        self.prompt_len = prompt_len
        self.last_messages = None

    def __call__(self, _prompt, return_tensors):
        assert return_tensors == "pt"
        return {
            "input_ids": torch.arange(self.prompt_len).unsqueeze(0),
            "attention_mask": torch.ones((1, self.prompt_len), dtype=torch.long),
        }

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.last_messages = messages
        return "\n".join(f"{message['role']}:{message['content']}" for message in messages)

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        table = {1: "你", 2: "好", 3: "A", 4: "B", 5: "世", 6: "界", 9: ""}
        return "".join(table[token] for token in values)


class FakeCache:
    pass


class FakeModel:
    def __init__(self, token_ids):
        self.token_ids = iter(token_ids)
        self.calls = []

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
    ):
        self.calls.append(
            {
                "input_len": input_ids.shape[1],
                "mask_len": attention_mask.shape[1],
                "position_ids": position_ids.tolist(),
                "cache_id": id(past_key_values) if past_key_values is not None else None,
                "use_cache": use_cache,
            }
        )
        token_id = next(self.token_ids)
        logits = torch.zeros((1, input_ids.shape[1], 10))
        logits[:, -1, token_id] = 10
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def make_engine(token_ids, *, use_kv_cache=False, prompt_len=2):
    engine = GenerationEngine(
        Settings(model_name="fake", device="cpu", max_model_len=64, use_kv_cache=use_kv_cache)
    )
    engine.tokenizer = FakeTokenizer(prompt_len)
    engine.model = FakeModel(token_ids)
    engine.device = "cpu"
    engine.cache_refs = []

    def create_cache():
        cache = FakeCache()
        engine.cache_refs.append(weakref.ref(cache))
        return cache

    engine._create_kv_cache = create_cache
    return engine


def generated_text(chunks):
    return "".join(chunk.text for chunk in chunks if chunk.text)


def test_stream_generate_is_incremental_and_reports_token_counts():
    engine = make_engine([1, 2, 9])
    chunks = engine.stream_generate("prompt", max_tokens=5, temperature=0)

    first = next(chunks)
    assert first.text == "你"
    assert len(engine.model.calls) == 1
    assert next(chunks).text == "好"
    final = next(chunks)
    assert final.finish_reason == "stop"
    assert final.prompt_tokens == 2
    assert final.completion_tokens == 3
    with pytest.raises(StopIteration):
        next(chunks)


@pytest.mark.parametrize(
    ("use_kv_cache", "input_lengths"),
    [(False, [4, 5, 6]), (True, [4, 1, 1])],
)
def test_kv_cache_controls_prefill_and_decode_input_lengths(use_kv_cache, input_lengths):
    engine = make_engine([1, 2, 9], use_kv_cache=use_kv_cache, prompt_len=4)
    chunks = list(engine.stream_generate("prompt", max_tokens=3, temperature=0))

    assert generated_text(chunks) == "你好"
    assert [call["input_len"] for call in engine.model.calls] == input_lengths
    assert [call["mask_len"] for call in engine.model.calls] == [4, 5, 6]
    assert [call["position_ids"] for call in engine.model.calls] == [
        [[0, 1, 2, 3]],
        [[4]] if use_kv_cache else [[0, 1, 2, 3, 4]],
        [[5]] if use_kv_cache else [[0, 1, 2, 3, 4, 5]],
    ]
    assert all(call["use_cache"] is use_kv_cache for call in engine.model.calls)


def test_cache_on_and_off_produce_identical_greedy_output():
    without_cache = make_engine([1, 2, 5, 6, 9], use_kv_cache=False)
    with_cache = make_engine([1, 2, 5, 6, 9], use_kv_cache=True)

    result_without = without_cache.generate("prompt", max_tokens=8, temperature=0)
    result_with = with_cache.generate("prompt", max_tokens=8, temperature=0)
    assert result_with == result_without


@pytest.mark.parametrize("prompt_len", [1, 7, 15])
def test_cache_supports_different_prompt_lengths(prompt_len):
    engine = make_engine([1, 9], use_kv_cache=True, prompt_len=prompt_len)
    list(engine.stream_generate("prompt", max_tokens=2, temperature=0))
    assert [call["input_len"] for call in engine.model.calls] == [prompt_len, 1]
    assert [call["mask_len"] for call in engine.model.calls] == [prompt_len, prompt_len + 1]


def test_multiturn_chat_uses_one_request_local_cache():
    engine = make_engine([1, 9], use_kv_cache=True, prompt_len=6)
    messages = [
        {"role": "system", "content": "简洁回答"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "继续"},
    ]
    chunks = list(engine.stream_chat(messages, max_tokens=2, temperature=0))
    assert generated_text(chunks) == "你"
    assert engine.tokenizer.last_messages == messages
    assert len({call["cache_id"] for call in engine.model.calls}) == 1


@pytest.mark.parametrize("use_kv_cache", [False, True])
def test_stop_sequence_is_not_leaked(use_kv_cache):
    engine = make_engine([3, 4], use_kv_cache=use_kv_cache)
    chunks = list(engine.stream_generate("prompt", max_tokens=2, temperature=0, stop=["AB"]))
    assert generated_text(chunks) == ""
    assert chunks[-1].finish_reason == "stop"


def test_streaming_with_cache_yields_in_order():
    engine = make_engine([1, 2, 9], use_kv_cache=True)
    chunks = engine.stream_generate("prompt", max_tokens=3, temperature=0)
    assert next(chunks).text == "你"
    assert next(chunks).text == "好"
    assert next(chunks).finish_reason == "stop"


def test_cancel_releases_request_cache_reference():
    engine = make_engine([1, 2], use_kv_cache=True)
    cancel = threading.Event()
    chunks = engine.stream_generate("prompt", cancel_event=cancel, temperature=0)
    assert next(chunks).text == "你"
    cache_ref = engine.cache_refs[0]

    cancel.set()
    assert list(chunks) == []
    gc.collect()
    assert cache_ref() is None


def test_pre_cancelled_request_does_not_allocate_cache():
    engine = make_engine([1], use_kv_cache=True)
    cancel = threading.Event()
    cancel.set()
    assert list(engine.stream_generate("prompt", cancel_event=cancel)) == []
    assert engine.cache_refs == []


def test_consecutive_requests_use_distinct_caches():
    engine = make_engine([1, 5], use_kv_cache=True)
    created_caches = []

    def create_cache():
        cache = FakeCache()
        created_caches.append(cache)
        return cache

    engine._create_kv_cache = create_cache
    first = list(engine.stream_generate("first", max_tokens=1, temperature=0))
    second = list(engine.stream_generate("second", max_tokens=1, temperature=0))

    assert generated_text(first) == "你"
    assert generated_text(second) == "世"
    assert len(created_caches) == 2
    assert created_caches[0] is not created_caches[1]


def test_generation_metrics_separate_prefill_and_decode():
    engine = make_engine([1, 2, 9], use_kv_cache=True)
    metrics = GenerationMetrics()
    list(engine.stream_generate("prompt", max_tokens=3, temperature=0, metrics=metrics))
    assert metrics.prefill_seconds > 0
    assert metrics.decode_seconds > 0
    assert metrics.decode_steps == 2
