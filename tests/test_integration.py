"""真实模型集成测试：验证 engine 端到端能跑通。

需要 torch + transformers，且首次运行会自动下载模型权重（约 0.5~3 GB）。
默认跳过：`pytest -m slow` 才会执行。可用 MINILLM_MODEL 环境变量换模型。

用法（建议在 .venv 内）：
    pytest -m slow tests/test_integration.py -v
"""

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.slow

from minillm.config import Settings  # noqa: E402
from minillm.engine import GenerationEngine  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    model = os.environ.get("MINILLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = os.environ.get("MINILLM_TEST_DEVICE", "cpu")
    eng = GenerationEngine(Settings(model_name=model, device=device, max_model_len=512, use_kv_cache=True))
    eng.load()
    return eng


def test_chat_returns_text(engine):
    result = engine.chat(
        messages=[{"role": "user", "content": "用一句话介绍你自己"}],
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
    )
    assert result.text.strip()
    assert result.prompt_tokens > 0
    assert 0 < result.completion_tokens <= 32
    assert result.finish_reason in {"stop", "length"}


def test_chat_template_uses_chat_format(engine):
    prompt = engine.build_prompt([{"role": "user", "content": "hi"}])
    # Qwen 系 chat template 的特征标记
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt


def test_kv_cache_matches_uncached_greedy_output(engine):
    prompt = engine.build_prompt(
        [
            {"role": "system", "content": "请简洁回答。"},
            {"role": "user", "content": "用一句话解释 KV Cache。"},
        ]
    )
    engine.settings.use_kv_cache = False
    without_cache = engine.generate(prompt, max_tokens=24, temperature=0.0, seed=42)
    engine.settings.use_kv_cache = True
    with_cache = engine.generate(prompt, max_tokens=24, temperature=0.0, seed=42)
    assert with_cache == without_cache


def test_cached_streaming_and_consecutive_requests_are_stable(engine):
    messages = [{"role": "user", "content": "只回答：缓存正常"}]
    engine.settings.use_kv_cache = True

    first_chunks = list(engine.stream_chat(messages, max_tokens=16, temperature=0.0, seed=42))
    second_chunks = list(engine.stream_chat(messages, max_tokens=16, temperature=0.0, seed=42))
    first_text = "".join(chunk.text for chunk in first_chunks)
    second_text = "".join(chunk.text for chunk in second_chunks)

    assert first_text
    assert first_text == second_text
    assert first_chunks[-1].finish_reason in {"stop", "length"}
    assert second_chunks[-1].finish_reason == first_chunks[-1].finish_reason
