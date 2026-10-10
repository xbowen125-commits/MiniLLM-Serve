"""schema 层单元测试：不依赖 torch / 模型。"""

import pytest
from pydantic import ValidationError

from minillm.schemas import ChatCompletionRequest, make_response, make_response_id


def test_request_defaults():
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert req.max_tokens == 512
    assert req.temperature == 0.7
    assert req.stream is False
    assert req.top_k == -1


def test_request_empty_messages_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[])


def test_request_bad_role_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "admin", "content": "hi"}])


def test_request_max_tokens_bounds():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=0,
        )


def test_request_temperature_bounds():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=3.0,
        )


def test_request_accepts_single_stop_string_and_normalizes_it():
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stop="END",
    )
    assert req.stop == ["END"]


@pytest.mark.parametrize("stop", [[], [""], ["a", "b", "c", "d", "e"]])
def test_request_rejects_invalid_stop(stop):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], stop=stop)


def test_response_shape_openai():
    resp = make_response(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        content="你好",
        finish_reason="stop",
        prompt_tokens=5,
        completion_tokens=3,
    )
    d = resp.model_dump()
    assert d["object"] == "chat.completion"
    assert d["id"].startswith("chatcmpl-")
    assert d["created"] > 0
    assert d["choices"][0]["message"]["role"] == "assistant"
    assert d["choices"][0]["message"]["content"] == "你好"
    assert d["choices"][0]["finish_reason"] == "stop"
    assert d["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def test_response_id_unique():
    assert make_response_id() != make_response_id()
