"""配置解析测试。"""

import pytest

from minillm.config import Settings


def test_kv_cache_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("MINILLM_USE_KV_CACHE", raising=False)
    assert Settings().use_kv_cache is True


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("1", True), ("off", False), ("0", False)])
def test_kv_cache_environment_flag(monkeypatch, value, expected):
    monkeypatch.setenv("MINILLM_USE_KV_CACHE", value)
    assert Settings().use_kv_cache is expected


def test_invalid_kv_cache_environment_flag(monkeypatch):
    monkeypatch.setenv("MINILLM_USE_KV_CACHE", "maybe")
    with pytest.raises(ValueError, match="MINILLM_USE_KV_CACHE"):
        Settings()


def test_scheduler_environment_settings(monkeypatch):
    monkeypatch.setenv("MINILLM_MAX_CONCURRENT_GENERATIONS", "2")
    monkeypatch.setenv("MINILLM_REQUEST_QUEUE_CAPACITY", "5")
    monkeypatch.setenv("MINILLM_QUEUE_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("MINILLM_INFERENCE_TIMEOUT_SECONDS", "9")
    settings = Settings()
    assert settings.max_concurrent_generations == 2
    assert settings.request_queue_capacity == 5
    assert settings.queue_timeout_seconds == 1.5
    assert settings.inference_timeout_seconds == 9


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_concurrent_generations", 0, "MINILLM_MAX_CONCURRENT_GENERATIONS"),
        ("request_queue_capacity", -1, "MINILLM_REQUEST_QUEUE_CAPACITY"),
        ("queue_timeout_seconds", 0, "MINILLM_QUEUE_TIMEOUT_SECONDS"),
        ("inference_timeout_seconds", 0, "MINILLM_INFERENCE_TIMEOUT_SECONDS"),
    ],
)
def test_invalid_scheduler_settings(field, value, message):
    with pytest.raises(ValueError, match=message):
        Settings(**{field: value})
