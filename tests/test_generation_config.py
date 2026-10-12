"""Tests for the internal, transport-independent generation options."""

from minillm.engine import GenerationConfig


def test_generation_config_defaults():
    config = GenerationConfig()

    assert config.max_tokens == 512
    assert config.temperature == 0.7
    assert config.top_k == -1
    assert config.top_p == 1.0
    assert config.stop is None
    assert config.stream is False
    assert config.seed is None


def test_generation_config_overrides():
    config = GenerationConfig(
        max_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        stream=True,
        seed=42,
    )

    assert config.max_tokens == 128
    assert config.temperature == 0.2
    assert config.top_k == 50
    assert config.top_p == 0.95
    assert config.stream is True
    assert config.seed == 42


def test_generation_config_preserves_stop_sequences():
    stop = ["END", "<stop>"]

    config = GenerationConfig(stop=stop)

    assert config.stop == stop
