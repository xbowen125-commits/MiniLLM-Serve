"""CPU input shape, padding masks, and stable request mapping tests."""

import pytest
import torch

from minillm.continuous import BatchInputBuilder
from minillm.contracts import ExecutionPhase
from minillm.engine import RequestState
from minillm.runtime import RuntimeRequestContext


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens=True):
        return {"short": [0, 5], "long": [1, 2, 3, 4]}[text]


def context(name, prompt):
    return RuntimeRequestContext(RequestState(request_id=name, prompt=prompt))


@pytest.mark.parametrize("side, ids, masks", [
    ("left", [[0, 0, 0, 5], [1, 2, 3, 4]], [[0, 0, 1, 1], [1, 1, 1, 1]]),
    ("right", [[0, 5, 0, 0], [1, 2, 3, 4]], [[1, 1, 0, 0], [1, 1, 1, 1]]),
])
def test_prefill_padding_and_mask_are_length_based(side, ids, masks):
    requests = [context("A", "short"), context("B", "long")]
    result = BatchInputBuilder(FakeTokenizer(), padding_side=side).build_prefill(requests)
    assert result.input_ids.tolist() == ids
    assert result.attention_mask.tolist() == masks
    assert result.request_ids == ("A", "B")
    assert result.phase is ExecutionPhase.PREFILL
    assert result.input_ids.dtype is torch.long
    assert result.input_ids.device.type == "cpu"
    assert requests[0].request.generated_tokens == []
    assert requests[0].decode_state.execution_phase is ExecutionPhase.WAITING


def test_request_mapping_follows_input_order():
    requests = [context("B", "long"), context("A", "short")]
    result = BatchInputBuilder(FakeTokenizer()).build_prefill(requests)
    assert result.request_ids == ("B", "A")
    assert result.input_ids[0].tolist() == [1, 2, 3, 4]


def test_decode_uses_only_latest_tokens_in_request_order():
    first, second = context("A", "short"), context("B", "long")
    for request, tokens in [(first, [10, 11, 12]), (second, [21, 22])]:
        request.start()
        request.decode_state.transition_phase(ExecutionPhase.PREFILL)
        request.decode_state.transition_phase(ExecutionPhase.DECODE)
        for token in tokens:
            request.append_token(token)
    result = BatchInputBuilder(FakeTokenizer()).build_decode([second, first])
    assert result.input_ids.tolist() == [[22], [12]]
    assert result.attention_mask.tolist() == [[1], [1]]
    assert result.request_ids == ("B", "A")
    assert result.phase is ExecutionPhase.DECODE
    assert first.request.generated_tokens == [10, 11, 12]


def test_empty_batch_and_duplicate_ids_are_rejected():
    builder = BatchInputBuilder(FakeTokenizer())
    for method in (builder.build_prefill, builder.build_decode):
        with pytest.raises(ValueError, match="empty batch"):
            method([])
    request = context("A", "short")
    with pytest.raises(ValueError, match="unique"):
        builder.build_prefill([request, request])


def test_missing_decode_token_is_rejected():
    request = context("A", "short")
    request.decode_state.transition_phase(ExecutionPhase.PREFILL)
    request.decode_state.transition_phase(ExecutionPhase.DECODE)
    with pytest.raises(ValueError, match="latest token"):
        BatchInputBuilder(FakeTokenizer()).build_decode([request])
