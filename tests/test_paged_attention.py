"""Ordered simulated attention block lookup without tensors or GPU execution."""

import pytest

from minillm.memory import PagedKVCacheManager, PagedAttentionSimulator


def setup():
    manager = PagedKVCacheManager(128, tokens_per_block=4, bytes_per_token=2)
    return manager, PagedAttentionSimulator(manager)


def test_multiple_blocks_read_in_logical_order():
    manager, simulator = setup()
    manager.allocate_request("A")
    blocks = manager.append_tokens("A", 13)
    assert simulator.get_attention_blocks("A") == [block.block_id for block in blocks]
    assert len(blocks) == 4


def test_interleaved_requests_are_isolated():
    manager, simulator = setup()
    manager.allocate_request("A")
    manager.allocate_request("B")
    manager.append_tokens("A", 4)
    second = manager.append_tokens("B", 8)
    first = manager.append_tokens("A", 8)
    assert simulator.get_attention_blocks("A") == [b.block_id for b in first]
    assert simulator.get_attention_blocks("B") == [b.block_id for b in second]
    assert not set(simulator.get_attention_blocks("A")).intersection(simulator.get_attention_blocks("B"))


def test_released_request_cannot_be_read():
    manager, simulator = setup()
    manager.allocate_request("A")
    manager.append_tokens("A", 5)
    manager.release_request("A")
    with pytest.raises(KeyError):
        simulator.get_attention_blocks("A")
    with pytest.raises(KeyError):
        simulator.get_attention_blocks("unknown")


def test_empty_request_returns_empty_list_and_growth_is_visible():
    manager, simulator = setup()
    manager.allocate_request("A")
    assert simulator.get_attention_blocks("A") == []
    blocks = manager.append_tokens("A", 1)
    assert simulator.get_attention_blocks("A") == [blocks[0].block_id]


def test_lookup_is_read_only_and_returned_list_is_detached():
    manager, simulator = setup()
    manager.allocate_request("A")
    blocks = manager.append_tokens("A", 5)
    before = manager.snapshot()
    selected = simulator.get_attention_blocks("A")
    selected.clear()
    assert simulator.get_attention_blocks("A") == [b.block_id for b in blocks]
    assert manager.snapshot() == before
