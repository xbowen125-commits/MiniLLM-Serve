"""Simulated paged sequence growth, isolation, capacity, and cleanup."""

import pytest

from minillm.memory import PagedKVCacheManager, KVCapacityExceeded


def manager(capacity=128):
    return PagedKVCacheManager(capacity, tokens_per_block=4, bytes_per_token=2)


def test_long_sequence_allocates_multiple_pages():
    paged = manager()
    paged.allocate_request("A")
    assert paged.get_block_table("A") == {}
    blocks = paged.append_tokens("A", 17)
    assert len(blocks) == 5
    assert all(block.size_bytes == 8 and block.allocated for block in blocks)
    assert paged.snapshot().request_tokens["A"] == 17


def test_logical_order_remains_stable_across_tail_growth():
    paged = manager()
    paged.allocate_request("A")
    paged.append_tokens("A", 3)
    first = paged.get_block_table("A")
    paged.append_tokens("A", 1)
    assert paged.get_block_table("A") == first
    paged.append_tokens("A", 5)
    grown = paged.get_block_table("A")
    assert list(grown) == [0, 1, 2]
    assert grown[0] == first[0]
    assert len(first) == 1
    with pytest.raises(TypeError):
        grown[0] = 99


def test_multiple_requests_have_disjoint_physical_pages():
    paged = manager()
    paged.allocate_request("A")
    paged.allocate_request("B")
    paged.append_tokens("A", 7)
    paged.append_tokens("B", 9)
    first = paged.get_block_table("A")
    second = paged.get_block_table("B")
    assert not set(first.values()).intersection(second.values())
    paged.append_tokens("A", 5)
    assert paged.get_block_table("B") == second


def test_release_frees_capacity_mapping_and_token_tracking():
    paged = manager()
    paged.allocate_request("A")
    paged.allocate_request("B")
    released = paged.append_tokens("A", 5)
    paged.append_tokens("B", 4)
    paged.release_request("A")
    snapshot = paged.snapshot()
    assert snapshot.request_tokens == {"B": 4}
    assert set(snapshot.block_table) == {"B"}
    assert snapshot.memory.allocated_bytes == 8
    assert all(not block.allocated for block in released)
    paged.release_request("A")
    paged.release_request("B")
    assert paged.snapshot().memory.allocated_bytes == 0
    assert paged.snapshot().block_table == {}


def test_snapshot_is_consistent_with_page_rounding():
    paged = manager()
    paged.allocate_request("A")
    paged.append_tokens("A", 9)
    snapshot = paged.snapshot()
    assert snapshot.tokens_per_block == 4
    assert snapshot.bytes_per_token == 2
    assert snapshot.memory.allocated_blocks == 3
    assert snapshot.memory.allocated_bytes == 24
    assert snapshot.memory.free_bytes == 104
    assert snapshot.block_table["A"] == paged.get_block_table("A")


def test_capacity_failure_preserves_sequence_and_mapping():
    paged = manager(16)
    paged.allocate_request("A")
    paged.append_tokens("A", 4)
    original = paged.get_block_table("A")
    with pytest.raises(KVCapacityExceeded):
        paged.append_tokens("A", 5)
    assert paged.snapshot().request_tokens["A"] == 4
    assert paged.get_block_table("A") == original
    assert paged.snapshot().memory.allocated_bytes == 8


def test_unknown_and_duplicate_requests():
    paged = manager()
    with pytest.raises(KeyError):
        paged.get_block_table("unknown")
    with pytest.raises(KeyError):
        paged.append_tokens("unknown", 1)
    paged.allocate_request("A")
    with pytest.raises(ValueError):
        paged.allocate_request("A")
    paged.release_request("A")
    with pytest.raises(KeyError):
        paged.get_block_table("A")
