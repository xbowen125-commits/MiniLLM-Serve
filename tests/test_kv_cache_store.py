"""Atomic simulated KV store growth and ledger consistency tests."""

import pytest

from minillm.memory import KVCacheStore, KVCapacityExceeded


def store(capacity=64):
    return KVCacheStore(capacity, tokens_per_block=4, bytes_per_token=2)


def test_request_allocates_multiple_ordered_blocks():
    cache = store()
    blocks = cache.allocate_request("A", 9)
    assert len(blocks) == 3
    assert all(b.size_bytes == 8 for b in blocks)
    assert cache.get_request_blocks("A") == blocks
    snapshot = cache.snapshot()
    assert snapshot.memory.allocated_bytes == 24
    assert tuple(snapshot.block_table["A"].values()) == tuple(b.block_id for b in blocks)
    assert snapshot.request_tokens == {"A": 9}


def test_append_reuses_tail_then_allocates_new_blocks():
    cache = store()
    initial = cache.allocate_request("A", 2)
    assert cache.append_tokens("A", 2) == initial
    grown = cache.append_tokens("A", 5)
    assert len(grown) == 3
    assert grown[0] is initial[0]
    assert cache.snapshot().request_tokens["A"] == 9


def test_release_frees_mapping_and_capacity_without_affecting_other_request():
    cache = store()
    first = cache.allocate_request("A", 5)
    second = cache.allocate_request("B", 4)
    cache.release_request("A")
    snapshot = cache.snapshot()
    assert "A" not in snapshot.block_table
    assert "A" not in snapshot.request_tokens
    assert snapshot.memory.allocated_bytes == 8
    assert cache.get_request_blocks("B") == second
    assert all(not b.allocated for b in first)
    cache.release_request("A")
    assert cache.snapshot() == snapshot


def test_insufficient_capacity_is_atomic_for_allocation_and_append():
    cache = store(16)
    with pytest.raises(KVCapacityExceeded):
        cache.allocate_request("A", 9)
    assert cache.snapshot().block_table == {}
    assert cache.snapshot().request_tokens == {}
    assert cache.snapshot().memory.allocated_bytes == 0
    initial = cache.allocate_request("A", 4)
    with pytest.raises(KVCapacityExceeded):
        cache.append_tokens("A", 5)
    assert cache.get_request_blocks("A") == initial
    assert cache.snapshot().request_tokens == {"A": 4}
    assert cache.snapshot().memory.allocated_bytes == 8


def test_zero_token_request_and_unknown_request():
    cache = store()
    assert cache.allocate_request("A", 0) == ()
    assert cache.append_tokens("A", 0) == ()
    assert len(cache.append_tokens("A", 1)) == 1
    with pytest.raises(KeyError):
        cache.append_tokens("unknown", 1)
    with pytest.raises(ValueError):
        cache.allocate_request("A", 2)
    cache.release_request("A")
    with pytest.raises(KeyError):
        cache.get_request_blocks("A")
