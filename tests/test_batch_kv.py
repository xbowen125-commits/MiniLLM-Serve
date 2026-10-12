"""Logical cache ownership, ordered lookup, replacement, and release tests."""

import gc
import weakref

import pytest

from minillm.continuous import BatchKVAdapter


class FakeCache:
    def __init__(self):
        self.tokens = []


def test_multiple_request_caches_are_independent():
    adapter = BatchKVAdapter()
    first, second = FakeCache(), FakeCache()
    adapter.add_request_cache("A", first)
    adapter.add_request_cache("B", second)
    adapter.get_batch_kv(["A"])[0].tokens.append(7)
    assert first.tokens == [7]
    assert second.tokens == []
    with pytest.raises(ValueError, match="multiple requests"):
        adapter.add_request_cache("C", first)


def test_batch_order_follows_request_ids():
    adapter = BatchKVAdapter()
    first, second = FakeCache(), FakeCache()
    adapter.add_request_cache("A", first)
    adapter.add_request_cache("B", second)
    selected = adapter.get_batch_kv(["B", "A"])
    assert selected[0] is second
    assert selected[1] is first
    assert adapter.get_batch_kv([]) == []


def test_decode_update_replaces_only_one_cache():
    adapter = BatchKVAdapter()
    first, second, updated = FakeCache(), FakeCache(), FakeCache()
    adapter.add_request_cache("A", first)
    adapter.add_request_cache("B", second)
    updated.tokens.append(9)
    adapter.update_request_cache("A", updated)
    assert adapter.get_batch_kv(["A"])[0] is updated
    assert adapter.get_batch_kv(["B"])[0] is second
    with pytest.raises(ValueError):
        adapter.update_request_cache("A", second)
    assert adapter.get_batch_kv(["A"])[0] is updated
    adapter.update_request_cache("A", updated)


def test_release_drops_owned_reference_and_preserves_other_request():
    adapter = BatchKVAdapter()
    first, second = FakeCache(), FakeCache()
    first_ref = weakref.ref(first)
    adapter.add_request_cache("A", first)
    adapter.add_request_cache("B", second)
    del first
    adapter.release_request("A")
    gc.collect()
    assert first_ref() is None
    assert len(adapter) == 1
    assert adapter.get_batch_kv(["B"])[0] is second
    with pytest.raises(KeyError):
        adapter.get_batch_kv(["A"])


@pytest.mark.parametrize("operation", ["lookup", "update", "release"])
def test_unknown_request_id_raises(operation):
    adapter = BatchKVAdapter()
    with pytest.raises(KeyError, match="unknown cache request_id"):
        if operation == "lookup":
            adapter.get_batch_kv(["unknown"])
        elif operation == "update":
            adapter.update_request_cache("unknown", FakeCache())
        else:
            adapter.release_request("unknown")


def test_invalid_registration_and_duplicate_batch_ids():
    adapter = BatchKVAdapter()
    with pytest.raises(ValueError):
        adapter.add_request_cache("", FakeCache())
    with pytest.raises(ValueError):
        adapter.add_request_cache("A", None)
    original = FakeCache()
    adapter.add_request_cache("A", original)
    with pytest.raises(ValueError):
        adapter.add_request_cache("A", FakeCache())
    with pytest.raises(ValueError):
        adapter.get_batch_kv(["A", "A"])
    assert adapter.get_batch_kv(["A"])[0] is original
