"""Independent logical allocations, capacity enforcement, and snapshots."""

from dataclasses import FrozenInstanceError

import pytest

from minillm.memory import KVCapacityExceeded, KVMemoryManager


def test_multiple_requests_have_independent_blocks():
    manager = KVMemoryManager(100)
    first = manager.allocate("A", 20)
    second = manager.allocate("B", 30)
    extra = manager.allocate("A", 10)
    assert len({first.block_id, second.block_id, extra.block_id}) == 3
    assert manager.get_blocks("A") == (first, extra)
    assert manager.get_blocks("B") == (second,)
    assert all(block.allocated for block in (first, second, extra))


def test_release_restores_capacity_and_preserves_other_request():
    manager = KVMemoryManager(100)
    first = manager.allocate("A", 40)
    second = manager.allocate("B", 60)
    manager.release("A")
    assert first.allocated is False
    assert second.allocated is True
    assert manager.get_blocks("A") == ()
    assert manager.snapshot().free_bytes == 40
    replacement = manager.allocate("C", 40)
    assert replacement.block_id != first.block_id


def test_capacity_rejection_does_not_partial_allocate():
    manager = KVMemoryManager(100)
    first = manager.allocate("A", 80)
    with pytest.raises(KVCapacityExceeded):
        manager.allocate("B", 21)
    assert manager.get_blocks("B") == ()
    assert manager.get_blocks("A") == (first,)
    snapshot = manager.snapshot()
    assert snapshot.allocated_bytes == 80
    assert snapshot.allocation_count == 1
    assert snapshot.rejected_allocations == 1
    manager.allocate("B", 20)
    assert manager.snapshot().free_bytes == 0


def test_snapshot_tracks_current_peak_and_release_counters():
    manager = KVMemoryManager(100)
    manager.allocate("A", 30)
    manager.allocate("A", 20)
    manager.allocate("B", 40)
    before = manager.snapshot()
    manager.release("A")
    after = manager.snapshot()
    assert before.active_requests == 2
    assert before.allocated_blocks == 3
    assert before.allocated_bytes == 90
    assert after.allocated_bytes == 40
    assert after.free_bytes == 60
    assert after.peak_allocated_bytes == 90
    assert after.active_requests == 1
    assert after.allocated_blocks == 1
    assert after.released_blocks == 2
    assert after.released_bytes == 50


def test_release_is_idempotent_and_metadata_is_protected():
    manager = KVMemoryManager(10)
    block = manager.allocate("A", 10)
    with pytest.raises(FrozenInstanceError):
        block.size_bytes = 100
    manager.release("A")
    first = manager.snapshot()
    manager.release("A")
    manager.release("unknown")
    assert manager.snapshot() == first


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_allocation_sizes(size):
    with pytest.raises(ValueError):
        KVMemoryManager(10).allocate("A", size)


def test_zero_capacity_and_invalid_configuration():
    with pytest.raises(KVCapacityExceeded):
        KVMemoryManager(0).allocate("A", 1)
    with pytest.raises(ValueError):
        KVMemoryManager(-1)
    with pytest.raises(ValueError):
        KVMemoryManager(10).allocate("", 1)
