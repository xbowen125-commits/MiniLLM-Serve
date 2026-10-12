"""Request-local logical indexes and detached simulated block mappings."""

import pytest

from minillm.memory import BlockTable, KVMemoryManager


def test_one_request_multiple_blocks_and_logical_order():
    table = BlockTable()
    assert table.append_block("A", 30) == 0
    assert table.append_block("A", 10) == 1
    assert table.append_block("A", 20) == 2
    assert table.get_blocks("A") == (30, 10, 20)
    assert table.snapshot()["A"] == {0: 30, 1: 10, 2: 20}


def test_requests_have_independent_logical_indexes():
    table = BlockTable()
    table.append_block("A", 1)
    assert table.append_block("B", 2) == 0
    table.append_block("A", 3)
    assert table.get_blocks("B") == (2,)
    assert table.snapshot() == {"A": {0: 1, 1: 3}, "B": {0: 2}}


def test_remove_only_removes_mapping_not_allocation():
    manager = KVMemoryManager(100)
    block = manager.allocate("A", 20)
    table = BlockTable()
    table.append_block("A", block.block_id)
    table.append_block("B", 999)
    table.remove_request("A")
    assert block.allocated is True
    assert manager.snapshot().allocated_bytes == 20
    assert table.get_blocks("B") == (999,)
    with pytest.raises(KeyError):
        table.get_blocks("A")


def test_snapshot_cannot_mutate_table_and_remains_stable():
    table = BlockTable()
    table.append_block("A", 1)
    snapshot = table.snapshot()
    with pytest.raises(TypeError):
        snapshot["A"][0] = 9
    table.append_block("A", 2)
    assert snapshot["A"] == {0: 1}
    assert table.get_blocks("A") == (1, 2)


@pytest.mark.parametrize("operation", ["get", "remove"])
def test_unknown_request_raises(operation):
    table = BlockTable()
    with pytest.raises(KeyError, match="unknown"):
        if operation == "get":
            table.get_blocks("unknown")
        else:
            table.remove_request("unknown")


def test_duplicate_physical_id_rejected_and_mapping_reusable_after_removal():
    table = BlockTable()
    table.append_block("A", 1)
    with pytest.raises(ValueError):
        table.append_block("B", 1)
    assert table.snapshot() == {"A": {0: 1}}
    table.remove_request("A")
    assert table.append_block("B", 1) == 0


def test_invalid_ids_rejected():
    table = BlockTable()
    for block_id in (-1, True, 1.5):
        with pytest.raises(ValueError):
            table.append_block("A", block_id)
    with pytest.raises(ValueError):
        table.append_block("", 1)
    assert table.snapshot() == {}
