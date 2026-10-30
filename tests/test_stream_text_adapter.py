"""Model-free correctness checks for the continuous SSE text boundary."""

import pytest

from minillm.runtime.streaming import IncrementalTextAdapter


def prefix_decoder(*prefixes):
    return lambda ids: prefixes[len(ids) - 1] if ids else ""


def test_words_are_incremental_and_finish_flushes_once():
    adapter = IncrementalTextAdapter(prefix_decoder("Hello", "Hello ", "Hello world"))

    assert [adapter.push(token) for token in (1, 2, 3)] == ["", "Hello ", ""]
    assert adapter.finish() == "world"
    assert adapter.finish() == ""
    assert adapter.text == "Hello world"
    assert adapter.token_ids == (1, 2, 3)
    with pytest.raises(RuntimeError, match="after finish"):
        adapter.push(4)


def test_stop_is_not_leaked_across_tokens():
    adapter = IncrementalTextAdapter(
        prefix_decoder("answer ", "answer <ST", "answer <STOP>not visible"),
        stop=["<STOP>"],
    )

    assert [adapter.push(token) for token in (1, 2, 3)] == ["answer ", "", ""]
    assert adapter.stopped
    assert adapter.push(4) == ""
    assert adapter.finish() == ""
    assert adapter.text == "answer "
    assert adapter.token_ids == (1, 2, 3)


def test_stop_prefix_containing_whitespace_is_held():
    adapter = IncrementalTextAdapter(
        prefix_decoder("safe ab ", "safe ab cd remainder"), stop=["ab cd"]
    )

    assert adapter.push(1) == "safe "
    assert adapter.push(2) == ""
    assert adapter.finish() == ""


def test_overlapping_stops_choose_first_visible_match():
    adapter = IncrementalTextAdapter(
        prefix_decoder("ok EN", "ok ENDING trailing"), stop=["ENDING", "END"]
    )

    assert adapter.push(1) == "ok "
    assert adapter.push(2) == ""
    assert adapter.finish() == ""
    assert adapter.text == "ok "


def test_earliest_stop_wins_independent_of_configuration_order():
    adapter = IncrementalTextAdapter(
        prefix_decoder("visible END hidden STOP"), stop=["STOP", "END"]
    )

    assert adapter.push(1) == "visible "
    assert adapter.finish() == ""


def test_unmatched_partial_stop_is_only_flushed_at_successful_end():
    adapter = IncrementalTextAdapter(prefix_decoder("answer EN"), stop=["END"])

    assert adapter.push(1) == "answer "
    assert not adapter.stopped
    assert adapter.finish() == "EN"


def test_unicode_byte_fallback_waits_for_complete_character():
    adapter = IncrementalTextAdapter(lambda ids: bytes(ids).decode("utf-8", errors="replace"))
    pieces = [adapter.push(token) for token in "中🙂 ".encode("utf-8")]

    assert pieces[:3] == ["", "", "中"]
    assert "".join(pieces) == "中🙂 "
    assert "\ufffd" not in "".join(pieces)
    assert adapter.finish() == ""


def test_cjk_and_newline_can_flush_without_spaces():
    adapter = IncrementalTextAdapter(prefix_decoder("你", "你好", "你好word", "你好word\n"))

    assert [adapter.push(token) for token in (1, 2, 3, 4)] == ["你", "好", "", "word\n"]
    assert adapter.finish() == ""


def test_stop_crossing_unicode_boundary_remains_hidden():
    adapter = IncrementalTextAdapter(prefix_decoder("内容结", "内容结束隐私"), stop=["结束"])

    assert adapter.push(1) == "内容"
    assert adapter.push(2) == ""
    assert adapter.finish() == ""


def test_unemitted_text_may_change_but_emitted_prefix_must_not():
    adapter = IncrementalTextAdapter(prefix_decoder("hel", "hello ", "hullo world"))

    assert adapter.push(1) == ""
    assert adapter.push(2) == "hello "
    with pytest.raises(RuntimeError, match="already emitted"):
        adapter.push(3)


def test_incomplete_final_unicode_raises_without_sending_replacement():
    adapter = IncrementalTextAdapter(lambda ids: bytes(ids).decode("utf-8", errors="replace"))

    assert adapter.push(0xE4) == ""
    with pytest.raises(RuntimeError, match="incomplete Unicode"):
        adapter.finish()
    assert adapter.text == ""


def test_eos_flushes_pending_word_and_is_not_visible():
    def decode(ids):
        return "word" if 1 in ids else ""

    adapter = IncrementalTextAdapter(decode, eos_token_ids=[99, 100])

    assert adapter.push(1) == ""
    assert adapter.push(99) == "word"
    assert adapter.stopped
    assert adapter.finish() == ""
    assert adapter.token_ids == (1, 99)


def test_empty_generation_can_finish():
    adapter = IncrementalTextAdapter(lambda ids: "")

    assert adapter.finish() == ""
    assert adapter.text == ""


def test_invalid_stop_configuration_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        IncrementalTextAdapter(lambda ids: "", stop=[""])


def test_decode_must_return_text():
    adapter = IncrementalTextAdapter(lambda ids: None)

    with pytest.raises(TypeError, match="return a string"):
        adapter.push(1)
