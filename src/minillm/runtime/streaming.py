"""Model-free, append-only text adaptation for continuous token streams.

The caller owns the generation-length limit and the tokenizer.  ``decode`` must
decode the *whole* generated token prefix, normally with special tokens skipped.
Keeping that prefix is intentional: byte-fallback tokenizers cannot safely be
decoded one token at a time.  This object is not a worker or a producer thread.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

__all__ = ["IncrementalTextAdapter"]


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return any(
        start <= codepoint <= end
        for start, end in (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
            (0x2F800, 0x2FA1F),
        )
    )


class IncrementalTextAdapter:
    """Turn token IDs into stable text without leaking a stop-string prefix.

    ``push`` emits complete words/whitespace, newlines, or CJK text, retaining
    unfinished Unicode and any suffix that may become a configured stop string.
    ``finish`` flushes a final incomplete word or unmatched stop prefix.  A
    literal U+FFFD is conservatively treated as undecodable Unicode: it is held
    during generation and raises at termination rather than sending corrupted
    text.  Rewriting an already emitted prefix also raises explicitly.

    Repeated ``finish`` calls are safe.  Tokens supplied after a stop/EOS but
    before ``finish`` are ignored; pushing after ``finish`` is an API error.
    The caller must finish only successful generations, not failed/cancelled
    ones, and translate adapter errors into its transport's error semantics.
    """

    def __init__(
        self,
        decode: Callable[[list[int]], str],
        stop: Sequence[str] | None = None,
        eos_token_ids: Iterable[int] | int | None = None,
    ) -> None:
        self._decode = decode
        self._stop = (stop,) if isinstance(stop, str) else tuple(stop or ())
        if any(not isinstance(value, str) or not value for value in self._stop):
            raise ValueError("stop strings must be non-empty strings")
        if isinstance(eos_token_ids, int):
            eos_token_ids = (eos_token_ids,)
        self._eos = frozenset(eos_token_ids or ())
        self._tokens: list[int] = []
        self._emitted = ""
        self._stopped = False
        self._closed = False

    @property
    def stopped(self) -> bool:
        """Whether a configured stop string or EOS token has been observed."""
        return self._stopped

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Accepted IDs, including the token that triggered stop/EOS."""
        return tuple(self._tokens)

    @property
    def text(self) -> str:
        """The immutable text prefix already returned to the consumer."""
        return self._emitted

    def _visible_text(self) -> str:
        text = self._decode(list(self._tokens))
        if not isinstance(text, str):
            raise TypeError("decode must return a string")
        stop_positions = [text.find(value) for value in self._stop]
        matched_positions = [position for position in stop_positions if position >= 0]
        if matched_positions:
            self._stopped = True
            text = text[: min(matched_positions)]
        if not text.startswith(self._emitted):
            raise RuntimeError("tokenizer rewrote already emitted streaming text")
        return text

    def _emit(self, text: str) -> str:
        delta = text[len(self._emitted) :]
        self._emitted = text
        return delta

    def _flush(self, text: str) -> str:
        if "\ufffd" in text:
            raise RuntimeError("incomplete Unicode at the end of streaming text")
        return self._emit(text)

    def push(self, token_id: int) -> str:
        """Accept one generated token and return a safe, append-only delta."""
        if self._closed:
            raise RuntimeError("cannot append a token after finish")
        if self._stopped:
            return ""
        self._tokens.append(token_id)
        text = self._visible_text()
        if token_id in self._eos:
            self._stopped = True
        if self._stopped:
            return self._flush(text)

        safe_end = text.find("\ufffd")
        if safe_end < 0:
            safe_end = len(text)
        # A stop sequence may start before a space or CJK boundary.  Hold its
        # complete prefix before selecting a printable word boundary.
        for value in self._stop:
            for size in range(1, min(len(value), len(text)) + 1):
                if text.endswith(value[:size]):
                    safe_end = min(safe_end, len(text) - size)
        safe_text = text[:safe_end]
        if safe_text and (safe_text.endswith("\n") or _is_cjk(safe_text[-1])):
            printable_end = safe_end
        else:
            printable_end = next(
                (index + 1 for index in range(len(safe_text) - 1, -1, -1) if safe_text[index].isspace()),
                0,
            )
        # A preceding CJK flush need not contain a whitespace boundary.  Never
        # shorten that already emitted prefix while buffering a following word.
        printable_end = max(len(self._emitted), printable_end)
        return self._emit(text[:printable_end])

    def finish(self) -> str:
        """Flush successful final text; repeated calls return an empty delta."""
        if self._closed:
            return ""
        text = self._visible_text()
        delta = self._flush(text)
        self._closed = True
        return delta
