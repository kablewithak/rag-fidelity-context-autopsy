"""Tokenizer adapters used for token-aware chunking and later context budgets."""

from __future__ import annotations

from typing import Protocol


class TokenizationError(RuntimeError):
    """Raised when the configured tokenizer cannot encode or decode text safely."""


class TokenCounter(Protocol):
    """Small deterministic tokenizer contract used by chunkers and future budget planners."""

    @property
    def name(self) -> str:
        """Return a stable, reportable tokenizer identifier."""

    def encode(self, text: str) -> list[int]:
        """Encode text into tokenizer-specific token identifiers."""

    def decode(self, token_ids: list[int]) -> str:
        """Decode a sequence of tokenizer-specific token identifiers."""

    def count(self, text: str) -> int:
        """Return the number of tokens required for text under this tokenizer."""

    def token_char_offsets(self, text: str) -> list[int]:
        """Return one source-character offset per token plus the final text length."""


class UnicodeCodePointTokenCounter:
    """Offline deterministic tokenizer for tests and local diagnostic chunking.

    This intentionally uses Unicode code points, not an LLM provider vocabulary. It keeps
    chunking tests network-free and proves the token-budget interface before a model-specific
    tokenizer is selected. Reports expose this distinction through ``name``.
    """

    @property
    def name(self) -> str:
        return "diagnostic:unicode_codepoint_v1"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(chr(token_id) for token_id in token_ids)
        except ValueError as error:
            raise TokenizationError("token identifier is not a valid Unicode code point") from error

    def count(self, text: str) -> int:
        return len(text)

    def token_char_offsets(self, text: str) -> list[int]:
        return list(range(len(text) + 1))


class TiktokenTokenCounter:
    """Optional `tiktoken` adapter with an explicit encoding name for runtime reports.

    The adapter is deliberately lazy: Phase 1 tests do not need to download a tokenizer
    vocabulary or touch a network. Install the optional dependency and pre-warm the encoding
    cache before selecting this adapter in a demo or deployed runtime.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
            from requests import RequestException
        except ModuleNotFoundError as error:
            raise TokenizationError(
                "tiktoken adapter is not installed; run: python -m pip install -e \".[tiktoken]\""
            ) from error

        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except (OSError, RequestException, ValueError) as error:
            raise TokenizationError(
                "could not load the requested tiktoken encoding; check connectivity or pre-warm "
                "the tokenizer cache before runtime"
            ) from error

        self._encoding_name = encoding_name

    @property
    def name(self) -> str:
        return f"tiktoken:{self._encoding_name}"

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        try:
            return self._encoding.encode(text, disallowed_special=())
        except ValueError as error:
            raise TokenizationError("failed to encode text with configured tokenizer") from error

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self._encoding.decode(token_ids)
        except ValueError as error:
            raise TokenizationError("failed to decode token identifiers with configured tokenizer") from error

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def token_char_offsets(self, text: str) -> list[int]:
        token_ids = self.encode(text)
        if not token_ids:
            return [0]
        try:
            decoded_text, offsets = self._encoding.decode_with_offsets(token_ids)
        except ValueError as error:
            raise TokenizationError("failed to calculate tokenizer character offsets") from error
        if decoded_text != text:
            raise TokenizationError("tokenizer decode did not round-trip source text for offset tracing")
        return [*offsets, len(text)]
