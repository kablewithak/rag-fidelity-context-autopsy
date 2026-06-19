"""Deterministic character, token-window, and sentence-aware token chunkers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import fmean
from typing import Protocol

from rag_lab.schemas import (
    ChunkBoundaryQuality,
    ChunkingReport,
    ChunkingStrategy,
    TextChunk,
)
from rag_lab.tokenizers import TokenCounter


class ChunkingInputError(ValueError):
    """Raised when a chunking request cannot produce trustworthy chunk boundaries."""


class Chunker(Protocol):
    """Minimal chunker contract shared by corpus preparation and later pipelines."""

    strategy: ChunkingStrategy

    def chunk(self, *, text: str, source_doc_id: str) -> list[TextChunk]:
        """Return deterministic chunks for one source document."""


@dataclass(frozen=True, slots=True)
class _SentenceUnit:
    """One sentence-like, table-row, or log-event unit with original text offsets."""

    start: int
    end: int
    token_count: int


class CharacterChunker:
    """Baseline chunker that cuts fixed character windows without semantic awareness."""

    strategy = ChunkingStrategy.CHARACTER
    boundary_quality = ChunkBoundaryQuality.CHARACTER_CUT

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        max_characters: int = 700,
        overlap_characters: int = 0,
    ) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be at least 1")
        if overlap_characters < 0 or overlap_characters >= max_characters:
            raise ValueError("overlap_characters must be non-negative and smaller than max_characters")

        self._token_counter = token_counter
        self._max_characters = max_characters
        self._overlap_characters = overlap_characters

    def chunk(self, *, text: str, source_doc_id: str) -> list[TextChunk]:
        source_text = _normalise_source_text(text)
        chunks: list[TextChunk] = []
        start = 0
        chunk_index = 0

        while start < len(source_text):
            end = min(start + self._max_characters, len(source_text))
            chunk = _make_chunk(
                source_text=source_text,
                source_doc_id=source_doc_id,
                strategy=self.strategy,
                chunk_index=chunk_index,
                start=start,
                end=end,
                boundary_quality=self.boundary_quality,
                token_counter=self._token_counter,
            )
            chunks.append(chunk)

            if end == len(source_text):
                break

            start = end - self._overlap_characters
            chunk_index += 1

        return chunks


class TokenWindowChunker:
    """Fixed token-window chunker that exposes tokenizer-specific segmentation pressure."""

    strategy = ChunkingStrategy.TOKEN_WINDOW
    boundary_quality = ChunkBoundaryQuality.TOKEN_WINDOW

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        max_tokens: int = 256,
        overlap_tokens: int = 0,
    ) -> None:
        _validate_token_window(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self._token_counter = token_counter
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, *, text: str, source_doc_id: str) -> list[TextChunk]:
        source_text = _normalise_source_text(text)
        token_ids = self._token_counter.encode(source_text)
        return _chunk_token_windows(
            source_text=source_text,
            source_doc_id=source_doc_id,
            strategy=self.strategy,
            token_ids=token_ids,
            token_counter=self._token_counter,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
            boundary_quality=self.boundary_quality,
        )


class SentenceAwareTokenChunker:
    """Token-bounded chunker that keeps sentence, table-row, and log-event units intact."""

    strategy = ChunkingStrategy.SENTENCE_AWARE_TOKEN

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        max_tokens: int = 256,
        overlap_tokens: int = 0,
    ) -> None:
        _validate_token_window(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self._token_counter = token_counter
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, *, text: str, source_doc_id: str) -> list[TextChunk]:
        source_text = _normalise_source_text(text)
        units = _sentence_like_units(source_text, self._token_counter)
        chunks: list[TextChunk] = []
        current_units: list[_SentenceUnit] = []
        current_token_count = 0

        for unit in units:
            if unit.token_count > self._max_tokens:
                chunks.extend(
                    self._flush_current(
                        source_text=source_text,
                        source_doc_id=source_doc_id,
                        chunks=chunks,
                        current_units=current_units,
                    )
                )
                current_units = []
                current_token_count = 0

                unit_token_ids = self._token_counter.encode(source_text[unit.start : unit.end])
                fallback_chunks = _chunk_token_windows(
                    source_text=source_text[unit.start : unit.end],
                    source_doc_id=source_doc_id,
                    strategy=self.strategy,
                    token_ids=unit_token_ids,
                    token_counter=self._token_counter,
                    max_tokens=self._max_tokens,
                    overlap_tokens=self._overlap_tokens,
                    boundary_quality=ChunkBoundaryQuality.TOKEN_WINDOW,
                    source_offset=unit.start,
                    start_index=len(chunks),
                )
                chunks.extend(fallback_chunks)
                continue

            if current_units and current_token_count + unit.token_count > self._max_tokens:
                chunks.extend(
                    self._flush_current(
                        source_text=source_text,
                        source_doc_id=source_doc_id,
                        chunks=chunks,
                        current_units=current_units,
                    )
                )
                current_units = _tail_units_within_token_overlap(
                    current_units,
                    overlap_tokens=self._overlap_tokens,
                )
                current_token_count = sum(item.token_count for item in current_units)

                while current_units and current_token_count + unit.token_count > self._max_tokens:
                    removed = current_units.pop(0)
                    current_token_count -= removed.token_count

            current_units.append(unit)
            current_token_count += unit.token_count

        chunks.extend(
            self._flush_current(
                source_text=source_text,
                source_doc_id=source_doc_id,
                chunks=chunks,
                current_units=current_units,
            )
        )
        return chunks

    def _flush_current(
        self,
        *,
        source_text: str,
        source_doc_id: str,
        chunks: list[TextChunk],
        current_units: list[_SentenceUnit],
    ) -> list[TextChunk]:
        if not current_units:
            return []

        return [
            _make_chunk(
                source_text=source_text,
                source_doc_id=source_doc_id,
                strategy=self.strategy,
                chunk_index=len(chunks),
                start=current_units[0].start,
                end=current_units[-1].end,
                boundary_quality=ChunkBoundaryQuality.CLEAN_SENTENCE_BOUNDARY,
                token_counter=self._token_counter,
            )
        ]


def build_chunking_report(
    *,
    source_text: str,
    source_doc_id: str,
    chunker_name: ChunkingStrategy,
    tokenizer_name: str,
    chunks: list[TextChunk],
    gold_evidence_text: str,
) -> ChunkingReport:
    """Create a traceable report showing whether a known evidence span survived chunking."""

    normalised_source = _normalise_source_text(source_text)
    if not chunks:
        raise ChunkingInputError("cannot build a chunking report without chunks")
    if not gold_evidence_text.strip():
        raise ChunkingInputError("gold_evidence_text must contain non-whitespace text")

    evidence_start = normalised_source.find(gold_evidence_text)
    if evidence_start < 0:
        raise ChunkingInputError("gold_evidence_text is not an exact substring of source_text")

    evidence_end = evidence_start + len(gold_evidence_text)
    evidence_preserved = any(
        chunk.source_char_start <= evidence_start and chunk.source_char_end >= evidence_end
        for chunk in chunks
    )
    overlapping_chunks = [
        chunk
        for chunk in chunks
        if chunk.source_char_start < evidence_end and chunk.source_char_end > evidence_start
    ]
    evidence_split = not evidence_preserved and len(overlapping_chunks) >= 2
    qualities = {chunk.boundary_quality for chunk in chunks}
    boundary_quality = qualities.pop() if len(qualities) == 1 else ChunkBoundaryQuality.MIXED

    return ChunkingReport(
        source_doc_id=source_doc_id,
        chunker_name=chunker_name,
        tokenizer_name=tokenizer_name,
        chunk_count=len(chunks),
        avg_token_count=round(fmean(chunk.token_count for chunk in chunks), 2),
        max_token_count=max(chunk.token_count for chunk in chunks),
        gold_evidence_preserved=evidence_preserved,
        gold_evidence_split=evidence_split,
        boundary_quality=boundary_quality,
        chunks=chunks,
    )


def _normalise_source_text(text: str) -> str:
    source_text = text.strip()
    if not source_text:
        raise ChunkingInputError("text must contain non-whitespace content")
    return source_text


def _validate_token_window(*, max_tokens: int, overlap_tokens: int) -> None:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")


def _chunk_token_windows(
    *,
    source_text: str,
    source_doc_id: str,
    strategy: ChunkingStrategy,
    token_ids: list[int],
    token_counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
    boundary_quality: ChunkBoundaryQuality,
    source_offset: int = 0,
    start_index: int = 0,
) -> list[TextChunk]:
    if not token_ids:
        return []

    chunks: list[TextChunk] = []
    step = max_tokens - overlap_tokens

    for token_start in range(0, len(token_ids), step):
        token_end = min(token_start + max_tokens, len(token_ids))
        char_start = len(token_counter.decode(token_ids[:token_start]))
        char_end = len(token_counter.decode(token_ids[:token_end]))
        chunks.append(
            _make_chunk(
                source_text=source_text,
                source_doc_id=source_doc_id,
                strategy=strategy,
                chunk_index=start_index + len(chunks),
                start=char_start,
                end=char_end,
                boundary_quality=boundary_quality,
                token_counter=token_counter,
                source_offset=source_offset,
            )
        )
        if token_end == len(token_ids):
            break

    return chunks


def _sentence_like_units(source_text: str, token_counter: TokenCounter) -> list[_SentenceUnit]:
    units: list[_SentenceUnit] = []
    sentence_boundary = re.compile(r"(?<=[.!?])\s+")

    for line_match in re.finditer(r"[^\n]+", source_text):
        line = line_match.group(0)
        line_offset = line_match.start()
        segment_start = 0

        for boundary_match in sentence_boundary.finditer(line):
            _append_sentence_unit(
                units=units,
                source_text=source_text,
                token_counter=token_counter,
                start=line_offset + segment_start,
                end=line_offset + boundary_match.start(),
            )
            segment_start = boundary_match.end()

        _append_sentence_unit(
            units=units,
            source_text=source_text,
            token_counter=token_counter,
            start=line_offset + segment_start,
            end=line_offset + len(line),
        )

    if not units:
        raise ChunkingInputError("text did not contain any sentence-like units")
    return units


def _append_sentence_unit(
    *,
    units: list[_SentenceUnit],
    source_text: str,
    token_counter: TokenCounter,
    start: int,
    end: int,
) -> None:
    text, normalised_start, normalised_end = _normalised_slice(source_text, start, end)
    if not text:
        return
    units.append(
        _SentenceUnit(
            start=normalised_start,
            end=normalised_end,
            token_count=token_counter.count(text),
        )
    )


def _tail_units_within_token_overlap(
    units: list[_SentenceUnit],
    *,
    overlap_tokens: int,
) -> list[_SentenceUnit]:
    if overlap_tokens == 0:
        return []

    selected: list[_SentenceUnit] = []
    retained_tokens = 0
    for unit in reversed(units):
        if retained_tokens + unit.token_count > overlap_tokens:
            break
        selected.append(unit)
        retained_tokens += unit.token_count

    selected.reverse()
    return selected


def _make_chunk(
    *,
    source_text: str,
    source_doc_id: str,
    strategy: ChunkingStrategy,
    chunk_index: int,
    start: int,
    end: int,
    boundary_quality: ChunkBoundaryQuality,
    token_counter: TokenCounter,
    source_offset: int = 0,
) -> TextChunk:
    chunk_text, normalised_start, normalised_end = _normalised_slice(source_text, start, end)
    if not chunk_text:
        raise ChunkingInputError("chunk window normalized to empty text")

    return TextChunk(
        chunk_id=f"{source_doc_id}_{strategy.value}_{chunk_index:03d}",
        source_doc_id=source_doc_id,
        strategy=strategy,
        chunk_index=chunk_index,
        text=chunk_text,
        token_count=token_counter.count(chunk_text),
        char_count=len(chunk_text),
        source_char_start=source_offset + normalised_start,
        source_char_end=source_offset + normalised_end,
        boundary_quality=boundary_quality,
    )


def _normalised_slice(source_text: str, start: int, end: int) -> tuple[str, int, int]:
    raw_text = source_text[start:end]
    left_trimmed = len(raw_text) - len(raw_text.lstrip())
    right_trimmed = len(raw_text) - len(raw_text.rstrip())
    normalised_start = start + left_trimmed
    normalised_end = end - right_trimmed
    return raw_text.strip(), normalised_start, normalised_end
