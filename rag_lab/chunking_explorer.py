"""Read-only typed chunking views for the Streamlit RAG reliability demo.

This module distinguishes the standard benchmark-aligned chunk configuration from the
separate controlled boundary probe. The standard view reports what the fixed 700-character
configuration actually does; the probe deliberately places a character boundary inside a
known clause to isolate one chunk-boundary failure mechanism.

Neither surface invokes embeddings, retrieval, reranking, context assembly, or answer
generation. Both emit local UI view models only and never modify reviewed benchmark artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.chunkers import (
    CharacterChunker,
    SentenceAwareTokenChunker,
    build_chunking_report,
)
from rag_lab.corpus_loader import load_synthetic_corpus
from rag_lab.diagnostic_scenarios import BOUNDARY_CASE_ID
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.schemas import (
    ChunkingReport,
    ChunkingStrategy,
    CorpusDocument,
    EvaluationCase,
    TextChunk,
)
from rag_lab.tokenizers import TiktokenTokenCounter, TokenCounter


DEFAULT_CHARACTER_MAX_CHARACTERS = 700
DEFAULT_SENTENCE_AWARE_MAX_TOKENS = 96
DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"
CONTROLLED_BOUNDARY_WINDOW_SUFFIX_CHARACTERS = 42
CONTROLLED_BOUNDARY_SENTENCE_MAX_TOKENS = 200


class ChunkingExplorerError(ValueError):
    """Raised when fixed cases and corpus documents cannot form a safe chunking view."""


class ChunkingStrategyView(BaseModel):
    """One deterministic chunking result, including emitted synthetic chunks."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    strategy: ChunkingStrategy
    configured_limit: int = Field(ge=1, le=100_000)
    configured_limit_unit: str = Field(pattern=r"^(characters|tokens)$")
    report: ChunkingReport

    @model_validator(mode="after")
    def validate_report_alignment(self) -> "ChunkingStrategyView":
        if self.report.chunker_name is not self.strategy:
            raise ValueError("report.chunker_name must match strategy")
        expected_unit = "characters" if self.strategy is ChunkingStrategy.CHARACTER else "tokens"
        if self.configured_limit_unit != expected_unit:
            raise ValueError("configured_limit_unit must match strategy")
        return self


class ControlledBoundaryProbeView(BaseModel):
    """A deliberately positioned boundary probe, separate from benchmark execution."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    character_window_characters: int = Field(ge=1, le=100_000)
    sentence_aware_max_tokens: int = Field(ge=1, le=100_000)
    character_chunking: ChunkingStrategyView
    sentence_aware_token_chunking: ChunkingStrategyView

    @model_validator(mode="after")
    def validate_controlled_probe(self) -> "ControlledBoundaryProbeView":
        if self.character_chunking.strategy is not ChunkingStrategy.CHARACTER:
            raise ValueError("controlled character_chunking must use character strategy")
        if self.sentence_aware_token_chunking.strategy is not ChunkingStrategy.SENTENCE_AWARE_TOKEN:
            raise ValueError("controlled sentence_aware_token_chunking must use sentence_aware_token")
        if self.character_chunking.configured_limit != self.character_window_characters:
            raise ValueError("controlled character limit must match character_window_characters")
        if self.sentence_aware_token_chunking.configured_limit != self.sentence_aware_max_tokens:
            raise ValueError("controlled sentence limit must match sentence_aware_max_tokens")
        if not self.character_chunking.report.gold_evidence_split:
            raise ValueError("controlled character probe must split the configured gold evidence")
        if not self.sentence_aware_token_chunking.report.gold_evidence_preserved:
            raise ValueError("controlled sentence-aware probe must preserve the configured gold evidence")
        return self


class ChunkingCaseView(BaseModel):
    """One fixed case with standard chunking results and an optional controlled probe."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case: EvaluationCase
    source_char_count: int = Field(ge=1)
    gold_evidence_start: int = Field(ge=0)
    gold_evidence_end: int = Field(ge=1)
    character_chunking: ChunkingStrategyView
    sentence_aware_token_chunking: ChunkingStrategyView
    controlled_boundary_probe: ControlledBoundaryProbeView | None = None

    @model_validator(mode="after")
    def validate_case_alignment(self) -> "ChunkingCaseView":
        if self.gold_evidence_end <= self.gold_evidence_start:
            raise ValueError("gold_evidence_end must be greater than gold_evidence_start")
        if self.gold_evidence_end > self.source_char_count:
            raise ValueError("gold evidence span cannot exceed source_char_count")

        character = self.character_chunking
        sentence_aware = self.sentence_aware_token_chunking
        if character.strategy is not ChunkingStrategy.CHARACTER:
            raise ValueError("character_chunking must use character strategy")
        if sentence_aware.strategy is not ChunkingStrategy.SENTENCE_AWARE_TOKEN:
            raise ValueError("sentence_aware_token_chunking must use sentence_aware_token strategy")

        for strategy_view in (character, sentence_aware):
            if strategy_view.report.source_doc_id != self.case.source_doc_id:
                raise ValueError("chunking report source_doc_id must match case.source_doc_id")
            if strategy_view.report.gold_evidence_preserved and strategy_view.report.gold_evidence_split:
                raise ValueError("gold evidence cannot be both preserved and split")

        if self.controlled_boundary_probe is not None:
            if self.case.case_id != BOUNDARY_CASE_ID:
                raise ValueError("controlled_boundary_probe is reserved for the fixed boundary case")
            probe_reports = (
                self.controlled_boundary_probe.character_chunking.report,
                self.controlled_boundary_probe.sentence_aware_token_chunking.report,
            )
            if any(report.source_doc_id != self.case.source_doc_id for report in probe_reports):
                raise ValueError("controlled probe reports must match case.source_doc_id")
        return self


def load_chunking_case_views(*, project_root: Path) -> tuple[ChunkingCaseView, ...]:
    """Load fixed synthetic assets and build local views with the reviewed tokenizer."""

    cases = load_evaluation_cases(project_root / "data" / "eval_cases.jsonl")
    documents = load_synthetic_corpus(corpus_directory=project_root / "data" / "corpus")
    token_counter = TiktokenTokenCounter(encoding_name=DEFAULT_TIKTOKEN_ENCODING)
    return build_chunking_case_views(cases=cases, documents=documents, token_counter=token_counter)


def build_chunking_case_views(
    *,
    cases: Iterable[EvaluationCase],
    documents: Iterable[CorpusDocument],
    token_counter: TokenCounter,
    character_max_characters: int = DEFAULT_CHARACTER_MAX_CHARACTERS,
    sentence_aware_max_tokens: int = DEFAULT_SENTENCE_AWARE_MAX_TOKENS,
) -> tuple[ChunkingCaseView, ...]:
    """Build standard views without changing corpus, cases, or benchmark artifacts."""

    ordered_cases = tuple(sorted(cases, key=lambda case: case.case_id))
    if not ordered_cases:
        raise ChunkingExplorerError("at least one fixed evaluation case is required")

    case_ids = [case.case_id for case in ordered_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ChunkingExplorerError("fixed evaluation cases must not repeat case_id")

    ordered_documents = tuple(sorted(documents, key=lambda document: document.source_doc_id))
    if not ordered_documents:
        raise ChunkingExplorerError("at least one synthetic corpus document is required")
    documents_by_id = {document.source_doc_id: document for document in ordered_documents}
    if len(documents_by_id) != len(ordered_documents):
        raise ChunkingExplorerError("synthetic corpus documents must not repeat source_doc_id")

    missing_document_ids = sorted({case.source_doc_id for case in ordered_cases} - set(documents_by_id))
    if missing_document_ids:
        raise ChunkingExplorerError(
            "fixed evaluation cases reference missing corpus documents: " + ", ".join(missing_document_ids)
        )

    character_chunker = CharacterChunker(
        token_counter=token_counter,
        max_characters=character_max_characters,
    )
    sentence_aware_chunker = SentenceAwareTokenChunker(
        token_counter=token_counter,
        max_tokens=sentence_aware_max_tokens,
    )

    character_chunks_by_doc = {
        document.source_doc_id: character_chunker.chunk(text=document.text, source_doc_id=document.source_doc_id)
        for document in ordered_documents
    }
    sentence_chunks_by_doc = {
        document.source_doc_id: sentence_aware_chunker.chunk(text=document.text, source_doc_id=document.source_doc_id)
        for document in ordered_documents
    }

    views: list[ChunkingCaseView] = []
    for case in ordered_cases:
        document = documents_by_id[case.source_doc_id]
        evidence_start = document.text.find(case.gold_evidence_text)
        if evidence_start < 0:
            raise ChunkingExplorerError(
                f"{case.case_id} gold evidence is not an exact substring of {case.source_doc_id}"
            )
        evidence_end = evidence_start + len(case.gold_evidence_text)

        character_report = _build_report(
            source_text=document.text,
            source_doc_id=document.source_doc_id,
            chunker_name=ChunkingStrategy.CHARACTER,
            token_counter=token_counter,
            chunks=character_chunks_by_doc[document.source_doc_id],
            gold_evidence_text=case.gold_evidence_text,
        )
        sentence_report = _build_report(
            source_text=document.text,
            source_doc_id=document.source_doc_id,
            chunker_name=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
            token_counter=token_counter,
            chunks=sentence_chunks_by_doc[document.source_doc_id],
            gold_evidence_text=case.gold_evidence_text,
        )

        probe = None
        if case.case_id == BOUNDARY_CASE_ID:
            probe = _build_controlled_boundary_probe(
                source_text=document.text,
                source_doc_id=document.source_doc_id,
                gold_evidence_text=case.gold_evidence_text,
                evidence_start=evidence_start,
                token_counter=token_counter,
            )

        views.append(
            ChunkingCaseView(
                case=case,
                source_char_count=document.char_count,
                gold_evidence_start=evidence_start,
                gold_evidence_end=evidence_end,
                character_chunking=ChunkingStrategyView(
                    strategy=ChunkingStrategy.CHARACTER,
                    configured_limit=character_max_characters,
                    configured_limit_unit="characters",
                    report=character_report,
                ),
                sentence_aware_token_chunking=ChunkingStrategyView(
                    strategy=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
                    configured_limit=sentence_aware_max_tokens,
                    configured_limit_unit="tokens",
                    report=sentence_report,
                ),
                controlled_boundary_probe=probe,
            )
        )
    return tuple(views)


def _build_controlled_boundary_probe(
    *,
    source_text: str,
    source_doc_id: str,
    gold_evidence_text: str,
    evidence_start: int,
    token_counter: TokenCounter,
) -> ControlledBoundaryProbeView:
    """Reproduce the separate controlled boundary diagnostic used by the harness."""

    character_window = evidence_start + CONTROLLED_BOUNDARY_WINDOW_SUFFIX_CHARACTERS
    if character_window >= evidence_start + len(gold_evidence_text):
        raise ChunkingExplorerError("controlled boundary window must cut inside the gold evidence")

    character_chunks = CharacterChunker(
        token_counter=token_counter,
        max_characters=character_window,
    ).chunk(text=source_text, source_doc_id=source_doc_id)
    sentence_chunks = SentenceAwareTokenChunker(
        token_counter=token_counter,
        max_tokens=CONTROLLED_BOUNDARY_SENTENCE_MAX_TOKENS,
    ).chunk(text=source_text, source_doc_id=source_doc_id)

    return ControlledBoundaryProbeView(
        character_window_characters=character_window,
        sentence_aware_max_tokens=CONTROLLED_BOUNDARY_SENTENCE_MAX_TOKENS,
        character_chunking=ChunkingStrategyView(
            strategy=ChunkingStrategy.CHARACTER,
            configured_limit=character_window,
            configured_limit_unit="characters",
            report=_build_report(
                source_text=source_text,
                source_doc_id=source_doc_id,
                chunker_name=ChunkingStrategy.CHARACTER,
                token_counter=token_counter,
                chunks=character_chunks,
                gold_evidence_text=gold_evidence_text,
            ),
        ),
        sentence_aware_token_chunking=ChunkingStrategyView(
            strategy=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
            configured_limit=CONTROLLED_BOUNDARY_SENTENCE_MAX_TOKENS,
            configured_limit_unit="tokens",
            report=_build_report(
                source_text=source_text,
                source_doc_id=source_doc_id,
                chunker_name=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
                token_counter=token_counter,
                chunks=sentence_chunks,
                gold_evidence_text=gold_evidence_text,
            ),
        ),
    )


def _build_report(
    *,
    source_text: str,
    source_doc_id: str,
    chunker_name: ChunkingStrategy,
    token_counter: TokenCounter,
    chunks: list[TextChunk],
    gold_evidence_text: str,
) -> ChunkingReport:
    """Build one provenance-bound report from already-emitted deterministic chunks."""

    return build_chunking_report(
        source_text=source_text,
        source_doc_id=source_doc_id,
        chunker_name=chunker_name,
        tokenizer_name=token_counter.name,
        chunks=chunks,
        gold_evidence_text=gold_evidence_text,
    )
