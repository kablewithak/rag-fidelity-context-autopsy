from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.chunkers import (
    CharacterChunker,
    SentenceAwareTokenChunker,
    TokenWindowChunker,
    build_chunking_report,
)
from rag_lab.schemas import ChunkBoundaryQuality, ChunkingStrategy
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGAL_TERMS_PATH = PROJECT_ROOT / "data" / "corpus" / "legal_terms.txt"


def test_character_chunker_can_split_a_gold_evidence_span() -> None:
    counter = UnicodeCodePointTokenCounter()
    evidence = "The customer may terminate the agreement within 30 days if the provider fails to cure a material breach."
    source_text = f"Opening context. {evidence} Closing context."
    chunker = CharacterChunker(token_counter=counter, max_characters=42)

    chunks = chunker.chunk(text=source_text, source_doc_id="legal_terms")
    report = build_chunking_report(
        source_text=source_text,
        source_doc_id="legal_terms",
        chunker_name=chunker.strategy,
        tokenizer_name=counter.name,
        chunks=chunks,
        gold_evidence_text=evidence,
    )

    assert report.gold_evidence_preserved is False
    assert report.gold_evidence_split is True
    assert report.boundary_quality is ChunkBoundaryQuality.CHARACTER_CUT


def test_token_window_chunker_respects_its_token_budget() -> None:
    counter = UnicodeCodePointTokenCounter()
    source_text = LEGAL_TERMS_PATH.read_text(encoding="utf-8")
    chunker = TokenWindowChunker(token_counter=counter, max_tokens=18, overlap_tokens=4)

    chunks = chunker.chunk(text=source_text, source_doc_id="legal_terms")

    assert len(chunks) >= 2
    assert all(chunk.token_count <= 18 for chunk in chunks)
    assert all(chunk.boundary_quality is ChunkBoundaryQuality.TOKEN_WINDOW for chunk in chunks)


def test_sentence_aware_chunker_preserves_complete_sentence_evidence() -> None:
    counter = UnicodeCodePointTokenCounter()
    evidence = "The customer may terminate the agreement within 30 days if the provider fails to cure a material breach."
    source_text = f"A short opening sentence. {evidence} A short closing sentence."
    chunker = SentenceAwareTokenChunker(token_counter=counter, max_tokens=120)

    chunks = chunker.chunk(text=source_text, source_doc_id="legal_terms")
    report = build_chunking_report(
        source_text=source_text,
        source_doc_id="legal_terms",
        chunker_name=chunker.strategy,
        tokenizer_name=counter.name,
        chunks=chunks,
        gold_evidence_text=evidence,
    )

    assert report.gold_evidence_preserved is True
    assert report.gold_evidence_split is False
    assert report.boundary_quality is ChunkBoundaryQuality.CLEAN_SENTENCE_BOUNDARY
    assert any(chunk.text == evidence for chunk in chunks)


def test_sentence_aware_chunker_respects_token_budget_when_units_fit() -> None:
    counter = UnicodeCodePointTokenCounter()
    source_text = "One short sentence. Two short sentences. Three short sentences. Four short sentences."
    chunker = SentenceAwareTokenChunker(token_counter=counter, max_tokens=30, overlap_tokens=0)

    chunks = chunker.chunk(text=source_text, source_doc_id="faq")

    assert all(chunk.token_count <= 30 for chunk in chunks)
    assert all(chunk.boundary_quality is ChunkBoundaryQuality.CLEAN_SENTENCE_BOUNDARY for chunk in chunks)



def test_sentence_aware_chunker_counts_inter_sentence_whitespace_in_budget() -> None:
    """A combined span must include the separator token between sentence units."""

    counter = UnicodeCodePointTokenCounter()
    source_text = "Alpha. Beta."
    chunker = SentenceAwareTokenChunker(token_counter=counter, max_tokens=11)

    chunks = chunker.chunk(text=source_text, source_doc_id="faq")

    assert [chunk.text for chunk in chunks] == ["Alpha.", "Beta."]
    assert all(chunk.token_count <= 11 for chunk in chunks)

def test_chunking_report_rejects_gold_evidence_missing_from_source_text() -> None:
    counter = UnicodeCodePointTokenCounter()
    source_text = "A policy sentence exists here."
    chunker = CharacterChunker(token_counter=counter, max_characters=100)
    chunks = chunker.chunk(text=source_text, source_doc_id="support_policy")

    with pytest.raises(ValueError, match="not an exact substring"):
        build_chunking_report(
            source_text=source_text,
            source_doc_id="support_policy",
            chunker_name=ChunkingStrategy.CHARACTER,
            tokenizer_name=counter.name,
            chunks=chunks,
            gold_evidence_text="Missing evidence sentence.",
        )
