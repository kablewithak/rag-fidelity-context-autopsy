from __future__ import annotations

import pytest

from rag_lab.chunkers import (
    ChunkingInputError,
    SentenceAwareTokenChunker,
    build_chunking_report,
)
from rag_lab.schemas import ChunkingStrategy
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


def test_chunker_records_tokenizer_provenance_on_every_emitted_chunk() -> None:
    counter = UnicodeCodePointTokenCounter()
    source_text = "A first sentence. A second sentence."
    chunks = SentenceAwareTokenChunker(
        token_counter=counter,
        max_tokens=100,
    ).chunk(text=source_text, source_doc_id="faq")

    assert chunks
    assert {chunk.tokenizer_name for chunk in chunks} == {counter.name}


def test_chunking_report_rejects_mismatched_chunk_tokenizer_provenance() -> None:
    counter = UnicodeCodePointTokenCounter()
    source_text = "A first sentence. A second sentence."
    chunks = SentenceAwareTokenChunker(
        token_counter=counter,
        max_tokens=100,
    ).chunk(text=source_text, source_doc_id="faq")

    with pytest.raises(ChunkingInputError, match="tokenizer provenance"):
        build_chunking_report(
            source_text=source_text,
            source_doc_id="faq",
            chunker_name=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
            tokenizer_name="tiktoken:cl100k_base",
            chunks=chunks,
            gold_evidence_text="A first sentence.",
        )
