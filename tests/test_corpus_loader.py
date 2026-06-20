from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.chunkers import SentenceAwareTokenChunker
from rag_lab.corpus_loader import CorpusLoadError, chunk_corpus, load_synthetic_corpus
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIRECTORY = PROJECT_ROOT / "data" / "corpus"


def test_load_synthetic_corpus_returns_all_manifest_documents_with_integrity_metadata() -> None:
    documents = load_synthetic_corpus(corpus_directory=CORPUS_DIRECTORY)

    assert [document.source_doc_id for document in documents] == [
        "code_logs",
        "faq",
        "legal_terms",
        "multilingual_support",
        "pricing_table",
        "support_policy",
        "tokenization_stress_policy",
    ]
    assert all(len(document.text_sha256) == 64 for document in documents)
    assert all(document.char_count == len(document.text) for document in documents)


def test_load_synthetic_corpus_rejects_undeclared_source_file(tmp_path: Path) -> None:
    for source_path in CORPUS_DIRECTORY.glob("*.txt"):
        (tmp_path / source_path.name).write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "unexpected_notes.txt").write_text("Synthetic but undeclared.", encoding="utf-8")

    with pytest.raises(CorpusLoadError, match="unexpected corpus documents: unexpected_notes"):
        load_synthetic_corpus(corpus_directory=tmp_path)


def test_chunk_corpus_produces_unique_chunk_identifiers() -> None:
    documents = load_synthetic_corpus(corpus_directory=CORPUS_DIRECTORY)
    chunker = SentenceAwareTokenChunker(
        token_counter=UnicodeCodePointTokenCounter(),
        max_tokens=500,
    )

    chunks = chunk_corpus(documents, chunker=chunker)

    assert chunks
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
