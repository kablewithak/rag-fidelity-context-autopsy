"""Strict loading and chunk preparation for the synthetic RAG diagnostic corpus."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from rag_lab.chunkers import Chunker
from rag_lab.schemas import CorpusDocument, DocumentType, TextChunk


SYNTHETIC_CORPUS_MANIFEST: dict[str, DocumentType] = {
    "code_logs": DocumentType.CODE_LOG,
    "faq": DocumentType.FAQ,
    "legal_terms": DocumentType.LEGAL_CLAUSE,
    "multilingual_support": DocumentType.MULTILINGUAL_SUPPORT,
    "pricing_table": DocumentType.PRICING_TABLE,
    "support_policy": DocumentType.SUPPORT_POLICY,
}


class CorpusLoadError(ValueError):
    """Raised when the fixed synthetic corpus is missing, altered, or ambiguous."""


def load_synthetic_corpus(*, corpus_directory: Path) -> list[CorpusDocument]:
    """Load the known v1 corpus in deterministic source-document order.

    The lab intentionally rejects undeclared or missing files. Eval cases, corpus text, and
    source-document types must remain a fixed diagnostic asset rather than an accidental local
    directory scan.
    """

    if not corpus_directory.exists():
        raise CorpusLoadError(f"synthetic corpus directory does not exist: {corpus_directory}")
    if not corpus_directory.is_dir():
        raise CorpusLoadError(f"synthetic corpus path is not a directory: {corpus_directory}")

    discovered_doc_ids = {path.stem for path in corpus_directory.glob("*.txt")}
    expected_doc_ids = set(SYNTHETIC_CORPUS_MANIFEST)
    missing_doc_ids = sorted(expected_doc_ids - discovered_doc_ids)
    unexpected_doc_ids = sorted(discovered_doc_ids - expected_doc_ids)

    if missing_doc_ids or unexpected_doc_ids:
        parts: list[str] = []
        if missing_doc_ids:
            parts.append(f"missing corpus documents: {', '.join(missing_doc_ids)}")
        if unexpected_doc_ids:
            parts.append(f"unexpected corpus documents: {', '.join(unexpected_doc_ids)}")
        raise CorpusLoadError("; ".join(parts))

    documents: list[CorpusDocument] = []
    for source_doc_id in sorted(SYNTHETIC_CORPUS_MANIFEST):
        source_path = corpus_directory / f"{source_doc_id}.txt"
        text = source_path.read_text(encoding="utf-8").strip()
        if not text:
            raise CorpusLoadError(f"synthetic corpus document is empty: {source_path.name}")

        documents.append(
            CorpusDocument(
                source_doc_id=source_doc_id,
                document_type=SYNTHETIC_CORPUS_MANIFEST[source_doc_id],
                text=text,
                char_count=len(text),
                text_sha256=sha256(text.encode("utf-8")).hexdigest(),
            )
        )

    return documents


def chunk_corpus(
    documents: Iterable[CorpusDocument],
    *,
    chunker: Chunker,
) -> list[TextChunk]:
    """Apply one chunker to loaded documents and reject ambiguous chunk identities."""

    chunks: list[TextChunk] = []
    for document in documents:
        chunks.extend(chunker.chunk(text=document.text, source_doc_id=document.source_doc_id))

    if not chunks:
        raise CorpusLoadError("chunker produced no corpus chunks")

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise CorpusLoadError("chunker produced duplicate chunk identifiers")

    return chunks
