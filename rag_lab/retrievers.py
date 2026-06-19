"""Deterministic first-stage lexical retrieval for the RAG reliability lab."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from rag_lab.schemas import (
    EvaluationCase,
    RetrievedChunk,
    RetrievalMethod,
    RetrievalTrace,
    TextChunk,
)


class RetrievalInputError(ValueError):
    """Raised when a retriever cannot construct a trustworthy result set."""


class LexicalAnalyzer(Protocol):
    """Explicit lexical normalization contract for BM25 indexing and querying."""

    @property
    def name(self) -> str:
        """Return the stable analyzer identifier written into retrieval traces."""

    def tokenize(self, text: str) -> list[str]:
        """Normalize one source string into lexical retrieval terms."""


class UnicodeWordLexicalAnalyzer:
    """Lowercase Unicode-aware word analyzer for local-first BM25 experiments.

    This is deliberately separate from the model tokenizer used for chunk boundaries and later
    context budgets. Lexical retrieval tokenization answers a different question: which exact
    normalized terms overlap between a query and a chunk?
    """

    _TOKEN_PATTERN = re.compile(r"(?u)\b[\w]+(?:[._/-][\w]+)*\b")

    @property
    def name(self) -> str:
        return "lexical:unicode_word_lowercase_v1"

    def tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in self._TOKEN_PATTERN.findall(text)]


class Bm25Retriever:
    """Immutable BM25 Okapi retriever with deterministic tie-breaking and typed traces."""

    method = RetrievalMethod.BM25_OKAPI

    def __init__(
        self,
        *,
        chunks: Sequence[TextChunk],
        lexical_analyzer: LexicalAnalyzer | None = None,
    ) -> None:
        if not chunks:
            raise RetrievalInputError("BM25 retrieval requires at least one corpus chunk")

        self._chunks = tuple(chunks)
        chunk_ids = [chunk.chunk_id for chunk in self._chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RetrievalInputError("BM25 retrieval requires unique chunk identifiers")

        self._lexical_analyzer = lexical_analyzer or UnicodeWordLexicalAnalyzer()
        tokenized_chunks = [self._lexical_analyzer.tokenize(chunk.text) for chunk in self._chunks]
        empty_chunk_ids = [
            chunk.chunk_id
            for chunk, tokens in zip(self._chunks, tokenized_chunks, strict=True)
            if not tokens
        ]
        if empty_chunk_ids:
            raise RetrievalInputError(
                "BM25 lexical analyzer produced no terms for chunks: " + ", ".join(empty_chunk_ids)
            )

        self._index = BM25Okapi(tokenized_chunks)

    def retrieve(self, *, case: EvaluationCase, top_k: int = 5) -> RetrievalTrace:
        """Return a typed first-stage trace for one fixed evaluation case."""

        if top_k < 1:
            raise RetrievalInputError("top_k must be at least 1")

        query_terms = self._lexical_analyzer.tokenize(case.query)
        if not query_terms:
            raise RetrievalInputError("lexical analyzer produced no query terms")

        scores = self._index.get_scores(query_terms)
        if len(scores) != len(self._chunks):
            raise RetrievalInputError("BM25 score count did not match the indexed chunk count")

        ranked = sorted(
            (
                (chunk, float(score))
                for chunk, score in zip(self._chunks, scores, strict=True)
            ),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        selected = ranked[:top_k]

        results: list[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(selected, start=1):
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    rank=rank,
                    score=score,
                    gold_evidence_match=case.gold_evidence_text in chunk.text,
                )
            )

        matching_results = [result for result in results if result.gold_evidence_match]
        if len(matching_results) > 1:
            raise RetrievalInputError(
                "gold evidence matched multiple returned chunks; chunking overlap makes rank ambiguous"
            )

        gold_evidence_found = bool(matching_results)
        gold_evidence_rank = matching_results[0].rank if matching_results else None

        return RetrievalTrace(
            case_id=case.case_id,
            retriever_name=self.method,
            lexical_analyzer_name=self._lexical_analyzer.name,
            query=case.query,
            requested_top_k=top_k,
            corpus_chunk_count=len(self._chunks),
            results=results,
            gold_evidence_found=gold_evidence_found,
            gold_evidence_rank=gold_evidence_rank,
        )
