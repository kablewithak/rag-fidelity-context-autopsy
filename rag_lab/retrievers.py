"""Deterministic lexical and dense first-stage retrieval for the RAG reliability lab."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from rag_lab.embedders import EmbeddingInputError, EmbeddingModel, validate_embedding_vectors
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
        self._chunks = _validate_chunks(chunks)

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
        """Return a typed first-stage lexical retrieval trace for one fixed evaluation case."""

        _validate_top_k(top_k)
        query_terms = self._lexical_analyzer.tokenize(case.query)
        if not query_terms:
            raise RetrievalInputError("lexical analyzer produced no query terms")

        scores = self._index.get_scores(query_terms)
        if len(scores) != len(self._chunks):
            raise RetrievalInputError("BM25 score count did not match the indexed chunk count")

        results = _build_ranked_results(
            chunks=self._chunks,
            scores=(float(score) for score in scores),
            case=case,
            top_k=top_k,
        )
        return _build_trace(
            case=case,
            method=self.method,
            results=results,
            requested_top_k=top_k,
            corpus_chunk_count=len(self._chunks),
            lexical_analyzer_name=self._lexical_analyzer.name,
        )


class DenseRetriever:
    """Cosine-similarity first-stage retriever over explicit fixed-dimension embeddings.

    The retriever does not know which provider produced vectors. Its trace records the embedder
    identity and dimension, while the embedding adapter owns model loading and text encoding.
    """

    method = RetrievalMethod.DENSE

    def __init__(
        self,
        *,
        chunks: Sequence[TextChunk],
        embedding_model: EmbeddingModel,
    ) -> None:
        self._chunks = _validate_chunks(chunks)
        self._embedding_model = embedding_model
        if embedding_model.dimension < 1:
            raise RetrievalInputError("embedding model dimension must be at least 1")

        try:
            raw_vectors = embedding_model.encode([chunk.text for chunk in self._chunks])
            self._chunk_vectors = validate_embedding_vectors(
                raw_vectors,
                expected_count=len(self._chunks),
                expected_dimension=embedding_model.dimension,
            )
        except EmbeddingInputError as error:
            raise RetrievalInputError(f"invalid corpus embeddings: {error}") from error

    def retrieve(self, *, case: EvaluationCase, top_k: int = 5) -> RetrievalTrace:
        """Return a typed cosine-similarity trace for one fixed evaluation case."""

        _validate_top_k(top_k)
        try:
            query_vectors = validate_embedding_vectors(
                self._embedding_model.encode([case.query]),
                expected_count=1,
                expected_dimension=self._embedding_model.dimension,
            )
        except EmbeddingInputError as error:
            raise RetrievalInputError(f"invalid query embedding: {error}") from error

        query_vector = query_vectors[0]
        scores = [
            _cosine_similarity(query_vector, chunk_vector)
            for chunk_vector in self._chunk_vectors
        ]
        results = _build_ranked_results(
            chunks=self._chunks,
            scores=scores,
            case=case,
            top_k=top_k,
        )
        return _build_trace(
            case=case,
            method=self.method,
            results=results,
            requested_top_k=top_k,
            corpus_chunk_count=len(self._chunks),
            embedding_model_name=self._embedding_model.name,
            embedding_dimension=self._embedding_model.dimension,
        )


def _validate_chunks(chunks: Sequence[TextChunk]) -> tuple[TextChunk, ...]:
    if not chunks:
        raise RetrievalInputError("retrieval requires at least one corpus chunk")

    materialized = tuple(chunks)
    chunk_ids = [chunk.chunk_id for chunk in materialized]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RetrievalInputError("retrieval requires unique chunk identifiers")
    return materialized


def _validate_top_k(top_k: int) -> None:
    if top_k < 1:
        raise RetrievalInputError("top_k must be at least 1")


def _build_ranked_results(
    *,
    chunks: Sequence[TextChunk],
    scores: Iterable[float],
    case: EvaluationCase,
    top_k: int,
) -> list[RetrievedChunk]:
    materialized_scores = list(scores)
    if len(materialized_scores) != len(chunks):
        raise RetrievalInputError("retrieval score count did not match the indexed chunk count")
    if any(not math.isfinite(score) for score in materialized_scores):
        raise RetrievalInputError("retrieval scores must be finite")

    ranked = sorted(
        zip(chunks, materialized_scores, strict=True),
        key=lambda item: (-item[1], item[0].chunk_id),
    )

    results: list[RetrievedChunk] = []
    for rank, (chunk, score) in enumerate(ranked[:top_k], start=1):
        results.append(
            RetrievedChunk(
                chunk=chunk,
                rank=rank,
                score=score,
                gold_evidence_match=case.gold_evidence_text in chunk.text,
            )
        )
    return results


def _build_trace(
    *,
    case: EvaluationCase,
    method: RetrievalMethod,
    results: list[RetrievedChunk],
    requested_top_k: int,
    corpus_chunk_count: int,
    lexical_analyzer_name: str | None = None,
    embedding_model_name: str | None = None,
    embedding_dimension: int | None = None,
) -> RetrievalTrace:
    matching_results = [result for result in results if result.gold_evidence_match]
    if len(matching_results) > 1:
        raise RetrievalInputError(
            "gold evidence matched multiple returned chunks; chunking overlap makes rank ambiguous"
        )

    gold_evidence_found = bool(matching_results)
    gold_evidence_rank = matching_results[0].rank if matching_results else None

    return RetrievalTrace(
        case_id=case.case_id,
        retriever_name=method,
        lexical_analyzer_name=lexical_analyzer_name,
        embedding_model_name=embedding_model_name,
        embedding_dimension=embedding_dimension,
        query=case.query,
        requested_top_k=requested_top_k,
        corpus_chunk_count=corpus_chunk_count,
        results=results,
        gold_evidence_found=gold_evidence_found,
        gold_evidence_rank=gold_evidence_rank,
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise RetrievalInputError("cosine similarity requires vectors with equal dimensions")

    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise RetrievalInputError("cosine similarity requires non-zero embedding vectors")
    return numerator / (left_norm * right_norm)
