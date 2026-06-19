from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.chunkers import CharacterChunker, SentenceAwareTokenChunker
from rag_lab.corpus_loader import chunk_corpus, load_synthetic_corpus
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.retrievers import Bm25Retriever, RetrievalInputError
from rag_lab.schemas import (
    ChunkBoundaryQuality,
    ChunkingStrategy,
    EvaluationCase,
    TextChunk,
)
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIRECTORY = PROJECT_ROOT / "data" / "corpus"
EVAL_CASES_PATH = PROJECT_ROOT / "data" / "eval_cases.jsonl"


def _case(case_id: str) -> EvaluationCase:
    return next(case for case in load_evaluation_cases(EVAL_CASES_PATH) if case.case_id == case_id)


def _sentence_aware_chunks() -> list[TextChunk]:
    documents = load_synthetic_corpus(corpus_directory=CORPUS_DIRECTORY)
    chunker = SentenceAwareTokenChunker(
        token_counter=UnicodeCodePointTokenCounter(),
        max_tokens=500,
    )
    return chunk_corpus(documents, chunker=chunker)


def test_bm25_recovers_exact_error_code_evidence_with_a_typed_trace() -> None:
    trace = Bm25Retriever(chunks=_sentence_aware_chunks()).retrieve(
        case=_case("code_sso_error_013"),
        top_k=5,
    )

    assert trace.retriever_name == "bm25_okapi"
    assert trace.lexical_analyzer_name == "lexical:unicode_word_lowercase_v1"
    assert trace.gold_evidence_found is True
    assert trace.gold_evidence_rank == 1
    assert trace.results[0].chunk.source_doc_id == "code_logs"
    assert trace.results[0].gold_evidence_match is True


def test_bm25_trace_records_when_character_chunking_split_the_gold_evidence() -> None:
    documents = load_synthetic_corpus(corpus_directory=CORPUS_DIRECTORY)
    chunks = chunk_corpus(
        documents,
        chunker=CharacterChunker(
            token_counter=UnicodeCodePointTokenCounter(),
            max_characters=60,
        ),
    )

    trace = Bm25Retriever(chunks=chunks).retrieve(
        case=_case("legal_termination_001"),
        top_k=5,
    )

    assert trace.gold_evidence_found is False
    assert trace.gold_evidence_rank is None
    assert any(result.chunk.source_doc_id == "legal_terms" for result in trace.results)


def test_bm25_tie_breaks_by_chunk_id_for_repeatable_traces() -> None:
    chunk_a = TextChunk(
        chunk_id="faq_character_000",
        source_doc_id="faq",
        strategy=ChunkingStrategy.CHARACTER,
        chunk_index=0,
        text="alpha",
        token_count=5,
        char_count=5,
        source_char_start=0,
        source_char_end=5,
        boundary_quality=ChunkBoundaryQuality.CHARACTER_CUT,
    )
    chunk_b = TextChunk(
        chunk_id="legal_terms_character_000",
        source_doc_id="legal_terms",
        strategy=ChunkingStrategy.CHARACTER,
        chunk_index=0,
        text="alpha",
        token_count=5,
        char_count=5,
        source_char_start=0,
        source_char_end=5,
        boundary_quality=ChunkBoundaryQuality.CHARACTER_CUT,
    )
    case = _case("legal_payment_002").model_copy(
        update={
            "query": "alpha query",
            "gold_evidence_text": "not present in either chunk",
        }
    )

    trace = Bm25Retriever(chunks=[chunk_b, chunk_a]).retrieve(case=case, top_k=2)

    assert [result.chunk.chunk_id for result in trace.results] == [
        "faq_character_000",
        "legal_terms_character_000",
    ]


def test_bm25_rejects_queries_that_produce_no_lexical_terms() -> None:
    case = _case("legal_payment_002").model_copy(update={"query": "!!!!!!!!"})

    with pytest.raises(RetrievalInputError, match="produced no query terms"):
        Bm25Retriever(chunks=_sentence_aware_chunks()).retrieve(case=case)


class FixtureEmbeddingModel:
    """Deterministic test double; it is not represented as a semantic production model."""

    name = "fixture:deterministic_dense_v1"
    dimension = 2

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors_by_text[text] for text in texts]


def _dense_case() -> EvaluationCase:
    return _case("legal_confidentiality_003").model_copy(
        update={
            "query": "semantic care query",
            "gold_evidence_text": "Gold evidence sentence.",
        }
    )


def _dense_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            chunk_id="faq_character_000",
            source_doc_id="faq",
            strategy=ChunkingStrategy.CHARACTER,
            chunk_index=0,
            text="Distractor sentence.",
            token_count=20,
            char_count=20,
            source_char_start=0,
            source_char_end=20,
            boundary_quality=ChunkBoundaryQuality.CHARACTER_CUT,
        ),
        TextChunk(
            chunk_id="legal_terms_character_000",
            source_doc_id="legal_terms",
            strategy=ChunkingStrategy.CHARACTER,
            chunk_index=0,
            text="Gold evidence sentence.",
            token_count=23,
            char_count=23,
            source_char_start=0,
            source_char_end=23,
            boundary_quality=ChunkBoundaryQuality.CHARACTER_CUT,
        ),
    ]


def test_dense_retriever_records_embedding_provenance_and_gold_evidence_rank() -> None:
    from rag_lab.retrievers import DenseRetriever

    embedding_model = FixtureEmbeddingModel(
        {
            "Distractor sentence.": [0.0, 1.0],
            "Gold evidence sentence.": [1.0, 0.0],
            "semantic care query": [0.9, 0.1],
        }
    )

    trace = DenseRetriever(chunks=_dense_chunks(), embedding_model=embedding_model).retrieve(
        case=_dense_case(),
        top_k=2,
    )

    assert trace.retriever_name == "dense"
    assert trace.lexical_analyzer_name is None
    assert trace.embedding_model_name == "fixture:deterministic_dense_v1"
    assert trace.embedding_dimension == 2
    assert trace.gold_evidence_found is True
    assert trace.gold_evidence_rank == 1
    assert trace.results[0].chunk.chunk_id == "legal_terms_character_000"


def test_dense_retriever_tie_breaks_by_chunk_id_for_repeatable_traces() -> None:
    from rag_lab.retrievers import DenseRetriever

    chunks = list(reversed(_dense_chunks()))
    embedding_model = FixtureEmbeddingModel(
        {
            "Distractor sentence.": [1.0, 0.0],
            "Gold evidence sentence.": [1.0, 0.0],
            "semantic care query": [1.0, 0.0],
        }
    )

    trace = DenseRetriever(chunks=chunks, embedding_model=embedding_model).retrieve(
        case=_dense_case(),
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in trace.results] == [
        "faq_character_000",
        "legal_terms_character_000",
    ]


def test_dense_retriever_rejects_bad_model_vector_dimension() -> None:
    from rag_lab.retrievers import DenseRetriever

    embedding_model = FixtureEmbeddingModel(
        {
            "Distractor sentence.": [1.0],
            "Gold evidence sentence.": [1.0],
        }
    )

    with pytest.raises(RetrievalInputError, match="invalid corpus embeddings"):
        DenseRetriever(chunks=_dense_chunks(), embedding_model=embedding_model)
