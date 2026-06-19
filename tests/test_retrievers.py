from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.chunkers import CharacterChunker, SentenceAwareTokenChunker
from rag_lab.corpus_loader import chunk_corpus, load_synthetic_corpus
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.retrievers import Bm25Retriever, DenseRetriever, HybridRetriever, RetrievalInputError
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
    chunk_a = _chunk("faq_character_000", "faq", "alpha")
    chunk_b = _chunk("legal_terms_character_000", "legal_terms", "alpha")
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
        _chunk("faq_character_000", "faq", "Distractor sentence."),
        _chunk("legal_terms_character_000", "legal_terms", "Gold evidence sentence."),
    ]


def test_dense_retriever_records_embedding_provenance_and_gold_evidence_rank() -> None:
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
    embedding_model = FixtureEmbeddingModel(
        {
            "Distractor sentence.": [1.0],
            "Gold evidence sentence.": [1.0],
        }
    )

    with pytest.raises(RetrievalInputError, match="invalid corpus embeddings"):
        DenseRetriever(chunks=_dense_chunks(), embedding_model=embedding_model)


def _hybrid_case() -> EvaluationCase:
    return _case("legal_confidentiality_003").model_copy(
        update={
            "query": "incident policy recovery",
            "gold_evidence_text": "Gold recovery evidence.",
        }
    )


def _hybrid_chunks() -> list[TextChunk]:
    return [
        _chunk("faq_character_000", "faq", "incident policy recovery details"),
        _chunk("legal_terms_character_000", "legal_terms", "Gold recovery evidence."),
        _chunk("pricing_table_character_000", "pricing_table", "secondary semantic distractor"),
        _chunk("support_policy_character_000", "support_policy", "third semantic distractor"),
    ]


def _hybrid_embedding_model() -> FixtureEmbeddingModel:
    return FixtureEmbeddingModel(
        {
            "incident policy recovery details": [0.0, 1.0],
            "Gold recovery evidence.": [1.0, 0.0],
            "secondary semantic distractor": [0.8, 0.2],
            "third semantic distractor": [0.7, 0.3],
            "incident policy recovery": [1.0, 0.0],
        }
    )


def test_hybrid_retriever_uses_rank_fusion_and_records_component_evidence() -> None:
    trace = HybridRetriever(
        chunks=_hybrid_chunks(),
        embedding_model=_hybrid_embedding_model(),
        rrf_k=60,
    ).retrieve(case=_hybrid_case(), top_k=4)

    assert trace.retriever_name == "hybrid"
    assert trace.lexical_analyzer_name == "lexical:unicode_word_lowercase_v1"
    assert trace.embedding_model_name == "fixture:deterministic_dense_v1"
    assert trace.embedding_dimension == 2
    assert trace.hybrid_fusion_method == "reciprocal_rank_fusion"
    assert trace.hybrid_rrf_k == 60
    assert trace.gold_evidence_found is True
    assert trace.gold_evidence_rank == 1

    gold = trace.results[0]
    assert gold.chunk.chunk_id == "legal_terms_character_000"
    assert gold.hybrid_score_breakdown is not None
    assert gold.hybrid_score_breakdown.bm25_rank == 2
    assert gold.hybrid_score_breakdown.dense_rank == 1
    assert gold.score == gold.hybrid_score_breakdown.fused_score


def test_hybrid_retriever_tie_breaks_by_chunk_id_after_fusion() -> None:
    chunks = list(reversed(_dense_chunks()))
    case = _dense_case().model_copy(update={"gold_evidence_text": "not present"})
    embedding_model = FixtureEmbeddingModel(
        {
            "Distractor sentence.": [1.0, 0.0],
            "Gold evidence sentence.": [1.0, 0.0],
            "semantic care query": [1.0, 0.0],
        }
    )

    trace = HybridRetriever(chunks=chunks, embedding_model=embedding_model).retrieve(
        case=case,
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in trace.results] == [
        "faq_character_000",
        "legal_terms_character_000",
    ]


def test_hybrid_retriever_rejects_non_positive_rrf_k() -> None:
    with pytest.raises(RetrievalInputError, match="rrf_k must be at least 1"):
        HybridRetriever(
            chunks=_hybrid_chunks(),
            embedding_model=_hybrid_embedding_model(),
            rrf_k=0,
        )


def _chunk(chunk_id: str, source_doc_id: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        source_doc_id=source_doc_id,
        strategy=ChunkingStrategy.CHARACTER,
        chunk_index=0,
        text=text,
        token_count=len(text),
        char_count=len(text),
        source_char_start=0,
        source_char_end=len(text),
        boundary_quality=ChunkBoundaryQuality.CHARACTER_CUT,
    )
