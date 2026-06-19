from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_lab.schemas import (
    ChunkBoundaryQuality,
    ChunkingStrategy,
    DocumentType,
    EvaluationCase,
    EvidenceLossStage,
    FailureDiagnosis,
    FailureLabel,
    HybridFusionMethod,
    HybridScoreBreakdown,
    QueryType,
    RetrievedChunk,
    RetrievalMethod,
    RetrievalTrace,
    TextChunk,
)


def valid_case_payload() -> dict[str, object]:
    return {
        "case_id": "valid_case_001",
        "document_type": DocumentType.FAQ,
        "query_type": QueryType.FAQ_QUERY,
        "query": "Where can an owner export workspace data?",
        "gold_evidence_text": "Workspace owners can export data from Settings > Data Management > Export.",
        "gold_answer": "Owners export data from Settings > Data Management > Export.",
        "expected_failure_mode": FailureLabel.DUPLICATE_CONTEXT_WASTE,
        "source_doc_id": "faq",
        "diagnostic_note": "This checks that the schema accepts a complete fixed diagnostic case.",
    }


def valid_chunk_payload() -> dict[str, object]:
    return {
        "chunk_id": "faq_character_000",
        "source_doc_id": "faq",
        "strategy": ChunkingStrategy.CHARACTER,
        "chunk_index": 0,
        "text": "A complete chunk.",
        "token_count": 17,
        "char_count": 17,
        "source_char_start": 0,
        "source_char_end": 17,
        "boundary_quality": ChunkBoundaryQuality.CHARACTER_CUT,
    }


def valid_retrieval_result() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=TextChunk.model_validate(valid_chunk_payload()),
        rank=1,
        score=1.5,
        gold_evidence_match=True,
    )


def valid_hybrid_result() -> RetrievedChunk:
    return valid_retrieval_result().model_copy(
        update={
            "score": 0.032,
            "hybrid_score_breakdown": HybridScoreBreakdown(
                bm25_rank=1,
                bm25_score=4.2,
                dense_rank=2,
                dense_score=0.73,
                fused_score=0.032,
            ),
        }
    )


def test_evaluation_case_accepts_valid_contract() -> None:
    case = EvaluationCase.model_validate(valid_case_payload())

    assert case.case_id == "valid_case_001"
    assert case.expected_failure_mode is FailureLabel.DUPLICATE_CONTEXT_WASTE


def test_evaluation_case_rejects_unknown_fields() -> None:
    payload = valid_case_payload()
    payload["untracked_field"] = "must fail"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvaluationCase.model_validate(payload)


def test_evaluation_case_rejects_non_snake_case_identifier() -> None:
    payload = valid_case_payload()
    payload["case_id"] = "Invalid Case ID"

    with pytest.raises(ValidationError, match="pattern"):
        EvaluationCase.model_validate(payload)


def test_text_chunk_rejects_source_span_that_does_not_match_char_count() -> None:
    payload = valid_chunk_payload()
    payload["source_char_end"] = 18

    with pytest.raises(ValidationError, match="source character span must equal char_count"):
        TextChunk.model_validate(payload)


def test_retrieval_trace_rejects_non_contiguous_ranks() -> None:
    result = valid_retrieval_result().model_copy(update={"rank": 2})

    with pytest.raises(ValidationError, match="ranks must be contiguous"):
        RetrievalTrace(
            case_id="valid_case_001",
            retriever_name=RetrievalMethod.BM25_OKAPI,
            lexical_analyzer_name="lexical:test_v1",
            query="Where can an owner export workspace data?",
            requested_top_k=2,
            corpus_chunk_count=2,
            results=[result],
            gold_evidence_found=True,
            gold_evidence_rank=2,
        )


def test_failure_diagnosis_rejects_duplicate_labels() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        FailureDiagnosis(
            case_id="valid_case_001",
            failure_labels=[
                FailureLabel.GOLD_EVIDENCE_SPLIT,
                FailureLabel.GOLD_EVIDENCE_SPLIT,
            ],
            loss_stage=EvidenceLossStage.CHUNKING,
            evidence_summary="The clause was split across a character boundary.",
            repair_recommendation="Use sentence-aware token chunking.",
        )


def test_dense_retrieval_trace_requires_embedding_metadata_and_excludes_lexical_metadata() -> None:
    result = valid_retrieval_result()

    with pytest.raises(ValidationError, match="dense traces require embedding model metadata"):
        RetrievalTrace(
            case_id="valid_case_001",
            retriever_name=RetrievalMethod.DENSE,
            query="Where can an owner export workspace data?",
            requested_top_k=1,
            corpus_chunk_count=1,
            results=[result],
            gold_evidence_found=True,
            gold_evidence_rank=1,
        )

    with pytest.raises(ValidationError, match="dense traces must not include lexical_analyzer_name"):
        RetrievalTrace(
            case_id="valid_case_001",
            retriever_name=RetrievalMethod.DENSE,
            lexical_analyzer_name="lexical:test_v1",
            embedding_model_name="fixture:test_dense_v1",
            embedding_dimension=2,
            query="Where can an owner export workspace data?",
            requested_top_k=1,
            corpus_chunk_count=1,
            results=[result],
            gold_evidence_found=True,
            gold_evidence_rank=1,
        )


def test_hybrid_trace_requires_fusion_metadata_and_component_breakdowns() -> None:
    result = valid_retrieval_result()

    with pytest.raises(ValidationError, match="require reciprocal_rank_fusion metadata"):
        RetrievalTrace(
            case_id="valid_case_001",
            retriever_name=RetrievalMethod.HYBRID,
            lexical_analyzer_name="lexical:test_v1",
            embedding_model_name="fixture:test_dense_v1",
            embedding_dimension=2,
            query="Where can an owner export workspace data?",
            requested_top_k=1,
            corpus_chunk_count=1,
            results=[result],
            gold_evidence_found=True,
            gold_evidence_rank=1,
        )

    with pytest.raises(ValidationError, match="hybrid results require hybrid score breakdowns"):
        RetrievalTrace(
            case_id="valid_case_001",
            retriever_name=RetrievalMethod.HYBRID,
            lexical_analyzer_name="lexical:test_v1",
            embedding_model_name="fixture:test_dense_v1",
            embedding_dimension=2,
            hybrid_fusion_method=HybridFusionMethod.RECIPROCAL_RANK_FUSION,
            hybrid_rrf_k=60,
            query="Where can an owner export workspace data?",
            requested_top_k=1,
            corpus_chunk_count=1,
            results=[result],
            gold_evidence_found=True,
            gold_evidence_rank=1,
        )


def test_hybrid_trace_rejects_mismatched_result_and_fused_scores() -> None:
    result = valid_hybrid_result().model_copy(update={"score": 0.031})

    with pytest.raises(ValidationError, match="must equal the fused score"):
        RetrievalTrace(
            case_id="valid_case_001",
            retriever_name=RetrievalMethod.HYBRID,
            lexical_analyzer_name="lexical:test_v1",
            embedding_model_name="fixture:test_dense_v1",
            embedding_dimension=2,
            hybrid_fusion_method=HybridFusionMethod.RECIPROCAL_RANK_FUSION,
            hybrid_rrf_k=60,
            query="Where can an owner export workspace data?",
            requested_top_k=1,
            corpus_chunk_count=1,
            results=[result],
            gold_evidence_found=True,
            gold_evidence_rank=1,
        )


def test_hybrid_score_breakdown_requires_rank_score_pairs() -> None:
    with pytest.raises(ValidationError, match="bm25_rank and bm25_score must be set together"):
        HybridScoreBreakdown(
            bm25_rank=1,
            dense_rank=1,
            dense_score=0.9,
            fused_score=0.03,
        )


def test_reranking_trace_rejects_candidate_that_does_not_match_first_stage_score() -> None:
    from rag_lab.schemas import RerankedChunk, RerankerMethod, RerankingTrace

    first_stage = RetrievalTrace(
        case_id="valid_case_001",
        retriever_name=RetrievalMethod.BM25_OKAPI,
        lexical_analyzer_name="lexical:test_v1",
        query="Where can an owner export workspace data?",
        requested_top_k=1,
        corpus_chunk_count=1,
        results=[valid_retrieval_result()],
        gold_evidence_found=True,
        gold_evidence_rank=1,
    )

    with pytest.raises(ValidationError, match="first_stage_score must match"):
        RerankingTrace(
            case_id="valid_case_001",
            first_stage_retriever_name=RetrievalMethod.BM25_OKAPI,
            first_stage_trace=first_stage,
            reranker_name=RerankerMethod.CROSS_ENCODER,
            reranker_model_name="fixture:cross_encoder_v1",
            candidate_count=1,
            results=[
                RerankedChunk(
                    chunk=first_stage.results[0].chunk,
                    rank=1,
                    first_stage_rank=1,
                    first_stage_score=9.9,
                    reranker_score=0.8,
                    gold_evidence_match=True,
                )
            ],
            gold_evidence_found=True,
            gold_evidence_rank_before_rerank=1,
            gold_evidence_rank_after_rerank=1,
        )
