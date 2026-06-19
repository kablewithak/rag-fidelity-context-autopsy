from __future__ import annotations

import math

import pytest

from rag_lab.rerankers import CrossEncoderReranker, RerankerInputError
from rag_lab.schemas import (
    ChunkBoundaryQuality,
    ChunkingStrategy,
    EvaluationCase,
    FailureLabel,
    QueryType,
    RetrievedChunk,
    RetrievalMethod,
    RetrievalTrace,
    TextChunk,
)


class FixturePairScoringModel:
    name = "fixture:pair_scorer_v1"

    def __init__(self, scores_by_document: dict[str, float] | None = None, raw_scores: list[float] | None = None) -> None:
        self._scores_by_document = scores_by_document or {}
        self._raw_scores = raw_scores

    def score(self, *, query: str, documents: list[str]) -> list[float]:
        assert query == "Which clause governs termination?"
        if self._raw_scores is not None:
            return self._raw_scores
        return [self._scores_by_document[document] for document in documents]


def _case() -> EvaluationCase:
    return EvaluationCase(
        case_id="legal_termination_001",
        document_type="legal_clause",
        query_type=QueryType.POLICY_CLAUSE_QUERY,
        query="Which clause governs termination?",
        gold_evidence_text="Gold termination evidence.",
        gold_answer="The gold termination evidence governs termination.",
        expected_failure_mode=FailureLabel.RERANKER_NEEDED,
        source_doc_id="legal_terms",
        diagnostic_note="The first-stage trace deliberately ranks the complete evidence below a distractor.",
    )


def _chunk(chunk_id: str, source_doc_id: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        source_doc_id=source_doc_id,
        strategy=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
        chunk_index=0,
        text=text,
        token_count=len(text),
        char_count=len(text),
        source_char_start=0,
        source_char_end=len(text),
        boundary_quality=ChunkBoundaryQuality.CLEAN_SENTENCE_BOUNDARY,
    )


def _first_stage_trace(*, include_gold: bool = True) -> RetrievalTrace:
    distractor = _chunk("faq_sentence_000", "faq", "Distractor answer.")
    results = [
        RetrievedChunk(
            chunk=distractor,
            rank=1,
            score=0.92,
            gold_evidence_match=False,
        )
    ]
    if include_gold:
        gold = _chunk("legal_terms_sentence_001", "legal_terms", "Gold termination evidence.")
        results.append(
            RetrievedChunk(
                chunk=gold,
                rank=2,
                score=0.71,
                gold_evidence_match=True,
            )
        )

    return RetrievalTrace(
        case_id=_case().case_id,
        retriever_name=RetrievalMethod.BM25_OKAPI,
        lexical_analyzer_name="lexical:unicode_word_lowercase_v1",
        query=_case().query,
        requested_top_k=len(results),
        corpus_chunk_count=len(results),
        results=results,
        gold_evidence_found=include_gold,
        gold_evidence_rank=2 if include_gold else None,
    )


def test_cross_encoder_reranker_moves_gold_evidence_and_records_before_after_ranks() -> None:
    trace = CrossEncoderReranker(
        scoring_model=FixturePairScoringModel(
            scores_by_document={
                "Distractor answer.": 0.15,
                "Gold termination evidence.": 0.95,
            }
        )
    ).rerank(first_stage_trace=_first_stage_trace())

    assert trace.first_stage_retriever_name is RetrievalMethod.BM25_OKAPI
    assert trace.reranker_name == "cross_encoder"
    assert trace.reranker_model_name == "fixture:pair_scorer_v1"
    assert trace.candidate_count == 2
    assert trace.gold_evidence_found is True
    assert trace.gold_evidence_rank_before_rerank == 2
    assert trace.gold_evidence_rank_after_rerank == 1
    assert trace.results[0].chunk.chunk_id == "legal_terms_sentence_001"
    assert trace.results[0].first_stage_rank == 2
    assert trace.results[0].reranker_score == 0.95


def test_cross_encoder_reranker_cannot_invent_missing_gold_evidence() -> None:
    trace = CrossEncoderReranker(
        scoring_model=FixturePairScoringModel(scores_by_document={"Distractor answer.": 0.9})
    ).rerank(first_stage_trace=_first_stage_trace(include_gold=False))

    assert trace.gold_evidence_found is False
    assert trace.gold_evidence_rank_before_rerank is None
    assert trace.gold_evidence_rank_after_rerank is None
    assert trace.results[0].gold_evidence_match is False


def test_cross_encoder_reranker_tie_breaks_by_first_stage_rank_then_chunk_id() -> None:
    trace = CrossEncoderReranker(
        scoring_model=FixturePairScoringModel(raw_scores=[0.5, 0.5])
    ).rerank(first_stage_trace=_first_stage_trace())

    assert [result.chunk.chunk_id for result in trace.results] == [
        "faq_sentence_000",
        "legal_terms_sentence_001",
    ]


def test_cross_encoder_reranker_rejects_score_count_mismatch() -> None:
    reranker = CrossEncoderReranker(
        scoring_model=FixturePairScoringModel(raw_scores=[0.9])
    )

    with pytest.raises(RerankerInputError, match="score count"):
        reranker.rerank(first_stage_trace=_first_stage_trace())


def test_cross_encoder_reranker_rejects_non_finite_scores() -> None:
    reranker = CrossEncoderReranker(
        scoring_model=FixturePairScoringModel(raw_scores=[0.9, math.nan])
    )

    with pytest.raises(RerankerInputError, match="must be finite"):
        reranker.rerank(first_stage_trace=_first_stage_trace())


def test_cross_encoder_reranker_rejects_blank_model_name() -> None:
    class BlankNameScorer:
        name = "   "

        def score(self, *, query: str, documents: list[str]) -> list[float]:
            return [0.0 for _ in documents]

    with pytest.raises(RerankerInputError, match="scoring_model.name"):
        CrossEncoderReranker(scoring_model=BlankNameScorer())
