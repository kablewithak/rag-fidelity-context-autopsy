from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_lab.comparison import PipelineId
from rag_lab.comparison_runner import (
    ComparisonExecutionConfig,
    ComparisonExecutionError,
    FourPipelineComparisonRunner,
)
from rag_lab.schemas import (
    CorpusDocument,
    DocumentType,
    EvaluationCase,
    EvidenceLossStage,
    FailureLabel,
    QueryType,
)
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


class FixtureEmbeddingModel:
    """Small deterministic embedding seam used without network or model downloads."""

    name = "fixture:comparison_embedding_v1"
    dimension = 3

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        normalized = text.lower()
        if "alpha" in normalized:
            return [1.0, 0.0, 0.0]
        if "beta" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class FixtureReranker:
    """Query-term scorer that preserves a deterministic local test seam."""

    name = "fixture:comparison_reranker_v1"

    def score(self, *, query: str, documents: Sequence[str]) -> list[float]:
        query_terms = set(query.lower().split())
        return [float(len(query_terms & set(document.lower().split()))) for document in documents]


def _document(*, source_doc_id: str, text: str) -> CorpusDocument:
    return CorpusDocument(
        source_doc_id=source_doc_id,
        document_type=DocumentType.FAQ,
        text=text,
        char_count=len(text),
        text_sha256="a" * 64,
    )


def _case(*, case_id: str, source_doc_id: str, query: str, gold: str) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        document_type=DocumentType.FAQ,
        query_type=QueryType.FAQ_QUERY,
        query=query,
        gold_evidence_text=gold,
        gold_answer=gold,
        expected_failure_mode=FailureLabel.RELEVANT_CHUNK_RANKED_TOO_LOW,
        source_doc_id=source_doc_id,
        diagnostic_note="A deterministic fixture case for execution-runner contract coverage.",
    )


def _runner(*, character_max_characters: int = 200) -> FourPipelineComparisonRunner:
    return FourPipelineComparisonRunner(
        token_counter=UnicodeCodePointTokenCounter(),
        embedding_model=FixtureEmbeddingModel(),
        reranker_scoring_model=FixtureReranker(),
        config=ComparisonExecutionConfig(
            character_max_characters=character_max_characters,
            sentence_aware_max_tokens=200,
            budgeted_render_profile="compact_citation",
        ),
    )


def test_runner_executes_all_fixed_pipelines_and_returns_bounded_trace_references() -> None:
    alpha_gold = "Alpha evidence is retained."
    beta_gold = "Beta evidence is retained."
    documents = [
        _document(source_doc_id="alpha_doc", text=alpha_gold),
        _document(source_doc_id="beta_doc", text=beta_gold),
        _document(source_doc_id="other_doc", text="Unrelated fallback evidence."),
    ]
    cases = [
        _case(
            case_id="alpha_case_001",
            source_doc_id="alpha_doc",
            query="alpha question",
            gold=alpha_gold,
        ),
        _case(
            case_id="beta_case_002",
            source_doc_id="beta_doc",
            query="beta question",
            gold=beta_gold,
        ),
    ]

    result = _runner().run(run_id="fixture_run_001", cases=cases, documents=documents)

    assert len(result.report.case_outcomes) == 8
    assert {outcome.pipeline_id for outcome in result.report.case_outcomes} == set(PipelineId)
    assert all(outcome.gold_evidence_included for outcome in result.report.case_outcomes)
    assert all(reference.trace_id.startswith("retrieval:") for outcome in result.report.case_outcomes if outcome.pipeline_id is not PipelineId.TOKEN_HYBRID_RERANK_BUDGETED for reference in outcome.trace_references)
    budgeted_outcomes = [
        outcome
        for outcome in result.report.case_outcomes
        if outcome.pipeline_id is PipelineId.TOKEN_HYBRID_RERANK_BUDGETED
    ]
    assert all(len(outcome.trace_references) == 3 for outcome in budgeted_outcomes)
    serialized = result.report.model_dump_json()
    assert alpha_gold not in serialized
    assert beta_gold not in serialized


def test_runner_classifies_split_character_evidence_as_chunking_loss() -> None:
    gold = "Alpha evidence must survive the full clause boundary."
    documents = [
        _document(source_doc_id="alpha_doc", text=f"Prefix. {gold} Suffix."),
        _document(source_doc_id="other_doc", text="Unrelated fallback evidence."),
    ]
    cases = [
        _case(
            case_id="alpha_case_001",
            source_doc_id="alpha_doc",
            query="alpha evidence",
            gold=gold,
        )
    ]

    result = _runner(character_max_characters=18).run(
        run_id="fixture_run_002",
        cases=cases,
        documents=documents,
    )
    char_outcome = next(
        outcome
        for outcome in result.report.case_outcomes
        if outcome.pipeline_id is PipelineId.CHAR_DENSE_NAIVE
    )
    token_outcome = next(
        outcome
        for outcome in result.report.case_outcomes
        if outcome.pipeline_id is PipelineId.TOKEN_DENSE_NAIVE
    )

    assert char_outcome.loss_stage is EvidenceLossStage.CHUNKING
    assert char_outcome.failure_labels == [
        FailureLabel.BAD_CHUNK_BOUNDARY,
        FailureLabel.GOLD_EVIDENCE_SPLIT,
    ]
    assert token_outcome.gold_evidence_included is True


def test_runner_classifies_hybrid_candidate_miss_with_generic_retrieval_label() -> None:
    gold = "Alpha evidence is retained."
    documents = [
        _document(source_doc_id="alpha_doc", text=gold),
        *[
            _document(source_doc_id=f"other_{index:02d}", text=f"Beta distractor fallback {index}.")
            for index in range(10)
        ],
    ]
    cases = [
        _case(
            case_id="alpha_case_001",
            source_doc_id="alpha_doc",
            query="beta question",
            gold=gold,
        )
    ]

    result = _runner().run(run_id="fixture_run_003", cases=cases, documents=documents)
    hybrid_outcome = next(
        outcome
        for outcome in result.report.case_outcomes
        if outcome.pipeline_id is PipelineId.TOKEN_HYBRID_NAIVE
    )

    assert hybrid_outcome.loss_stage is EvidenceLossStage.RETRIEVAL
    assert hybrid_outcome.failure_labels == [FailureLabel.RETRIEVAL_MISS]


def test_runner_rejects_duplicate_case_ids_before_model_execution() -> None:
    gold = "Alpha evidence is retained."
    documents = [_document(source_doc_id="alpha_doc", text=gold)]
    duplicate_case = _case(
        case_id="alpha_case_001",
        source_doc_id="alpha_doc",
        query="alpha question",
        gold=gold,
    )

    with pytest.raises(ComparisonExecutionError, match="unique evaluation case IDs"):
        _runner().run(
            run_id="fixture_run_004",
            cases=[duplicate_case, duplicate_case],
            documents=documents,
        )


def test_runner_rejects_metric_cutoff_above_fixed_candidate_pool() -> None:
    with pytest.raises(ComparisonExecutionError, match="retrieval_top_k must be at least retrieval_metric_k"):
        FourPipelineComparisonRunner(
            token_counter=UnicodeCodePointTokenCounter(),
            embedding_model=FixtureEmbeddingModel(),
            reranker_scoring_model=FixtureReranker(),
            config=ComparisonExecutionConfig(
                sentence_aware_max_tokens=200,
                retrieval_metric_k=9,
            ),
        )
