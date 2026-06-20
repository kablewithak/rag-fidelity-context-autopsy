from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.comparison import PipelineId
from rag_lab.comparison_artifacts import DEFAULT_BASELINE_ARTIFACT_PATH, load_baseline_artifact
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.retrieval_explorer import (
    CandidateSetState,
    RetrievalExplorerError,
    build_retrieval_case_views,
    load_retrieval_case_views,
)
from rag_lab.schemas import EvidenceLossStage, FailureLabel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _view(case_id: str):
    return next(
        view
        for view in load_retrieval_case_views(project_root=PROJECT_ROOT)
        if view.case.case_id == case_id
    )


def _pipeline(view, pipeline_id: PipelineId):
    return next(
        pipeline
        for pipeline in view.pipeline_views
        if pipeline.pipeline_id is pipeline_id
    )


def test_retrieval_explorer_loads_every_fixed_case_and_pipeline() -> None:
    views = load_retrieval_case_views(project_root=PROJECT_ROOT)

    assert len(views) == 18
    assert [view.case.case_id for view in views] == sorted(view.case.case_id for view in views)
    assert all({pipeline.pipeline_id for pipeline in view.pipeline_views} == set(PipelineId) for view in views)
    assert all(view.retrieval_metric_k == 5 for view in views)


def test_retrieval_explorer_distinguishes_dense_candidate_miss_from_hybrid_recovery() -> None:
    view = _view("multilingual_security_012")

    baseline = _pipeline(view, PipelineId.CHAR_DENSE_NAIVE)
    hybrid = _pipeline(view, PipelineId.TOKEN_HYBRID_NAIVE)

    assert baseline.candidate_set_state is CandidateSetState.MISSING_FROM_CANDIDATE_SET
    assert baseline.retrieved_gold_rank is None
    assert baseline.loss_stage is EvidenceLossStage.RETRIEVAL
    assert baseline.failure_labels == (FailureLabel.DENSE_RETRIEVAL_MISS,)

    assert hybrid.candidate_set_state is CandidateSetState.PRESENT
    assert hybrid.retrieved_gold_rank == 3
    assert hybrid.gold_evidence_included is True


def test_retrieval_explorer_distinguishes_candidate_presence_from_ranking_loss() -> None:
    view = _view("support_enterprise_s1_006")

    baseline = _pipeline(view, PipelineId.CHAR_DENSE_NAIVE)
    hybrid = _pipeline(view, PipelineId.TOKEN_HYBRID_NAIVE)

    assert baseline.candidate_set_state is CandidateSetState.PRESENT
    assert baseline.retrieved_gold_rank == baseline.rank_used_for_context == 4
    assert baseline.gold_evidence_included is False
    assert baseline.loss_stage is EvidenceLossStage.RANKING
    assert baseline.failure_labels == (FailureLabel.RELEVANT_CHUNK_RANKED_TOO_LOW,)

    assert hybrid.retrieved_gold_rank == 1
    assert hybrid.gold_evidence_included is True


def test_retrieval_explorer_uses_reranked_order_only_for_reranker_pipeline() -> None:
    view = _view("support_enterprise_s1_006")

    no_reranker = _pipeline(view, PipelineId.TOKEN_HYBRID_NAIVE)
    reranked = _pipeline(view, PipelineId.TOKEN_HYBRID_RERANK_BUDGETED)

    assert no_reranker.reranker_enabled is False
    assert no_reranker.reranked_gold_rank is None
    assert no_reranker.rank_used_for_context == no_reranker.retrieved_gold_rank

    assert reranked.reranker_enabled is True
    assert reranked.candidate_set_state is CandidateSetState.PRESENT
    assert reranked.reranked_gold_rank is not None
    assert reranked.rank_used_for_context == reranked.reranked_gold_rank


def test_retrieval_explorer_rejects_case_and_artifact_coverage_drift() -> None:
    artifact = load_baseline_artifact(PROJECT_ROOT / DEFAULT_BASELINE_ARTIFACT_PATH)
    cases = load_evaluation_cases(PROJECT_ROOT / "data" / "eval_cases.jsonl")

    with pytest.raises(RetrievalExplorerError, match="coverage differ"):
        build_retrieval_case_views(cases=cases[:-1], artifact=artifact)
