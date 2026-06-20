from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.case_explorer import (
    CaseExplorerError,
    build_failure_case_views,
    load_failure_case_views,
)
from rag_lab.comparison import PipelineId
from rag_lab.comparison_artifacts import DEFAULT_BASELINE_ARTIFACT_PATH, load_baseline_artifact
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.schemas import EvidenceLossStage, FailureLabel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_failure_case_explorer_loads_every_fixed_case_and_pipeline() -> None:
    views = load_failure_case_views(project_root=PROJECT_ROOT)

    assert len(views) == 18
    assert [view.case.case_id for view in views] == sorted(
        view.case.case_id for view in views
    )
    assert all(len(view.pipeline_statuses) == 4 for view in views)
    assert all(
        {status.pipeline_id for status in view.pipeline_statuses} == set(PipelineId)
        for view in views
    )


def test_failure_case_explorer_surfaces_known_baseline_retrieval_loss() -> None:
    views = load_failure_case_views(project_root=PROJECT_ROOT)
    view = next(
        item for item in views if item.case.case_id == "multilingual_security_012"
    )

    baseline = view.baseline_status

    assert view.baseline_pipeline_id is PipelineId.CHAR_DENSE_NAIVE
    assert baseline.gold_evidence_included is False
    assert baseline.loss_stage is EvidenceLossStage.RETRIEVAL
    assert baseline.failure_labels == (FailureLabel.DENSE_RETRIEVAL_MISS,)
    assert baseline.rank_used_for_context is None


def test_failure_case_explorer_surfaces_known_baseline_ranking_loss() -> None:
    views = load_failure_case_views(project_root=PROJECT_ROOT)
    view = next(
        item for item in views if item.case.case_id == "support_enterprise_s1_006"
    )

    baseline = view.baseline_status

    assert baseline.gold_evidence_included is False
    assert baseline.loss_stage is EvidenceLossStage.RANKING
    assert baseline.rank_used_for_context == baseline.retrieved_gold_rank == 4
    assert baseline.failure_labels == (
        FailureLabel.RELEVANT_CHUNK_RANKED_TOO_LOW,
    )


def test_failure_case_explorer_rejects_case_and_artifact_coverage_drift() -> None:
    artifact = load_baseline_artifact(
        PROJECT_ROOT / DEFAULT_BASELINE_ARTIFACT_PATH
    )
    cases = load_evaluation_cases(PROJECT_ROOT / "data" / "eval_cases.jsonl")

    with pytest.raises(CaseExplorerError, match="coverage differ"):
        build_failure_case_views(cases=cases[:-1], artifact=artifact)
