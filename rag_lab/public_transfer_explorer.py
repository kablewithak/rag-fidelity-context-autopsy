"""Read-only explorer view for reviewed public-corpus transfer evidence.

This module renders no UI and runs no models. It loads the committed synthetic baseline,
reviewed public-transfer artifact, and deterministic review report through a fail-closed
boundary, then exposes only bounded metrics and provenance for the Streamlit Executive Report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.comparison import MetricRate, PipelineId
from rag_lab.comparison_artifacts import (
    DEFAULT_BASELINE_ARTIFACT_PATH,
    load_baseline_artifact,
)
from rag_lab.public_transfer_artifacts import load_public_transfer_artifact
from rag_lab.public_transfer_review import (
    DEFAULT_REVIEWED_PUBLIC_TRANSFER_ARTIFACT_PATH,
    DEFAULT_REVIEWED_PUBLIC_TRANSFER_REPORT_PATH,
    assert_public_transfer_review_matches,
    assert_review_boundary,
)


class PublicTransferExplorerError(ValueError):
    """Raised when the external-review surface cannot load trusted reviewed evidence."""


class PublicTransferPipelineScorecard(BaseModel):
    """Bounded public-transfer metric row for one fixed pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_id: PipelineId
    retrieval_recall_at_k: MetricRate
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    evidence_inclusion_rate: MetricRate
    dropped_evidence_rate: MetricRate


class PublicTransferReviewView(BaseModel):
    """Separate, side-by-side public-transfer evidence for the Executive Report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_mode: Literal["side_by_side_not_pooled"] = "side_by_side_not_pooled"
    public_artifact_id: str = Field(min_length=8, max_length=120)
    public_source_run_id: str = Field(min_length=5, max_length=160)
    synthetic_artifact_id: str = Field(min_length=8, max_length=120)
    synthetic_case_count: int = Field(gt=0)
    public_dataset_id: str = Field(min_length=3, max_length=160)
    public_dataset_version: str = Field(min_length=1, max_length=80)
    public_license_name: str = Field(min_length=3, max_length=160)
    public_source_url: str = Field(min_length=8, max_length=1_000)
    public_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    public_fixture_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    public_source_document_count: int = Field(gt=0)
    public_evaluation_case_count: int = Field(gt=0)
    pipeline_scorecards: tuple[PublicTransferPipelineScorecard, ...] = Field(min_length=4)
    baseline_scorecard: PublicTransferPipelineScorecard
    strongest_scorecard: PublicTransferPipelineScorecard
    evidence_inclusion_delta_pp: float = Field(ge=-100.0, le=100.0)
    full_pipeline_recall_delta_pp: float = Field(ge=-100.0, le=100.0)
    sentence_aware_dense_recall_delta_pp: float = Field(ge=-100.0, le=100.0)
    mrr_delta: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_side_by_side_contract(self) -> "PublicTransferReviewView":
        scorecard_ids = {card.pipeline_id for card in self.pipeline_scorecards}
        if scorecard_ids != set(PipelineId) or len(self.pipeline_scorecards) != len(scorecard_ids):
            raise ValueError("pipeline_scorecards must contain every fixed pipeline exactly once")
        if self.baseline_scorecard.pipeline_id is not PipelineId.CHAR_DENSE_NAIVE:
            raise ValueError("baseline_scorecard must be char_dense_naive")
        if self.strongest_scorecard.pipeline_id is not PipelineId.TOKEN_HYBRID_RERANK_BUDGETED:
            raise ValueError("strongest_scorecard must be token_hybrid_rerank_budgeted")
        return self


def _scorecard_from_metric(metric: object) -> PublicTransferPipelineScorecard:
    return PublicTransferPipelineScorecard(
        pipeline_id=metric.pipeline_id,
        retrieval_recall_at_k=metric.retrieval_recall_at_k,
        mrr_at_10=metric.mrr_at_10,
        evidence_inclusion_rate=metric.evidence_inclusion_rate,
        dropped_evidence_rate=metric.dropped_evidence_rate,
    )


def load_public_transfer_review_view(*, project_root: Path) -> PublicTransferReviewView:
    """Load verified reviewed public-transfer evidence without rerunning models.

    The loader verifies that the public artifact and report match, that public and
    synthetic runs use the same fixed execution contract, and that the suites remain
    separate. It returns no raw public documents, questions, answers, chunks, prompts,
    candidate scores, or generated answers.
    """

    project_root = project_root.resolve()
    try:
        public_artifact = load_public_transfer_artifact(
            project_root / DEFAULT_REVIEWED_PUBLIC_TRANSFER_ARTIFACT_PATH
        )
        synthetic_baseline = load_baseline_artifact(
            project_root / DEFAULT_BASELINE_ARTIFACT_PATH
        )
        assert_review_boundary(
            public_artifact=public_artifact,
            synthetic_baseline=synthetic_baseline,
        )
        assert_public_transfer_review_matches(
            public_artifact=public_artifact,
            synthetic_baseline=synthetic_baseline,
            path=project_root / DEFAULT_REVIEWED_PUBLIC_TRANSFER_REPORT_PATH,
        )
    except (OSError, ValueError) as error:
        raise PublicTransferExplorerError(
            "could not load verified reviewed public-transfer evidence"
        ) from error

    scorecards = tuple(
        _scorecard_from_metric(metric)
        for metric in public_artifact.report.pipeline_metrics
    )
    scorecard_by_id = {scorecard.pipeline_id: scorecard for scorecard in scorecards}
    baseline = scorecard_by_id[PipelineId.CHAR_DENSE_NAIVE]
    strongest = scorecard_by_id[PipelineId.TOKEN_HYBRID_RERANK_BUDGETED]
    sentence_aware_dense = scorecard_by_id[PipelineId.TOKEN_DENSE_NAIVE]
    provenance = public_artifact.provenance

    return PublicTransferReviewView(
        public_artifact_id=public_artifact.artifact_id,
        public_source_run_id=public_artifact.reference_run_id,
        synthetic_artifact_id=synthetic_baseline.artifact_id,
        synthetic_case_count=synthetic_baseline.provenance.evaluation_case_count,
        public_dataset_id=provenance.external_dataset_id,
        public_dataset_version=provenance.dataset_version,
        public_license_name=provenance.license_name,
        public_source_url=provenance.source_url,
        public_source_sha256=provenance.source_sha256,
        public_fixture_manifest_sha256=provenance.fixture_manifest_sha256,
        public_source_document_count=provenance.source_document_count,
        public_evaluation_case_count=provenance.evaluation_case_count,
        pipeline_scorecards=scorecards,
        baseline_scorecard=baseline,
        strongest_scorecard=strongest,
        evidence_inclusion_delta_pp=(
            strongest.evidence_inclusion_rate.value
            - baseline.evidence_inclusion_rate.value
        )
        * 100,
        full_pipeline_recall_delta_pp=(
            strongest.retrieval_recall_at_k.value
            - baseline.retrieval_recall_at_k.value
        )
        * 100,
        sentence_aware_dense_recall_delta_pp=(
            sentence_aware_dense.retrieval_recall_at_k.value
            - baseline.retrieval_recall_at_k.value
        )
        * 100,
        mrr_delta=strongest.mrr_at_10 - baseline.mrr_at_10,
    )
