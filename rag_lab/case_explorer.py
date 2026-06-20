"""Read-only typed case views for the Streamlit Failure Case Explorer.

The explorer loads fixed synthetic evaluation cases and the reviewed baseline artifact.
It never runs retrievers, rerankers, embeddings, or context assembly. This keeps the
first demo surface deterministic, low-cost, and faithful to the committed benchmark.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.comparison import CasePipelineOutcome, PipelineId
from rag_lab.comparison_artifacts import (
    DEFAULT_BASELINE_ARTIFACT_PATH,
    ComparisonBaselineArtifact,
    load_baseline_artifact,
)
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.schemas import EvaluationCase, EvidenceLossStage, FailureLabel


class CaseExplorerError(ValueError):
    """Raised when fixed cases and the reviewed artifact cannot form one safe view."""


class PipelineEvidenceStatus(BaseModel):
    """Privacy-bounded evidence state for one case through one fixed pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    pipeline_id: PipelineId
    gold_evidence_included: bool
    retrieved_gold_rank: int | None = Field(default=None, ge=1, le=100)
    reranked_gold_rank: int | None = Field(default=None, ge=1, le=100)
    rank_used_for_context: int | None = Field(default=None, ge=1, le=100)
    loss_stage: EvidenceLossStage | None = None
    failure_labels: tuple[FailureLabel, ...] = Field(default_factory=tuple, max_length=5)

    @model_validator(mode="after")
    def validate_lifecycle_summary(self) -> "PipelineEvidenceStatus":
        expected_rank = self.reranked_gold_rank or self.retrieved_gold_rank
        if self.rank_used_for_context != expected_rank:
            raise ValueError(
                "rank_used_for_context must match reranked_gold_rank or retrieved_gold_rank"
            )
        if self.gold_evidence_included:
            if self.loss_stage is not None or self.failure_labels:
                raise ValueError(
                    "included evidence cannot carry a loss_stage or failure_labels"
                )
            return self
        if self.loss_stage is None or not self.failure_labels:
            raise ValueError(
                "missing evidence requires a loss_stage and at least one failure label"
            )
        return self


class FailureCaseView(BaseModel):
    """One selected fixed case plus its reviewed four-pipeline evidence lifecycle."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_run_id: str = Field(pattern=r"^[a-z0-9_:-]+$", min_length=5, max_length=160)
    baseline_pipeline_id: PipelineId
    case: EvaluationCase
    pipeline_statuses: tuple[PipelineEvidenceStatus, ...] = Field(
        min_length=4,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_fixed_pipeline_coverage(self) -> "FailureCaseView":
        expected_pipeline_ids = set(PipelineId)
        actual_pipeline_ids = {
            status.pipeline_id for status in self.pipeline_statuses
        }
        if actual_pipeline_ids != expected_pipeline_ids:
            raise ValueError(
                "pipeline_statuses must contain each fixed pipeline exactly once"
            )
        if len(actual_pipeline_ids) != len(self.pipeline_statuses):
            raise ValueError("pipeline_statuses must not repeat pipeline_id")
        return self

    @property
    def baseline_status(self) -> PipelineEvidenceStatus:
        """Return the selected baseline's evidence state for the fixed case."""

        return next(
            status
            for status in self.pipeline_statuses
            if status.pipeline_id is self.baseline_pipeline_id
        )


def load_failure_case_views(*, project_root: Path) -> tuple[FailureCaseView, ...]:
    """Load the fixed cases and reviewed baseline into deterministic explorer views."""

    evaluation_cases_path = project_root / "data" / "eval_cases.jsonl"
    baseline_artifact_path = project_root / DEFAULT_BASELINE_ARTIFACT_PATH

    cases = load_evaluation_cases(evaluation_cases_path)
    artifact = load_baseline_artifact(baseline_artifact_path)
    return build_failure_case_views(cases=cases, artifact=artifact)


def build_failure_case_views(
    *,
    cases: Iterable[EvaluationCase],
    artifact: ComparisonBaselineArtifact,
) -> tuple[FailureCaseView, ...]:
    """Join typed fixed cases to reviewed outcomes without altering benchmark state."""

    ordered_cases = tuple(sorted(cases, key=lambda case: case.case_id))
    case_ids = [case.case_id for case in ordered_cases]
    if not ordered_cases:
        raise CaseExplorerError("at least one fixed evaluation case is required")
    if len(case_ids) != len(set(case_ids)):
        raise CaseExplorerError("fixed evaluation cases must not repeat case_id")

    report_case_ids = {outcome.case_id for outcome in artifact.report.case_outcomes}
    expected_case_ids = set(case_ids)
    if report_case_ids != expected_case_ids:
        missing_from_artifact = sorted(expected_case_ids - report_case_ids)
        missing_from_cases = sorted(report_case_ids - expected_case_ids)
        raise CaseExplorerError(
            "fixed evaluation cases and reviewed artifact coverage differ: "
            f"missing_from_artifact={missing_from_artifact}, "
            f"missing_from_cases={missing_from_cases}"
        )

    outcomes_by_key = {
        (outcome.pipeline_id, outcome.case_id): outcome
        for outcome in artifact.report.case_outcomes
    }
    views: list[FailureCaseView] = []
    for case in ordered_cases:
        statuses = tuple(
            _build_pipeline_status(
                outcomes_by_key[(pipeline_id, case.case_id)]
            )
            for pipeline_id in PipelineId
        )
        views.append(
            FailureCaseView(
                source_run_id=artifact.report.run_id,
                baseline_pipeline_id=artifact.report.baseline_pipeline_id,
                case=case,
                pipeline_statuses=statuses,
            )
        )
    return tuple(views)


def _build_pipeline_status(outcome: CasePipelineOutcome) -> PipelineEvidenceStatus:
    """Reduce one already-validated outcome to UI-safe status fields."""

    return PipelineEvidenceStatus(
        pipeline_id=outcome.pipeline_id,
        gold_evidence_included=outcome.gold_evidence_included,
        retrieved_gold_rank=outcome.retrieved_gold_rank,
        reranked_gold_rank=outcome.reranked_gold_rank,
        rank_used_for_context=outcome.rank_used_for_context,
        loss_stage=outcome.loss_stage,
        failure_labels=tuple(outcome.failure_labels),
    )
