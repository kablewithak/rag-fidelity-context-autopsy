"""Read-only typed retrieval views for the Streamlit RAG reliability demo.

The Retrieval Explorer reads fixed synthetic evaluation cases and the reviewed comparison
artifact. It never invokes embeddings, retrievers, rerankers, or context assembly. This keeps
the UI faithful to committed benchmark evidence rather than presenting a fresh local rerun as a
reviewed result.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.comparison import (
    ContextSelectionMode,
    PipelineDefinition,
    PipelineId,
)
from rag_lab.comparison_artifacts import (
    DEFAULT_BASELINE_ARTIFACT_PATH,
    ComparisonBaselineArtifact,
    load_baseline_artifact,
)
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.schemas import (
    ChunkingStrategy,
    EvaluationCase,
    EvidenceLossStage,
    FailureLabel,
    RetrievalMethod,
)


class RetrievalExplorerError(ValueError):
    """Raised when fixed cases and reviewed retrieval outcomes cannot form one safe view."""


class CandidateSetState(StrEnum):
    """Whether complete gold evidence was available to the first-stage candidate set."""

    PRESENT = "present"
    MISSING_FROM_CANDIDATE_SET = "missing_from_candidate_set"
    UNAVAILABLE_AFTER_CHUNKING = "unavailable_after_chunking"


class RetrievalPipelineView(BaseModel):
    """Privacy-bounded retrieval and ranking state for one reviewed pipeline outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    pipeline_id: PipelineId
    chunking_strategy: ChunkingStrategy
    retrieval_method: RetrievalMethod
    reranker_enabled: bool
    context_selection_mode: ContextSelectionMode
    candidate_depth: int = Field(ge=1, le=100)
    final_evidence_chunk_limit: int = Field(ge=1, le=100)
    retrieval_metric_k: int = Field(ge=1, le=100)
    candidate_set_state: CandidateSetState
    retrieved_gold_rank: int | None = Field(default=None, ge=1, le=100)
    reranked_gold_rank: int | None = Field(default=None, ge=1, le=100)
    rank_used_for_context: int | None = Field(default=None, ge=1, le=100)
    gold_evidence_included: bool
    loss_stage: EvidenceLossStage | None = None
    failure_labels: tuple[FailureLabel, ...] = Field(default_factory=tuple, max_length=5)
    trace_ids: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_retrieval_lifecycle(self) -> "RetrievalPipelineView":
        if self.candidate_depth < self.retrieval_metric_k:
            raise ValueError("candidate_depth must be at least retrieval_metric_k")

        if self.reranked_gold_rank is not None and not self.reranker_enabled:
            raise ValueError("reranked_gold_rank requires reranker_enabled")

        expected_context_rank = self.reranked_gold_rank or self.retrieved_gold_rank
        if self.rank_used_for_context != expected_context_rank:
            raise ValueError(
                "rank_used_for_context must match reranked_gold_rank or retrieved_gold_rank"
            )

        if self.candidate_set_state is CandidateSetState.PRESENT:
            if self.retrieved_gold_rank is None:
                raise ValueError("present candidate evidence requires retrieved_gold_rank")
            return self

        if self.retrieved_gold_rank is not None or self.reranked_gold_rank is not None:
            raise ValueError("missing or unavailable candidate evidence cannot carry ranks")

        if self.candidate_set_state is CandidateSetState.UNAVAILABLE_AFTER_CHUNKING:
            if self.loss_stage is not EvidenceLossStage.CHUNKING:
                raise ValueError("unavailable_after_chunking requires chunking loss_stage")
            return self

        if self.loss_stage is not EvidenceLossStage.RETRIEVAL:
            raise ValueError("missing_from_candidate_set requires retrieval loss_stage")
        return self


class RetrievalCaseView(BaseModel):
    """One fixed case plus its reviewed candidate and rank state across all pipelines."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_run_id: str = Field(pattern=r"^[a-z0-9_:-]+$", min_length=5, max_length=160)
    retrieval_metric_k: int = Field(ge=1, le=100)
    baseline_pipeline_id: PipelineId
    case: EvaluationCase
    pipeline_views: tuple[RetrievalPipelineView, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_fixed_pipeline_coverage(self) -> "RetrievalCaseView":
        expected_pipeline_ids = set(PipelineId)
        actual_pipeline_ids = {view.pipeline_id for view in self.pipeline_views}
        if actual_pipeline_ids != expected_pipeline_ids:
            raise ValueError("pipeline_views must contain each fixed pipeline exactly once")
        if len(actual_pipeline_ids) != len(self.pipeline_views):
            raise ValueError("pipeline_views must not repeat pipeline_id")
        return self

    @property
    def baseline_view(self) -> RetrievalPipelineView:
        """Return the reviewed baseline pipeline's retrieval state for the selected case."""

        return next(
            view
            for view in self.pipeline_views
            if view.pipeline_id is self.baseline_pipeline_id
        )


def load_retrieval_case_views(*, project_root: Path) -> tuple[RetrievalCaseView, ...]:
    """Load fixed cases and the reviewed artifact into deterministic retrieval explorer views."""

    cases = load_evaluation_cases(project_root / "data" / "eval_cases.jsonl")
    artifact = load_baseline_artifact(project_root / DEFAULT_BASELINE_ARTIFACT_PATH)
    return build_retrieval_case_views(cases=cases, artifact=artifact)


def build_retrieval_case_views(
    *,
    cases: Iterable[EvaluationCase],
    artifact: ComparisonBaselineArtifact,
) -> tuple[RetrievalCaseView, ...]:
    """Join typed cases, pipeline definitions, and bounded retrieval outcomes without rerunning."""

    ordered_cases = tuple(sorted(cases, key=lambda case: case.case_id))
    case_ids = [case.case_id for case in ordered_cases]
    if not ordered_cases:
        raise RetrievalExplorerError("at least one fixed evaluation case is required")
    if len(case_ids) != len(set(case_ids)):
        raise RetrievalExplorerError("fixed evaluation cases must not repeat case_id")

    expected_case_ids = set(case_ids)
    report_case_ids = {outcome.case_id for outcome in artifact.report.case_outcomes}
    if report_case_ids != expected_case_ids:
        missing_from_artifact = sorted(expected_case_ids - report_case_ids)
        missing_from_cases = sorted(report_case_ids - expected_case_ids)
        raise RetrievalExplorerError(
            "fixed evaluation cases and reviewed artifact coverage differ: "
            f"missing_from_artifact={missing_from_artifact}, "
            f"missing_from_cases={missing_from_cases}"
        )

    definitions_by_pipeline_id = {
        definition.pipeline_id: definition
        for definition in artifact.report.pipeline_definitions
    }
    if set(definitions_by_pipeline_id) != set(PipelineId):
        raise RetrievalExplorerError("reviewed artifact is missing a fixed pipeline definition")

    outcomes_by_key = {
        (outcome.pipeline_id, outcome.case_id): outcome
        for outcome in artifact.report.case_outcomes
    }
    expected_outcome_keys = {
        (pipeline_id, case_id)
        for pipeline_id in PipelineId
        for case_id in expected_case_ids
    }
    if set(outcomes_by_key) != expected_outcome_keys:
        raise RetrievalExplorerError(
            "reviewed artifact does not contain one retrieval outcome for every pipeline/case pair"
        )

    views: list[RetrievalCaseView] = []
    for case in ordered_cases:
        pipeline_views = tuple(
            _build_retrieval_pipeline_view(
                definition=definitions_by_pipeline_id[pipeline_id],
                outcome=outcomes_by_key[(pipeline_id, case.case_id)],
                retrieval_metric_k=artifact.report.retrieval_metric_k,
            )
            for pipeline_id in PipelineId
        )
        views.append(
            RetrievalCaseView(
                source_run_id=artifact.report.run_id,
                retrieval_metric_k=artifact.report.retrieval_metric_k,
                baseline_pipeline_id=artifact.report.baseline_pipeline_id,
                case=case,
                pipeline_views=pipeline_views,
            )
        )
    return tuple(views)


def _build_retrieval_pipeline_view(
    *,
    definition: PipelineDefinition,
    outcome: object,
    retrieval_metric_k: int,
) -> RetrievalPipelineView:
    """Reduce one already-reviewed outcome into an operator-safe retrieval view."""

    if outcome.requested_top_k != definition.retrieval_top_k:
        raise RetrievalExplorerError(
            "reviewed outcome candidate depth does not match its pipeline definition"
        )

    return RetrievalPipelineView(
        pipeline_id=definition.pipeline_id,
        chunking_strategy=definition.chunking_strategy,
        retrieval_method=definition.retrieval_method,
        reranker_enabled=definition.reranker_enabled,
        context_selection_mode=definition.context_selection_mode,
        candidate_depth=outcome.requested_top_k,
        final_evidence_chunk_limit=definition.final_evidence_chunk_limit,
        retrieval_metric_k=retrieval_metric_k,
        candidate_set_state=_candidate_set_state(outcome=outcome),
        retrieved_gold_rank=outcome.retrieved_gold_rank,
        reranked_gold_rank=outcome.reranked_gold_rank,
        rank_used_for_context=outcome.rank_used_for_context,
        gold_evidence_included=outcome.gold_evidence_included,
        loss_stage=outcome.loss_stage,
        failure_labels=tuple(outcome.failure_labels),
        trace_ids=tuple(reference.trace_id for reference in outcome.trace_references),
    )


def _candidate_set_state(*, outcome: object) -> CandidateSetState:
    """Classify availability before ranking without treating chunking loss as a retrieval miss."""

    if outcome.retrieved_gold_rank is not None:
        return CandidateSetState.PRESENT
    if outcome.loss_stage is EvidenceLossStage.CHUNKING:
        return CandidateSetState.UNAVAILABLE_AFTER_CHUNKING
    if outcome.loss_stage is EvidenceLossStage.RETRIEVAL:
        return CandidateSetState.MISSING_FROM_CANDIDATE_SET
    raise RetrievalExplorerError(
        "reviewed outcome has no retrieved gold rank but is not classified as chunking or retrieval loss"
    )
