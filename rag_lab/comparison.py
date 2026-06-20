"""Schema-first comparison contracts and deterministic metric aggregation.

This module does not run retrieval models. It accepts one normalized, bounded outcome
per (pipeline, evaluation case) and produces the machine-readable comparison artifact
that later runner slices will populate from typed chunking, retrieval, reranking, and
context-autopsy traces.

No raw chunk text, rendered prompt, or source document content belongs in this report.
"""
from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_lab.schemas import (
    ChunkingStrategy,
    EvidenceLossStage,
    FailureLabel,
    RetrievalMethod,
)


class ComparisonInputError(ValueError):
    """Raised when a comparison report would make an ambiguous metric claim."""


class PipelineId(StrEnum):
    """The four fixed ablation pipelines required by the lab PRD."""

    CHAR_DENSE_NAIVE = "char_dense_naive"
    TOKEN_DENSE_NAIVE = "token_dense_naive"
    TOKEN_HYBRID_NAIVE = "token_hybrid_naive"
    TOKEN_HYBRID_RERANK_BUDGETED = "token_hybrid_rerank_budgeted"


class ContextSelectionMode(StrEnum):
    """How a pipeline chooses evidence after its final ranking stage."""

    NAIVE_TOP_K = "naive_top_k"
    TOKEN_BUDGETED = "token_budgeted"


class PipelineDefinition(BaseModel):
    """Inspectable configuration for one fixed ablation pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    pipeline_id: PipelineId
    chunking_strategy: ChunkingStrategy
    retrieval_method: RetrievalMethod
    reranker_enabled: bool
    context_selection_mode: ContextSelectionMode
    retrieval_top_k: int = Field(ge=1, le=100)
    final_evidence_chunk_limit: int = Field(ge=1, le=100)
    max_context_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    reserved_output_tokens: int | None = Field(default=None, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_context_selection_contract(self) -> "PipelineDefinition":
        if self.context_selection_mode is ContextSelectionMode.NAIVE_TOP_K:
            if self.max_context_tokens is not None or self.reserved_output_tokens is not None:
                raise ValueError(
                    "naive_top_k pipelines must not claim measured token-budget settings"
                )
            return self

        if self.max_context_tokens is None or self.reserved_output_tokens is None:
            raise ValueError(
                "token_budgeted pipelines require max_context_tokens and reserved_output_tokens"
            )
        if self.reserved_output_tokens >= self.max_context_tokens:
            raise ValueError("reserved_output_tokens must be smaller than max_context_tokens")
        return self


DEFAULT_PIPELINE_DEFINITIONS: tuple[PipelineDefinition, ...] = (
    PipelineDefinition(
        pipeline_id=PipelineId.CHAR_DENSE_NAIVE,
        chunking_strategy=ChunkingStrategy.CHARACTER,
        retrieval_method=RetrievalMethod.DENSE,
        reranker_enabled=False,
        context_selection_mode=ContextSelectionMode.NAIVE_TOP_K,
        retrieval_top_k=8,
        final_evidence_chunk_limit=3,
    ),
    PipelineDefinition(
        pipeline_id=PipelineId.TOKEN_DENSE_NAIVE,
        chunking_strategy=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
        retrieval_method=RetrievalMethod.DENSE,
        reranker_enabled=False,
        context_selection_mode=ContextSelectionMode.NAIVE_TOP_K,
        retrieval_top_k=8,
        final_evidence_chunk_limit=3,
    ),
    PipelineDefinition(
        pipeline_id=PipelineId.TOKEN_HYBRID_NAIVE,
        chunking_strategy=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
        retrieval_method=RetrievalMethod.HYBRID,
        reranker_enabled=False,
        context_selection_mode=ContextSelectionMode.NAIVE_TOP_K,
        retrieval_top_k=8,
        final_evidence_chunk_limit=3,
    ),
    PipelineDefinition(
        pipeline_id=PipelineId.TOKEN_HYBRID_RERANK_BUDGETED,
        chunking_strategy=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
        retrieval_method=RetrievalMethod.HYBRID,
        reranker_enabled=True,
        context_selection_mode=ContextSelectionMode.TOKEN_BUDGETED,
        retrieval_top_k=8,
        final_evidence_chunk_limit=8,
        max_context_tokens=1_200,
        reserved_output_tokens=240,
    ),
)


class TraceReference(BaseModel):
    """Bounded identifier for a trace retained elsewhere in an execution run.

    The comparison report stores identifiers and hashes, not raw chunks, prompts, or source text.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    trace_id: str = Field(pattern=r"^[a-z0-9_:-]+$", min_length=5, max_length=160)
    trace_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CasePipelineOutcome(BaseModel):
    """Normalized evidence outcome for one fixed case through one pipeline.

    ``retrieved_gold_rank`` measures first-stage recall. ``reranked_gold_rank`` is present
    only when a reranker produced the final order used for context selection. A missing gold
    evidence result must be classified at the stage where it became unusable.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    pipeline_id: PipelineId
    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    requested_top_k: int = Field(ge=1, le=100)
    retrieved_gold_rank: int | None = Field(default=None, ge=1, le=100)
    reranked_gold_rank: int | None = Field(default=None, ge=1, le=100)
    gold_evidence_included: bool
    loss_stage: EvidenceLossStage | None = None
    failure_labels: list[FailureLabel] = Field(default_factory=list)
    trace_references: list[TraceReference] = Field(min_length=1, max_length=8)

    @field_validator("failure_labels")
    @classmethod
    def require_unique_failure_labels(cls, labels: list[FailureLabel]) -> list[FailureLabel]:
        if len(labels) != len(set(labels)):
            raise ValueError("failure_labels must not contain duplicates")
        return labels

    @model_validator(mode="after")
    def validate_evidence_lifecycle(self) -> "CasePipelineOutcome":
        for rank_name, rank in (
            ("retrieved_gold_rank", self.retrieved_gold_rank),
            ("reranked_gold_rank", self.reranked_gold_rank),
        ):
            if rank is not None and rank > self.requested_top_k:
                raise ValueError(f"{rank_name} cannot exceed requested_top_k")

        if self.reranked_gold_rank is not None and self.retrieved_gold_rank is None:
            raise ValueError("reranked_gold_rank requires retrieved_gold_rank")

        if self.gold_evidence_included:
            if self.rank_used_for_context is None:
                raise ValueError("included gold evidence requires a retrieval or reranking rank")
            if self.loss_stage is not None or self.failure_labels:
                raise ValueError("included gold evidence must not carry a loss stage or failure labels")
            return self

        if self.loss_stage is None:
            raise ValueError("missing gold evidence requires loss_stage")
        if not self.failure_labels:
            raise ValueError("missing gold evidence requires at least one failure label")

        if self.loss_stage in {EvidenceLossStage.RANKING, EvidenceLossStage.CONTEXT_ASSEMBLY}:
            if self.rank_used_for_context is None:
                raise ValueError("ranking or context loss requires gold evidence in the candidate set")
        if self.loss_stage is EvidenceLossStage.RETRIEVAL and self.retrieved_gold_rank is not None:
            raise ValueError("retrieval loss cannot include a retrieved gold rank")
        return self

    @property
    def rank_used_for_context(self) -> int | None:
        """Return the last ranking stage that governed context selection."""

        return self.reranked_gold_rank or self.retrieved_gold_rank


class MetricRate(BaseModel):
    """A named rate retaining the numerator and denominator behind a percentage claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_value(self) -> "MetricRate":
        expected = 0.0 if self.denominator == 0 else self.numerator / self.denominator
        if abs(self.value - expected) > 1e-12:
            raise ValueError("value must equal numerator divided by denominator")
        return self


class PipelineMetrics(BaseModel):
    """Metrics for one pipeline, computed only from fixed case outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_id: PipelineId
    evaluated_case_count: int = Field(ge=1)
    retrieval_recall_at_k: MetricRate
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    evidence_inclusion_rate: MetricRate
    dropped_evidence_rate: MetricRate
    failure_label_counts: dict[FailureLabel, int]
    loss_stage_counts: dict[EvidenceLossStage, int]


class PipelineMetricDelta(BaseModel):
    """Intervention minus baseline delta for one metric family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_pipeline_id: PipelineId
    comparison_pipeline_id: PipelineId
    retrieval_recall_at_k_delta: float = Field(ge=-1.0, le=1.0)
    mrr_at_10_delta: float = Field(ge=-1.0, le=1.0)
    evidence_inclusion_rate_delta: float = Field(ge=-1.0, le=1.0)
    dropped_evidence_rate_delta: float = Field(ge=-1.0, le=1.0)


class PipelineComparisonReport(BaseModel):
    """Machine-readable, privacy-bounded report for the four-pipeline evaluation harness."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    schema_version: str = Field(default="comparison_report_v1", min_length=5, max_length=80)
    run_id: str = Field(pattern=r"^[a-z0-9_:-]+$", min_length=5, max_length=160)
    baseline_pipeline_id: PipelineId
    pipeline_definitions: list[PipelineDefinition] = Field(min_length=4, max_length=4)
    case_outcomes: list[CasePipelineOutcome] = Field(min_length=4)
    pipeline_metrics: list[PipelineMetrics] = Field(min_length=4, max_length=4)
    metric_deltas: list[PipelineMetricDelta] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_fixed_four_pipeline_coverage(self) -> "PipelineComparisonReport":
        expected_ids = set(PipelineId)
        definition_ids = {definition.pipeline_id for definition in self.pipeline_definitions}
        metric_ids = {metric.pipeline_id for metric in self.pipeline_metrics}
        if definition_ids != expected_ids:
            raise ValueError("pipeline_definitions must contain each fixed pipeline exactly once")
        if metric_ids != expected_ids:
            raise ValueError("pipeline_metrics must contain each fixed pipeline exactly once")
        if len(self.pipeline_definitions) != len(definition_ids):
            raise ValueError("pipeline_definitions must not repeat pipeline_id")

        outcome_keys = {(outcome.pipeline_id, outcome.case_id) for outcome in self.case_outcomes}
        if len(outcome_keys) != len(self.case_outcomes):
            raise ValueError("case_outcomes must not repeat a pipeline/case pair")

        case_ids_by_pipeline: dict[PipelineId, set[str]] = {pipeline_id: set() for pipeline_id in expected_ids}
        for outcome in self.case_outcomes:
            case_ids_by_pipeline[outcome.pipeline_id].add(outcome.case_id)
        expected_case_ids = next(iter(case_ids_by_pipeline.values()))
        if not expected_case_ids or any(case_ids != expected_case_ids for case_ids in case_ids_by_pipeline.values()):
            raise ValueError("every fixed pipeline must evaluate the same non-empty case set")

        if self.baseline_pipeline_id not in expected_ids:
            raise ValueError("baseline_pipeline_id must be one of the fixed pipelines")
        compared_ids = {delta.comparison_pipeline_id for delta in self.metric_deltas}
        expected_comparisons = expected_ids - {self.baseline_pipeline_id}
        if compared_ids != expected_comparisons:
            raise ValueError("metric_deltas must compare baseline to every non-baseline pipeline")
        if any(delta.baseline_pipeline_id is not self.baseline_pipeline_id for delta in self.metric_deltas):
            raise ValueError("every metric delta must use the declared baseline pipeline")
        return self


def build_comparison_report(
    *,
    run_id: str,
    baseline_pipeline_id: PipelineId,
    pipeline_definitions: Iterable[PipelineDefinition] = DEFAULT_PIPELINE_DEFINITIONS,
    case_outcomes: Iterable[CasePipelineOutcome],
) -> PipelineComparisonReport:
    """Aggregate fixed case outcomes into a validated comparison report.

    This is intentionally a pure reducer: it does not invoke models, access files, or mutate
    caller-owned objects. The later execution slice will be responsible for deriving each
    ``CasePipelineOutcome`` from real typed traces.
    """

    definitions = list(pipeline_definitions)
    outcomes = list(case_outcomes)
    _validate_input_coverage(definitions=definitions, outcomes=outcomes)

    metrics_by_pipeline = {
        pipeline_id: _build_pipeline_metrics(
            pipeline_id=pipeline_id,
            outcomes=[outcome for outcome in outcomes if outcome.pipeline_id is pipeline_id],
        )
        for pipeline_id in PipelineId
    }
    baseline_metrics = metrics_by_pipeline[baseline_pipeline_id]
    deltas = [
        _build_metric_delta(
            baseline=baseline_metrics,
            comparison=metrics_by_pipeline[pipeline_id],
        )
        for pipeline_id in PipelineId
        if pipeline_id is not baseline_pipeline_id
    ]

    return PipelineComparisonReport(
        run_id=run_id,
        baseline_pipeline_id=baseline_pipeline_id,
        pipeline_definitions=definitions,
        case_outcomes=outcomes,
        pipeline_metrics=[metrics_by_pipeline[pipeline_id] for pipeline_id in PipelineId],
        metric_deltas=deltas,
    )


def _validate_input_coverage(
    *,
    definitions: list[PipelineDefinition],
    outcomes: list[CasePipelineOutcome],
) -> None:
    expected_ids = set(PipelineId)
    definition_ids = {definition.pipeline_id for definition in definitions}
    if len(definitions) != len(definition_ids) or definition_ids != expected_ids:
        raise ComparisonInputError("definitions must contain the four fixed pipeline IDs exactly once")
    if not outcomes:
        raise ComparisonInputError("comparison requires at least one case outcome per pipeline")

    outcomes_by_pipeline: dict[PipelineId, list[CasePipelineOutcome]] = {
        pipeline_id: [] for pipeline_id in PipelineId
    }
    for outcome in outcomes:
        outcomes_by_pipeline[outcome.pipeline_id].append(outcome)

    case_sets = {
        pipeline_id: {outcome.case_id for outcome in pipeline_outcomes}
        for pipeline_id, pipeline_outcomes in outcomes_by_pipeline.items()
    }
    expected_case_ids = next(iter(case_sets.values()))
    if not expected_case_ids or any(case_ids != expected_case_ids for case_ids in case_sets.values()):
        raise ComparisonInputError("every pipeline must provide exactly the same non-empty case IDs")
    if any(len(pipeline_outcomes) != len(expected_case_ids) for pipeline_outcomes in outcomes_by_pipeline.values()):
        raise ComparisonInputError("each pipeline/case pair must appear exactly once")


def _build_pipeline_metrics(
    *,
    pipeline_id: PipelineId,
    outcomes: list[CasePipelineOutcome],
) -> PipelineMetrics:
    if not outcomes:
        raise ComparisonInputError(f"no outcomes supplied for pipeline {pipeline_id}")

    evaluated_case_count = len(outcomes)
    retrieved_count = sum(outcome.retrieved_gold_rank is not None for outcome in outcomes)
    reciprocal_rank_sum = sum(
        1.0 / outcome.rank_used_for_context
        for outcome in outcomes
        if outcome.rank_used_for_context is not None and outcome.rank_used_for_context <= 10
    )
    included_count = sum(outcome.gold_evidence_included for outcome in outcomes)
    dropped_count = sum(
        outcome.loss_stage is EvidenceLossStage.CONTEXT_ASSEMBLY for outcome in outcomes
    )
    failure_counts = Counter(label for outcome in outcomes for label in outcome.failure_labels)
    stage_counts = Counter(
        outcome.loss_stage for outcome in outcomes if outcome.loss_stage is not None
    )

    return PipelineMetrics(
        pipeline_id=pipeline_id,
        evaluated_case_count=evaluated_case_count,
        retrieval_recall_at_k=_rate(retrieved_count, evaluated_case_count),
        mrr_at_10=reciprocal_rank_sum / evaluated_case_count,
        evidence_inclusion_rate=_rate(included_count, evaluated_case_count),
        dropped_evidence_rate=_rate(dropped_count, retrieved_count),
        failure_label_counts=dict(failure_counts),
        loss_stage_counts=dict(stage_counts),
    )


def _build_metric_delta(
    *,
    baseline: PipelineMetrics,
    comparison: PipelineMetrics,
) -> PipelineMetricDelta:
    return PipelineMetricDelta(
        baseline_pipeline_id=baseline.pipeline_id,
        comparison_pipeline_id=comparison.pipeline_id,
        retrieval_recall_at_k_delta=(
            comparison.retrieval_recall_at_k.value - baseline.retrieval_recall_at_k.value
        ),
        mrr_at_10_delta=comparison.mrr_at_10 - baseline.mrr_at_10,
        evidence_inclusion_rate_delta=(
            comparison.evidence_inclusion_rate.value - baseline.evidence_inclusion_rate.value
        ),
        dropped_evidence_rate_delta=(
            comparison.dropped_evidence_rate.value - baseline.dropped_evidence_rate.value
        ),
    )


def _rate(numerator: int, denominator: int) -> MetricRate:
    return MetricRate(
        numerator=numerator,
        denominator=denominator,
        value=0.0 if denominator == 0 else numerator / denominator,
    )
