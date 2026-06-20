"""Deterministic repair recommendations for the RAG comparison harness.

This module translates observed evidence-loss labels from a typed four-pipeline
comparison report into bounded, explainable next interventions. It intentionally
uses report metadata, failure labels, and aggregate metrics only. Raw documents,
chunks, rendered context, prompts, and generated answers remain outside this
report surface.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_lab.comparison import PipelineComparisonReport, PipelineId
from rag_lab.failure_taxonomy import get_failure_taxonomy_entry
from rag_lab.schemas import EvidenceLossStage, FailureLabel


REPAIR_RECOMMENDATION_REPORT_SCHEMA_VERSION: Final[str] = "repair_recommendation_report_v1"


class RepairPriority(StrEnum):
    """Delivery order for a repair, based on where evidence first became unusable."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class RepairRecommendationId(StrEnum):
    """Stable, report-safe identifiers for deterministic repair families."""

    SENTENCE_AWARE_TOKEN_CHUNKING = "sentence_aware_token_chunking"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    CROSS_ENCODER_RERANKING = "cross_encoder_reranking"
    TOKEN_BUDGETED_CONTEXT_PACKING = "token_budgeted_context_packing"
    GROUNDED_GENERATION_GUARDS = "grounded_generation_guards"


class RepairRecommendation(BaseModel):
    """One evidence-backed intervention derived from observed baseline failures."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    recommendation_id: RepairRecommendationId
    priority: RepairPriority
    loss_stage: EvidenceLossStage
    observed_failure_labels: tuple[FailureLabel, ...] = Field(min_length=1, max_length=5)
    observed_failure_count: int = Field(ge=1)
    recommendation: str = Field(min_length=24, max_length=500)
    expected_signal: str = Field(min_length=24, max_length=700)
    metrics_to_watch: tuple[str, ...] = Field(min_length=1, max_length=4)
    trade_off: str = Field(min_length=24, max_length=500)
    supporting_pipeline_id: PipelineId | None = None
    evidence_summary: str = Field(min_length=24, max_length=900)

    @field_validator("observed_failure_labels")
    @classmethod
    def require_unique_failure_labels(
        cls, labels: tuple[FailureLabel, ...]
    ) -> tuple[FailureLabel, ...]:
        if len(labels) != len(set(labels)):
            raise ValueError("observed_failure_labels must not contain duplicates")
        return labels

    @model_validator(mode="after")
    def validate_stage_matches_taxonomy(self) -> "RepairRecommendation":
        invalid_labels = [
            label
            for label in self.observed_failure_labels
            if get_failure_taxonomy_entry(label).loss_stage is not self.loss_stage
        ]
        if invalid_labels:
            rendered = ", ".join(label.value for label in invalid_labels)
            raise ValueError(
                "observed_failure_labels must map to the recommendation loss_stage: "
                f"{rendered}"
            )
        return self


class RepairRecommendationReport(BaseModel):
    """Privacy-bounded recommendation surface for one comparison report."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: str = Field(
        default=REPAIR_RECOMMENDATION_REPORT_SCHEMA_VERSION,
        min_length=5,
        max_length=120,
    )
    source_run_id: str = Field(pattern=r"^[a-z0-9_:-]+$", min_length=5, max_length=160)
    baseline_pipeline_id: PipelineId
    evaluated_case_count: int = Field(ge=1, le=10_000)
    observed_failure_label_counts: dict[FailureLabel, int]
    recommendations: tuple[RepairRecommendation, ...] = Field(max_length=5)

    @model_validator(mode="after")
    def validate_recommendation_coverage(self) -> "RepairRecommendationReport":
        if self.schema_version != REPAIR_RECOMMENDATION_REPORT_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must equal "
                f"{REPAIR_RECOMMENDATION_REPORT_SCHEMA_VERSION}"
            )
        if any(count <= 0 for count in self.observed_failure_label_counts.values()):
            raise ValueError("observed_failure_label_counts values must be positive")

        observed_labels = set(self.observed_failure_label_counts)
        recommendation_labels = {
            label
            for recommendation in self.recommendations
            for label in recommendation.observed_failure_labels
        }
        if observed_labels != recommendation_labels:
            missing = sorted(label.value for label in observed_labels - recommendation_labels)
            extra = sorted(label.value for label in recommendation_labels - observed_labels)
            raise ValueError(
                "recommendations must cover observed failure labels exactly; "
                f"missing={missing}, extra={extra}"
            )

        recommendation_ids = [
            recommendation.recommendation_id for recommendation in self.recommendations
        ]
        if len(recommendation_ids) != len(set(recommendation_ids)):
            raise ValueError("recommendations must not repeat recommendation_id")
        return self


@dataclass(frozen=True, slots=True)
class _RepairPolicy:
    recommendation_id: RepairRecommendationId
    priority: RepairPriority
    loss_stage: EvidenceLossStage
    failure_labels: tuple[FailureLabel, ...]
    recommendation: str
    metrics_to_watch: tuple[str, ...]
    trade_off: str
    supporting_pipeline_id: PipelineId | None


REPAIR_POLICIES: Final[tuple[_RepairPolicy, ...]] = (
    _RepairPolicy(
        recommendation_id=RepairRecommendationId.SENTENCE_AWARE_TOKEN_CHUNKING,
        priority=RepairPriority.CRITICAL,
        loss_stage=EvidenceLossStage.CHUNKING,
        failure_labels=(
            FailureLabel.BAD_CHUNK_BOUNDARY,
            FailureLabel.GOLD_EVIDENCE_SPLIT,
        ),
        recommendation=(
            "Replace character-boundary chunking with sentence-aware token chunking "
            "and retain bounded overlap only where a clause, table row, or event can "
            "cross the boundary."
        ),
        metrics_to_watch=("Recall@{k}", "evidence inclusion"),
        trade_off=(
            "Sentence-aware splitting increases implementation and tokenizer-provenance "
            "discipline; count final emitted text, including separators, rather than "
            "trusting configured unit budgets."
        ),
        supporting_pipeline_id=PipelineId.TOKEN_DENSE_NAIVE,
    ),
    _RepairPolicy(
        recommendation_id=RepairRecommendationId.HYBRID_RETRIEVAL,
        priority=RepairPriority.CRITICAL,
        loss_stage=EvidenceLossStage.RETRIEVAL,
        failure_labels=(
            FailureLabel.DENSE_RETRIEVAL_MISS,
            FailureLabel.RETRIEVAL_MISS,
            FailureLabel.KEYWORD_RETRIEVAL_NEEDED,
        ),
        recommendation=(
            "Add BM25-backed hybrid retrieval so exact terms, identifiers, legal clauses, "
            "prices, and error codes can complement dense semantic recall."
        ),
        metrics_to_watch=("Recall@{k}", "evidence inclusion"),
        trade_off=(
            "Hybrid retrieval adds lexical-index and fusion parameters that must remain "
            "fixed and attributable across comparisons; do not conceal candidate-depth "
            "changes behind a higher headline cutoff."
        ),
        supporting_pipeline_id=PipelineId.TOKEN_HYBRID_NAIVE,
    ),
    _RepairPolicy(
        recommendation_id=RepairRecommendationId.CROSS_ENCODER_RERANKING,
        priority=RepairPriority.HIGH,
        loss_stage=EvidenceLossStage.RANKING,
        failure_labels=(
            FailureLabel.RERANKER_NEEDED,
            FailureLabel.RELEVANT_CHUNK_RANKED_TOO_LOW,
        ),
        recommendation=(
            "Apply cross-encoder reranking after candidate recall succeeds and before "
            "final context selection, so the most query-specific evidence reaches the "
            "context window."
        ),
        metrics_to_watch=("MRR@10", "evidence inclusion"),
        trade_off=(
            "Reranking adds per-candidate inference cost and latency; bound candidate "
            "depth first, then verify that ordering improved without trading away recall."
        ),
        supporting_pipeline_id=PipelineId.TOKEN_HYBRID_RERANK_BUDGETED,
    ),
    _RepairPolicy(
        recommendation_id=RepairRecommendationId.TOKEN_BUDGETED_CONTEXT_PACKING,
        priority=RepairPriority.HIGH,
        loss_stage=EvidenceLossStage.CONTEXT_ASSEMBLY,
        failure_labels=(
            FailureLabel.RELEVANT_CHUNK_DROPPED_BY_BUDGET,
            FailureLabel.CONTEXT_BUDGET_EXCEEDED,
            FailureLabel.DUPLICATE_CONTEXT_WASTE,
            FailureLabel.TOKEN_BUDGET_REGRESSION,
        ),
        recommendation=(
            "Use measured token-budgeted context packing that preserves high-relevance "
            "evidence, reserves output capacity, and records every dropped chunk with a "
            "machine-readable reason."
        ),
        metrics_to_watch=("dropped-evidence rate", "evidence inclusion"),
        trade_off=(
            "Budget-aware packing requires model- and tokenizer-specific accounting; "
            "rendering wrappers, labels, and separators are part of the real budget and "
            "must remain observable."
        ),
        supporting_pipeline_id=PipelineId.TOKEN_HYBRID_RERANK_BUDGETED,
    ),
    _RepairPolicy(
        recommendation_id=RepairRecommendationId.GROUNDED_GENERATION_GUARDS,
        priority=RepairPriority.MEDIUM,
        loss_stage=EvidenceLossStage.GENERATION,
        failure_labels=(
            FailureLabel.ANSWER_UNSUPPORTED_BY_CONTEXT,
            FailureLabel.CITATION_MISSING_OR_WRONG,
        ),
        recommendation=(
            "Add answer-to-evidence and citation-to-evidence checks with an unsupported "
            "fallback before presenting a generated answer as grounded."
        ),
        metrics_to_watch=("citation correctness", "unsupported-answer rate"),
        trade_off=(
            "Generation checks require a separate fixed-case evaluation layer and spot "
            "checks; they must not be inferred from retrieval or context-selection metrics alone."
        ),
        supporting_pipeline_id=None,
    ),
)


def build_repair_recommendation_report(
    *, comparison_report: PipelineComparisonReport
) -> RepairRecommendationReport:
    """Build deterministic, evidence-backed recommendations from baseline failures.

    The function does not alter benchmark data, pipeline definitions, metrics, or
    baseline governance. It only interprets already-normalized failure labels from the
    selected baseline pipeline.
    """

    baseline_outcomes = [
        outcome
        for outcome in comparison_report.case_outcomes
        if outcome.pipeline_id is comparison_report.baseline_pipeline_id
        and not outcome.gold_evidence_included
    ]
    failure_counts = Counter(
        label for outcome in baseline_outcomes for label in outcome.failure_labels
    )
    ordered_failure_counts = {
        label: failure_counts[label]
        for label in sorted(failure_counts, key=lambda item: item.value)
    }
    metrics_by_pipeline = {
        metric.pipeline_id: metric for metric in comparison_report.pipeline_metrics
    }
    baseline_metrics = metrics_by_pipeline[comparison_report.baseline_pipeline_id]

    recommendations = tuple(
        recommendation
        for policy in REPAIR_POLICIES
        if (
            recommendation := _build_recommendation(
                policy=policy,
                failure_counts=failure_counts,
                baseline_metrics=baseline_metrics,
                metrics_by_pipeline=metrics_by_pipeline,
                retrieval_metric_k=comparison_report.retrieval_metric_k,
            )
        )
        is not None
    )

    return RepairRecommendationReport(
        source_run_id=comparison_report.run_id,
        baseline_pipeline_id=comparison_report.baseline_pipeline_id,
        evaluated_case_count=baseline_metrics.evaluated_case_count,
        observed_failure_label_counts=ordered_failure_counts,
        recommendations=recommendations,
    )


def render_repair_recommendations_markdown(*, report: RepairRecommendationReport) -> str:
    """Render a concise, deterministic Markdown decision surface."""

    lines = [
        "# Deterministic Repair Recommendations",
        "",
        "## Scope",
        "",
        "This report translates observed baseline evidence-loss labels into deterministic "
        "repair actions. It uses only typed comparison outcomes, aggregate metrics, and "
        "failure taxonomy. It does not contain raw documents, chunks, prompts, rendered "
        "context, or generated answers.",
        "",
        "## Source",
        "",
        f"- **Comparison run:** `{report.source_run_id}`",
        f"- **Baseline pipeline:** `{report.baseline_pipeline_id.value}`",
        f"- **Fixed cases:** {report.evaluated_case_count}",
        "",
        "## Observed baseline failure labels",
        "",
    ]

    if report.observed_failure_label_counts:
        for label, count in report.observed_failure_label_counts.items():
            lines.append(f"- `{label.value}`: {count}")
    else:
        lines.append("- No baseline failure labels were observed in this report.")

    lines.extend(["", "## Recommended repair sequence", ""])
    if not report.recommendations:
        lines.extend(
            [
                "No repair recommendation is emitted because the selected baseline has no "
                "observed evidence-loss labels. Preserve the fixed benchmark and use a "
                "controlled diagnostic before adding speculative complexity.",
                "",
            ]
        )
    else:
        for position, recommendation in enumerate(report.recommendations, start=1):
            supporting_pipeline = (
                f"`{recommendation.supporting_pipeline_id.value}`"
                if recommendation.supporting_pipeline_id is not None
                else "a future generation-evaluation layer"
            )
            observed_labels = ", ".join(
                f"`{label.value}`" for label in recommendation.observed_failure_labels
            )
            metrics_to_watch = ", ".join(recommendation.metrics_to_watch)
            lines.extend(
                [
                    f"### {position}. {recommendation.priority.value.title()} — "
                    f"{recommendation.recommendation_id.value.replace('_', ' ')}",
                    "",
                    f"- **Observed labels:** {observed_labels}",
                    f"- **Observed failure count:** {recommendation.observed_failure_count}",
                    f"- **Repair:** {recommendation.recommendation}",
                    f"- **Expected signal:** {recommendation.expected_signal}",
                    f"- **Metrics to watch:** {metrics_to_watch}",
                    f"- **Trade-off:** {recommendation.trade_off}",
                    f"- **Supporting comparison:** {supporting_pipeline}",
                    f"- **Evidence:** {recommendation.evidence_summary}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Validation command",
            "",
            "```powershell",
            "python .\\scripts\\run_comparison_baseline.py `",
            "    --tokenizer tiktoken `",
            "    --tiktoken-encoding cl100k_base",
            "```",
            "",
            "## Non-claims",
            "",
            "These recommendations are limited to the fixed synthetic benchmark and "
            "recorded failure labels. They do not prove customer-data performance, "
            "production readiness, answer grounding, citation correctness, or model "
            "behavior across providers.",
            "",
        ]
    )
    return "\n".join(lines)


def write_repair_recommendation_json(
    *, report: RepairRecommendationReport, path: Path
) -> None:
    """Write a JSON-safe recommendation artifact without raw retrieval payloads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_repair_recommendations_markdown(
    *, report: RepairRecommendationReport, path: Path
) -> None:
    """Write the deterministic Markdown recommendation surface."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_repair_recommendations_markdown(report=report), encoding="utf-8")


def _build_recommendation(
    *,
    policy: _RepairPolicy,
    failure_counts: Counter[FailureLabel],
    baseline_metrics: object,
    metrics_by_pipeline: dict[PipelineId, object],
    retrieval_metric_k: int,
) -> RepairRecommendation | None:
    observed_labels = tuple(
        label for label in policy.failure_labels if failure_counts.get(label, 0) > 0
    )
    if not observed_labels:
        return None

    observed_failure_count = sum(failure_counts[label] for label in observed_labels)
    target_metrics = (
        metrics_by_pipeline[policy.supporting_pipeline_id]
        if policy.supporting_pipeline_id is not None
        else None
    )
    return RepairRecommendation(
        recommendation_id=policy.recommendation_id,
        priority=policy.priority,
        loss_stage=policy.loss_stage,
        observed_failure_labels=observed_labels,
        observed_failure_count=observed_failure_count,
        recommendation=policy.recommendation,
        expected_signal=_build_expected_signal(
            recommendation_id=policy.recommendation_id,
            baseline_metrics=baseline_metrics,
            target_metrics=target_metrics,
            retrieval_metric_k=retrieval_metric_k,
        ),
        metrics_to_watch=tuple(
            metric.format(k=retrieval_metric_k) for metric in policy.metrics_to_watch
        ),
        trade_off=policy.trade_off,
        supporting_pipeline_id=policy.supporting_pipeline_id,
        evidence_summary=_build_evidence_summary(
            observed_labels=observed_labels,
            failure_counts=failure_counts,
            loss_stage=policy.loss_stage,
        ),
    )


def _build_expected_signal(
    *,
    recommendation_id: RepairRecommendationId,
    baseline_metrics: object,
    target_metrics: object | None,
    retrieval_metric_k: int,
) -> str:
    """Build a bounded evidence sentence without exposing raw runtime content."""

    if target_metrics is None:
        return (
            "Introduce a separate generation-evaluation layer and measure citation "
            "correctness plus unsupported-answer rate; the current four-pipeline report "
            "stops at final evidence selection."
        )

    baseline_recall = baseline_metrics.retrieval_recall_at_k.value
    target_recall = target_metrics.retrieval_recall_at_k.value
    baseline_inclusion = baseline_metrics.evidence_inclusion_rate.value
    target_inclusion = target_metrics.evidence_inclusion_rate.value
    baseline_drops = baseline_metrics.dropped_evidence_rate.value
    target_drops = target_metrics.dropped_evidence_rate.value

    if recommendation_id is RepairRecommendationId.SENTENCE_AWARE_TOKEN_CHUNKING:
        return (
            f"Watch Recall@{retrieval_metric_k} and evidence inclusion. In this fixed run, "
            f"the supporting pipeline moved Recall@{retrieval_metric_k} from "
            f"{_percent(baseline_recall)} to {_percent(target_recall)} and evidence "
            f"inclusion from {_percent(baseline_inclusion)} to {_percent(target_inclusion)}."
        )
    if recommendation_id is RepairRecommendationId.HYBRID_RETRIEVAL:
        return (
            f"Watch Recall@{retrieval_metric_k} before changing ranking. In this fixed run, "
            f"the supporting pipeline moved Recall@{retrieval_metric_k} from "
            f"{_percent(baseline_recall)} to {_percent(target_recall)} and evidence "
            f"inclusion from {_percent(baseline_inclusion)} to {_percent(target_inclusion)}."
        )
    if recommendation_id is RepairRecommendationId.CROSS_ENCODER_RERANKING:
        return (
            "Watch MRR@10 after candidate recall is already sufficient. In this fixed run, "
            f"the supporting pipeline moved MRR@10 from {baseline_metrics.mrr_at_10:.3f} "
            f"to {target_metrics.mrr_at_10:.3f} while preserving Recall@{retrieval_metric_k} "
            f"at {_percent(target_recall)}."
        )
    if recommendation_id is RepairRecommendationId.TOKEN_BUDGETED_CONTEXT_PACKING:
        return (
            "Watch dropped-evidence rate and evidence inclusion under measured rendered "
            f"context budgets. In this fixed run, dropped-evidence rate changed from "
            f"{_percent(baseline_drops)} to {_percent(target_drops)} and evidence inclusion "
            f"changed from {_percent(baseline_inclusion)} to {_percent(target_inclusion)}."
        )
    raise ValueError(f"unsupported recommendation_id: {recommendation_id}")


def _build_evidence_summary(
    *,
    observed_labels: tuple[FailureLabel, ...],
    failure_counts: Counter[FailureLabel],
    loss_stage: EvidenceLossStage,
) -> str:
    rendered_labels = ", ".join(
        f"{label.value} ({failure_counts[label]})" for label in observed_labels
    )
    return (
        f"The baseline recorded {sum(failure_counts[label] for label in observed_labels)} "
        f"failure label occurrence(s) at the {loss_stage.value} stage: {rendered_labels}."
    )


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"
