"""Deterministic executive evaluation report for the RAG reliability lab.

This module converts the reviewed four-pipeline baseline, its deterministic repair
recommendations, and the separate controlled context-pressure proof into one
CTO-readable decision surface. It does not run embeddings, retrieval, reranking,
or answer generation, and it retains no raw documents, chunks, prompts, candidate
scores, rendered context, or generated answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.comparison import MetricRate, PipelineId
from rag_lab.comparison_artifacts import (
    DEFAULT_BASELINE_ARTIFACT_PATH,
    ComparisonBaselineArtifact,
    load_baseline_artifact,
)
from rag_lab.context_assembly import ContextDropReason
from rag_lab.context_autopsy_explorer import (
    ContextAutopsyCaseView,
    load_context_autopsy_case_view,
)
from rag_lab.repair_recommendations import (
    RepairRecommendation,
    build_repair_recommendation_report,
)
from rag_lab.schemas import EvidenceLossStage, FailureLabel


EXECUTIVE_EVALUATION_REPORT_SCHEMA_VERSION: Final[str] = "executive_evaluation_report_v1"
DEFAULT_EXECUTIVE_EVALUATION_REPORT_PATH: Final[Path] = Path(
    "docs/reports/executive_evaluation_report_v1.md"
)


class ExecutiveEvaluationReportError(ValueError):
    """Raised when reviewed evidence cannot form one trustworthy executive report."""


class ExecutivePipelineScorecard(BaseModel):
    """One fixed pipeline summarized without exposing raw retrieval or context payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    pipeline_id: PipelineId
    pipeline_label: str = Field(min_length=8, max_length=120)
    retrieval_recall_at_k: MetricRate
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    evidence_inclusion_rate: MetricRate
    dropped_evidence_rate: MetricRate
    dropped_evidence_eligible_case_count: int = Field(ge=0)


class BaselineFailureStageSummary(BaseModel):
    """Observed fixed-case failures at one evidence-loss boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    loss_stage: EvidenceLossStage
    affected_case_count: int = Field(ge=1)
    failure_labels: tuple[FailureLabel, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_unique_labels(self) -> "BaselineFailureStageSummary":
        if len(self.failure_labels) != len(set(self.failure_labels)):
            raise ValueError("failure_labels must not contain duplicates")
        return self


class ControlledContextFinding(BaseModel):
    """A separately labelled local proof of context-wrapper budget displacement."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    tokenizer_name: str = Field(min_length=3, max_length=160)
    calibrated_context_tokens: int = Field(ge=1)
    reserved_output_tokens: int = Field(ge=0)
    gold_evidence_rank_before_context: int = Field(ge=1)
    verbose_gold_evidence_dropped: bool
    verbose_drop_reason: ContextDropReason
    compact_gold_evidence_included: bool
    verbose_wrapper_tax_tokens: int
    compact_wrapper_tax_tokens: int

    @model_validator(mode="after")
    def validate_controlled_proof(self) -> "ControlledContextFinding":
        if not self.verbose_gold_evidence_dropped:
            raise ValueError("controlled verbose profile must drop gold evidence")
        if self.verbose_drop_reason is not ContextDropReason.BUDGET_EXHAUSTED:
            raise ValueError("controlled verbose profile must drop gold evidence by budget exhaustion")
        if not self.compact_gold_evidence_included:
            raise ValueError("controlled compact profile must retain gold evidence")
        return self


class ExecutiveEvaluationReport(BaseModel):
    """Versioned executive decision surface over reviewed and controlled evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: str = Field(
        default=EXECUTIVE_EVALUATION_REPORT_SCHEMA_VERSION,
        min_length=8,
        max_length=120,
    )
    artifact_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=8, max_length=120)
    source_run_id: str = Field(pattern=r"^[a-z0-9_:-]+$", min_length=5, max_length=160)
    tokenizer_name: str = Field(min_length=3, max_length=160)
    evaluated_case_count: int = Field(ge=1, le=10_000)
    retrieval_metric_k: int = Field(ge=1, le=100)
    baseline_pipeline_id: PipelineId
    pipeline_scorecards: tuple[ExecutivePipelineScorecard, ...] = Field(min_length=4, max_length=4)
    baseline_failure_stages: tuple[BaselineFailureStageSummary, ...] = Field(max_length=5)
    repair_sequence: tuple[RepairRecommendation, ...] = Field(max_length=5)
    controlled_context_finding: ControlledContextFinding

    @model_validator(mode="after")
    def validate_report_contract(self) -> "ExecutiveEvaluationReport":
        if self.schema_version != EXECUTIVE_EVALUATION_REPORT_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must equal "
                f"{EXECUTIVE_EVALUATION_REPORT_SCHEMA_VERSION}"
            )

        scorecard_ids = {scorecard.pipeline_id for scorecard in self.pipeline_scorecards}
        if scorecard_ids != set(PipelineId):
            raise ValueError("pipeline_scorecards must contain each fixed pipeline exactly once")
        if len(scorecard_ids) != len(self.pipeline_scorecards):
            raise ValueError("pipeline_scorecards must not repeat pipeline_id")

        for scorecard in self.pipeline_scorecards:
            if scorecard.retrieval_recall_at_k.denominator != self.evaluated_case_count:
                raise ValueError("scorecard recall denominator must match evaluated_case_count")
            if scorecard.evidence_inclusion_rate.denominator != self.evaluated_case_count:
                raise ValueError(
                    "scorecard evidence inclusion denominator must match evaluated_case_count"
                )
            if scorecard.dropped_evidence_eligible_case_count > self.evaluated_case_count:
                raise ValueError(
                    "scorecard dropped evidence eligible count cannot exceed evaluated_case_count"
                )
            if (
                scorecard.dropped_evidence_rate.denominator
                != scorecard.dropped_evidence_eligible_case_count
            ):
                raise ValueError(
                    "scorecard dropped evidence denominator must match "
                    "dropped_evidence_eligible_case_count"
                )

        stages = [summary.loss_stage for summary in self.baseline_failure_stages]
        if len(stages) != len(set(stages)):
            raise ValueError("baseline_failure_stages must not repeat loss_stage")
        return self

    @property
    def baseline_scorecard(self) -> ExecutivePipelineScorecard:
        """Return the reviewed baseline scorecard."""
        return next(
            scorecard
            for scorecard in self.pipeline_scorecards
            if scorecard.pipeline_id is self.baseline_pipeline_id
        )

    @property
    def strongest_scorecard(self) -> ExecutivePipelineScorecard:
        """Return the highest-evidence-inclusion scorecard, breaking ties by MRR."""
        return max(
            self.pipeline_scorecards,
            key=lambda scorecard: (
                scorecard.evidence_inclusion_rate.value,
                scorecard.mrr_at_10,
                scorecard.retrieval_recall_at_k.value,
            ),
        )


_PIPELINE_LABELS: Final[dict[PipelineId, str]] = {
    PipelineId.CHAR_DENSE_NAIVE: "Character + dense",
    PipelineId.TOKEN_DENSE_NAIVE: "Token + dense",
    PipelineId.TOKEN_HYBRID_NAIVE: "Token + hybrid",
    PipelineId.TOKEN_HYBRID_RERANK_BUDGETED: "Token + hybrid + rerank + budget",
}

_OBSERVED_BASELINE_STAGE_ORDER: Final[tuple[EvidenceLossStage, ...]] = (
    EvidenceLossStage.CHUNKING,
    EvidenceLossStage.RETRIEVAL,
    EvidenceLossStage.RANKING,
    EvidenceLossStage.CONTEXT_ASSEMBLY,
)


def load_executive_evaluation_report(*, project_root: Path) -> ExecutiveEvaluationReport:
    """Load the reviewed baseline and controlled context proof into one executive contract."""

    artifact = load_baseline_artifact(project_root / DEFAULT_BASELINE_ARTIFACT_PATH)
    context_view = load_context_autopsy_case_view(project_root=project_root)
    return build_executive_evaluation_report(
        artifact=artifact,
        controlled_context_view=context_view,
    )


def build_executive_evaluation_report(
    *,
    artifact: ComparisonBaselineArtifact,
    controlled_context_view: ContextAutopsyCaseView,
) -> ExecutiveEvaluationReport:
    """Build a deterministic report from already validated local evidence.

    The comparison artifact is the source for scorecards, observed baseline failures,
    and observed repair sequence. The context-autopsy view is separately labelled as a
    controlled mechanism proof and is never counted as a benchmark failure.
    """

    report = artifact.report
    metrics_by_pipeline = {
        metric.pipeline_id: metric for metric in report.pipeline_metrics
    }
    if set(metrics_by_pipeline) != set(PipelineId):
        raise ExecutiveEvaluationReportError(
            "reviewed artifact must provide metrics for every fixed pipeline"
        )

    outcomes_by_pipeline = {
        pipeline_id: tuple(
            outcome
            for outcome in report.case_outcomes
            if outcome.pipeline_id is pipeline_id
        )
        for pipeline_id in PipelineId
    }
    scorecards = tuple(
        ExecutivePipelineScorecard(
            pipeline_id=pipeline_id,
            pipeline_label=_PIPELINE_LABELS[pipeline_id],
            retrieval_recall_at_k=metrics_by_pipeline[pipeline_id].retrieval_recall_at_k,
            mrr_at_10=metrics_by_pipeline[pipeline_id].mrr_at_10,
            evidence_inclusion_rate=metrics_by_pipeline[pipeline_id].evidence_inclusion_rate,
            dropped_evidence_rate=metrics_by_pipeline[pipeline_id].dropped_evidence_rate,
            dropped_evidence_eligible_case_count=sum(
                outcome.retrieved_gold_rank is not None
                for outcome in outcomes_by_pipeline[pipeline_id]
            ),
        )
        for pipeline_id in PipelineId
    )

    baseline_outcomes = tuple(
        outcome
        for outcome in report.case_outcomes
        if outcome.pipeline_id is report.baseline_pipeline_id
        and not outcome.gold_evidence_included
    )
    baseline_failure_stages = _build_baseline_failure_stage_summaries(
        baseline_outcomes=baseline_outcomes,
    )

    repair_report = build_repair_recommendation_report(comparison_report=report)

    return ExecutiveEvaluationReport(
        artifact_id=artifact.artifact_id,
        source_run_id=artifact.reference_run_id,
        tokenizer_name=artifact.provenance.tokenizer_name,
        evaluated_case_count=artifact.provenance.evaluation_case_count,
        retrieval_metric_k=report.retrieval_metric_k,
        baseline_pipeline_id=report.baseline_pipeline_id,
        pipeline_scorecards=scorecards,
        baseline_failure_stages=baseline_failure_stages,
        repair_sequence=repair_report.recommendations,
        controlled_context_finding=ControlledContextFinding(
            case_id=controlled_context_view.case.case_id,
            tokenizer_name=controlled_context_view.tokenizer_name,
            calibrated_context_tokens=controlled_context_view.calibrated_context_tokens,
            reserved_output_tokens=controlled_context_view.reserved_output_tokens,
            gold_evidence_rank_before_context=(
                controlled_context_view.loss_diagnosis.gold_evidence_rank_before_context
            ),
            verbose_gold_evidence_dropped=(
                controlled_context_view.verbose_audit.gold_evidence_dropped
            ),
            verbose_drop_reason=(
                controlled_context_view.verbose_audit.gold_evidence_drop_reason
                or ContextDropReason.BUDGET_EXHAUSTED
            ),
            compact_gold_evidence_included=(
                controlled_context_view.compact_citation.gold_evidence_included
            ),
            verbose_wrapper_tax_tokens=(
                controlled_context_view.verbose_audit.rendering_token_tax_tokens
            ),
            compact_wrapper_tax_tokens=(
                controlled_context_view.compact_citation.rendering_token_tax_tokens
            ),
        ),
    )


def render_executive_evaluation_report_markdown(
    *,
    report: ExecutiveEvaluationReport,
) -> str:
    """Render deterministic Markdown with report boundaries suitable for executive review."""

    baseline = report.baseline_scorecard
    strongest = report.strongest_scorecard
    inclusion_delta_points = (
        strongest.evidence_inclusion_rate.value - baseline.evidence_inclusion_rate.value
    ) * 100
    recall_delta_points = (
        strongest.retrieval_recall_at_k.value - baseline.retrieval_recall_at_k.value
    ) * 100
    mrr_delta = strongest.mrr_at_10 - baseline.mrr_at_10

    lines = [
        "# Executive Evaluation Report v1",
        "",
        "## Decision context",
        "",
        (
            "This is a local, fixed synthetic-data evidence-selection evaluation. It shows "
            "whether complete known evidence survives chunking, retrieval, ranking, and final "
            "context selection. It does not make a customer-data, production-readiness, "
            "generated-answer-grounding, or citation-correctness claim."
        ),
        "",
        f"- **Reviewed artifact:** `{report.artifact_id}`",
        f"- **Reference run:** `{report.source_run_id}`",
        f"- **Fixed evaluation cases:** {report.evaluated_case_count}",
        f"- **Reported retrieval metric:** Recall@{report.retrieval_metric_k}",
        f"- **Tokenizer provenance:** `{report.tokenizer_name}`",
        "",
        "## Executive finding",
        "",
        (
            f"On the fixed {report.evaluated_case_count}-case synthetic benchmark, "
            f"`{baseline.pipeline_id.value}` reached {baseline.evidence_inclusion_rate.value:.1%} "
            "evidence inclusion. The strongest reviewed pipeline, "
            f"`{strongest.pipeline_id.value}`, reached {strongest.evidence_inclusion_rate.value:.1%}. "
            f"That is a {inclusion_delta_points:.1f}-point evidence-inclusion difference, a "
            f"{recall_delta_points:.1f}-point Recall@{report.retrieval_metric_k} difference, and "
            f"a {mrr_delta:.3f} MRR@10 difference within this fixed benchmark."
        ),
        "",
        "## Four-pipeline scorecard",
        "",
        "| Pipeline | Recall@{k} | MRR@10 | Evidence inclusion | Dropped evidence among eligible candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    for scorecard in report.pipeline_scorecards:
        lines.append(
            "| "
            f"{scorecard.pipeline_label} (`{scorecard.pipeline_id.value}`) | "
            f"{scorecard.retrieval_recall_at_k.value:.1%} | "
            f"{scorecard.mrr_at_10:.3f} | "
            f"{scorecard.evidence_inclusion_rate.value:.1%} | "
            f"{scorecard.dropped_evidence_rate.value:.1%} "
            f"({scorecard.dropped_evidence_rate.numerator}/"
            f"{scorecard.dropped_evidence_eligible_case_count}) |"
        )

    lines.extend(
        [
            "",
            (
                "Dropped-evidence rate uses only cases whose complete gold evidence entered "
                "the first-stage candidate set. It is not divided by all fixed evaluation cases, "
                "because evidence that never reached candidates cannot be dropped by context packing."
            ),
            "",
            "## Where the baseline lost evidence",
            "",
        ]
    )
    if not report.baseline_failure_stages:
        lines.append(
            "No observed baseline evidence-loss stage was recorded. Do not add repair "
            "complexity without a controlled diagnostic."
        )
    else:
        lines.extend(
            [
                "| Evidence boundary | Affected fixed cases | Observed failure labels |",
                "|---|---:|---|",
            ]
        )
        for summary in report.baseline_failure_stages:
            labels = ", ".join(f"`{label.value}`" for label in summary.failure_labels)
            lines.append(
                "| "
                f"{summary.loss_stage.value.replace('_', ' ').title()} | "
                f"{summary.affected_case_count} | {labels} |"
            )

    finding = report.controlled_context_finding
    lines.extend(
        [
            "",
            "## Controlled context-budget finding",
            "",
            (
                "This is a separate local mechanism proof and is not counted as a standard "
                "four-pipeline benchmark failure."
            ),
            "",
            f"- **Fixed pressure case:** `{finding.case_id}`",
            f"- **Tokenizer:** `{finding.tokenizer_name}`",
            f"- **Same calibrated context window:** {finding.calibrated_context_tokens} tokens",
            f"- **Reserved output allowance:** {finding.reserved_output_tokens} tokens",
            f"- **Gold-evidence rank before context:** #{finding.gold_evidence_rank_before_context}",
            (
                f"- **Verbose audit wrappers:** gold evidence dropped by "
                f"`{finding.verbose_drop_reason.value}`; wrapper tax "
                f"{finding.verbose_wrapper_tax_tokens} tokens"
            ),
            (
                f"- **Compact citation wrappers:** gold evidence retained; wrapper tax "
                f"{finding.compact_wrapper_tax_tokens} tokens"
            ),
            "",
            "## Ordered repair sequence",
            "",
        ]
    )
    if not report.repair_sequence:
        lines.append(
            "No observed baseline repair is emitted. Preserve the benchmark and use a controlled "
            "diagnostic before adding speculative complexity."
        )
    else:
        for index, recommendation in enumerate(report.repair_sequence, start=1):
            labels = ", ".join(
                f"`{label.value}`" for label in recommendation.observed_failure_labels
            )
            supporting_pipeline = (
                f"`{recommendation.supporting_pipeline_id.value}`"
                if recommendation.supporting_pipeline_id is not None
                else "not applicable"
            )
            lines.extend(
                [
                    f"### {index}. {recommendation.recommendation_id.value}",
                    "",
                    f"- **Priority:** {recommendation.priority.value}",
                    f"- **Observed boundary:** {recommendation.loss_stage.value.replace('_', ' ')}",
                    f"- **Observed labels:** {labels}",
                    f"- **Action:** {recommendation.recommendation}",
                    f"- **Expected signal:** {recommendation.expected_signal}",
                    f"- **Metrics to watch:** {', '.join(recommendation.metrics_to_watch)}",
                    f"- **Trade-off:** {recommendation.trade_off}",
                    f"- **Supporting reviewed pipeline:** {supporting_pipeline}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Next evaluation gate",
            "",
            (
                "Before extending this result to a customer pilot, freeze an approved and "
                "privacy-reviewed evaluation set, preserve artifact provenance and trace fields, "
                "define acceptance thresholds before intervention, and add final-answer grounding "
                "and citation evaluation as a separate boundary."
            ),
            "",
            "## Evidence boundary",
            "",
            (
                "The report intentionally excludes raw source text, chunks, prompts, candidate "
                "scores, rendered context, and generated answers. It is evidence-selection and "
                "mechanism proof, not a claim that generated answers are correct."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_executive_evaluation_report(
    *,
    report: ExecutiveEvaluationReport,
    path: Path,
) -> None:
    """Write the deterministic executive Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_executive_evaluation_report_markdown(report=report),
        encoding="utf-8",
    )


def _build_baseline_failure_stage_summaries(
    *,
    baseline_outcomes: tuple[object, ...],
) -> tuple[BaselineFailureStageSummary, ...]:
    """Group observed baseline loss cases by the exact stage where evidence became unusable."""

    summaries: list[BaselineFailureStageSummary] = []
    for stage in _OBSERVED_BASELINE_STAGE_ORDER:
        matching = [
            outcome
            for outcome in baseline_outcomes
            if outcome.loss_stage is stage
        ]
        if not matching:
            continue
        labels = tuple(
            sorted(
                {
                    label
                    for outcome in matching
                    for label in outcome.failure_labels
                },
                key=lambda label: label.value,
            )
        )
        summaries.append(
            BaselineFailureStageSummary(
                loss_stage=stage,
                affected_case_count=len(matching),
                failure_labels=labels,
            )
        )
    return tuple(summaries)
