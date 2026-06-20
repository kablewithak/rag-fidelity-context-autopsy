from __future__ import annotations

from hashlib import sha256

import pytest
from pydantic import ValidationError

from rag_lab.comparison import (
    DEFAULT_PIPELINE_DEFINITIONS,
    CasePipelineOutcome,
    ComparisonInputError,
    ContextSelectionMode,
    PipelineId,
    TraceReference,
    build_comparison_report,
)
from rag_lab.schemas import EvidenceLossStage, FailureLabel


CASE_IDS = ("legal_termination_001", "code_sso_error_013", "token_context_notice_018")


def _trace_reference(*, pipeline_id: PipelineId, case_id: str) -> TraceReference:
    trace_id = f"trace:{pipeline_id.value}:{case_id}"
    return TraceReference(
        trace_id=trace_id,
        trace_sha256=sha256(trace_id.encode("utf-8")).hexdigest(),
    )


def _outcome(
    *,
    pipeline_id: PipelineId,
    case_id: str,
    retrieved_rank: int | None,
    reranked_rank: int | None = None,
    included: bool,
    loss_stage: EvidenceLossStage | None = None,
    labels: list[FailureLabel] | None = None,
) -> CasePipelineOutcome:
    return CasePipelineOutcome(
        pipeline_id=pipeline_id,
        case_id=case_id,
        requested_top_k=8,
        retrieved_gold_rank=retrieved_rank,
        reranked_gold_rank=reranked_rank,
        gold_evidence_included=included,
        loss_stage=loss_stage,
        failure_labels=labels or [],
        trace_references=[_trace_reference(pipeline_id=pipeline_id, case_id=case_id)],
    )


def _outcomes_for_all_pipelines() -> list[CasePipelineOutcome]:
    outcomes: list[CasePipelineOutcome] = []
    for pipeline_id in PipelineId:
        if pipeline_id is PipelineId.CHAR_DENSE_NAIVE:
            outcomes.extend(
                [
                    _outcome(
                        pipeline_id=pipeline_id,
                        case_id=CASE_IDS[0],
                        retrieved_rank=1,
                        included=True,
                    ),
                    _outcome(
                        pipeline_id=pipeline_id,
                        case_id=CASE_IDS[1],
                        retrieved_rank=None,
                        included=False,
                        loss_stage=EvidenceLossStage.RETRIEVAL,
                        labels=[FailureLabel.DENSE_RETRIEVAL_MISS],
                    ),
                    _outcome(
                        pipeline_id=pipeline_id,
                        case_id=CASE_IDS[2],
                        retrieved_rank=2,
                        included=False,
                        loss_stage=EvidenceLossStage.CONTEXT_ASSEMBLY,
                        labels=[
                            FailureLabel.RELEVANT_CHUNK_DROPPED_BY_BUDGET,
                            FailureLabel.CONTEXT_BUDGET_EXCEEDED,
                        ],
                    ),
                ]
            )
            continue

        outcomes.extend(
            [
                _outcome(
                    pipeline_id=pipeline_id,
                    case_id=CASE_IDS[0],
                    retrieved_rank=1,
                    included=True,
                ),
                _outcome(
                    pipeline_id=pipeline_id,
                    case_id=CASE_IDS[1],
                    retrieved_rank=2,
                    reranked_rank=1 if pipeline_id is PipelineId.TOKEN_HYBRID_RERANK_BUDGETED else None,
                    included=True,
                ),
                _outcome(
                    pipeline_id=pipeline_id,
                    case_id=CASE_IDS[2],
                    retrieved_rank=3,
                    reranked_rank=1 if pipeline_id is PipelineId.TOKEN_HYBRID_RERANK_BUDGETED else None,
                    included=True,
                ),
            ]
        )
    return outcomes


def test_default_definitions_are_the_required_four_ablation_pipelines() -> None:
    assert {definition.pipeline_id for definition in DEFAULT_PIPELINE_DEFINITIONS} == set(PipelineId)
    assert (
        next(
            definition
            for definition in DEFAULT_PIPELINE_DEFINITIONS
            if definition.pipeline_id is PipelineId.TOKEN_HYBRID_RERANK_BUDGETED
        ).context_selection_mode
        is ContextSelectionMode.TOKEN_BUDGETED
    )


def test_case_outcome_rejects_included_evidence_without_a_rank() -> None:
    with pytest.raises(ValidationError, match="requires a retrieval or reranking rank"):
        _outcome(
            pipeline_id=PipelineId.CHAR_DENSE_NAIVE,
            case_id=CASE_IDS[0],
            retrieved_rank=None,
            included=True,
        )


def test_case_outcome_rejects_context_loss_without_retrieved_evidence() -> None:
    with pytest.raises(ValidationError, match="context loss requires gold evidence"):
        _outcome(
            pipeline_id=PipelineId.CHAR_DENSE_NAIVE,
            case_id=CASE_IDS[0],
            retrieved_rank=None,
            included=False,
            loss_stage=EvidenceLossStage.CONTEXT_ASSEMBLY,
            labels=[FailureLabel.RELEVANT_CHUNK_DROPPED_BY_BUDGET],
        )


def test_comparison_report_calculates_traceable_metrics_and_deltas() -> None:
    report = build_comparison_report(
        run_id="comparison_fixture_001",
        baseline_pipeline_id=PipelineId.CHAR_DENSE_NAIVE,
        case_outcomes=_outcomes_for_all_pipelines(),
    )

    baseline = next(
        metric
        for metric in report.pipeline_metrics
        if metric.pipeline_id is PipelineId.CHAR_DENSE_NAIVE
    )
    intervention = next(
        metric
        for metric in report.pipeline_metrics
        if metric.pipeline_id is PipelineId.TOKEN_HYBRID_RERANK_BUDGETED
    )
    delta = next(
        item
        for item in report.metric_deltas
        if item.comparison_pipeline_id is PipelineId.TOKEN_HYBRID_RERANK_BUDGETED
    )

    assert baseline.retrieval_recall_at_k.numerator == 2
    assert baseline.retrieval_recall_at_k.denominator == 3
    assert baseline.mrr_at_10 == pytest.approx(0.5)
    assert baseline.evidence_inclusion_rate.value == pytest.approx(1 / 3)
    assert baseline.dropped_evidence_rate.value == pytest.approx(1 / 2)
    assert baseline.failure_label_counts[FailureLabel.DENSE_RETRIEVAL_MISS] == 1
    assert baseline.loss_stage_counts[EvidenceLossStage.CONTEXT_ASSEMBLY] == 1

    assert intervention.retrieval_recall_at_k.value == 1.0
    assert intervention.mrr_at_10 == 1.0
    assert intervention.evidence_inclusion_rate.value == 1.0
    assert intervention.dropped_evidence_rate.value == 0.0
    assert delta.evidence_inclusion_rate_delta == pytest.approx(2 / 3)
    assert delta.dropped_evidence_rate_delta == pytest.approx(-1 / 2)


def test_comparison_rejects_missing_case_coverage_for_one_pipeline() -> None:
    outcomes = _outcomes_for_all_pipelines()
    incomplete_outcomes = [
        outcome
        for outcome in outcomes
        if not (
            outcome.pipeline_id is PipelineId.TOKEN_DENSE_NAIVE
            and outcome.case_id == CASE_IDS[-1]
        )
    ]

    with pytest.raises(ComparisonInputError, match="same non-empty case IDs"):
        build_comparison_report(
            run_id="comparison_fixture_incomplete",
            baseline_pipeline_id=PipelineId.CHAR_DENSE_NAIVE,
            case_outcomes=incomplete_outcomes,
        )
