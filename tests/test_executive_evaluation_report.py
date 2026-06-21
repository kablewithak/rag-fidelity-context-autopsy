from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.comparison import PipelineId
from rag_lab.comparison_artifacts import DEFAULT_BASELINE_ARTIFACT_PATH, load_baseline_artifact
from rag_lab.context_autopsy_explorer import load_context_autopsy_case_view
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.executive_evaluation_report import (
    ExecutiveEvaluationReportError,
    build_executive_evaluation_report,
    load_executive_evaluation_report,
    render_executive_evaluation_report_markdown,
)
from rag_lab.repair_recommendations import RepairRecommendationId
from rag_lab.schemas import EvidenceLossStage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _report():
    return load_executive_evaluation_report(project_root=PROJECT_ROOT)


def test_executive_report_has_fixed_four_pipeline_scorecard() -> None:
    report = _report()

    assert report.evaluated_case_count == 18
    assert report.retrieval_metric_k == 5
    assert {scorecard.pipeline_id for scorecard in report.pipeline_scorecards} == set(PipelineId)

    baseline = report.baseline_scorecard
    strongest = report.strongest_scorecard

    assert baseline.pipeline_id is PipelineId.CHAR_DENSE_NAIVE
    assert baseline.evidence_inclusion_rate.numerator == 13
    assert baseline.evidence_inclusion_rate.denominator == 18
    assert baseline.dropped_evidence_rate.numerator == 0
    assert baseline.dropped_evidence_eligible_case_count == 14
    assert baseline.dropped_evidence_rate.denominator == 14
    assert strongest.pipeline_id is PipelineId.TOKEN_HYBRID_RERANK_BUDGETED
    assert strongest.evidence_inclusion_rate.numerator == 18
    assert strongest.evidence_inclusion_rate.denominator == 18
    assert strongest.dropped_evidence_eligible_case_count == 18
    assert strongest.dropped_evidence_rate.denominator == 18


def test_executive_report_groups_observed_baseline_loss_stages() -> None:
    report = _report()

    summaries = {
        summary.loss_stage: summary
        for summary in report.baseline_failure_stages
    }

    assert summaries[EvidenceLossStage.CHUNKING].affected_case_count == 3
    assert summaries[EvidenceLossStage.RETRIEVAL].affected_case_count == 1
    assert summaries[EvidenceLossStage.RANKING].affected_case_count == 1
    assert EvidenceLossStage.CONTEXT_ASSEMBLY not in summaries


def test_executive_report_emits_only_observed_baseline_repairs() -> None:
    report = _report()

    assert [recommendation.recommendation_id for recommendation in report.repair_sequence] == [
        RepairRecommendationId.SENTENCE_AWARE_TOKEN_CHUNKING,
        RepairRecommendationId.HYBRID_RETRIEVAL,
        RepairRecommendationId.CROSS_ENCODER_RERANKING,
    ]


def test_executive_report_labels_context_as_separate_controlled_proof() -> None:
    report = _report()
    finding = report.controlled_context_finding

    assert finding.case_id == "token_context_notice_018"
    assert finding.gold_evidence_rank_before_context == 3
    assert finding.verbose_gold_evidence_dropped is True
    assert finding.compact_gold_evidence_included is True


def test_executive_report_render_is_deterministic_and_privacy_bounded() -> None:
    report = _report()

    first = render_executive_evaluation_report_markdown(report=report)
    second = render_executive_evaluation_report_markdown(report=report)

    assert first == second

    # Privacy testing must inspect actual payloads, not the report's prose explaining
    # that those payloads are excluded. The report may legitimately describe its
    # privacy boundary using terms such as "raw source text".
    fixed_cases = load_evaluation_cases(PROJECT_ROOT / "data" / "eval_cases.jsonl")
    raw_case_payloads = [
        payload
        for case in fixed_cases
        for payload in (case.query, case.gold_evidence_text, case.gold_answer)
    ]
    assert all(payload not in first for payload in raw_case_payloads)

    assert "candidate scores" in first
    assert "token_context_notice_018" in first
    assert "Dropped evidence among eligible candidates" in first
    assert "0.0% (0/14)" in first


def test_executive_report_rejects_missing_pipeline_metric_coverage() -> None:
    artifact = load_baseline_artifact(PROJECT_ROOT / DEFAULT_BASELINE_ARTIFACT_PATH)
    context_view = load_context_autopsy_case_view(project_root=PROJECT_ROOT)
    incomplete_report = artifact.report.model_copy(
        update={"pipeline_metrics": artifact.report.pipeline_metrics[:-1]}
    )
    incomplete_artifact = artifact.model_copy(update={"report": incomplete_report})

    with pytest.raises(ExecutiveEvaluationReportError, match="metrics for every fixed pipeline"):
        build_executive_evaluation_report(
            artifact=incomplete_artifact,
            controlled_context_view=context_view,
        )
