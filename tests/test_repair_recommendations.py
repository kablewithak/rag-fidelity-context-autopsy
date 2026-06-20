from __future__ import annotations

from pathlib import Path

from rag_lab.comparison import PipelineId, build_comparison_report
from rag_lab.comparison_artifacts import DEFAULT_BASELINE_ARTIFACT_PATH, load_baseline_artifact
from rag_lab.repair_recommendations import (
    REPAIR_POLICIES,
    RepairRecommendationId,
    build_repair_recommendation_report,
    render_repair_recommendations_markdown,
    write_repair_recommendation_json,
    write_repair_recommendations_markdown,
)
from rag_lab.schemas import FailureLabel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _baseline():
    return load_baseline_artifact(PROJECT_ROOT / DEFAULT_BASELINE_ARTIFACT_PATH)


def test_repair_policies_cover_each_supported_failure_label_once() -> None:
    policy_labels = [
        label for policy in REPAIR_POLICIES for label in policy.failure_labels
    ]

    assert set(policy_labels) == set(FailureLabel)
    assert len(policy_labels) == len(set(policy_labels))


def test_reviewed_baseline_produces_ordered_evidence_backed_recommendations() -> None:
    artifact = _baseline()

    report = build_repair_recommendation_report(comparison_report=artifact.report)

    assert report.source_run_id == artifact.report.run_id
    assert report.baseline_pipeline_id is PipelineId.CHAR_DENSE_NAIVE
    assert report.evaluated_case_count == 18
    assert report.recommendations
    assert {
        recommendation.recommendation_id for recommendation in report.recommendations
    } == {
        RepairRecommendationId.SENTENCE_AWARE_TOKEN_CHUNKING,
        RepairRecommendationId.HYBRID_RETRIEVAL,
        RepairRecommendationId.CROSS_ENCODER_RERANKING,
    }
    assert all(
        recommendation.observed_failure_count >= 1
        for recommendation in report.recommendations
    )
    assert all(
        "raw document" not in recommendation.evidence_summary.lower()
        for recommendation in report.recommendations
    )


def test_report_rendering_is_deterministic_and_uses_the_valid_gate_command() -> None:
    artifact = _baseline()
    report = build_repair_recommendation_report(comparison_report=artifact.report)

    first_render = render_repair_recommendations_markdown(report=report)
    second_render = render_repair_recommendations_markdown(report=report)

    assert first_render == second_render
    assert "# Deterministic Repair Recommendations" in first_render
    assert "--tiktoken-encoding cl100k_base" in first_render
    assert "--verify" not in first_render


def test_clean_selected_baseline_emits_no_speculative_recommendations() -> None:
    artifact = _baseline()
    clean_baseline_report = build_comparison_report(
        run_id="clean_repair_recommendation_fixture",
        baseline_pipeline_id=PipelineId.TOKEN_HYBRID_RERANK_BUDGETED,
        pipeline_definitions=artifact.report.pipeline_definitions,
        case_outcomes=artifact.report.case_outcomes,
        retrieval_metric_k=artifact.report.retrieval_metric_k,
    )

    report = build_repair_recommendation_report(comparison_report=clean_baseline_report)

    assert report.observed_failure_label_counts == {}
    assert report.recommendations == ()
    assert "No repair recommendation is emitted" in render_repair_recommendations_markdown(
        report=report
    )


def test_json_and_markdown_writers_create_local_privacy_bounded_outputs(tmp_path: Path) -> None:
    artifact = _baseline()
    report = build_repair_recommendation_report(comparison_report=artifact.report)
    json_path = tmp_path / "recommendations.json"
    markdown_path = tmp_path / "recommendations.md"

    write_repair_recommendation_json(report=report, path=json_path)
    write_repair_recommendations_markdown(report=report, path=markdown_path)

    assert json_path.exists()
    assert markdown_path.exists()
    assert '"schema_version": "repair_recommendation_report_v1"' in json_path.read_text(
        encoding="utf-8"
    )
    assert markdown_path.read_text(encoding="utf-8") == render_repair_recommendations_markdown(
        report=report
    )
