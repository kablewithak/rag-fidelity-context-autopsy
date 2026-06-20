from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.comparison import PipelineId, build_comparison_report
from rag_lab.comparison_artifacts import (
    DEFAULT_BASELINE_ARTIFACT_PATH,
    DEFAULT_BASELINE_READOUT_PATH,
    ComparisonRegressionError,
    canonical_json_sha256,
    load_baseline_artifact,
    render_executive_markdown,
    verify_against_baseline,
)
from rag_lab.schemas import EvidenceLossStage, FailureLabel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _baseline():
    return load_baseline_artifact(PROJECT_ROOT / DEFAULT_BASELINE_ARTIFACT_PATH)


def _candidate_from_baseline(*, outcomes=None):
    baseline = _baseline()
    return build_comparison_report(
        run_id="comparison_regression_check_fixture",
        baseline_pipeline_id=baseline.report.baseline_pipeline_id,
        pipeline_definitions=baseline.report.pipeline_definitions,
        case_outcomes=outcomes or baseline.report.case_outcomes,
        retrieval_metric_k=baseline.report.retrieval_metric_k,
    )


def test_committed_baseline_artifact_is_valid_and_has_complete_coverage() -> None:
    artifact = _baseline()

    assert artifact.artifact_id == "four_pipeline_baseline_v1"
    assert artifact.provenance.evaluation_case_count == 18
    assert len(artifact.report.case_outcomes) == 72
    assert artifact.report.retrieval_metric_k == 5


def test_committed_readout_is_the_deterministic_render_of_the_artifact() -> None:
    artifact = _baseline()
    committed_readout = (PROJECT_ROOT / DEFAULT_BASELINE_READOUT_PATH).read_text(encoding="utf-8")

    assert committed_readout == render_executive_markdown(artifact=artifact)
    assert "Recall@5" in committed_readout
    assert "## Deterministic repair sequence" in committed_readout
    assert "`sentence_aware_token_chunking`" in committed_readout
    assert "`hybrid_retrieval`" in committed_readout
    assert "`cross_encoder_reranking`" in committed_readout
    assert "--tiktoken-encoding cl100k_base" in committed_readout
    assert "--verify" not in committed_readout
    assert "model-version stability" in committed_readout


def test_baseline_regression_gate_accepts_same_result_with_a_new_run_id() -> None:
    artifact = _baseline()
    candidate = _candidate_from_baseline()

    result = verify_against_baseline(
        baseline=artifact,
        candidate_report=candidate,
        candidate_provenance=artifact.provenance,
    )

    assert result.passed is True
    assert result.checked_pipeline_count == 4
    assert result.checked_case_count == 18


def test_baseline_regression_gate_rejects_loss_of_previously_included_evidence() -> None:
    artifact = _baseline()
    regressed_outcomes = list(artifact.report.case_outcomes)
    target_index = next(
        index
        for index, outcome in enumerate(regressed_outcomes)
        if outcome.pipeline_id is PipelineId.TOKEN_HYBRID_NAIVE
        and outcome.case_id == "code_sso_error_013"
    )
    target = regressed_outcomes[target_index]
    regressed_outcomes[target_index] = target.model_copy(
        update={
            "gold_evidence_included": False,
            "loss_stage": EvidenceLossStage.RANKING,
            "failure_labels": [FailureLabel.RELEVANT_CHUNK_RANKED_TOO_LOW],
        }
    )
    candidate = _candidate_from_baseline(outcomes=regressed_outcomes)

    with pytest.raises(ComparisonRegressionError, match="lost evidence that baseline included"):
        verify_against_baseline(
            baseline=artifact,
            candidate_report=candidate,
            candidate_provenance=artifact.provenance,
        )


def test_baseline_regression_gate_rejects_provenance_change() -> None:
    artifact = _baseline()
    candidate = _candidate_from_baseline()
    changed_provenance = artifact.provenance.model_copy(
        update={"tokenizer_name": "diagnostic:unicode_codepoint_v1"}
    )

    with pytest.raises(ComparisonRegressionError, match="provenance mismatch for tokenizer_name"):
        verify_against_baseline(
            baseline=artifact,
            candidate_report=candidate,
            candidate_provenance=changed_provenance,
        )


def test_canonical_json_hash_is_key_order_independent() -> None:
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256({"a": 1, "b": 2})
