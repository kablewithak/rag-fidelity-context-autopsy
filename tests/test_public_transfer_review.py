from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.comparison import DEFAULT_PIPELINE_DEFINITIONS, CasePipelineOutcome, PipelineId, TraceReference, build_comparison_report
from rag_lab.comparison_artifacts import (
    ComparisonBaselineArtifact,
    ComparisonRegressionPolicy,
    ComparisonRunProvenance,
)
from rag_lab.comparison_runner import ComparisonExecutionConfig
from rag_lab.public_transfer_artifacts import PublicTransferComparisonArtifact
from rag_lab.public_transfer_review import (
    EXPECTED_PUBLIC_CASE_COUNT,
    PublicTransferReviewError,
    REVIEWED_PUBLIC_TRANSFER_ARTIFACT_ID,
    assert_public_transfer_review_matches,
    assert_review_boundary,
    build_reviewed_public_transfer_artifact,
    render_public_transfer_review_markdown,
    write_public_transfer_review_markdown,
)
from rag_lab.public_transfer_runtime import PublicTransferCaseReference, PublicTransferRunProvenance


def _outcomes(case_ids: tuple[str, ...]) -> list[CasePipelineOutcome]:
    values: list[CasePipelineOutcome] = []
    for pipeline_id in PipelineId:
        for case_id in case_ids:
            values.append(CasePipelineOutcome(
                pipeline_id=pipeline_id,
                case_id=case_id,
                requested_top_k=8,
                retrieved_gold_rank=1,
                reranked_gold_rank=1 if pipeline_id is PipelineId.TOKEN_HYBRID_RERANK_BUDGETED else None,
                gold_evidence_included=True,
                trace_references=[TraceReference(trace_id=f"review:{pipeline_id.value}:{case_id}", trace_sha256="a" * 64)],
            ))
    return values


def _synthetic_baseline() -> ComparisonBaselineArtifact:
    case_ids = tuple(f"synthetic_case_{index:03d}" for index in range(18))
    report = build_comparison_report(run_id="synthetic_review_fixture", baseline_pipeline_id=PipelineId.CHAR_DENSE_NAIVE, pipeline_definitions=DEFAULT_PIPELINE_DEFINITIONS, case_outcomes=_outcomes(case_ids), retrieval_metric_k=5)
    provenance = ComparisonRunProvenance(tokenizer_name="tiktoken:cl100k_base", embedding_model_name="sentence-transformers:fixture", reranker_model_name="sentence-transformers-cross-encoder:fixture", device="cpu", execution_config=ComparisonExecutionConfig(), corpus_manifest_sha256="b" * 64, evaluation_cases_sha256="c" * 64, source_document_count=7, evaluation_case_count=18)
    return ComparisonBaselineArtifact(
        artifact_id="four_pipeline_baseline_v1",
        reference_run_id=report.run_id,
        provenance=provenance,
        regression_policy=ComparisonRegressionPolicy(),
        report=report,
    )


def _public_candidate() -> PublicTransferComparisonArtifact:
    case_ids = tuple(f"public_case_{index:03d}" for index in range(EXPECTED_PUBLIC_CASE_COUNT))
    report = build_comparison_report(run_id="public_review_fixture", baseline_pipeline_id=PipelineId.CHAR_DENSE_NAIVE, pipeline_definitions=DEFAULT_PIPELINE_DEFINITIONS, case_outcomes=_outcomes(case_ids), retrieval_metric_k=5)
    provenance = PublicTransferRunProvenance(fixture_format_version="public_transfer_fixture_v1", external_dataset_id="squad_v1.1_dev", dataset_version="1.1", source_url="https://example.test/squad.json", source_sha256="d" * 64, license_name="CC BY-SA 4.0", fixture_manifest_sha256="e" * 64, corpus_contract_sha256="f" * 64, case_contract_sha256="1" * 64, tokenizer_name="tiktoken:cl100k_base", embedding_model_name="sentence-transformers:fixture", reranker_model_name="sentence-transformers-cross-encoder:fixture", device="cpu", execution_config=ComparisonExecutionConfig(), source_document_count=10, evaluation_case_count=30)
    references = tuple(PublicTransferCaseReference(case_id=case_id, external_case_id=f"external-{index:03d}", source_document_id=f"public_doc_{index // 3:03d}", source_document_text_sha256="2" * 64, source_answer_start=index, question_sha256="3" * 64, answer_sha256="4" * 64, gold_evidence_sha256="5" * 64) for index, case_id in enumerate(case_ids))
    return PublicTransferComparisonArtifact(artifact_id="public_transfer_squad_v1_dev_v1_review_candidate", reference_run_id=report.run_id, provenance=provenance, case_references=references, report=report)


def test_review_renderer_keeps_boundaries_separate() -> None:
    baseline = _synthetic_baseline()
    reviewed = build_reviewed_public_transfer_artifact(source_artifact=_public_candidate())
    rendered = render_public_transfer_review_markdown(public_artifact=reviewed, synthetic_baseline=baseline)
    assert reviewed.artifact_id == REVIEWED_PUBLIC_TRANSFER_ARTIFACT_ID
    assert "## Controlled synthetic benchmark" in rendered
    assert "## Public-corpus transfer probe" in rendered
    assert "must not be pooled" in rendered
    assert "## Non-claims" in rendered


def test_review_boundary_rejects_changed_public_case_count() -> None:
    baseline = _synthetic_baseline()
    reviewed = build_reviewed_public_transfer_artifact(source_artifact=_public_candidate())
    invalid = reviewed.model_copy(update={"provenance": reviewed.provenance.model_copy(update={"evaluation_case_count": EXPECTED_PUBLIC_CASE_COUNT - 1})})
    with pytest.raises(PublicTransferReviewError, match="fixed 30-case fixture"):
        assert_review_boundary(public_artifact=invalid, synthetic_baseline=baseline)


def test_written_review_must_match_its_artifacts(tmp_path: Path) -> None:
    baseline = _synthetic_baseline()
    reviewed = build_reviewed_public_transfer_artifact(source_artifact=_public_candidate())
    report_path = tmp_path / "review.md"
    write_public_transfer_review_markdown(public_artifact=reviewed, synthetic_baseline=baseline, path=report_path)
    assert_public_transfer_review_matches(public_artifact=reviewed, synthetic_baseline=baseline, path=report_path)
    report_path.write_text("stale", encoding="utf-8")
    with pytest.raises(PublicTransferReviewError, match="does not match"):
        assert_public_transfer_review_matches(public_artifact=reviewed, synthetic_baseline=baseline, path=report_path)
