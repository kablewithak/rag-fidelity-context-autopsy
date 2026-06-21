from __future__ import annotations

from pathlib import Path

from rag_lab.comparison_artifacts import (
    DEFAULT_BASELINE_ARTIFACT_PATH,
    load_baseline_artifact,
)
from rag_lab.public_transfer_artifacts import (
    PublicTransferComparisonArtifact,
    load_public_transfer_artifact,
    render_public_transfer_markdown,
    write_public_transfer_artifact,
)
from rag_lab.public_transfer_runtime import (
    PublicTransferCaseReference,
    PublicTransferRunProvenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _artifact() -> PublicTransferComparisonArtifact:
    baseline = load_baseline_artifact(PROJECT_ROOT / DEFAULT_BASELINE_ARTIFACT_PATH)
    case_ids = sorted({outcome.case_id for outcome in baseline.report.case_outcomes})
    references = tuple(
        PublicTransferCaseReference(
            case_id=case_id,
            external_case_id=f"external-{case_id}",
            source_document_id="public_fixture_doc",
            source_document_text_sha256="0" * 64,
            source_answer_start=0,
            question_sha256="1" * 64,
            answer_sha256="2" * 64,
            gold_evidence_sha256="3" * 64,
        )
        for case_id in case_ids
    )
    provenance = PublicTransferRunProvenance(
        fixture_format_version="public_transfer_fixture_v1",
        external_dataset_id="fixture_public_dataset",
        dataset_version="fixture-v1",
        source_url="https://example.test/public-fixture",
        source_sha256="4" * 64,
        license_name="fixture-license",
        fixture_manifest_sha256="5" * 64,
        corpus_contract_sha256="6" * 64,
        case_contract_sha256="7" * 64,
        tokenizer_name=baseline.provenance.tokenizer_name,
        embedding_model_name=baseline.provenance.embedding_model_name,
        reranker_model_name=baseline.provenance.reranker_model_name,
        device=baseline.provenance.device,
        execution_config=baseline.provenance.execution_config,
        source_document_count=10,
        evaluation_case_count=len(references),
    )
    return PublicTransferComparisonArtifact(
        artifact_id="public_transfer_fixture_run",
        reference_run_id=baseline.report.run_id,
        provenance=provenance,
        case_references=references,
        report=baseline.report,
    )


def test_public_transfer_artifact_round_trips_without_raw_source_text(tmp_path: Path) -> None:
    artifact = _artifact()
    output_path = tmp_path / "public_transfer.json"

    write_public_transfer_artifact(artifact=artifact, path=output_path)
    loaded = load_public_transfer_artifact(output_path)

    assert loaded == artifact
    payload = output_path.read_text(encoding="utf-8")
    assert "Super Bowl" not in payload
    assert "public_transfer_fixture_run" in payload


def test_public_transfer_readout_labels_the_result_as_a_separate_measurement() -> None:
    readout = render_public_transfer_markdown(artifact=_artifact())

    assert "# Public-Corpus RAG Transfer Run" in readout
    assert "separate from the 18-case synthetic benchmark" in readout
    assert "No regression gate is applied" in readout
