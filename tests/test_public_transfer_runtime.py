from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.comparison_runner import ComparisonExecutionConfig
from rag_lab.public_transfer import load_public_transfer_fixture
from rag_lab.public_transfer_runtime import (
    PublicTransferRuntimeError,
    adapt_public_transfer_fixture,
    build_public_transfer_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "data" / "public_transfer" / "squad_v1_dev_v1"


def test_public_fixture_adapts_to_the_existing_runner_boundary_without_synthetic_relabeling() -> None:
    fixture = load_public_transfer_fixture(FIXTURE_PATH)

    adapted = adapt_public_transfer_fixture(fixture)

    assert len(adapted.documents) == 10
    assert len(adapted.cases) == 30
    assert len(adapted.case_references) == 30
    assert adapted.documents[0].source_doc_id == fixture.documents[0].source_document_id
    assert adapted.cases[0].query == fixture.cases[0].question
    assert adapted.cases[0].gold_evidence_text == fixture.cases[0].gold_evidence_text
    assert adapted.case_references[0].external_case_id == fixture.cases[0].external_case_id
    assert all(
        runtime_case.source_doc_id == reference.source_document_id
        for runtime_case, reference in zip(
            adapted.cases,
            adapted.case_references,
            strict=True,
        )
    )


def test_adapter_rejects_a_public_answer_span_that_no_longer_matches_source_text() -> None:
    fixture = load_public_transfer_fixture(FIXTURE_PATH)
    first_case = fixture.cases[0]
    malformed_case = first_case.model_copy(
        update={"source_answer_start": first_case.source_answer_start + 1}
    )
    malformed_fixture = fixture.model_copy(
        update={"cases": (malformed_case, *fixture.cases[1:])}
    )

    with pytest.raises(PublicTransferRuntimeError, match="answer span"):
        adapt_public_transfer_fixture(malformed_fixture)


def test_public_transfer_provenance_hashes_fixture_identity_without_source_text() -> None:
    fixture = load_public_transfer_fixture(FIXTURE_PATH)
    adapted = adapt_public_transfer_fixture(fixture)

    provenance = build_public_transfer_provenance(
        manifest=fixture.manifest,
        adapter=adapted,
        token_counter_name="tiktoken:cl100k_base",
        embedding_model_name="fixture:embedding",
        reranker_model_name="fixture:reranker",
        device="cpu",
        execution_config=ComparisonExecutionConfig(),
    )

    assert provenance.external_dataset_id == "squad_v1.1_dev"
    assert provenance.source_document_count == 10
    assert provenance.evaluation_case_count == 30
    assert len(provenance.fixture_manifest_sha256) == 64
    assert len(provenance.corpus_contract_sha256) == 64
    assert len(provenance.case_contract_sha256) == 64
