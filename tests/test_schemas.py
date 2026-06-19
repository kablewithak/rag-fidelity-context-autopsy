from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_lab.schemas import (
    DocumentType,
    EvaluationCase,
    EvidenceLossStage,
    FailureDiagnosis,
    FailureLabel,
    QueryType,
)


def valid_case_payload() -> dict[str, object]:
    return {
        "case_id": "valid_case_001",
        "document_type": DocumentType.FAQ,
        "query_type": QueryType.FAQ_QUERY,
        "query": "Where can an owner export workspace data?",
        "gold_evidence_text": "Workspace owners can export data from Settings > Data Management > Export.",
        "gold_answer": "Owners export data from Settings > Data Management > Export.",
        "expected_failure_mode": FailureLabel.DUPLICATE_CONTEXT_WASTE,
        "source_doc_id": "faq",
        "diagnostic_note": "This checks that the schema accepts a complete fixed diagnostic case.",
    }


def test_evaluation_case_accepts_valid_contract() -> None:
    case = EvaluationCase.model_validate(valid_case_payload())

    assert case.case_id == "valid_case_001"
    assert case.expected_failure_mode is FailureLabel.DUPLICATE_CONTEXT_WASTE


def test_evaluation_case_rejects_unknown_fields() -> None:
    payload = valid_case_payload()
    payload["untracked_field"] = "must fail"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvaluationCase.model_validate(payload)


def test_evaluation_case_rejects_non_snake_case_identifier() -> None:
    payload = valid_case_payload()
    payload["case_id"] = "Invalid Case ID"

    with pytest.raises(ValidationError, match="pattern"):
        EvaluationCase.model_validate(payload)


def test_failure_diagnosis_rejects_duplicate_labels() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        FailureDiagnosis(
            case_id="valid_case_001",
            failure_labels=[
                FailureLabel.GOLD_EVIDENCE_SPLIT,
                FailureLabel.GOLD_EVIDENCE_SPLIT,
            ],
            loss_stage=EvidenceLossStage.CHUNKING,
            evidence_summary="The clause was split across a character boundary.",
            repair_recommendation="Use sentence-aware token chunking.",
        )
