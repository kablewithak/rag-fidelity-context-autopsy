from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.eval_cases import (
    EvaluationCaseLoadError,
    assert_gold_evidence_exists,
    load_evaluation_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_CASES_PATH = PROJECT_ROOT / "data" / "eval_cases.jsonl"
CORPUS_DIRECTORY = PROJECT_ROOT / "data" / "corpus"


def test_eval_case_gate_has_at_least_fifteen_fixed_cases() -> None:
    cases = load_evaluation_cases(EVAL_CASES_PATH)

    assert len(cases) >= 15
    assert len({case.case_id for case in cases}) == len(cases)


def test_every_gold_evidence_string_exists_in_its_declared_source_document() -> None:
    cases = load_evaluation_cases(EVAL_CASES_PATH)

    assert_gold_evidence_exists(cases, corpus_directory=CORPUS_DIRECTORY)


def test_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    duplicate_payload = (
        '{"case_id":"duplicate_case_001","document_type":"faq","query_type":"faq_query",'
        '"query":"Where can a user export workspace data?",'
        '"gold_evidence_text":"Users can export data from Settings > Export.",'
        '"gold_answer":"Use Settings > Export.",'
        '"expected_failure_mode":"duplicate_context_waste",'
        '"source_doc_id":"faq",'
        '"diagnostic_note":"A valid synthetic case used to test duplicate IDs."}'
    )
    path = tmp_path / "duplicate_cases.jsonl"
    path.write_text(f"{duplicate_payload}\n{duplicate_payload}\n", encoding="utf-8")

    with pytest.raises(EvaluationCaseLoadError, match="duplicate case_id"):
        load_evaluation_cases(path)
