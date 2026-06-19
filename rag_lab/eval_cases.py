"""JSONL loading utilities for deterministic, schema-validated evaluation cases."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from rag_lab.schemas import EvaluationCase


class EvaluationCaseLoadError(ValueError):
    """Raised when a JSONL eval asset cannot be parsed or validated."""

    def __init__(self, *, path: Path, line_number: int, reason: str) -> None:
        super().__init__(f"{path}:{line_number}: {reason}")
        self.path = path
        self.line_number = line_number
        self.reason = reason


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load non-empty JSONL lines as strict ``EvaluationCase`` contracts."""

    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationCaseLoadError(
                path=path,
                line_number=line_number,
                reason=f"invalid JSON: {error.msg}",
            ) from error

        try:
            case = EvaluationCase.model_validate(payload)
        except ValidationError as error:
            raise EvaluationCaseLoadError(
                path=path,
                line_number=line_number,
                reason=error.json(),
            ) from error

        if case.case_id in seen_case_ids:
            raise EvaluationCaseLoadError(
                path=path,
                line_number=line_number,
                reason=f"duplicate case_id: {case.case_id}",
            )

        seen_case_ids.add(case.case_id)
        cases.append(case)

    if not cases:
        raise EvaluationCaseLoadError(
            path=path,
            line_number=0,
            reason="evaluation case file contained no valid cases",
        )

    return cases


def source_document_path(*, corpus_directory: Path, source_doc_id: str) -> Path:
    """Return the canonical local source-document path for an eval case."""

    return corpus_directory / f"{source_doc_id}.txt"


def assert_gold_evidence_exists(
    cases: Iterable[EvaluationCase],
    *,
    corpus_directory: Path,
) -> None:
    """Fail fast when an eval case points to missing or mismatched synthetic evidence."""

    for case in cases:
        document_path = source_document_path(
            corpus_directory=corpus_directory,
            source_doc_id=case.source_doc_id,
        )
        if not document_path.exists():
            raise FileNotFoundError(
                f"{case.case_id} declares missing source document: {document_path}"
            )

        document_text = document_path.read_text(encoding="utf-8")
        if case.gold_evidence_text not in document_text:
            raise ValueError(
                f"{case.case_id} gold evidence is not an exact substring of {document_path.name}"
            )
