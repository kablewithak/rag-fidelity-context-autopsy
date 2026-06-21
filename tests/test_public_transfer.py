from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from rag_lab.public_transfer import (
    PublicTransferError,
    build_squad_v1_transfer_fixture,
    build_squad_v1_transfer_fixture_from_bytes,
    load_public_transfer_fixture,
    write_public_transfer_fixture,
)


def _paragraph(context: str, question_prefix: str) -> dict[str, object]:
    answer = "evidence"
    answer_start = context.index(answer)
    return {
        "context": context,
        "qas": [
            {
                "id": f"{question_prefix}-a",
                "question": f"Where is {answer}?",
                "answers": [{"text": answer, "answer_start": answer_start}],
            },
            {
                "id": f"{question_prefix}-b",
                "question": "What does the record retain?",
                "answers": [{"text": answer, "answer_start": answer_start}],
            },
        ],
    }


def _squad_payload() -> dict[str, object]:
    return {
        "version": "1.1",
        "data": [
            {
                "title": "First Article",
                "paragraphs": [
                    _paragraph("First sentence contains evidence. Second sentence adds context.", "one"),
                    _paragraph("Another paragraph preserves evidence for retrieval.", "two"),
                ],
            },
            {
                "title": "Second Article",
                "paragraphs": [
                    _paragraph("A separate source keeps evidence in its opening sentence.", "three"),
                    _paragraph("The second source paragraph also contains evidence.", "four"),
                ],
            },
        ],
    }


def _fixture():
    return build_squad_v1_transfer_fixture(
        _squad_payload(),
        source_sha256="a" * 64,
        document_count=2,
        paragraphs_per_document=2,
        cases_per_document=2,
    )


def test_builds_deterministic_grouped_documents_and_answerable_cases() -> None:
    fixture = _fixture()

    assert fixture.manifest.document_count == 2
    assert len(fixture.documents) == 2
    assert len(fixture.cases) == 4
    assert fixture.manifest.case_ids == tuple(case.case_id for case in fixture.cases)

    for case in fixture.cases:
        document = next(
            item for item in fixture.documents if item.source_document_id == case.source_document_id
        )
        answer_end = case.source_answer_start + len(case.answer_text)
        assert document.text[case.source_answer_start:answer_end] == case.answer_text
        assert case.answer_text in case.gold_evidence_text
        assert case.source_document_text_sha256 == document.text_sha256


def test_preserves_raw_source_digest_when_materializing_bytes() -> None:
    raw_payload = b'{"version":"1.1","data":[{"title":"One","paragraphs":[{"context":"evidence is here.","qas":[{"id":"a","question":"Where?","answers":[{"text":"evidence","answer_start":0}]}]},{"context":"evidence returns.","qas":[{"id":"b","question":"Where again?","answers":[{"text":"evidence","answer_start":0}]}]}]}]}'

    fixture = build_squad_v1_transfer_fixture_from_bytes(
        raw_payload,
        document_count=1,
        paragraphs_per_document=2,
        cases_per_document=2,
    )

    assert fixture.manifest.source_sha256 == sha256(raw_payload).hexdigest()


def test_rejects_answer_spans_that_do_not_match_source_text() -> None:
    payload = _squad_payload()
    first_article_paragraphs = payload["data"][0]["paragraphs"]
    for paragraph in first_article_paragraphs:
        for qa in paragraph["qas"]:
            qa["answers"][0]["answer_start"] = 1

    with pytest.raises(PublicTransferError, match="enough articles"):
        build_squad_v1_transfer_fixture(
            payload,
            source_sha256="b" * 64,
            document_count=2,
            paragraphs_per_document=2,
            cases_per_document=2,
        )


def test_writes_and_loads_a_self_consistent_fixture(tmp_path: Path) -> None:
    fixture = _fixture()
    output_directory = tmp_path / "squad_v1_dev_v1"

    write_public_transfer_fixture(fixture, output_directory)
    loaded = load_public_transfer_fixture(output_directory)

    assert loaded == fixture
    assert (output_directory / "ATTRIBUTION.md").read_text(encoding="utf-8").startswith(
        "# Public Transfer Fixture Attribution"
    )

    with pytest.raises(PublicTransferError, match="refusing to replace"):
        write_public_transfer_fixture(fixture, output_directory)
