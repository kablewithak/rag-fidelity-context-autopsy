"""Deterministic public-corpus transfer-fixture contracts.

This module isolates a small, reproducible external-validity probe from the
repository's fixed synthetic benchmark. It materializes a curated subset of
SQuAD v1.1 development data into auditable corpus and case artifacts without
changing synthetic evaluation cases, baseline artifacts, or regression policy.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator


SQUAD_V1_DATASET_ID = "squad_v1.1_dev"
SQUAD_V1_DATASET_VERSION = "1.1"
SQUAD_V1_SOURCE_URL = (
    "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"
    "master/dataset/dev-v1.1.json"
)
SQUAD_V1_LICENSE = "CC BY-SA 4.0"
SQUAD_V1_ATTRIBUTION = (
    "Stanford Question Answering Dataset (SQuAD) v1.1 development set; "
    "Wikipedia-derived passages and crowdworker-authored questions."
)
FIXTURE_FORMAT_VERSION = "public_transfer_fixture_v1"
DEFAULT_DOCUMENT_COUNT = 10
DEFAULT_PARAGRAPHS_PER_DOCUMENT = 8
DEFAULT_CASES_PER_DOCUMENT = 3


class PublicTransferError(ValueError):
    """Raised when a public transfer fixture cannot be materialized safely."""


class PublicTransferDocument(BaseModel):
    """One grouped external source document retained for transfer retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_article_index: int = Field(ge=0)
    paragraph_indices: tuple[int, ...] = Field(min_length=1)
    text: str = Field(min_length=1)
    text_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("text_sha256")
    @classmethod
    def validate_text_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("text_sha256 must be a lowercase SHA-256 hex digest")
        return value


class PublicTransferCase(BaseModel):
    """One answerable external evaluation case with exact source provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    external_case_id: str = Field(min_length=1)
    external_dataset_id: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_document_text_sha256: str = Field(min_length=64, max_length=64)
    title: str = Field(min_length=1)
    paragraph_index: int = Field(ge=0)
    source_answer_start: int = Field(ge=0)
    question: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    gold_evidence_text: str = Field(min_length=1)

    @field_validator("source_document_text_sha256")
    @classmethod
    def validate_source_document_text_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                "source_document_text_sha256 must be a lowercase SHA-256 hex digest"
            )
        return value


class PublicTransferManifest(BaseModel):
    """Dataset provenance and deterministic selection contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: str = FIXTURE_FORMAT_VERSION
    external_dataset_id: str = SQUAD_V1_DATASET_ID
    dataset_version: str = SQUAD_V1_DATASET_VERSION
    source_url: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    license_name: str = SQUAD_V1_LICENSE
    attribution: str = SQUAD_V1_ATTRIBUTION
    article_selection_rule: str = Field(min_length=1)
    document_count: int = Field(gt=0)
    paragraphs_per_document: int = Field(gt=0)
    cases_per_document: int = Field(gt=0)
    source_document_ids: tuple[str, ...] = Field(min_length=1)
    case_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("source_url must use HTTPS")
        return value

    @field_validator("source_sha256")
    @classmethod
    def validate_source_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
        return value


class PublicTransferFixture(BaseModel):
    """The independently versioned public transfer corpus and evaluation cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: PublicTransferManifest
    documents: tuple[PublicTransferDocument, ...] = Field(min_length=1)
    cases: tuple[PublicTransferCase, ...] = Field(min_length=1)

    def assert_consistent(self) -> None:
        """Validate cross-file invariants before artifacts are written or loaded."""

        document_ids = {document.source_document_id for document in self.documents}
        if len(document_ids) != len(self.documents):
            raise PublicTransferError("duplicate source_document_id in public transfer fixture")

        document_hashes = {
            document.source_document_id: document.text_sha256 for document in self.documents
        }
        case_ids = {case.case_id for case in self.cases}
        external_case_ids = {case.external_case_id for case in self.cases}

        if len(case_ids) != len(self.cases):
            raise PublicTransferError("duplicate case_id in public transfer fixture")
        if len(external_case_ids) != len(self.cases):
            raise PublicTransferError("duplicate external_case_id in public transfer fixture")
        if tuple(document.source_document_id for document in self.documents) != (
            self.manifest.source_document_ids
        ):
            raise PublicTransferError("manifest source_document_ids do not match document order")
        if tuple(case.case_id for case in self.cases) != self.manifest.case_ids:
            raise PublicTransferError("manifest case_ids do not match case order")
        if len(self.documents) != self.manifest.document_count:
            raise PublicTransferError("manifest document_count does not match documents")
        expected_case_count = (
            self.manifest.document_count * self.manifest.cases_per_document
        )
        if len(self.cases) != expected_case_count:
            raise PublicTransferError(
                "case count must equal document_count * cases_per_document"
            )

        for case in self.cases:
            if case.source_document_id not in document_ids:
                raise PublicTransferError(
                    f"case {case.case_id} references an unknown source document"
                )
            if document_hashes[case.source_document_id] != case.source_document_text_sha256:
                raise PublicTransferError(
                    f"case {case.case_id} does not carry its source document hash"
                )


def sha256_text(value: str) -> str:
    """Return a stable lower-case SHA-256 digest for UTF-8 text."""

    return sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return a stable lower-case SHA-256 digest for raw source bytes."""

    return sha256(value).hexdigest()


def _canonical_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise PublicTransferError("unable to derive a stable source-document identifier")
    return slug


def _extract_evidence_sentence(paragraph: str, answer_start: int, answer_text: str) -> str:
    """Return the sentence containing the answer span without inventing evidence."""

    answer_end = answer_start + len(answer_text)
    if paragraph[answer_start:answer_end] != answer_text:
        raise PublicTransferError("answer span does not match its paragraph text")

    left_matches = list(re.finditer(r"[.!?](?:[\"'”’\)]*)\s+|\n+", paragraph[:answer_start]))
    sentence_start = left_matches[-1].end() if left_matches else 0

    right_match = re.search(r"[.!?](?:[\"'”’\)]*)(?=\s|$)|\n+", paragraph[answer_end:])
    sentence_end = answer_end + right_match.end() if right_match else len(paragraph)
    evidence = paragraph[sentence_start:sentence_end].strip()

    if not evidence or answer_text not in evidence:
        raise PublicTransferError("could not retain answer-bearing evidence sentence")
    return evidence


def _first_valid_answer(qa: dict[str, Any], paragraph: str) -> tuple[str, int] | None:
    answers = qa.get("answers")
    if not isinstance(answers, list):
        return None

    for answer in answers:
        if not isinstance(answer, dict):
            continue
        answer_text = answer.get("text")
        answer_start = answer.get("answer_start")
        if not isinstance(answer_text, str) or not answer_text:
            continue
        if not isinstance(answer_start, int) or answer_start < 0:
            continue
        answer_end = answer_start + len(answer_text)
        if paragraph[answer_start:answer_end] == answer_text:
            return answer_text, answer_start
    return None


def _select_article(
    article: dict[str, Any],
    *,
    article_index: int,
    paragraphs_per_document: int,
    cases_per_document: int,
) -> tuple[PublicTransferDocument, tuple[PublicTransferCase, ...]] | None:
    title = article.get("title")
    paragraphs = article.get("paragraphs")
    if not isinstance(title, str) or not title or not isinstance(paragraphs, list):
        return None

    selected_paragraphs: list[tuple[int, str, list[tuple[dict[str, Any], str, int]]]] = []
    for paragraph_index, paragraph_record in enumerate(paragraphs):
        if not isinstance(paragraph_record, dict):
            continue
        paragraph_text = paragraph_record.get("context")
        qas = paragraph_record.get("qas")
        if not isinstance(paragraph_text, str) or not paragraph_text:
            continue
        if not isinstance(qas, list):
            continue

        valid_qas: list[tuple[dict[str, Any], str, int]] = []
        for qa in qas:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question")
            question_id = qa.get("id")
            if not isinstance(question, str) or not question:
                continue
            if not isinstance(question_id, str) or not question_id:
                continue
            valid_answer = _first_valid_answer(qa, paragraph_text)
            if valid_answer is None:
                continue
            answer_text, answer_start = valid_answer
            valid_qas.append((qa, answer_text, answer_start))

        if not valid_qas:
            continue
        selected_paragraphs.append((paragraph_index, paragraph_text, valid_qas))
        if len(selected_paragraphs) == paragraphs_per_document:
            break

    if len(selected_paragraphs) != paragraphs_per_document:
        return None

    selected_cases: list[tuple[int, dict[str, Any], str, int]] = []
    for paragraph_index, _, valid_qas in selected_paragraphs:
        for qa, answer_text, answer_start in valid_qas:
            selected_cases.append((paragraph_index, qa, answer_text, answer_start))
            if len(selected_cases) == cases_per_document:
                break
        if len(selected_cases) == cases_per_document:
            break

    if len(selected_cases) != cases_per_document:
        return None

    source_document_id = (
        f"squad_v1_dev__a{article_index:03d}__{_canonical_slug(title)}"
    )
    document_parts: list[str] = []
    paragraph_offsets: dict[int, int] = {}
    cursor = 0
    for offset, (paragraph_index, paragraph_text, _) in enumerate(selected_paragraphs):
        if offset:
            document_parts.append("\n\n")
            cursor += 2
        paragraph_offsets[paragraph_index] = cursor
        document_parts.append(paragraph_text)
        cursor += len(paragraph_text)

    document_text = "".join(document_parts)
    document_hash = sha256_text(document_text)
    document = PublicTransferDocument(
        source_document_id=source_document_id,
        title=title,
        source_article_index=article_index,
        paragraph_indices=tuple(item[0] for item in selected_paragraphs),
        text=document_text,
        text_sha256=document_hash,
    )

    cases: list[PublicTransferCase] = []
    paragraph_lookup = {item[0]: item[1] for item in selected_paragraphs}
    for paragraph_index, qa, answer_text, answer_start in selected_cases:
        paragraph_text = paragraph_lookup[paragraph_index]
        question_id = qa["id"]
        cases.append(
            PublicTransferCase(
                case_id=f"squad_v1_dev__{question_id}",
                external_case_id=question_id,
                external_dataset_id=SQUAD_V1_DATASET_ID,
                source_document_id=source_document_id,
                source_document_text_sha256=document_hash,
                title=title,
                paragraph_index=paragraph_index,
                source_answer_start=paragraph_offsets[paragraph_index] + answer_start,
                question=qa["question"],
                answer_text=answer_text,
                gold_evidence_text=_extract_evidence_sentence(
                    paragraph_text,
                    answer_start,
                    answer_text,
                ),
            )
        )

    return document, tuple(cases)


def build_squad_v1_transfer_fixture(
    payload: dict[str, Any],
    *,
    source_url: str = SQUAD_V1_SOURCE_URL,
    source_sha256: str,
    document_count: int = DEFAULT_DOCUMENT_COUNT,
    paragraphs_per_document: int = DEFAULT_PARAGRAPHS_PER_DOCUMENT,
    cases_per_document: int = DEFAULT_CASES_PER_DOCUMENT,
) -> PublicTransferFixture:
    """Build the fixed real-text transfer probe from SQuAD v1.1 source JSON.

    Selection is source-order deterministic: retain the first source articles
    with enough answer-bearing paragraphs, keep the first valid paragraphs, and
    then retain the first answerable questions from those paragraphs.
    """

    if document_count <= 0 or paragraphs_per_document <= 0 or cases_per_document <= 0:
        raise PublicTransferError("selection sizes must all be greater than zero")
    if payload.get("version") != SQUAD_V1_DATASET_VERSION:
        raise PublicTransferError("expected a SQuAD v1.1 payload")

    articles = payload.get("data")
    if not isinstance(articles, list):
        raise PublicTransferError("SQuAD payload must contain a data list")

    documents: list[PublicTransferDocument] = []
    cases: list[PublicTransferCase] = []
    for article_index, article in enumerate(articles):
        if not isinstance(article, dict):
            continue
        selected = _select_article(
            article,
            article_index=article_index,
            paragraphs_per_document=paragraphs_per_document,
            cases_per_document=cases_per_document,
        )
        if selected is None:
            continue
        document, selected_cases = selected
        documents.append(document)
        cases.extend(selected_cases)
        if len(documents) == document_count:
            break

    if len(documents) != document_count:
        raise PublicTransferError(
            "source payload did not contain enough articles for the requested transfer fixture"
        )

    fixture = PublicTransferFixture(
        manifest=PublicTransferManifest(
            source_url=source_url,
            source_sha256=source_sha256,
            article_selection_rule=(
                "Traverse SQuAD v1.1 dev source order; retain the first "
                f"{document_count} articles with at least {paragraphs_per_document} "
                "answer-bearing paragraphs; keep those first qualifying paragraphs and "
                f"the first {cases_per_document} answerable questions per retained article."
            ),
            document_count=document_count,
            paragraphs_per_document=paragraphs_per_document,
            cases_per_document=cases_per_document,
            source_document_ids=tuple(document.source_document_id for document in documents),
            case_ids=tuple(case.case_id for case in cases),
        ),
        documents=tuple(documents),
        cases=tuple(cases),
    )
    fixture.assert_consistent()
    return fixture


def build_squad_v1_transfer_fixture_from_bytes(
    raw_payload: bytes,
    *,
    source_url: str = SQUAD_V1_SOURCE_URL,
    document_count: int = DEFAULT_DOCUMENT_COUNT,
    paragraphs_per_document: int = DEFAULT_PARAGRAPHS_PER_DOCUMENT,
    cases_per_document: int = DEFAULT_CASES_PER_DOCUMENT,
) -> PublicTransferFixture:
    """Decode a downloaded SQuAD payload and preserve the exact raw-byte digest."""

    try:
        parsed = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicTransferError("downloaded SQuAD payload is not valid UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise PublicTransferError("downloaded SQuAD payload must decode to a JSON object")

    return build_squad_v1_transfer_fixture(
        parsed,
        source_url=source_url,
        source_sha256=sha256_bytes(raw_payload),
        document_count=document_count,
        paragraphs_per_document=paragraphs_per_document,
        cases_per_document=cases_per_document,
    )


def _write_text(path: Path, value: str) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(value, encoding="utf-8")
    temporary_path.replace(path)


def _json_line_records(items: Iterable[BaseModel]) -> str:
    return "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for item in items
    )


def attribution_markdown(manifest: PublicTransferManifest) -> str:
    """Render the required attribution file for the curated public fixture."""

    return (
        "# Public Transfer Fixture Attribution\n\n"
        f"- **Dataset:** {manifest.external_dataset_id} (version {manifest.dataset_version})\n"
        f"- **Source:** {manifest.source_url}\n"
        f"- **Source SHA-256:** `{manifest.source_sha256}`\n"
        f"- **License:** {manifest.license_name}\n"
        f"- **Attribution:** {manifest.attribution}\n\n"
        "This directory is a deterministic, limited evaluation subset. It is separate from "
        "the repository's synthetic baseline and must not be represented as customer data or "
        "as a production benchmark.\n"
    )


def write_public_transfer_fixture(
    fixture: PublicTransferFixture,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write manifest, corpus, cases, and attribution with safe replacement semantics."""

    fixture.assert_consistent()
    output_directory = output_directory.resolve()
    expected_paths = (
        output_directory / "manifest.json",
        output_directory / "corpus.jsonl",
        output_directory / "cases.jsonl",
        output_directory / "ATTRIBUTION.md",
    )
    if output_directory.exists() and any(path.exists() for path in expected_paths) and not overwrite:
        raise PublicTransferError(
            f"refusing to replace an existing public transfer fixture: {output_directory}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_text(
        output_directory / "manifest.json",
        json.dumps(fixture.manifest.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
    )
    _write_text(output_directory / "corpus.jsonl", _json_line_records(fixture.documents))
    _write_text(output_directory / "cases.jsonl", _json_line_records(fixture.cases))
    _write_text(output_directory / "ATTRIBUTION.md", attribution_markdown(fixture.manifest))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PublicTransferError(f"missing public transfer artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise PublicTransferError(f"invalid JSON artifact: {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise PublicTransferError(f"missing public transfer artifact: {path}") from error

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise PublicTransferError(f"blank JSONL record at {path}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PublicTransferError(f"invalid JSONL record at {path}:{line_number}") from error
        if not isinstance(record, dict):
            raise PublicTransferError(f"non-object JSONL record at {path}:{line_number}")
        records.append(record)
    return records


def load_public_transfer_fixture(output_directory: Path) -> PublicTransferFixture:
    """Load and validate a previously materialized public transfer fixture."""

    output_directory = output_directory.resolve()
    fixture = PublicTransferFixture(
        manifest=PublicTransferManifest.model_validate(
            _read_json(output_directory / "manifest.json")
        ),
        documents=tuple(
            PublicTransferDocument.model_validate(record)
            for record in _read_jsonl(output_directory / "corpus.jsonl")
        ),
        cases=tuple(
            PublicTransferCase.model_validate(record)
            for record in _read_jsonl(output_directory / "cases.jsonl")
        ),
    )
    fixture.assert_consistent()
    attribution_path = output_directory / "ATTRIBUTION.md"
    if not attribution_path.is_file():
        raise PublicTransferError(f"missing public transfer attribution: {attribution_path}")
    return fixture
