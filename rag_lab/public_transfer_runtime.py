"""Runtime adapter for the separately versioned public-corpus transfer probe.

This module deliberately reuses the fixed four-pipeline execution engine without
loading the synthetic corpus or synthetic evaluation cases. It adapts the
public SQuAD-derived fixture through a narrow structural boundary and retains
public provenance separately from the synthetic baseline contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_lab.comparison_artifacts import canonical_json_sha256
from rag_lab.comparison_runner import (
    ComparisonExecutionConfig,
    ComparisonExecutionResult,
    FourPipelineComparisonRunner,
)
from rag_lab.comparison_runtime import (
    ComparisonRuntimeSettings,
    build_token_counter,
)
from rag_lab.embedders import SentenceTransformerEmbeddingModel
from rag_lab.public_transfer import (
    PublicTransferCase,
    PublicTransferDocument,
    PublicTransferFixture,
    PublicTransferManifest,
    load_public_transfer_fixture,
)
from rag_lab.rerankers import SentenceTransformersCrossEncoderModel


DEFAULT_PUBLIC_TRANSFER_FIXTURE_PATH = Path(
    "data/public_transfer/squad_v1_dev_v1"
)


class PublicTransferRuntimeError(ValueError):
    """Raised when the public transfer probe cannot preserve source provenance."""


class PublicTransferRuntimeDocument(BaseModel):
    """Minimal document boundary accepted by the existing chunking/retrieval runner.

    This is intentionally separate from ``CorpusDocument``. The synthetic document
    taxonomy must not be used to relabel public reference text as synthetic FAQ,
    policy, or code content merely to reuse the four-pipeline execution engine.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_doc_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=3, max_length=96)
    text: str = Field(min_length=1, max_length=100_000)
    char_count: int = Field(ge=1)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_integrity(self) -> "PublicTransferRuntimeDocument":
        if self.char_count != len(self.text):
            raise ValueError("char_count must equal source text length")
        if self.text_sha256 != sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("text_sha256 must match source text")
        return self


class PublicTransferRuntimeCase(BaseModel):
    """Minimal case boundary required by the existing four-pipeline runner.

    The public fixture's answer text and external identifiers remain in
    ``PublicTransferCaseReference``. The runner needs only query, exact evidence,
    and source-document identity to measure evidence survival.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    query: str = Field(min_length=1, max_length=2_000)
    gold_evidence_text: str = Field(min_length=1, max_length=10_000)
    source_doc_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=3, max_length=96)


class PublicTransferCaseReference(BaseModel):
    """Public-fixture provenance retained without serializing raw source text."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    external_case_id: str = Field(min_length=1, max_length=160)
    source_document_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=3, max_length=96)
    source_document_text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_answer_start: int = Field(ge=0)
    question_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    answer_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    gold_evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class PublicTransferRunProvenance(BaseModel):
    """Complete bounded provenance for one public-corpus comparison execution."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    fixture_format_version: str = Field(min_length=3, max_length=120)
    external_dataset_id: str = Field(min_length=3, max_length=160)
    dataset_version: str = Field(min_length=1, max_length=80)
    source_url: str = Field(min_length=8, max_length=1_000)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    license_name: str = Field(min_length=3, max_length=160)
    fixture_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    case_contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tokenizer_name: str = Field(min_length=3, max_length=160)
    embedding_model_name: str = Field(min_length=3, max_length=240)
    reranker_model_name: str = Field(min_length=3, max_length=240)
    device: str = Field(min_length=2, max_length=80)
    execution_config: ComparisonExecutionConfig
    source_document_count: int = Field(ge=1, le=10_000)
    evaluation_case_count: int = Field(ge=1, le=10_000)


@dataclass(frozen=True, slots=True)
class PublicTransferAdapterResult:
    """Normalized runner inputs plus public source references."""

    documents: tuple[PublicTransferRuntimeDocument, ...]
    cases: tuple[PublicTransferRuntimeCase, ...]
    case_references: tuple[PublicTransferCaseReference, ...]


@dataclass(frozen=True, slots=True)
class PublicTransferComparisonRun:
    """One public-corpus comparison result with bounded provenance."""

    execution: ComparisonExecutionResult
    provenance: PublicTransferRunProvenance
    case_references: tuple[PublicTransferCaseReference, ...]


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _runtime_document(document: PublicTransferDocument) -> PublicTransferRuntimeDocument:
    return PublicTransferRuntimeDocument(
        source_doc_id=document.source_document_id,
        text=document.text,
        char_count=len(document.text),
        text_sha256=document.text_sha256,
    )


def _runtime_case(case: PublicTransferCase) -> PublicTransferRuntimeCase:
    return PublicTransferRuntimeCase(
        case_id=case.case_id,
        query=case.question,
        gold_evidence_text=case.gold_evidence_text,
        source_doc_id=case.source_document_id,
    )


def _case_reference(case: PublicTransferCase) -> PublicTransferCaseReference:
    return PublicTransferCaseReference(
        case_id=case.case_id,
        external_case_id=case.external_case_id,
        source_document_id=case.source_document_id,
        source_document_text_sha256=case.source_document_text_sha256,
        source_answer_start=case.source_answer_start,
        question_sha256=_sha256_text(case.question),
        answer_sha256=_sha256_text(case.answer_text),
        gold_evidence_sha256=_sha256_text(case.gold_evidence_text),
    )


def adapt_public_transfer_fixture(
    fixture: PublicTransferFixture,
) -> PublicTransferAdapterResult:
    """Validate and adapt a public fixture without changing synthetic contracts."""

    fixture.assert_consistent()
    documents = tuple(_runtime_document(document) for document in fixture.documents)
    document_by_id = {document.source_doc_id: document for document in documents}
    cases = tuple(_runtime_case(case) for case in fixture.cases)
    references = tuple(_case_reference(case) for case in fixture.cases)

    for source_case, runtime_case in zip(fixture.cases, cases, strict=True):
        document = document_by_id.get(runtime_case.source_doc_id)
        if document is None:
            raise PublicTransferRuntimeError(
                f"missing adapted source document for {runtime_case.case_id}"
            )
        answer_end = source_case.source_answer_start + len(source_case.answer_text)
        if document.text[source_case.source_answer_start:answer_end] != source_case.answer_text:
            raise PublicTransferRuntimeError(
                f"{runtime_case.case_id} answer span does not match its adapted source document"
            )
        if runtime_case.gold_evidence_text not in document.text:
            raise PublicTransferRuntimeError(
                f"{runtime_case.case_id} gold evidence is not an exact source substring"
            )

    return PublicTransferAdapterResult(
        documents=documents,
        cases=cases,
        case_references=references,
    )


def _corpus_contract_sha256(
    documents: Sequence[PublicTransferRuntimeDocument],
) -> str:
    return canonical_json_sha256(
        [
            {
                "source_doc_id": document.source_doc_id,
                "char_count": document.char_count,
                "text_sha256": document.text_sha256,
            }
            for document in sorted(documents, key=lambda item: item.source_doc_id)
        ]
    )


def _case_contract_sha256(
    references: Sequence[PublicTransferCaseReference],
) -> str:
    return canonical_json_sha256(
        [
            reference.model_dump(mode="json")
            for reference in sorted(references, key=lambda item: item.case_id)
        ]
    )


def build_public_transfer_provenance(
    *,
    manifest: PublicTransferManifest,
    adapter: PublicTransferAdapterResult,
    token_counter_name: str,
    embedding_model_name: str,
    reranker_model_name: str,
    device: str,
    execution_config: ComparisonExecutionConfig,
) -> PublicTransferRunProvenance:
    """Build provenance without storing public source text in output artifacts."""

    return PublicTransferRunProvenance(
        fixture_format_version=manifest.format_version,
        external_dataset_id=manifest.external_dataset_id,
        dataset_version=manifest.dataset_version,
        source_url=manifest.source_url,
        source_sha256=manifest.source_sha256,
        license_name=manifest.license_name,
        fixture_manifest_sha256=canonical_json_sha256(
            manifest.model_dump(mode="json")
        ),
        corpus_contract_sha256=_corpus_contract_sha256(adapter.documents),
        case_contract_sha256=_case_contract_sha256(adapter.case_references),
        tokenizer_name=token_counter_name,
        embedding_model_name=embedding_model_name,
        reranker_model_name=reranker_model_name,
        device=device,
        execution_config=execution_config,
        source_document_count=len(adapter.documents),
        evaluation_case_count=len(adapter.cases),
    )


def run_local_public_transfer_comparison(
    *,
    project_root: Path,
    settings: ComparisonRuntimeSettings,
    fixture_directory: Path = DEFAULT_PUBLIC_TRANSFER_FIXTURE_PATH,
) -> PublicTransferComparisonRun:
    """Run the fixed four pipelines on the public fixture without baseline mutation."""

    resolved_fixture_directory = (
        fixture_directory
        if fixture_directory.is_absolute()
        else project_root / fixture_directory
    )
    fixture = load_public_transfer_fixture(resolved_fixture_directory)
    adapter = adapt_public_transfer_fixture(fixture)

    token_counter = build_token_counter(
        tokenizer_kind=settings.tokenizer_kind,
        encoding_name=settings.tiktoken_encoding,
    )
    embedding_model = SentenceTransformerEmbeddingModel(
        model_name=settings.embedding_model_name,
        device=settings.device,
    )
    reranker_model = SentenceTransformersCrossEncoderModel(
        model_name=settings.reranker_model_name,
        device=settings.device,
    )
    runner = FourPipelineComparisonRunner(
        token_counter=token_counter,
        embedding_model=embedding_model,
        reranker_scoring_model=reranker_model,
        config=settings.execution_config,
    )
    execution = runner.run(
        run_id=settings.run_id,
        cases=adapter.cases,  # structural runner boundary; no synthetic schema relabeling
        documents=adapter.documents,  # structural runner boundary; no synthetic taxonomy
    )
    provenance = build_public_transfer_provenance(
        manifest=fixture.manifest,
        adapter=adapter,
        token_counter_name=token_counter.name,
        embedding_model_name=embedding_model.name,
        reranker_model_name=reranker_model.name,
        device=settings.device,
        execution_config=settings.execution_config,
    )
    return PublicTransferComparisonRun(
        execution=execution,
        provenance=provenance,
        case_references=adapter.case_references,
    )
