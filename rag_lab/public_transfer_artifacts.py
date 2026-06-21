"""Bounded, separately versioned artifacts for public-corpus transfer runs.

Public-transfer artifacts preserve provenance, case references, metrics, and bounded
trace IDs. They do not contain raw public documents, chunk text, rendered prompts,
or generated answers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.comparison import PipelineComparisonReport, PipelineId
from rag_lab.public_transfer_runtime import (
    PublicTransferCaseReference,
    PublicTransferComparisonRun,
    PublicTransferRunProvenance,
)


PUBLIC_TRANSFER_ARTIFACT_SCHEMA_VERSION: Final[str] = (
    "public_transfer_comparison_artifact_v1"
)


class PublicTransferArtifactError(ValueError):
    """Raised when a public-transfer artifact is malformed or ambiguous."""


class PublicTransferComparisonArtifact(BaseModel):
    """One measured public-corpus transfer run, explicitly separate from the baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    artifact_schema_version: str = Field(
        default=PUBLIC_TRANSFER_ARTIFACT_SCHEMA_VERSION,
        min_length=8,
        max_length=120,
    )
    artifact_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=8, max_length=120)
    reference_run_id: str = Field(
        pattern=r"^[a-z0-9_:-]+$", min_length=5, max_length=160
    )
    provenance: PublicTransferRunProvenance
    case_references: tuple[PublicTransferCaseReference, ...] = Field(min_length=1)
    report: PipelineComparisonReport

    @model_validator(mode="after")
    def validate_consistency(self) -> "PublicTransferComparisonArtifact":
        if self.artifact_schema_version != PUBLIC_TRANSFER_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "artifact_schema_version must equal "
                f"{PUBLIC_TRANSFER_ARTIFACT_SCHEMA_VERSION}"
            )
        if self.reference_run_id != self.report.run_id:
            raise ValueError("reference_run_id must match report.run_id")
        if self.provenance.execution_config.retrieval_metric_k != self.report.retrieval_metric_k:
            raise ValueError(
                "provenance execution_config retrieval_metric_k must match report"
            )
        if len(self.case_references) != self.provenance.evaluation_case_count:
            raise ValueError(
                "case_references count must match provenance evaluation_case_count"
            )

        reference_case_ids = [reference.case_id for reference in self.case_references]
        if len(reference_case_ids) != len(set(reference_case_ids)):
            raise ValueError("case_references must not repeat case_id")
        report_case_ids = {outcome.case_id for outcome in self.report.case_outcomes}
        if report_case_ids != set(reference_case_ids):
            raise ValueError(
                "report case IDs must exactly match public-transfer case references"
            )
        expected_outcome_count = len(PipelineId) * len(self.case_references)
        if len(self.report.case_outcomes) != expected_outcome_count:
            raise ValueError(
                "report must cover every public-transfer case through all four pipelines"
            )
        return self


def build_public_transfer_artifact(
    *,
    artifact_id: str,
    run: PublicTransferComparisonRun,
) -> PublicTransferComparisonArtifact:
    """Wrap a completed run in a separately named public-transfer artifact."""

    return PublicTransferComparisonArtifact(
        artifact_id=artifact_id,
        reference_run_id=run.execution.report.run_id,
        provenance=run.provenance,
        case_references=run.case_references,
        report=run.execution.report,
    )


def _pretty_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_public_transfer_artifact(
    *,
    artifact: PublicTransferComparisonArtifact,
    path: Path,
) -> None:
    """Write a deterministic public-transfer artifact with parent creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _pretty_json(artifact.model_dump(mode="json")),
        encoding="utf-8",
    )


def load_public_transfer_artifact(path: Path) -> PublicTransferComparisonArtifact:
    """Load one public-transfer artifact and validate all coverage invariants."""

    if not path.is_file():
        raise PublicTransferArtifactError(
            f"public transfer artifact does not exist: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PublicTransferArtifactError(
            f"public transfer artifact is invalid JSON: {path}"
        ) from error
    try:
        return PublicTransferComparisonArtifact.model_validate(payload)
    except ValueError as error:
        raise PublicTransferArtifactError(
            f"public transfer artifact validation failed: {path}: {error}"
        ) from error


def render_public_transfer_markdown(
    *,
    artifact: PublicTransferComparisonArtifact,
) -> str:
    """Render a deterministic, bounded public-transfer readout."""

    report = artifact.report
    lines = [
        "# Public-Corpus RAG Transfer Run",
        "",
        "## Status",
        "",
        (
            "Measured external-validity probe on a fixed public SQuAD v1.1 subset. "
            "This is separate from the 18-case synthetic benchmark and is not a "
            "customer-data evaluation, production benchmark, final-answer evaluation, "
            "or production-readiness claim."
        ),
        "",
        "## Public fixture",
        "",
        f"- **Dataset:** `{artifact.provenance.external_dataset_id}` "
        f"(version `{artifact.provenance.dataset_version}`)",
        f"- **Source SHA-256:** `{artifact.provenance.source_sha256}`",
        f"- **Fixture manifest SHA-256:** `{artifact.provenance.fixture_manifest_sha256}`",
        f"- **License:** `{artifact.provenance.license_name}`",
        f"- **Public source documents:** {artifact.provenance.source_document_count}",
        f"- **Public evaluation cases:** {artifact.provenance.evaluation_case_count}",
        f"- **Pipeline outcomes:** {len(report.case_outcomes)}",
        "",
        "## Runtime",
        "",
        f"- **Run ID:** `{artifact.reference_run_id}`",
        f"- **Tokenizer:** `{artifact.provenance.tokenizer_name}`",
        f"- **Embedding model:** `{artifact.provenance.embedding_model_name}`",
        f"- **Reranker model:** `{artifact.provenance.reranker_model_name}`",
        f"- **Device:** `{artifact.provenance.device}`",
        f"- **Recall cutoff:** Recall@{report.retrieval_metric_k}",
        "",
        "## Measured evidence-survival results",
        "",
        "| Pipeline | Recall@k | MRR@10 | Evidence inclusion | Context drops |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in report.pipeline_metrics:
        lines.append(
            "| "
            f"`{metric.pipeline_id.value}` | "
            f"{metric.retrieval_recall_at_k.value:.1%} | "
            f"{metric.mrr_at_10:.3f} | "
            f"{metric.evidence_inclusion_rate.value:.1%} | "
            f"{metric.dropped_evidence_rate.value:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The outcomes measure whether exact known public evidence survives "
                "chunking, retrieval, ranking, and final context selection. They do "
                "not measure whether a language model produced a correct answer."
            ),
            "",
            (
                "No regression gate is applied to this first public run. It is a "
                "separate measurement artifact, not a replacement for the reviewed "
                "synthetic baseline."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_public_transfer_readout(
    *,
    artifact: PublicTransferComparisonArtifact,
    path: Path,
) -> None:
    """Write the deterministic public-transfer Markdown readout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_public_transfer_markdown(artifact=artifact),
        encoding="utf-8",
    )
