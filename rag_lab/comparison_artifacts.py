"""Versioned comparison artifacts, executive readouts, and regression gates.

This module is deliberately separate from ``comparison.py``:

- ``comparison.py`` reduces runtime outcomes into a typed report.
- This module freezes one reviewed report with provenance, renders a concise
  readout, and checks future executions against that reviewed baseline.

Artifacts store only report metadata, IDs, hashes, ranks, counts, and metrics.
They must not contain raw source documents, chunks, prompts, or model answers.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.comparison import (
    DEFAULT_PIPELINE_DEFINITIONS,
    CasePipelineOutcome,
    PipelineComparisonReport,
    PipelineDefinition,
    PipelineId,
    build_comparison_report,
)
from rag_lab.comparison_runner import ComparisonExecutionConfig
from rag_lab.schemas import CorpusDocument, EvaluationCase


ARTIFACT_SCHEMA_VERSION: Final[str] = "comparison_baseline_artifact_v1"
BASELINE_ARTIFACT_ID: Final[str] = "four_pipeline_baseline_v1"
DEFAULT_BASELINE_ARTIFACT_PATH: Final[Path] = Path(
    "artifacts/comparisons/four_pipeline_baseline_v1.json"
)
DEFAULT_BASELINE_READOUT_PATH: Final[Path] = Path(
    "docs/reports/four_pipeline_baseline_v1.md"
)


class ComparisonArtifactError(ValueError):
    """Raised when a comparison artifact is malformed, unsafe, or inconsistent."""


class ComparisonRegressionError(ValueError):
    """Raised when a fresh comparison run violates the reviewed baseline gate."""


class ComparisonRunProvenance(BaseModel):
    """Stable run metadata required to interpret a versioned comparison artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    tokenizer_name: str = Field(min_length=3, max_length=160)
    embedding_model_name: str = Field(min_length=3, max_length=240)
    reranker_model_name: str = Field(min_length=3, max_length=240)
    device: str = Field(min_length=2, max_length=80)
    execution_config: ComparisonExecutionConfig
    corpus_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_cases_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_document_count: int = Field(ge=1, le=10_000)
    evaluation_case_count: int = Field(ge=1, le=10_000)


class ComparisonRegressionPolicy(BaseModel):
    """Fixed synthetic-benchmark guardrails for one reviewed baseline.

    The policy permits improvements but rejects regressions in the metrics that the
    artifact publicly reports. Zero tolerance is intentional: this is a fixed corpus,
    fixed case set, fixed model-name, fixed tokenizer benchmark. An intentional model,
    corpus, or benchmark change requires a reviewed new baseline artifact rather than
    silently widening the gate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_recall_allowed_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    mrr_at_10_allowed_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_inclusion_allowed_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    dropped_evidence_allowed_increase: float = Field(default=0.0, ge=0.0, le=1.0)
    require_baseline_included_evidence_to_remain_included: bool = True
    require_baseline_top_k_retrieval_to_remain_top_k: bool = True


class ComparisonBaselineArtifact(BaseModel):
    """A reviewed local benchmark report plus the provenance and gate that constrain it."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    artifact_schema_version: str = Field(default=ARTIFACT_SCHEMA_VERSION)
    artifact_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=8, max_length=120)
    reference_run_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=5, max_length=160)
    provenance: ComparisonRunProvenance
    regression_policy: ComparisonRegressionPolicy
    report: PipelineComparisonReport

    @model_validator(mode="after")
    def validate_artifact_consistency(self) -> "ComparisonBaselineArtifact":
        if self.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact_schema_version must equal {ARTIFACT_SCHEMA_VERSION}"
            )
        if self.reference_run_id != self.report.run_id:
            raise ValueError("reference_run_id must match report.run_id")
        if self.provenance.execution_config.retrieval_metric_k != self.report.retrieval_metric_k:
            raise ValueError(
                "provenance execution_config retrieval_metric_k must match report retrieval_metric_k"
            )
        if self.provenance.evaluation_case_count * len(PipelineId) != len(self.report.case_outcomes):
            raise ValueError(
                "evaluation_case_count must match complete fixed-pipeline report coverage"
            )
        return self


@dataclass(frozen=True, slots=True)
class ComparisonRegressionGateResult:
    """Structured regression-gate result without serializing raw runtime traces."""

    passed: bool
    checked_pipeline_count: int
    checked_case_count: int
    messages: tuple[str, ...]


def canonical_json_sha256(value: object) -> str:
    """Return a stable SHA-256 digest for an already JSON-safe value."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def build_corpus_manifest_sha256(documents: Iterable[CorpusDocument]) -> str:
    """Hash fixed corpus identity without retaining raw corpus text in artifacts."""

    manifest = [
        {
            "source_doc_id": document.source_doc_id,
            "document_type": document.document_type.value,
            "char_count": document.char_count,
            "text_sha256": document.text_sha256,
        }
        for document in sorted(documents, key=lambda item: item.source_doc_id)
    ]
    return canonical_json_sha256(manifest)


def build_evaluation_cases_sha256(cases: Iterable[EvaluationCase]) -> str:
    """Hash the fixed case contracts without serializing their source text into provenance."""

    manifest = [
        {
            "case_id": case.case_id,
            "document_type": case.document_type.value,
            "query_type": case.query_type.value,
            "source_doc_id": case.source_doc_id,
            "gold_evidence_sha256": sha256(
                case.gold_evidence_text.encode("utf-8")
            ).hexdigest(),
            "gold_answer_sha256": sha256(case.gold_answer.encode("utf-8")).hexdigest(),
            "expected_failure_mode": case.expected_failure_mode.value,
        }
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    return canonical_json_sha256(manifest)


def build_baseline_artifact(
    *,
    artifact_id: str,
    report: PipelineComparisonReport,
    provenance: ComparisonRunProvenance,
    regression_policy: ComparisonRegressionPolicy | None = None,
) -> ComparisonBaselineArtifact:
    """Wrap one validated report in a reviewed-artifact contract."""

    return ComparisonBaselineArtifact(
        artifact_id=artifact_id,
        reference_run_id=report.run_id,
        provenance=provenance,
        regression_policy=regression_policy or ComparisonRegressionPolicy(),
        report=report,
    )


def write_baseline_artifact(*, artifact: ComparisonBaselineArtifact, path: Path) -> None:
    """Write a deterministically formatted artifact with parent-directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_pretty_json(artifact.model_dump(mode="json")), encoding="utf-8")


def load_baseline_artifact(path: Path) -> ComparisonBaselineArtifact:
    """Load and validate one committed baseline artifact."""

    if not path.exists():
        raise ComparisonArtifactError(f"baseline artifact does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ComparisonArtifactError(f"baseline artifact is invalid JSON: {path}") from error

    try:
        return ComparisonBaselineArtifact.model_validate(payload)
    except ValueError as error:
        raise ComparisonArtifactError(f"baseline artifact validation failed: {path}: {error}") from error


def render_executive_markdown(*, artifact: ComparisonBaselineArtifact) -> str:
    """Render a concise deterministic decision readout from the reviewed artifact."""

    report = artifact.report
    metrics = {metric.pipeline_id: metric for metric in report.pipeline_metrics}
    deltas = {delta.comparison_pipeline_id: delta for delta in report.metric_deltas}
    baseline = metrics[report.baseline_pipeline_id]
    token_dense = metrics[PipelineId.TOKEN_DENSE_NAIVE]
    hybrid = metrics[PipelineId.TOKEN_HYBRID_NAIVE]
    full = metrics[PipelineId.TOKEN_HYBRID_RERANK_BUDGETED]

    baseline_chunking_losses = baseline.loss_stage_counts.get("chunking", 0)
    baseline_retrieval_losses = baseline.loss_stage_counts.get("retrieval", 0)
    baseline_ranking_losses = baseline.loss_stage_counts.get("ranking", 0)

    token_dense_delta = deltas[PipelineId.TOKEN_DENSE_NAIVE]
    hybrid_delta = deltas[PipelineId.TOKEN_HYBRID_NAIVE]
    full_delta = deltas[PipelineId.TOKEN_HYBRID_RERANK_BUDGETED]

    lines = [
        "# Four-Pipeline Reliability Baseline v1",
        "",
        "## Status",
        "",
        "Locally validated synthetic-data benchmark. This is not a production deployment, customer-data evaluation, or final-answer grounding claim.",
        "",
        "## Reproducibility",
        "",
        f"- **Artifact ID:** `{artifact.artifact_id}`",
        f"- **Reference run:** `{artifact.reference_run_id}`",
        f"- **Tokenizer:** `{artifact.provenance.tokenizer_name}`",
        f"- **Embedding model:** `{artifact.provenance.embedding_model_name}`",
        f"- **Reranker model:** `{artifact.provenance.reranker_model_name}`",
        f"- **Device:** `{artifact.provenance.device}`",
        f"- **Fixed cases:** {artifact.provenance.evaluation_case_count}",
        f"- **Recall cutoff:** Recall@{report.retrieval_metric_k}",
        f"- **Candidate pool:** {report.pipeline_definitions[0].retrieval_top_k}",
        "",
        "## Measured results",
        "",
        "| Pipeline | Recall@5 | MRR@10 | Evidence inclusion | Context drops |",
        "|---|---:|---:|---:|---:|",
    ]
    for pipeline_id in PipelineId:
        metric = metrics[pipeline_id]
        lines.append(
            "| "
            f"`{pipeline_id.value}` | "
            f"{_percent(metric.retrieval_recall_at_k.value)} "
            f"({metric.retrieval_recall_at_k.numerator}/{metric.retrieval_recall_at_k.denominator}) | "
            f"{metric.mrr_at_10:.3f} | "
            f"{_percent(metric.evidence_inclusion_rate.value)} "
            f"({metric.evidence_inclusion_rate.numerator}/{metric.evidence_inclusion_rate.denominator}) | "
            f"{_percent(metric.dropped_evidence_rate.value)} "
            f"({metric.dropped_evidence_rate.numerator}/{metric.dropped_evidence_rate.denominator}) |"
        )

    lines.extend(
        [
            "",
            "## Evidence-backed interpretation",
            "",
            (
                f"- **Token-aware chunking:** `token_dense_naive` raised Recall@{report.retrieval_metric_k} "
                f"by {_percentage_points(token_dense_delta.retrieval_recall_at_k_delta)} versus the character+dense baseline "
                f"and raised evidence inclusion by {_percentage_points(token_dense_delta.evidence_inclusion_rate_delta)}."
            ),
            (
                f"- **Hybrid retrieval:** `token_hybrid_naive` reached "
                f"{_percent(hybrid.retrieval_recall_at_k.value)} Recall@{report.retrieval_metric_k} "
                f"and {_percent(hybrid.evidence_inclusion_rate.value)} evidence inclusion, "
                f"a {_percentage_points(hybrid_delta.retrieval_recall_at_k_delta)} Recall@{report.retrieval_metric_k} gain versus baseline."
            ),
            (
                f"- **Reranking:** `token_hybrid_rerank_budgeted` preserved the hybrid retrieval result "
                f"and increased MRR@10 by {full_delta.mrr_at_10_delta:+.3f} versus baseline "
                f"({full.mrr_at_10:.3f} versus {baseline.mrr_at_10:.3f})."
            ),
            (
                "- **Context budgeting:** the standard comparison suite recorded zero context-assembly "
                "drops. The separate controlled Phase 6 pressure diagnostic remains the mechanism "
                "proof that rendered wrapper overhead can displace otherwise retrieved evidence."
            ),
            "",
            "## Baseline failure pattern",
            "",
            (
                f"The `char_dense_naive` baseline recorded {baseline_chunking_losses} chunking-stage loss(es), "
                f"{baseline_retrieval_losses} retrieval-stage loss(es), and {baseline_ranking_losses} ranking-stage loss(es)."
            ),
            "",
            "## Regression gate",
            "",
            "A fresh run must preserve the fixed provenance, pipeline definitions, case set, Recall cutoff, and all baseline-included evidence. It may improve metrics, but it fails when Recall@5, MRR@10, or evidence inclusion falls below the reviewed baseline, or when dropped-evidence rate increases.",
            "",
            "```powershell",
            "python .\\scripts\\run_comparison_baseline.py `",
            "    --tokenizer tiktoken `",
            "    --tiktoken-encoding cl100k_base `",
            "    --verify",
            "```",
            "",
            "## Non-claims",
            "",
            "This benchmark does not evaluate final generated answers, citation correctness in generated output, customer data, production latency, production cost, model-version stability across vendors, or production readiness.",
            "",
        ]
    )
    return "\n".join(lines)


def write_executive_markdown(*, artifact: ComparisonBaselineArtifact, path: Path) -> None:
    """Write the deterministic executive readout for one baseline artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_executive_markdown(artifact=artifact), encoding="utf-8")


def verify_against_baseline(
    *,
    baseline: ComparisonBaselineArtifact,
    candidate_report: PipelineComparisonReport,
    candidate_provenance: ComparisonRunProvenance,
) -> ComparisonRegressionGateResult:
    """Reject a fresh run that regresses the reviewed synthetic benchmark.

    Improvements are allowed. Any intentional change to model names, tokenizer,
    corpus/eval manifests, pipeline definitions, or strict metric floor must become a
    newly reviewed baseline artifact rather than silently rewriting this one.
    """

    messages: list[str] = []
    _assert_provenance_matches(
        expected=baseline.provenance,
        actual=candidate_provenance,
        messages=messages,
    )
    _assert_report_shape_matches(
        expected=baseline.report,
        actual=candidate_report,
        messages=messages,
    )
    _assert_metric_floors(
        baseline=baseline,
        candidate=candidate_report,
        messages=messages,
    )
    _assert_case_level_guards(
        baseline=baseline,
        candidate=candidate_report,
        messages=messages,
    )

    if messages:
        summary = "\n".join(f"- {message}" for message in messages)
        raise ComparisonRegressionError(f"comparison baseline regression gate failed:\n{summary}")

    return ComparisonRegressionGateResult(
        passed=True,
        checked_pipeline_count=len(PipelineId),
        checked_case_count=baseline.provenance.evaluation_case_count,
        messages=("Baseline regression gate passed.",),
    )


def _assert_provenance_matches(
    *,
    expected: ComparisonRunProvenance,
    actual: ComparisonRunProvenance,
    messages: list[str],
) -> None:
    fields = (
        "tokenizer_name",
        "embedding_model_name",
        "reranker_model_name",
        "device",
        "corpus_manifest_sha256",
        "evaluation_cases_sha256",
        "source_document_count",
        "evaluation_case_count",
    )
    for field_name in fields:
        if getattr(expected, field_name) != getattr(actual, field_name):
            messages.append(
                f"provenance mismatch for {field_name}: expected {getattr(expected, field_name)!r}, "
                f"got {getattr(actual, field_name)!r}"
            )

    if expected.execution_config != actual.execution_config:
        messages.append("provenance mismatch for execution_config")


def _assert_report_shape_matches(
    *,
    expected: PipelineComparisonReport,
    actual: PipelineComparisonReport,
    messages: list[str],
) -> None:
    if expected.schema_version != actual.schema_version:
        messages.append(
            f"report schema mismatch: expected {expected.schema_version}, got {actual.schema_version}"
        )
    if expected.retrieval_metric_k != actual.retrieval_metric_k:
        messages.append(
            "retrieval_metric_k mismatch: "
            f"expected {expected.retrieval_metric_k}, got {actual.retrieval_metric_k}"
        )
    if expected.baseline_pipeline_id is not actual.baseline_pipeline_id:
        messages.append("baseline_pipeline_id mismatch")
    if expected.pipeline_definitions != actual.pipeline_definitions:
        messages.append("pipeline_definitions mismatch")

    expected_cases = _case_ids_by_pipeline(expected)
    actual_cases = _case_ids_by_pipeline(actual)
    if expected_cases != actual_cases:
        messages.append("case coverage mismatch")


def _assert_metric_floors(
    *,
    baseline: ComparisonBaselineArtifact,
    candidate: PipelineComparisonReport,
    messages: list[str],
) -> None:
    policy = baseline.regression_policy
    expected_metrics = {metric.pipeline_id: metric for metric in baseline.report.pipeline_metrics}
    actual_metrics = {metric.pipeline_id: metric for metric in candidate.pipeline_metrics}

    for pipeline_id in PipelineId:
        expected = expected_metrics[pipeline_id]
        actual = actual_metrics[pipeline_id]
        if actual.retrieval_recall_at_k.value < (
            expected.retrieval_recall_at_k.value - policy.retrieval_recall_allowed_drop
        ):
            messages.append(
                f"{pipeline_id.value} Recall@{candidate.retrieval_metric_k} regressed: "
                f"expected >= {expected.retrieval_recall_at_k.value:.6f}, "
                f"got {actual.retrieval_recall_at_k.value:.6f}"
            )
        if actual.mrr_at_10 < (expected.mrr_at_10 - policy.mrr_at_10_allowed_drop):
            messages.append(
                f"{pipeline_id.value} MRR@10 regressed: expected >= {expected.mrr_at_10:.6f}, "
                f"got {actual.mrr_at_10:.6f}"
            )
        if actual.evidence_inclusion_rate.value < (
            expected.evidence_inclusion_rate.value - policy.evidence_inclusion_allowed_drop
        ):
            messages.append(
                f"{pipeline_id.value} evidence inclusion regressed: "
                f"expected >= {expected.evidence_inclusion_rate.value:.6f}, "
                f"got {actual.evidence_inclusion_rate.value:.6f}"
            )
        if actual.dropped_evidence_rate.value > (
            expected.dropped_evidence_rate.value + policy.dropped_evidence_allowed_increase
        ):
            messages.append(
                f"{pipeline_id.value} dropped-evidence rate regressed: "
                f"expected <= {expected.dropped_evidence_rate.value:.6f}, "
                f"got {actual.dropped_evidence_rate.value:.6f}"
            )


def _assert_case_level_guards(
    *,
    baseline: ComparisonBaselineArtifact,
    candidate: PipelineComparisonReport,
    messages: list[str],
) -> None:
    policy = baseline.regression_policy
    expected_outcomes = _outcomes_by_key(baseline.report)
    actual_outcomes = _outcomes_by_key(candidate)

    for key, expected in expected_outcomes.items():
        actual = actual_outcomes.get(key)
        if actual is None:
            continue
        if policy.require_baseline_included_evidence_to_remain_included:
            if expected.gold_evidence_included and not actual.gold_evidence_included:
                messages.append(
                    f"{key[0].value}/{key[1]} lost evidence that baseline included"
                )
        if policy.require_baseline_top_k_retrieval_to_remain_top_k:
            expected_top_k = (
                expected.retrieved_gold_rank is not None
                and expected.retrieved_gold_rank <= baseline.report.retrieval_metric_k
            )
            actual_top_k = (
                actual.retrieved_gold_rank is not None
                and actual.retrieved_gold_rank <= candidate.retrieval_metric_k
            )
            if expected_top_k and not actual_top_k:
                messages.append(
                    f"{key[0].value}/{key[1]} fell outside Recall@{candidate.retrieval_metric_k}"
                )


def _case_ids_by_pipeline(report: PipelineComparisonReport) -> dict[PipelineId, tuple[str, ...]]:
    return {
        pipeline_id: tuple(
            sorted(
                outcome.case_id
                for outcome in report.case_outcomes
                if outcome.pipeline_id is pipeline_id
            )
        )
        for pipeline_id in PipelineId
    }


def _outcomes_by_key(
    report: PipelineComparisonReport,
) -> dict[tuple[PipelineId, str], CasePipelineOutcome]:
    return {
        (outcome.pipeline_id, outcome.case_id): outcome
        for outcome in report.case_outcomes
    }


def _pretty_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _percentage_points(value: float) -> str:
    return f"{value * 100:+.1f} percentage points"
