"""Reviewed, separate reporting for the public-corpus transfer probe."""
from __future__ import annotations

from pathlib import Path

from rag_lab.comparison import PipelineId, PipelineMetrics
from rag_lab.comparison_artifacts import ComparisonBaselineArtifact
from rag_lab.public_transfer_artifacts import PublicTransferComparisonArtifact

REVIEWED_PUBLIC_TRANSFER_ARTIFACT_ID = "public_transfer_squad_v1_dev_v1_reviewed_v1"
DEFAULT_REVIEWED_PUBLIC_TRANSFER_ARTIFACT_PATH = Path(
    "artifacts/public_transfer/public_transfer_squad_v1_dev_v1_reviewed_v1.json"
)
DEFAULT_REVIEWED_PUBLIC_TRANSFER_REPORT_PATH = Path(
    "docs/reports/public_transfer_squad_v1_dev_v1_reviewed_v1.md"
)
EXPECTED_PUBLIC_DATASET_ID = "squad_v1.1_dev"
EXPECTED_PUBLIC_DOCUMENT_COUNT = 10
EXPECTED_PUBLIC_CASE_COUNT = 30


class PublicTransferReviewError(ValueError):
    """Raised when a public-transfer review blurs evaluation boundaries."""


def build_reviewed_public_transfer_artifact(
    *, source_artifact: PublicTransferComparisonArtifact
) -> PublicTransferComparisonArtifact:
    """Assign the stable reviewed identity after an explicit human review decision."""
    return source_artifact.model_copy(
        update={"artifact_id": REVIEWED_PUBLIC_TRANSFER_ARTIFACT_ID}
    )


def assert_review_boundary(
    *,
    public_artifact: PublicTransferComparisonArtifact,
    synthetic_baseline: ComparisonBaselineArtifact,
) -> None:
    """Validate side-by-side reporting without blending public and synthetic suites."""
    public = public_artifact.provenance
    synthetic = synthetic_baseline.provenance
    if public_artifact.artifact_id != REVIEWED_PUBLIC_TRANSFER_ARTIFACT_ID:
        raise PublicTransferReviewError("public review must use the stable reviewed artifact identity")
    if public.external_dataset_id != EXPECTED_PUBLIC_DATASET_ID:
        raise PublicTransferReviewError("reviewed public artifact must use squad_v1.1_dev")
    if public.source_document_count != EXPECTED_PUBLIC_DOCUMENT_COUNT:
        raise PublicTransferReviewError("reviewed public artifact must preserve the fixed 10-document fixture")
    if public.evaluation_case_count != EXPECTED_PUBLIC_CASE_COUNT:
        raise PublicTransferReviewError("reviewed public artifact must preserve the fixed 30-case fixture")
    if synthetic.evaluation_case_count != 18:
        raise PublicTransferReviewError("synthetic baseline must preserve the fixed 18-case fixture")
    if public_artifact.report.pipeline_definitions != synthetic_baseline.report.pipeline_definitions:
        raise PublicTransferReviewError("public and synthetic reports must use the same four pipeline definitions")
    for name, public_value, synthetic_value in (
        ("tokenizer", public.tokenizer_name, synthetic.tokenizer_name),
        ("embedding model", public.embedding_model_name, synthetic.embedding_model_name),
        ("reranker model", public.reranker_model_name, synthetic.reranker_model_name),
        ("device", public.device, synthetic.device),
        ("execution configuration", public.execution_config, synthetic.execution_config),
    ):
        if public_value != synthetic_value:
            raise PublicTransferReviewError(
                f"public and synthetic reports must use the same {name} for side-by-side review"
            )


def _metrics_by_pipeline(metrics: list[PipelineMetrics]) -> dict[PipelineId, PipelineMetrics]:
    values = {metric.pipeline_id: metric for metric in metrics}
    if set(values) != set(PipelineId):
        raise PublicTransferReviewError("review report requires all four fixed pipeline metrics")
    return values


def _metric_row(metric: PipelineMetrics) -> str:
    return (
        f"| `{metric.pipeline_id.value}` | "
        f"{metric.retrieval_recall_at_k.value:.1%} ({metric.retrieval_recall_at_k.numerator}/{metric.retrieval_recall_at_k.denominator}) | "
        f"{metric.mrr_at_10:.3f} | "
        f"{metric.evidence_inclusion_rate.value:.1%} ({metric.evidence_inclusion_rate.numerator}/{metric.evidence_inclusion_rate.denominator}) | "
        f"{metric.dropped_evidence_rate.value:.1%} |"
    )


def _pp(value: float) -> str:
    return f"{value * 100:+.1f} percentage points"


def render_public_transfer_review_markdown(
    *,
    public_artifact: PublicTransferComparisonArtifact,
    synthetic_baseline: ComparisonBaselineArtifact,
) -> str:
    """Render a deterministic evidence review that never aggregates benchmark rates."""
    assert_review_boundary(public_artifact=public_artifact, synthetic_baseline=synthetic_baseline)
    public = _metrics_by_pipeline(public_artifact.report.pipeline_metrics)
    synthetic = _metrics_by_pipeline(synthetic_baseline.report.pipeline_metrics)
    p_char = public[PipelineId.CHAR_DENSE_NAIVE]
    p_token = public[PipelineId.TOKEN_DENSE_NAIVE]
    p_hybrid = public[PipelineId.TOKEN_HYBRID_NAIVE]
    p_full = public[PipelineId.TOKEN_HYBRID_RERANK_BUDGETED]

    lines = [
        "# Public-Corpus Transfer Review v1",
        "",
        "## Status",
        "",
        "Reviewed external-validity probe on a fixed public SQuAD v1.1 subset. This report presents the public run beside the controlled synthetic benchmark without averaging their scores or changing the synthetic regression gate.",
        "",
        "## Review boundary",
        "",
        f"- **Synthetic baseline:** `{synthetic_baseline.artifact_id}` ({synthetic_baseline.provenance.evaluation_case_count} fixed synthetic cases, {synthetic_baseline.provenance.source_document_count} source documents)",
        f"- **Public transfer artifact:** `{public_artifact.artifact_id}` ({public_artifact.provenance.evaluation_case_count} fixed public cases, {public_artifact.provenance.source_document_count} source documents)",
        f"- **Public dataset:** `{public_artifact.provenance.external_dataset_id}` (version `{public_artifact.provenance.dataset_version}`, license `{public_artifact.provenance.license_name}`)",
        f"- **Public source SHA-256:** `{public_artifact.provenance.source_sha256}`",
        f"- **Public fixture manifest SHA-256:** `{public_artifact.provenance.fixture_manifest_sha256}`",
        "",
        "The synthetic benchmark remains the controlled mechanism test. The public fixture is a separate transfer probe. Their rates are displayed side by side for interpretation only and must not be pooled into one headline score.",
        "",
        "## Controlled synthetic benchmark",
        "",
        "| Pipeline | Recall@5 | MRR@10 | Evidence inclusion | Context drops |",
        "|---|---:|---:|---:|---:|",
    ]
    for pipeline_id in PipelineId:
        lines.append(_metric_row(synthetic[pipeline_id]))
    lines.extend(["", "## Public-corpus transfer probe", "", "| Pipeline | Recall@5 | MRR@10 | Evidence inclusion | Context drops |", "|---|---:|---:|---:|---:|"])
    for pipeline_id in PipelineId:
        lines.append(_metric_row(public[pipeline_id]))
    lines.extend([
        "",
        "## Measured transfer findings",
        "",
        f"- **Public ranking signal:** within the 30-case public fixture, `token_hybrid_rerank_budgeted` changed MRR@10 by {_pp(p_full.mrr_at_10 - p_char.mrr_at_10)} relative to `char_dense_naive` ({p_full.mrr_at_10:.3f} versus {p_char.mrr_at_10:.3f}).",
        f"- **Public evidence-survival signal:** within that same fixture, the full pipeline changed evidence inclusion by {_pp(p_full.evidence_inclusion_rate.value - p_char.evidence_inclusion_rate.value)} relative to `char_dense_naive` ({p_full.evidence_inclusion_rate.value:.1%} versus {p_char.evidence_inclusion_rate.value:.1%}).",
        f"- **Public hybrid signal:** `token_hybrid_naive` changed MRR@10 by {_pp(p_hybrid.mrr_at_10 - p_char.mrr_at_10)} relative to `char_dense_naive` ({p_hybrid.mrr_at_10:.3f} versus {p_char.mrr_at_10:.3f}).",
        f"- **Non-uniform chunking result:** on this public fixture, `token_dense_naive` changed Recall@5 by {_pp(p_token.retrieval_recall_at_k.value - p_char.retrieval_recall_at_k.value)} relative to `char_dense_naive` ({p_token.retrieval_recall_at_k.value:.1%} versus {p_char.retrieval_recall_at_k.value:.1%}). This is not evidence for a universal chunking rule.",
        "",
        "## Interpretation",
        "",
        "The public run supports a limited transfer claim: the harness can measure evidence survival on non-authored public prose, and hybrid retrieval plus reranking improved ordering and final evidence inclusion on this fixed probe. It does not support a claim that every synthetic intervention transfers uniformly across corpora.",
        "",
        "## Reproducibility",
        "",
        f"- **Tokenizer:** `{public_artifact.provenance.tokenizer_name}`",
        f"- **Embedding model:** `{public_artifact.provenance.embedding_model_name}`",
        f"- **Reranker model:** `{public_artifact.provenance.reranker_model_name}`",
        f"- **Device:** `{public_artifact.provenance.device}`",
        f"- **Candidate pool:** {public_artifact.report.pipeline_definitions[0].retrieval_top_k}",
        f"- **Recall cutoff:** Recall@{public_artifact.report.retrieval_metric_k}",
        "",
        "```powershell",
        "python .\\scripts\\publish_public_transfer_review.py --check",
        "```",
        "",
        "## Non-claims",
        "",
        "This review does not evaluate final generated answers, citation correctness in generated output, customer data, production latency, production cost, cross-vendor stability, security posture, or production readiness.",
        "",
    ])
    return "\n".join(lines)


def write_public_transfer_review_markdown(*, public_artifact: PublicTransferComparisonArtifact, synthetic_baseline: ComparisonBaselineArtifact, path: Path) -> None:
    """Write the deterministic public-transfer review report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_public_transfer_review_markdown(public_artifact=public_artifact, synthetic_baseline=synthetic_baseline), encoding="utf-8")


def assert_public_transfer_review_matches(*, public_artifact: PublicTransferComparisonArtifact, synthetic_baseline: ComparisonBaselineArtifact, path: Path) -> None:
    """Fail closed when the committed report differs from its source artifacts."""
    if not path.is_file():
        raise PublicTransferReviewError(f"review report does not exist: {path}")
    expected = render_public_transfer_review_markdown(public_artifact=public_artifact, synthetic_baseline=synthetic_baseline)
    if path.read_text(encoding="utf-8") != expected:
        raise PublicTransferReviewError("committed public-transfer review report does not match its artifacts")
