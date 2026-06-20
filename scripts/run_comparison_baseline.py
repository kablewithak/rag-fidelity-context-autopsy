"""Generate bounded comparison artifacts and verify a reviewed local baseline.

Default operation writes fresh, git-ignored outputs under ``outputs/comparisons`` and
checks them against the committed baseline artifact. A baseline update requires both
``--update-baseline`` and ``--confirm-baseline-update`` so an intentional benchmark,
model, or configuration change cannot silently rewrite the reviewed reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rag_lab.comparison_artifacts import (
    BASELINE_ARTIFACT_ID,
    DEFAULT_BASELINE_ARTIFACT_PATH,
    DEFAULT_BASELINE_READOUT_PATH,
    ComparisonRegressionPolicy,
    build_baseline_artifact,
    load_baseline_artifact,
    verify_against_baseline,
    write_baseline_artifact,
    write_executive_markdown,
)
from rag_lab.comparison_runtime import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    build_runtime_settings,
    run_local_four_pipeline_comparison,
)
from rag_lab.context_assembly import ContextRenderProfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = Path("outputs/comparisons/four_pipeline_current.json")
DEFAULT_OUTPUT_MARKDOWN = Path("outputs/comparisons/four_pipeline_current.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed four-pipeline benchmark, write bounded local artifacts, and "
            "verify the result against the committed baseline."
        )
    )
    parser.add_argument("--run-id", default="local_four_pipeline_regression_check_v1")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokenizer", choices=("tiktoken", "diagnostic"), default="tiktoken")
    parser.add_argument("--tiktoken-encoding", default="cl100k_base")
    parser.add_argument("--character-max-characters", type=int, default=700)
    parser.add_argument("--sentence-aware-max-tokens", type=int, default=96)
    parser.add_argument("--hybrid-rrf-k", type=int, default=60)
    parser.add_argument("--retrieval-metric-k", type=int, default=5)
    parser.add_argument(
        "--budgeted-render-profile",
        choices=tuple(profile.value for profile in ContextRenderProfile),
        default=ContextRenderProfile.COMPACT_CITATION.value,
    )
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        default=DEFAULT_BASELINE_ARTIFACT_PATH,
        help="Committed reviewed baseline artifact path, relative to the repository root.",
    )
    parser.add_argument(
        "--baseline-readout",
        type=Path,
        default=DEFAULT_BASELINE_READOUT_PATH,
        help="Committed executive markdown path, relative to the repository root.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Fresh git-ignored artifact path, relative to the repository root.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MARKDOWN,
        help="Fresh git-ignored markdown path, relative to the repository root.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Generate fresh local outputs without comparing them to the committed baseline.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the fresh result as the committed baseline. Requires explicit confirmation.",
    )
    parser.add_argument(
        "--confirm-baseline-update",
        action="store_true",
        help="Required alongside --update-baseline to prevent silent reference rewrites.",
    )
    return parser.parse_args()


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    if args.update_baseline and not args.confirm_baseline_update:
        raise SystemExit(
            "Refusing baseline update. Use --update-baseline together with "
            "--confirm-baseline-update after reviewing the result."
        )

    settings = build_runtime_settings(
        run_id=args.run_id,
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
        device=args.device,
        tokenizer_kind=args.tokenizer,
        tiktoken_encoding=args.tiktoken_encoding,
        character_max_characters=args.character_max_characters,
        sentence_aware_max_tokens=args.sentence_aware_max_tokens,
        hybrid_rrf_k=args.hybrid_rrf_k,
        retrieval_metric_k=args.retrieval_metric_k,
        budgeted_render_profile=ContextRenderProfile(args.budgeted_render_profile),
    )
    current = run_local_four_pipeline_comparison(
        project_root=PROJECT_ROOT,
        settings=settings,
    )
    current_artifact = build_baseline_artifact(
        artifact_id="four_pipeline_current_run",
        report=current.execution.report,
        provenance=current.provenance,
    )
    output_json = _repo_path(args.output_json)
    output_markdown = _repo_path(args.output_markdown)
    write_baseline_artifact(artifact=current_artifact, path=output_json)
    write_executive_markdown(artifact=current_artifact, path=output_markdown)

    if args.update_baseline:
        reviewed_artifact = build_baseline_artifact(
            artifact_id=BASELINE_ARTIFACT_ID,
            report=current.execution.report,
            provenance=current.provenance,
            regression_policy=ComparisonRegressionPolicy(),
        )
        baseline_path = _repo_path(args.baseline_artifact)
        readout_path = _repo_path(args.baseline_readout)
        write_baseline_artifact(artifact=reviewed_artifact, path=baseline_path)
        write_executive_markdown(artifact=reviewed_artifact, path=readout_path)
        print(f"BASELINE UPDATED: {baseline_path.relative_to(PROJECT_ROOT)}")
        print(f"READOUT UPDATED: {readout_path.relative_to(PROJECT_ROOT)}")
        return

    if args.no_verify:
        print(f"FRESH ARTIFACT WRITTEN: {output_json.relative_to(PROJECT_ROOT)}")
        print(f"FRESH READOUT WRITTEN: {output_markdown.relative_to(PROJECT_ROOT)}")
        return

    baseline = load_baseline_artifact(_repo_path(args.baseline_artifact))
    gate = verify_against_baseline(
        baseline=baseline,
        candidate_report=current.execution.report,
        candidate_provenance=current.provenance,
    )
    print(f"FRESH ARTIFACT WRITTEN: {output_json.relative_to(PROJECT_ROOT)}")
    print(f"FRESH READOUT WRITTEN: {output_markdown.relative_to(PROJECT_ROOT)}")
    print(
        "BASELINE REGRESSION GATE: PASS "
        f"({gate.checked_pipeline_count} pipelines, {gate.checked_case_count} cases)"
    )


if __name__ == "__main__":
    main()
