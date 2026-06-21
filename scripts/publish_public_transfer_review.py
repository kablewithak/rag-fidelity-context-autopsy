"""Publish or verify the reviewed public-corpus transfer evidence asset."""
from __future__ import annotations

import argparse
from pathlib import Path

from rag_lab.comparison_artifacts import DEFAULT_BASELINE_ARTIFACT_PATH, load_baseline_artifact
from rag_lab.comparison_runtime import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL, build_runtime_settings
from rag_lab.context_assembly import ContextRenderProfile
from rag_lab.public_transfer_artifacts import build_public_transfer_artifact, load_public_transfer_artifact, write_public_transfer_artifact
from rag_lab.public_transfer_review import (
    DEFAULT_REVIEWED_PUBLIC_TRANSFER_ARTIFACT_PATH,
    DEFAULT_REVIEWED_PUBLIC_TRANSFER_REPORT_PATH,
    REVIEWED_PUBLIC_TRANSFER_ARTIFACT_ID,
    assert_public_transfer_review_matches,
    assert_review_boundary,
    build_reviewed_public_transfer_artifact,
    write_public_transfer_review_markdown,
)
from rag_lab.public_transfer_runtime import DEFAULT_PUBLIC_TRANSFER_FIXTURE_PATH, run_local_public_transfer_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish or verify the reviewed public-transfer evidence asset.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--publish", action="store_true", help="Run the public fixture and write reviewed files.")
    mode.add_argument("--check", action="store_true", help="Verify committed review files without rerunning models.")
    parser.add_argument("--confirm-public-transfer-review", action="store_true", help="Required with --publish.")
    parser.add_argument("--replace-reviewed-artifact", action="store_true", help="Allow intentional replacement of existing reviewed files.")
    parser.add_argument("--run-id", default="public_transfer_squad_v1_dev_v1_reviewed_v1")
    parser.add_argument("--fixture-directory", type=Path, default=DEFAULT_PUBLIC_TRANSFER_FIXTURE_PATH)
    parser.add_argument("--synthetic-baseline-artifact", type=Path, default=DEFAULT_BASELINE_ARTIFACT_PATH)
    parser.add_argument("--reviewed-artifact", type=Path, default=DEFAULT_REVIEWED_PUBLIC_TRANSFER_ARTIFACT_PATH)
    parser.add_argument("--reviewed-report", type=Path, default=DEFAULT_REVIEWED_PUBLIC_TRANSFER_REPORT_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokenizer", choices=("tiktoken", "diagnostic"), default="tiktoken")
    parser.add_argument("--tiktoken-encoding", default="cl100k_base")
    parser.add_argument("--character-max-characters", type=int, default=700)
    parser.add_argument("--sentence-aware-max-tokens", type=int, default=96)
    parser.add_argument("--hybrid-rrf-k", type=int, default=60)
    parser.add_argument("--retrieval-metric-k", type=int, default=5)
    parser.add_argument("--budgeted-render-profile", choices=tuple(profile.value for profile in ContextRenderProfile), default=ContextRenderProfile.COMPACT_CITATION.value)
    return parser.parse_args()


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _publish(args: argparse.Namespace) -> None:
    if not args.confirm_public_transfer_review:
        raise SystemExit("Refusing publication. Use --publish with --confirm-public-transfer-review after review.")
    artifact_path = _repo_path(args.reviewed_artifact)
    report_path = _repo_path(args.reviewed_report)
    existing = [path for path in (artifact_path, report_path) if path.exists()]
    if existing and not args.replace_reviewed_artifact:
        raise SystemExit("Refusing to replace existing reviewed public-transfer files. Use --replace-reviewed-artifact only after review.")
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
    run = run_local_public_transfer_comparison(project_root=PROJECT_ROOT, settings=settings, fixture_directory=args.fixture_directory)
    candidate = build_public_transfer_artifact(artifact_id="public_transfer_squad_v1_dev_v1_review_candidate", run=run)
    reviewed = build_reviewed_public_transfer_artifact(source_artifact=candidate)
    baseline = load_baseline_artifact(_repo_path(args.synthetic_baseline_artifact))
    assert_review_boundary(public_artifact=reviewed, synthetic_baseline=baseline)
    write_public_transfer_artifact(artifact=reviewed, path=artifact_path)
    write_public_transfer_review_markdown(public_artifact=reviewed, synthetic_baseline=baseline, path=report_path)
    print("PUBLIC TRANSFER REVIEW PUBLISHED: PASS")
    print(f"artifact_id={REVIEWED_PUBLIC_TRANSFER_ARTIFACT_ID}")
    print(f"fixture={reviewed.provenance.external_dataset_id}")
    print(f"documents={reviewed.provenance.source_document_count}")
    print(f"cases={reviewed.provenance.evaluation_case_count}")
    print(f"pipelines={len(reviewed.report.pipeline_definitions)}")
    print(f"artifact={artifact_path.relative_to(PROJECT_ROOT)}")
    print(f"report={report_path.relative_to(PROJECT_ROOT)}")


def _check(args: argparse.Namespace) -> None:
    artifact_path = _repo_path(args.reviewed_artifact)
    report_path = _repo_path(args.reviewed_report)
    reviewed = load_public_transfer_artifact(artifact_path)
    baseline = load_baseline_artifact(_repo_path(args.synthetic_baseline_artifact))
    assert_review_boundary(public_artifact=reviewed, synthetic_baseline=baseline)
    assert_public_transfer_review_matches(public_artifact=reviewed, synthetic_baseline=baseline, path=report_path)
    print("PUBLIC TRANSFER REVIEW CHECK: PASS")
    print(f"artifact={artifact_path.relative_to(PROJECT_ROOT)}")
    print(f"report={report_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    args = parse_args()
    if args.publish:
        _publish(args)
    else:
        _check(args)


if __name__ == "__main__":
    main()
