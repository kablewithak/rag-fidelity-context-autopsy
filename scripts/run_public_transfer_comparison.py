"""Run the fixed four-pipeline harness against the public SQuAD transfer fixture.

This command intentionally creates a separate local artifact. It never updates,
verifies against, or writes over the reviewed synthetic 18-case baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rag_lab.comparison_runtime import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    build_runtime_settings,
)
from rag_lab.context_assembly import ContextRenderProfile
from rag_lab.public_transfer_artifacts import (
    build_public_transfer_artifact,
    write_public_transfer_artifact,
    write_public_transfer_readout,
)
from rag_lab.public_transfer_runtime import (
    DEFAULT_PUBLIC_TRANSFER_FIXTURE_PATH,
    run_local_public_transfer_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = Path(
    "outputs/public_transfer/squad_v1_dev_v1_current.json"
)
DEFAULT_OUTPUT_MARKDOWN = Path(
    "outputs/public_transfer/squad_v1_dev_v1_current.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing fixed four pipelines against the public SQuAD "
            "transfer fixture and write separate bounded artifacts."
        )
    )
    parser.add_argument(
        "--run-id",
        default="public_transfer_squad_v1_dev_v1",
    )
    parser.add_argument(
        "--fixture-directory",
        type=Path,
        default=DEFAULT_PUBLIC_TRANSFER_FIXTURE_PATH,
        help="Public transfer fixture directory, relative to the repository root.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tokenizer",
        choices=("tiktoken", "diagnostic"),
        default="tiktoken",
    )
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
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Separate local JSON artifact path, relative to the repository root.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MARKDOWN,
        help="Separate local Markdown readout path, relative to the repository root.",
    )
    return parser.parse_args()


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
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
    run = run_local_public_transfer_comparison(
        project_root=PROJECT_ROOT,
        settings=settings,
        fixture_directory=args.fixture_directory,
    )
    artifact = build_public_transfer_artifact(
        artifact_id="public_transfer_squad_v1_dev_v1_current_run",
        run=run,
    )
    output_json = _repo_path(args.output_json)
    output_markdown = _repo_path(args.output_markdown)
    write_public_transfer_artifact(artifact=artifact, path=output_json)
    write_public_transfer_readout(artifact=artifact, path=output_markdown)

    print("PUBLIC TRANSFER RUN: PASS")
    print("fixture=squad_v1_dev_v1")
    print(f"documents={artifact.provenance.source_document_count}")
    print(f"cases={artifact.provenance.evaluation_case_count}")
    print(f"pipelines={len(artifact.report.pipeline_definitions)}")
    print(f"artifact={output_json.relative_to(PROJECT_ROOT)}")
    print(f"readout={output_markdown.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
