"""Execute the four fixed RAG pipelines and print the bounded JSON report.

This remains the raw runner command. For the versioned baseline, executive markdown
readout, and regression gate, use ``run_comparison_baseline.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_lab.comparison_runtime import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    build_runtime_settings,
    run_local_four_pipeline_comparison,
)
from rag_lab.context_assembly import ContextRenderProfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all fixed evaluation cases through the four RAG comparison pipelines and "
            "print a JSON-safe report with bounded trace references."
        )
    )
    parser.add_argument("--run-id", default="local_four_pipeline_run_v1")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokenizer", choices=("tiktoken", "diagnostic"), default="tiktoken")
    parser.add_argument("--tiktoken-encoding", default="cl100k_base")
    parser.add_argument("--character-max-characters", type=int, default=700)
    parser.add_argument("--sentence-aware-max-tokens", type=int, default=96)
    parser.add_argument("--hybrid-rrf-k", type=int, default=60)
    parser.add_argument(
        "--retrieval-metric-k",
        type=int,
        default=5,
        help="Recall cutoff to aggregate from the shared first-stage candidate pool.",
    )
    parser.add_argument(
        "--budgeted-render-profile",
        choices=tuple(profile.value for profile in ContextRenderProfile),
        default=ContextRenderProfile.COMPACT_CITATION.value,
    )
    return parser.parse_args()


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
    result = run_local_four_pipeline_comparison(
        project_root=PROJECT_ROOT,
        settings=settings,
    )
    print(json.dumps(result.execution.report.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
