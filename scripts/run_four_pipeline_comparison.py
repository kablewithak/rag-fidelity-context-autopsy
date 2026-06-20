"""Run the fixed real four-pipeline RAG comparison locally.

The command prints a privacy-bounded JSON comparison artifact to stdout. It does not
write report files in this slice; Phase 7C will add explicit versioned output writing
and a buyer-facing markdown render once this execution boundary is stable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_lab.comparison_runner import (
    ComparisonExecutionConfig,
    FourPipelineComparisonRunner,
)
from rag_lab.context_assembly import ContextRenderProfile
from rag_lab.corpus_loader import load_synthetic_corpus
from rag_lab.embedders import SentenceTransformerEmbeddingModel
from rag_lab.eval_cases import assert_gold_evidence_exists, load_evaluation_cases
from rag_lab.rerankers import SentenceTransformersCrossEncoderModel
from rag_lab.tokenizers import TiktokenTokenCounter, TokenCounter, UnicodeCodePointTokenCounter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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


def build_token_counter(*, tokenizer_kind: str, encoding_name: str) -> TokenCounter:
    if tokenizer_kind == "tiktoken":
        return TiktokenTokenCounter(encoding_name=encoding_name)
    if tokenizer_kind == "diagnostic":
        return UnicodeCodePointTokenCounter()
    raise ValueError(f"unsupported tokenizer kind: {tokenizer_kind}")


def main() -> None:
    args = parse_args()
    token_counter = build_token_counter(
        tokenizer_kind=args.tokenizer,
        encoding_name=args.tiktoken_encoding,
    )
    cases = load_evaluation_cases(PROJECT_ROOT / "data" / "eval_cases.jsonl")
    corpus_directory = PROJECT_ROOT / "data" / "corpus"
    assert_gold_evidence_exists(cases, corpus_directory=corpus_directory)
    documents = load_synthetic_corpus(corpus_directory=corpus_directory)

    runner = FourPipelineComparisonRunner(
        token_counter=token_counter,
        embedding_model=SentenceTransformerEmbeddingModel(
            model_name=args.embedding_model,
            device=args.device,
        ),
        reranker_scoring_model=SentenceTransformersCrossEncoderModel(
            model_name=args.reranker_model,
            device=args.device,
        ),
        config=ComparisonExecutionConfig(
            character_max_characters=args.character_max_characters,
            sentence_aware_max_tokens=args.sentence_aware_max_tokens,
            hybrid_rrf_k=args.hybrid_rrf_k,
            retrieval_metric_k=args.retrieval_metric_k,
            budgeted_render_profile=ContextRenderProfile(args.budgeted_render_profile),
        ),
    )
    result = runner.run(
        run_id=args.run_id,
        cases=cases,
        documents=documents,
    )
    print(json.dumps(result.report.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
