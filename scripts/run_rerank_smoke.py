"""Run one real hybrid-retrieval plus cross-encoder reranking trace over the synthetic corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_lab.chunkers import SentenceAwareTokenChunker
from rag_lab.corpus_loader import chunk_corpus, load_synthetic_corpus
from rag_lab.embedders import SentenceTransformerEmbeddingModel
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.rerankers import CrossEncoderReranker, SentenceTransformersCrossEncoderModel
from rag_lab.retrievers import HybridRetriever
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CASE_ID = "legal_termination_001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real hybrid retrieval trace followed by cross-encoder reranking over the "
            "synthetic corpus."
        )
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_k < 1:
        raise SystemExit("--candidate-k must be at least 1")
    if args.rrf_k < 1:
        raise SystemExit("--rrf-k must be at least 1")

    cases = load_evaluation_cases(PROJECT_ROOT / "data" / "eval_cases.jsonl")
    case = next((candidate for candidate in cases if candidate.case_id == args.case_id), None)
    if case is None:
        available = ", ".join(item.case_id for item in cases)
        raise SystemExit(f"unknown --case-id {args.case_id!r}; available: {available}")

    documents = load_synthetic_corpus(corpus_directory=PROJECT_ROOT / "data" / "corpus")
    chunks = chunk_corpus(
        documents,
        chunker=SentenceAwareTokenChunker(
            token_counter=UnicodeCodePointTokenCounter(),
            max_tokens=500,
        ),
    )
    embedding_model = SentenceTransformerEmbeddingModel(
        model_name=args.embedding_model,
        device="cpu",
    )
    first_stage_trace = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        rrf_k=args.rrf_k,
    ).retrieve(case=case, top_k=args.candidate_k)
    reranking_trace = CrossEncoderReranker(
        scoring_model=SentenceTransformersCrossEncoderModel(
            model_name=args.reranker_model,
            device="cpu",
        )
    ).rerank(first_stage_trace=first_stage_trace)
    print(json.dumps(reranking_trace.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
