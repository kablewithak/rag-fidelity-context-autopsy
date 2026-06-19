"""Optional local smoke run for the real Sentence Transformers dense retrieval adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_lab.chunkers import SentenceAwareTokenChunker
from rag_lab.corpus_loader import chunk_corpus, load_synthetic_corpus
from rag_lab.embedders import SentenceTransformerEmbeddingModel
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.retrievers import DenseRetriever
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CASE_ID = "legal_confidentiality_003"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real dense retrieval trace over the synthetic corpus."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

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
    embedding_model = SentenceTransformerEmbeddingModel(model_name=args.model, device="cpu")
    trace = DenseRetriever(chunks=chunks, embedding_model=embedding_model).retrieve(
        case=case,
        top_k=args.top_k,
    )
    print(json.dumps(trace.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
