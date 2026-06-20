"""Run real retrieval/reranking plus a measured rendered-context autopsy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from rag_lab.chunkers import SentenceAwareTokenChunker
from rag_lab.context_assembly import (
    ContextAssembler,
    ContextAssemblyConfig,
    ContextRenderConfig,
    ContextRenderProfile,
    build_lost_evidence_report,
)
from rag_lab.corpus_loader import chunk_corpus, load_synthetic_corpus
from rag_lab.embedders import SentenceTransformerEmbeddingModel
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.rerankers import CrossEncoderReranker, SentenceTransformersCrossEncoderModel
from rag_lab.retrievers import HybridRetriever
from rag_lab.tokenizers import TiktokenTokenCounter, TokenCounter, UnicodeCodePointTokenCounter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CASE_ID = "token_context_notice_018"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real hybrid retrieval and reranking, then measure rendered prompt and "
            "evidence capacity with an explicit tokenizer."
        )
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--max-context-tokens", type=int, default=1_200)
    parser.add_argument("--reserved-output-tokens", type=int, default=240)
    parser.add_argument("--max-evidence-chunks", type=int, default=None)
    parser.add_argument(
        "--render-profile",
        choices=tuple(profile.value for profile in ContextRenderProfile),
        default=ContextRenderProfile.VERBOSE_AUDIT.value,
    )
    parser.add_argument(
        "--tokenizer",
        choices=("diagnostic", "tiktoken"),
        default="diagnostic",
        help="Tokenizer used for raw chunk counts and final rendered-context capacity.",
    )
    parser.add_argument(
        "--chunking-tokenizer",
        choices=("match-context", "diagnostic", "tiktoken"),
        default="match-context",
        help=(
            "Tokenizer used before retrieval. The default aligns it with final context assembly. "
            "Choose another value only for an explicit tokenizer-mismatch diagnostic."
        ),
    )
    parser.add_argument("--tiktoken-encoding", default="cl100k_base")
    parser.add_argument("--chunk-max-tokens", type=int, default=96)
    return parser.parse_args()


def build_token_counter(*, tokenizer_kind: str, tiktoken_encoding: str) -> TokenCounter:
    if tokenizer_kind == "diagnostic":
        return UnicodeCodePointTokenCounter()
    if tokenizer_kind == "tiktoken":
        return TiktokenTokenCounter(encoding_name=tiktoken_encoding)
    raise ValueError(f"unsupported tokenizer kind: {tokenizer_kind}")


def main() -> None:
    args = parse_args()
    if args.candidate_k < 1:
        raise SystemExit("--candidate-k must be at least 1")
    if args.rrf_k < 1:
        raise SystemExit("--rrf-k must be at least 1")
    if args.chunk_max_tokens < 1:
        raise SystemExit("--chunk-max-tokens must be at least 1")

    context_token_counter = build_token_counter(
        tokenizer_kind=args.tokenizer,
        tiktoken_encoding=args.tiktoken_encoding,
    )
    chunking_tokenizer_kind = (
        args.tokenizer
        if args.chunking_tokenizer == "match-context"
        else args.chunking_tokenizer
    )
    chunking_token_counter = build_token_counter(
        tokenizer_kind=chunking_tokenizer_kind,
        tiktoken_encoding=args.tiktoken_encoding,
    )
    tokenizer_mismatch_requested = (
        chunking_token_counter.name != context_token_counter.name
    )

    cases = load_evaluation_cases(PROJECT_ROOT / "data" / "eval_cases.jsonl")
    case = next((candidate for candidate in cases if candidate.case_id == args.case_id), None)
    if case is None:
        available = ", ".join(item.case_id for item in cases)
        raise SystemExit(f"unknown --case-id {args.case_id!r}; available: {available}")

    documents = load_synthetic_corpus(corpus_directory=PROJECT_ROOT / "data" / "corpus")
    chunks = chunk_corpus(
        documents,
        chunker=SentenceAwareTokenChunker(
            token_counter=chunking_token_counter,
            max_tokens=args.chunk_max_tokens,
        ),
    )
    source_chunk_counts = dict(sorted(Counter(chunk.source_doc_id for chunk in chunks).items()))
    embedding_model = SentenceTransformerEmbeddingModel(
        model_name=args.embedding_model,
        device="cpu",
    )
    retrieval_trace = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        rrf_k=args.rrf_k,
    ).retrieve(case=case, top_k=args.candidate_k)
    reranking_trace = CrossEncoderReranker(
        scoring_model=SentenceTransformersCrossEncoderModel(
            model_name=args.reranker_model,
            device="cpu",
        )
    ).rerank(first_stage_trace=retrieval_trace)

    assembly = ContextAssembler(
        token_counter=context_token_counter,
        config=ContextAssemblyConfig(
            max_context_tokens=args.max_context_tokens,
            reserved_output_tokens=args.reserved_output_tokens,
            render_config=ContextRenderConfig(
                profile=ContextRenderProfile(args.render_profile)
            ),
            max_evidence_chunks=args.max_evidence_chunks,
            allow_tokenizer_mismatch=tokenizer_mismatch_requested,
        ),
    ).assemble(reranking_trace=reranking_trace)
    lost_evidence = build_lost_evidence_report(
        reranking_trace=reranking_trace,
        autopsy_report=assembly.report,
    )
    print(
        json.dumps(
            {
                "run_metadata": {
                    "case_id": case.case_id,
                    "source_doc_id": case.source_doc_id,
                    "chunking_tokenizer_name": chunking_token_counter.name,
                    "context_tokenizer_name": context_token_counter.name,
                    "tokenizer_mismatch_requested": tokenizer_mismatch_requested,
                    "chunk_max_tokens": args.chunk_max_tokens,
                    "corpus_chunk_count": len(chunks),
                    "source_chunk_counts": source_chunk_counts,
                },
                "context_autopsy": assembly.report.model_dump(mode="json"),
                "lost_evidence": lost_evidence.model_dump(mode="json") if lost_evidence else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
