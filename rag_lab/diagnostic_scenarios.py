"""Fixed, inspectable diagnostic scenarios for tokenization and context-pressure proof.

These helpers intentionally avoid live embedding and reranking model calls. They
construct deterministic traces over the same synthetic source material so a
specific failure mechanism can be reproduced and regression-tested.
"""
from __future__ import annotations

from pathlib import Path

from rag_lab.chunkers import SentenceAwareTokenChunker
from rag_lab.corpus_loader import load_synthetic_corpus
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.schemas import (
    EvaluationCase,
    RerankedChunk,
    RerankerMethod,
    RerankingTrace,
    RetrievedChunk,
    RetrievalMethod,
    RetrievalTrace,
    TextChunk,
)
from rag_lab.tokenizers import TokenCounter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRESS_SOURCE_DOC_ID = "tokenization_stress_policy"
BOUNDARY_CASE_ID = "token_boundary_export_017"
CONTEXT_PRESSURE_CASE_ID = "token_context_notice_018"
DEFAULT_STRESS_CHUNK_MAX_TOKENS = 96
MINIMUM_STRESS_CHUNK_COUNT = 4


class DiagnosticScenarioError(ValueError):
    """Raised when a fixed diagnostic case cannot be reconstructed exactly."""


def load_diagnostic_case(case_id: str) -> EvaluationCase:
    """Load one named synthetic diagnostic case from the fixed JSONL asset."""
    cases = load_evaluation_cases(PROJECT_ROOT / "data" / "eval_cases.jsonl")
    case = next((item for item in cases if item.case_id == case_id), None)
    if case is None:
        raise DiagnosticScenarioError(f"missing diagnostic case: {case_id}")
    return case


def load_stress_source_text() -> str:
    """Load only the controlled stress document through the strict corpus manifest."""
    documents = load_synthetic_corpus(corpus_directory=PROJECT_ROOT / "data" / "corpus")
    document = next((item for item in documents if item.source_doc_id == STRESS_SOURCE_DOC_ID), None)
    if document is None:
        raise DiagnosticScenarioError(f"missing stress source document: {STRESS_SOURCE_DOC_ID}")
    return document.text


def build_stress_chunks(
    *,
    token_counter: TokenCounter,
    max_tokens: int = DEFAULT_STRESS_CHUNK_MAX_TOKENS,
) -> list[TextChunk]:
    """Create sentence-aware stress chunks under an explicit tokenizer budget.

    The default budget is deliberately small enough to create at least four
    chunks under the supported real tokenizer path. This ensures the context
    pressure diagnostic has two distractors, one gold chunk, and one unused
    source chunk rather than silently collapsing into a document-level demo.
    """
    source_text = load_stress_source_text()
    source_token_count = token_counter.count(source_text)
    chunks = SentenceAwareTokenChunker(
        token_counter=token_counter,
        max_tokens=max_tokens,
    ).chunk(
        text=source_text,
        source_doc_id=STRESS_SOURCE_DOC_ID,
    )
    if len(chunks) < MINIMUM_STRESS_CHUNK_COUNT:
        raise DiagnosticScenarioError(
            "stress document produced "
            f"{len(chunks)} chunks under tokenizer={token_counter.name!r} "
            f"with max_tokens={max_tokens}; expected at least "
            f"{MINIMUM_STRESS_CHUNK_COUNT}. source_token_count={source_token_count}. "
            "Lower max_tokens or expand the controlled stress document."
        )
    return chunks


def build_context_pressure_trace(
    *,
    token_counter: TokenCounter,
    max_tokens: int = DEFAULT_STRESS_CHUNK_MAX_TOKENS,
) -> RerankingTrace:
    """Create a deterministic candidate order where complete gold evidence is rank 3.

    The trace deliberately models a ranking outcome rather than claiming it was
    produced by a live reranker. This lets the autopsy isolate context packing and
    rendered prompt tax from retrieval-model variability.
    """
    case = load_diagnostic_case(CONTEXT_PRESSURE_CASE_ID)
    chunks = build_stress_chunks(token_counter=token_counter, max_tokens=max_tokens)
    gold_chunk = next(
        (chunk for chunk in chunks if case.gold_evidence_text in chunk.text),
        None,
    )
    if gold_chunk is None:
        raise DiagnosticScenarioError(
            "gold context-pressure evidence was not preserved in sentence-aware chunks "
            f"under tokenizer={token_counter.name!r} with max_tokens={max_tokens}; "
            f"source_chunk_count={len(chunks)}. This scenario requires a tokenizer and "
            "chunk budget that keep the complete gold sentence intact."
        )

    distractors = sorted(
        (chunk for chunk in chunks if chunk.chunk_id != gold_chunk.chunk_id),
        key=lambda chunk: (-chunk.token_count, chunk.chunk_id),
    )[:2]
    if len(distractors) != 2:
        raise DiagnosticScenarioError("context-pressure scenario requires two distractor chunks")

    ordered_chunks = [*distractors, gold_chunk]
    retrieved_results = [
        RetrievedChunk(
            chunk=chunk,
            rank=rank,
            score=1.0 - rank * 0.1,
            gold_evidence_match=chunk.chunk_id == gold_chunk.chunk_id,
        )
        for rank, chunk in enumerate(ordered_chunks, start=1)
    ]
    first_stage_trace = RetrievalTrace(
        case_id=case.case_id,
        retriever_name=RetrievalMethod.BM25_OKAPI,
        lexical_analyzer_name="diagnostic:fixed_candidate_order_v1",
        query=case.query,
        requested_top_k=len(retrieved_results),
        corpus_chunk_count=len(chunks),
        results=retrieved_results,
        gold_evidence_found=True,
        gold_evidence_rank=3,
    )
    reranked_results = [
        RerankedChunk(
            chunk=result.chunk,
            rank=result.rank,
            first_stage_rank=result.rank,
            first_stage_score=result.score,
            reranker_score=result.score,
            gold_evidence_match=result.gold_evidence_match,
        )
        for result in retrieved_results
    ]
    return RerankingTrace(
        case_id=case.case_id,
        first_stage_retriever_name=RetrievalMethod.BM25_OKAPI,
        first_stage_trace=first_stage_trace,
        reranker_name=RerankerMethod.CROSS_ENCODER,
        reranker_model_name="diagnostic:fixed_rank_context_pressure_v1",
        candidate_count=len(reranked_results),
        results=reranked_results,
        gold_evidence_found=True,
        gold_evidence_rank_before_rerank=3,
        gold_evidence_rank_after_rerank=3,
    )
