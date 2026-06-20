"""Shared local-runtime construction for four-pipeline comparison commands.

This module centralizes the model, tokenizer, corpus, and fixed-case wiring so the raw
comparison command and the committed-baseline command cannot drift apart silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag_lab.comparison_artifacts import (
    ComparisonRunProvenance,
    build_corpus_manifest_sha256,
    build_evaluation_cases_sha256,
)
from rag_lab.comparison_runner import (
    ComparisonExecutionConfig,
    ComparisonExecutionResult,
    FourPipelineComparisonRunner,
)
from rag_lab.context_assembly import ContextRenderProfile
from rag_lab.corpus_loader import load_synthetic_corpus
from rag_lab.embedders import SentenceTransformerEmbeddingModel
from rag_lab.eval_cases import assert_gold_evidence_exists, load_evaluation_cases
from rag_lab.rerankers import SentenceTransformersCrossEncoderModel
from rag_lab.tokenizers import TiktokenTokenCounter, TokenCounter, UnicodeCodePointTokenCounter


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ComparisonRuntimeSettings(BaseModel):
    """Explicit runtime settings for a local four-pipeline execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=5, max_length=160)
    embedding_model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=3, max_length=240)
    reranker_model_name: str = Field(default=DEFAULT_RERANKER_MODEL, min_length=3, max_length=240)
    device: str = Field(default="cpu", min_length=2, max_length=80)
    tokenizer_kind: str = Field(default="tiktoken", pattern=r"^(tiktoken|diagnostic)$")
    tiktoken_encoding: str = Field(default="cl100k_base", min_length=3, max_length=160)
    execution_config: ComparisonExecutionConfig = Field(default_factory=ComparisonExecutionConfig)


@dataclass(frozen=True, slots=True)
class LocalComparisonRun:
    """The typed report and artifact provenance from one local execution."""

    execution: ComparisonExecutionResult
    provenance: ComparisonRunProvenance


def build_token_counter(*, tokenizer_kind: str, encoding_name: str) -> TokenCounter:
    """Build the requested explicit token counter without fallback substitution."""

    if tokenizer_kind == "tiktoken":
        return TiktokenTokenCounter(encoding_name=encoding_name)
    if tokenizer_kind == "diagnostic":
        return UnicodeCodePointTokenCounter()
    raise ValueError(f"unsupported tokenizer kind: {tokenizer_kind}")


def run_local_four_pipeline_comparison(
    *,
    project_root: Path,
    settings: ComparisonRuntimeSettings,
) -> LocalComparisonRun:
    """Execute the fixed local benchmark and return report plus reproducibility metadata."""

    token_counter = build_token_counter(
        tokenizer_kind=settings.tokenizer_kind,
        encoding_name=settings.tiktoken_encoding,
    )
    cases = load_evaluation_cases(project_root / "data" / "eval_cases.jsonl")
    corpus_directory = project_root / "data" / "corpus"
    assert_gold_evidence_exists(cases, corpus_directory=corpus_directory)
    documents = load_synthetic_corpus(corpus_directory=corpus_directory)

    embedding_model = SentenceTransformerEmbeddingModel(
        model_name=settings.embedding_model_name,
        device=settings.device,
    )
    reranker_model = SentenceTransformersCrossEncoderModel(
        model_name=settings.reranker_model_name,
        device=settings.device,
    )
    runner = FourPipelineComparisonRunner(
        token_counter=token_counter,
        embedding_model=embedding_model,
        reranker_scoring_model=reranker_model,
        config=settings.execution_config,
    )
    execution = runner.run(
        run_id=settings.run_id,
        cases=cases,
        documents=documents,
    )
    provenance = ComparisonRunProvenance(
        tokenizer_name=token_counter.name,
        embedding_model_name=embedding_model.name,
        reranker_model_name=reranker_model.name,
        device=settings.device,
        execution_config=settings.execution_config,
        corpus_manifest_sha256=build_corpus_manifest_sha256(documents),
        evaluation_cases_sha256=build_evaluation_cases_sha256(cases),
        source_document_count=len(documents),
        evaluation_case_count=len(cases),
    )
    return LocalComparisonRun(execution=execution, provenance=provenance)


def build_runtime_settings(
    *,
    run_id: str,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    device: str = "cpu",
    tokenizer_kind: str = "tiktoken",
    tiktoken_encoding: str = "cl100k_base",
    character_max_characters: int = 700,
    sentence_aware_max_tokens: int = 96,
    hybrid_rrf_k: int = 60,
    retrieval_metric_k: int = 5,
    budgeted_render_profile: ContextRenderProfile = ContextRenderProfile.COMPACT_CITATION,
) -> ComparisonRuntimeSettings:
    """Build a runtime settings contract from CLI-safe primitive values."""

    return ComparisonRuntimeSettings(
        run_id=run_id,
        embedding_model_name=embedding_model_name,
        reranker_model_name=reranker_model_name,
        device=device,
        tokenizer_kind=tokenizer_kind,
        tiktoken_encoding=tiktoken_encoding,
        execution_config=ComparisonExecutionConfig(
            character_max_characters=character_max_characters,
            sentence_aware_max_tokens=sentence_aware_max_tokens,
            hybrid_rrf_k=hybrid_rrf_k,
            retrieval_metric_k=retrieval_metric_k,
            budgeted_render_profile=budgeted_render_profile,
        ),
    )
