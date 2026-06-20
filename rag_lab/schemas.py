"""Schema-first boundary contracts for the RAG reliability lab."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentType(StrEnum):
    """Supported synthetic corpus document categories for the v1 lab."""

    LEGAL_CLAUSE = "legal_clause"
    SUPPORT_POLICY = "support_policy"
    PRICING_TABLE = "pricing_table"
    MULTILINGUAL_SUPPORT = "multilingual_support"
    CODE_LOG = "code_log"
    FAQ = "faq"


class QueryType(StrEnum):
    """Diagnostic query shapes used to exercise distinct RAG failure modes."""

    EXACT_TERM_QUERY = "exact_term_query"
    SEMANTIC_SUMMARY_QUERY = "semantic_summary_query"
    TABLE_LOOKUP_QUERY = "table_lookup_query"
    POLICY_CLAUSE_QUERY = "policy_clause_query"
    MULTI_HOP_QUERY = "multi_hop_query"
    MULTILINGUAL_QUERY = "multilingual_query"
    CODE_LOG_QUERY = "code_log_query"
    FAQ_QUERY = "faq_query"


class FailureLabel(StrEnum):
    """Standardized labels used by the lab's failure taxonomy."""

    BAD_CHUNK_BOUNDARY = "bad_chunk_boundary"
    GOLD_EVIDENCE_SPLIT = "gold_evidence_split"
    DENSE_RETRIEVAL_MISS = "dense_retrieval_miss"
    KEYWORD_RETRIEVAL_NEEDED = "keyword_retrieval_needed"
    RERANKER_NEEDED = "reranker_needed"
    RELEVANT_CHUNK_RANKED_TOO_LOW = "relevant_chunk_ranked_too_low"
    RELEVANT_CHUNK_DROPPED_BY_BUDGET = "relevant_chunk_dropped_by_budget"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
    DUPLICATE_CONTEXT_WASTE = "duplicate_context_waste"
    ANSWER_UNSUPPORTED_BY_CONTEXT = "answer_unsupported_by_context"
    CITATION_MISSING_OR_WRONG = "citation_missing_or_wrong"
    TOKEN_BUDGET_REGRESSION = "token_budget_regression"


class EvidenceLossStage(StrEnum):
    """The pipeline stage at which evidence becomes unusable for generation."""

    CHUNKING = "chunking"
    RETRIEVAL = "retrieval"
    RANKING = "ranking"
    CONTEXT_ASSEMBLY = "context_assembly"
    GENERATION = "generation"


class ChunkingStrategy(StrEnum):
    """The fixed chunking strategies compared by the lab."""

    CHARACTER = "character"
    TOKEN_WINDOW = "token_window"
    SENTENCE_AWARE_TOKEN = "sentence_aware_token"


class ChunkBoundaryQuality(StrEnum):
    """How safely a chunker preserves semantic, table, or event boundaries."""

    CHARACTER_CUT = "character_cut"
    TOKEN_WINDOW = "token_window"
    CLEAN_SENTENCE_BOUNDARY = "clean_sentence_boundary"
    MIXED = "mixed"


class RetrievalMethod(StrEnum):
    """Retriever implementations compared by the lab."""

    BM25_OKAPI = "bm25_okapi"
    DENSE = "dense"
    HYBRID = "hybrid"


class RerankerMethod(StrEnum):
    """Second-stage ranking implementations compared by the lab."""

    CROSS_ENCODER = "cross_encoder"


class HybridFusionMethod(StrEnum):
    """Rank-fusion algorithms that can combine lexical and dense retrieval evidence."""

    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"


class EvaluationCase(BaseModel):
    """A deterministic RAG diagnostic case with explicit gold evidence."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
        min_length=5,
        max_length=96,
        description="Stable snake_case identifier for traces, tests, and reports.",
    )
    document_type: DocumentType
    query_type: QueryType
    query: str = Field(min_length=8, max_length=1000)
    gold_evidence_text: str = Field(
        min_length=12,
        max_length=4000,
        description="Exact text expected to appear in the declared synthetic source document.",
    )
    gold_answer: str = Field(min_length=8, max_length=2000)
    expected_failure_mode: FailureLabel
    source_doc_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
        min_length=3,
        max_length=96,
        description="File stem in data/corpus, without the .txt extension.",
    )
    diagnostic_note: str = Field(
        min_length=12,
        max_length=1000,
        description="Why this case is useful for a specific reliability diagnosis.",
    )

    @field_validator("gold_evidence_text", "gold_answer", "diagnostic_note")
    @classmethod
    def reject_whitespace_only_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace text")
        return value


class CorpusDocument(BaseModel):
    """A loaded synthetic source document with deterministic integrity metadata."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_doc_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=3, max_length=96)
    document_type: DocumentType
    text: str = Field(min_length=1, max_length=100_000)
    char_count: int = Field(ge=1)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_text_length(self) -> "CorpusDocument":
        if len(self.text) != self.char_count:
            raise ValueError("char_count must equal the source text length")
        return self


class TextChunk(BaseModel):
    """A deterministic chunk with token and source-span evidence for later traces."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=160)
    source_doc_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=3, max_length=96)
    strategy: ChunkingStrategy
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=20000)
    token_count: int = Field(ge=1)
    tokenizer_name: str = Field(
        default="unattributed:unknown_v1",
        min_length=3,
        max_length=160,
        description=(
            "Tokenizer that produced token_count. Context assembly rejects unattributed "
            "chunks so final budget decisions always have traceable provenance."
        ),
    )
    char_count: int = Field(ge=1)
    source_char_start: int = Field(ge=0)
    source_char_end: int = Field(ge=1)
    boundary_quality: ChunkBoundaryQuality

    @model_validator(mode="after")
    def validate_source_span_matches_text(self) -> "TextChunk":
        if self.source_char_end <= self.source_char_start:
            raise ValueError("source_char_end must be greater than source_char_start")
        if self.source_char_end - self.source_char_start != self.char_count:
            raise ValueError("source character span must equal char_count")
        if len(self.text) != self.char_count:
            raise ValueError("char_count must equal the normalized chunk text length")
        return self


class ChunkingReport(BaseModel):
    """Inspectable evidence-preservation result for one chunking strategy and case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_doc_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=3, max_length=96)
    chunker_name: ChunkingStrategy
    tokenizer_name: str = Field(min_length=3, max_length=160)
    chunk_count: int = Field(ge=1)
    avg_token_count: float = Field(ge=0)
    max_token_count: int = Field(ge=1)
    gold_evidence_preserved: bool
    gold_evidence_split: bool
    boundary_quality: ChunkBoundaryQuality
    chunks: list[TextChunk] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chunk_summary(self) -> "ChunkingReport":
        if len(self.chunks) != self.chunk_count:
            raise ValueError("chunk_count must equal the number of chunk records")
        if max(chunk.token_count for chunk in self.chunks) != self.max_token_count:
            raise ValueError("max_token_count must match the emitted chunks")
        if self.gold_evidence_preserved and self.gold_evidence_split:
            raise ValueError("preserved gold evidence cannot also be marked as split")
        return self


class HybridScoreBreakdown(BaseModel):
    """Per-result component ranks and scores used to make hybrid fusion auditable."""

    model_config = ConfigDict(extra="forbid")

    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    fused_score: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_component_pairs(self) -> "HybridScoreBreakdown":
        if (self.bm25_rank is None) != (self.bm25_score is None):
            raise ValueError("bm25_rank and bm25_score must be set together")
        if (self.dense_rank is None) != (self.dense_score is None):
            raise ValueError("dense_rank and dense_score must be set together")
        if self.bm25_rank is None and self.dense_rank is None:
            raise ValueError("hybrid score breakdown requires at least one component rank")

        values = [
            value
            for value in (self.bm25_score, self.dense_score, self.fused_score)
            if value is not None
        ]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("hybrid scores must be finite")
        return self


class RetrievedChunk(BaseModel):
    """One first-stage retrieval candidate with an inspectable score and rank."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk: TextChunk
    rank: int = Field(ge=1)
    score: float
    gold_evidence_match: bool
    hybrid_score_breakdown: HybridScoreBreakdown | None = None

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retrieval score must be finite")
        return value


class RetrievalTrace(BaseModel):
    """Typed trace proving what a first-stage retriever returned for one eval case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    retriever_name: RetrievalMethod
    lexical_analyzer_name: str | None = Field(default=None, min_length=3, max_length=160)
    embedding_model_name: str | None = Field(default=None, min_length=3, max_length=240)
    embedding_dimension: int | None = Field(default=None, ge=1, le=16_384)
    hybrid_fusion_method: HybridFusionMethod | None = None
    hybrid_rrf_k: int | None = Field(default=None, ge=1, le=10_000)
    query: str = Field(min_length=1, max_length=1000)
    requested_top_k: int = Field(ge=1)
    corpus_chunk_count: int = Field(ge=1)
    results: list[RetrievedChunk] = Field(min_length=1)
    gold_evidence_found: bool
    gold_evidence_rank: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_retrieval_trace(self) -> "RetrievalTrace":
        if len(self.results) > self.requested_top_k:
            raise ValueError("result count cannot exceed requested_top_k")
        if len(self.results) > self.corpus_chunk_count:
            raise ValueError("result count cannot exceed corpus_chunk_count")

        expected_ranks = list(range(1, len(self.results) + 1))
        actual_ranks = [result.rank for result in self.results]
        if actual_ranks != expected_ranks:
            raise ValueError("retrieval result ranks must be contiguous and start at 1")

        chunk_ids = [result.chunk.chunk_id for result in self.results]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("retrieval results must not repeat a chunk")

        self._validate_retriever_metadata()

        match_ranks = [result.rank for result in self.results if result.gold_evidence_match]
        if self.gold_evidence_found:
            if len(match_ranks) != 1:
                raise ValueError("gold_evidence_found requires exactly one matching result")
            if self.gold_evidence_rank != match_ranks[0]:
                raise ValueError("gold_evidence_rank must match the gold evidence result rank")
        elif self.gold_evidence_rank is not None:
            raise ValueError("gold_evidence_rank must be null when gold evidence was not found")
        elif match_ranks:
            raise ValueError("gold evidence matches require gold_evidence_found to be true")

        return self

    def _validate_retriever_metadata(self) -> None:
        if self.retriever_name is RetrievalMethod.BM25_OKAPI:
            if self.lexical_analyzer_name is None:
                raise ValueError("bm25 traces require lexical_analyzer_name")
            if self.embedding_model_name is not None or self.embedding_dimension is not None:
                raise ValueError("bm25 traces must not include embedding metadata")
            if self.hybrid_fusion_method is not None or self.hybrid_rrf_k is not None:
                raise ValueError("bm25 traces must not include hybrid fusion metadata")
            if any(result.hybrid_score_breakdown is not None for result in self.results):
                raise ValueError("bm25 results must not include hybrid score breakdowns")
            return

        if self.retriever_name is RetrievalMethod.DENSE:
            if self.lexical_analyzer_name is not None:
                raise ValueError("dense traces must not include lexical_analyzer_name")
            if self.embedding_model_name is None or self.embedding_dimension is None:
                raise ValueError("dense traces require embedding model metadata")
            if self.hybrid_fusion_method is not None or self.hybrid_rrf_k is not None:
                raise ValueError("dense traces must not include hybrid fusion metadata")
            if any(result.hybrid_score_breakdown is not None for result in self.results):
                raise ValueError("dense results must not include hybrid score breakdowns")
            return

        if self.retriever_name is RetrievalMethod.HYBRID:
            if self.lexical_analyzer_name is None:
                raise ValueError("hybrid traces require lexical_analyzer_name")
            if self.embedding_model_name is None or self.embedding_dimension is None:
                raise ValueError("hybrid traces require embedding model metadata")
            if self.hybrid_fusion_method is not HybridFusionMethod.RECIPROCAL_RANK_FUSION:
                raise ValueError("hybrid traces require reciprocal_rank_fusion metadata")
            if self.hybrid_rrf_k is None:
                raise ValueError("hybrid traces require hybrid_rrf_k")
            for result in self.results:
                breakdown = result.hybrid_score_breakdown
                if breakdown is None:
                    raise ValueError("hybrid results require hybrid score breakdowns")
                if not math.isclose(result.score, breakdown.fused_score, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("hybrid result score must equal the fused score")
            return

        raise ValueError("unsupported retriever_name")


class RerankedChunk(BaseModel):
    """One fixed first-stage candidate after query-document rescoring."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk: TextChunk
    rank: int = Field(ge=1)
    first_stage_rank: int = Field(ge=1)
    first_stage_score: float
    reranker_score: float
    gold_evidence_match: bool

    @field_validator("first_stage_score", "reranker_score")
    @classmethod
    def require_finite_scores(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reranking scores must be finite")
        return value


class RerankingTrace(BaseModel):
    """Typed before/after rank evidence for a fixed first-stage candidate set."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    first_stage_retriever_name: RetrievalMethod
    first_stage_trace: RetrievalTrace
    reranker_name: RerankerMethod
    reranker_model_name: str = Field(min_length=3, max_length=240)
    candidate_count: int = Field(ge=1)
    results: list[RerankedChunk] = Field(min_length=1)
    gold_evidence_found: bool
    gold_evidence_rank_before_rerank: int | None = Field(default=None, ge=1)
    gold_evidence_rank_after_rerank: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_reranking_trace(self) -> "RerankingTrace":
        if self.case_id != self.first_stage_trace.case_id:
            raise ValueError("case_id must match the first-stage trace")
        if self.first_stage_retriever_name is not self.first_stage_trace.retriever_name:
            raise ValueError("first_stage_retriever_name must match the first-stage trace")
        if self.candidate_count != len(self.first_stage_trace.results):
            raise ValueError("candidate_count must match the first-stage trace result count")
        if len(self.results) != self.candidate_count:
            raise ValueError("reranking results must contain every first-stage candidate")

        expected_ranks = list(range(1, len(self.results) + 1))
        actual_ranks = [result.rank for result in self.results]
        if actual_ranks != expected_ranks:
            raise ValueError("reranking result ranks must be contiguous and start at 1")

        first_stage_by_chunk_id = {
            result.chunk.chunk_id: result for result in self.first_stage_trace.results
        }
        reranked_chunk_ids = [result.chunk.chunk_id for result in self.results]
        if len(reranked_chunk_ids) != len(set(reranked_chunk_ids)):
            raise ValueError("reranking results must not repeat a chunk")
        if set(reranked_chunk_ids) != set(first_stage_by_chunk_id):
            raise ValueError("reranking results must match the first-stage candidate set")

        for result in self.results:
            first_stage_result = first_stage_by_chunk_id[result.chunk.chunk_id]
            if result.chunk != first_stage_result.chunk:
                raise ValueError("reranked chunk must match the first-stage chunk")
            if result.first_stage_rank != first_stage_result.rank:
                raise ValueError("first_stage_rank must match the first-stage trace")
            if not math.isclose(
                result.first_stage_score,
                first_stage_result.score,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("first_stage_score must match the first-stage trace")
            if result.gold_evidence_match != first_stage_result.gold_evidence_match:
                raise ValueError("gold_evidence_match must match the first-stage trace")

        if self.gold_evidence_found != self.first_stage_trace.gold_evidence_found:
            raise ValueError("gold_evidence_found must match the first-stage trace")
        if self.gold_evidence_rank_before_rerank != self.first_stage_trace.gold_evidence_rank:
            raise ValueError("gold_evidence_rank_before_rerank must match the first-stage trace")

        reranked_match_ranks = [
            result.rank for result in self.results if result.gold_evidence_match
        ]
        if self.gold_evidence_found:
            if len(reranked_match_ranks) != 1:
                raise ValueError("gold evidence must appear exactly once in reranking results")
            if self.gold_evidence_rank_after_rerank != reranked_match_ranks[0]:
                raise ValueError(
                    "gold_evidence_rank_after_rerank must match the reranked gold evidence rank"
                )
        elif self.gold_evidence_rank_before_rerank is not None:
            raise ValueError("gold evidence rank before rerank must be null when not found")
        elif self.gold_evidence_rank_after_rerank is not None:
            raise ValueError("gold evidence rank after rerank must be null when not found")
        elif reranked_match_ranks:
            raise ValueError("reranking gold evidence matches require gold_evidence_found to be true")

        return self


class FailureDiagnosis(BaseModel):
    """Structured contract reserved for later chunking, retrieval, and context traces."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    failure_labels: list[FailureLabel] = Field(min_length=1)
    loss_stage: EvidenceLossStage
    evidence_summary: str = Field(min_length=8, max_length=2000)
    repair_recommendation: str = Field(min_length=8, max_length=2000)

    @field_validator("failure_labels")
    @classmethod
    def require_unique_failure_labels(
        cls,
        labels: list[FailureLabel],
    ) -> list[FailureLabel]:
        if len(labels) != len(set(labels)):
            raise ValueError("failure_labels must not contain duplicates")
        return labels
