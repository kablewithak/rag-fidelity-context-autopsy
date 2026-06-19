"""Schema-first boundary contracts for the RAG reliability lab."""

from __future__ import annotations

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


class RetrievedChunk(BaseModel):
    """One first-stage retrieval candidate with an inspectable score and rank."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk: TextChunk
    rank: int = Field(ge=1)
    score: float
    gold_evidence_match: bool


class RetrievalTrace(BaseModel):
    """Typed trace proving what a first-stage retriever returned for one eval case."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    retriever_name: RetrievalMethod
    lexical_analyzer_name: str = Field(min_length=3, max_length=160)
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
