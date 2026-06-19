"""Schema-first boundary contracts for fixed RAG diagnostic evaluation cases."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
