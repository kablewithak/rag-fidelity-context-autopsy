"""Typed, local-only context-autopsy view for the Streamlit RAG reliability demo.

This surface isolates a controlled context-pressure mechanism. It does not use the reviewed
four-pipeline artifact as evidence of a standard benchmark regression: the committed benchmark
has no context drops. Instead, it reconstructs the existing fixed synthetic pressure trace,
measures rendered prompt costs with one explicit tokenizer, and compares verbose versus compact
rendering under the same calibrated context window.

No raw rendered context or chunk text is retained in the view models.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.context_assembly import (
    ContextAssembler,
    ContextAssemblyConfig,
    ContextAutopsyReport,
    ContextDropReason,
    ContextRenderConfig,
    ContextRenderProfile,
    build_lost_evidence_report,
)
from rag_lab.diagnostic_scenarios import (
    CONTEXT_PRESSURE_CASE_ID,
    DEFAULT_STRESS_CHUNK_MAX_TOKENS,
    build_context_pressure_trace,
)
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.schemas import EvaluationCase, FailureLabel
from rag_lab.tokenizers import TiktokenTokenCounter, TokenCounter


DEFAULT_RESERVED_OUTPUT_TOKENS = 120


class ContextAutopsyExplorerError(ValueError):
    """Raised when the fixed controlled context-pressure proof cannot be reconstructed."""


class ContextDecisionView(BaseModel):
    """Privacy-bounded include/drop decision for one reranked synthetic candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    chunk_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=160)
    source_doc_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=3, max_length=96)
    first_stage_rank: int = Field(ge=1)
    reranked_rank: int = Field(ge=1)
    raw_context_token_count: int = Field(ge=1)
    rendered_context_token_count: int = Field(ge=1)
    rendering_token_tax: int
    gold_evidence_match: bool
    included: bool
    drop_reason: ContextDropReason | None = None

    @model_validator(mode="after")
    def validate_decision_state(self) -> "ContextDecisionView":
        if self.included and self.drop_reason is not None:
            raise ValueError("included decision cannot carry drop_reason")
        if not self.included and self.drop_reason is None:
            raise ValueError("dropped decision requires drop_reason")
        if self.rendering_token_tax != (
            self.rendered_context_token_count - self.raw_context_token_count
        ):
            raise ValueError(
                "rendering_token_tax must equal rendered_context_token_count minus raw_context_token_count"
            )
        return self


class ContextAssemblyView(BaseModel):
    """A bounded accounting view for one fixed render profile under one context window."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    render_profile: ContextRenderProfile
    max_context_tokens: int = Field(ge=1)
    static_prompt_tokens: int = Field(ge=1)
    reserved_output_tokens: int = Field(ge=0)
    available_evidence_tokens: int = Field(ge=1)
    used_raw_evidence_tokens: int = Field(ge=0)
    used_rendered_evidence_tokens: int = Field(ge=0)
    rendering_token_tax_tokens: int
    remaining_evidence_tokens: int = Field(ge=0)
    candidate_count: int = Field(ge=1)
    included_chunk_count: int = Field(ge=0)
    dropped_chunk_count: int = Field(ge=0)
    gold_evidence_found_in_candidates: bool
    gold_evidence_included: bool
    gold_evidence_dropped: bool
    gold_evidence_drop_reason: ContextDropReason | None = None
    decisions: tuple[ContextDecisionView, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_accounting(self) -> "ContextAssemblyView":
        if self.candidate_count != len(self.decisions):
            raise ValueError("candidate_count must match decision count")
        if self.included_chunk_count != sum(item.included for item in self.decisions):
            raise ValueError("included_chunk_count must match decisions")
        if self.dropped_chunk_count != sum(not item.included for item in self.decisions):
            raise ValueError("dropped_chunk_count must match decisions")
        if self.used_raw_evidence_tokens != sum(
            item.raw_context_token_count for item in self.decisions if item.included
        ):
            raise ValueError("used_raw_evidence_tokens must match included decisions")
        if self.used_rendered_evidence_tokens != sum(
            item.rendered_context_token_count for item in self.decisions if item.included
        ):
            raise ValueError("used_rendered_evidence_tokens must match included decisions")
        if self.rendering_token_tax_tokens != (
            self.used_rendered_evidence_tokens - self.used_raw_evidence_tokens
        ):
            raise ValueError("rendering_token_tax_tokens must match rendered-minus-raw use")
        if self.remaining_evidence_tokens != (
            self.available_evidence_tokens - self.used_rendered_evidence_tokens
        ):
            raise ValueError("remaining_evidence_tokens must match measured budget")
        return self


class ContextLossDiagnosisView(BaseModel):
    """The bounded, deterministic diagnosis for the deliberately displaced gold evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    gold_evidence_rank_before_context: int = Field(ge=1)
    drop_reason: ContextDropReason
    failure_labels: tuple[FailureLabel, ...] = Field(min_length=1, max_length=5)
    evidence_summary: str = Field(min_length=20, max_length=1_000)
    repair_recommendation: str = Field(min_length=20, max_length=1_000)


class ContextAutopsyCaseView(BaseModel):
    """One fixed controlled pressure case, rendered under two measured wrapper profiles."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case: EvaluationCase
    tokenizer_name: str = Field(min_length=3, max_length=160)
    sentence_aware_max_tokens: int = Field(ge=1)
    calibrated_context_tokens: int = Field(ge=1)
    reserved_output_tokens: int = Field(ge=0)
    verbose_audit: ContextAssemblyView
    compact_citation: ContextAssemblyView
    loss_diagnosis: ContextLossDiagnosisView

    @model_validator(mode="after")
    def validate_controlled_repair(self) -> "ContextAutopsyCaseView":
        if self.case.case_id != CONTEXT_PRESSURE_CASE_ID:
            raise ValueError("context autopsy view must use the fixed context-pressure case")
        if self.verbose_audit.render_profile is not ContextRenderProfile.VERBOSE_AUDIT:
            raise ValueError("verbose_audit must use verbose_audit profile")
        if self.compact_citation.render_profile is not ContextRenderProfile.COMPACT_CITATION:
            raise ValueError("compact_citation must use compact_citation profile")
        if (
            self.verbose_audit.max_context_tokens != self.calibrated_context_tokens
            or self.compact_citation.max_context_tokens != self.calibrated_context_tokens
        ):
            raise ValueError("both profiles must use the same calibrated context window")
        if self.verbose_audit.reserved_output_tokens != self.reserved_output_tokens:
            raise ValueError("verbose reserved_output_tokens must match case view")
        if self.compact_citation.reserved_output_tokens != self.reserved_output_tokens:
            raise ValueError("compact reserved_output_tokens must match case view")
        if not self.verbose_audit.gold_evidence_dropped:
            raise ValueError("controlled verbose profile must drop the gold evidence")
        if self.verbose_audit.gold_evidence_drop_reason is not ContextDropReason.BUDGET_EXHAUSTED:
            raise ValueError("controlled verbose profile must expose budget_exhausted")
        if not self.compact_citation.gold_evidence_included:
            raise ValueError("controlled compact profile must retain the gold evidence")
        return self


def load_context_autopsy_case_view(*, project_root: Path) -> ContextAutopsyCaseView:
    """Build the fixed controlled pressure view with the selected local model tokenizer."""

    cases = load_evaluation_cases(project_root / "data" / "eval_cases.jsonl")
    case = next((item for item in cases if item.case_id == CONTEXT_PRESSURE_CASE_ID), None)
    if case is None:
        raise ContextAutopsyExplorerError(
            f"missing fixed context-pressure case: {CONTEXT_PRESSURE_CASE_ID}"
        )

    token_counter = TiktokenTokenCounter(encoding_name="cl100k_base")
    return build_context_autopsy_case_view(
        case=case,
        token_counter=token_counter,
        sentence_aware_max_tokens=DEFAULT_STRESS_CHUNK_MAX_TOKENS,
        reserved_output_tokens=DEFAULT_RESERVED_OUTPUT_TOKENS,
    )


def build_context_autopsy_case_view(
    *,
    case: EvaluationCase,
    token_counter: TokenCounter,
    sentence_aware_max_tokens: int = DEFAULT_STRESS_CHUNK_MAX_TOKENS,
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
) -> ContextAutopsyCaseView:
    """Measure the fixed pressure trace under verbose and compact rendering profiles.

    The calibrated window is deliberately chosen so both profiles receive the identical total
    context budget. Verbose wrappers fill capacity before the rank-three gold candidate; compact
    wrappers retain it. This is a controlled diagnostic, not a live retrieval result or a
    replacement for the reviewed four-pipeline benchmark.
    """

    if case.case_id != CONTEXT_PRESSURE_CASE_ID:
        raise ContextAutopsyExplorerError(
            f"expected fixed context-pressure case {CONTEXT_PRESSURE_CASE_ID}, got {case.case_id}"
        )
    if sentence_aware_max_tokens < 1:
        raise ContextAutopsyExplorerError("sentence_aware_max_tokens must be at least 1")
    if reserved_output_tokens < 0:
        raise ContextAutopsyExplorerError("reserved_output_tokens must be non-negative")

    trace = build_context_pressure_trace(
        token_counter=token_counter,
        max_tokens=sentence_aware_max_tokens,
    )
    if trace.case_id != case.case_id:
        raise ContextAutopsyExplorerError(
            "fixed pressure trace and evaluation case must share one case_id"
        )
    if trace.gold_evidence_rank_after_rerank is None:
        raise ContextAutopsyExplorerError(
            "fixed pressure trace must retain a reranked gold-evidence rank"
        )

    verbose_full = _assemble(
        token_counter=token_counter,
        trace=trace,
        max_context_tokens=50_000,
        reserved_output_tokens=reserved_output_tokens,
        profile=ContextRenderProfile.VERBOSE_AUDIT,
    )
    compact_full = _assemble(
        token_counter=token_counter,
        trace=trace,
        max_context_tokens=50_000,
        reserved_output_tokens=reserved_output_tokens,
        profile=ContextRenderProfile.COMPACT_CITATION,
    )

    gold_index = next(
        (
            index
            for index, decision in enumerate(verbose_full.report.decisions)
            if decision.gold_evidence_match
        ),
        None,
    )
    if gold_index is None or gold_index < 1:
        raise ContextAutopsyExplorerError(
            "fixed pressure trace must place gold evidence after at least one distractor"
        )

    verbose_prefix_tokens = sum(
        decision.rendered_context_token_count
        for decision in verbose_full.report.decisions[:gold_index]
    )
    calibrated_context_tokens = max(
        verbose_full.report.actual_static_prompt_tokens
        + verbose_prefix_tokens
        + reserved_output_tokens,
        compact_full.report.actual_static_prompt_tokens
        + compact_full.report.used_rendered_evidence_tokens
        + reserved_output_tokens,
    )

    verbose_result = _assemble(
        token_counter=token_counter,
        trace=trace,
        max_context_tokens=calibrated_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
        profile=ContextRenderProfile.VERBOSE_AUDIT,
    )
    compact_result = _assemble(
        token_counter=token_counter,
        trace=trace,
        max_context_tokens=calibrated_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
        profile=ContextRenderProfile.COMPACT_CITATION,
    )

    verbose_view = _build_assembly_view(verbose_result.report)
    compact_view = _build_assembly_view(compact_result.report)
    lost_report = build_lost_evidence_report(
        reranking_trace=trace,
        autopsy_report=verbose_result.report,
    )
    if lost_report is None:
        raise ContextAutopsyExplorerError(
            "controlled verbose profile did not emit a lost-evidence diagnosis"
        )

    return ContextAutopsyCaseView(
        case=case,
        tokenizer_name=token_counter.name,
        sentence_aware_max_tokens=sentence_aware_max_tokens,
        calibrated_context_tokens=calibrated_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
        verbose_audit=verbose_view,
        compact_citation=compact_view,
        loss_diagnosis=ContextLossDiagnosisView(
            gold_evidence_rank_before_context=lost_report.gold_evidence_rank_before_context,
            drop_reason=lost_report.reason,
            failure_labels=tuple(lost_report.failure_labels),
            evidence_summary=lost_report.evidence_summary,
            repair_recommendation=lost_report.repair_recommendation,
        ),
    )


def _assemble(
    *,
    token_counter: TokenCounter,
    trace: object,
    max_context_tokens: int,
    reserved_output_tokens: int,
    profile: ContextRenderProfile,
):
    """Run only deterministic context accounting over the fixed synthetic candidate trace."""

    return ContextAssembler(
        token_counter=token_counter,
        config=ContextAssemblyConfig(
            max_context_tokens=max_context_tokens,
            reserved_output_tokens=reserved_output_tokens,
            render_config=ContextRenderConfig(profile=profile),
        ),
    ).assemble(reranking_trace=trace)


def _build_assembly_view(report: ContextAutopsyReport) -> ContextAssemblyView:
    """Reduce the report to a UI-safe view that excludes raw rendered context and chunk text."""

    return ContextAssemblyView(
        render_profile=report.render_profile,
        max_context_tokens=report.max_context_tokens,
        static_prompt_tokens=report.actual_static_prompt_tokens,
        reserved_output_tokens=report.reserved_output_tokens,
        available_evidence_tokens=report.available_evidence_tokens,
        used_raw_evidence_tokens=report.used_raw_evidence_tokens,
        used_rendered_evidence_tokens=report.used_rendered_evidence_tokens,
        rendering_token_tax_tokens=report.rendering_token_tax_tokens,
        remaining_evidence_tokens=report.remaining_evidence_tokens,
        candidate_count=report.candidate_count,
        included_chunk_count=report.included_chunk_count,
        dropped_chunk_count=report.dropped_chunk_count,
        gold_evidence_found_in_candidates=report.gold_evidence_found_in_candidates,
        gold_evidence_included=report.gold_evidence_included,
        gold_evidence_dropped=report.gold_evidence_dropped,
        gold_evidence_drop_reason=report.gold_evidence_drop_reason,
        decisions=tuple(
            ContextDecisionView(
                chunk_id=decision.chunk_id,
                source_doc_id=decision.source_doc_id,
                first_stage_rank=decision.first_stage_rank,
                reranked_rank=decision.reranked_rank,
                raw_context_token_count=decision.raw_context_token_count,
                rendered_context_token_count=decision.rendered_context_token_count,
                rendering_token_tax=decision.rendering_token_tax,
                gold_evidence_match=decision.gold_evidence_match,
                included=decision.included,
                drop_reason=decision.drop_reason,
            )
            for decision in report.decisions
        ),
    )
