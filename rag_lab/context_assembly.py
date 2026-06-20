"""Rendered-context budgeting and privacy-conscious lost-evidence autopsies.

The assembler consumes a fixed reranked candidate set. It never retrieves,
reranks, or truncates evidence silently. It renders the actual static prompt and
per-chunk citation wrappers, counts them with the selected tokenizer, then records
why every candidate was included or excluded.
"""
from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.schemas import EvidenceLossStage, FailureLabel, RerankingTrace
from rag_lab.tokenizers import TokenCounter


class ContextAssemblyInputError(ValueError):
    """Raised when context packing cannot make a trustworthy budget claim."""


class ContextDropReason(StrEnum):
    """Why a candidate was intentionally excluded from rendered context."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    DUPLICATE_TEXT = "duplicate_text"
    MAX_EVIDENCE_CHUNKS_REACHED = "max_evidence_chunks_reached"


class TokenizerAlignmentStatus(StrEnum):
    """Whether raw chunk counts came from the tokenizer used for packing."""

    ALIGNED = "aligned"
    MISMATCHED = "mismatched"


class ContextRenderProfile(StrEnum):
    """Deterministic evidence-wrapper profiles used for measurable prompt-tax tests."""

    VERBOSE_AUDIT = "verbose_audit"
    COMPACT_CITATION = "compact_citation"


class ContextRenderConfig(BaseModel):
    """Actual prompt and evidence-rendering policy counted by the selected tokenizer.

    The strings are synthetic, fixed, and safe to retain in the repository. In a
    customer deployment, the same interface would receive approved prompt text and
    a policy-controlled response contract.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    profile: ContextRenderProfile = ContextRenderProfile.VERBOSE_AUDIT
    system_instruction: str = Field(
        default=(
            "Answer the user only from the supplied evidence. State unsupported when the "
            "evidence does not establish the answer. Preserve source references in the answer."
        ),
        min_length=20,
        max_length=4_000,
    )
    response_contract: str = Field(
        default=(
            'Return JSON with keys "answer", "citations", and "unsupported". '
            "Citations must reference only supplied evidence chunk identifiers."
        ),
        min_length=20,
        max_length=4_000,
    )

    def render_static_prompt(self, *, query: str) -> str:
        if not query.strip():
            raise ContextAssemblyInputError("query must contain non-whitespace text")
        return (
            f"System instruction:\n{self.system_instruction}\n\n"
            f"Question:\n{query}\n\n"
            f"Response contract:\n{self.response_contract}\n\n"
            "Evidence:"
        )

    def render_evidence(self, *, chunk_id: str, source_doc_id: str, rank: int, text: str) -> str:
        if self.profile is ContextRenderProfile.VERBOSE_AUDIT:
            return (
                "\n\n<evidence "
                f'source_doc_id="{source_doc_id}" '
                f'chunk_id="{chunk_id}" '
                f'reranked_rank="{rank}">\n'
                f"{text}\n"
                "</evidence>"
            )
        if self.profile is ContextRenderProfile.COMPACT_CITATION:
            return f"\n\n[{source_doc_id}:{rank}]\n{text}"
        raise ContextAssemblyInputError("unsupported context render profile")


class ContextAssemblyConfig(BaseModel):
    """Explicit final-window capacity and deterministic rendered-context policy."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    max_context_tokens: int = Field(ge=1, le=1_000_000)
    reserved_output_tokens: int = Field(default=0, ge=0)
    render_config: ContextRenderConfig = Field(default_factory=ContextRenderConfig)
    max_evidence_chunks: int | None = Field(default=None, ge=1)
    deduplicate_exact_text: bool = True
    allow_tokenizer_mismatch: bool = False

    @model_validator(mode="after")
    def validate_window_leaves_possible_capacity(self) -> "ContextAssemblyConfig":
        if self.reserved_output_tokens >= self.max_context_tokens:
            raise ValueError("max_context_tokens must exceed reserved_output_tokens")
        return self


class ContextChunkDecision(BaseModel):
    """One include/drop decision without persisting raw chunk text."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=160)
    source_doc_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=3, max_length=96)
    reranked_rank: int = Field(ge=1)
    first_stage_rank: int = Field(ge=1)
    chunk_token_count: int = Field(ge=1)
    chunking_tokenizer_name: str = Field(min_length=3, max_length=160)
    raw_context_token_count: int = Field(ge=1)
    tokenizer_count_delta: int
    rendered_context_token_count: int = Field(ge=1)
    rendering_token_tax: int
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    gold_evidence_match: bool
    included: bool
    drop_reason: ContextDropReason | None = None

    @model_validator(mode="after")
    def validate_decision_state(self) -> "ContextChunkDecision":
        if self.included and self.drop_reason is not None:
            raise ValueError("included context chunks must not have a drop_reason")
        if not self.included and self.drop_reason is None:
            raise ValueError("dropped context chunks require a drop_reason")
        if self.tokenizer_count_delta != self.raw_context_token_count - self.chunk_token_count:
            raise ValueError(
                "tokenizer_count_delta must equal raw_context_token_count minus chunk_token_count"
            )
        if self.rendering_token_tax != self.rendered_context_token_count - self.raw_context_token_count:
            raise ValueError(
                "rendering_token_tax must equal rendered_context_token_count minus raw_context_token_count"
            )
        return self


class ContextAutopsyReport(BaseModel):
    """Bounded JSON-safe proof of final prompt capacity and evidence retention."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    context_tokenizer_name: str = Field(min_length=3, max_length=160)
    chunking_tokenizer_names: list[str] = Field(min_length=1)
    tokenizer_alignment_status: TokenizerAlignmentStatus
    render_profile: ContextRenderProfile
    max_context_tokens: int = Field(ge=1)
    actual_static_prompt_tokens: int = Field(ge=1)
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
    tokenizer_count_delta_detected: bool
    budget_underestimation_detected: bool
    budget_overestimation_detected: bool
    rendering_token_tax_detected: bool
    decisions: list[ContextChunkDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_autopsy_consistency(self) -> "ContextAutopsyReport":
        if self.candidate_count != len(self.decisions):
            raise ValueError("candidate_count must equal the number of context decisions")
        included = [decision for decision in self.decisions if decision.included]
        dropped = [decision for decision in self.decisions if not decision.included]
        if self.included_chunk_count != len(included):
            raise ValueError("included_chunk_count must equal the included decision count")
        if self.dropped_chunk_count != len(dropped):
            raise ValueError("dropped_chunk_count must equal the dropped decision count")
        if self.used_raw_evidence_tokens != sum(item.raw_context_token_count for item in included):
            raise ValueError("used_raw_evidence_tokens must equal included raw context token counts")
        if self.used_rendered_evidence_tokens != sum(
            item.rendered_context_token_count for item in included
        ):
            raise ValueError(
                "used_rendered_evidence_tokens must equal included rendered context token counts"
            )
        if self.rendering_token_tax_tokens != (
            self.used_rendered_evidence_tokens - self.used_raw_evidence_tokens
        ):
            raise ValueError(
                "rendering_token_tax_tokens must equal rendered evidence tokens minus raw evidence tokens"
            )
        if self.available_evidence_tokens != (
            self.max_context_tokens
            - self.actual_static_prompt_tokens
            - self.reserved_output_tokens
        ):
            raise ValueError("available_evidence_tokens must reflect the measured static prompt tax")
        if self.remaining_evidence_tokens != (
            self.available_evidence_tokens - self.used_rendered_evidence_tokens
        ):
            raise ValueError("remaining_evidence_tokens must equal available minus rendered evidence use")
        if self.remaining_evidence_tokens < 0:
            raise ValueError("remaining_evidence_tokens must not be negative")

        expected_ranks = list(range(1, len(self.decisions) + 1))
        if [decision.reranked_rank for decision in self.decisions] != expected_ranks:
            raise ValueError("context decisions must preserve contiguous reranked ranks")

        gold_decisions = [decision for decision in self.decisions if decision.gold_evidence_match]
        if self.gold_evidence_found_in_candidates:
            if len(gold_decisions) != 1:
                raise ValueError("gold evidence candidates must appear exactly once")
            gold_decision = gold_decisions[0]
            if self.gold_evidence_included != gold_decision.included:
                raise ValueError("gold_evidence_included must match the gold decision")
            if self.gold_evidence_dropped == gold_decision.included:
                raise ValueError("gold_evidence_dropped must be the inverse of gold inclusion")
            if self.gold_evidence_drop_reason != gold_decision.drop_reason:
                raise ValueError("gold_evidence_drop_reason must match the gold decision")
        elif gold_decisions:
            raise ValueError("gold evidence decisions require gold_evidence_found_in_candidates")
        elif self.gold_evidence_included or self.gold_evidence_dropped or self.gold_evidence_drop_reason is not None:
            raise ValueError("gold evidence fields must be empty when it was not in candidates")

        expected_chunking_names = sorted(
            {decision.chunking_tokenizer_name for decision in self.decisions}
        )
        if self.chunking_tokenizer_names != expected_chunking_names:
            raise ValueError("chunking_tokenizer_names must match decision tokenizer provenance")
        expected_alignment = (
            TokenizerAlignmentStatus.ALIGNED
            if expected_chunking_names == [self.context_tokenizer_name]
            else TokenizerAlignmentStatus.MISMATCHED
        )
        if self.tokenizer_alignment_status is not expected_alignment:
            raise ValueError("tokenizer_alignment_status must reflect tokenizer provenance")
        if self.tokenizer_count_delta_detected != any(
            item.tokenizer_count_delta != 0 for item in self.decisions
        ):
            raise ValueError("tokenizer_count_delta_detected must reflect decision deltas")
        if self.budget_underestimation_detected != any(
            item.tokenizer_count_delta > 0 for item in self.decisions
        ):
            raise ValueError("budget_underestimation_detected must reflect positive tokenizer deltas")
        if self.budget_overestimation_detected != any(
            item.tokenizer_count_delta < 0 for item in self.decisions
        ):
            raise ValueError("budget_overestimation_detected must reflect negative tokenizer deltas")
        if self.rendering_token_tax_detected != any(
            item.rendering_token_tax != 0 for item in self.decisions
        ):
            raise ValueError("rendering_token_tax_detected must reflect decision rendering tax")
        return self


class ContextAssemblyResult(BaseModel):
    """Ephemeral rendered prompt plus the report safe to persist or export."""

    model_config = ConfigDict(extra="forbid")

    context_text: str
    included_chunk_ids: list[str]
    report: ContextAutopsyReport

    @model_validator(mode="after")
    def validate_included_ids(self) -> "ContextAssemblyResult":
        expected = [
            decision.chunk_id for decision in self.report.decisions if decision.included
        ]
        if self.included_chunk_ids != expected:
            raise ValueError("included_chunk_ids must match included report decisions")
        return self


class LostEvidenceReport(BaseModel):
    """A narrow diagnosis emitted only when evidence dies in context assembly."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^[a-z0-9_]+$", min_length=5, max_length=96)
    loss_stage: EvidenceLossStage
    failure_labels: list[FailureLabel] = Field(min_length=1)
    gold_evidence_rank_before_context: int = Field(ge=1)
    reason: ContextDropReason
    evidence_summary: str = Field(min_length=20, max_length=1_000)
    repair_recommendation: str = Field(min_length=20, max_length=1_000)

    @model_validator(mode="after")
    def validate_context_loss_contract(self) -> "LostEvidenceReport":
        if self.loss_stage is not EvidenceLossStage.CONTEXT_ASSEMBLY:
            raise ValueError("lost evidence reports from this assembler must use context_assembly")
        if len(self.failure_labels) != len(set(self.failure_labels)):
            raise ValueError("failure_labels must not contain duplicates")
        return self


class ContextAssembler:
    """Pack reranked chunks under a measured rendered-prompt budget."""

    def __init__(self, *, token_counter: TokenCounter, config: ContextAssemblyConfig) -> None:
        self._token_counter = token_counter
        self._config = config
        if not self._token_counter.name.strip():
            raise ContextAssemblyInputError("token counter name must contain non-whitespace text")

    def assemble(self, *, reranking_trace: RerankingTrace) -> ContextAssemblyResult:
        candidates = reranking_trace.results
        if not candidates:
            raise ContextAssemblyInputError("context assembly requires reranked candidates")

        chunking_tokenizer_names = sorted({candidate.chunk.tokenizer_name for candidate in candidates})
        if any(name.startswith("unattributed:") for name in chunking_tokenizer_names):
            raise ContextAssemblyInputError(
                "context assembly requires attributed chunk tokenizer provenance"
            )
        aligned = chunking_tokenizer_names == [self._token_counter.name]
        if not aligned and not self._config.allow_tokenizer_mismatch:
            raise ContextAssemblyInputError(
                "context assembly received chunks from different tokenizers; set "
                "allow_tokenizer_mismatch=True only for an explicit diagnostic experiment"
            )

        static_prompt = self._config.render_config.render_static_prompt(
            query=reranking_trace.first_stage_trace.query
        )
        static_prompt_tokens = self._token_counter.count(static_prompt)
        available_evidence_tokens = (
            self._config.max_context_tokens
            - static_prompt_tokens
            - self._config.reserved_output_tokens
        )
        if available_evidence_tokens < 1:
            raise ContextAssemblyInputError(
                "measured static prompt and reserved output leave no evidence capacity; "
                "reduce prompt or schema text, reserve fewer output tokens, or increase the window"
            )

        current_prompt = static_prompt
        current_prompt_tokens = static_prompt_tokens
        included_count = 0
        seen_text_hashes: set[str] = set()
        decisions: list[ContextChunkDecision] = []

        for candidate in candidates:
            chunk = candidate.chunk
            text_hash = sha256(chunk.text.encode("utf-8")).hexdigest()
            raw_context_tokens = self._token_counter.count(chunk.text)
            rendered_fragment = self._config.render_config.render_evidence(
                chunk_id=chunk.chunk_id,
                source_doc_id=chunk.source_doc_id,
                rank=candidate.rank,
                text=chunk.text,
            )
            proposed_prompt = current_prompt + rendered_fragment
            proposed_prompt_tokens = self._token_counter.count(proposed_prompt)
            rendered_increment_tokens = proposed_prompt_tokens - current_prompt_tokens
            if rendered_increment_tokens < 1:
                raise ContextAssemblyInputError(
                    "rendered evidence chunk did not increase prompt token count; check renderer"
                )

            drop_reason: ContextDropReason | None = None
            if self._config.deduplicate_exact_text and text_hash in seen_text_hashes:
                drop_reason = ContextDropReason.DUPLICATE_TEXT
            elif (
                self._config.max_evidence_chunks is not None
                and included_count >= self._config.max_evidence_chunks
            ):
                drop_reason = ContextDropReason.MAX_EVIDENCE_CHUNKS_REACHED
            elif proposed_prompt_tokens + self._config.reserved_output_tokens > self._config.max_context_tokens:
                drop_reason = ContextDropReason.BUDGET_EXHAUSTED

            included = drop_reason is None
            decisions.append(
                ContextChunkDecision(
                    chunk_id=chunk.chunk_id,
                    source_doc_id=chunk.source_doc_id,
                    reranked_rank=candidate.rank,
                    first_stage_rank=candidate.first_stage_rank,
                    chunk_token_count=chunk.token_count,
                    chunking_tokenizer_name=chunk.tokenizer_name,
                    raw_context_token_count=raw_context_tokens,
                    tokenizer_count_delta=raw_context_tokens - chunk.token_count,
                    rendered_context_token_count=rendered_increment_tokens,
                    rendering_token_tax=rendered_increment_tokens - raw_context_tokens,
                    text_sha256=text_hash,
                    gold_evidence_match=candidate.gold_evidence_match,
                    included=included,
                    drop_reason=drop_reason,
                )
            )

            if included:
                current_prompt = proposed_prompt
                current_prompt_tokens = proposed_prompt_tokens
                included_count += 1
                seen_text_hashes.add(text_hash)

        included_decisions = [decision for decision in decisions if decision.included]
        gold_decision = next(
            (decision for decision in decisions if decision.gold_evidence_match),
            None,
        )
        used_raw = sum(item.raw_context_token_count for item in included_decisions)
        used_rendered = sum(item.rendered_context_token_count for item in included_decisions)
        report = ContextAutopsyReport(
            case_id=reranking_trace.case_id,
            context_tokenizer_name=self._token_counter.name,
            chunking_tokenizer_names=chunking_tokenizer_names,
            tokenizer_alignment_status=(
                TokenizerAlignmentStatus.ALIGNED
                if aligned
                else TokenizerAlignmentStatus.MISMATCHED
            ),
            render_profile=self._config.render_config.profile,
            max_context_tokens=self._config.max_context_tokens,
            actual_static_prompt_tokens=static_prompt_tokens,
            reserved_output_tokens=self._config.reserved_output_tokens,
            available_evidence_tokens=available_evidence_tokens,
            used_raw_evidence_tokens=used_raw,
            used_rendered_evidence_tokens=used_rendered,
            rendering_token_tax_tokens=used_rendered - used_raw,
            remaining_evidence_tokens=available_evidence_tokens - used_rendered,
            candidate_count=len(decisions),
            included_chunk_count=len(included_decisions),
            dropped_chunk_count=len(decisions) - len(included_decisions),
            gold_evidence_found_in_candidates=reranking_trace.gold_evidence_found,
            gold_evidence_included=gold_decision.included if gold_decision else False,
            gold_evidence_dropped=(not gold_decision.included) if gold_decision else False,
            gold_evidence_drop_reason=gold_decision.drop_reason if gold_decision else None,
            tokenizer_count_delta_detected=any(
                item.tokenizer_count_delta != 0 for item in decisions
            ),
            budget_underestimation_detected=any(
                item.tokenizer_count_delta > 0 for item in decisions
            ),
            budget_overestimation_detected=any(
                item.tokenizer_count_delta < 0 for item in decisions
            ),
            rendering_token_tax_detected=any(
                item.rendering_token_tax != 0 for item in decisions
            ),
            decisions=decisions,
        )
        return ContextAssemblyResult(
            context_text=current_prompt,
            included_chunk_ids=[item.chunk_id for item in included_decisions],
            report=report,
        )


def build_lost_evidence_report(
    *,
    reranking_trace: RerankingTrace,
    autopsy_report: ContextAutopsyReport,
) -> LostEvidenceReport | None:
    """Return a diagnosis only when complete gold evidence dies in context assembly."""
    if reranking_trace.case_id != autopsy_report.case_id:
        raise ContextAssemblyInputError("reranking trace and autopsy report must share a case_id")
    if not reranking_trace.gold_evidence_found:
        return None
    if not autopsy_report.gold_evidence_dropped:
        return None
    if autopsy_report.gold_evidence_drop_reason is None:
        raise ContextAssemblyInputError("dropped gold evidence requires a drop reason")
    if reranking_trace.gold_evidence_rank_after_rerank is None:
        raise ContextAssemblyInputError("gold evidence rank after rerank is required for context loss")

    labels = [FailureLabel.RELEVANT_CHUNK_DROPPED_BY_BUDGET]
    if autopsy_report.gold_evidence_drop_reason is ContextDropReason.BUDGET_EXHAUSTED:
        labels.append(FailureLabel.CONTEXT_BUDGET_EXCEEDED)
        recommendation = (
            "Use measured rendered-context accounting, reduce static prompt or response-contract "
            "tax, and compact citation wrappers before displacing high-relevance evidence."
        )
    elif autopsy_report.gold_evidence_drop_reason is ContextDropReason.DUPLICATE_TEXT:
        labels.append(FailureLabel.DUPLICATE_CONTEXT_WASTE)
        recommendation = (
            "Deduplicate exact or near-identical context before packing so high-value evidence "
            "is not displaced by repetition."
        )
    else:
        recommendation = (
            "Review the maximum evidence-chunk cap against the measured rendered budget and "
            "retain capacity for high-relevance evidence."
        )

    return LostEvidenceReport(
        case_id=autopsy_report.case_id,
        loss_stage=EvidenceLossStage.CONTEXT_ASSEMBLY,
        failure_labels=labels,
        gold_evidence_rank_before_context=reranking_trace.gold_evidence_rank_after_rerank,
        reason=autopsy_report.gold_evidence_drop_reason,
        evidence_summary=(
            "Complete gold evidence was present in the reranked candidate set but was not "
            f"included in rendered final context because {autopsy_report.gold_evidence_drop_reason.value}."
        ),
        repair_recommendation=recommendation,
    )
