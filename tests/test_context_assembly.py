from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_lab.context_assembly import (
    ContextAssembler,
    ContextAssemblyConfig,
    ContextAssemblyInputError,
    ContextDropReason,
    ContextRenderConfig,
    ContextRenderProfile,
    TokenizerAlignmentStatus,
    build_lost_evidence_report,
)
from rag_lab.schemas import (
    ChunkBoundaryQuality,
    ChunkingStrategy,
    EvaluationCase,
    RerankedChunk,
    RerankerMethod,
    RerankingTrace,
    RetrievedChunk,
    RetrievalMethod,
    RetrievalTrace,
    TextChunk,
)
from rag_lab.tokenizers import TokenCounter, UnicodeCodePointTokenCounter


class DoubleCountTokenizer:
    @property
    def name(self) -> str:
        return "fixture:double_count_v1"

    def encode(self, text: str) -> list[int]:
        return list(range(len(text) * 2))

    def decode(self, token_ids: list[int]) -> str:
        return "x" * (len(token_ids) // 2)

    def count(self, text: str) -> int:
        return len(text) * 2


class HalfCountTokenizer:
    @property
    def name(self) -> str:
        return "fixture:half_count_v1"

    def encode(self, text: str) -> list[int]:
        return list(range(max(1, len(text) // 2)))

    def decode(self, token_ids: list[int]) -> str:
        return "x" * (len(token_ids) * 2)

    def count(self, text: str) -> int:
        return max(1, len(text) // 2)


def _case() -> EvaluationCase:
    return EvaluationCase(
        case_id="legal_termination_001",
        document_type="legal_clause",
        query_type="policy_clause_query",
        query="When can the customer terminate after a material breach?",
        gold_evidence_text="Gold termination evidence.",
        gold_answer="After the uncured breach period.",
        expected_failure_mode="relevant_chunk_dropped_by_budget",
        source_doc_id="legal_terms",
        diagnostic_note="A fixed fixture for final rendered-context budgeting tests.",
    )


def _chunk(
    chunk_id: str,
    source_doc_id: str,
    text: str,
    *,
    tokenizer_name: str = "diagnostic:unicode_codepoint_v1",
) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        source_doc_id=source_doc_id,
        strategy=ChunkingStrategy.SENTENCE_AWARE_TOKEN,
        chunk_index=0,
        text=text,
        token_count=len(text),
        tokenizer_name=tokenizer_name,
        char_count=len(text),
        source_char_start=0,
        source_char_end=len(text),
        boundary_quality=ChunkBoundaryQuality.CLEAN_SENTENCE_BOUNDARY,
    )


def _reranking_trace(
    *,
    candidates: list[tuple[str, str, bool]],
    tokenizer_name: str = "diagnostic:unicode_codepoint_v1",
) -> RerankingTrace:
    retrieved = [
        RetrievedChunk(
            chunk=_chunk(
                chunk_id=chunk_id,
                source_doc_id=chunk_id.split("_")[0],
                text=text,
                tokenizer_name=tokenizer_name,
            ),
            rank=rank,
            score=1.0 - (rank * 0.1),
            gold_evidence_match=is_gold,
        )
        for rank, (chunk_id, text, is_gold) in enumerate(candidates, start=1)
    ]
    gold_rank = next((item.rank for item in retrieved if item.gold_evidence_match), None)
    first_stage = RetrievalTrace(
        case_id=_case().case_id,
        retriever_name=RetrievalMethod.BM25_OKAPI,
        lexical_analyzer_name="fixture:fixed_candidate_order_v1",
        query=_case().query,
        requested_top_k=len(retrieved),
        corpus_chunk_count=len(retrieved),
        results=retrieved,
        gold_evidence_found=gold_rank is not None,
        gold_evidence_rank=gold_rank,
    )
    reranked = [
        RerankedChunk(
            chunk=item.chunk,
            rank=item.rank,
            first_stage_rank=item.rank,
            first_stage_score=item.score,
            reranker_score=item.score,
            gold_evidence_match=item.gold_evidence_match,
        )
        for item in retrieved
    ]
    return RerankingTrace(
        case_id=_case().case_id,
        first_stage_retriever_name=RetrievalMethod.BM25_OKAPI,
        first_stage_trace=first_stage,
        reranker_name=RerankerMethod.CROSS_ENCODER,
        reranker_model_name="fixture:fixed_rank_v1",
        candidate_count=len(reranked),
        results=reranked,
        gold_evidence_found=gold_rank is not None,
        gold_evidence_rank_before_rerank=gold_rank,
        gold_evidence_rank_after_rerank=gold_rank,
    )


def _assemble(
    *,
    trace: RerankingTrace,
    max_context_tokens: int = 1_000,
    profile: ContextRenderProfile = ContextRenderProfile.VERBOSE_AUDIT,
    token_counter: TokenCounter | None = None,
    allow_tokenizer_mismatch: bool = False,
    max_evidence_chunks: int | None = None,
):
    return ContextAssembler(
        token_counter=token_counter or UnicodeCodePointTokenCounter(),
        config=ContextAssemblyConfig(
            max_context_tokens=max_context_tokens,
            reserved_output_tokens=40,
            render_config=ContextRenderConfig(profile=profile),
            max_evidence_chunks=max_evidence_chunks,
            allow_tokenizer_mismatch=allow_tokenizer_mismatch,
        ),
    ).assemble(reranking_trace=trace)


def _prompt_total(report: object) -> int:
    return (
        report.actual_static_prompt_tokens
        + report.used_rendered_evidence_tokens
        + report.reserved_output_tokens
    )


def test_context_assembler_counts_rendered_wrappers_not_only_raw_chunk_text() -> None:
    trace = _reranking_trace(
        candidates=[("legal_terms_sentence_001", "Gold termination evidence.", True)]
    )
    result = _assemble(trace=trace)

    decision = result.report.decisions[0]
    assert decision.raw_context_token_count == len("Gold termination evidence.")
    assert decision.rendered_context_token_count > decision.raw_context_token_count
    assert decision.rendering_token_tax > 0
    assert result.report.actual_static_prompt_tokens > 0
    assert result.report.rendering_token_tax_detected is True
    assert result.report.tokenizer_alignment_status is TokenizerAlignmentStatus.ALIGNED
    assert result.report.tokenizer_count_delta_detected is False
    assert result.context_text.startswith("System instruction:")


def test_compact_render_profile_recovers_gold_evidence_under_same_window() -> None:
    trace = _reranking_trace(
        candidates=[
            ("faq_sentence_001", "A" * 80, False),
            ("support_sentence_002", "B" * 80, False),
            ("legal_terms_sentence_003", "Gold termination evidence.", True),
        ]
    )
    verbose_full = _assemble(trace=trace, max_context_tokens=10_000)
    compact_full = _assemble(
        trace=trace,
        max_context_tokens=10_000,
        profile=ContextRenderProfile.COMPACT_CITATION,
    )
    verbose_prefix_before_gold = (
        verbose_full.report.actual_static_prompt_tokens
        + verbose_full.report.decisions[0].rendered_context_token_count
        + verbose_full.report.decisions[1].rendered_context_token_count
        + verbose_full.report.reserved_output_tokens
    )
    calibrated_budget = max(verbose_prefix_before_gold, _prompt_total(compact_full.report))
    assert calibrated_budget < _prompt_total(verbose_full.report)

    baseline = _assemble(trace=trace, max_context_tokens=calibrated_budget)
    repair = _assemble(
        trace=trace,
        max_context_tokens=calibrated_budget,
        profile=ContextRenderProfile.COMPACT_CITATION,
    )

    assert baseline.report.gold_evidence_dropped is True
    assert baseline.report.gold_evidence_drop_reason is ContextDropReason.BUDGET_EXHAUSTED
    assert repair.report.gold_evidence_included is True
    assert repair.report.rendering_token_tax_tokens < baseline.report.rendering_token_tax_tokens


def test_context_assembler_reports_gold_evidence_dropped_by_budget() -> None:
    trace = _reranking_trace(
        candidates=[
            ("faq_sentence_001", "A" * 100, False),
            ("legal_terms_sentence_001", "Gold termination evidence.", True),
        ]
    )
    unconstrained = _assemble(trace=trace, max_context_tokens=10_000)
    budget = (
        unconstrained.report.actual_static_prompt_tokens
        + unconstrained.report.decisions[0].rendered_context_token_count
        + unconstrained.report.reserved_output_tokens
    )
    result = _assemble(trace=trace, max_context_tokens=budget)

    assert result.report.gold_evidence_dropped is True
    lost = build_lost_evidence_report(reranking_trace=trace, autopsy_report=result.report)
    assert lost is not None
    assert lost.loss_stage.value == "context_assembly"
    assert [label.value for label in lost.failure_labels] == [
        "relevant_chunk_dropped_by_budget",
        "context_budget_exceeded",
    ]


def test_context_assembler_deduplicates_exact_text_without_persisting_raw_text() -> None:
    trace = _reranking_trace(
        candidates=[
            ("faq_sentence_001", "Repeated answer.", False),
            ("support_sentence_002", "Repeated answer.", False),
            ("legal_terms_sentence_001", "Gold termination evidence.", True),
        ]
    )
    result = _assemble(trace=trace)

    assert result.included_chunk_ids == [
        "faq_sentence_001",
        "legal_terms_sentence_001",
    ]
    assert result.report.decisions[1].drop_reason is ContextDropReason.DUPLICATE_TEXT
    serialized_report = result.report.model_dump(mode="json")
    assert "text" not in serialized_report["decisions"][0]
    assert "Repeated answer." not in str(serialized_report)


def test_context_assembler_rejects_unattributed_chunk_token_counts() -> None:
    trace = _reranking_trace(
        candidates=[("legal_terms_sentence_001", "Gold termination evidence.", True)]
    )
    altered_chunk = trace.results[0].chunk.model_copy(
        update={"tokenizer_name": "unattributed:unknown_v1"}
    )
    altered = trace.results[0].model_copy(update={"chunk": altered_chunk})
    altered_trace = trace.model_copy(update={"results": [altered]})

    with pytest.raises(ContextAssemblyInputError, match="attributed chunk tokenizer provenance"):
        _assemble(trace=altered_trace)


def test_context_assembler_rejects_tokenizer_mismatch_without_explicit_opt_in() -> None:
    trace = _reranking_trace(
        candidates=[("legal_terms_sentence_001", "Gold termination evidence.", True)],
        tokenizer_name="fixture:double_count_v1",
    )

    with pytest.raises(ContextAssemblyInputError, match="different tokenizers"):
        _assemble(trace=trace)


def test_context_assembler_records_positive_delta_as_budget_underestimation() -> None:
    trace = _reranking_trace(
        candidates=[("legal_terms_sentence_001", "Gold", True)]
    )
    result = _assemble(
        trace=trace,
        token_counter=DoubleCountTokenizer(),
        allow_tokenizer_mismatch=True,
    )

    decision = result.report.decisions[0]
    assert decision.chunk_token_count == 4
    assert decision.raw_context_token_count == 8
    assert decision.tokenizer_count_delta == 4
    assert result.report.budget_underestimation_detected is True
    assert result.report.budget_overestimation_detected is False


def test_context_assembler_records_negative_delta_as_budget_overestimation() -> None:
    trace = _reranking_trace(
        candidates=[("legal_terms_sentence_001", "Gold", True)],
        tokenizer_name="fixture:double_count_v1",
    )
    result = _assemble(
        trace=trace,
        token_counter=HalfCountTokenizer(),
        allow_tokenizer_mismatch=True,
    )

    decision = result.report.decisions[0]
    assert decision.chunk_token_count == 4
    assert decision.raw_context_token_count == 2
    assert decision.tokenizer_count_delta == -2
    assert result.report.budget_underestimation_detected is False
    assert result.report.budget_overestimation_detected is True


def test_context_assembler_respects_max_evidence_chunks() -> None:
    trace = _reranking_trace(
        candidates=[
            ("faq_sentence_001", "First", False),
            ("legal_terms_sentence_001", "Gold termination evidence.", True),
        ]
    )
    result = _assemble(trace=trace, max_evidence_chunks=1)

    assert result.included_chunk_ids == ["faq_sentence_001"]
    assert result.report.gold_evidence_drop_reason is ContextDropReason.MAX_EVIDENCE_CHUNKS_REACHED
    lost = build_lost_evidence_report(reranking_trace=trace, autopsy_report=result.report)
    assert lost is not None
    assert [label.value for label in lost.failure_labels] == ["relevant_chunk_dropped_by_budget"]


def test_context_assembler_does_not_mislabel_missing_candidate_as_context_loss() -> None:
    trace = _reranking_trace(candidates=[("faq_sentence_001", "Fallback", False)])
    result = _assemble(trace=trace)

    assert result.report.gold_evidence_found_in_candidates is False
    assert build_lost_evidence_report(reranking_trace=trace, autopsy_report=result.report) is None


def test_context_budget_must_exceed_reserved_output_tokens() -> None:
    with pytest.raises(ValidationError, match="exceed reserved_output_tokens"):
        ContextAssemblyConfig(max_context_tokens=10, reserved_output_tokens=10)


def test_lost_evidence_report_rejects_cross_case_pairing() -> None:
    trace = _reranking_trace(
        candidates=[("legal_terms_sentence_001", "Gold termination evidence.", True)]
    )
    result = _assemble(trace=trace)
    altered = result.report.model_copy(update={"case_id": "support_refund_001"})

    with pytest.raises(ContextAssemblyInputError, match="share a case_id"):
        build_lost_evidence_report(reranking_trace=trace, autopsy_report=altered)
