from __future__ import annotations

from rag_lab.chunkers import CharacterChunker, SentenceAwareTokenChunker, build_chunking_report
from rag_lab.context_assembly import (
    ContextAssembler,
    ContextAssemblyConfig,
    ContextRenderConfig,
    ContextRenderProfile,
    build_lost_evidence_report,
)
from rag_lab.diagnostic_scenarios import (
    BOUNDARY_CASE_ID,
    DEFAULT_STRESS_CHUNK_MAX_TOKENS,
    MINIMUM_STRESS_CHUNK_COUNT,
    build_context_pressure_trace,
    build_stress_chunks,
    load_diagnostic_case,
    load_stress_source_text,
)
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


# This is an offline structural fixture, not a model-token claim. With the
# Unicode-codepoint counter, 800 preserves the complete gold sentence while
# leaving four controlled source chunks for the context-packing comparison.
OFFLINE_UNICODE_CONTEXT_PRESSURE_MAX_TOKENS = 800


def test_controlled_character_boundary_splits_gold_clause_and_sentence_aware_repair_preserves_it() -> None:
    counter = UnicodeCodePointTokenCounter()
    case = load_diagnostic_case(BOUNDARY_CASE_ID)
    source_text = load_stress_source_text()
    evidence_start = source_text.index(case.gold_evidence_text)

    baseline_chunks = CharacterChunker(
        token_counter=counter,
        max_characters=evidence_start + 42,
    ).chunk(text=source_text, source_doc_id=case.source_doc_id)
    repair_chunks = SentenceAwareTokenChunker(
        token_counter=counter,
        max_tokens=200,
    ).chunk(text=source_text, source_doc_id=case.source_doc_id)

    baseline_report = build_chunking_report(
        source_text=source_text,
        source_doc_id=case.source_doc_id,
        chunker_name=CharacterChunker.strategy,
        tokenizer_name=counter.name,
        chunks=baseline_chunks,
        gold_evidence_text=case.gold_evidence_text,
    )
    repair_report = build_chunking_report(
        source_text=source_text,
        source_doc_id=case.source_doc_id,
        chunker_name=SentenceAwareTokenChunker.strategy,
        tokenizer_name=counter.name,
        chunks=repair_chunks,
        gold_evidence_text=case.gold_evidence_text,
    )

    assert baseline_report.gold_evidence_split is True
    assert repair_report.gold_evidence_preserved is True


def test_context_pressure_default_chunk_budget_creates_real_multi_chunk_pressure() -> None:
    counter = UnicodeCodePointTokenCounter()

    chunks = build_stress_chunks(token_counter=counter)

    assert len(chunks) >= MINIMUM_STRESS_CHUNK_COUNT
    assert {chunk.tokenizer_name for chunk in chunks} == {counter.name}
    assert max(chunk.token_count for chunk in chunks) <= DEFAULT_STRESS_CHUNK_MAX_TOKENS


def test_rendered_context_pressure_baseline_loses_gold_and_compact_repair_restores_it() -> None:
    counter = UnicodeCodePointTokenCounter()
    trace = build_context_pressure_trace(
        token_counter=counter,
        max_tokens=OFFLINE_UNICODE_CONTEXT_PRESSURE_MAX_TOKENS,
    )

    verbose_full = ContextAssembler(
        token_counter=counter,
        config=ContextAssemblyConfig(
            max_context_tokens=50_000,
            reserved_output_tokens=120,
            render_config=ContextRenderConfig(profile=ContextRenderProfile.VERBOSE_AUDIT),
        ),
    ).assemble(reranking_trace=trace)
    compact_full = ContextAssembler(
        token_counter=counter,
        config=ContextAssemblyConfig(
            max_context_tokens=50_000,
            reserved_output_tokens=120,
            render_config=ContextRenderConfig(profile=ContextRenderProfile.COMPACT_CITATION),
        ),
    ).assemble(reranking_trace=trace)
    budget = max(
        verbose_full.report.actual_static_prompt_tokens
        + verbose_full.report.decisions[0].rendered_context_token_count
        + verbose_full.report.decisions[1].rendered_context_token_count
        + verbose_full.report.reserved_output_tokens,
        compact_full.report.actual_static_prompt_tokens
        + compact_full.report.used_rendered_evidence_tokens
        + compact_full.report.reserved_output_tokens,
    )

    baseline = ContextAssembler(
        token_counter=counter,
        config=ContextAssemblyConfig(
            max_context_tokens=budget,
            reserved_output_tokens=120,
            render_config=ContextRenderConfig(profile=ContextRenderProfile.VERBOSE_AUDIT),
        ),
    ).assemble(reranking_trace=trace)
    repair = ContextAssembler(
        token_counter=counter,
        config=ContextAssemblyConfig(
            max_context_tokens=budget,
            reserved_output_tokens=120,
            render_config=ContextRenderConfig(profile=ContextRenderProfile.COMPACT_CITATION),
        ),
    ).assemble(reranking_trace=trace)

    assert baseline.report.gold_evidence_dropped is True
    assert build_lost_evidence_report(reranking_trace=trace, autopsy_report=baseline.report)
    assert repair.report.gold_evidence_included is True
    assert build_lost_evidence_report(reranking_trace=trace, autopsy_report=repair.report) is None
