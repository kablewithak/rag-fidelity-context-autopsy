from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.chunking_explorer import (
    ChunkingExplorerError,
    build_chunking_case_views,
)
from rag_lab.corpus_loader import load_synthetic_corpus
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cases():
    return load_evaluation_cases(PROJECT_ROOT / "data" / "eval_cases.jsonl")


def _documents():
    return load_synthetic_corpus(corpus_directory=PROJECT_ROOT / "data" / "corpus")


def _views():
    return build_chunking_case_views(
        cases=_cases(),
        documents=_documents(),
        token_counter=UnicodeCodePointTokenCounter(),
        character_max_characters=700,
        sentence_aware_max_tokens=512,
    )


def test_chunking_views_cover_each_fixed_case_with_both_standard_strategies() -> None:
    views = _views()

    assert len(views) == 18
    assert [view.case.case_id for view in views] == sorted(view.case.case_id for view in views)
    assert all(view.character_chunking.report.chunks for view in views)
    assert all(view.sentence_aware_token_chunking.report.chunks for view in views)
    assert all(
        view.character_chunking.report.tokenizer_name == "diagnostic:unicode_codepoint_v1"
        for view in views
    )


def test_standard_configuration_reports_actual_boundary_result_without_forcing_a_split() -> None:
    view = next(view for view in _views() if view.case.case_id == "token_boundary_export_017")

    assert view.character_chunking.configured_limit == 700
    assert view.character_chunking.report.gold_evidence_preserved is True
    assert view.character_chunking.report.gold_evidence_split is False
    assert view.sentence_aware_token_chunking.report.gold_evidence_preserved is True


def test_controlled_boundary_probe_splits_character_window_and_preserves_sentence_aware_evidence() -> None:
    view = next(view for view in _views() if view.case.case_id == "token_boundary_export_017")

    assert view.controlled_boundary_probe is not None
    probe = view.controlled_boundary_probe
    assert probe.character_window_characters == view.gold_evidence_start + 42
    assert probe.character_chunking.report.gold_evidence_preserved is False
    assert probe.character_chunking.report.gold_evidence_split is True
    assert probe.sentence_aware_token_chunking.report.gold_evidence_preserved is True
    assert probe.sentence_aware_token_chunking.report.gold_evidence_split is False


def test_missing_case_source_document_fails_closed() -> None:
    cases = _cases()
    invalid_case = cases[0].model_copy(update={"source_doc_id": "missing_document"})

    with pytest.raises(
        ChunkingExplorerError,
        match="fixed evaluation cases reference missing corpus documents",
    ):
        build_chunking_case_views(
            cases=[invalid_case],
            documents=_documents(),
            token_counter=UnicodeCodePointTokenCounter(),
        )


def test_case_with_nonexistent_gold_evidence_fails_closed() -> None:
    cases = _cases()
    invalid_case = cases[0].model_copy(
        update={"gold_evidence_text": "This evidence text does not exist in the source document."}
    )

    with pytest.raises(
        ChunkingExplorerError,
        match="gold evidence is not an exact substring",
    ):
        build_chunking_case_views(
            cases=[invalid_case],
            documents=_documents(),
            token_counter=UnicodeCodePointTokenCounter(),
        )
