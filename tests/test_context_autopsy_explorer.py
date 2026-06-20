from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.context_assembly import ContextDropReason, ContextRenderProfile
from rag_lab.context_autopsy_explorer import (
    ContextAutopsyExplorerError,
    build_context_autopsy_case_view,
)
from rag_lab.diagnostic_scenarios import CONTEXT_PRESSURE_CASE_ID
from rag_lab.eval_cases import load_evaluation_cases
from rag_lab.schemas import FailureLabel
from rag_lab.tokenizers import UnicodeCodePointTokenCounter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_UNICODE_CONTEXT_PRESSURE_MAX_TOKENS = 800


def _case(case_id: str = CONTEXT_PRESSURE_CASE_ID):
    return next(
        case
        for case in load_evaluation_cases(PROJECT_ROOT / "data" / "eval_cases.jsonl")
        if case.case_id == case_id
    )


def _view():
    return build_context_autopsy_case_view(
        case=_case(),
        token_counter=UnicodeCodePointTokenCounter(),
        sentence_aware_max_tokens=OFFLINE_UNICODE_CONTEXT_PRESSURE_MAX_TOKENS,
    )


def test_context_autopsy_proves_verbose_budget_drop_and_compact_recovery() -> None:
    view = _view()

    assert view.case.case_id == CONTEXT_PRESSURE_CASE_ID
    assert view.verbose_audit.render_profile is ContextRenderProfile.VERBOSE_AUDIT
    assert view.compact_citation.render_profile is ContextRenderProfile.COMPACT_CITATION
    assert view.verbose_audit.gold_evidence_dropped is True
    assert view.verbose_audit.gold_evidence_drop_reason is ContextDropReason.BUDGET_EXHAUSTED
    assert view.compact_citation.gold_evidence_included is True


def test_context_autopsy_gives_both_profiles_the_same_measured_window() -> None:
    view = _view()

    assert view.verbose_audit.max_context_tokens == view.calibrated_context_tokens
    assert view.compact_citation.max_context_tokens == view.calibrated_context_tokens
    assert view.verbose_audit.reserved_output_tokens == view.reserved_output_tokens
    assert view.compact_citation.reserved_output_tokens == view.reserved_output_tokens
    assert view.verbose_audit.candidate_count == view.compact_citation.candidate_count == 3


def test_context_autopsy_measures_wrapper_tax_without_retaining_raw_context() -> None:
    view = _view()

    assert view.verbose_audit.rendering_token_tax_tokens > 0
    assert view.verbose_audit.used_rendered_evidence_tokens > view.verbose_audit.used_raw_evidence_tokens
    assert all("text" not in decision.model_fields for decision in view.verbose_audit.decisions)
    assert "context_text" not in view.model_fields


def test_context_autopsy_emits_a_specific_context_budget_diagnosis() -> None:
    view = _view()

    assert view.loss_diagnosis.gold_evidence_rank_before_context == 3
    assert view.loss_diagnosis.drop_reason is ContextDropReason.BUDGET_EXHAUSTED
    assert view.loss_diagnosis.failure_labels == (
        FailureLabel.RELEVANT_CHUNK_DROPPED_BY_BUDGET,
        FailureLabel.CONTEXT_BUDGET_EXCEEDED,
    )


def test_context_autopsy_rejects_any_case_other_than_the_fixed_pressure_case() -> None:
    with pytest.raises(ContextAutopsyExplorerError, match="expected fixed context-pressure case"):
        build_context_autopsy_case_view(
            case=_case("legal_payment_002"),
            token_counter=UnicodeCodePointTokenCounter(),
            sentence_aware_max_tokens=OFFLINE_UNICODE_CONTEXT_PRESSURE_MAX_TOKENS,
        )
