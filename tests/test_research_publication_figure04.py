"""Figure 04 generated context-budget parity, privacy and presentation guardrails."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.export_research_publication_figure04 import (
    OUTPUT,
    Figure04ExportError,
    build_context_projection,
    build_from_repo,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'design/research_publication_gate01'


def _decision(rank: int, raw: int, wrapper: int, *, gold=False, included=True):
    return NS(
        reranked_rank=rank,
        first_stage_rank=rank,
        raw_context_token_count=raw,
        rendered_context_token_count=raw + wrapper,
        rendering_token_tax=wrapper,
        gold_evidence_match=gold,
        included=included,
        drop_reason=None if included else 'budget_exhausted',
    )


def _view():
    verbose = [
        _decision(1, 100, 35), _decision(2, 100, 42),
        _decision(3, 30, 10, gold=True, included=False),
    ]
    compact = [
        _decision(1, 70, 8), _decision(2, 70, 8),
        _decision(3, 80, 9, gold=True),
    ]
    def profile(name, static, raw, tax, left, ds, included):
        return NS(
            render_profile=name,
            max_context_tokens=468,
            reserved_output_tokens=120,
            static_prompt_tokens=static,
            used_raw_evidence_tokens=raw,
            used_rendered_evidence_tokens=raw + tax,
            rendering_token_tax_tokens=tax,
            remaining_evidence_tokens=left,
            available_evidence_tokens=468 - static - 120,
            candidate_count=len(ds),
            included_chunk_count=sum(d.included for d in ds),
            dropped_chunk_count=sum(not d.included for d in ds),
            gold_evidence_included=included,
            gold_evidence_dropped=not included,
            decisions=ds,
        )
    return NS(
        case=NS(case_id='token_context_notice_018'),
        calibrated_context_tokens=468,
        reserved_output_tokens=120,
        tokenizer_name='tiktoken:cl100k_base',
        loss_diagnosis=NS(gold_evidence_rank_before_context=3, drop_reason='budget_exhausted'),
        verbose_audit=profile('verbose_audit', 50, 200, 77, 21, verbose, False),
        compact_citation=profile('compact_citation', 35, 220, 25, 68, compact, True),
    )


def test_figure04_projection_preserves_exact_accounting_and_privacy() -> None:
    data = build_context_projection(_view())
    assert data['schemaVersion'] == 'figure04_context_budget_v1'
    assert data['status'] == 'measured'
    assert data['windowTokens'] == 468
    assert data['reservedOutputTokens'] == 120
    assert data['goldRank'] == 3
    assert [x['wrapperTaxTokens'] for x in data['profiles']] == [77, 25]
    assert [x['goldIncluded'] for x in data['profiles']] == [False, True]
    for p in data['profiles']:
        assert sum(p[k] for k in ('staticPromptTokens','rawEvidenceTokens','wrapperTaxTokens','remainingTokens')) + 120 == 468
        assert set(p) == {'id','staticPromptTokens','rawEvidenceTokens','wrapperTaxTokens','remainingTokens','includedCount','droppedCount','goldIncluded','goldDropReason','decisions'}
        assert all(set(d) == {'rank','firstStageRank','rawTokens','renderedTokens','wrapperTokens','isGold','included','dropReason'} for d in p['decisions'])


def test_figure04_rejects_broken_accounting() -> None:
    view = _view()
    view.verbose_audit.remaining_evidence_tokens = 22
    with pytest.raises(Figure04ExportError, match='conserve total context'):
        build_context_projection(view)


def test_figure04_rejects_changed_headline_finding() -> None:
    view = _view()
    view.compact_citation.rendering_token_tax_tokens = 26
    with pytest.raises(Figure04ExportError):
        build_context_projection(view)


def test_figure04_generated_asset_matches_local_measured_context() -> None:
    # Exact deterministic local parity; requires optional tiktoken dependency.
    text = build_from_repo(ROOT)
    assert (ROOT / OUTPUT).read_text(encoding='utf-8') == text, (
        'Regenerate with python -m scripts.export_research_publication_figure04 --write'
    )
    assert '"status": "measured"' in text
    for forbidden in ('gold_evidence_text','gold_answer','source_document_text','rendered_context','candidate_chunk_text'):
        assert forbidden not in text


def test_figure04_publication_and_workflow_integration() -> None:
    html = (PAGE / 'index.html').read_text(encoding='utf-8')
    js = (PAGE / 'app.js').read_text(encoding='utf-8')
    css = (PAGE / 'styles.css').read_text(encoding='utf-8')
    workflow = (ROOT / '.github/workflows/deploy-huggingface-space.yml').read_text(encoding='utf-8')
    assert '<script defer src="figure04_data.js"></script>' in html
    assert 'id="figure-four"' in html
    assert 'id="budget-profile-rows"' in html
    assert 'id="budget-decisions"' in html
    assert 'renderBudget();' in js
    assert 'assertFigure04Proof();' in js
    assert 'prefers-reduced-motion:reduce' in css
    assert 'position:sticky' in css
    assert 'python -m scripts.export_research_publication_figure04 --check' in workflow
    assert 'tests/test_research_publication_figure04.py' in workflow
    assert 'git push --porcelain hf HEAD:main' in workflow
    assert '--force' not in workflow


def test_figure04_node_syntax() -> None:
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is optional for the Python repo; browser smoke check separately')
    for file in ('app.js','figure04_data.js'):
        subprocess.run([node,'--check',str(PAGE/file)],check=True,capture_output=True,text=True)
