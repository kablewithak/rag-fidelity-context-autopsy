"""Export a measured, privacy-bounded Figure 04 context-budget proof.

Only reuses the existing deterministic synthetic pressure trace and tokenizer.
This is independent of the 18-case reviewed four-pipeline comparison: it
never executes embedding retrieval, reranking, or generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.export_research_publication_gate01 import ROOT

OUTPUT = Path('design/research_publication_gate01/figure04_data.js')
SCHEMA_VERSION = 'figure04_context_budget_v1'


class Figure04ExportError(ValueError):
    """The controlled context proof no longer matches its reviewed contract."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise Figure04ExportError(message)


def build_context_projection(view: Any) -> dict[str, Any]:
    """Project validated view-model fields; never expose raw chunks or prompts."""
    _check(view.case.case_id == 'token_context_notice_018', 'Unexpected fixed pressure case')
    _check(view.calibrated_context_tokens == 468, 'Calibrated context window drifted')
    _check(view.reserved_output_tokens == 120, 'Output reserve changed')
    _check(view.loss_diagnosis.gold_evidence_rank_before_context == 3, 'Gold rank changed')
    _check(str(view.loss_diagnosis.drop_reason) == 'budget_exhausted', 'Loss reason changed')
    _check(view.tokenizer_name == 'tiktoken:cl100k_base', 'Tokenizer identity changed')

    rows: list[dict[str, Any]] = []
    for expected, assembly in (
        ('verbose_audit', view.verbose_audit),
        ('compact_citation', view.compact_citation),
    ):
        _check(str(assembly.render_profile) == expected, f'Render profile mismatch: {expected}')
        _check(assembly.max_context_tokens == 468, 'Profile context window drifted')
        _check(assembly.reserved_output_tokens == 120, 'Profile output reserve drifted')
        static = assembly.static_prompt_tokens
        raw = assembly.used_raw_evidence_tokens
        tax = assembly.rendering_token_tax_tokens
        remain = assembly.remaining_evidence_tokens
        _check(all(isinstance(x, int) and x >= 0 for x in (static, raw, tax, remain)), 'Nonnegative integer budget required')
        _check(static + raw + tax + remain + 120 == 468, 'Budget segments do not conserve total context')
        _check(assembly.available_evidence_tokens == 468 - static - 120, 'Available-evidence budget mismatch')
        _check(assembly.used_rendered_evidence_tokens == raw + tax, 'Rendered evidence budget mismatch')
        _check(assembly.candidate_count == len(assembly.decisions), 'Candidate count mismatch')
        _check(assembly.included_chunk_count == sum(bool(d.included) for d in assembly.decisions), 'Included candidate count mismatch')
        _check(assembly.dropped_chunk_count == sum(not d.included for d in assembly.decisions), 'Dropped candidate count mismatch')
        _check(sum(d.gold_evidence_match for d in assembly.decisions) == 1, 'Expected exactly one gold candidate')

        decisions: list[dict[str, Any]] = []
        for d in assembly.decisions:
            _check(d.raw_context_token_count > 0, 'Candidate raw cost must be positive')
            _check(d.rendered_context_token_count - d.raw_context_token_count == d.rendering_token_tax, 'Candidate wrapper tax mismatch')
            _check(d.rendering_token_tax >= 0, 'Negative candidate wrapper overhead unsupported')
            _check(not d.included or d.drop_reason is None, 'Included candidate has a drop reason')
            _check(d.included or d.drop_reason is not None, 'Dropped candidate lacks a reason')
            decisions.append({
                'rank': d.reranked_rank,
                'firstStageRank': d.first_stage_rank,
                'rawTokens': d.raw_context_token_count,
                'renderedTokens': d.rendered_context_token_count,
                'wrapperTokens': d.rendering_token_tax,
                'isGold': d.gold_evidence_match,
                'included': d.included,
                'dropReason': str(d.drop_reason) if d.drop_reason is not None else None,
            })
        _check(len({d['rank'] for d in decisions}) == len(decisions), 'Repeated reranked candidate positions')
        _check(sum(d['rawTokens'] for d in decisions if d['included']) == raw, 'Candidate raw-token accounting mismatch')
        _check(sum(d['renderedTokens'] for d in decisions if d['included']) == raw + tax, 'Candidate rendered-token accounting mismatch')
        gold = next(d for d in decisions if d['isGold'])
        _check(gold['rank'] == 3, 'Gold candidate rank does not match diagnosis')
        if expected == 'verbose_audit':
            _check(tax == 77 and not gold['included'], 'Verbose controlled result changed')
            _check(gold['dropReason'] == 'budget_exhausted', 'Verbose gold drop reason changed')
            _check(assembly.gold_evidence_dropped and not assembly.gold_evidence_included, 'Verbose gold flags changed')
        else:
            _check(tax == 25 and gold['included'], 'Compact controlled result changed')
            _check(assembly.gold_evidence_included and not assembly.gold_evidence_dropped, 'Compact gold flags changed')
        rows.append({
            'id': expected,
            'staticPromptTokens': static,
            'rawEvidenceTokens': raw,
            'wrapperTaxTokens': tax,
            'remainingTokens': remain,
            'includedCount': assembly.included_chunk_count,
            'droppedCount': assembly.dropped_chunk_count,
            'goldIncluded': gold['included'],
            'goldDropReason': gold['dropReason'],
            'decisions': sorted(decisions, key=lambda d: d['rank']),
        })
    return {
        'schemaVersion': SCHEMA_VERSION,
        'status': 'measured',
        'caseId': view.case.case_id,
        'tokenizer': view.tokenizer_name,
        'windowTokens': 468,
        'reservedOutputTokens': 120,
        'goldRank': 3,
        'profiles': rows,
        'scope': 'controlled_context_pressure_not_four_pipeline_benchmark',
    }


def render_js(projection: dict[str, Any]) -> str:
    encoded = json.dumps(projection, ensure_ascii=False, indent=2, allow_nan=False)
    encoded = encoded.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return (
        '/* GENERATED from the existing deterministic controlled context-pressure view.\n'
        ' * No raw text, chunks, prompts, answers, or model calls.\n'
        ' * Regenerate: python scripts/export_research_publication_figure04.py --write */\n'
        f'const FIGURE04_PROOF = Object.freeze({encoded});\n'
    )


def build_from_repo(root: Path) -> str:
    from rag_lab.context_autopsy_explorer import load_context_autopsy_case_view

    return render_js(build_context_projection(load_context_autopsy_case_view(project_root=root)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--write', action='store_true')
    modes.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        rendered = build_from_repo(ROOT)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f'FAIL: Could not validate the controlled context evidence: {error}')
        return 1
    path = ROOT / OUTPUT
    if args.check:
        if not path.is_file() or path.read_text(encoding='utf-8') != rendered:
            print('FAIL: Figure 04 evidence export missing/stale; regenerate with --write')
            return 1
        print('PASS: Figure 04 matches deterministic context-pressure accounting')
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding='utf-8', newline='\n')
    print(f'WROTE: {path.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
