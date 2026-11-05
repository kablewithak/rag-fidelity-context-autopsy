"""Export a deterministic, privacy-bounded research Figure 01 JS data contract.

Reads only the reviewed synthetic benchmark and fixed evaluation cases; it does
not rerun models. Run from the repo root; see design/research_publication_gate01/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    from rag_lab.comparison_artifacts import ComparisonBaselineArtifact
    from rag_lab.schemas import EvaluationCase

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path('artifacts/comparisons/four_pipeline_baseline_v1.json')
CASES = Path('data/eval_cases.jsonl')
LABELS = Path('design/research_publication_gate01/case_labels_v1.json')
OUTPUT = Path('design/research_publication_gate01/data.js')
PIPELINE_ORDER = (
    'char_dense_naive',
    'token_dense_naive',
    'token_hybrid_naive',
    'token_hybrid_rerank_budgeted',
)
PIPELINE_DISPLAY = (
    ('C + D', 'Character + dense', 'Character chunking, dense retrieval'),
    ('T + D', 'Token + dense', 'Sentence-aware token chunking, dense retrieval'),
    ('T + H', 'Token + hybrid', 'Sentence-aware token chunking, hybrid retrieval'),
    ('T + H + R', 'Hybrid + rerank', 'Sentence-aware token chunking, hybrid retrieval, reranking and budget-aware selection'),
)
GROUP_DISPLAY = {
    'legal_clause': 'Legal clause',
    'support_policy': 'Support policy',
    'pricing_table': 'Pricing table',
    'multilingual_support': 'Multilingual',
    'code_log': 'Incident log',
    'faq': 'FAQ',
}
ALLOWED_LOSS_STAGES = {'chunking', 'retrieval', 'ranking'}


class PublicationExportError(ValueError):
    """Evidence coverage or scope disagrees with this frozen Figure 01 contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_projection(
    *,
    cases: Iterable[EvaluationCase],
    artifact: ComparisonBaselineArtifact,
    display_labels: Mapping[str, str],
    artifact_sha256: str,
    cases_sha256: str,
) -> dict[str, Any]:
    """Build reviewed facts only; no gold answers, source text, or raw traces."""
    ordered_cases = tuple(cases)
    case_ids = [c.case_id for c in ordered_cases]
    if len(case_ids) != 18 or len(set(case_ids)) != 18:
        raise PublicationExportError('Figure 01 requires exactly 18 unique fixed cases')
    if set(display_labels) != set(case_ids):
        raise PublicationExportError('Editorial case labels must have exact case-ID coverage')
    if any(not isinstance(display_labels[cid], str) or not display_labels[cid].strip() or len(display_labels[cid]) > 80 for cid in case_ids):
        raise PublicationExportError('Every editorial label must be nonempty and <=80 characters')
    if artifact.artifact_id != 'four_pipeline_baseline_v1':
        raise PublicationExportError('Figure 01 artifact ID mismatch')
    if artifact.reference_run_id != artifact.report.run_id:
        raise PublicationExportError('Reference run and report run differ')
    if artifact.provenance.evaluation_case_count != len(ordered_cases):
        raise PublicationExportError('Artifact case count differs from fixed cases')
    definitions = tuple(str(definition.pipeline_id) for definition in artifact.report.pipeline_definitions)
    if len(definitions) != 4 or set(definitions) != set(PIPELINE_ORDER):
        raise PublicationExportError('Pipeline definitions differ from the four reviewed configurations')
    outcomes = {}
    for outcome in artifact.report.case_outcomes:
        key = (outcome.case_id, str(outcome.pipeline_id))
        if key in outcomes or outcome.case_id not in case_ids or key[1] not in PIPELINE_ORDER:
            raise PublicationExportError(f'Duplicate or unknown outcome: {key}')
        outcomes[key] = outcome
    expected = {(cid, pid) for cid in case_ids for pid in PIPELINE_ORDER}
    if set(outcomes) != expected:
        raise PublicationExportError('Missing case-pipeline outcome or unexpected coverage')
    metrics = {str(metric.pipeline_id): metric for metric in artifact.report.pipeline_metrics}
    if set(metrics) != set(PIPELINE_ORDER):
        raise PublicationExportError('Pipeline metric coverage mismatch')

    public_rows = []
    counts = Counter()
    baseline_losses = Counter()
    for case in ordered_cases:
        category = str(case.document_type)
        if category not in GROUP_DISPLAY:
            raise PublicationExportError(f'Unexpected document type: {category}')
        states = []
        for index, pid in enumerate(PIPELINE_ORDER):
            outcome = outcomes[(case.case_id, pid)]
            included = bool(outcome.gold_evidence_included)
            loss = str(outcome.loss_stage) if outcome.loss_stage is not None else None
            if included and loss is not None:
                raise PublicationExportError('Included evidence has a loss stage')
            if not included and loss not in ALLOWED_LOSS_STAGES:
                raise PublicationExportError(f'Unsupported Figure 01 loss stage: {loss}')
            rank = outcome.retrieved_gold_rank
            rerank = outcome.reranked_gold_rank
            if (rank is not None and rank < 1) or (rerank is not None and (rerank < 1 or rank is None)):
                raise PublicationExportError('Invalid recorded rank state')
            if not included and loss == 'ranking' and rank is None:
                raise PublicationExportError('Ranking loss must have an observed first-stage rank')
            states.append([int(included), rank, rerank, loss])
            counts[pid] += int(included)
            if index == 0 and not included:
                baseline_losses[loss] += 1
        public_rows.append({
            'id': case.case_id,
            'label': display_labels[case.case_id],
            'group': GROUP_DISPLAY[category],
            'query': case.query,
            'states': states,
        })
    for pid in PIPELINE_ORDER:
        metric = metrics[pid].evidence_inclusion_rate
        if metric.numerator != counts[pid] or metric.denominator != 18:
            raise PublicationExportError(f'Aggregate metric differs from case outcomes: {pid}')
    if [counts[pid] for pid in PIPELINE_ORDER] != [13, 16, 18, 18]:
        raise PublicationExportError('Frozen Figure 01 evidence inclusion totals changed')
    if baseline_losses != Counter({'chunking': 3, 'retrieval': 1, 'ranking': 1}):
        raise PublicationExportError('Frozen Figure 01 baseline-loss taxonomy changed')
    for item in (artifact_sha256, cases_sha256):
        if len(item) != 64 or any(c not in '0123456789abcdef' for c in item):
            raise PublicationExportError('Invalid source SHA-256 provenance')
    return {
        'pipelines': [
            {'id': pid, 'short': text[0], 'name': text[1], 'description': text[2]}
            for pid, text in zip(PIPELINE_ORDER, PIPELINE_DISPLAY, strict=True)
        ],
        'cases': public_rows,
        'provenance': {
            'artifactId': artifact.artifact_id,
            'runId': artifact.report.run_id,
            'artifactSha256': artifact_sha256,
            'casesSha256': cases_sha256,
            'url': 'https://github.com/kablewithak/rag-fidelity-context-autopsy',
        },
    }


def render_js(projection: Mapping[str, Any]) -> str:
    """Render deterministic JavaScript constants consumed by the frozen UI."""
    def json_literal(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')

    return (
        '/* AUTO-GENERATED — do not edit. Source: reviewed Python evaluation artifacts.\n'
        ' * Regenerate: python scripts/export_research_publication_gate01.py --write\n'
        ' * Figure 01 only. Synthetic evidence selection; no model reruns. */\n'
        f'const PIPELINES = Object.freeze({json_literal(projection["pipelines"])});\n'
        f'const CASES = Object.freeze({json_literal(projection["cases"])});\n'
        f'const DATA_PROVENANCE = Object.freeze({json_literal(projection["provenance"])});\n'
    )


def build_from_repo(repo_root: Path) -> str:
    """Validate sources via existing Pydantic v2 loaders before projecting."""
    from rag_lab.comparison_artifacts import load_baseline_artifact
    from rag_lab.eval_cases import load_evaluation_cases

    artifact_path = repo_root / ARTIFACT
    cases_path = repo_root / CASES
    label_path = repo_root / LABELS
    artifact = load_baseline_artifact(artifact_path)
    cases = load_evaluation_cases(cases_path)
    labels = json.loads(label_path.read_text(encoding='utf-8'))
    if not isinstance(labels, dict):
        raise PublicationExportError('Editorial label asset must be a mapping')
    return render_js(build_projection(
        cases=cases, artifact=artifact, display_labels=labels,
        artifact_sha256=_sha256(artifact_path), cases_sha256=_sha256(cases_path),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--write', action='store_true', help='Rebuild the reviewed data.js from validated source artifacts')
    modes.add_argument('--check', action='store_true', help='Fail if the committed data.js differs from deterministic generation')
    args = parser.parse_args()
    destination = ROOT / OUTPUT
    generated = build_from_repo(ROOT)
    if args.check:
        if not destination.is_file() or destination.read_text(encoding='utf-8') != generated:
            print('FAIL: research publication data.js differs from reviewed artifact projection. Run --write and review the diff.')
            return 1
        print('PASS: Figure 01 data.js exactly matches the reviewed sources')
        return 0
    destination.write_text(generated, encoding='utf-8', newline='\n')
    print(f'WROTE: {destination.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
