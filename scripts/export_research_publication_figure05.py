"""Deterministic privacy-bounded export of the reviewed, separate SQuAD transfer probe.

Read-only: this does not rerun retrieval, models, or generation. The exporter
validates the reviewed public-transfer report and cross-checks all case outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.export_research_publication_gate01 import PIPELINE_ORDER, ROOT

if TYPE_CHECKING:
    from rag_lab.public_transfer_artifacts import PublicTransferComparisonArtifact
    from rag_lab.public_transfer_explorer import PublicTransferReviewView

OUTPUT = Path('design/research_publication_gate01/figure05_data.js')
ALLOWED_LOSSES = frozenset({'chunking', 'retrieval', 'ranking', 'context_assembly', 'generation'})
EXPECTED_INCLUSION = (26, 27, 28, 29)
EXPECTED_RECALL = (30, 29, 29, 29)


class Figure05ExportError(ValueError):
    """The reviewed public transfer does not meet the frozen figure contract."""


def build_projection(
    *,
    view: PublicTransferReviewView,
    artifact: PublicTransferComparisonArtifact,
) -> dict[str, Any]:
    """Emit only aggregate reviewed evidence and bounded provenance; never fixture text."""
    if (view.comparison_mode != 'side_by_side_not_pooled'
            or view.public_artifact_id != artifact.artifact_id
            or view.public_source_run_id != artifact.reference_run_id
            or view.public_evaluation_case_count != 30
            or view.public_source_document_count != 10
            or view.synthetic_case_count != 18
            or artifact.report.retrieval_metric_k != 5
            or view.public_dataset_id != 'squad_v1.1_dev'
            or view.public_dataset_version != '1.1'
            or artifact.artifact_schema_version != 'public_transfer_comparison_artifact_v1'):
        raise Figure05ExportError('Public fixture, provenance or separation boundary changed')
    if (artifact.provenance.evaluation_case_count != 30
            or artifact.provenance.source_document_count != 10
            or view.public_source_sha256 != artifact.provenance.source_sha256
            or view.public_fixture_manifest_sha256 != artifact.provenance.fixture_manifest_sha256):
        raise Figure05ExportError('Public fixture provenance mismatch')

    scorecards = {str(m.pipeline_id): m for m in view.pipeline_scorecards}
    artifact_metrics = {str(m.pipeline_id): m for m in artifact.report.pipeline_metrics}
    if (len(scorecards) != 4 or len(artifact_metrics) != 4
            or set(scorecards) != set(PIPELINE_ORDER)
            or set(artifact_metrics) != set(PIPELINE_ORDER)):
        raise Figure05ExportError('Unexpected reviewed pipeline scorecard coverage')
    outcomes = {(str(o.pipeline_id), o.case_id): o for o in artifact.report.case_outcomes}
    case_ids = {o.case_id for o in artifact.report.case_outcomes}
    if (len(case_ids) != 30 or len(outcomes) != 120
            or len(artifact.report.case_outcomes) != 120):
        raise Figure05ExportError('Expected exactly 120 distinct public outcomes on 30 cases')
    rows: list[dict[str, Any]] = []
    for pipeline_id in PIPELINE_ORDER:
        scores = scorecards[pipeline_id]
        reported = artifact_metrics[pipeline_id]
        selected = [outcomes.get((pipeline_id, case_id)) for case_id in sorted(case_ids)]
        if any(o is None for o in selected):
            raise Figure05ExportError(f'Incomplete public case coverage: {pipeline_id}')
        inc = sum(o.gold_evidence_included for o in selected)
        recalled = sum(o.retrieved_gold_rank is not None and o.retrieved_gold_rank <= 5 for o in selected)
        mrr = sum(
            1 / rank for o in selected
            if (rank := (o.reranked_gold_rank or o.retrieved_gold_rank)) is not None
            and rank <= 10
        ) / 30
        losses = Counter(str(o.loss_stage) for o in selected if not o.gold_evidence_included)
        if not set(losses) <= ALLOWED_LOSSES or sum(losses.values()) != 30 - inc:
            raise Figure05ExportError(f'Invalid loss taxonomy: {pipeline_id}')
        if losses != Counter({str(k): int(v) for k, v in reported.loss_stage_counts.items()}):
            raise Figure05ExportError(f'Loss-stage aggregation mismatch: {pipeline_id}')
        if (scores.evidence_inclusion_rate.numerator != inc
                or scores.retrieval_recall_at_k.numerator != recalled
                or scores.evidence_inclusion_rate.denominator != 30
                or scores.retrieval_recall_at_k.denominator != 30
                or reported.evidence_inclusion_rate.numerator != inc
                or reported.retrieval_recall_at_k.numerator != recalled
                or not math.isclose(float(scores.mrr_at_10), mrr, abs_tol=1e-12, rel_tol=0)
                or not math.isclose(float(reported.mrr_at_10), mrr, abs_tol=1e-12, rel_tol=0)):
            raise Figure05ExportError(f'Aggregate metric disagreement: {pipeline_id}')
        rows.append({
            'id': pipeline_id,
            'inclusion': {'numerator': inc, 'denominator': 30, 'value': inc / 30},
            'recallAt5': {'numerator': recalled, 'denominator': 30, 'value': recalled / 30},
            'mrrAt10': float(scores.mrr_at_10),
            'lossStages': {stage: losses.get(stage, 0) for stage in ('chunking', 'retrieval', 'ranking', 'context_assembly', 'generation')},
        })
    if tuple(x['inclusion']['numerator'] for x in rows) != EXPECTED_INCLUSION or tuple(x['recallAt5']['numerator'] for x in rows) != EXPECTED_RECALL:
        raise Figure05ExportError('Reviewed public fixture outcomes changed; new review required')
    if not math.isclose(rows[-1]['mrrAt10'] - rows[0]['mrrAt10'], view.mrr_delta, abs_tol=1e-12, rel_tol=0):
        raise Figure05ExportError('Public transfer MRR delta mismatch')
    return {
        'schemaVersion': 'figure05_public_transfer_v1',
        'scope': 'separate_reviewed_public_transfer_not_pooled',
        'artifactId': artifact.artifact_id,
        'runId': artifact.reference_run_id,
        'datasetId': view.public_dataset_id,
        'datasetVersion': view.public_dataset_version,
        'license': view.public_license_name,
        'sourceSha256': view.public_source_sha256,
        'fixtureManifestSha256': view.public_fixture_manifest_sha256,
        'sourceDocumentCount': 10,
        'publicCaseCount': 30,
        'syntheticCaseCount': 18,
        'recallCutoff': 5,
        'rows': rows,
    }


def render_js(projection: dict[str, Any]) -> str:
    encoded = json.dumps(projection, ensure_ascii=False, indent=2, allow_nan=False)
    encoded = encoded.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return ('/* GENERATED from a separately reviewed public transfer artifact.\n'
            ' * No public fixture questions, answers, chunks, prompts or model calls.\n'
            ' * Regenerate: python -m scripts.export_research_publication_figure05 --write */\n'
            f'const FIGURE05_PUBLIC = Object.freeze({encoded});\n')


def build_from_repo(root: Path) -> str:
    from rag_lab.public_transfer_explorer import load_public_transfer_review_view
    from rag_lab.public_transfer_artifacts import load_public_transfer_artifact
    from rag_lab.public_transfer_review import DEFAULT_REVIEWED_PUBLIC_TRANSFER_ARTIFACT_PATH
    view = load_public_transfer_review_view(project_root=root)
    artifact = load_public_transfer_artifact(root / DEFAULT_REVIEWED_PUBLIC_TRANSFER_ARTIFACT_PATH)
    return render_js(build_projection(view=view, artifact=artifact))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    generated = build_from_repo(ROOT)
    target = ROOT / OUTPUT
    if args.check:
        if not target.is_file() or target.read_text(encoding='utf-8') != generated:
            print('FAIL: Figure 05 reviewed public transfer export missing or stale; run --write')
            return 1
        print('PASS: Figure 05 matches reviewed, separate public transfer evidence')
        return 0
    target.write_text(generated, encoding='utf-8', newline='\n')
    print(f'WROTE: {target.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
