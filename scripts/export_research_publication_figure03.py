"""Deterministic Figure 03 metric projection from the frozen synthetic comparison.

This does not run retrieval, generation or any model inference. It exports only
aggregated, schema-validated metrics and reviewed provenance identifiers.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_lab.comparison_artifacts import ComparisonBaselineArtifact

from scripts.export_research_publication_gate01 import ARTIFACT, PIPELINE_ORDER, ROOT

OUTPUT = Path('design/research_publication_gate01/figure03_data.js')


class Figure03ExportError(ValueError):
    """The reviewed data is inconsistent or outside Figure 03's fixed scope."""


def build_metric_projection(artifact: ComparisonBaselineArtifact) -> dict[str, Any]:
    """Cross-check aggregate metrics against the reviewed case-level outcomes."""
    report = artifact.report
    if artifact.artifact_id != 'four_pipeline_baseline_v1' or artifact.reference_run_id != report.run_id:
        raise Figure03ExportError('Unreviewed artifact identity or source run')
    if report.retrieval_metric_k != 5 or artifact.provenance.evaluation_case_count != 18:
        raise Figure03ExportError('Unexpected retrieval cutoff or evaluation case count')

    metrics = {str(row.pipeline_id): row for row in report.pipeline_metrics}
    if len(report.pipeline_metrics) != 4 or set(metrics) != set(PIPELINE_ORDER):
        raise Figure03ExportError('Incomplete or duplicated pipeline metric coverage')
    outcomes = {(str(o.pipeline_id), o.case_id): o for o in report.case_outcomes}
    if len(outcomes) != 72 or len(report.case_outcomes) != 72:
        raise Figure03ExportError('Expected exactly 72 unique case-pipeline outcomes')
    cases = {case_id for _pipeline, case_id in outcomes}
    if len(cases) != 18 or {pipeline for pipeline, _case in outcomes} != set(PIPELINE_ORDER):
        raise Figure03ExportError('Incomplete reviewed case coverage')

    rows: list[dict[str, Any]] = []
    for pipeline_id in PIPELINE_ORDER:
        metric = metrics[pipeline_id]
        evidence = [outcomes.get((pipeline_id, case_id)) for case_id in cases]
        if any(o is None for o in evidence):
            raise Figure03ExportError(f'Missing reviewed case outcome for {pipeline_id}')
        included = sum(bool(o.gold_evidence_included) for o in evidence)
        recalled = sum(o.retrieved_gold_rank is not None and o.retrieved_gold_rank <= 5 for o in evidence)
        mrr = sum(
            1 / final_rank
            for o in evidence
            if (final_rank := (o.reranked_gold_rank or o.retrieved_gold_rank)) is not None
            and final_rank <= 10
        ) / 18
        loss_counts = Counter(str(o.loss_stage) for o in evidence if not o.gold_evidence_included)
        if (metric.evidence_inclusion_rate.numerator, metric.evidence_inclusion_rate.denominator) != (included, 18):
            raise Figure03ExportError(f'Evidence inclusion aggregation mismatch: {pipeline_id}')
        if (metric.retrieval_recall_at_k.numerator, metric.retrieval_recall_at_k.denominator) != (recalled, 18):
            raise Figure03ExportError(f'Recall@5 aggregation mismatch: {pipeline_id}')
        if not math.isclose(float(metric.mrr_at_10), mrr, rel_tol=0, abs_tol=1e-12):
            raise Figure03ExportError(f'MRR@10 aggregation mismatch: {pipeline_id}')
        if loss_counts != Counter({str(stage): count for stage, count in metric.loss_stage_counts.items()}):
            raise Figure03ExportError(f'Loss-stage count mismatch: {pipeline_id}')
        rows.append({
            'id': pipeline_id,
            'inclusion': {'numerator': included, 'denominator': 18, 'value': metric.evidence_inclusion_rate.value},
            'recallAt5': {'numerator': recalled, 'denominator': 18, 'value': metric.retrieval_recall_at_k.value},
            'mrrAt10': float(metric.mrr_at_10),
        })
    if [x['inclusion']['numerator'] for x in rows] != [13, 16, 18, 18]:
        raise Figure03ExportError('Frozen evidence inclusion progression changed')
    if [x['recallAt5']['numerator'] for x in rows] != [14, 17, 18, 18]:
        raise Figure03ExportError('Frozen Recall@5 progression changed')
    return {'artifactId': artifact.artifact_id, 'runId': report.run_id, 'retrievalMetricK': 5, 'rows': rows}


def render_js(projection: dict[str, Any]) -> str:
    encoded = json.dumps(projection, ensure_ascii=False, indent=2, allow_nan=False)
    encoded = encoded.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return ('/* AUTO-GENERATED — reviewed synthetic metric summary for Figure 03.\n'
            ' * Regenerate: python scripts/export_research_publication_figure03.py --write\n'
            ' * Figures 01–02 remain on the separate fixed-case projection. */\n'
            f'const FIGURE03_METRICS = Object.freeze({encoded});\n')


def build_from_repo(root: Path) -> str:
    from rag_lab.comparison_artifacts import load_baseline_artifact
    return render_js(build_metric_projection(load_baseline_artifact(root / ARTIFACT)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--write', action='store_true')
    modes.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = build_from_repo(ROOT)
    target = ROOT / OUTPUT
    if args.check:
        if not target.is_file() or target.read_text(encoding='utf-8') != result:
            print('FAIL: Figure 03 metric projection is missing or stale; regenerate with --write')
            return 1
        print('PASS: Figure 03 projection matches reviewed synthetic benchmark')
        return 0
    target.write_text(result, encoding='utf-8', newline='\n')
    print(f'WROTE: {target.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
