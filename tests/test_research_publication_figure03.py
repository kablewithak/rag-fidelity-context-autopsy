"""Figure 03 aggregate-only export, parity, and editorial UI contract."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rag_lab.comparison_artifacts import load_baseline_artifact
from scripts.export_research_publication_figure03 import (
    OUTPUT,
    Figure03ExportError,
    build_from_repo,
    build_metric_projection,
)
from scripts.export_research_publication_gate01 import ARTIFACT

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'design' / 'research_publication_gate01'


def test_figure03_generated_projection_exactly_matches_artifact() -> None:
    generated = build_from_repo(ROOT)
    assert (ROOT / OUTPUT).read_text(encoding='utf-8') == generated, (
        'Regenerate with python scripts/export_research_publication_figure03.py --write'
    )
    artifact = load_baseline_artifact(ROOT / ARTIFACT)
    projection = build_metric_projection(artifact)
    assert set(projection) == {'artifactId', 'runId', 'retrievalMetricK', 'rows'}
    assert [row['inclusion']['numerator'] for row in projection['rows']] == [13, 16, 18, 18]
    assert [row['recallAt5']['numerator'] for row in projection['rows']] == [14, 17, 18, 18]
    assert projection['rows'][0]['mrrAt10'] == pytest.approx(0.7361111111111112)
    assert projection['rows'][3]['mrrAt10'] == pytest.approx(0.9722222222222222)
    assert 'gold_answer' not in generated
    assert 'gold_evidence_text' not in generated
    assert 'trace_references' not in generated


def test_figure03_rejects_metric_corruption() -> None:
    artifact = load_baseline_artifact(ROOT / ARTIFACT)
    metrics = list(artifact.report.pipeline_metrics)
    metrics[0] = metrics[0].model_copy(update={'mrr_at_10': metrics[0].mrr_at_10 - 0.01})
    report = artifact.report.model_copy(update={'pipeline_metrics': metrics})
    changed = artifact.model_copy(update={'report': report})
    with pytest.raises(Figure03ExportError, match='MRR@10 aggregation mismatch'):
        build_metric_projection(changed)


def test_figure03_static_markup_and_deployment_gate() -> None:
    html = (PAGE / 'index.html').read_text(encoding='utf-8')
    script = (PAGE / 'app.js').read_text(encoding='utf-8')
    workflow = (ROOT / '.github/workflows/deploy-huggingface-space.yml').read_text(encoding='utf-8')
    css = (PAGE / 'styles.css').read_text(encoding='utf-8')
    for ref in ('#figure-one', '#figure-two', '#figure-three', '#method'):
        assert ref in html
    assert '<script defer src="figure03_data.js"></script>' in html
    assert 'id="comparison-plots"' in html
    assert 'id="comparison-step-list"' in html
    assert 'id="comparison-active-copy"' in html
    assert "renderComparisonSteps();" in script
    assert 'assertFigure03Metrics();' in script
    assert "button.setAttribute('aria-pressed'" in script
    assert 'position:sticky' in css
    assert 'prefers-reduced-motion:reduce' in css
    assert 'python -m scripts.export_research_publication_figure03 --check' in workflow
    assert 'tests/test_research_publication_figure03.py' in workflow
    assert 'git push --porcelain hf HEAD:main' in workflow
    assert '--force' not in workflow


def test_figure03_node_runtime_consistency() -> None:
    node = shutil.which('node')
    if not node:
        pytest.skip('Node optional; validated independently in supported browser')
    program = r'''
const fs=require('fs'),vm=require('vm');
const ctx=vm.createContext({document:{readyState:'loading',addEventListener:()=>{}},URL});
for(const file of process.argv.slice(1))vm.runInContext(fs.readFileSync(file,'utf8'),ctx);
const verdict=vm.runInContext(`(() => {
  assertEvidenceProjection();assertFigure03Metrics();
  if(METRIC_SERIES.length!==3||FIGURE03_METRICS.rows.length!==4)return false;
  if(metricLabel(FIGURE03_METRICS.rows[0],'inclusion')!=='72.2%')return false;
  if(countLabel(FIGURE03_METRICS.rows[0],'recallAt5')!=='14/18')return false;
  if(metricLabel(FIGURE03_METRICS.rows[3],'mrrAt10')!=='0.972')return false;
  return true;
})()`,ctx);
if(!verdict)process.exit(1);
'''
    files = [PAGE / name for name in ('data.js', 'figure03_data.js', 'app.js')]
    result = subprocess.run([node, '-e', program, *map(str, files)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
