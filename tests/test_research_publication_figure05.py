"""Gate 06: public-transfer figure provenance, strict parity and UI integration."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from rag_lab.public_transfer_artifacts import load_public_transfer_artifact
from rag_lab.public_transfer_explorer import load_public_transfer_review_view
from rag_lab.public_transfer_review import DEFAULT_REVIEWED_PUBLIC_TRANSFER_ARTIFACT_PATH
from scripts.export_research_publication_figure05 import (
    OUTPUT,
    Figure05ExportError,
    build_from_repo,
    build_projection,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'design' / 'research_publication_gate01'


def _reviewed():
    return (
        load_public_transfer_review_view(project_root=ROOT),
        load_public_transfer_artifact(ROOT / DEFAULT_REVIEWED_PUBLIC_TRANSFER_ARTIFACT_PATH),
    )


def test_figure05_generated_asset_exactly_matches_reviewed_public_transfer() -> None:
    generated = build_from_repo(ROOT)
    assert (ROOT / OUTPUT).read_text(encoding='utf-8') == generated, (
        'Regenerate with python -m scripts.export_research_publication_figure05 --write'
    )
    view, artifact = _reviewed()
    projection = build_projection(view=view, artifact=artifact)
    assert projection['scope'] == 'separate_reviewed_public_transfer_not_pooled'
    assert projection['publicCaseCount'] == 30
    assert projection['syntheticCaseCount'] == 18
    assert [r['inclusion']['numerator'] for r in projection['rows']] == [26, 27, 28, 29]
    assert [r['recallAt5']['numerator'] for r in projection['rows']] == [30, 29, 29, 29]
    assert projection['rows'][3]['lossStages']['retrieval'] == 1
    assert projection['rows'][3]['mrrAt10'] == pytest.approx(0.9333333333333333)
    for forbidden in ('gold_answer', 'gold_evidence_text', 'trace_references', 'question_sha256', 'source_doc_id'):
        assert forbidden not in generated


def test_figure05_rejects_corrupted_metric() -> None:
    view, artifact = _reviewed()
    metrics = list(artifact.report.pipeline_metrics)
    metrics[0] = metrics[0].model_copy(update={'mrr_at_10': metrics[0].mrr_at_10 - 0.05})
    corrupted = artifact.model_copy(update={
        'report': artifact.report.model_copy(update={'pipeline_metrics': metrics})
    })
    with pytest.raises(Figure05ExportError, match='Aggregate metric disagreement'):
        build_projection(view=view, artifact=corrupted)


def test_figure05_rejects_changed_fixture_or_pooled_scope() -> None:
    view, artifact = _reviewed()
    changed_view = view.model_copy(update={'synthetic_case_count': 30})
    with pytest.raises(Figure05ExportError, match='separation boundary changed'):
        build_projection(view=changed_view, artifact=artifact)
    changed_view = view.model_copy(update={'comparison_mode': 'pooled'})
    with pytest.raises(Figure05ExportError, match='separation boundary changed'):
        build_projection(view=changed_view, artifact=artifact)


def test_figure05_rejects_corrupted_loss_stages() -> None:
    view, artifact = _reviewed()
    metrics = list(artifact.report.pipeline_metrics)
    metrics[3] = metrics[3].model_copy(update={'loss_stage_counts': {'ranking': 1}})
    corrupted = artifact.model_copy(update={
        'report': artifact.report.model_copy(update={'pipeline_metrics': metrics})
    })
    with pytest.raises(Figure05ExportError, match='Loss-stage aggregation mismatch'):
        build_projection(view=view, artifact=corrupted)


def test_figure05_page_and_deployment_contract() -> None:
    html = (PAGE / 'index.html').read_text(encoding='utf-8')
    css = (PAGE / 'styles.css').read_text(encoding='utf-8')
    js = (PAGE / 'app.js').read_text(encoding='utf-8')
    workflow = (ROOT / '.github/workflows/deploy-huggingface-space.yml').read_text(encoding='utf-8')
    assert html.index('src="figure05_data.js"') < html.index('src="app.js"')
    assert 'id="figure-five"' in html
    assert 'id="transfer-pipeline-rows"' in html
    assert 'id="transfer-comparisons"' in html
    assert 'id="transfer-inspector-copy"' in html
    assert 'assertFigure05Public();' in js
    assert 'renderTransfer();' in js
    assert '.transfer-section' in css
    assert 'position:sticky' in css
    assert 'prefers-reduced-motion:reduce' in css
    assert 'python -m scripts.export_research_publication_figure05 --check' in workflow
    assert 'tests/test_research_publication_figure05.py' in workflow
    assert 'git push --porcelain hf HEAD:main' in workflow
    assert '--force' not in workflow
    assert 'streamlit run app/streamlit_app.py' not in html  # no premature runtime claim


def test_figure05_node_validation_and_reading_state() -> None:
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is optional; browser smoke remains required')
    program = r'''
const fs=require('fs'),vm=require('vm');
const ctx=vm.createContext({document:{readyState:'loading',addEventListener:()=>{}},URL});
for(const f of process.argv.slice(1))vm.runInContext(fs.readFileSync(f,'utf8'),ctx);
const ok=vm.runInContext(`(() => {
  assertEvidenceProjection();assertFigure03Metrics();assertFigure04Proof();assertFigure05Public();
  if(FIGURE05_PUBLIC.rows.length!==4)return false;
  if(transferFormat(FIGURE05_PUBLIC.rows[0],'inclusion')!=='26/30')return false;
  if(transferFormat(FIGURE05_PUBLIC.rows[3],'inclusion')!=='29/30')return false;
  if(transferFormat(FIGURE05_PUBLIC.rows[0],'recallAt5')!=='30/30')return false;
  if(transferFormat(FIGURE05_PUBLIC.rows[3],'recallAt5')!=='29/30')return false;
  if(transferDelta(FIGURE05_PUBLIC.rows[0],FIGURE05_PUBLIC.rows[3],'recallAt5')!=='-3.3 pp')return false;
  return true;
})()`,ctx);
if(!ok)process.exit(1);
'''
    files = [PAGE / x for x in ('data.js', 'figure03_data.js', 'figure04_data.js', 'figure05_data.js', 'app.js')]
    result = subprocess.run([node, '-e', program, *map(str, files)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
