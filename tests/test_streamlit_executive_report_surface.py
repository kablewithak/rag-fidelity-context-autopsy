from __future__ import annotations

import ast
from pathlib import Path

from rag_lab.executive_evaluation_report import load_executive_evaluation_report
from rag_lab.schemas import EvidenceLossStage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"


def test_streamlit_app_compiles_and_wires_executive_report_route() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    ast.parse(source, filename=str(APP_PATH))

    assert "load_executive_evaluation_report" in source
    assert 'options=("Executive report", "Failure case", "Chunking", "Retrieval", "Context autopsy")' in source
    assert "def _render_executive_report(" in source
    assert 'with st.expander("Guided CTO demo route", expanded=True):' in source


def test_executive_route_preserves_reviewed_pipeline_story() -> None:
    report = load_executive_evaluation_report(project_root=PROJECT_ROOT)
    baseline = report.baseline_scorecard
    strongest = report.strongest_scorecard

    assert baseline.evidence_inclusion_rate.numerator == 13
    assert baseline.evidence_inclusion_rate.denominator == 18
    assert strongest.evidence_inclusion_rate.numerator == 18
    assert strongest.evidence_inclusion_rate.denominator == 18
    assert strongest.mrr_at_10 > baseline.mrr_at_10

    assert [item.loss_stage for item in report.baseline_failure_stages] == [
        EvidenceLossStage.CHUNKING,
        EvidenceLossStage.RETRIEVAL,
        EvidenceLossStage.RANKING,
    ]
    assert len(report.repair_sequence) == 3


def test_executive_route_keeps_controlled_context_finding_separate() -> None:
    report = load_executive_evaluation_report(project_root=PROJECT_ROOT)
    finding = report.controlled_context_finding

    assert finding.verbose_gold_evidence_dropped is True
    assert finding.compact_gold_evidence_included is True

    source = APP_PATH.read_text(encoding="utf-8")
    assert "not counted as a standard four-pipeline benchmark regression" in source
