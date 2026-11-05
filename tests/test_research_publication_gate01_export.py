"""Figure 01 public projection: deterministic parity and fail-closed coverage."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rag_lab.comparison_artifacts import load_baseline_artifact
from rag_lab.eval_cases import load_evaluation_cases
from scripts.export_research_publication_gate01 import (
    ARTIFACT,
    CASES,
    LABELS,
    OUTPUT,
    PublicationExportError,
    build_from_repo,
    build_projection,
    render_js,
)

ROOT = Path(__file__).resolve().parents[1]


def _sources():
    cases = load_evaluation_cases(ROOT / CASES)
    artifact = load_baseline_artifact(ROOT / ARTIFACT)
    labels = json.loads((ROOT / LABELS).read_text(encoding='utf-8'))
    return cases, artifact, labels


def test_figure01_export_matches_committed_file_byte_for_byte() -> None:
    generated = build_from_repo(ROOT)
    assert (ROOT / OUTPUT).read_text(encoding='utf-8') == generated, (
        'Run python scripts/export_research_publication_gate01.py --write and review the diff'
    )
    assert generated == build_from_repo(ROOT)
    assert generated.startswith('/* AUTO-GENERATED')


def test_figure01_projection_limits_output_to_approved_public_fields() -> None:
    from scripts.export_research_publication_gate01 import _sha256

    cases, artifact, labels = _sources()
    projection = build_projection(
        cases=cases, artifact=artifact, display_labels=labels,
        artifact_sha256=_sha256(ROOT / ARTIFACT),
        cases_sha256=_sha256(ROOT / CASES),
    )
    assert set(projection) == {'pipelines', 'cases', 'provenance'}
    assert len(projection['cases']) == 18
    assert all(set(row) == {'id', 'label', 'group', 'query', 'states'} for row in projection['cases'])
    assert all(len(row['states']) == 4 for row in projection['cases'])
    assert [sum(row['states'][i][0] for row in projection['cases']) for i in range(4)] == [13, 16, 18, 18]

    rows_by_id = {row['id']: row for row in projection['cases']}
    assert rows_by_id['legal_confidentiality_003']['states'] == [
        [0, None, None, 'chunking'],
        [0, 5, None, 'ranking'],
        [1, 3, None, None],
        [1, 3, 1, None],
    ]
    text = render_js(projection)
    assert 'const CASES = Object.freeze(' in text
    assert not any(case.gold_evidence_text in text for case in cases)
    assert not any(case.gold_answer in text for case in cases)
    assert not any(case.diagnostic_note in text for case in cases)
    assert 'trace_sha256' not in text
    assert 'source_doc_id' not in text
    assert 'generated_answer' not in text
    assert re.search(r'"artifactSha256": "[a-f0-9]{64}"', text)


def test_figure01_export_rejects_missing_labels() -> None:
    cases, artifact, labels = _sources()
    labels.pop(cases[0].case_id)
    with pytest.raises(PublicationExportError, match='exact case-ID coverage'):
        build_projection(
            cases=cases, artifact=artifact, display_labels=labels,
            artifact_sha256='a' * 64, cases_sha256='b' * 64,
        )


def test_figure01_export_rejects_unreviewed_artifact_identity() -> None:
    cases, artifact, labels = _sources()
    changed = artifact.model_copy(update={'artifact_id': 'unreviewed_variant'})
    with pytest.raises(PublicationExportError, match='artifact ID mismatch'):
        build_projection(
            cases=cases, artifact=changed, display_labels=labels,
            artifact_sha256='a' * 64, cases_sha256='b' * 64,
        )


def test_figure01_export_rejects_incomplete_coverage() -> None:
    cases, artifact, labels = _sources()
    changed_report = artifact.report.model_copy(
        update={'case_outcomes': artifact.report.case_outcomes[:-1]}
    )
    changed = artifact.model_copy(update={'report': changed_report})
    with pytest.raises(PublicationExportError, match='Missing case-pipeline outcome'):
        build_projection(
            cases=cases, artifact=changed, display_labels=labels,
            artifact_sha256='a' * 64, cases_sha256='b' * 64,
        )
