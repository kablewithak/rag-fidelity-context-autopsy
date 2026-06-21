from __future__ import annotations

from pathlib import Path

from rag_lab.comparison import PipelineId
from rag_lab.public_transfer_explorer import load_public_transfer_review_view


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_committed_public_transfer_review_loads_as_a_separate_view() -> None:
    view = load_public_transfer_review_view(project_root=PROJECT_ROOT)

    assert view.comparison_mode == "side_by_side_not_pooled"
    assert view.synthetic_artifact_id == "four_pipeline_baseline_v1"
    assert view.synthetic_case_count == 18
    assert view.public_artifact_id == "public_transfer_squad_v1_dev_v1_reviewed_v1"
    assert view.public_dataset_id == "squad_v1.1_dev"
    assert view.public_source_document_count == 10
    assert view.public_evaluation_case_count == 30
    assert len(view.pipeline_scorecards) == 4


def test_public_transfer_view_preserves_measured_non_uniformity() -> None:
    view = load_public_transfer_review_view(project_root=PROJECT_ROOT)
    metrics = {card.pipeline_id: card for card in view.pipeline_scorecards}

    character = metrics[PipelineId.CHAR_DENSE_NAIVE]
    full = metrics[PipelineId.TOKEN_HYBRID_RERANK_BUDGETED]

    assert character.retrieval_recall_at_k.numerator == 30
    assert full.retrieval_recall_at_k.numerator == 29
    assert full.mrr_at_10 > character.mrr_at_10
    assert full.evidence_inclusion_rate.numerator > character.evidence_inclusion_rate.numerator
    assert view.full_pipeline_recall_delta_pp < 0
    assert view.sentence_aware_dense_recall_delta_pp < 0
    assert view.evidence_inclusion_delta_pp > 0
    assert view.mrr_delta > 0
