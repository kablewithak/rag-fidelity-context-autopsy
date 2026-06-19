from rag_lab.failure_taxonomy import FAILURE_TAXONOMY, get_failure_taxonomy_entry
from rag_lab.schemas import FailureLabel


def test_failure_taxonomy_covers_every_standardized_failure_label() -> None:
    assert set(FAILURE_TAXONOMY) == set(FailureLabel)


def test_every_taxonomy_entry_has_actionable_repair_guidance() -> None:
    for label in FailureLabel:
        entry = get_failure_taxonomy_entry(label)

        assert entry.label is label
        assert len(entry.definition) >= 20
        assert len(entry.repair_recommendation) >= 20
