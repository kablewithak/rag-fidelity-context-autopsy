from __future__ import annotations

import pytest

from rag_lab.embedders import EmbeddingInputError, validate_embedding_vectors


def test_validate_embedding_vectors_accepts_finite_non_zero_fixed_dimension_vectors() -> None:
    vectors = validate_embedding_vectors(
        [[1.0, 0.0], [0.5, -0.5]],
        expected_count=2,
        expected_dimension=2,
    )

    assert vectors == [(1.0, 0.0), (0.5, -0.5)]


def test_validate_embedding_vectors_rejects_dimension_mismatch() -> None:
    with pytest.raises(EmbeddingInputError, match="dimension did not match"):
        validate_embedding_vectors(
            [[1.0]],
            expected_count=1,
            expected_dimension=2,
        )


def test_validate_embedding_vectors_rejects_zero_and_non_finite_vectors() -> None:
    with pytest.raises(EmbeddingInputError, match="all zero"):
        validate_embedding_vectors(
            [[0.0, 0.0]],
            expected_count=1,
            expected_dimension=2,
        )

    with pytest.raises(EmbeddingInputError, match="finite"):
        validate_embedding_vectors(
            [[float("nan"), 1.0]],
            expected_count=1,
            expected_dimension=2,
        )
