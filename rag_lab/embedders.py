"""Embedding adapters for dense retrieval with explicit runtime and test seams."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol


class EmbeddingInputError(ValueError):
    """Raised when an embedding request cannot form a trustworthy dense vector."""


class EmbeddingRuntimeError(RuntimeError):
    """Raised when the configured runtime embedding provider cannot be loaded or called."""


class EmbeddingModel(Protocol):
    """Minimal model boundary used by dense retrieval.

    Implementations must return one finite vector per input text with a stable fixed dimension.
    The retriever owns cosine scoring; an embedder owns only text-to-vector conversion.
    """

    @property
    def name(self) -> str:
        """Return a stable model identifier suitable for a retrieval trace."""

    @property
    def dimension(self) -> int:
        """Return the fixed embedding-vector dimension."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts into one dense vector per text in input order."""


class SentenceTransformerEmbeddingModel:
    """Lazy CPU-first adapter for an explicitly selected Sentence Transformers model.

    The dependency and model weights are intentionally optional during unit testing. Install the
    ``dense`` extra before selecting this adapter in a local runtime or demo. The adapter does
    not silently fall back to another model because a trace must identify the actual embedding
    model that produced its ranking.
    """

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        if not model_name.strip():
            raise EmbeddingInputError("model_name must contain non-whitespace text")
        if not device.strip():
            raise EmbeddingInputError("device must contain non-whitespace text")
        if batch_size < 1:
            raise EmbeddingInputError("batch_size must be at least 1")

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as error:
            raise EmbeddingRuntimeError(
                "sentence-transformers is not installed; run: "
                'python -m pip install -e ".[dev,dense]"'
            ) from error

        try:
            self._model = SentenceTransformer(model_name, device=device)
        except (OSError, RuntimeError, ValueError) as error:
            raise EmbeddingRuntimeError(
                "could not load the configured sentence-transformers model; verify the model "
                "identifier, network access for the initial download, or local Hugging Face cache"
            ) from error

        dimension = self._model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension < 1:
            raise EmbeddingRuntimeError(
                "configured sentence-transformers model did not expose a valid embedding dimension"
            )

        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            raise EmbeddingInputError("embedding requests must contain at least one text")
        if any(not text.strip() for text in text_list):
            raise EmbeddingInputError("embedding requests must not contain blank text")

        try:
            encoded = self._model.encode(
                text_list,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise EmbeddingRuntimeError("sentence-transformers failed to encode the supplied text") from error

        vectors = [[float(value) for value in row] for row in encoded]
        if len(vectors) != len(text_list):
            raise EmbeddingRuntimeError(
                "sentence-transformers returned a vector count that did not match the input text count"
            )

        for vector in vectors:
            _validate_vector(vector, expected_dimension=self._dimension)
        return vectors


def validate_embedding_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    expected_dimension: int,
) -> list[tuple[float, ...]]:
    """Validate model output before a retriever stores or scores a dense vector."""

    if expected_count < 1:
        raise EmbeddingInputError("expected_count must be at least 1")
    if expected_dimension < 1:
        raise EmbeddingInputError("expected_dimension must be at least 1")
    if len(vectors) != expected_count:
        raise EmbeddingInputError(
            "embedding model returned a vector count that did not match the input text count"
        )

    return [
        _validate_vector(vector, expected_dimension=expected_dimension)
        for vector in vectors
    ]


def _validate_vector(
    vector: Sequence[float],
    *,
    expected_dimension: int,
) -> tuple[float, ...]:
    if len(vector) != expected_dimension:
        raise EmbeddingInputError(
            "embedding vector dimension did not match the configured embedding model"
        )

    normalized = tuple(float(value) for value in vector)
    if any(not math.isfinite(value) for value in normalized):
        raise EmbeddingInputError("embedding vectors must contain only finite numeric values")
    if not any(value != 0.0 for value in normalized):
        raise EmbeddingInputError("embedding vectors must not be all zero")
    return normalized
