"""Cross-encoder reranking with typed before/after rank evidence."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol

from rag_lab.schemas import (
    RerankedChunk,
    RerankerMethod,
    RerankingTrace,
    RetrievalTrace,
)


class RerankerInputError(ValueError):
    """Raised when a reranking request cannot produce an auditable ranking."""


class RerankerRuntimeError(RuntimeError):
    """Raised when a configured cross-encoder runtime cannot load or score candidates."""


class PairScoringModel(Protocol):
    """Minimal query-document scoring boundary used by the reranker.

    The implementation owns model loading and pair scoring. The reranker owns candidate ordering,
    deterministic tie-breaking, trace construction, and validation of model output.
    """

    @property
    def name(self) -> str:
        """Return the stable model identifier recorded in reranking traces."""

    def score(self, *, query: str, documents: Sequence[str]) -> list[float]:
        """Return one finite relevance score per supplied document, in input order."""


class SentenceTransformersCrossEncoderModel:
    """Lazy CPU-first adapter for an explicitly selected Sentence Transformers CrossEncoder.

    The adapter intentionally has no fallback model. A reranking trace must identify the exact
    model that scored the candidate pairs. Unit tests use fixture scoring models and never require
    a model download.
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        if not model_name.strip():
            raise RerankerInputError("model_name must contain non-whitespace text")
        if not device.strip():
            raise RerankerInputError("device must contain non-whitespace text")
        if batch_size < 1:
            raise RerankerInputError("batch_size must be at least 1")

        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as error:
            raise RerankerRuntimeError(
                "sentence-transformers is not installed; run: "
                'python -m pip install -e ".[dev,dense]"'
            ) from error

        try:
            self._model = CrossEncoder(model_name, device=device)
        except (OSError, RuntimeError, ValueError) as error:
            raise RerankerRuntimeError(
                "could not load the configured cross-encoder model; verify the model identifier, "
                "network access for the initial download, or local Hugging Face cache"
            ) from error

        self._model_name = model_name
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return f"sentence-transformers-cross-encoder:{self._model_name}"

    def score(self, *, query: str, documents: Sequence[str]) -> list[float]:
        if not query.strip():
            raise RerankerInputError("query must contain non-whitespace text")
        document_list = list(documents)
        if not document_list:
            raise RerankerInputError("reranking requires at least one candidate document")
        if any(not document.strip() for document in document_list):
            raise RerankerInputError("reranking candidate documents must not contain blank text")

        pairs = [(query, document) for document in document_list]
        try:
            raw_scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RerankerRuntimeError("cross-encoder failed to score the supplied candidate pairs") from error

        try:
            scores = [float(score) for score in raw_scores]
        except (TypeError, ValueError) as error:
            raise RerankerRuntimeError(
                "cross-encoder returned scores that could not be converted to scalar floats"
            ) from error

        _validate_scores(scores, expected_count=len(document_list))
        return scores


class CrossEncoderReranker:
    """Rescore an existing first-stage candidate trace with an auditable cross-encoder seam.

    Reranking cannot recover evidence that the first-stage trace never retrieved. It only changes
    the order of the fixed candidate set. The resulting trace therefore records both ranks for
    every candidate and explicitly distinguishes a retrieval miss from a ranking repair.
    """

    method = RerankerMethod.CROSS_ENCODER

    def __init__(self, *, scoring_model: PairScoringModel) -> None:
        self._scoring_model = scoring_model
        if not self._scoring_model.name.strip():
            raise RerankerInputError("scoring_model.name must contain non-whitespace text")

    def rerank(self, *, first_stage_trace: RetrievalTrace) -> RerankingTrace:
        """Return the full candidate set ordered by cross-encoder score.

        ``first_stage_trace`` must be a real typed retrieval trace. Its result list fixes the
        candidate universe; the reranker cannot silently add chunks or discard candidates.
        """

        candidates = first_stage_trace.results
        if not candidates:
            raise RerankerInputError("reranking requires a first-stage trace with candidates")

        scores = self._scoring_model.score(
            query=first_stage_trace.query,
            documents=[candidate.chunk.text for candidate in candidates],
        )
        _validate_scores(scores, expected_count=len(candidates))

        scored_candidates = list(zip(candidates, scores, strict=True))
        ordered_candidates = sorted(
            scored_candidates,
            key=lambda item: (-item[1], item[0].rank, item[0].chunk.chunk_id),
        )

        results = [
            RerankedChunk(
                chunk=candidate.chunk,
                rank=rerank_rank,
                first_stage_rank=candidate.rank,
                first_stage_score=candidate.score,
                reranker_score=score,
                gold_evidence_match=candidate.gold_evidence_match,
            )
            for rerank_rank, (candidate, score) in enumerate(ordered_candidates, start=1)
        ]

        gold_after_rerank = next(
            (result.rank for result in results if result.gold_evidence_match),
            None,
        )

        return RerankingTrace(
            case_id=first_stage_trace.case_id,
            first_stage_retriever_name=first_stage_trace.retriever_name,
            first_stage_trace=first_stage_trace,
            reranker_name=self.method,
            reranker_model_name=self._scoring_model.name,
            candidate_count=len(candidates),
            results=results,
            gold_evidence_found=first_stage_trace.gold_evidence_found,
            gold_evidence_rank_before_rerank=first_stage_trace.gold_evidence_rank,
            gold_evidence_rank_after_rerank=gold_after_rerank,
        )


def _validate_scores(scores: Sequence[float], *, expected_count: int) -> None:
    if len(scores) != expected_count:
        raise RerankerInputError(
            "scoring model returned a score count that did not match the candidate count"
        )
    if any(not math.isfinite(score) for score in scores):
        raise RerankerInputError("reranker scores must be finite")
