"""Real execution runner for the fixed four-pipeline RAG comparison harness.

The runner uses existing typed chunkers, retrievers, reranking, and context assembly
components to derive one :class:`CasePipelineOutcome` per fixed evaluation case and
pipeline. It keeps complete raw traces in memory only for local inspection while the
comparison artifact retains only bounded trace identifiers and SHA-256 fingerprints.

The runner does not generate answers. Its evidence-inclusion metric means that complete
known gold evidence reached the pipeline's final selected context boundary, not that an
LLM answered correctly or grounded every claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_lab.chunkers import CharacterChunker, SentenceAwareTokenChunker
from rag_lab.comparison import (
    DEFAULT_PIPELINE_DEFINITIONS,
    DEFAULT_RETRIEVAL_METRIC_K,
    CasePipelineOutcome,
    ContextSelectionMode,
    PipelineComparisonReport,
    PipelineDefinition,
    PipelineId,
    TraceReference,
    build_comparison_report,
)
from rag_lab.context_assembly import (
    ContextAssembler,
    ContextAssemblyConfig,
    ContextAutopsyReport,
    ContextRenderConfig,
    ContextRenderProfile,
    LostEvidenceReport,
    build_lost_evidence_report,
)
from rag_lab.corpus_loader import chunk_corpus
from rag_lab.embedders import EmbeddingModel
from rag_lab.rerankers import CrossEncoderReranker, PairScoringModel
from rag_lab.retrievers import DenseRetriever, HybridRetriever
from rag_lab.schemas import (
    ChunkBoundaryQuality,
    CorpusDocument,
    EvaluationCase,
    EvidenceLossStage,
    FailureLabel,
    RerankingTrace,
    RetrievalMethod,
    RetrievalTrace,
    TextChunk,
)
from rag_lab.tokenizers import TokenCounter


class ComparisonExecutionError(ValueError):
    """Raised when the runner cannot make an auditable pipeline outcome claim."""


class ComparisonExecutionConfig(BaseModel):
    """Frozen execution settings shared by the four fixed ablation pipelines.

    These settings are intentionally explicit rather than inferred from the corpus. A
    comparison only has causal value when each pipeline receives the same source corpus,
    evaluation cases, token counter, dense model, and fixed candidate depth.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    character_max_characters: int = Field(default=700, ge=1, le=20_000)
    character_overlap_characters: int = Field(default=0, ge=0, le=19_999)
    sentence_aware_max_tokens: int = Field(default=96, ge=1, le=20_000)
    sentence_aware_overlap_tokens: int = Field(default=0, ge=0, le=19_999)
    hybrid_rrf_k: int = Field(default=60, ge=1, le=10_000)
    retrieval_metric_k: int = Field(default=DEFAULT_RETRIEVAL_METRIC_K, ge=1, le=100)
    budgeted_render_profile: ContextRenderProfile = ContextRenderProfile.COMPACT_CITATION

    @model_validator(mode="after")
    def validate_overlap_bounds(self) -> "ComparisonExecutionConfig":
        if self.character_overlap_characters >= self.character_max_characters:
            raise ValueError(
                "character_overlap_characters must be smaller than character_max_characters"
            )
        if self.sentence_aware_overlap_tokens >= self.sentence_aware_max_tokens:
            raise ValueError(
                "sentence_aware_overlap_tokens must be smaller than sentence_aware_max_tokens"
            )
        return self


@dataclass(frozen=True, slots=True)
class PipelineCaseExecution:
    """Ephemeral local evidence for one pipeline/case run.

    These rich typed traces can contain synthetic chunk text and remain in process memory
    for local inspection. They are deliberately excluded from serialized comparison reports.
    """

    outcome: CasePipelineOutcome
    retrieval_trace: RetrievalTrace
    reranking_trace: RerankingTrace | None
    context_autopsy_report: ContextAutopsyReport | None
    lost_evidence_report: LostEvidenceReport | None


@dataclass(frozen=True, slots=True)
class ComparisonExecutionResult:
    """A validated report plus ephemeral typed trace objects from one local run."""

    report: PipelineComparisonReport
    executions: tuple[PipelineCaseExecution, ...]


@dataclass(frozen=True, slots=True)
class _EvidenceAvailability:
    """Whether a complete known evidence string survived the chunking boundary."""

    complete_chunk_exists: bool
    evidence_split: bool
    has_character_boundary_cut: bool


class FourPipelineComparisonRunner:
    """Execute the lab's fixed four ablations from real local component traces.

    The runner invokes actual dense or hybrid retrieval for every case. The budgeted
    intervention additionally invokes the configured cross-encoder and the measured context
    assembler. Naive pipelines intentionally select the first ``final_evidence_chunk_limit``
    first-stage results without token-budget accounting; this is a baseline behavior, not a
    claim about final model capacity.
    """

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        embedding_model: EmbeddingModel,
        reranker_scoring_model: PairScoringModel,
        pipeline_definitions: Iterable[PipelineDefinition] = DEFAULT_PIPELINE_DEFINITIONS,
        config: ComparisonExecutionConfig | None = None,
    ) -> None:
        self._token_counter = token_counter
        self._embedding_model = embedding_model
        self._reranker_scoring_model = reranker_scoring_model
        self._pipeline_definitions = tuple(pipeline_definitions)
        self._config = config or ComparisonExecutionConfig()

        if not self._token_counter.name.strip():
            raise ComparisonExecutionError("token_counter.name must contain non-whitespace text")
        if self._embedding_model.dimension < 1:
            raise ComparisonExecutionError("embedding_model.dimension must be at least 1")
        if not self._embedding_model.name.strip():
            raise ComparisonExecutionError("embedding_model.name must contain non-whitespace text")
        if not self._reranker_scoring_model.name.strip():
            raise ComparisonExecutionError(
                "reranker_scoring_model.name must contain non-whitespace text"
            )
        _validate_fixed_pipeline_definitions(self._pipeline_definitions)
        _validate_retrieval_metric_cutoff(
            definitions=self._pipeline_definitions,
            retrieval_metric_k=self._config.retrieval_metric_k,
        )

    def run(
        self,
        *,
        run_id: str,
        cases: Sequence[EvaluationCase],
        documents: Sequence[CorpusDocument],
        baseline_pipeline_id: PipelineId = PipelineId.CHAR_DENSE_NAIVE,
    ) -> ComparisonExecutionResult:
        """Run every fixed case through every fixed pipeline exactly once.

        ``run_id`` is supplied by the caller rather than generated implicitly so the caller can
        make a local execution reproducible and link later exported reports to a specific config.
        """

        ordered_cases = _validate_cases(cases)
        ordered_documents = _validate_documents(documents)
        document_by_id = {document.source_doc_id: document for document in ordered_documents}
        _validate_case_sources(cases=ordered_cases, document_by_id=document_by_id)

        chunks_by_strategy = self._build_chunks(documents=ordered_documents)
        retrievers = self._build_retrievers(chunks_by_strategy=chunks_by_strategy)
        reranker = CrossEncoderReranker(scoring_model=self._reranker_scoring_model)

        executions: list[PipelineCaseExecution] = []
        for definition in self._pipeline_definitions:
            pipeline_chunks = chunks_by_strategy[definition.pipeline_id]
            retriever = retrievers[definition.pipeline_id]
            for case in ordered_cases:
                retrieval_trace = retriever.retrieve(
                    case=case,
                    top_k=definition.retrieval_top_k,
                )
                availability = _inspect_evidence_availability(
                    case=case,
                    source_document=document_by_id[case.source_doc_id],
                    chunks=pipeline_chunks,
                )
                executions.append(
                    self._derive_execution(
                        definition=definition,
                        case=case,
                        availability=availability,
                        retrieval_trace=retrieval_trace,
                        reranker=reranker,
                    )
                )

        outcomes = [execution.outcome for execution in executions]
        report = build_comparison_report(
            run_id=run_id,
            baseline_pipeline_id=baseline_pipeline_id,
            pipeline_definitions=self._pipeline_definitions,
            case_outcomes=outcomes,
            retrieval_metric_k=self._config.retrieval_metric_k,
        )
        return ComparisonExecutionResult(report=report, executions=tuple(executions))

    def _build_chunks(
        self,
        *,
        documents: Sequence[CorpusDocument],
    ) -> dict[PipelineId, tuple[TextChunk, ...]]:
        char_chunks = tuple(
            chunk_corpus(
                documents,
                chunker=CharacterChunker(
                    token_counter=self._token_counter,
                    max_characters=self._config.character_max_characters,
                    overlap_characters=self._config.character_overlap_characters,
                ),
            )
        )
        token_chunks = tuple(
            chunk_corpus(
                documents,
                chunker=SentenceAwareTokenChunker(
                    token_counter=self._token_counter,
                    max_tokens=self._config.sentence_aware_max_tokens,
                    overlap_tokens=self._config.sentence_aware_overlap_tokens,
                ),
            )
        )
        if not char_chunks or not token_chunks:
            raise ComparisonExecutionError("comparison chunkers must produce non-empty corpora")
        return {
            PipelineId.CHAR_DENSE_NAIVE: char_chunks,
            PipelineId.TOKEN_DENSE_NAIVE: token_chunks,
            PipelineId.TOKEN_HYBRID_NAIVE: token_chunks,
            PipelineId.TOKEN_HYBRID_RERANK_BUDGETED: token_chunks,
        }

    def _build_retrievers(
        self,
        *,
        chunks_by_strategy: dict[PipelineId, tuple[TextChunk, ...]],
    ) -> dict[PipelineId, DenseRetriever | HybridRetriever]:
        return {
            PipelineId.CHAR_DENSE_NAIVE: DenseRetriever(
                chunks=chunks_by_strategy[PipelineId.CHAR_DENSE_NAIVE],
                embedding_model=self._embedding_model,
            ),
            PipelineId.TOKEN_DENSE_NAIVE: DenseRetriever(
                chunks=chunks_by_strategy[PipelineId.TOKEN_DENSE_NAIVE],
                embedding_model=self._embedding_model,
            ),
            PipelineId.TOKEN_HYBRID_NAIVE: HybridRetriever(
                chunks=chunks_by_strategy[PipelineId.TOKEN_HYBRID_NAIVE],
                embedding_model=self._embedding_model,
                rrf_k=self._config.hybrid_rrf_k,
            ),
            PipelineId.TOKEN_HYBRID_RERANK_BUDGETED: HybridRetriever(
                chunks=chunks_by_strategy[PipelineId.TOKEN_HYBRID_RERANK_BUDGETED],
                embedding_model=self._embedding_model,
                rrf_k=self._config.hybrid_rrf_k,
            ),
        }

    def _derive_execution(
        self,
        *,
        definition: PipelineDefinition,
        case: EvaluationCase,
        availability: _EvidenceAvailability,
        retrieval_trace: RetrievalTrace,
        reranker: CrossEncoderReranker,
    ) -> PipelineCaseExecution:
        references = [_trace_reference(stage="retrieval", definition=definition, case=case, trace=retrieval_trace)]

        if not availability.complete_chunk_exists:
            outcome = CasePipelineOutcome(
                pipeline_id=definition.pipeline_id,
                case_id=case.case_id,
                requested_top_k=definition.retrieval_top_k,
                retrieved_gold_rank=None,
                reranked_gold_rank=None,
                gold_evidence_included=False,
                loss_stage=EvidenceLossStage.CHUNKING,
                failure_labels=_chunking_failure_labels(availability=availability),
                trace_references=references,
            )
            return PipelineCaseExecution(
                outcome=outcome,
                retrieval_trace=retrieval_trace,
                reranking_trace=None,
                context_autopsy_report=None,
                lost_evidence_report=None,
            )

        if not retrieval_trace.gold_evidence_found:
            outcome = CasePipelineOutcome(
                pipeline_id=definition.pipeline_id,
                case_id=case.case_id,
                requested_top_k=definition.retrieval_top_k,
                retrieved_gold_rank=None,
                reranked_gold_rank=None,
                gold_evidence_included=False,
                loss_stage=EvidenceLossStage.RETRIEVAL,
                failure_labels=[_retrieval_failure_label(definition=definition)],
                trace_references=references,
            )
            return PipelineCaseExecution(
                outcome=outcome,
                retrieval_trace=retrieval_trace,
                reranking_trace=None,
                context_autopsy_report=None,
                lost_evidence_report=None,
            )

        if definition.context_selection_mode is ContextSelectionMode.NAIVE_TOP_K:
            included = retrieval_trace.gold_evidence_rank <= definition.final_evidence_chunk_limit
            outcome = CasePipelineOutcome(
                pipeline_id=definition.pipeline_id,
                case_id=case.case_id,
                requested_top_k=definition.retrieval_top_k,
                retrieved_gold_rank=retrieval_trace.gold_evidence_rank,
                reranked_gold_rank=None,
                gold_evidence_included=included,
                loss_stage=None if included else EvidenceLossStage.RANKING,
                failure_labels=[]
                if included
                else [FailureLabel.RELEVANT_CHUNK_RANKED_TOO_LOW],
                trace_references=references,
            )
            return PipelineCaseExecution(
                outcome=outcome,
                retrieval_trace=retrieval_trace,
                reranking_trace=None,
                context_autopsy_report=None,
                lost_evidence_report=None,
            )

        reranking_trace = reranker.rerank(first_stage_trace=retrieval_trace)
        references.append(
            _trace_reference(
                stage="reranking",
                definition=definition,
                case=case,
                trace=reranking_trace,
            )
        )
        if not reranking_trace.gold_evidence_found or reranking_trace.gold_evidence_rank_after_rerank is None:
            raise ComparisonExecutionError(
                "cross-encoder reranking must preserve retrieved gold evidence in the fixed candidate set"
            )

        assembly = ContextAssembler(
            token_counter=self._token_counter,
            config=ContextAssemblyConfig(
                max_context_tokens=_required_int(definition.max_context_tokens, "max_context_tokens"),
                reserved_output_tokens=_required_int(
                    definition.reserved_output_tokens,
                    "reserved_output_tokens",
                ),
                render_config=ContextRenderConfig(profile=self._config.budgeted_render_profile),
                max_evidence_chunks=definition.final_evidence_chunk_limit,
            ),
        ).assemble(reranking_trace=reranking_trace)
        references.append(
            _trace_reference(
                stage="context_autopsy",
                definition=definition,
                case=case,
                trace=assembly.report,
            )
        )
        lost_evidence = build_lost_evidence_report(
            reranking_trace=reranking_trace,
            autopsy_report=assembly.report,
        )
        if assembly.report.gold_evidence_included:
            outcome = CasePipelineOutcome(
                pipeline_id=definition.pipeline_id,
                case_id=case.case_id,
                requested_top_k=definition.retrieval_top_k,
                retrieved_gold_rank=retrieval_trace.gold_evidence_rank,
                reranked_gold_rank=reranking_trace.gold_evidence_rank_after_rerank,
                gold_evidence_included=True,
                trace_references=references,
            )
        else:
            if lost_evidence is None:
                raise ComparisonExecutionError(
                    "budgeted pipeline dropped known gold evidence without a lost-evidence report"
                )
            outcome = CasePipelineOutcome(
                pipeline_id=definition.pipeline_id,
                case_id=case.case_id,
                requested_top_k=definition.retrieval_top_k,
                retrieved_gold_rank=retrieval_trace.gold_evidence_rank,
                reranked_gold_rank=reranking_trace.gold_evidence_rank_after_rerank,
                gold_evidence_included=False,
                loss_stage=lost_evidence.loss_stage,
                failure_labels=lost_evidence.failure_labels,
                trace_references=references,
            )
        return PipelineCaseExecution(
            outcome=outcome,
            retrieval_trace=retrieval_trace,
            reranking_trace=reranking_trace,
            context_autopsy_report=assembly.report,
            lost_evidence_report=lost_evidence,
        )


def _validate_fixed_pipeline_definitions(definitions: Sequence[PipelineDefinition]) -> None:
    expected_ids = set(PipelineId)
    observed_ids = {definition.pipeline_id for definition in definitions}
    if len(definitions) != len(observed_ids) or observed_ids != expected_ids:
        raise ComparisonExecutionError(
            "runner requires the four fixed pipeline definitions exactly once"
        )


def _validate_retrieval_metric_cutoff(
    *,
    definitions: Sequence[PipelineDefinition],
    retrieval_metric_k: int,
) -> None:
    retrieval_depths = {definition.retrieval_top_k for definition in definitions}
    if len(retrieval_depths) != 1:
        raise ComparisonExecutionError(
            "runner requires all fixed pipelines to use the same retrieval_top_k candidate-pool depth"
        )
    retrieval_depth = retrieval_depths.pop()
    if retrieval_depth < retrieval_metric_k:
        raise ComparisonExecutionError(
            "runner retrieval_top_k must be at least retrieval_metric_k"
        )


def _validate_cases(cases: Sequence[EvaluationCase]) -> tuple[EvaluationCase, ...]:
    if not cases:
        raise ComparisonExecutionError("comparison runner requires at least one evaluation case")
    ordered = tuple(sorted(cases, key=lambda case: case.case_id))
    case_ids = [case.case_id for case in ordered]
    if len(case_ids) != len(set(case_ids)):
        raise ComparisonExecutionError("comparison runner requires unique evaluation case IDs")
    return ordered


def _validate_documents(documents: Sequence[CorpusDocument]) -> tuple[CorpusDocument, ...]:
    if not documents:
        raise ComparisonExecutionError("comparison runner requires at least one corpus document")
    ordered = tuple(sorted(documents, key=lambda document: document.source_doc_id))
    source_doc_ids = [document.source_doc_id for document in ordered]
    if len(source_doc_ids) != len(set(source_doc_ids)):
        raise ComparisonExecutionError("comparison runner requires unique source document IDs")
    return ordered


def _validate_case_sources(
    *,
    cases: Sequence[EvaluationCase],
    document_by_id: dict[str, CorpusDocument],
) -> None:
    for case in cases:
        document = document_by_id.get(case.source_doc_id)
        if document is None:
            raise ComparisonExecutionError(
                f"{case.case_id} references missing source document {case.source_doc_id!r}"
            )
        if case.gold_evidence_text not in document.text:
            raise ComparisonExecutionError(
                f"{case.case_id} gold evidence is not an exact substring of {case.source_doc_id}"
            )


def _inspect_evidence_availability(
    *,
    case: EvaluationCase,
    source_document: CorpusDocument,
    chunks: Sequence[TextChunk],
) -> _EvidenceAvailability:
    source_chunks = [chunk for chunk in chunks if chunk.source_doc_id == case.source_doc_id]
    if not source_chunks:
        raise ComparisonExecutionError(
            f"chunker produced no chunks for source document {case.source_doc_id!r}"
        )
    evidence_start = source_document.text.find(case.gold_evidence_text)
    if evidence_start < 0:
        raise ComparisonExecutionError(
            f"{case.case_id} gold evidence was missing from its declared source document"
        )
    evidence_end = evidence_start + len(case.gold_evidence_text)
    complete_chunk_exists = any(case.gold_evidence_text in chunk.text for chunk in source_chunks)
    overlapping_chunks = [
        chunk
        for chunk in source_chunks
        if chunk.source_char_start < evidence_end and chunk.source_char_end > evidence_start
    ]
    evidence_split = not complete_chunk_exists and len(overlapping_chunks) >= 2
    has_character_boundary_cut = any(
        chunk.boundary_quality is ChunkBoundaryQuality.CHARACTER_CUT
        for chunk in overlapping_chunks
    )
    return _EvidenceAvailability(
        complete_chunk_exists=complete_chunk_exists,
        evidence_split=evidence_split,
        has_character_boundary_cut=has_character_boundary_cut,
    )


def _chunking_failure_labels(*, availability: _EvidenceAvailability) -> list[FailureLabel]:
    labels = [FailureLabel.GOLD_EVIDENCE_SPLIT]
    if availability.has_character_boundary_cut:
        labels.insert(0, FailureLabel.BAD_CHUNK_BOUNDARY)
    return labels


def _retrieval_failure_label(*, definition: PipelineDefinition) -> FailureLabel:
    if definition.retrieval_method is RetrievalMethod.DENSE:
        return FailureLabel.DENSE_RETRIEVAL_MISS
    return FailureLabel.RETRIEVAL_MISS


def _trace_reference(
    *,
    stage: str,
    definition: PipelineDefinition,
    case: EvaluationCase,
    trace: BaseModel,
) -> TraceReference:
    payload = trace.model_dump_json(by_alias=True, exclude_none=False)
    trace_id = f"{stage}:{definition.pipeline_id.value}:{case.case_id}"
    return TraceReference(
        trace_id=trace_id,
        trace_sha256=sha256(payload.encode("utf-8")).hexdigest(),
    )


def _required_int(value: int | None, field_name: str) -> int:
    if value is None:
        raise ComparisonExecutionError(f"budgeted pipeline requires {field_name}")
    return value
