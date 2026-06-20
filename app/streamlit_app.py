"""Read-only Streamlit explorers for the fixed synthetic RAG reliability benchmark."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_lab.case_explorer import (
    FailureCaseView,
    PipelineEvidenceStatus,
    load_failure_case_views,
)
from rag_lab.chunking_explorer import (
    ChunkingCaseView,
    ChunkingStrategyView,
    ControlledBoundaryProbeView,
    load_chunking_case_views,
)
from rag_lab.context_autopsy_explorer import (
    ContextAssemblyView,
    ContextAutopsyCaseView,
    ContextDecisionView,
    load_context_autopsy_case_view,
)
from rag_lab.retrieval_explorer import (
    CandidateSetState,
    RetrievalCaseView,
    RetrievalPipelineView,
    load_retrieval_case_views,
)
from rag_lab.schemas import TextChunk


PROJECT_ROOT = Path(__file__).resolve().parents[1]


st.set_page_config(
    page_title="RAG Fidelity & Context Autopsy",
    page_icon="ðŸ”Ž",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_failure_views() -> tuple[FailureCaseView, ...]:
    """Cache the reviewed static baseline view used by the Failure Case Explorer."""

    return load_failure_case_views(project_root=PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def load_chunking_views() -> tuple[ChunkingCaseView, ...]:
    """Cache deterministic chunking views over the fixed synthetic corpus."""

    return load_chunking_case_views(project_root=PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def load_retrieval_views() -> tuple[RetrievalCaseView, ...]:
    """Cache reviewed ranks and bounded trace references without rerunning retrieval."""

    return load_retrieval_case_views(project_root=PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def load_context_autopsy_view() -> ContextAutopsyCaseView:
    """Cache the fixed controlled context-pressure accounting view."""

    return load_context_autopsy_case_view(project_root=PROJECT_ROOT)


def main() -> None:
    """Render read-only evidence, chunking, retrieval, and context-autopsy surfaces."""

    st.title("RAG Fidelity & Context Autopsy")
    st.caption(
        "Read-only reliability demo Â· fixed synthetic cases Â· reviewed four-pipeline baseline"
    )

    try:
        failure_views = load_failure_views()
        chunking_views = load_chunking_views()
        retrieval_views = load_retrieval_views()
        context_autopsy_view = load_context_autopsy_view()
    except (OSError, RuntimeError, ValueError) as error:
        st.error("The explorer could not load its fixed synthetic benchmark assets.")
        st.exception(error)
        st.stop()

    failure_by_case_id = {view.case.case_id: view for view in failure_views}
    chunking_by_case_id = {view.case.case_id: view for view in chunking_views}
    retrieval_by_case_id = {view.case.case_id: view for view in retrieval_views}
    if (
        set(failure_by_case_id) != set(chunking_by_case_id)
        or set(failure_by_case_id) != set(retrieval_by_case_id)
    ):
        st.error("The explorer surfaces do not cover the same fixed evaluation cases.")
        st.stop()

    with st.sidebar:
        st.header("Explorer")
        selected_surface = st.radio(
            "Read-only surface",
            options=("Failure case", "Chunking", "Retrieval", "Context autopsy"),
            horizontal=False,
        )
        st.divider()
        st.header("Case selection")
        selected_case_id = st.selectbox(
            "Fixed diagnostic case",
            options=tuple(failure_by_case_id),
            format_func=lambda case_id: _case_label(failure_by_case_id[case_id]),
        )
        st.divider()
        st.caption("Demo boundary")
        st.write(
            "This demo reads fixed synthetic cases and a reviewed baseline artifact. "
            "The Chunking Explorer deterministically emits local chunks with "
            "`tiktoken:cl100k_base`. The Retrieval Explorer reads committed candidate "
            "presence, ranks, loss labels, and trace IDs. The Context Autopsy Explorer "
            "runs only the fixed local context-pressure trace to measure rendered prompt "
            "tax under an explicit tokenizer. It does not run embeddings, retrieval, "
            "reranking, or answer generation."
        )
        st.caption(
            "No customer data, credentials, prompts, raw retrieval candidates, raw "
            "rendered context, or generated answers are loaded."
        )

    if selected_surface == "Failure case":
        _render_failure_case(failure_by_case_id[selected_case_id])
    elif selected_surface == "Chunking":
        _render_chunking_case(chunking_by_case_id[selected_case_id])
    elif selected_surface == "Retrieval":
        _render_retrieval_case(retrieval_by_case_id[selected_case_id])
    else:
        _render_context_autopsy_case(context_autopsy_view)


def _render_failure_case(view: FailureCaseView) -> None:
    """Render one fixed case and its evidence lifecycle across four pipelines."""

    case = view.case
    baseline = view.baseline_status

    st.subheader(case.case_id.replace("_", " ").title())
    first_column, second_column, third_column = st.columns(3)
    first_column.metric("Document type", case.document_type.value.replace("_", " "))
    second_column.metric("Query type", case.query_type.value.replace("_", " "))
    third_column.metric(
        "Expected diagnostic",
        case.expected_failure_mode.value.replace("_", " "),
    )

    st.markdown("### Question")
    st.info(case.query)

    content_column, diagnosis_column = st.columns((3, 2))
    with content_column:
        st.markdown("### Gold evidence")
        st.code(case.gold_evidence_text, language=None)

        st.markdown("### Expected answer")
        st.write(case.gold_answer)

        st.markdown("### Why this case is diagnostic")
        st.write(case.diagnostic_note)

    with diagnosis_column:
        st.markdown("### Baseline evidence state")
        _render_baseline_status(baseline)

        st.markdown("### Scope")
        st.write(
            "The status table shows evidence selection only. It does not claim "
            "anything about generated-answer correctness or citation correctness."
        )
        st.caption(f"Reviewed comparison run: {view.source_run_id}")

    st.markdown("### Four-pipeline evidence lifecycle")
    st.dataframe(
        [_status_row(status) for status in view.pipeline_statuses],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("How to read this screen", expanded=False):
        st.markdown(
            "- **Chunking loss:** the evidence became unusable before retrieval.\n"
            "- **Retrieval loss:** the candidate set did not contain the evidence.\n"
            "- **Ranking loss:** the evidence was retrieved but did not reach final selection.\n"
            "- **Context assembly loss:** the evidence was retrieved and ranked, then excluded "
            "by measured context packing.\n"
            "- **Included:** gold evidence reached the final evidence-selection boundary."
        )


def _render_chunking_case(view: ChunkingCaseView) -> None:
    """Render character versus sentence-aware token chunking for one synthetic case."""

    case = view.case
    st.subheader(f"{case.case_id.replace('_', ' ').title()} Â· Chunking Autopsy")
    st.caption(
        "This is a deterministic local chunking comparison. It is not a retrieval, "
        "reranking, context-packing, or answer-generation result."
    )

    st.markdown("### Question")
    st.info(case.query)

    st.markdown("### Gold evidence under inspection")
    st.code(case.gold_evidence_text, language=None)
    st.caption(
        f"Synthetic source span: characters {view.gold_evidence_start}â€“"
        f"{view.gold_evidence_end} of {view.source_char_count}"
    )

    metric_columns = st.columns(4)
    _render_chunking_metric(
        metric_columns[0],
        label="Character chunks",
        value=str(view.character_chunking.report.chunk_count),
        detail=f"{view.character_chunking.configured_limit} characters",
    )
    _render_chunking_metric(
        metric_columns[1],
        label="Character evidence",
        value=_preservation_label(view.character_chunking.report.gold_evidence_preserved),
        detail=_split_detail(view.character_chunking.report.gold_evidence_split),
    )
    _render_chunking_metric(
        metric_columns[2],
        label="Sentence-aware chunks",
        value=str(view.sentence_aware_token_chunking.report.chunk_count),
        detail=f"{view.sentence_aware_token_chunking.configured_limit} tokens",
    )
    _render_chunking_metric(
        metric_columns[3],
        label="Sentence-aware evidence",
        value=_preservation_label(
            view.sentence_aware_token_chunking.report.gold_evidence_preserved
        ),
        detail=_split_detail(
            view.sentence_aware_token_chunking.report.gold_evidence_split
        ),
    )

    st.markdown("### Standard local configuration")
    st.caption(
        "These results use the benchmark-aligned local settings: 700 character windows and "
        "96 sentence-aware tokens. They report the actual outcome for this case and are not "
        "a deliberately forced failure fixture."
    )
    character_column, token_column = st.columns(2)
    with character_column:
        _render_chunking_strategy(
            title="Standard character configuration",
            strategy_view=view.character_chunking,
            gold_start=view.gold_evidence_start,
            gold_end=view.gold_evidence_end,
        )
    with token_column:
        _render_chunking_strategy(
            title="Standard sentence-aware token configuration",
            strategy_view=view.sentence_aware_token_chunking,
            gold_start=view.gold_evidence_start,
            gold_end=view.gold_evidence_end,
        )

    if view.controlled_boundary_probe is not None:
        _render_controlled_boundary_probe(
            probe=view.controlled_boundary_probe,
            gold_start=view.gold_evidence_start,
            gold_end=view.gold_evidence_end,
        )

    with st.expander("How to read this screen", expanded=False):
        st.markdown(
            "- **Standard configuration:** reports what the benchmark-aligned local chunk settings actually do.\n"
            "- **Controlled boundary probe:** a separate diagnostic fixture that intentionally ends a "
            "character window inside the known clause; it is not a four-pipeline benchmark metric.\n"
            "- **Preserved:** one emitted chunk fully contains the gold evidence span.\n"
            "- **Split:** the gold span overlaps two or more chunks but no single chunk "
            "contains it completely.\n"
            "- **Boundary quality:** character windows cut at fixed offsets; sentence-aware "
            "chunks preserve sentence, table-row, or log-event units when they fit the token budget.\n"
            "- **Token count:** measured with `tiktoken:cl100k_base` on final emitted chunk text."
        )


def _render_retrieval_case(view: RetrievalCaseView) -> None:
    """Render reviewed candidate availability and ranks without rerunning retrieval."""

    case = view.case
    baseline = view.baseline_view

    st.subheader(f"{case.case_id.replace('_', ' ').title()} Â· Retrieval Autopsy")
    st.caption(
        "This surface reads reviewed candidate presence and rank fields from the committed "
        "baseline artifact. It does not perform a fresh retrieval or reranking run."
    )

    st.markdown("### Question")
    st.info(case.query)
    st.caption(case.diagnostic_note)

    metric_columns = st.columns(4)
    _render_retrieval_metric(
        metric_columns[0],
        label="Baseline candidate state",
        value=_candidate_state_label(baseline.candidate_set_state),
        detail=f"{baseline.retrieval_method.value} retrieval",
    )
    _render_retrieval_metric(
        metric_columns[1],
        label="Candidate pool",
        value=f"Top {baseline.candidate_depth}",
        detail=f"Recall reported at {view.retrieval_metric_k}",
    )
    _render_retrieval_metric(
        metric_columns[2],
        label="Baseline first-stage rank",
        value=_rank_label(baseline.retrieved_gold_rank),
        detail="Gold evidence rank before reranking",
    )
    _render_retrieval_metric(
        metric_columns[3],
        label="Baseline final evidence",
        value="Included" if baseline.gold_evidence_included else "Not included",
        detail=_loss_detail(baseline),
    )

    st.markdown("### Baseline diagnosis")
    _render_retrieval_baseline_diagnosis(baseline)

    st.markdown("### Reviewed candidate and rank path")
    st.dataframe(
        [_retrieval_row(pipeline) for pipeline in view.pipeline_views],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Candidate presence is measured against each pipeline's retained top-8 candidate pool. "
        f"Recall@{view.retrieval_metric_k} remains a separate reported metric cutoff."
    )

    with st.expander("Bounded trace references", expanded=False):
        st.caption(
            "The reviewed artifact retains trace IDs, not raw candidate chunks, similarity scores, "
            "source documents, prompts, or rendered context."
        )
        st.dataframe(
            [_trace_reference_row(pipeline) for pipeline in view.pipeline_views],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("How to read this screen", expanded=False):
        st.markdown(
            "- **Unavailable after chunking:** the complete evidence span was not a usable chunk, "
            "so this is not a first-stage retrieval miss.\n"
            "- **Missing from candidate set:** the complete evidence was usable but absent from the "
            "pipeline's retained candidate pool.\n"
            "- **Present:** the evidence entered the candidate pool; its first-stage rank is shown.\n"
            "- **Reranked rank:** exists only for the cross-encoder reranking pipeline and governs "
            "the final rank used for context selection.\n"
            "- **Final evidence:** records whether the gold evidence reached the final "
            "evidence-selection boundary. A later loss stage can still be ranking or context assembly.\n"
            "- **No candidate text or score:** the committed artifact intentionally excludes raw "
            "retrieval payloads and model scores."
        )



def _render_context_autopsy_case(view: ContextAutopsyCaseView) -> None:
    """Render the fixed controlled context-pressure autopsy without rerunning retrieval."""

    case = view.case
    verbose = view.verbose_audit
    compact = view.compact_citation

    st.subheader(f"{case.case_id.replace('_', ' ').title()} Â· Context Autopsy")
    st.warning(
        "Controlled local diagnostic: both render profiles receive the same calibrated context "
        "window and the same fixed reranked candidate order. This is a mechanism proof, not a "
        "claim that the reviewed four-pipeline benchmark has a standard context-budget regression."
    )
    st.caption(
        "The screen runs deterministic context accounting only. It does not run embeddings, "
        "retrieval, reranking, or answer generation."
    )

    st.markdown("### Question")
    st.info(case.query)

    st.markdown("### Gold evidence under inspection")
    st.code(case.gold_evidence_text, language=None)
    st.caption(
        f"Tokenizer: `{view.tokenizer_name}` Â· fixed sentence-aware source budget: "
        f"{view.sentence_aware_max_tokens} tokens"
    )

    metric_columns = st.columns(4)
    _render_context_metric(
        metric_columns[0],
        label="Calibrated context window",
        value=str(view.calibrated_context_tokens),
        detail=f"{view.reserved_output_tokens} reserved output tokens",
    )
    _render_context_metric(
        metric_columns[1],
        label="Verbose static prompt tax",
        value=str(verbose.static_prompt_tokens),
        detail=f"{verbose.available_evidence_tokens} evidence tokens available",
    )
    _render_context_metric(
        metric_columns[2],
        label="Verbose gold evidence",
        value="Dropped" if verbose.gold_evidence_dropped else "Included",
        detail=(
            verbose.gold_evidence_drop_reason.value.replace("_", " ")
            if verbose.gold_evidence_drop_reason is not None
            else "no drop reason"
        ),
    )
    _render_context_metric(
        metric_columns[3],
        label="Compact gold evidence",
        value="Included" if compact.gold_evidence_included else "Dropped",
        detail=f"{compact.remaining_evidence_tokens} evidence tokens remaining",
    )

    st.markdown("### Same budget, different rendered context cost")
    verbose_column, compact_column = st.columns(2)
    with verbose_column:
        _render_context_profile(
            title="Verbose audit wrappers",
            assembly=verbose,
        )
    with compact_column:
        _render_context_profile(
            title="Compact citation wrappers",
            assembly=compact,
        )

    st.markdown("### Controlled loss diagnosis")
    diagnosis = view.loss_diagnosis
    st.error(
        f"Gold evidence was reranked at #{diagnosis.gold_evidence_rank_before_context}, then "
        f"dropped because `{diagnosis.drop_reason.value}` under verbose audit wrappers."
    )
    st.write(diagnosis.evidence_summary)
    st.info(diagnosis.repair_recommendation)
    st.caption(
        "Failure labels: "
        + ", ".join(label.value.replace("_", " ") for label in diagnosis.failure_labels)
    )

    with st.expander("How to read this screen", expanded=False):
        st.markdown(
            "- **Static prompt tax:** tokens consumed by the measured system instruction, question, "
            "and response contract before any evidence enters context.\n"
            "- **Rendered context cost:** raw evidence tokens plus wrapper and citation tokens.\n"
            "- **Same calibrated window:** both profiles share one total context budget and one "
            "reserved-output allowance.\n"
            "- **Verbose audit wrappers:** retain richer per-chunk metadata but can displace "
            "lower-ranked evidence under pressure.\n"
            "- **Compact citation wrappers:** reduce wrapper tax and retain the rank-three gold "
            "candidate in this fixed diagnostic.\n"
            "- **No raw rendered context:** this screen displays decision metadata only, not the "
            "assembled prompt or raw candidate text."
        )


def _render_context_profile(*, title: str, assembly: ContextAssemblyView) -> None:
    """Render bounded accounting and per-candidate decisions for one fixed wrapper profile."""

    st.markdown(f"#### {title}")
    if assembly.gold_evidence_included:
        st.success("Gold evidence reached final rendered context.")
    else:
        reason = (
            assembly.gold_evidence_drop_reason.value.replace("_", " ")
            if assembly.gold_evidence_drop_reason is not None
            else "unknown reason"
        )
        st.warning(f"Gold evidence did not reach final rendered context: {reason}.")

    st.caption(
        f"Profile: `{assembly.render_profile.value}` Â· candidates: {assembly.candidate_count} Â· "
        f"included: {assembly.included_chunk_count} Â· dropped: {assembly.dropped_chunk_count}"
    )
    st.caption(
        f"Raw evidence used: {assembly.used_raw_evidence_tokens} Â· "
        f"rendered evidence used: {assembly.used_rendered_evidence_tokens} Â· "
        f"wrapper tax: {assembly.rendering_token_tax_tokens} Â· "
        f"remaining: {assembly.remaining_evidence_tokens}"
    )
    st.dataframe(
        [_context_decision_row(decision) for decision in assembly.decisions],
        use_container_width=True,
        hide_index=True,
    )


def _render_context_metric(
    container: object,
    *,
    label: str,
    value: str,
    detail: str,
) -> None:
    """Render a compact context-autopsy metric without returning UI state."""

    container.metric(label, value)
    container.caption(detail)


def _context_decision_row(decision: ContextDecisionView) -> dict[str, object]:
    """Render bounded candidate accounting without raw chunk text or rendered prompt content."""

    return {
        "Reranked rank": f"#{decision.reranked_rank}",
        "First-stage rank": f"#{decision.first_stage_rank}",
        "Chunk ID": decision.chunk_id,
        "Raw tokens": decision.raw_context_token_count,
        "Rendered tokens": decision.rendered_context_token_count,
        "Wrapper tax": decision.rendering_token_tax,
        "Gold evidence": "Yes" if decision.gold_evidence_match else "No",
        "Decision": "Included" if decision.included else "Dropped",
        "Drop reason": decision.drop_reason.value if decision.drop_reason else "â€”",
    }


def _render_controlled_boundary_probe(
    *,
    probe: ControlledBoundaryProbeView,
    gold_start: int,
    gold_end: int,
) -> None:
    """Render the existing controlled boundary diagnostic without calling it benchmark evidence."""

    st.divider()
    st.markdown("### Controlled boundary probe")
    st.warning(
        "Separate diagnostic fixture: the character window is deliberately positioned inside "
        "the known gold clause. This isolates boundary damage; it is not the standard 700-character "
        "benchmark configuration and does not change any benchmark metric."
    )
    st.caption(
        f"Controlled character window: {probe.character_window_characters} characters Â· "
        f"Sentence-aware repair budget: {probe.sentence_aware_max_tokens} tokens"
    )
    character_column, token_column = st.columns(2)
    with character_column:
        _render_chunking_strategy(
            title="Controlled character boundary",
            strategy_view=probe.character_chunking,
            gold_start=gold_start,
            gold_end=gold_end,
        )
    with token_column:
        _render_chunking_strategy(
            title="Controlled sentence-aware repair",
            strategy_view=probe.sentence_aware_token_chunking,
            gold_start=gold_start,
            gold_end=gold_end,
        )


def _render_chunking_metric(
    container: object,
    *,
    label: str,
    value: str,
    detail: str,
) -> None:
    """Render a compact chunking metric without returning UI state."""

    container.metric(label, value)
    container.caption(detail)


def _render_retrieval_metric(
    container: object,
    *,
    label: str,
    value: str,
    detail: str,
) -> None:
    """Render a compact retrieval metric without returning UI state."""

    container.metric(label, value)
    container.caption(detail)


def _render_chunking_strategy(
    *,
    title: str,
    strategy_view: ChunkingStrategyView,
    gold_start: int,
    gold_end: int,
) -> None:
    """Render one strategy report and the emitted synthetic chunks."""

    report = strategy_view.report
    st.markdown(f"#### {title}")
    if report.gold_evidence_preserved:
        st.success("Gold evidence is preserved in one emitted chunk.")
    elif report.gold_evidence_split:
        st.warning("Gold evidence is split across emitted chunks.")
    else:
        st.warning("Gold evidence is not preserved by this emitted chunk set.")

    st.caption(
        f"Tokenizer: `{report.tokenizer_name}` Â· "
        f"Boundary quality: `{report.boundary_quality.value}` Â· "
        f"Average chunk size: {report.avg_token_count} tokens Â· "
        f"Maximum: {report.max_token_count} tokens"
    )

    for chunk in report.chunks:
        relation = _gold_relation(
            chunk=chunk,
            gold_start=gold_start,
            gold_end=gold_end,
        )
        label = (
            f"Chunk {chunk.chunk_index + 1} Â· {chunk.token_count} tokens Â· "
            f"source chars {chunk.source_char_start}â€“{chunk.source_char_end} Â· {relation}"
        )
        with st.expander(label, expanded=relation != "no gold evidence"):
            st.code(chunk.text, language=None)


def _render_baseline_status(status: PipelineEvidenceStatus) -> None:
    """Render the baseline state with a clear no-claim boundary."""

    if status.gold_evidence_included:
        st.success("Gold evidence reached the final evidence-selection boundary.")
        st.caption(
            f"Rank used for final context selection: {status.rank_used_for_context}"
        )
        return

    st.warning("Gold evidence did not reach the final evidence-selection boundary.")
    st.write(f"**Loss stage:** {status.loss_stage.value.replace('_', ' ')}")
    st.write(
        "**Failure labels:** "
        + ", ".join(label.value.replace("_", " ") for label in status.failure_labels)
    )
    if status.rank_used_for_context is not None:
        st.caption(f"Rank available before loss: {status.rank_used_for_context}")


def _render_retrieval_baseline_diagnosis(view: RetrievalPipelineView) -> None:
    """Render a stage-accurate baseline explanation from the reviewed outcome."""

    if view.candidate_set_state is CandidateSetState.UNAVAILABLE_AFTER_CHUNKING:
        st.warning(
            "The complete gold evidence was unavailable to first-stage retrieval because "
            "chunking failed before a usable evidence unit reached the candidate boundary."
        )
    elif view.candidate_set_state is CandidateSetState.MISSING_FROM_CANDIDATE_SET:
        st.warning(
            f"The complete gold evidence was absent from the reviewed top-{view.candidate_depth} "
            "first-stage candidate pool."
        )
    elif view.gold_evidence_included:
        st.success(
            f"The gold evidence entered the candidate set at rank {view.retrieved_gold_rank} "
            "and reached the final evidence-selection boundary."
        )
    else:
        rank = _rank_label(view.rank_used_for_context)
        stage = view.loss_stage.value.replace("_", " ") if view.loss_stage else "unknown stage"
        st.warning(
            f"The gold evidence entered the candidate set at rank {rank}, then failed at "
            f"{stage} before final evidence selection."
        )

    if view.failure_labels:
        st.caption(
            "Failure labels: "
            + ", ".join(label.value.replace("_", " ") for label in view.failure_labels)
        )


def _status_row(status: PipelineEvidenceStatus) -> dict[str, object]:
    """Convert one typed status into a compact table row."""

    return {
        "Pipeline": status.pipeline_id.value,
        "Gold evidence included": "Yes" if status.gold_evidence_included else "No",
        "Retrieved rank": status.retrieved_gold_rank,
        "Reranked rank": status.reranked_gold_rank,
        "Rank used for context": status.rank_used_for_context,
        "Loss stage": status.loss_stage.value if status.loss_stage else "â€”",
        "Failure labels": (
            ", ".join(label.value for label in status.failure_labels)
            if status.failure_labels
            else "â€”"
        ),
    }


def _retrieval_row(view: RetrievalPipelineView) -> dict[str, object]:
    """Convert one typed retrieval view into a compact comparison row."""

    return {
        "Pipeline": view.pipeline_id.value,
        "Chunking": view.chunking_strategy.value,
        "Retrieval": view.retrieval_method.value,
        "Reranker": "Yes" if view.reranker_enabled else "No",
        "Candidate pool": f"Top {view.candidate_depth}",
        "Candidate state": _candidate_state_label(view.candidate_set_state),
        "First-stage rank": _rank_label(view.retrieved_gold_rank),
        "Reranked rank": _rank_label(view.reranked_gold_rank),
        "Rank used for context": _rank_label(view.rank_used_for_context),
        "Final evidence": "Included" if view.gold_evidence_included else "Not included",
        "Loss stage": view.loss_stage.value if view.loss_stage else "â€”",
    }


def _trace_reference_row(view: RetrievalPipelineView) -> dict[str, str]:
    """Render bounded trace IDs without exposing the underlying trace payload."""

    return {
        "Pipeline": view.pipeline_id.value,
        "Trace IDs": "\n".join(view.trace_ids),
    }


def _gold_relation(*, chunk: TextChunk, gold_start: int, gold_end: int) -> str:
    """Classify a chunk's relation to one exact gold-evidence source span."""

    if chunk.source_char_start <= gold_start and chunk.source_char_end >= gold_end:
        return "contains complete gold evidence"
    if chunk.source_char_start < gold_end and chunk.source_char_end > gold_start:
        return "overlaps gold evidence"
    return "no gold evidence"


def _preservation_label(is_preserved: bool) -> str:
    """Render a concise yes/no evidence-preservation label."""

    return "Preserved" if is_preserved else "Not preserved"


def _split_detail(is_split: bool) -> str:
    """Render a concise split-state label."""

    return "Split across chunks" if is_split else "Not split"


def _candidate_state_label(state: CandidateSetState) -> str:
    """Render a concise candidate-state label."""

    if state is CandidateSetState.PRESENT:
        return "Present"
    if state is CandidateSetState.MISSING_FROM_CANDIDATE_SET:
        return "Missing"
    return "Unavailable after chunking"


def _rank_label(rank: int | None) -> str:
    """Render a rank without representing absence as rank zero."""

    return f"#{rank}" if rank is not None else "â€”"


def _loss_detail(view: RetrievalPipelineView) -> str:
    """Render final-boundary detail without overclaiming generated-answer behavior."""

    if view.gold_evidence_included:
        return f"Rank used: {_rank_label(view.rank_used_for_context)}"
    if view.loss_stage is None:
        return "No reviewed loss stage"
    return f"Lost at {view.loss_stage.value.replace('_', ' ')}"


def _case_label(view: FailureCaseView) -> str:
    """Make the selector useful without leaking more corpus data into the sidebar."""

    return f"{view.case.case_id} Â· {view.case.document_type.value.replace('_', ' ')}"


if __name__ == "__main__":
    main()
