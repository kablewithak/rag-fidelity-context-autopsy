"""Read-only Streamlit Failure Case Explorer for the fixed synthetic benchmark."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_lab.case_explorer import (
    FailureCaseView,
    PipelineEvidenceStatus,
    load_failure_case_views,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


st.set_page_config(
    page_title="RAG Fidelity & Context Autopsy",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_views() -> tuple[FailureCaseView, ...]:
    """Cache only the small, reviewed static view model for this read-only demo."""

    return load_failure_case_views(project_root=PROJECT_ROOT)


def main() -> None:
    """Render the first operator-facing surface without rerunning models."""

    st.title("RAG Fidelity & Context Autopsy")
    st.caption(
        "Read-only Failure Case Explorer · fixed synthetic cases · reviewed four-pipeline baseline"
    )

    try:
        views = load_views()
    except (OSError, ValueError) as error:
        st.error(
            "The explorer could not load the fixed evaluation cases or reviewed baseline artifact."
        )
        st.exception(error)
        st.stop()

    views_by_case_id = {view.case.case_id: view for view in views}

    with st.sidebar:
        st.header("Case selection")
        selected_case_id = st.selectbox(
            "Fixed diagnostic case",
            options=tuple(views_by_case_id),
            format_func=lambda case_id: _case_label(views_by_case_id[case_id]),
        )
        st.divider()
        st.caption("Demo boundary")
        st.write(
            "This screen reads committed synthetic evaluation cases and the reviewed "
            "baseline artifact. It does not run embeddings, retrieval, reranking, "
            "context assembly, or answer generation."
        )
        st.caption(
            "No customer data, raw corpus text, prompts, or generated answers are loaded."
        )

    view = views_by_case_id[selected_case_id]
    _render_case(view)


def _render_case(view: FailureCaseView) -> None:
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


def _status_row(status: PipelineEvidenceStatus) -> dict[str, object]:
    """Convert one typed status into a compact table row."""

    return {
        "Pipeline": status.pipeline_id.value,
        "Gold evidence included": "Yes" if status.gold_evidence_included else "No",
        "Retrieved rank": status.retrieved_gold_rank,
        "Reranked rank": status.reranked_gold_rank,
        "Rank used for context": status.rank_used_for_context,
        "Loss stage": status.loss_stage.value if status.loss_stage else "—",
        "Failure labels": (
            ", ".join(label.value for label in status.failure_labels)
            if status.failure_labels
            else "—"
        ),
    }


def _case_label(view: FailureCaseView) -> str:
    """Make the selector useful without leaking more corpus data into the sidebar."""

    return f"{view.case.case_id} · {view.case.document_type.value.replace('_', ' ')}"


if __name__ == "__main__":
    main()
