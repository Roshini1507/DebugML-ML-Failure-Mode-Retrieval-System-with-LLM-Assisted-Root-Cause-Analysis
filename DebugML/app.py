"""DebugML: Production-ready Streamlit interface for ML failure analysis."""

import streamlit as st

from rag import RAGError, RAGPipeline

st.set_page_config(
    page_title="DebugML: Failure Analysis Assistant",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom styles for clean UI
st.markdown(
    """
    <style>
    .section-header {
        color: #0d6efd;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_similar_failure(rank: int, score: float, failure: dict) -> None:
    """Render a single similar failure card."""
    similarity_pct = max(0, min(100, int(score * 100)))
    desc = failure.get("failure_description", "—")
    short_desc = desc[:80] + "…" if len(desc) > 80 else desc
    with st.expander(f"#{rank} — Relevance: {similarity_pct}% — {short_desc}", expanded=(rank == 1)):
        st.markdown(f"**Description:** {desc}")
        st.markdown(f"**Root cause:** {failure.get('root_cause', '—')}")
        st.markdown(f"**Solution:** {failure.get('solution', '—')}")
        st.caption(f"Tags: {', '.join(failure.get('tags', []))}")


def main() -> None:
    st.title("🔧 DebugML: Failure Analysis Assistant")
    st.markdown(
        "Describe your ML model failure and get root cause analysis and solutions based on similar past failures."
    )
    st.divider()

    failure_input = st.text_area(
        "Failure description",
        placeholder="e.g., Model achieved 99% accuracy on training but only 60% on test. Performance collapsed in production.",
        height=120,
        help="Enter a detailed description of the ML failure you're experiencing.",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_clicked = st.button("Analyze Failure", type="primary", use_container_width=True)

    if analyze_clicked:
        if not failure_input or not failure_input.strip():
            st.error("Please enter a failure description.")
            return

        with st.spinner("Analyzing failure... Retrieving similar cases and generating analysis."):
            try:
                pipeline = RAGPipeline(top_k=3)
                response = pipeline.query(failure_input.strip())

            except RAGError as e:
                st.error(f"Analysis failed: {e}")
                st.info(
                    "Ensure Ollama is running (`ollama serve`) and llama3 is pulled (`ollama pull llama3`). "
                    "Also verify the vector index exists in data/vector_db/."
                )
                return
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                return

        st.divider()

        # Root cause
        st.markdown('<p class="section-header">Root Cause</p>', unsafe_allow_html=True)
        st.markdown(response["root_cause"])

        # Confidence
        st.markdown('<p class="section-header">Confidence</p>', unsafe_allow_html=True)
        st.caption(response["confidence"].capitalize())

        # Recommended solution
        st.markdown('<p class="section-header">Recommended Solution</p>', unsafe_allow_html=True)
        st.markdown(response["recommended_solution"])

        # Similar past failures
        st.markdown('<p class="section-header">Similar Past Failures</p>', unsafe_allow_html=True)
        for sr in response["similar_failures"]:
            render_similar_failure(
                rank=sr["rank"],
                score=sr["score"],
                failure=sr["failure"],
            )


if __name__ == "__main__":
    main()
