import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, DEFAULT_TOP_K, FAISS_INDEX_DIR
from src.document_loader import SUPPORTED_EXTENSIONS
from src.mcp_tools import export_answer_to_markdown, save_study_card_to_notion
from src.rag_chain import answer_question
from src.vector_store import build_faiss_index, index_exists


st.set_page_config(
    page_title="Computational Linear Algebra Study Assistant",
    page_icon="AI",
    layout="wide",
)

st.title("Computational Linear Algebra Study Assistant")

with st.sidebar:
    st.header("Knowledge Base")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    uploaded_files = st.file_uploader(
        "Add documents",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )

    if uploaded_files:
        saved_files = []
        for uploaded_file in uploaded_files:
            destination = DATA_DIR / Path(uploaded_file.name).name
            destination.write_bytes(uploaded_file.getbuffer())
            saved_files.append(destination.name)
        st.success("Saved: " + ", ".join(saved_files))

    current_files = sorted(
        file_path.name
        for file_path in DATA_DIR.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if current_files:
        st.caption(f"{len(current_files)} document file(s) in data/")
        with st.expander("View documents"):
            for file_name in current_files:
                st.write(file_name)
    else:
        st.caption("No documents in data/ yet.")

    if index_exists():
        st.success(f"FAISS index ready: {FAISS_INDEX_DIR}")
    else:
        st.warning("No FAISS index found. Build one after adding files to data/.")

    if st.button("Build / Rebuild Index", type="primary"):
        with st.spinner("Loading documents, chunking, embedding, and saving FAISS index..."):
            try:
                store = build_faiss_index()
                st.success(f"Indexed {store.index.ntotal} chunks.")
            except Exception as exc:
                st.error(str(exc))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_export" not in st.session_state:
    st.session_state.pending_export = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not index_exists():
            st.error("Build the FAISS index first from the sidebar.")
        else:
            with st.spinner("Retrieving context and generating an answer..."):
                try:
                    result = answer_question(question, top_k=top_k)
                    st.markdown(result["answer"])
                    if result["sources"]:
                        st.caption("Sources: " + ", ".join(result["sources"]))
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )
                    st.session_state.pending_export = {
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "topic": result.get("query_topic", "general"),
                    }
                except Exception as exc:
                    st.error(str(exc))

if st.session_state.pending_export:
    st.divider()
    st.subheader("Review Answer")
    st.markdown(st.session_state.pending_export["answer"])
    st.caption("Sources: " + ", ".join(st.session_state.pending_export["sources"]))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve and Export"):
            try:
                exported_path = export_answer_to_markdown.invoke(
                    {
                        "question": st.session_state.pending_export["question"],
                        "answer": st.session_state.pending_export["answer"],
                        "sources": ", ".join(st.session_state.pending_export["sources"]),
                    }
                )
                st.success(f"Exported approved answer to {exported_path}")
                st.session_state.pending_export = None
            except Exception as exc:
                st.error(str(exc))
    with col2:
        if st.button("Save Study Card to Notion"):
            try:
                result = save_study_card_to_notion.invoke(
                    {
                        "question": st.session_state.pending_export["question"],
                        "answer": st.session_state.pending_export["answer"],
                        "topic": st.session_state.pending_export.get("topic", "general"),
                        "tags": ["computational-linear-algebra", "rag-study-card"],
                        "sources": st.session_state.pending_export["sources"],
                    }
                )
                st.success(f"Saved Notion study card: {result.get('url', 'created')}")
                st.session_state.pending_export = None
            except Exception as exc:
                st.error(str(exc))

    if st.button("Keep Editing / Ask Follow-up"):
        st.info("Ask a follow-up question or request changes in the chat box.")
