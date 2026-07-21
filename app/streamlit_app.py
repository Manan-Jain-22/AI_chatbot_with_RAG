import sys
from pathlib import Path
import re

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, DEFAULT_TOP_K, FAISS_INDEX_DIR
import src.config as app_config
from src.document_loader import SUPPORTED_EXTENSIONS
from src.mcp_tools import save_study_card_to_notion
from src.rag_chain import answer_question
from src.vector_store import build_faiss_index, index_exists


PUBLIC_DEMO_MODE = getattr(app_config, "PUBLIC_DEMO_MODE", False)

st.set_page_config(
    page_title="Computational Linear Algebra Study Assistant",
    page_icon="AI",
    layout="wide",
)

st.title("Computational Linear Algebra Study Assistant")
st.caption(
    "A hosted RAG application using LangChain, Gemini/OpenAI models, FAISS retrieval, "
    "LangGraph orchestration, and MCP export tools."
)

def get_demo_result(question: str) -> dict:
    return {
        "question": question,
        "answer": (
            "Public demo mode is running without external model calls. To answer "
            "arbitrary uploaded documents, disable `PUBLIC_DEMO_MODE`, configure a "
            "live provider key, build the FAISS index, and ask again."
        ),
        "query_topic": "demo_mode",
        "rewritten_question": "cost safe hosted portfolio demo without external model API calls",
        "sources": ["data/demo_computational_linear_algebra_notes.md"],
        "documents": [],
    }


def slugify_filename(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return f"{slug[:80] or 'rag-answer'}.md"


def build_markdown_export(question: str, answer: str, sources: list[str]) -> str:
    source_text = "\n".join(f"- {source}" for source in sources) if sources else "- No sources"
    return (
        "# RAG Study Assistant Answer\n\n"
        f"## Question\n\n{question}\n\n"
        f"## Answer\n\n{answer}\n\n"
        f"## Sources\n\n{source_text}\n"
    )


if PUBLIC_DEMO_MODE:
    st.info(
        "Public demo mode is on. Demo questions work without OpenAI calls, so this "
        "portfolio app will not consume API credits."
    )

if "last_index_message" not in st.session_state:
    st.session_state.last_index_message = None

with st.sidebar:
    st.header("Knowledge Base")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PUBLIC_DEMO_MODE:
        provider = app_config.LLM_PROVIDER
        st.caption(f"Live provider: {provider}")
        if provider == "gemini" and app_config.GOOGLE_API_KEY:
            st.success("Google Gemini key detected from configuration.")
        elif provider == "openai" and app_config.OPENAI_API_KEY:
            st.success("OpenAI key detected from configuration.")
        else:
            st.warning("Live provider key not detected. Add GOOGLE_API_KEY or GEMINI_API_KEY in Streamlit secrets.")
            with st.expander("Secrets Diagnostics"):
                secret_names = app_config.get_visible_secret_names()
                if secret_names:
                    st.write("Detected secret names:")
                    for secret_name in secret_names:
                        st.code(secret_name)
                else:
                    st.write("No Streamlit secrets are visible to the app.")

    if PUBLIC_DEMO_MODE:
        st.caption("Uploads and index rebuilding are disabled in public demo mode.")
    else:
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

    if PUBLIC_DEMO_MODE:
        st.success("Cost-safe demo ready.")
    else:
        if st.session_state.last_index_message:
            st.success(st.session_state.last_index_message)
            st.session_state.last_index_message = None

        if index_exists():
            st.success(f"FAISS index ready: {FAISS_INDEX_DIR}")
        else:
            st.warning("No FAISS index found. Build one after adding files to data/.")

        if st.button("Build / Rebuild Index", type="primary"):
            with st.spinner("Loading documents, chunking, embedding, and saving FAISS index..."):
                try:
                    store = build_faiss_index()
                    st.session_state.last_index_message = f"Indexed {store.index.ntotal} chunks."
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with st.expander("Agent Workflow"):
        st.write("1. Classify the query topic")
        st.write("2. Rewrite the question with course-specific terms")
        st.write("3. Retrieve semantic matches from FAISS")
        st.write("4. Select the best context")
        st.write("5. Generate a grounded answer")
        st.write("6. Export only after user approval")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_export" not in st.session_state:
    st.session_state.pending_export = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))
        if message.get("query_topic") or message.get("rewritten_question"):
            with st.expander("Agent Trace"):
                if message.get("query_topic"):
                    st.write(f"Classified topic: {message['query_topic']}")
                if message.get("rewritten_question"):
                    st.write(f"Retrieval query: {message['rewritten_question']}")

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if PUBLIC_DEMO_MODE:
            result = get_demo_result(question)
            st.markdown(result["answer"])
            st.caption("Sources: " + ", ".join(result["sources"]))
            with st.expander("Agent Trace"):
                st.write(f"Classified topic: {result['query_topic']}")
                st.write(f"Retrieval query: {result['rewritten_question']}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "query_topic": result["query_topic"],
                    "rewritten_question": result["rewritten_question"],
                }
            )
            st.session_state.pending_export = {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "topic": result.get("query_topic", "general"),
            }
        elif not index_exists():
            st.error("Build the FAISS index first from the sidebar.")
        else:
            with st.spinner("Retrieving context and generating an answer..."):
                try:
                    result = answer_question(question, top_k=top_k)
                    st.markdown(result["answer"])
                    if result["sources"]:
                        st.caption("Sources: " + ", ".join(result["sources"]))
                    with st.expander("Agent Trace"):
                        st.write(f"Classified topic: {result['query_topic']}")
                        st.write(f"Retrieval query: {result['rewritten_question']}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "query_topic": result["query_topic"],
                            "rewritten_question": result["rewritten_question"],
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
    st.subheader("Finalize Answer")
    st.caption(
        "The answer is shown once in the chat above. Approve it here to download "
        "a Markdown copy or send it to the configured Notion MCP-style export."
    )
    st.caption("Sources: " + ", ".join(st.session_state.pending_export["sources"]))

    col1, col2 = st.columns(2)
    with col1:
        markdown_export = build_markdown_export(
            st.session_state.pending_export["question"],
            st.session_state.pending_export["answer"],
            st.session_state.pending_export["sources"],
        )
        st.download_button(
            "Approve and Download Markdown",
            data=markdown_export,
            file_name=slugify_filename(st.session_state.pending_export["question"]),
            mime="text/markdown",
        )
    with col2:
        notion_disabled = (
            PUBLIC_DEMO_MODE
            or not app_config.NOTION_API_KEY
            or not app_config.NOTION_DATABASE_ID
        )
        if st.button("Save Study Card to Notion", disabled=notion_disabled):
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
        if PUBLIC_DEMO_MODE:
            st.caption("Notion export is disabled in public demo mode.")
        elif notion_disabled:
            st.caption(
                "Notion export needs NOTION_API_KEY and NOTION_DATABASE_ID in "
                "Streamlit secrets. This is the MCP-style external tool connector."
            )

    if st.button("Keep Editing / Ask Follow-up"):
        st.info("Ask a follow-up question or request changes in the chat box.")
