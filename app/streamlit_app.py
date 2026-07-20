import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, DEFAULT_TOP_K, FAISS_INDEX_DIR
import src.config as app_config
from src.document_loader import SUPPORTED_EXTENSIONS
from src.mcp_tools import export_answer_to_markdown, save_study_card_to_notion
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

DEMO_QUESTIONS = [
    "How should I solve an overdetermined system, and why is QR better than normal equations?",
    "When should I use conjugate gradient instead of LU factorization?",
    "What is the difference between Jacobi, Gauss-Seidel, and SOR?",
    "Why are sparse and banded solvers useful for large linear systems?",
    "How does SVD help with ill-conditioned least squares problems?",
]

DEMO_RESPONSES = {
    DEMO_QUESTIONS[0]: {
        "answer": (
            "For an overdetermined system, there are more equations than unknowns, "
            "so an exact solution may not exist. The standard goal is to find the "
            "least squares solution that minimizes the residual norm ||Ax - b||_2.\n\n"
            "QR is usually preferred over normal equations because normal equations "
            "form A^T A, which can square the condition number and amplify numerical "
            "error. With QR, A = QR and the least squares problem becomes Rx = Q^T b, "
            "where Q has orthonormal columns and R is triangular. That keeps the "
            "computation more stable for many practical problems."
        ),
        "query_topic": "least_squares",
        "rewritten_question": "overdetermined linear system least squares QR factorization normal equations conditioning residual norm",
    },
    DEMO_QUESTIONS[1]: {
        "answer": (
            "Use conjugate gradient when the matrix is large, sparse, and symmetric "
            "positive definite. In that setting, explicitly factoring A with LU can "
            "be expensive in both memory and time, while conjugate gradient only needs "
            "matrix-vector products and a few vector operations per iteration.\n\n"
            "LU is still a good choice for moderate dense systems or when you need to "
            "solve many systems with the same matrix. Conjugate gradient is better "
            "when preserving sparsity and avoiding factorization are more important."
        ),
        "query_topic": "conjugate_gradient",
        "rewritten_question": "conjugate gradient versus LU factorization large sparse symmetric positive definite systems",
    },
    DEMO_QUESTIONS[2]: {
        "answer": (
            "Jacobi, Gauss-Seidel, and SOR are stationary iterative methods for linear "
            "systems. Jacobi updates every variable using values from the previous "
            "iteration, so updates are independent within one sweep.\n\n"
            "Gauss-Seidel uses newly computed values immediately, which often improves "
            "convergence compared with Jacobi. SOR extends Gauss-Seidel by adding a "
            "relaxation parameter omega. A good omega can accelerate convergence, while "
            "a poor omega can slow the method or make it fail to converge."
        ),
        "query_topic": "iterative_methods",
        "rewritten_question": "Jacobi Gauss-Seidel SOR stationary iterative methods relaxation parameter convergence",
    },
    DEMO_QUESTIONS[3]: {
        "answer": (
            "Sparse and banded solvers are useful because they exploit matrix structure. "
            "A sparse matrix has many zeros, so storing and operating on every entry "
            "wastes memory and computation.\n\n"
            "A banded matrix has nonzero entries concentrated near the main diagonal. "
            "Specialized solvers operate mainly inside that band, reducing both storage "
            "and arithmetic cost. For tridiagonal systems, this can reduce the solve to "
            "linear time in the number of unknowns."
        ),
        "query_topic": "sparse_banded",
        "rewritten_question": "sparse matrix banded matrix tridiagonal solver memory arithmetic cost large linear systems",
    },
    DEMO_QUESTIONS[4]: {
        "answer": (
            "SVD helps with ill-conditioned least squares problems because it exposes "
            "the singular values of A. Small singular values reveal directions where "
            "the data provides weak information and where the solution can become very "
            "sensitive to noise.\n\n"
            "Compared with normal equations, SVD gives more diagnostic information: "
            "rank, numerical rank, and unstable directions. That makes it useful for "
            "rank-deficient or nearly rank-deficient least squares problems."
        ),
        "query_topic": "svd",
        "rewritten_question": "SVD ill-conditioned least squares singular values numerical rank rank deficient systems",
    },
}


def get_demo_result(question: str) -> dict:
    response = DEMO_RESPONSES.get(question)
    if response is None:
        return {
            "answer": (
                "This hosted portfolio demo is running in cost-safe mode, so it only "
                "answers the curated demo questions in the sidebar. The full project "
                "supports arbitrary uploaded documents and provider-backed FAISS RAG when "
                "`PUBLIC_DEMO_MODE` is disabled and a live provider key is configured."
            ),
            "query_topic": "demo_mode",
            "rewritten_question": "cost safe hosted portfolio demo without external model API calls",
            "sources": ["data/demo_computational_linear_algebra_notes.md"],
        }

    return {
        "question": question,
        "answer": response["answer"],
        "query_topic": response["query_topic"],
        "rewritten_question": response["rewritten_question"],
        "sources": ["data/demo_computational_linear_algebra_notes.md"],
        "documents": [],
    }


if PUBLIC_DEMO_MODE:
    st.info(
        "Public demo mode is on. Demo questions work without OpenAI calls, so this "
        "portfolio app will not consume API credits."
    )

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
            st.warning("Live provider key not detected in Streamlit secrets.")

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

    st.divider()
    st.header("Demo Questions")
    st.caption("Use these to quickly show semantic retrieval and topic-aware query rewriting.")
    for demo_question in DEMO_QUESTIONS:
        if st.button(demo_question, use_container_width=True):
            st.session_state.selected_demo_question = demo_question

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
if "selected_demo_question" not in st.session_state:
    st.session_state.selected_demo_question = None

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
if st.session_state.selected_demo_question and not question:
    question = st.session_state.selected_demo_question
    st.session_state.selected_demo_question = None

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
        notion_disabled = PUBLIC_DEMO_MODE
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
        if notion_disabled:
            st.caption("Notion export is disabled in public demo mode.")

    if st.button("Keep Editing / Ask Follow-up"):
        st.info("Ask a follow-up question or request changes in the chat box.")
