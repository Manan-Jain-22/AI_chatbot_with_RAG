from __future__ import annotations

from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.config import (
    CHAT_MODEL,
    DEFAULT_TOP_K,
    GEMINI_CHAT_MODEL,
    LLM_PROVIDER,
    MAX_REWRITE_ATTEMPTS,
    validate_model_config,
)
from src.course_topics import TOPIC_KEYWORDS, expand_query_with_topic_terms, infer_topic
from src.vector_store import get_retriever


class RAGState(TypedDict):
    question: str
    query_topic: str
    rewritten_question: str
    documents: List[Document]
    selected_documents: List[Document]
    answer: str
    attempts: int
    messages: Annotated[list, add_messages]


def get_chat_model(temperature: float = 0):
    validate_model_config()

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=temperature)

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=CHAT_MODEL, temperature=temperature)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def format_documents(documents: List[Document]) -> str:
    if not documents:
        return "No context was retrieved."

    formatted = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        topic = doc.metadata.get("topic", "general")
        section = doc.metadata.get("section_heading", "")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        section_label = f", section: {section}" if section else ""
        formatted.append(
            f"[{i}] Source: {source}{page_label}, topic: {topic}{section_label}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted)


def collect_sources(documents: List[Document]) -> List[str]:
    seen = set()
    sources = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        section = doc.metadata.get("section_heading", "")
        page_part = f" p.{page + 1}" if isinstance(page, int) else ""
        section_part = f" - {section}" if section else ""
        label = f"{source}{page_part}{section_part}"
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return sources


def build_rag_graph(top_k: int = DEFAULT_TOP_K):
    """
    Build a LangGraph RAG workflow.

    The graph performs query classification, query rewriting, FAISS retrieval,
    context selection, and grounded answer generation.
    """
    llm = get_chat_model()
    retriever = get_retriever(top_k=top_k + 4)

    def classify_query(state: RAGState) -> dict:
        topics = ", ".join(TOPIC_KEYWORDS)
        prompt = [
            SystemMessage(
                content=(
                    "Classify the computational linear algebra topic for the question. "
                    f"Choose one of: {topics}. Return only the topic name."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
        response = llm.invoke(prompt)
        candidate = response.content.strip().lower()
        topic = candidate if candidate in TOPIC_KEYWORDS else infer_topic(state["question"])
        return {"query_topic": topic, "messages": [response]}

    def rewrite_query(state: RAGState) -> dict:
        prompt = [
            SystemMessage(
                content=(
                    "Rewrite the student's computational linear algebra question for "
                    "semantic retrieval. Expand vague wording into likely course terms. "
                    "Do not answer the question. Return only the retrieval query."
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['query_topic']}\n"
                    f"Question: {state['question']}"
                )
            ),
        ]
        response = llm.invoke(prompt)
        rewritten = response.content.strip() or state["question"]
        expanded = expand_query_with_topic_terms(rewritten, state["query_topic"])
        return {
            "rewritten_question": expanded,
            "attempts": state.get("attempts", 0) + 1,
            "messages": [response],
        }

    def retrieve(state: RAGState) -> dict:
        query = state["rewritten_question"] or state["question"]
        documents = retriever.invoke(query)
        return {"documents": documents}

    def select_context(state: RAGState) -> dict:
        topic = state["query_topic"]
        topic_matches = [
            document
            for document in state["documents"]
            if document.metadata.get("topic") == topic
        ]
        selected = (topic_matches + state["documents"])[:top_k]
        return {"selected_documents": selected}

    def generate_answer(state: RAGState) -> dict:
        context = format_documents(state["selected_documents"])
        prompt = [
            SystemMessage(
                content=(
                    "You are a Computational Linear Algebra study assistant. "
                    "Answer using only the provided course context. "
                    "If the answer is not in the context, say you do not know. "
                    "Explain which numerical method applies and why. "
                    "Cite sources inline as [1], [2], etc."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n\n"
                    f"Retrieval query: {state['rewritten_question']}\n\n"
                    f"Context:\n{context}"
                )
            ),
        ]
        response = llm.invoke(prompt)
        return {"answer": response.content, "messages": [response]}

    graph = StateGraph(RAGState)
    graph.add_node("classify_query", classify_query)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("select_context", select_context)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("classify_query")
    graph.add_edge("classify_query", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "select_context")
    graph.add_edge("select_context", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


def answer_question(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Run the RAG workflow for one question and return answer plus sources."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    graph = build_rag_graph(top_k=top_k)
    result = graph.invoke(
        {
            "question": question.strip(),
            "query_topic": "",
            "rewritten_question": "",
            "documents": [],
            "selected_documents": [],
            "answer": "",
            "attempts": 0,
            "messages": [],
        },
        config={"recursion_limit": 10 + MAX_REWRITE_ATTEMPTS},
    )
    return {
        "question": question,
        "query_topic": result["query_topic"],
        "rewritten_question": result["rewritten_question"],
        "answer": result["answer"],
        "sources": collect_sources(result["selected_documents"]),
        "documents": result["selected_documents"],
    }


if __name__ == "__main__":
    user_question = input("Ask a question: ")
    response = answer_question(user_question)
    print("\nAnswer:")
    print(response["answer"])
    print("\nSources:")
    for source in response["sources"]:
        print(f"- {source}")
