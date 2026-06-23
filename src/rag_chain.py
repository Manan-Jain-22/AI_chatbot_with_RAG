from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.config import CHAT_MODEL, DEFAULT_TOP_K, MAX_REWRITE_ATTEMPTS, validate_openai_key
from src.mcp_tools import find_tool, get_agent_tools
from src.vector_store import get_retriever


class RAGState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    answer: str
    attempts: int
    messages: Annotated[list, add_messages]


def get_chat_model(temperature: float = 0) -> ChatOpenAI:
    validate_openai_key()
    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature)


def format_documents(documents: List[Document]) -> str:
    if not documents:
        return "No context was retrieved."

    formatted = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        formatted.append(f"[{i}] Source: {source}{page_label}\n{doc.page_content}")
    return "\n\n".join(formatted)


def collect_sources(documents: List[Document]) -> List[str]:
    seen = set()
    sources = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        label = f"{source} p.{page + 1}" if isinstance(page, int) else source
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return sources


def build_rag_graph(
    top_k: int = DEFAULT_TOP_K,
    tools: Optional[List[BaseTool]] = None,
):
    """
    Build a LangGraph RAG workflow.

    The graph performs query rewriting, optional tool execution, FAISS retrieval,
    and grounded answer generation. The default tool is intentionally local and
    deterministic; MCP-backed tools can be passed in by callers that expose them.
    """
    llm = get_chat_model()
    retriever = get_retriever(top_k=top_k)
    available_tools = tools or get_agent_tools()
    query_tool = find_tool(available_tools, "prepare_retrieval_query")
    response_tool = find_tool(available_tools, "prepare_final_response")

    def rewrite_query(state: RAGState) -> dict:
        prompt = [
            SystemMessage(
                content=(
                    "Rewrite the user's question for semantic retrieval. Keep named "
                    "entities, dates, and technical terms. Return only the rewritten query."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
        response = llm.invoke(prompt)
        rewritten = response.content.strip() or state["question"]
        return {
            "rewritten_question": rewritten,
            "attempts": state.get("attempts", 0) + 1,
            "messages": [response],
        }

    def prepare_query_with_mcp(state: RAGState) -> dict:
        prepared_query = query_tool.invoke(
            {
                "question": state["question"],
                "rewritten_question": state["rewritten_question"],
            }
        )
        return {"rewritten_question": str(prepared_query)}

    def retrieve(state: RAGState) -> dict:
        query = state["rewritten_question"] or state["question"]
        documents = retriever.invoke(query)
        return {"documents": documents}

    def generate_answer(state: RAGState) -> dict:
        context = format_documents(state["documents"])
        prompt = [
            SystemMessage(
                content=(
                    "You are a RAG assistant. Answer using only the provided context. "
                    "If the answer is not in the context, say you do not know. "
                    "Cite sources inline as [1], [2], etc. Keep the answer concise."
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

    def format_answer_with_mcp(state: RAGState) -> dict:
        formatted_answer = response_tool.invoke(
            {
                "answer": state["answer"],
                "sources": ", ".join(collect_sources(state["documents"])),
            }
        )
        return {"answer": str(formatted_answer)}

    graph = StateGraph(RAGState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("prepare_query_with_mcp", prepare_query_with_mcp)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("format_answer_with_mcp", format_answer_with_mcp)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "prepare_query_with_mcp")
    graph.add_edge("prepare_query_with_mcp", "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", "format_answer_with_mcp")
    graph.add_edge("format_answer_with_mcp", END)
    return graph.compile()


def answer_question(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Run the RAG workflow for one question and return answer plus sources."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    graph = build_rag_graph(top_k=top_k)
    result = graph.invoke(
        {
            "question": question.strip(),
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "attempts": 0,
            "messages": [],
        },
        config={"recursion_limit": 10 + MAX_REWRITE_ATTEMPTS},
    )
    return {
        "question": question,
        "rewritten_question": result["rewritten_question"],
        "answer": result["answer"],
        "sources": collect_sources(result["documents"]),
        "documents": result["documents"],
    }


if __name__ == "__main__":
    user_question = input("Ask a question: ")
    response = answer_question(user_question)
    print("\nAnswer:")
    print(response["answer"])
    print("\nSources:")
    for source in response["sources"]:
        print(f"- {source}")
