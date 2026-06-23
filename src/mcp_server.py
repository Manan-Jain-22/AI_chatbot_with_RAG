from pathlib import Path
from re import sub
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.config import BASE_DIR
from src.whatsapp_client import send_whatsapp_text_message_impl


mcp = FastMCP(
    "rag-agent-tools",
    instructions=(
        "Tools that support the RAG agent by preparing retrieval queries, "
        "formatting grounded answers, and exporting approved responses."
    ),
)


def _clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def prepare_retrieval_query_impl(question: str, rewritten_question: Optional[str] = None) -> str:
    """
    Convert a conversational user question into a compact semantic-search query.
    """
    candidate = rewritten_question or question
    cleaned = _clean_text(candidate)
    cleaned = sub(r"\b(please|could you|can you|tell me|explain)\b", "", cleaned, flags=2)
    return _clean_text(cleaned) or _clean_text(question)


def prepare_final_response_impl(answer: str, sources: str = "") -> str:
    """
    Normalize a grounded RAG answer and append source labels for review/export.
    """
    cleaned_answer = answer.strip()
    cleaned_sources = _clean_text(sources)

    if not cleaned_answer:
        return "I do not know based on the retrieved context."

    if cleaned_sources and "Sources:" not in cleaned_answer:
        return f"{cleaned_answer}\n\nSources: {cleaned_sources}"
    return cleaned_answer


def export_answer_to_markdown_impl(
    question: str,
    answer: str,
    sources: str = "",
    output_dir: str = "exports",
) -> str:
    """
    Export an approved answer to a local Markdown document.
    """
    target_dir = BASE_DIR / output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    slug = sub(r"[^a-zA-Z0-9]+", "-", question.lower()).strip("-")[:60] or "rag-answer"
    target_path = target_dir / f"{slug}.md"
    content = (
        f"# RAG Answer\n\n"
        f"## Question\n\n{question.strip()}\n\n"
        f"## Answer\n\n{answer.strip()}\n\n"
        f"## Sources\n\n{sources.strip() or 'No sources recorded.'}\n"
    )
    target_path.write_text(content, encoding="utf-8")
    return str(target_path)


@mcp.tool()
def prepare_retrieval_query(question: str, rewritten_question: Optional[str] = None) -> str:
    """Prepare a concise query for FAISS semantic retrieval."""
    return prepare_retrieval_query_impl(question, rewritten_question)


@mcp.tool()
def prepare_final_response(answer: str, sources: str = "") -> str:
    """Format the draft RAG answer before returning it to the user."""
    return prepare_final_response_impl(answer, sources)


@mcp.tool()
def export_answer_to_markdown(
    question: str,
    answer: str,
    sources: str = "",
    output_dir: str = "exports",
) -> str:
    """Export a human-approved RAG answer as a Markdown document."""
    return export_answer_to_markdown_impl(question, answer, sources, output_dir)


@mcp.tool()
def send_whatsapp_text_message(
    recipient_phone_number: str,
    message: str,
    preview_url: bool = False,
) -> dict:
    """Send a text message through Meta WhatsApp Cloud API."""
    return send_whatsapp_text_message_impl(recipient_phone_number, message, preview_url)


if __name__ == "__main__":
    mcp.run(transport="stdio")
