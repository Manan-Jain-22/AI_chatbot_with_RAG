from pathlib import Path
from re import sub
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.config import BASE_DIR
from src.notion_client import save_study_card_to_notion_impl


mcp = FastMCP(
    "rag-agent-tools",
    instructions=(
        "Tools that support the Computational Linear Algebra RAG assistant "
        "by exporting finalized answers to external study systems."
    ),
)


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
def export_answer_to_markdown(
    question: str,
    answer: str,
    sources: str = "",
    output_dir: str = "exports",
) -> str:
    """Export a human-approved RAG answer as a Markdown document."""
    return export_answer_to_markdown_impl(question, answer, sources, output_dir)


@mcp.tool()
def save_study_card_to_notion(
    question: str,
    answer: str,
    topic: str = "general",
    tags: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
) -> dict:
    """Save a finalized answer as a structured Notion study card."""
    return save_study_card_to_notion_impl(question, answer, topic, tags, sources)


if __name__ == "__main__":
    mcp.run(transport="stdio")
