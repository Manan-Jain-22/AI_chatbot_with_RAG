from __future__ import annotations

import asyncio
import sys
from typing import List

from langchain_core.tools import BaseTool, tool

from src.mcp_server import (
    export_answer_to_markdown_impl,
    prepare_final_response_impl,
    prepare_retrieval_query_impl,
)


@tool
def prepare_retrieval_query(question: str, rewritten_question: str = "") -> str:
    """Prepare a concise query for FAISS semantic retrieval."""
    return prepare_retrieval_query_impl(question, rewritten_question or None)


@tool
def prepare_final_response(answer: str, sources: str = "") -> str:
    """Format the draft RAG answer before returning it to the user."""
    return prepare_final_response_impl(answer, sources)


@tool
def export_answer_to_markdown(
    question: str,
    answer: str,
    sources: str = "",
    output_dir: str = "exports",
) -> str:
    """Export a human-approved RAG answer as a Markdown document."""
    return export_answer_to_markdown_impl(question, answer, sources, output_dir)


def get_local_mcp_compatible_tools() -> List[BaseTool]:
    """
    Return LangChain tools with the same schemas as the MCP server tools.

    These keep the app runnable in environments where spawning a stdio MCP
    server is not available, while the real MCP server remains in src.mcp_server.
    """
    return [prepare_retrieval_query, prepare_final_response, export_answer_to_markdown]


async def load_mcp_tools() -> List[BaseTool]:
    """Load tools from the local MCP server through the LangChain MCP adapter."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "rag_agent_tools": {
                "command": sys.executable,
                "args": ["-m", "src.mcp_server"],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools(server_name="rag_agent_tools")


def get_agent_tools(prefer_mcp: bool = True) -> List[BaseTool]:
    """
    Prefer real MCP tools, then fall back to same-schema local tools.
    """
    if not prefer_mcp:
        return get_local_mcp_compatible_tools()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(load_mcp_tools())
        except Exception:
            return get_local_mcp_compatible_tools()

    return get_local_mcp_compatible_tools()


def find_tool(tools: List[BaseTool], name: str) -> BaseTool:
    for candidate in tools:
        if candidate.name == name or candidate.name.endswith(f"_{name}"):
            return candidate
    raise ValueError(f"Tool not found: {name}")
