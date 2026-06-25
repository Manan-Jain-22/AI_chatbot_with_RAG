from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from src.config import (
    NOTION_API_KEY,
    NOTION_API_VERSION,
    NOTION_DATABASE_ID,
    validate_notion_config,
)


NOTION_API_BASE_URL = "https://api.notion.com/v1"


def _rich_text(text: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": {"content": text[:2000]}}] if text else []


def _multi_select(values: List[str]) -> List[Dict[str, str]]:
    return [{"name": value[:100]} for value in values if value]


def save_study_card_to_notion_impl(
    question: str,
    answer: str,
    topic: str = "general",
    tags: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Save a finalized RAG answer as a structured Notion study card.

    Expected Notion database properties:
    - Question: title
    - Answer: rich_text
    - Topic: select
    - Tags: multi_select
    - Sources: rich_text
    """
    validate_notion_config()

    url = f"{NOTION_API_BASE_URL}/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_API_VERSION,
        "Content-Type": "application/json",
    }
    source_text = "; ".join(sources or [])
    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Question": {"title": _rich_text(question)},
            "Answer": {"rich_text": _rich_text(answer)},
            "Topic": {"select": {"name": topic or "general"}},
            "Tags": {"multi_select": _multi_select(tags or [])},
            "Sources": {"rich_text": _rich_text(source_text)},
        },
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()
