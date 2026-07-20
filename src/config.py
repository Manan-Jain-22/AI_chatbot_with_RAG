import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Project root:
# AI_chatbot_with_RAG/
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")


def _get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read config from environment variables, with Streamlit secrets fallback."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        return str(secret_value) if secret_value else default
    except Exception:
        return default


def _get_bool_setting(name: str, default: str = "false") -> bool:
    value = _get_setting(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Main folders
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_DIR = Path(_get_setting("FAISS_INDEX_DIR", str(INDEX_DIR / "faiss")))


# OpenAI settings
OPENAI_API_KEY = _get_setting("OPENAI_API_KEY")
CHAT_MODEL = _get_setting("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = _get_setting("EMBEDDING_MODEL", "text-embedding-3-small")


# RAG settings
CHUNK_SIZE = int(_get_setting("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_get_setting("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(_get_setting("DEFAULT_TOP_K", "4"))
MAX_REWRITE_ATTEMPTS = int(_get_setting("MAX_REWRITE_ATTEMPTS", "1"))
PUBLIC_DEMO_MODE = _get_bool_setting("PUBLIC_DEMO_MODE", "false")


# Notion settings for optional MCP export
NOTION_API_KEY = _get_setting("NOTION_API_KEY")
NOTION_DATABASE_ID = _get_setting("NOTION_DATABASE_ID")
NOTION_API_VERSION = _get_setting("NOTION_API_VERSION", "2022-06-28")


def validate_openai_key() -> None:
    """
    Checks whether the OpenAI API key exists.
    We call this before making OpenAI embedding/chat requests.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your local .env file, Streamlit "
            "Cloud secrets, or the app sidebar before running the AI pipeline."
        )


def validate_notion_config() -> None:
    """
    Checks whether Notion export credentials exist.
    """
    missing = [
        name
        for name, value in {
            "NOTION_API_KEY": NOTION_API_KEY,
            "NOTION_DATABASE_ID": NOTION_DATABASE_ID,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing Notion configuration: "
            + ", ".join(missing)
            + ". Add these values to your .env file."
        )
