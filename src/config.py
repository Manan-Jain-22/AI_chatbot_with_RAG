import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


# Project root:
# AI_chatbot_with_RAG/
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")


def _read_streamlit_secret(name: str) -> Optional[str]:
    """Read Streamlit secrets from top-level or common nested sections."""
    try:
        import streamlit as st

        candidates = {name, name.lower(), name.upper()}

        for candidate in candidates:
            value = st.secrets.get(candidate)
            if value:
                return str(value).strip()

        for section_name in ("general", "default", "secrets"):
            section: Any = st.secrets.get(section_name)
            if hasattr(section, "get"):
                for candidate in candidates:
                    value = section.get(candidate)
                    if value:
                        return str(value).strip()
    except Exception:
        return None

    return None


def get_visible_secret_names() -> list[str]:
    """Return secret key names for diagnostics without exposing secret values."""
    try:
        import streamlit as st

        names = []
        for key, value in st.secrets.items():
            if hasattr(value, "keys"):
                names.append(f"[{key}]")
                names.extend(f"{key}.{nested_key}" for nested_key in value.keys())
            else:
                names.append(str(key))
        return sorted(names)
    except Exception:
        return []


def _get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read config from Streamlit secrets first, then environment variables."""
    secret_value = _read_streamlit_secret(name)
    if secret_value:
        return secret_value

    value = os.getenv(name)
    return value if value else default


def _get_bool_setting(name: str, default: str = "false") -> bool:
    value = _get_setting(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Main folders
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_DIR = Path(_get_setting("FAISS_INDEX_DIR", str(INDEX_DIR / "faiss")))


# OpenAI settings
LLM_PROVIDER = _get_setting("LLM_PROVIDER", "gemini").lower()
OPENAI_API_KEY = _get_setting("OPENAI_API_KEY")
CHAT_MODEL = _get_setting("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = _get_setting("EMBEDDING_MODEL", "text-embedding-3-small")


# Gemini settings
GOOGLE_API_KEY = (
    _get_setting("GOOGLE_API_KEY")
    or _get_setting("GEMINI_API_KEY")
    or _get_setting("GOOGLE_AI_API_KEY")
)
GEMINI_CHAT_MODEL = _get_setting("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = _get_setting("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")


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


def validate_model_config() -> None:
    """Validate credentials for the configured LLM/embedding provider."""
    if LLM_PROVIDER == "openai":
        validate_openai_key()
        return

    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError(
                "Gemini API key is missing. Add GOOGLE_API_KEY or GEMINI_API_KEY "
                "to Streamlit Cloud secrets before running live Gemini RAG."
            )
        return

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Use 'gemini' or 'openai'."
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
