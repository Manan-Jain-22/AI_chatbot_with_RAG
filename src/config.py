import os
from pathlib import Path
from dotenv import load_dotenv


# Project root:
# AI_chatbot_with_RAG/
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")


# Main folders
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", INDEX_DIR / "faiss"))


# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
MAX_REWRITE_ATTEMPTS = int(os.getenv("MAX_REWRITE_ATTEMPTS", "1"))


# Notion settings for optional MCP export
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_API_VERSION = os.getenv("NOTION_API_VERSION", "2022-06-28")


def validate_openai_key() -> None:
    """
    Checks whether the OpenAI API key exists.
    We call this before making OpenAI embedding/chat requests.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your .env file before running the AI pipeline."
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
