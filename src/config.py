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


# WhatsApp Cloud API settings
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "rag-chatbot-verify-token")
WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v20.0")


def validate_openai_key() -> None:
    """
    Checks whether the OpenAI API key exists.
    We call this before making OpenAI embedding/chat requests.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your .env file before running the AI pipeline."
        )


def validate_whatsapp_config() -> None:
    """
    Checks whether WhatsApp Cloud API credentials exist.
    """
    missing = [
        name
        for name, value in {
            "WHATSAPP_ACCESS_TOKEN": WHATSAPP_ACCESS_TOKEN,
            "WHATSAPP_PHONE_NUMBER_ID": WHATSAPP_PHONE_NUMBER_ID,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing WhatsApp configuration: "
            + ", ".join(missing)
            + ". Add these values to your .env file."
        )
