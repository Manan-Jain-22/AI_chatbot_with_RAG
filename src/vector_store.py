from pathlib import Path
from typing import Iterable, Optional

from langchain_core.documents import Document

from src.chunking import split_documents_into_chunks
from src.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    GOOGLE_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    LLM_PROVIDER,
    validate_model_config,
)
from src.document_loader import load_documents_from_directory


def get_embeddings():
    """Return the embedding model used for indexing and retrieval."""
    validate_model_config()

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _import_faiss_vector_store():
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "FAISS support is not installed. Run `pip install -r requirements.txt` "
            "inside the project virtual environment."
        ) from exc
    return FAISS


def build_faiss_index(
    documents: Optional[Iterable[Document]] = None,
    index_dir: Path = FAISS_INDEX_DIR,
):
    """
    Build and persist a FAISS vector store from raw documents.

    If documents are not provided, files are loaded from DATA_DIR and split into
    retrieval-sized chunks before embedding.
    """
    FAISS = _import_faiss_vector_store()
    raw_documents = list(documents) if documents is not None else load_documents_from_directory()
    chunks = split_documents_into_chunks(raw_documents)

    if not chunks:
        raise ValueError("No chunks were produced. Add documents with extractable text.")

    vector_store = FAISS.from_documents(chunks, get_embeddings())
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    return vector_store


def load_faiss_index(index_dir: Path = FAISS_INDEX_DIR):
    """Load a persisted FAISS index from disk."""
    FAISS = _import_faiss_vector_store()
    index_dir = Path(index_dir)

    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_dir}. Build it first with "
            "`python -m src.vector_store`."
        )

    return FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def get_retriever(top_k: int = 4, index_dir: Path = FAISS_INDEX_DIR):
    """Return a retriever backed by the persisted FAISS index."""
    vector_store = load_faiss_index(index_dir=index_dir)
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def index_exists(index_dir: Path = FAISS_INDEX_DIR) -> bool:
    """Check whether the FAISS index files exist."""
    index_dir = Path(index_dir)
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


if __name__ == "__main__":
    store = build_faiss_index()
    print(f"FAISS index saved to: {FAISS_INDEX_DIR}")
    print(f"Indexed vectors: {store.index.ntotal}")
