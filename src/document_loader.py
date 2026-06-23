from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.config import DATA_DIR


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_single_document(file_path: Path) -> List[Document]:
    """
    Loads one file and returns a list of LangChain Document objects.

    A Document has:
    - page_content: the actual text
    - metadata: information like source file name, page number, file type
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        loader = TextLoader(str(file_path), encoding="utf-8")

    documents = loader.load()

    # Add consistent metadata for citations and debugging
    for doc in documents:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_path"] = str(file_path)
        doc.metadata["file_type"] = extension

    return documents


def load_documents_from_directory(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    Loads all supported files from the data directory.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_documents: List[Document] = []

    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents = load_single_document(file_path)
            all_documents.extend(documents)

    if not all_documents:
        raise ValueError(
            f"No supported documents found in {data_dir}. "
            "Add .pdf, .txt, or .md files to the data folder."
        )

    return all_documents


if __name__ == "__main__":
    docs = load_documents_from_directory()
    print(f"Loaded {len(docs)} document pages/files.")

    print("\nSample metadata:")
    print(docs[0].metadata)

    print("\nSample text:")
    print(docs[0].page_content[:500])
