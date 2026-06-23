from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.document_loader import load_documents_from_directory


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits large documents into smaller chunks.

    Why?
    - Embedding entire PDFs is too broad and often impossible.
    - Smaller chunks improve retrieval precision.
    - Overlap helps avoid losing context at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk ids for easier debugging/citation
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


if __name__ == "__main__":
    docs = load_documents_from_directory()
    chunks = split_documents_into_chunks(docs)

    print(f"Original documents/pages: {len(docs)}")
    print(f"Generated chunks: {len(chunks)}")

    print("\nSample chunk metadata:")
    print(chunks[0].metadata)

    print("\nSample chunk text:")
    print(chunks[0].page_content[:500])