import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.document_loader import load_documents_from_directory


HEADING_PATTERN = re.compile(
    r"^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 ,:/\\()\-]{4,80})$",
    re.MULTILINE,
)


def _extract_section_heading(text: str) -> Optional[str]:
    match = HEADING_PATTERN.search(text.strip())
    if not match:
        return None
    return match.group(1).lstrip("#").strip()


def _split_document_by_sections(document: Document) -> List[Document]:
    """
    Preserve nearby headings for math notes before character chunking.
    """
    text = document.page_content
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        heading = _extract_section_heading(text)
        document.metadata["section_heading"] = heading or document.metadata.get("section_heading", "")
        return [document]

    sections: List[Document] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        metadata = dict(document.metadata)
        metadata["section_heading"] = match.group(1).lstrip("#").strip()
        sections.append(Document(page_content=section_text, metadata=metadata))

    return sections or [document]


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

    section_documents: List[Document] = []
    for document in documents:
        section_documents.extend(_split_document_by_sections(document))

    chunks = splitter.split_documents(section_documents)

    # Add chunk ids for easier debugging/citation
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata.setdefault("section_heading", _extract_section_heading(chunk.page_content) or "")

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
