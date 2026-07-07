import os
from pathlib import Path
roject_root = Path(__file__).resolve().parent.parent

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split documents into smaller chunks.

    Args:
        documents: List of LangChain Documents.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of overlapping characters.

    Returns:
        List of chunked Documents.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    print("Text Splitter Summary")
    print("---------------------")
    print(f"Original Documents : {len(documents)}")
    print(f"Generated Chunks   : {len(chunks)}")
    print()

    return chunks


if __name__ == "__main__":

    from src.document_loader import load_pdf

    documents = load_pdf()

    chunks = split_documents(documents)

    print("First Chunk\n")

    print(chunks[0].page_content)

    print("\nMetadata\n")

    print(chunks[0].metadata)