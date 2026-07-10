"""Entry point for the document ingestion pipeline.

This module connects the loading, chunking, and vector-store creation steps
into one simple workflow that can be run from the command line.
"""

from pathlib import Path

from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.db.chroma_store import create_vector_store


def ingest(
    pdf_path,
    persist_directory="./chroma_db",
):
    """Load a PDF, split it into chunks, and create a vector store."""

    # Step 1: read the PDF into LangChain document objects.
    documents = load_pdf(pdf_path)

    # Step 2: split long documents into smaller reusable chunks.
    chunks = split_documents(documents)

    # Step 3: embed the chunks and store them in Chroma.
    vectorstore = create_vector_store(
        chunks,
        persist_directory=persist_directory,
    )

    return vectorstore


if __name__ == "__main__":
    # Resolve the project root so the sample PDF can be found reliably.
    project_root = Path(__file__).resolve().parent.parent

    pdf_path = project_root / "data" / "sample.pdf"

    vectorstore = ingest(pdf_path)

    print(f"Stored {vectorstore._collection.count()} chunks.")