from pathlib import Path
import sys

# Add the project root (RAG/) to Python's path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from document_loader import load_pdf
from text_splitter import split_documents
from db.chroma_store import create_vector_store


def ingest():

    print("=" * 60)
    print("Starting document ingestion...")
    print("=" * 60)

    # Step 1: Load documents
    documents = load_pdf()

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create Chroma DB
    vector_store = create_vector_store(chunks)

    print("\nIngestion completed successfully!")

    return vector_store


if __name__ == "__main__":
    ingest()