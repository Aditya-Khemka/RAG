from pathlib import Path
import sys

# Add the project root (RAG/) to Python's path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.document_loader import load_pdf
from src.text_splitter import split_documents


documents = load_pdf()

chunks = split_documents(documents)

print("=" * 80)

print(f"Documents : {len(documents)}")

print(f"Chunks    : {len(chunks)}")

print("=" * 80)

print("\nFirst Chunk\n")

print(chunks[0].page_content)

print("\nMetadata")

print(chunks[0].metadata)