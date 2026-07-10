# Add the project root (RAG/) to Python's path
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.db.chroma_store import (
    load_vector_store,
    similarity_search,
)

vectorstore = load_vector_store(
    "./chroma_db"
)

results = similarity_search(
    vectorstore,
    "What is this document about?",
    k=3,
)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print(doc.page_content[:500])