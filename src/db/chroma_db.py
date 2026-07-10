"""Standalone Chroma example.

This file is a small low-level example of using Chroma directly.
It is not part of the main RAG pipeline and is kept here as a reference.
"""

# Create a local Chroma client.
import chromadb

chroma_client = chromadb.Client()

# Add the project root (RAG/) to Python's path.
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Create a collection, which acts like a table in a traditional database.
collection = chroma_client.create_collection(name="test_collection")

# Add a few example documents to the collection.
documents = [
    {"id": "doc1", "content": "Hello World"},
    {"id": "doc2", "content": "This is the second document."},
    {"id": "doc3", "content": "This is the third document."},
]

for doc in documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["content"]],
    )

# Query the collection for the most similar documents.
query = "Hello World"
results = collection.query(
    query_texts=[query],
    n_results=2,
)

print(results)