"""Standalone Chroma example.

This file is a small low-level example of using Chroma directly.
It is not part of the main RAG pipeline and is kept here as a reference.
This uses the default embedding model
"""

# Add the project root (RAG/) to Python's path.
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Create a local Chroma client.
import chromadb

# chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient()

# Persist the Chroma database to disk in the "chroma_db" directory.
chroma_client.persist_directory = "chroma_db"


'''
# Create a collection, which acts like a table in a traditional database.
collection = chroma_client.get_or_create_collection(name="test_collection")

# Add a few example documents to the collection.
documents = [
    {"id": "doc1", "content": "Hello World"},
    {"id": "doc2", "content": "This is the second document."},
    {"id": "doc3", "content": "This is the third document."},
    {"id": "id1", "content": "This is the apple document."},
    {"id": "id2", "content": "This is the banana document."},
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
    n_results=4,
    # where_document={"$contains": "banana"} #Retrieve documents that contain the word "banana" in their content
)

print(results)
'''

'''
Response : w/o where_documents filter
{
    'ids': [['doc1', 'id2', 'doc3', 'doc2']], 
    'embeddings': None, 
    'documents': [['Hello World', 'This is the banana document.', 'This is the third document.', 'This is the second document.']], 
    'uris': None, 
    'included': ['metadatas', 'documents', 'distances'], 
    'data': None, 
    'metadatas': [[None, None, None, None]], 
    'distances': [[0.0, 1.726358413696289, 1.798677921295166, 1.8283600807189941]]
}


Response with contains 
{
    'ids': [['id2']], 
    'embeddings': None, 
    'documents': [['This is the banana document.']], 
    'uris': None, 
    'included': ['metadatas', 'documents', 'distances'], 
    'data': None, 
    'metadatas': [[None]], 
    'distances': [[1.726358413696289]]
}
'''