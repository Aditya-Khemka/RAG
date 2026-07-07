# Simple reprsentation  of cheoma db 

import chromadb
chroma_client = chromadb.Client()

# create a collection (collection is like a table in a database)
collection = chroma_client.create_collection(name="test_collection")

# Adding some documents to the collection
documents = [
    {"id": "doc1", "content": "Hello World"},
    {"id": "doc2", "content": "This is the second document."},
    {"id": "doc3", "content": "This is the third document."}
]

for doc in documents:
    collection.upsert(
        ids = [doc["id"]],
        documents = [doc["content"]]
    )
# the syntax for upsert is collection.upsert(ids=[...], documents=[...])


# Now you can query the collection

query = "Hello World"

# Query the collection for the most similar documents to the query
results = collection.query(
    query_texts=[query],
    n_results=2
)
# The syntax for query is collection.query(query_texts=[...], n_results=...)
print(results)