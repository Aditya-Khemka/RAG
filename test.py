from src.db.chroma_store import similarity_search, similarity_search_with_scores
from langchain_core.documents import Document
from src.db.chroma_store import create_vector_store

'''
# Create 3 dummy documents and store them in the vector store
docs = [
    Document(page_content="LangChain is a framework."),
    Document(page_content="Chroma stores embeddings."),
    Document(page_content="Azure OpenAI generates embeddings."),
]
create_vector_store(docs)
'''

# Perform a similarity search for the query "What is LangChain?" and retrieve the top 2 results
results = similarity_search_with_scores(
    "What is LangChain?",
    k=2,
)


for i, (doc, score) in enumerate(results, start=1):

    print(f"Result {i}")
    print(f"Distance : {score:.4f}")
    print(f"Content  : {doc.page_content}")
    print(f"Metadata : {doc.metadata}")


'''
Result 1
Distance : 0.6033
Content  : LangChain is a framework.
Metadata : {}
------------------------------------------------------------
Result 2
Distance : 1.5567
Content  : Chroma stores embeddings.
Metadata : {}
------------------------------------------------------------
'''

