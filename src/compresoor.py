# Add the project root (RAG/) to Python's path
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.chat_model import * 
from src.embeddings import *
from src.db.chroma_store import load_vector_store

from langchain_classic.retrievers import *
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langsmith import traceable

@traceable
def create_vector_store():
    # Create embeddings
    embeddings = get_embedding_model()

    # Load the vector store
    vectorstore = load_vector_store(
        "./chroma_db"
    )

    return vectorstore

@traceable
def contextual_compression ():
    vectorstore = create_vector_store()
    llm = get_chat_model()

    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )

    query = "What frameworks exist for building LLM applications?"

    print(f"\nQuery: {query}")

    # Without compression
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)
    print(f"\n--- WITHOUT Compression (full chunks) ---")
    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content[:150]}...\n")

    # With compression
    # With compression
    compressed_docs = compression_retriever.invoke(query)
    print(f"\n--- WITH Compression (relevant only) ---")
    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content}\n")


if __name__ == "__main__":
    contextual_compression()


