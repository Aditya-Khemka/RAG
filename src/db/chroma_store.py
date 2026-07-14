"""Helpers for creating, loading, and querying a Chroma vector store."""

from langchain_chroma import Chroma
from src.embeddings import get_embedding_model


def create_vector_store(
    documents,
    persist_directory=None,
):
    """Create a Chroma vector store from a list of documents."""

    # Build the embedding model once and use it for all chunks.
    embedding_model = get_embedding_model()

    # Store the documents in Chroma and persist them to disk if requested.
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )

    return vectorstore


def load_vector_store(
    persist_directory,
):
    """Load an existing Chroma vector store (python object of the db) from disk."""

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )


def similarity_search(
    vectorstore,
    query,
    k=3,
):
    """Return the top-k documents most similar to a query."""

    return vectorstore.similarity_search(
        query,
        k=k,
    )


def similarity_search_with_scores(
    vectorstore,
    query,
    k=3,
):
    """Return similar documents together with their distance score."""

    return vectorstore.similarity_search_with_score(
        query,
        k=k,
    )


def metadata_search(
    vectorstore,
    query,
    filter_criteria,
    k=3,
):
    """Return similar documents while applying a metadata filter."""

    return vectorstore.similarity_search(
        query,
        k=k,
        filter=filter_criteria,
    )