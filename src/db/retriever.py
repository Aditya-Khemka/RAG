"""Retriever helpers for the Chroma-backed RAG workflow."""


def get_retriever(
    vectorstore,
    k=3,
):
    """Create a similarity-based retriever for the given vector store."""

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def get_mmr_retriever(
    vectorstore,
    k=3,
    fetch_k=5,
):
    """Create a maximum-marginal-relevance retriever."""

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
        },
    )