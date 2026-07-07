from langchain_core.documents import Document

from src.db.chroma_store import load_vector_store


# Rectriever = vector store + search method (similarity, mmr, etc)
def get_retriever(
    search_type: str = "similarity",
    k: int = 2,
):
    # Returns a configured Chroma retriever.

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

    return retriever


def retrieve_documents(query: str, k: int = 2) -> list[Document]:
    # Retrieve the most relevant documents.

    retriever = get_retriever(k=k)
    return retriever.invoke(query)