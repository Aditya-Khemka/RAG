from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings import get_embedding_model

# Database will be stored here:
DB_DIR = Path(__file__).resolve().parent.parent.parent / "chroma_db"


# Create a vector DB 
def create_vector_store(documents: list[Document]) -> Chroma:
    # Create and persist a Chroma vector database.

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Creating Chroma vector store...")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(DB_DIR), #persist = save on disc 
    )

    print(f"Database created successfully.")
    print(f"Stored {vector_store._collection.count()} documents.")
    print(f"Location: {DB_DIR}")

    return vector_store



# Retreiving a vector DB 
def load_vector_store() -> Chroma:
    # Load an existing Chroma database.

    print("Loading existing Chroma database...")

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedding_model,
    )

    print(f"Loaded {vector_store._collection.count()} documents.")

    return vector_store




def similarity_search(query: str, k: int = 3) -> list[Document]:
    # Perform semantic similarity search on the vector store.

    vector_store = load_vector_store()
    print(f"\nSearching for: '{query}'\n")
    results = vector_store.similarity_search(
        query=query,
        k=k,
    )

    return results


def similarity_search_with_scores(query: str, k: int = 3):
    # Perform semantic similarity search and return the distance score.

    vector_store = load_vector_store()
    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
    )

    return results