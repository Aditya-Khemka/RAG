"""Embedding helpers for the RAG pipeline.

This module wraps the Azure embedding client so the rest of the project can
create embeddings without repeating the Azure configuration each time.
"""

import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langsmith import traceable

# Load environment variables from the local .env file.
load_dotenv()

@traceable(name="get_embedding_model")
def get_embedding_model():
    """Create and return the Azure embedding model client."""

    print("[embeddings / get_embedding_model] Loading embedding model from Azure OpenAI...")

    # Build the Azure OpenAI embeddings client using environment variables.
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["DEPLOYMENT_EMBEDDING"],
    )


@traceable(name="embed_text")
def embed_text(text: str):
    """Convert a single text string into an embedding vector."""

    print(f"[embeddings / embed_text] Creating embedding for text: {text}")
    embedding_model = get_embedding_model()
    embedding = embedding_model.embed_query(text)

    return embedding


if __name__ == "__main__":
    text = "LangChain makes building RAG applications easier."
    embedding = embed_text(text)

    print(f"[embeddings / __main__] Embedding dimension: {len(embedding)}")
    print(embedding[:10])