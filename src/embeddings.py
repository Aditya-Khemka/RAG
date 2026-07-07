import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


def get_embedding_model():
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["DEPLOYMENT_EMBEDDING"],
    )


def embed_text(text: str):
    embedding_model = get_embedding_model()
    embedding = embedding_model.embed_query(text)

    return embedding


if __name__ == "__main__":
    text = "LangChain makes building RAG applications easier."
    embedding = embed_text(text)

    print(f"Embedding dimension: {len(embedding)}")
    print(embedding[:10])