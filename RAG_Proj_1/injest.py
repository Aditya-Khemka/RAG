from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

from src.text_splitter import split_documents
from src.db.chroma_store import create_vector_store


LANGCHAIN_KB = """
# LangChain Framework
LangChain is a framework for developing applications powered by language models.

It was created by Harrison Chase in October 2022.

## Core Components
1. Models
2. Prompts
3. Chains
4. Agents
5. Memory

## LangGraph
LangGraph is a library for building stateful, multi-actor applications.

## Pricing
LangChain is open source and free.
LangSmith has paid plans starting at $39/month.
"""

LANGVICTA_KB = """
# LangVicta Platform

LangVicta is an enterprise AI platform for document intelligence and workflow automation.

## Features
1. Document Search
2. RAG Pipelines
3. Multi-agent Workflows
4. Analytics Dashboard

## Pricing
Starter Plan: $50/month
Professional Plan: $200/month
Enterprise Plan: Contact Sales
"""


def ingest():

    docs = [
        Document(
            page_content=LANGCHAIN_KB,
            metadata={"source": "langchain_docs"},
        ),
        Document(
            page_content=LANGVICTA_KB,
            metadata={"source": "langvicta_docs"},
        ),
    ]

    chunks = split_documents(
        docs,
        chunk_size=250,
        chunk_overlap=30,
    )

    vectorstore = create_vector_store(
        documents=chunks,
        persist_directory="./chroma_db",
    )

    print(
        f"Database created with {vectorstore._collection.count()} chunks."
    )


if __name__ == "__main__":
    ingest()