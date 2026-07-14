"""
Flow:

Question
    ↓
Retriever
    ↓
Relevant Documents
    ↓
Prompt + Context
    ↓
LLM
    ↓
Final Answer
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.chat_model import *
from src.text_splitter import *
from src.db.chroma_store import *


# Dummy knowledge base.
LANGCHAIN_KB = """
# LangChain Framework
LangChain is a framework for developing applications powered by language models.

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


# Build a vector database from our knowledge base.
def create_kb():

    # Create a Document object from the knowledge base text.
    docs = [
        Document(
            page_content=LANGCHAIN_KB,
            metadata={
                "source": "langchain_docs"
            },
        ),
        Document(
            page_content=LANGVICTA_KB,
            metadata={
                "source": "langvicta_docs"
            },
        ),
    ]

    # Split the document into smaller chunks
    chunks = split_documents(
        docs,
        chunk_size=250,
        chunk_overlap=30,
    )

    # Create a Chroma vector store from the chunks.
    vector_store = create_vector_store(chunks)

    return vector_store


# --------------------------------------------------
# Convert:
#
# [Document(...), Document(...)]
#
# into:
#
# "doc1 text\n\ndoc2 text"
#
# LLMs cannot consume Document objects directly.
# They need plain text.
# --------------------------------------------------
def format_docs(docs):

    formatted = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")

        formatted.append(
            f"""
SOURCE: {source}

{doc.page_content}
"""
        )

    return "\n\n".join(formatted)


# Main RAG function.
def ask_question(question):

    # Step 1: Create the vector database.
    vector_store = create_kb()

    # Step 2: Create a retriever.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    # Step 3: Retrieve relevant chunks.
    docs = retriever.invoke(question)

    # Step 3.1 : Extract all unique sources used by the retriever.
    sources = sorted(
        {
            doc.metadata.get("source", "unknown")
            for doc in docs
        }
    )

    for i, doc in enumerate(docs, start=1):
        print(f"----- Document {i} -----")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print()

    # Step 4: Convert retrieved docs into plain text.
    context = format_docs(docs)

    # --------------------------------------------------
    # Step 5: Build the prompt.
    #
    # We tell the model:
    # - Use ONLY the provided context.
    # - Answer the question.
    #
    # This reduces hallucinations.
    # --------------------------------------------------
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using ONLY the context below.

At the end, mention which sources were used.

If the answer cannot be found in the context,
say:

"I don't have that information in my knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # Fill the template.
    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # Step 6: Prepare the LLM.
    llm = get_chat_model()

    # Step 7: Send the prompt to the LLM.
    response = llm.invoke(messages)

    # Step 8: Return both answer and sources.
    return {
        "answer": response.content,
        "sources": sources,
    }


# --------------------------------------------------
# Entry point.
# --------------------------------------------------
if __name__ == "__main__":

    questions = [
        "How much does LangVicta cost per month?",
        "Who founded LangVicta?",
        "What is LangGraph?",
        "Who created OpenAI?",
        "How much does LangChain cost per month? Who created LangChain?",
    ]

    for question in questions:
        print("=" * 80)
        print(f"QUESTION: {question}")

        result = ask_question(question)

        print()
        print("FINAL ANSWER:")
        print(result["answer"])

        print()
        print("SOURCES:")
        print(result["sources"])
        print("=" * 80)