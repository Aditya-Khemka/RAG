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
KNOWLEDGE_BASE = """
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
LangSmith has paid plans starting at $39/month while LangGraph is free for now. 
LangVicta is a new product that is currently being offered to a select few for a cost of $200/month.
"""


# Build a vector database from our knowledge base.
def create_kb():

    # Create a Document object from the knowledge base text.
    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata={
            "source": "langchain_docs"
        },
    )

    # Split the document into smaller chunks 
    chunks = split_documents(
        [doc],
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
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# Main RAG function.
def ask_question(question):

    print(f"QUESTION:\n{question}")
    print("=" * 80)

    # Step 1: Create the vector database.
    vector_store = create_kb()

    # Step 2: Create a retriever.
    #
    # search_type="similarity"
    # means:
    # "Find chunks whose embeddings are closest to the embedding of the question."
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )


    # Step 3: Retrieve relevant chunks.
    docs = retriever.invoke(question)

    print("\nRetrieved Documents:\n")

    for i, doc in enumerate(docs, start=1):
        print(f"----- Document {i} -----")
        print(doc.page_content)
        print()

    # Step 4: Convert retrieved docs into plain text.
    context = format_docs(docs)

    print("=" * 80)
    print(f"CONTEXT SENT TO THE LLM: \n {context}")
    print("=" * 80)
    print()

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

    print("=" * 80)
    print("FINAL PROMPT:")
    print("=" * 80)
    print(messages)
    print()

    # Step 6: Prepare the LLM.
    llm = get_chat_model()

    # Step 7: Send the prompt to the LLM.
    response = llm.invoke(messages)

    return response.content


# --------------------------------------------------
# Entry point.
# --------------------------------------------------
if __name__ == "__main__":

    question = "How much does LangVicta cost per month? Who is the creator of LangVicta?"

    answer = ask_question(question)

    print("=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(answer)