from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.chat_model import *
from src.db.chroma_store import *
from src.db.retriever import *
from src.schemas import *



def format_docs(docs):
    """
    Convert retrieved Document objects into a plain text
    context string that can be inserted into the prompt.
    """

    formatted = []

    for doc in docs:
        source = doc.metadata.get("source","unknown")

        formatted.append(
            f"""
SOURCE: {source}

{doc.page_content}
"""
        )

    return "\n\n".join(formatted)



def ask_question(question: str) -> RAGResponse:
    """
    Ask a question against the vector database.

    Returns:
        RAGResponse
    """

    # Load existing Chroma database.

    vectorstore = load_vector_store("./chroma_db")

    # Create retriever.
    retriever = get_mmr_retriever(
        vectorstore,
        k=4,
        fetch_k=10,
    )


    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I don't have that information in my knowledge base."

Provide:

1. answer
2. confidence (0 to 1)
3. sources_used
4. follow_up

Context:
{context}

Question:
{question}
"""
    )

    # -------------------------------------------------
    # Structured LLM
    # Output will automatically become a RAGResponse
    # object instead of plain text.
    # -------------------------------------------------
    structured_llm = get_structured_chat_model()

    # -------------------------------------------------
    # LCEL RAG Chain
    #
    # Input Question
    #        │
    #        ├──> Retriever
    #        │         │
    #        │         ▼
    #        │   Retrieved Docs
    #        │         │
    #        │         ▼
    #        │    format_docs
    #        │
    #        └──> Original Question
    #
    # Both are passed into the prompt.
    # -------------------------------------------------
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | structured_llm
    )

    result = rag_chain.invoke(question)

    return result


if __name__ == "__main__":

    questions = [
        "How much does LangVicta cost per month?",
        "Who founded LangVicta?",
        "What is LangGraph?",
        "How much does LangChain cost?",
        "Who created LangChain?",
        "Who created OpenAI?",
        "Who created LangVicta?",
    ]

    for question in questions:

        print("=" * 80)
        print(f"QUESTION:\n{question}")

        result = ask_question(question)

        print("\nANSWER:")
        print(result.answer)

        print("\nCONFIDENCE:")
        print(result.confidence)

        print("\nSOURCES:")
        print(result.sources_used)

        print("\nFOLLOW-UP:")
        print(result.follow_up)

        print("=" * 80)
        print()