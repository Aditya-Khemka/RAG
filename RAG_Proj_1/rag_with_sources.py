from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.prompts import ChatPromptTemplate

from src.chat_model import *
from src.text_splitter import *
from src.db.chroma_store import *
from src.db.retriever import *



def format_docs(docs):

    formatted = []

    for doc in docs:
        source = doc.metadata.get(
            "source",
            "unknown",
        )

        formatted.append(
            f"""
SOURCE: {source}

{doc.page_content}
"""
        )

    return "\n\n".join(formatted)



def ask_question(question):

    vectorstore = load_vector_store(
        "./chroma_db"
    )

    retriever = get_mmr_retriever(
        vectorstore,
        k=4,
        fetch_k=10,
    )

    docs = retriever.invoke(question)

    print("\nRetrieved Documents:\n")

    for i, doc in enumerate(docs, start=1):
        print(f"----- Document {i} -----")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print()

    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using ONLY the context below.

If the answer is not present,
say:

"I don't have that information in my knowledge base."

At the end include:

Sources Used:
<list>

Context:
{context}

Question:
{question}

Answer:
"""
    )

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    llm = get_chat_model()

    response = llm.invoke(messages)

    return response.content



if __name__ == "__main__":

    questions = [
        "How much does LangVicta cost per month?",
        "Who founded LangVicta?",
        "What is LangGraph?",
        "How much does LangChain cost?",
        "Who created LangChain?",
        "Who created OpenAI?",
        "Who Created LangVicta?",
    ]

    for q in questions:

        print("=" * 80)
        print(f"QUESTION: {q}")

        answer = ask_question(q)

        print("\nFINAL ANSWER:")
        print(answer)