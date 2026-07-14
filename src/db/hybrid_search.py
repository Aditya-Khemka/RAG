# Add the project root (RAG/) to Python's path.
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import get_embedding_model
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever,BM25Retriever,EnsembleRetriever
from langchain_core.documents import Document

embedding_model = get_embedding_model()

# Documents 
documents = [
    Document(
        page_content=(
            "Product SKU-7742X is our flagship router. "
            "It supports gigabit speeds and advanced QoS features."
        ),
        metadata={"type": "product"}
    ),

    Document(
        page_content=(
            "For network connectivity issues, first check the "
            "ethernet cable and router status lights."
        ),
        metadata={"type": "troubleshooting"}
    ),

    Document(
        page_content=(
            "Error code E_CONN_REFUSED indicates the server "
            "rejected the connection. Check firewall settings."
        ),
        metadata={"type": "error"}
    ),

    Document(
        page_content=(
            "Product SKU-9988Z is a high-performance firewall "
            "appliance designed for enterprise security."
        ),
        metadata={"type": "product"}
    ),

    Document(
        page_content=(
            "To reset your router, hold the reset button for "
            "10 seconds until the LED starts blinking."
        ),
        metadata={"type": "troubleshooting"}
    ),

    Document(
        page_content=(
            "Error code DNS_404 means the DNS server could not "
            "resolve the requested domain name."
        ),
        metadata={"type": "error"}
    ),

    Document(
        page_content=(
            "Product SKU-5566A is a managed switch with "
            "24 gigabit ports and VLAN support."
        ),
        metadata={"type": "product"}
    ),

    Document(
        page_content=(
            "If Wi-Fi performance is poor, try changing the "
            "wireless channel to avoid interference."
        ),
        metadata={"type": "troubleshooting"}
    ),

    Document(
        page_content=(
            "The company refund policy allows customers to "
            "return products within 30 days of purchase."
        ),
        metadata={"type": "policy"}
    ),

    Document(
        page_content=(
            "Employee onboarding requires completing security "
            "training and setting up multi-factor authentication."
        ),
        metadata={"type": "hr"}
    ),

    Document(
        page_content=(
            "Error code AUTH_401 indicates authentication failed "
            "because the API key is invalid or expired."
        ),
        metadata={"type": "error"}
    ),

    Document(
        page_content=(
            "Product SKU-1122B is a compact access point that "
            "supports Wi-Fi 6 and mesh networking."
        ),
        metadata={"type": "product"}
    ),
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="chroma_db",
    collection_name="hybrid_search_collection"
)


vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Vector retriever ready!")

bm25_retriever = BM25Retriever.from_documents(documents,kw_args={"k": 3})
print("BM25 retriever ready!")

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Adjust weights as needed
)   

if __name__ == "__main__":
    # Test queries designed to challenge retrieval systems
    test_queries = [
        'SKU-7742X',                          # 100% - Exact product code
        'E_CONN_REFUSED',                     # 100% - Exact error code
        'AUTH_401 invalid API key',           # 100% - Exact error description

        'router specifications',             # ~70% - Semantic product query
        'server rejected connection',         # ~70% - Semantic error query
        'wifi problems and interference',     # ~70% - Semantic troubleshooting

        'How do I authenticate?',             # ~50% - Indirect authentication question
        'How can I improve my internet?',     # ~50% - Broad troubleshooting query
        'What is your return process?',       # ~50% - Policy question

        'enterprise networking equipment',    # ~30% - Broad product category
        'security requirements for employees',# ~30% - Broad HR query
        'my website is not opening',          # ~30% - Ambiguous problem

        'What are your office timings?',      # 0% - No matching document
        'How do I configure Kubernetes?',     # 0% - Out-of-domain query
        'machine learning algorithms',        # 0% - Completely unrelated
    ]

    results = {}
    for query in test_queries:
        results[query] = ensemble_retriever.invoke(query)
        print(f"Query: '{query}' | Retrieved {len(results[query])} documents")
        print("Retrieved Documents:")
        for doc in results[query]:
            print(f"- {doc.page_content[:100]}... (type: {doc.metadata.get('type')})")
        print("\n" + "-"*80 + "\n")
        
        