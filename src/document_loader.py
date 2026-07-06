from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


def load_pdf():

    # Locate the project root (RAG/)
    project_root = Path(__file__).resolve().parent.parent

    pdf_path = project_root / "data" / "sample.pdf"

    print(f"Looking for PDF at:\n{pdf_path}\n")

    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} not found.")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    load_pdf()