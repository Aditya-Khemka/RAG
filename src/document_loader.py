from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


def load_pdf(file_name: str = "sample.pdf") -> list[Document]:
    """
    Load a PDF from the project's data directory.

    Args:
        file_name: Name of the PDF inside the data folder.

    Returns:
        List of LangChain Document objects (one per page).
    """

    # Project Root
    project_root = Path(__file__).resolve().parent.parent

    # Path to PDF
    pdf_path = project_root / "data" / file_name

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found:\n{pdf_path}")

    print(f"Loading PDF: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    print(f"Loaded {len(documents)} pages.\n")

    return documents


if __name__ == "__main__":

    docs = load_pdf()

    print("First Page Preview\n")
    print(docs[0].page_content[:500])

    print("\nMetadata\n")
    print(docs[0].metadata)