"""Utilities for loading PDF documents into LangChain document objects."""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str | Path):
    """Load a PDF file and return one LangChain document per page."""

    # Convert the input to a string so the PDF loader can work with either
    # a Path object or a plain string path.
    loader = PyPDFLoader(str(pdf_path))

    # Load the PDF content into memory as LangChain documents.
    documents = loader.load()

    print(f"Loaded {len(documents)} page(s).")

    return documents