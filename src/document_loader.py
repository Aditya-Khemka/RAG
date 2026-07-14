"""Utilities for loading PDF documents into LangChain document objects."""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: str | Path):
    """Load a PDF file and return one LangChain document per page."""

    # Convert the input to a string so the PDF loader can work with either
    # a Path object or a plain string path.
    loader = PyPDFLoader(str(pdf_path))

    # Load the PDF content into memory as LangChain documents.
    documents = loader.load()

    print(f"Loaded {len(documents)} page(s).")

    return documents



def document_splitter():
   
    docs = load_pdf("data/sample.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # split the docs
    split_docs = splitter.split_documents(docs)

    print(f"Split into {len(split_docs)} chunks")
    print(f"\nFirst chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content: {split_docs[0].page_content[:200]}...")


if __name__ == "__main__":
    document_splitter()