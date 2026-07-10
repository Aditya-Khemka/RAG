"""Helpers for splitting long documents into smaller chunks."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=50,
):
    """Split a list of documents into smaller overlapping chunks."""

    # Configure the text splitter with a fixed chunk size and overlap.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Create smaller document chunks that can be embedded and stored.
    split_docs = splitter.split_documents(documents)

    print(f"Created {len(split_docs)} chunks.")

    return split_docs