"""
Document Ingestion Pipeline
----------------------------
Reads .txt files in the knowledge base folder, splits them into chunks,
creates embeddings with an Ollama embedding model, and stores them in ChromaDB.
"""

import os

from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

BASE_DIR = Path(__file__).parent.parent  

KB_DIR      = str(BASE_DIR / "data" / "knowledge_base")
PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "knowledge_base"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

def load_documents(docs_path: str):
    """Load documents from the specified directory."""
    print(f"Loading documents from {docs_path}...")

    # Check if the directory exists   
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    # Load all .txt files from the knowledge base directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    return documents

def split_documents(documents, chunk_size: int, chunk_overlap: int):
    """Split documents into chunks."""
    print("Splitting documents into chunks with overlap...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks.")

    return chunks

def create_vector_store(chunks, persist_directory: str):
    """Create vector embeddings and persist them in ChromaDB vector store."""

    print("Creating vector embeddings and storing in ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    #Create ChromaDB vector store and persist it
    print("Creating ChromaDB vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    print(f"Vector store created and persisted in {persist_directory}.")
    return vector_store


def main():
    print("Document Ingestion Pipeline")
    # Load documents
    documents = load_documents(KB_DIR)

    # Split documents into chunks
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # Create vector embeddings
    create_vector_store(chunks, PERSIST_DIR)
    

if __name__ == "__main__":
    main()