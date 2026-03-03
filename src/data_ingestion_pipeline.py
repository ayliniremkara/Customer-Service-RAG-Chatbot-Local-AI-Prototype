"""
Document Ingestion Pipeline
----------------------------
Reads .txt files in the knowledge base folder, splits them into chunks,
creates embeddings with an Ollama embedding model, and stores them in ChromaDB.
"""

import os, shutil
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #Allows defining specific separators 
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


BASE_DIR = Path(__file__).parent.parent  #repo root

KB_DIR = BASE_DIR / "data" / "knowledge_base" #Knowledge base embeddings storage 
PERSIST_DIR = BASE_DIR / "data" / "chroma_db"  #ChromaDB persistence directory where embeddings are stored
EMBEDDING_MODEL = "nomic-embed-text"    #Embedding model used to vectorize documents and queries
COLLECTION_NAME = "knowledge_base"      #Vector store collection name

CHUNK_SIZE = 500                    #Adjust based on the average length of documents
CHUNK_OVERLAP = 20                  #Adjust according to how much context to retain between chunks


def extract_metadata(content: str):
    """Extract title and category from TXT format"""
    lines = content.strip().split("\n")
    metadata = {}
    
    for line in lines[:2]:
        line = line.strip()
        if line.startswith("Title:"):
            metadata["title"] = line.replace("Title:", "").strip()
        elif line.startswith("Category:"):
            metadata["category"] = line.replace("Category:", "").strip()
    
    return metadata


def load_documents(docs_path: str):
    """Load documents from the knowledge base directory"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    raw_documents = loader.load()

    documents = []
    for doc in raw_documents:
        extra_meta = extract_metadata(doc.page_content)
        doc.metadata.update(extra_meta)  # Merge into existing metadata
        documents.append(doc)

    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents, chunk_size: int, chunk_overlap: int):
    """Split documents into chunks with overlap"""
    print("Splitting documents into chunks with overlap...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",               # Section breaks (Charging Options:, Battery Technology:)
            "\n",                 # Bullet points and lines
            ". ",                 # End of sentences
            " ",                  # Words
            ""                    # Characters (last resort)
        ],
        keep_separator=True       # Preserve bullet points and section headers
    )

    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks[:50]): 
        print("-------------")
        print(f"Metadata: ", chunk.metadata)
        print("----")
        print(f"Chunk {i+1}: {chunk.page_content}")  
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory: str, collection_name: str):
    """Create vector embeddings for the document chunks and store them in ChromaDB"""
    print("Creating vector embeddings and storing in ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print("Creating ChromaDB vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Vector store created and persisted in {persist_directory}.")
    return vector_store

def main():
    """Run the document ingestion pipeline: load, split, and create vector store"""
    print("Document Ingestion Pipeline Started")

    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print("Old vector store removed.")

    documents = load_documents(KB_DIR)
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    
    create_vector_store(chunks, PERSIST_DIR, COLLECTION_NAME)

if __name__ == "__main__":
    main()