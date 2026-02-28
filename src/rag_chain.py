"""
RAG Pipeline
----------------------------
Loads persisted ChromaDB, retrieves Top-K chunks, and generates an answer using Ollama.
"""

from pathlib import Path
from operator import itemgetter

from pathlib import Path 
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_community.chat_models import ChatOllama 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR    = Path(__file__).parent.parent
PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db")

EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL      = "llama3.2"
COLLECTION_NAME = "knowledge_base"
TEMPERATURE     = 0.1

SYSTEM_PROMPT = """You are a chatbot that answers using the provided context.
If the answer is not in the context, say: "I don't have information about that."
"""

def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def format_docs(docs):
    formatted = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).stem
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def get_sources(docs):
    sources = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).stem
        if source not in sources:
            sources.append(source)
    return sources


def build_rag_chain(top_k: int):
    vector_store = load_vectorstore()

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])

    llm = ChatOllama(model=CHAT_MODEL, temperature=TEMPERATURE)

    # IMPORTANT:
    # - We expect input like {"question": "..."}.
    # - itemgetter("question") extracts the actual query string.
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever