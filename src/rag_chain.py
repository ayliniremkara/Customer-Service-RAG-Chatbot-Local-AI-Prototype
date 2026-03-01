"""
RAG Pipeline
----------------------------
Loads persisted ChromaDB, retrieves Top-K chunks, and generates an answer using Ollama.
question -> retrieve ->  build context ->  prompt -> LLM -> answer
"""

from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

PERSIST_DIR = Path("data/chroma_db") #ChromaDB persistence directory where embeddings are stored

EMBEDDING_MODEL = "nomic-embed-text" #Embedding model used to vectorize documents and queries
CHAT_MODEL      = "llama3.2"         #LLM used for generating answers based on retrieved context
COLLECTION_NAME = "knowledge_base"   #Vector store collection name 
TEMPERATURE     = 0.1                #Settings for the LLM to control randomness in answer generation

SYSTEM_PROMPT = """You are an customer chatbot at a large automotive company.
You are an expert in answering questions using the provided context. 
Always include sources. If the answer is not in the context, say: "Sorry, I don't have information about this question."
"""

def load_vectorstore():
    """Load vector store to be used for retrieval task"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore

class RAGChain:
    """RAGChain class to handle the retrieval and generation process"""
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def invoke(self, inputs: dict) -> dict:
        """Invoke the RAG chain with a question input. Returns a dict with the answer and source documents"""
        question = inputs["question"]
        docs = self.retriever.invoke(question)
        
        #If no relevant documents are found, return a default answer 
        if not docs: 
            return {
                "answer": "Sorry, I don't have information about this question. Please contact our support team.",
                "source_documents": []
            }
        
        texts = []
        for doc in docs:
            source_name = Path(doc.metadata['source']).stem     #Without file extension for cleaner display 
            text = f"Source: {source_name}\n{doc.page_content}" 
            texts.append(text) 
        context = "\n\n".join(texts)

        #Create LLM prompt 
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {question}"
            )
        ]
        response = self.llm.invoke(messages)

        #Return dictionary with answer and sources to be displayed in the Streamlit App
        return {
            "answer": response.content,
            "source_documents": [
                Path(doc.metadata["source"]).stem for doc in docs
            ]
        }

def build_rag_chain(top_k: int):
    """Build the RAG chain by loading the vector store, creating a retriever, and initializing the LLM"""
    vector_store = load_vectorstore()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOllama(model=CHAT_MODEL, temperature=TEMPERATURE)
    return RAGChain(retriever, llm)