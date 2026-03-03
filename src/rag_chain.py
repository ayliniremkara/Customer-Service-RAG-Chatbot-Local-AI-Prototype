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

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PERSIST_DIR = BASE_DIR / "data" / "chroma_db"   #ChromaDB persistence directory where embeddings are stored

EMBEDDING_MODEL = "nomic-embed-text" #Embedding model used to vectorize documents and queries
CHAT_MODEL      = "llama3.2"         #LLM used for generating answers based on retrieved context
COLLECTION_NAME = "knowledge_base"   #Vector store collection name 
TEMPERATURE     = 0.1                #Settings for the LLM to control randomness in answer generation

SYSTEM_PROMPT = """You are a helpful and professional customer service assistant for an automotive company. 

Your goal is to answer questions accurately based ONLY on the provided context. 

Guidelines:
1. Use the provided context to answer the user's question. 
2. If the context contains the information, even if it's not a direct word-for-word match, try to summarize or explain it to help the user.
3. Be specific. If the context mentions durations (like years), limits (like mileage), or specific conditions, include them in your answer.
4. If you find relevant information in multiple documents, combine them into a coherent response.
5. If the answer is absolutely not present in the provided context, ONLY then say: "Sorry, I don't have information about this question."
6. If the user asks a follow-up question, use the chat history to understand the context.
7. Do NOT mention document names inside the answer text. 
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

    def invoke(self, inputs: dict):
        """Invoke the RAG chain with a question input. Returns a dict with the answer and source documents"""
        question = inputs["question"]

        history_text= ""
        chat_history = inputs.get("chat_history", []) 
        for m in chat_history[-8:]:
            history_text += m["role"] + ": " + m["content"] + "\n"

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
                content=(
                    f"Chat History:\n{history_text}\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            )
        ]
        response = self.llm.invoke(messages)
        print("DEBUG sources:", [Path(d.metadata["source"]).stem for d in docs])


        answer = response.content.strip()
        answer_l = answer.lower()
        # If model says "no info", don't show sources
        if "don't have information" in answer_l or "dont have information" in answer_l:
            return {"answer": answer, "source_documents": []}
        return {
            "answer": answer,
            "source_documents": list(dict.fromkeys(Path(doc.metadata["source"]).stem for doc in docs))
        }


def build_rag_chain(top_k: int):
    """Build the RAG chain by loading the vector store, creating a retriever, and initializing the LLM"""
    vector_store = load_vectorstore()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    llm = ChatOllama(model=CHAT_MODEL, temperature=TEMPERATURE)
    return RAGChain(retriever, llm)