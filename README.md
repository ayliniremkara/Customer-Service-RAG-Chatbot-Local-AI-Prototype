# 🚗 Customer Service RAG Chatbot — Local AI Prototype

## 📖 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Key Components](#key-components)
  - [Data Ingestion Pipeline](#data-ingestion-pipeline)
  - [RAG Chain](#rag-chain)
  - [Streamlit App](#streamlit-app)
- [Setup Instructions](#️-setup-instructions)
- [Knowledge Base](#-knowledge-base)
- [Acknowledgement](#-acknowledgement)

## 💻 Overview

This project implements a **local AI-powered customer service chatbot** using a Retrieval-Augmented Generation (RAG) pipeline at a large automotive company. 
It is designed to assist users with questions about vehicle features, maintenance, warranty, ordering process, electric vehicles and customer support.

## ⭐️ Key Features

- Runs fully locally (external APIs not used)
- Uses [Ollama](https://ollama.com/) for LLM and embeddings
- Vector search with [ChromaDB](https://www.trychroma.com/)
- RAG architecture supported with the knowledge base
- [Streamlit](https://streamlit.io/) chat interface
- Displays source documents alongside answers for transparency

## 🧩 Key Components

### Data Ingestion Pipeline
- **File**: `src/data_ingestion_pipeline.py`
- **Purpose**: Loads input files from the `knowledge_base` directory, splits them into chunks, generates embeddings, and stores them in a ChromaDB vector store.

### RAG Chain
- **File**: `src/rag_chain.py`
- **Purpose**: Implements the RAG chain, including retrieving relevant chunks from the vector store and generating answers using a language model.

### Streamlit App
- **File**: `src/app.py`
- **Purpose**: Provides a user interface for interacting with the chatbot. Users can ask questions, and the chatbot retrieves relevant information and generates answers.

## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayliniremkara/Customer-Service-RAG-Chatbot-Local-AI-Prototype.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd 2026---AI-Engineering-Case-Study
   ```
3. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate       
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install Ollama and pull the required models:**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
5. **Run the data ingestion pipeline to create embeddings and stores them in ChromaDb:**
   ```bash
   python src/data_ingestion_pipeline.py
   ```
6. **Start Customer Service Chatbot:**
   ```bash
   streamlit run src/app.py      
   ```

## 📚 Knowledge Base
The knowledge base is located in `data/knowledge_base/` and contains the following documents:

- `doc_01_vehicle_features.txt`: Vehicle Features & Configuration Options
- `doc_02_service_maintenance.txt`: After-Sales Service
- `doc_03_warranty.txt`: Warranty Coverage
- `doc_04_ordering_process.txt`: Ordering Process
- `doc_05_electric_vehicles.txt`: Electric Vehicles & Charging
- `doc_06_customer_support.txt`: Customer Support Channels


To extend the knowledge base, add new `.txt` files to `data/knowledge_base/` directory. Before re-running, clear the existing ChromaDB collection to avoid duplicate entries:
```bash
rm -rf data/chroma_db
python src/data_ingestion_pipeline.py
```

## 💡 Acknowledgement
- [LangChain](https://www.langchain.com) framework for building agents and LLM-powered applications. 
- [Streamlit](https://streamlit.io/) for the interactive user interface.
- [Ollama](https://ollama.com/) — local LLM inference and embeddings.
- [ChromaDB](https://docs.trychroma.com/docs/overview/introduction) lightweight and persistent vector database.
