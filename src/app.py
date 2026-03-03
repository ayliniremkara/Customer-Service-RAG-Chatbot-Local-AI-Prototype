"""
Streamlit Chat Interface for the RAG Chatbot.

This is a starter template — feel free to modify, extend, or replace it entirely.
Run with: streamlit run src/app.py
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rag_chain import build_rag_chain
import random
from langchain_core.messages import HumanMessage, AIMessage

@st.cache_resource
def get_cached_chain(top_k: int):
    """Initialize the RAG chain and cache it to avoid reloading on every interaction"""
    return build_rag_chain(top_k)

# Greetings to be randomly choosen when the app is first loaded
GREETINGS = [
    "Hello, I'm your AI Assistant. How can I help you today?",
    "Hi there, I'm here to assist you with any questions you may have.",
    "Welcome! I'm your AI Assistant. What would you like to know?",
    "Hello! I can help with vehicles, services, or warranty information.",
    "Hi, feel free to ask me anything about our products or services.",
    "Hello, how can I assist you today?",
]

# ──────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar settings. Returns a dict of user-configured parameters."""
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider( 
            "Retrieved chunks (Top-K)",
            min_value=1,
            max_value=10,
            value=3,
            help="How many document chunks to retrieve per query.",
        )

        st.divider()
        st.markdown("**Models**")
        st.markdown("- Chat model: `llama3.2`")
        st.markdown("- Embedding model: `nomic-embed-text`")

        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "1. Your question is embedded\n"
            "2. Relevant document chunks are retrieved\n"
            "3. An LLM generates an answer based on the context"
        )

    return {"top_k": top_k}


def render_message(message: dict) -> None:
    """Render a single chat message with optional source expander."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")
                    

def render_chat_history() -> None:
    """Display all messages stored in session state."""
    for message in st.session_state.messages:
        render_message(message)


def get_bot_response(query: str, top_k: int) -> tuple[str, list[str]]:
    chain = get_cached_chain(top_k=top_k)

    history = st.session_state.messages[-8:] #Chat history to understand follow-up question

    response = chain.invoke({
        "question": query,
        "chat_history": history
    })

    answer = response["answer"]
    sources = response["source_documents"]
    return answer, sources



# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Customer Service Chatbot",
        page_icon="🚗",
        layout="centered",
    )

    st.title("🚗 Customer Service Chatbot")
    st.caption("Ask questions about vehicles, services, warranty, and more.")

    # Sidebar
    settings = render_sidebar()

    # Session state
    if "messages" not in st.session_state:
        greeting = random.choice(GREETINGS)
        st.session_state.messages = [
            {"role": "assistant", "content": greeting}
        ]

    render_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot response
        answer, sources = get_bot_response(prompt, top_k=settings["top_k"])

        response = {"role": "assistant", "content": answer, "sources": sources}
        render_message(response)
        st.session_state.messages.append(response)


if __name__ == "__main__":
    main()