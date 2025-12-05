# src/app.py
import os
import pickle
from src.vectorizer import create_chunks_and_vectorize
import streamlit as st
from rag_pipeline import rag_loader_and_pipeliner # This imports our new chain
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(
    page_title="VruCare",
    page_icon="🩺",
    layout="wide"
)
VECTORSTORE_PATH = "vectorstore"
if not os.path.exists(VECTORSTORE_PATH):
    with st.spinner(" performing one-time setup: Building the knowledge base..."):
        create_chunks_and_vectorize()
    st.success("Knowledge base built successfully!")

with st.sidebar:
    st.title("🩺 VruCare")
    st.markdown("---")
    
    st.header("About This Chatbot")
    st.info(
        "This is **VruCare**, an AI assistant for kidney stone information. "
        "It uses a Retrieval-Augmented Generation (RAG) model to answer "
        "your questions based on a curated set of trusted medical documents. "
        "\n\n"
        "**Note:** This bot is for informational purposes only and is **not** "
        "a substitute for professional medical advice."
    )
    
    st.markdown("---")
    
    st.header("Chat Controls")
    if st.button("Clear Chat History", type="primary"):
        st.session_state.chat_history = []
        st.toast("Chat history cleared!", icon="🗑️")
        st.rerun()

# --- Title and Disclaimer (same) ---
st.title("🩺VruCare:Kidney Stone Information Chatbot")
st.warning(
    "**⚠️ Disclaimer:** I am an AI assistant and not a medical professional. "
    "The information I provide is based on curated medical documents and "
    "is for informational purposes only. It is **not** a substitute for "
    "professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider with any health concerns."
)

# --- Caching the RAG Chain (same) ---
@st.cache_resource
def load_rag_chain():
    try:
        chain = rag_loader_and_pipeliner()
        return chain
    except Exception as e:
        st.error(f"Failed to load RAG chain. Error: {e}")
        st.stop()

rag_chain = load_rag_chain()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    else:
        continue
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Type Your Query Regarding Kidney Stones..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    history_window = st.session_state.chat_history[-12:]
    chain_input = {
        "input": prompt,
        "chat_history": history_window
    }
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                def get_answer_stream(stream):
                    for chunk in stream:
                        if "answer" in chunk:
                            yield chunk["answer"]
                
                response_generator = get_answer_stream(rag_chain.stream(chain_input))
                answer = st.write_stream(response_generator)
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=answer))

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)