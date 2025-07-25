# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from utils.document_loader import load_documents
from utils.vectorstore_setup import setup_vectorstore
from utils.chain_factory import create_chain

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)

st.title("🦙 Chat with PDF - LLAMA")

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDFs
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    documents = load_documents(file_paths)

    if "uploaded_files" not in st.session_state or st.session_state.uploaded_files != uploaded_files:
        st.cache_data.clear()
        st.session_state.uploaded_files = uploaded_files

    vectorstore = setup_vectorstore(documents)
    st.session_state.conversation_chain = create_chain(vectorstore)

# Chat UI
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask Llama...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
