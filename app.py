from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

import os

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#set up streamlit
st.title("Conversational RAG With PDF Uploadsand chat history")
st.write("Upload a PDF file and start chatting with it!")

#input the api_key
api_key=st.text_input("Enter the api key", type="password")

#check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    #chat interface
    session_id=st.text_input("Session ID ", value="default_session")

    #statefully manage chathistory
    if 'store' not in st.session_state:
        st.session_state['store'] = {}

    uploaded_file=st.file_uploader("Upload a PDF file", type=["pdf"],accept_multiple_files=False)
     
    # Process Uploaded files
    








