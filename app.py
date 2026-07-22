from langchain.document_loaders import PyPDFLoader
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
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

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_files:

    documents = []

    for uploaded_file in uploaded_files:

        # Create a temporary file name
        temppdf = f"./{uploaded_file.name}"

        # Save the uploaded PDF
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        # Load the PDF
        loader = PyPDFLoader(temppdf)

        # Extract pages/documents
        docs = loader.load()

        # Add the pages to the main documents list
        documents.extend(docs)

        #split and create embeddings for the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever=






