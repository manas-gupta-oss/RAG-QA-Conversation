from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain_community.chat_message_histories import (
    ChatMessageHistory
)

from langchain_core.chat_history import (
    BaseChatMessageHistory
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

import streamlit as st
from dotenv import load_dotenv
import os


load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Set up Streamlit
st.title("Conversational RAG With PDF Uploads and Chat History")

st.write("Upload PDF files and start chatting with them!")


# Input API key
api_key = st.text_input(
    "Enter the API key",
    type="password"
)


# Check if API key is provided
if api_key:

    # Create LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Gemma2-9b-It"
    )


    # Chat interface
    session_id = st.text_input(
        "Session ID",
        value="default_session"
    )


    # Statefully manage chat history
    if "store" not in st.session_state:
        st.session_state["store"] = {}


    # Upload multiple PDF files
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )


    # Process uploaded files
    if uploaded_files:

        documents = []


        # Process each PDF
        for uploaded_file in uploaded_files:

            # Create temporary PDF path
            temppdf = f"./{uploaded_file.name}"


            # Save uploaded PDF
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())


            # Load PDF
            loader = PyPDFLoader(temppdf)


            # Extract pages
            docs = loader.load()


            # Add pages to main list
            documents.extend(docs)


        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )


        splits = text_splitter.split_documents(documents)


        # Create vector database
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )


        # Create retriever
        retriever = vectorstore.as_retriever()


        # Prompt for reformulating question
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just reformulate it if needed and otherwise return it as it is."
        )


        # Create contextualization prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),

                ("human", "{input}"),
            ]
        )






