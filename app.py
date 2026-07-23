from langchain_community.embeddings import FastEmbedEmbeddings
from chromadb import Client
from langchain_chroma import Chroma

client = Client()


from langchain_community.document_loaders import PyPDFLoader

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

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

from langchain_core.chat_history import BaseChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()


embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# Streamlit UI
st.title("Conversational RAG With PDF Uploads and Chat History")

st.write("Upload PDF files and start chatting with them!")


# API key input
api_key = st.text_input(
    "Enter the Groq API key",
    type="password"
)


if api_key:

    # Create LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.3-70b-versatile"
    )


    # Session ID
    session_id = st.text_input(
        "Session ID",
        value="default_session"
    )


    # Store chat histories
    if "store" not in st.session_state:
        st.session_state.store = {}


    # Upload PDFs
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )


    if uploaded_files:

        documents = []


        # Load all PDFs
        for uploaded_file in uploaded_files:

            temppdf = f"./{uploaded_file.name}"


            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())


            loader = PyPDFLoader(temppdf)

            docs = loader.load()

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
             embedding=embeddings,
              collection_name="pdf_collection")
    


        # Create retriever
        retriever = vectorstore.as_retriever()


        # Contextualization prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just reformulate it if needed and otherwise return it as it is."
        )


        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),

                ("human", "{input}"),
            ]
        )


        # History-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )


        # Question-answer prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n"
            "{context}"
        )


        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),

                ("human", "{input}"),
            ]
        )


        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )


        # Create RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )


        # Chat history function
        def get_session_history(
            session_id: str
        ) -> BaseChatMessageHistory:

            if session_id not in st.session_state.store:

                st.session_state.store[session_id] = (
                    ChatMessageHistory()
                )

            return st.session_state.store[session_id]


        # Add message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


        # User question
        user_input = st.text_input(
            "Your Question:"
        )


        if user_input:

            response = conversational_rag_chain.invoke(
                {
                    "input": user_input
                },

                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )


            st.success(response["answer"])


            session_history = get_session_history(session_id)

            st.write(
                "Chat History:",
                session_history.messages
            )

else:

    st.warning(
        "Please enter your Groq API key."
    )

        




