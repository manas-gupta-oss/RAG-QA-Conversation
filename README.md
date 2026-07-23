# Conversational PDF Chatbot

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content. The application uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant sections from the uploaded documents before generating an answer.

## What it does

* Upload one or more PDF files
* Extract and split document content into smaller chunks
* Convert document chunks into embeddings
* Store the embeddings in Chroma
* Retrieve relevant content for a user query
* Answer questions using a Groq-hosted language model
* Maintain conversation history for follow-up questions

## How the application works

The basic workflow is:

```text
PDF Files
   ↓
PDF Text Extraction
   ↓
Text Splitting
   ↓
Embedding Generation
   ↓
Chroma Vector Store
   ↓
Relevant Document Retrieval
   ↓
Groq LLM
   ↓
Answer
```

For follow-up questions, the previous conversation is also used to understand the context of the latest query.

For example:

```text
User: What is the main idea of this document?

User: What are its advantages?
```

The application uses the previous question and answer to understand what "its" refers to.

## Technologies Used

* Python
* Streamlit
* LangChain
* Groq
* Chroma
* FastEmbed
* PyPDF

## Project Structure

```text
chat_with_pdf/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Running the project locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd chat_with_pdf
```

### 2. Create a virtual environment

```bash
python -m venv pdfchatbot
```

Activate it on Windows:

```bash
pdfchatbot\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## API Key

The application requires a Groq API key to generate responses.

For local development, the key can be provided through the application or stored securely using environment variables.

Do not commit API keys or `.env` files to GitHub.

Example `.gitignore`:

```text
.env
__pycache__/
*.pyc
pdfchatbot/
.venv/
```

## Main components

### PDF processing

Uploaded PDFs are loaded and their content is extracted page by page. The extracted text is divided into smaller chunks so that relevant sections can be retrieved efficiently.

### Embeddings

The document chunks are converted into vector representations using FastEmbed. These vectors allow the application to find content that is semantically related to a user's question.

### Vector search

Chroma is used to store the document embeddings and retrieve the most relevant chunks during a query.

### Conversational retrieval

The application uses the conversation history to reformulate follow-up questions into standalone questions before searching the document collection.

### Answer generation

The retrieved document context is passed to a Groq language model, which generates the final response.

## Possible improvements

Some features that could be added in the future:

* Persistent vector database storage
* Streaming responses
* Source references for answers
* Better chat interface
* Support for DOCX and TXT files
* User authentication
* Deployment with persistent storage
* Improved document processing for scanned PDFs

## Author

Manas Gupta

This project was built to understand and implement a complete conversational RAG pipeline, from document ingestion and embeddings to retrieval and context-aware response generation.
