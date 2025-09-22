# Agentic RAG Chatbot (MCP + LangChain + FAISS + Ollama)

This project implements an agentic Retrieval-Augmented Generation (RAG) chatbot with a Streamlit UI.  
It integrates multi-agent orchestration (via MCP), FAISS vector storage, SentenceTransformers embeddings, Cross-Encoder reranking, and an Ollama local LLM for end-to-end document-based question answering.

---

## Key Features
- Streamlit UI for multi-turn document-based Q&A.
- Agentic design: Ingestion, Retrieval, and LLM Response agents coordinated via MCP.
- Vector search: FAISS + SentenceTransformers (`all-MiniLM-L6-v2`) for embeddings and similarity search.
- Reranking with cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for improved accuracy.
- Local LLM: Responses generated using Ollama with `llama3.2:1b` running at `http://localhost:11434`.

---

## Project Files

| File | Description |
|------|-------------|
| `requirements.txt` | Python dependencies required for the project. |
| `utils.py` | Utility functions for logging, metadata handling, and other helpers. |
| `ingestion_agent.py` | Extracts text from PDFs/DOCX/PPTX, splits into chunks, generates embeddings, and stores them in FAISS. |
| `vector_store.py` | Manages FAISS vector database: upsert, search, and persistence. |
| `retrieval_agent.py` | Retrieves relevant chunks and reranks them with the cross-encoder. |
| `llm_response_agent.py` | Calls the Ollama API to generate answers using retrieved context. |
| `coordinator.py` | Coordinates communication between agents via MCP. |
| `agent_processes.py` | Manages running agents as separate processes using multiprocessing and queues. |
| `streamlit_app.py` | Streamlit front-end for uploading documents, querying, and chatting with the system. |

---

## Full Setup Instructions

Follow these steps to get the project running locally.

### 1. Install project dependencies

pip install --upgrade pip
pip install -r requirements.txt

### 2. Install and configure Ollama

1. Download and install Ollama from https://ollama.ai
2. Start the Ollama service (default API: http://localhost:11434).
3. Pull the required model:

   ollama pull llama3.2:1b

4. Verify installation:

   ollama list

5. Start the Ollama server

   ollama serve

6. Run the Streamlit application

   streamlit run streamlit_app.py


## Work Flow

-> Upload a document in the Streamlit UI.

-> Ingestion Agent extracts and chunks text, embeds with SentenceTransformers, and stores in FAISS.

-> Retrieval Agent retrieves top-k relevant chunks and reranks them.

-> The LLM Response Agent uses retrieved context and the user query to generate an answer via Ollama.

-> The MCP Broker manages communication between agents.

-> The final answer appears in the chat interface.