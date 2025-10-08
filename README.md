---
title: AI Portfolio Backend
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---


# AI Portfolio Chatbot with RAG

A resume‑focused, first‑person Q&A chatbot that uses Retrieval‑Augmented Generation (RAG) to deliver accurate, grounded answers about skills, experience, education, and projects. Built with LangChain, FAISS, Sentence Transformers, FastAPI, Streamlit, and Groq for low‑latency inference.

## Overview

This app ingests a resume (PDF/DOCX/TXT), chunks and embeds it locally, stores vectors in FAISS, retrieves relevant context for each query, and generates concise first‑person answers—ideal for recruiter interactions.

## Features

- Retrieval‑Augmented Generation for grounded, low‑hallucination answers.
- Local embeddings via Sentence Transformers (all‑MiniLM‑L6‑v2) for fast semantic search on CPU/GPU.
- FAISS vector database for high‑performance similarity search and local persistence.
- Multi‑format loaders (PDF/DOCX/TXT) + resume‑tuned chunking.
- FastAPI backend (typed models, /chat, /health, /info) and Streamlit UI (dark/light themes, quick actions).
- Groq chat inference for ultra‑low latency with Llama and Mixtral.

## Architecture

User → Streamlit UI → FastAPI Backend → LangChain RAG
↓
FAISS Vector DB
↓
Groq/OpenAI Chat LLM
↓
First‑person response


Indexing (load → split → embed → store) happens once; serving (embed → retrieve → generate) runs per query.

## Tech Stack

| Layer          | Choice                                     | Why |
|----------------|---------------------------------------------|-----|
| Orchestration  | LangChain                                   | Chains, retrievers, loaders, prompts for RAG |
| Embeddings     | Sentence Transformers (all‑MiniLM‑L6‑v2)    | Fast, 384‑dim embeddings, excellent CPU perf |
| Vector DB      | FAISS                                       | Efficient similarity search, local persistence |
| Loaders        | PyPDFLoader, Docx2txtLoader                 | Reliable PDF/DOCX parsing |
| Backend        | FastAPI                                     | High‑performance Python API, OpenAPI |
| Frontend       | Streamlit                                   | Rapid, modern UI with simple deployment |
| LLM Inference  | Groq Python SDK                             | Low‑latency Llama/Mixtral chat completions |

## Installation

Create a virtual environment and install dependencies:

python -m venv .venv

macOS/Linux
source .venv/bin/activate

Windows
..venv\Scripts\activate

pip install -r requirements.txt


## Configuration

Provide API keys and set resume path:

.env
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key # optional
TAVILY_API_KEY=your_tavily_key # optional




In `ai_agent.py`, set the resume file path to one of:
- `./resume.pdf` (text‑based PDF recommended)
- `./resume.docx`
- `./resume.txt` (most robust if parsing issues occur)

## Running

Start backend and frontend:

Terminal 1
python Backend.py

Terminal 2
streamlit run frontend.py


- Backend: FastAPI on http://127.0.0.1:9999 (endpoints: `/chat`, `/health`, `/info`)
- Frontend: Streamlit on http://localhost:8501

## Usage

Ask resume‑related questions such as:
- “What programming languages do you know?”
- “Tell me about your work experience.”
- “What projects have you worked on?”
- “What’s your educational background?”

Non‑resume queries are politely redirected to keep the conversation professional and on-topic.

## API (FastAPI)

- Base URL: `http://127.0.0.1:9999`

Example request:
curl -X POST "http://127.0.0.1:9999/chat"
-H "Content-Type: application/json"
-d '{
"model_name": "llama-3.3-70b-versatile",
"model_provider": "Groq",
"prompt": "Act as AI Assistant",
"messages": ["What programming languages do you know?"],
"allow_search": false
}'


Example response:
{
"response": "I'm proficient in Python, Java, and JavaScript...",
"is_resume_related": true
}


## Tuning

- Retrieval: adjust `search_kwargs={"k": 3..5}` for breadth vs. precision.
- Chunking: tune `chunk_size` and `chunk_overlap` for your resume format.
- Models: switch between Llama 3.3 70B and Mixtral 8x7B via Groq for latency/quality tradeoffs.

## Troubleshooting

- Poor PDF parsing? Try DOCX or export to TXT and re‑index.
- Vector store reload issues? Ensure compatible FAISS versions and safe deserialization logic.
- Frontend connection errors? Confirm backend is running and port/URL match.
- Embedding download failures? Check Sentence Transformers version and network access.

## Performance

- First run: downloads all‑MiniLM‑L6‑v2 (~90 MB) and builds FAISS index.
- Subsequent runs: fast (2–3 s typical), thanks to cached models and persisted vectors.

## Project Structure

.
├── ai_agent.py # RAG pipeline & AI agent
├── Backend.py # FastAPI server (/chat, /health, /info)
├── frontend.py # Streamlit UI
├── requirements.txt # Dependencies
├── .env # API keys (not committed)
├── .gitignore # Protects secrets, resume files, vectors, etc.
└── faiss_resume_vectorstore/ # Local vector DB (not committed)




## Sources

- LangChain RAG Tutorial: https://python.langchain.com/docs/tutorials/rag/  
- LangChain Vector Stores: https://python.langchain.com/docs/integrations/vectorstores/  
- FAISS (LangChain Integration): https://python.langchain.com/docs/integrations/vectorstores/faiss/  
- FAISS Docs: https://faiss.ai/index.html  
- FAISS GitHub: https://github.com/facebookresearch/faiss  
- Sentence Transformers (all‑MiniLM‑L6‑v2): https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  
- PyPDFLoader: https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/  
- Docx2txtLoader: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.word_document.Docx2txtLoader.html  
- FastAPI: https://fastapi.tiangolo.com  
- Streamlit: https://docs.streamlit.io  
- Groq Python SDK: https://github.com/groq/groq-python




pip install langgraph
pip install langchain_groq langchain_openai langchain_community
pip install -U langchain-tavily langchain
pip install -U langchain-tavily
pip install -U langchain
pip install pydantic
pip install fastapi
pip install uvicorn
pip install streamlit