# NileTech Policy Assistant

A Retrieval-Augmented Generation (RAG) application for answering policy-related questions using company documents.

---

## Overview

The NileTech Policy Assistant combines semantic search with a Large Language Model to generate grounded answers from indexed policy documents.

The application uses a modular architecture consisting of Streamlit, Flask, ChromaDB, and the Gemma 4 31B Instruct model via OpenRouter. A hybrid chunking strategy and optimized retrieval settings improve retrieval accuracy while providing supporting citations for generated answers.

---

## Features

- Retrieval-Augmented Generation (RAG)
- Semantic search over policy documents
- Local ChromaDB vector database
- Streamlit web interface
- Flask REST API
- Automatic source citations
- Evaluation framework for retrieval performance
- Support for both Markdown and PDF documents

---

## Project Structure

```text
rag-policy-assistant/
│
├── api.py
├── app.py
├── README.md
├── design-and-evaluation.md
├── requirements.txt
├── eval_dataset.json
│
├── documents/
├── data/
│   └── chroma_db/
│
├── src/
│   ├── evaluation.py
│   ├── ingestion.py
│   ├── prompts.py
│   ├── rag_chain.py
│   ├── retrieval.py
│   └── utils.py
│
└── tests/
```

---

## System Architecture

```text
             User
               │
               ▼
     Streamlit Frontend
               │
               ▼
          Flask REST API
               │
               ▼
          PolicyRAG Engine
          ├── Retriever
          ├── ChromaDB
          └── Gemma LLM
```

---

## Installation

Python **3.11** is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file.

```env
OPENROUTER_API_KEY=YOUR_API_KEY

LLM_MODEL=google/gemma-4-31b-it
OPENAI_BASE_URL=https://openrouter.ai/api/v1

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

BACKEND_URL=http://127.0.0.1:5000

CHROMA_DIR=data/chroma_db
CORPUS_DIR=documents

CHUNK_SIZE=500
CHUNK_OVERLAP=100

TOP_K=4
MAX_DISTANCE_THRESHOLD=1.1

SEED=42
```

---

## Build the Vector Database

```bash
python -m src.ingestion
```

---

## Run the Backend

```bash
python api.py
```

Available endpoints:

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Returns API health status |
| POST | `/chat` | Returns answers with citations |

---

## Run the Frontend

```bash
streamlit run app.py
```

---

## Run the Evaluation

```bash
python -m src.evaluation
```

---

## Technology Stack

- Python
- Streamlit
- Flask
- ChromaDB
- Sentence Transformers (all-MiniLM-L6-v2)
- OpenRouter API
- Gemma 4 31B free tier

---

## Testing

Run the unit tests using:

```bash
pytest
```

---

## License

This project was developed for an academic Retrieval-Augmented Generation (RAG) assignment.