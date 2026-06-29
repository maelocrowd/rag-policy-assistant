# NileTech Policy Assistant (RAG)

## Overview

The **NileTech Policy Assistant** is a Retrieval-Augmented Generation
(RAG) application that enables employees to ask natural language
questions about company policies and receive accurate, context-aware
answers grounded in indexed policy documents.

## Features

-   Semantic search over company policy documents
-   Retrieval-Augmented Generation (RAG)
-   Local Chroma vector database
-   Streamlit chat interface
-   Flask REST API
-   Source citations and snippets
-   Evaluation module for retrieval performance

## Project Structure

``` text
rag-policy-assistant/
│
├── api.py
├── app.py
├── requirements.txt
├── README.md
├── documents/
├── data/chroma_db/
├── src/
│   ├── evaluation.py
│   ├── ingestion.py
│   ├── prompts.py
│   ├── rag_chain.py
│   ├── retrieval.py
│   ├── utils.py
│   
└── tests/
```

## Architecture

``` text
User
 │
 ▼
Streamlit Frontend
 │
 ▼
Flask API
 │
 ▼
PolicyRAG
 ├── PolicyRetriever
 ├── ChromaDB
 └── OpenRouter LLM
```

## Installation

``` bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

``` text
OPENROUTER_API_KEY=your_api_key
LLM_MODEL=your_model_name
```

## Build the Vector Database

``` bash
python -m src.ingestion.py
```

## Run the Backend

``` bash
python api.py
```

Available endpoints:

-   `GET /` -- Redirects to the Streamlit interface
-   `POST /chat` -- Returns answers with citations
-   `GET /health` -- Returns API health status

## Run the Frontend

``` bash
streamlit run app.py
```

## Run Evaluation

``` bash
python -m src.evaluation
```

## Example Question

**Question**

> How many annual leave days do employees receive?

The assistant retrieves relevant policy chunks, generates a grounded
response, and displays supporting citations and snippets.

## Technologies

-   Python
-   Streamlit
-   Flask
-   ChromaDB
-   Sentence Transformers
-   OpenRouter API

## License

Developed for an academic Retrieval-Augmented Generation (RAG) project.
