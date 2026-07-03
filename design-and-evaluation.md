# Design and Evaluation

## Technical Design

The application follows a modular Retrieval-Augmented Generation (RAG) architecture. It uses:

- **Streamlit** for the web-based user interface
- **Flask** for the backend REST API
- **Sentence Transformers (all-MiniLM-L6-v2)** for document embeddings
- **Gemma 4 31B** (via OpenRouter) as the Large Language Model (LLM)
- **ChromaDB** as the vector database

These technologies were selected because they are lightweight, freely available, and satisfy the project requirements.

---

## Corpus

The knowledge base consists of publicly available Ethiopian Civil Servants proclamations. Additional Markdown documents were created from these proclamations to provide structured content for the assignment.

---

## Document Ingestion

The ingestion pipeline processes both **Markdown (.md)** and **PDF (.pdf)** documents by:

1. Loading documents
2. Splitting them into chunks
3. Generating vector embeddings
4. Storing the embeddings in ChromaDB

Both document formats were indexed to compare retrieval consistency and ensure the system performs reliably across different file types.

---

## Chunking Strategy

A hybrid chunking strategy was implemented:

- **Heading-aware chunking** for Markdown documents to preserve document structure.
- **Recursive character chunking** for other document types using:
  - Chunk size: **500 characters**
  - Chunk overlap: **100 characters**

This approach preserves semantic structure while maintaining sufficient context for retrieval.

---

## Retrieval Strategy

The retriever returns the **Top-K = 4** most relevant document chunks.

To improve retrieval quality, a similarity distance threshold(L2) was applied before passing retrieved contexts to the LLM.

The threshold was initially set to **0.7**, but testing showed that relevant policy sections with minor wording differences were often excluded. After empirical evaluation, the threshold was increased to **1.1**, allowing the system to retrieve semantically similar policy statements while maintaining answer quality.

---

## RAG Pipeline

The `PolicyRAG` pipeline performs the following steps:

1. Retrieves the most relevant document chunks.
2. Filters results using the similarity threshold.
3. Constructs the LLM prompt.
4. Generates grounded responses using:
   - Temperature = **0.0**
   - Seed = **42**
5. Returns the answer together with supporting citations.

The pipeline also handles duplicate citations, missing retrieval results, and API exceptions.

---

## Prompt Design

The system prompt instructs the model to:

- Answer only using indexed policy documents.
- Avoid generating unsupported information.
- Provide concise and professional responses (2–3 sentences).
- Cite every factual statement.
- Combine multiple sources only when supported by retrieved evidence.
- Return a fallback response when sufficient information is unavailable.

---

## Evaluation

The system was evaluated using a benchmark dataset of 20 question-answer pairs, each mapped to expected answers, source documents, and specific section references. This benchmark verified whether the generated responses were fully supported by the indexed source documents.
The evaluation assessed the following performance areas:
- Accuracy Metrics: Groundedness, citation accuracy, and exact/partial match percentages.
- Latency Metrics: p50 (median) and p95 (tail) latencies, along with minimum and maximum bounds.

---

## Deployment

Due to free-tier hardware and API limitations, the application is deployed and demonstrated locally.

---

## Flask API

The backend is implemented using Flask and runs locally on **port 5000**.

The API:

- Receives user queries from the Streamlit interface
- Executes the RAG pipeline
- Returns generated answers with citations

---

## User Interface

The frontend is implemented using Streamlit for its simplicity and seamless integration with Python.

The interface communicates with the backend through the `/chat` endpoint to send user queries and display generated responses.

---

## Testing

Three unit tests were implemented using **pytest** to validate:

- Flask API endpoints
- Streamlit application functionality
