# PolicyRAG System Architecture and Evaluation

## Technical Design

The system uses Streamlit for the web interface, Flask for the API, all-MiniLM-L6-v2 for embeddings, gemma-4-31b-it:free via OpenRouter as the language model, and ChromaDB as the vector database. These tools were chosen because they are free to use and meet the project requirements.

## Document Ingestion

The pipeline processes Markdown (.md) and PDF (.pdf) files by loading, chunking, embedding, and storing them in ChromaDB. Both formats were indexed to compare retrieval consistency.

## Chunking Strategy

Documents were chunked using both heading-aware and recursive chunking (500-character chunks with 100-character overlap) to preserve structure and improve retrieval.

## Retrieval

The retriever uses Top-K = 4 to balance answer quality and efficiency. The similarity threshold was adjusted from 0.7 to 1.1 to capture more relevant policy variations and supporting context with slightly different phrasing and summaries.

## RAG Pipeline

The PolicyRAG class retrieves relevant policy chunks, builds a prompt, and generates answers using gemma-4-31b-it:free (temperature=0.0, seed=42). It also handles missing results, API errors, and duplicate citations.

## System Prompt

The model is restricted to indexed company policy documents only. It provides concise, professional answers (2–3 sentences), cites all factual statements, combines sources only when supported, and returns a fallback message when information is unavailable.

## Evaluation and Deployment

The system was evaluated on 20 datasets (see evaluation-matrices.md). Deployment on Render was started, but due to free-tier limitations, the project is demonstrated locally.

