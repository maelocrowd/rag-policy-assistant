# utils.py

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables once
load_dotenv()

# ============================================================================
# Environment Configuration
# ============================================================================

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
CORPUS_DIR = os.getenv("CORPUS_DIR", "documents/")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K = int(os.getenv("TOP_K", 4))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-31b-it:free")
SEED = int(os.getenv("SEED", 42))


MAX_DISTANCE_THRESHOLD = float(
    os.getenv("MAX_DISTANCE_THRESHOLD", 1.1)
)

# ============================================================================
# Shared Helper Functions
# ============================================================================

def get_embeddings():
    """
    Returns a HuggingFace embedding model instance.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def get_vector_store():
    """
    Returns a Chroma vector store instance connected to the
    persisted database.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Chroma directory '{CHROMA_DIR}' missing. "
            "Please make sure to run your document ingestion pipeline first."
        )

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )
