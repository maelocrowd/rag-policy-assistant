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

CHROMA_DIR = os.getenv("CHROMA_DIR")
CORPUS_DIR = os.getenv("CORPUS_DIR")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
TOP_K = int(os.getenv("TOP_K"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_API_KEY = os.getenv("GEMINI_API_KEY")  # Use Gemini API key for fallback
LLM_MODEL = os.getenv("LLM_MODEL")
SEED = int(os.getenv("SEED"))


MAX_DISTANCE_THRESHOLD = float( os.getenv("MAX_DISTANCE_THRESHOLD"))

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
