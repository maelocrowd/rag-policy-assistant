import os
import glob
from pathlib import Path
from dotenv import load_dotenv

# Import LangChain utilities
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load environment variables
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# UPDATED: Targets the root-level or local /documents folder
CORPUS_DIR = os.getenv("CORPUS_DIR", "documents/")

def load_and_chunk_markdown_files(directory_path: str):
    """Loads markdown files and splits them preserving header hierarchies."""
    
    # Define headers to track during splitting
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    # Initialize splitters
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False  # Keeps markdown headers inside chunk text for system context
    )
    
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    final_chunks = []
    
    # UPDATED: Evaluates both standard .md and extended .markdown file definitions recursively
    md_files = []
    for ext in ("/**/*.md", "/**/*.markdown"):
        md_files.extend(glob.glob(directory_path + ext, recursive=True))
        
    print(f"Found {len(md_files)} Markdown file(s) inside {directory_path} for ingestion.")
    
    for file_path in md_files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            print(f"Skipping file due to read error {file_path}: {e}")
            continue
            
        # First layer: Split cleanly by Markdown headers
        header_splits = markdown_splitter.split_text(file_content)
        
        # Second layer: Sub-split any section that exceeds our chunk size limit
        for chunk in header_splits:
            # Inject file origin into metadata for accurate citations later
            chunk.metadata["source"] = os.path.basename(file_path)
            chunk.metadata["file_path"] = file_path
            
            if len(chunk.page_content) > CHUNK_SIZE:
                # Sub-split oversized segments recursively
                sub_chunks = recursive_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
                
    return final_chunks

def main():
    # Target directory verification
    if not os.path.exists(CORPUS_DIR):
        print(f"Error: Target directory '{CORPUS_DIR}' does not exist on this machine.")
        return

    # 2. Extract and structure chunks
    print("Beginning document chunking strategy...")
    chunks = load_and_chunk_markdown_files(CORPUS_DIR)
    
    if not chunks:
        print("No documents were processed. Pipeline terminating.")
        return
        
    print(f"Generated {len(chunks)} total text chunks.")
    
    # 3. Initialize local open-source embeddings model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 4. Generate embeddings and save vectors to ChromaDB disk persistent storage
    print(f"Initializing and storing vectors inside {CHROMA_DIR}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print("Ingestion execution completed successfully!")

if __name__ == "__main__":
    main()
