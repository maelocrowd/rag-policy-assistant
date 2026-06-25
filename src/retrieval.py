# retrieval.py
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load environment variables
load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", 4))

# ADJUSTED FOR DEFAULT CHROMA L2 SPACE:
# For L2 distance, smaller is better (0.0 is an exact match).
# A distance threshold of 1.25 allows valid policy matches while safely filtering out extreme outliers like "Brad Pitt".
MAX_DISTANCE_THRESHOLD = 1.25

class PolicyRetriever:
    def __init__(self):
        """Initializes embeddings and establishes a direct connection to ChromaDB."""
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(f"Chroma directory '{CHROMA_DIR}' missing.")
            
        self.vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )

    def retrieve_context(self, query: str, top_k: int = TOP_K):
        """
        Uses native similarity_search_with_score to extract raw distance scores,
        printing them out to assist with configuration.
        """
        docs_with_distances = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        filtered_docs = []
        for doc, distance in docs_with_distances:
            # Diagnostic print: Helps determine exactly what space Chroma is using
            print(f"   [Debug] Found chunk in '{doc.metadata.get('source')}' with raw distance: {distance:.4f}")
            
            # Keep the chunk only if the distance is below our maximum allowed threshold
            if distance <= MAX_DISTANCE_THRESHOLD:
                doc.metadata["raw_distance"] = round(distance, 4)
                filtered_docs.append(doc)
                
        return filtered_docs

    def format_context(self, docs):
        """Formats retrieved chunks, preventing text block duplication."""
        context_blocks = []
        seen_contents = set()
        
        for idx, doc in enumerate(docs, 1):
            clean_content = doc.page_content.strip()
            
            if clean_content in seen_contents:
                continue
            seen_contents.add(clean_content)
            
            source_file = doc.metadata.get("source", "Unknown Document")
            distance = doc.metadata.get("raw_distance", 0.0)
            
            # Rebuild structural header paths
            header_path = []
            for lvl in ["Header 1", "Header 2", "Header 3", "Header 4"]:
                if lvl in doc.metadata:
                    header_path.append(doc.metadata[lvl])
            structure_str = " > ".join(header_path) if header_path else "General Section"
            
            block = (
                f"[Source Document {idx}]: {source_file} (Raw Distance: {distance})\n"
                f"[Section Hierarchy]: {structure_str}\n"
                f"[Content]:\n{clean_content}\n"
                f"{'-'*40}"
            )
            context_blocks.append(block)
            
        return "\n\n".join(context_blocks)


def main():
    try:
        retriever = PolicyRetriever()
        
        # Define a list of automated sample questions to check your pipeline performance
        sample_questions = [
            "What is the policy for requesting time off?",
            "What is the total number of paid time off leaves for paternity leave?",
            "What are the basic security protocols for laptop usage?",
            "Who is brad pitt?"
        ]
        
        print("=" * 60)
        print("RUNNING AUTOMATED RETRIEVAL TEST SUITE")
        print("=" * 60)
        
        for idx, question in enumerate(sample_questions, 1):
            print(f"\n[TEST {idx}/{len(sample_questions)}] Query: \"{question}\"")
            print(f"Searching vector store for top {TOP_K} matches...")
            
            matched_docs = retriever.retrieve_context(question)
            
            if not matched_docs:
                print(" -> RESULT: No relevant context found inside the vector store.")
                print("-" * 50)
                continue
                
            formatted_text = retriever.format_context(matched_docs)
            print("\n--- RETRIEVED CONTEXT START ---")
            print(formatted_text)
            print("--- RETRIEVED CONTEXT END ---\n")
            print("-" * 50)
            
        print("\nTest suite execution completed successfully!")
            
    except Exception as e:
        print(f"Initialization or runtime error: {e}")

if __name__ == "__main__":
    main()
