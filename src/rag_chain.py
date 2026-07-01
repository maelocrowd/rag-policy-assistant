import re
import openai

from .prompts import SYSTEM_PROMPT
from .retrieval import PolicyRetriever
from .utils import (
    LLM_MODEL,
    OPENROUTER_API_KEY,
    SEED,
)


class PolicyRAG:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline.
    """

    def __init__(self):
        """Initializes the retriever and OpenRouter client."""
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "API_KEY":
            raise ValueError(
                "Please replace the placeholder 'API_KEY' "
                "with your actual OpenRouter API key "
                "inside your .env file."
            )

        self.retriever = PolicyRetriever()
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

    def retrieve_context(self, question: str):
        """Retrieves relevant policy documents from vector store."""
        documents = self.retriever.retrieve_context(question)
        return documents if documents else []

    def build_prompt(self, question: str, context: str):
        """Builds the user prompt using the retrieved context."""
        return (
            "--- START RETRIEVED POLICY CONTEXT ---\n"
            f"{context}\n"
            "--- END RETRIEVED POLICY CONTEXT ---\n\n"
            f"Employee Question: {question}\n"
            "Answer:"
        )

    def generate(self, question: str) -> dict:
        """
        Executes the complete RAG pipeline exactly once.
        
        Returns:
            dict: {
                "answer": str,
                "sources": list of dicts (UI readable),
                "source_documents": list of raw document chunks (Evaluation engine readable)
            }
        """
        documents = self.retrieve_context(question)

        if not documents:
            return {
                "answer": "I can only answer about our policies within the available company policy documents.",
                "sources": [],
                "source_documents": []
            }

        context = self.retriever.format_context(documents)
        prompt = self.build_prompt(question, context)

        try:
            print("\n" + "=" * 80)
            print(f"[RAG Pipeline Executing Query]: {question}")
            print("==================================================\n")
            
            # Send payload with explicit OpenRouter tracking validation headers
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                seed=SEED,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8501", 
                    "X-Title": "NileTech Policy Assistant",   
                }
            )

            # Safety Guard: Intercept structural server faults or connectivity timeouts
            if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
                return {
                    "answer": "⚠️ OpenRouter returned an empty validation object. Verify API quota balances.",
                    "sources": [],
                    "source_documents": documents
                }

            answer = response.choices[0].message.content
            
            # Catch raw HTML errors emitted by platform routing proxies
            if str(answer).strip().startswith("<!DOCTYPE html") or "clerk.openrouter.ai" in str(answer):
                return {
                    "answer": "🔒 **Authentication Challenge**: Request blocked by proxy layer challenge check.",
                    "sources": [],
                    "source_documents": documents
                }
            
            # --- POST-PROCESSING & SOURCE CITATION FILTERING ---
            fallback_phrases = ["i can only answer", "not present in the indexed corpus", "available company policy"]
            if any(phrase in answer.lower() for phrase in fallback_phrases):
                return {
                    "answer": answer,
                    "sources": [],
                    "source_documents": documents
                }

            # Gather cited markers within the output text stream
            citations = re.findall(r'\[([^\]]+)\]', answer)
            citations_clean = [c.strip().lower() for c in citations]

            seen = set()
            sources = []

            for doc in documents:
                source_name = doc.metadata.get("source", "Unknown Document")
                clean_filename = source_name.lower()
                
                # Deduplication and baseline evaluation filtering match rules
                is_cited = any(cit in clean_filename for cit in citations_clean) or (len(documents) == 1)
                if not is_cited:
                    continue  

                # Retain structural data framing for downstream evaluations
                snippet = doc.page_content.strip()

                headers = []
                for level in ("Header 1", "Header 2", "Header 3", "Header 4"):
                    if level in doc.metadata:
                        headers.append(doc.metadata[level])

                section = " > ".join(headers) if headers else "General Section"

                dup_key = (source_name, section)
                if dup_key not in seen:
                    seen.add(dup_key)
                    sources.append({
                        "source": source_name,
                        "section": section,
                        "snippet": snippet[:300] + "..." if len(snippet) > 300 else snippet,
                    })

            return {
                "answer": answer,
                "sources": sources,
                "source_documents": documents  # Critical key added here to maximize evaluation efficiency
            }

        except Exception as e:
            return {
                "answer": f"Pipeline connection error via OpenRouter API:\n{e}",
                "sources": [],
                "source_documents": documents if 'documents' in locals() else []
            }