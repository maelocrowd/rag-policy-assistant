# rag_chain.py

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

    Workflow:
        User Question
              ↓
        Retrieve Context
              ↓
        Build Prompt
              ↓
        Query LLM
              ↓
        Return Response
    """

    def __init__(self):
        """Initializes the retriever and OpenRouter client."""

        if (
            not OPENROUTER_API_KEY
            or OPENROUTER_API_KEY == "API_KEY"
        ):
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
        """
        Retrieves relevant policy documents.
        """

        documents = self.retriever.retrieve_context(question)

        if not documents:
            return None

        return documents

    def build_prompt(self,question: str,context: str):
        """
        Builds the user prompt using the retrieved context.
        """

        return (
            "--- START RETRIEVED POLICY CONTEXT ---\n"
            f"{context}\n"
            "--- END RETRIEVED POLICY CONTEXT ---\n\n"
            f"Employee Question: {question}\n"
            "Answer:"
        )

    def generate(self, question: str):
        """
        Executes the complete RAG pipeline.
        """

        documents = self.retrieve_context(question)

        if documents is None:
            return {
                "answer": (
                    "I can only answer about our policies within the available company policy documents."
                ),
                "sources": [],
            }

        context = self.retriever.format_context(documents)

        prompt = self.build_prompt(
            question,
            context,
        )

        try:

            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                seed=SEED,
            )

            if isinstance(response, str):
                return {
                    "answer": (
                        "OpenRouter returned an unexpected response:\n"
                        f"{response}"
                    ),
                    "sources": [],
                }

            answer = response.choices[0].message.content

            sources = []

            for doc in documents:
                sources.append(
                    {
                        "source": doc.metadata.get("source", "Unknown Document"),
                        "snippet": doc.page_content.strip()[:300],
                    }
                )

            return {
                "answer": answer,
                "sources": sources,
            }

        except Exception as e:
            return {
                "answer": (
                    "Pipeline connection error via OpenRouter API:\n"
                    f"{e}"
                ),
                "sources": [],
            }

    