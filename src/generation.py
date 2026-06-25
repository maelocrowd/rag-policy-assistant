# generation.py
import os
import sys
from dotenv import load_dotenv
import openai

# Add project root to path to ensure clean internal package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval import PolicyRetriever

# 1. Load system configurations
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# FIX: Adjusted to the exact OpenRouter free tier model string identifier
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-2-27b-it:free")
SEED = int(os.getenv("SEED", 42))

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "API_KEY":
    raise ValueError("Error: Please replace the placeholder 'API_KEY' with your actual OpenRouter key inside your .env file.")

# Initialize OpenAI SDK Client targeting OpenRouter endpoint
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 2. Establish RAG System Prompt
SYSTEM_PROMPT = """You are an expert Corporate Policy Assistant. Your goal is to answer employee questions accurately using ONLY the provided policy context blocks.

Strict Operational Guidelines:
1. Grounding: Rely strictly on the facts directly mentioned in the context. Do not invent details or project rules not explicitly detailed in the files.
2. Citations: You must explicitly cite the source document filename (e.g., [employee-handbook.md]) whenever you state a policy guideline or parameter.
3. Out-Of-Bounds Handling: If the retrieved context is empty, or if the provided context does not contain the answer to the user's specific question, reply exactly with:
   "I cannot find the answer to this question within the available company policy documents."
4. Tone: Maintain a professional, clear, objective corporate tone. Avoid fluff.
"""

def generate_policy_response(user_query: str):
    """Retrieves document context and queries the model via OpenRouter."""
    
    # 1. Initialize our tuned retrieval engine and pull chunks
    retriever = PolicyRetriever()
    matched_docs = retriever.retrieve_context(user_query)
    
    # 2. Check if the context passed our distance filtering thresholds
    if not matched_docs:
        return "I cannot find the answer to this question within the available company policy documents."
    
    # 3. Format the text snippets cleanly for prompt construction
    formatted_context = retriever.format_context(matched_docs)
    
    # 4. Construct user payload combining structured context with user's question
    user_payload = (
        f"--- START RETRIEVED POLICY CONTEXT ---\n"
        f"{formatted_context}\n"
        f"--- END RETRIEVED POLICY CONTEXT ---\n\n"
        f"Employee Question: {user_query}\n"
        f"Answer:"
    )
    
    # 5. Route payload to OpenRouter API
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload}
            ],
            temperature=0.0, 
            seed=SEED
        )
        
        # Diagnostic: Catch if the endpoint returns a raw string response error instead of an object
        if isinstance(response, str):
            return f"OpenRouter Server Error Raw String: {response}"
            
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Pipeline connection error via OpenRouter API: {e}"

def main():
    sample_questions = [
        "What is the policy for sexual harassment?",
        "Is remote work allowed? If so, what are the guidelines?",
        "What is the total working hours per week?",
        "What is the total number of paid time off leaves for maternity leave?"
    ]
    
    print("=" * 60)
    print("RUNNING END-TO-END RAG GENERATION TEST SUITE")
    print("=" * 60)
    
    for idx, question in enumerate(sample_questions, 1):
        print(f"\n[TEST {idx}/{len(sample_questions)}] User Query: \"{question}\"")
        print("Processing context retrieval and LLM response generation...")
        
        output_answer = generate_policy_response(question)
        
        print("\n--- LLM DIRECT RESPONSE ---")
        print(output_answer)
        print("---------------------------")
        print("-" * 50)
        
    print("\nGeneration test suite execution completed successfully!")

if __name__ == "__main__":
    main()
