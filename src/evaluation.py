import json
import time
import os
import string
import re
import numpy as np
from tabulate import tabulate
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from .rag_chain import PolicyRAG
from .utils import (
    LLM_MODEL,
    OPENROUTER_API_KEY
)

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "eval_dataset.json"))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

# Initialize application pipeline globally once
rag_pipeline = PolicyRAG()

def run_llm_application(case_id: str, question: str) -> dict:
    """
    Executes the PolicyRAG pipeline cleanly via a single-pass payload.
    Leverages the optimized single-pass source_documents return array.
    """
    start_time = time.time()
    result = rag_pipeline.generate(question)
    latency = time.time() - start_time

    generated_answer = result.get("answer", "")
    retrieved_docs = result.get("source_documents", [])

    if retrieved_docs:
        retrieved_context = "\n\n".join(
            doc.page_content if hasattr(doc, 'page_content') else str(doc)
            for doc in retrieved_docs
        )
    else:
        retrieved_context = "No context retrieved."

    documents, sections = extract_citations_from_answer(generated_answer)

    return {
        "id": case_id,
        "generated_answer": generated_answer,
        "retrieved_context": retrieved_context,
        "cited_documents": documents,
        "cited_sections": sections,
        "latency": latency
    }

def extract_citations_from_answer(answer: str):
    """
    Extracts every cited document and section from the generated answer text layer.
    """
    documents = []
    sections = []

    if not answer:
        return documents, sections

    citations = re.findall(r"\[(.*?)\]", answer)

    for citation in citations:
        citation = citation.strip()

        # Pattern: [Section: xxx, file.md]
        m1 = re.search(r"Section:\s*(.*?),\s*([A-Za-z0-9_.\-]+\.(?:md|pdf|txt|html))", citation, re.IGNORECASE)
        if m1:
            sections.append(m1.group(1).strip())
            documents.append(m1.group(2).strip())
            continue

        # Pattern: [file.md, Section: xxx]
        m2 = re.search(r"([A-Za-z0-9_.\-]+\.(?:md|pdf|txt|html))\s*,\s*Section:\s*(.*)", citation, re.IGNORECASE)
        if m2:
            documents.append(m2.group(1).strip())
            sections.append(m2.group(2).strip())
            continue

        # Pattern: Standalone file marker [file.md]
        m3 = re.search(r"([A-Za-z0-9_.\-]+\.(?:md|pdf|txt|html))", citation, re.IGNORECASE)
        if m3:
            documents.append(m3.group(1).strip())
            sections.append("General Section")

    return documents, sections

# ---------------------------------------------------------
# FREE-TIER SELF-HEALING GRADING LOGIC (LLM-AS-A-JUDGE)
# ---------------------------------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=15),
    reraise=False
)
def llm_judge_grade(prompt: str) -> bool:
    """
    Queries OpenRouter using exponential backoff to handle 429 rate limits.
    Intentionally pauses execution to stay safe under the 15 RPM platform threshold.
    """
    time.sleep(2.0)
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict QA evaluator. Respond with exactly 'YES' or 'NO'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        verdict = response.choices[0].message.content.strip().upper()
        return "YES" in verdict
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower() or "exhausted" in str(e).lower():
            print(f"⚠️ [API Blocked - 429 Rate Limit]: Activating exponential backoff retry rule...")
            raise e  
        
        print(f"Evaluator API structural error: {e}")
        return False

def evaluate_groundedness(answer: str, context: str) -> bool:
    """Checks if the answer contains any information absent from the retrieved text context."""
    if not context or "No context retrieved" in context:
        return False

    prompt = f"""
    Context: {context}
    Answer: {answer}
    
    Determine whether every factual claim in the answer is supported by at least one statement in the retrieved context.
    Minor rewording, summarization, or combining multiple retrieved passages should still be considered grounded.
    Only answer NO if the response introduces unsupported factual claims.
    Respond only YES or NO.
    """
    return llm_judge_grade(prompt)

def evaluate_semantic_completeness(generated_answer: str, gold_answer: str) -> bool:
    """
    LLM-Judge metric that checks if the generated answer covers all core facts 
    present within the target golden answer, ignoring cosmetic formatting (Exact Match Requirement).
    """
    if not generated_answer or not gold_answer:
        return False

    prompt = f"""
    Target Correct Answer: {gold_answer}
    Generated System Answer: {generated_answer}
    
    Compare the generated system answer against the target correct answer. 
    Verify if the generated answer contains all the core factual rules, timelines, or numbers present in the target correct answer.
    
    Ignore cosmetic formatting differences, bullet points, or exact word choices.
    Respond with exactly 'YES' if the core information matches completely, otherwise respond 'NO'.
    """
    return llm_judge_grade(prompt)

def evaluate_citation_accuracy(cited_docs, cited_secs, gold_documents, gold_sections):
    """
    Validates generated model citations against expected golden evaluation targets.
    """
    def normalize(text):
        text = str(text).lower()
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"\.md|\.pdf|\.txt|\.html", "", text)
        text = re.sub(r"[^a-z0-9 ]", "", text)
        return " ".join(text.split())

    if not isinstance(gold_documents, list): gold_documents = [gold_documents]
    if not isinstance(gold_sections, list): gold_sections = [gold_sections]

    gold_documents = [normalize(x) for x in gold_documents]
    gold_sections = [normalize(x) for x in gold_sections]
    cited_docs = [normalize(x) for x in cited_docs]
    cited_secs = [normalize(x) for x in cited_secs]

    for doc, sec in zip(cited_docs, cited_secs):
        for gold_doc, gold_sec in zip(gold_documents, gold_sections):
            doc_match = (doc == gold_doc or doc in gold_doc or gold_doc in doc)
            sec_match = (sec == gold_sec or sec in gold_sec or gold_sec in sec)
            if doc_match and sec_match:
                return True
    return False

def evaluate_match_metrics(generated_answer: str, gold_answer: str) -> tuple:
    """
    Calculates standard token string equivalence alongside partial token F1-Score metrics.
    """
    def clean_text(text: str) -> list:
        text = text.strip().lower()
        text = re.sub(r'\[[^\]]+\]', '', text)  
        text = text.replace("**", "").replace("__", "").replace("`", "")
        return text.translate(str.maketrans('', '', string.punctuation)).split()

    gen_words = clean_text(generated_answer)
    gold_words = clean_text(gold_answer)
    
    if not gen_words or not gold_words:
        return False, 0.0

    gen_set, gold_set = set(gen_words), set(gold_words)
    overlap = gen_set.intersection(gold_set)
    
    precision = len(overlap) / len(gen_set)
    recall = len(overlap) / len(gold_set)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    partial_match = f1_score >= 0.70
    return partial_match, f1_score

# ---------------------------------------------------------
# PACED SEQUENTIAL RUNTIME CORE ENGINE
# ---------------------------------------------------------
# ---------------------------------------------------------
# PACED SEQUENTIAL RUNTIME CORE ENGINE
# ---------------------------------------------------------
def main():
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"Error: Evaluation data matrix could not be resolved at: {EVAL_DATASET_PATH}")
        return

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Note: To run your temporary 10-item diagnostic slice, uncomment the line below:
    # dataset = dataset[:10]

    print(f"🚀 Launching paced sequential evaluation framework across {len(dataset)} suite profiles...")
    eval_results = []
    start_eval_time = time.time()

    for idx, case in enumerate(dataset):
        case_id = case.get("id", "UNKNOWN")
        print(f"[{idx+1}/{len(dataset)}] Processing Evaluation Matrix Profile: {case_id}...")
        
        # 1. Execute RAG application inference task
        app_output = run_llm_application(case_id, case.get("question", ""))
        
        # 2. Add an explicit safety buffer to protect OpenRouter free rate states from query overlap
        time.sleep(2.0)
        
        # 3. Fire off evaluation criteria checks sequentially
        is_grounded = evaluate_groundedness(app_output["generated_answer"], app_output["retrieved_context"])
        is_citation_accurate = evaluate_citation_accuracy(
            app_output["cited_documents"], app_output["cited_sections"], 
            case.get("document_references", []), case.get("section_references", [])
        )
        semantic_match = evaluate_semantic_completeness(app_output["generated_answer"], case.get("answer", ""))
        partial_m, token_f1 = evaluate_match_metrics(app_output["generated_answer"], case.get("answer", ""))
        
        eval_results.append({
            "id": case_id,
            "latency": app_output["latency"],
            "grounded": is_grounded,
            "citation_accurate": is_citation_accurate,
            "exact_match": semantic_match,
            "partial_match": partial_m,
            "f1_score": token_f1
        })

    total_eval_duration = time.time() - start_eval_time

    # Aggregate evaluation metrics
    latencies = [r["latency"] for r in eval_results]
    grounded_rate = sum(1 for r in eval_results if r["grounded"]) / len(eval_results) * 100
    citation_rate = sum(1 for r in eval_results if r["citation_accurate"]) / len(eval_results) * 100
    exact_rate = sum(1 for r in eval_results if r["exact_match"]) / len(eval_results) * 100
    partial_rate = sum(1 for r in eval_results if r["partial_match"]) / len(eval_results) * 100
    avg_f1_score = np.mean([r["f1_score"] for r in eval_results]) * 100

    # Format reports
    quality_table = [
        ["Groundedness (Fact Consistency)", f"{grounded_rate:.2f}%"],
        ["Citation Accuracy (Attribution)", f"{citation_rate:.2f}%"],
        ["Exact Match (Semantic Completeness)", f"{exact_rate:.2f}%"],
        ["Partial Match (F1 Overlap >= 70%)", f"{partial_rate:.2f}%"],
        ["Average Semantic Token F1 Score", f"{avg_f1_score:.2f}%"]
    ]

    performance_table = [
        ["p50 (Median Latency)", f"{np.median(latencies):.3f} seconds"],
        ["p95 (Tail Latency)", f"{np.percentile(latencies, 95):.3f} seconds"],
        ["Min / Max Bounds", f"{min(latencies):.3f}s / {max(latencies):.3f}s"],
        ["Total Batch Process Duration", f"{total_eval_duration:.2f} seconds"]
    ]

    print("\n==================================================")
    print("           LLM APPLICATION EVALUATION REPORT      ")
    print("==================================================")
    print("\n### ANSWER QUALITY METRICS")
    print(tabulate(quality_table, headers=["Metric", "Score / Rate"], tablefmt="grid"))
    print("\n### SYSTEM PERFORMANCE METRICS")
    print(tabulate(performance_table, headers=["Latency Metric", "Value"], tablefmt="grid"))
if __name__ == "__main__":
    main()