import os
# Force underlying Hugging Face pipelines to run purely local and skip online HEAD requests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import time
import string
import re
import numpy as np
from tabulate import tabulate
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from .rag_chain import PolicyRAG
from .utils import (
    LLM_MODEL,
    OPENROUTER_API_KEY,
    SEED,
    OPENAI_BASE_URL
)

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "eval_dataset.json"))
OPENROUTER_BASE_URL = OPENAI_BASE_URL

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
            temperature=0.0,
            seed=SEED,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return "YES" in verdict
    except Exception as e:
        # Force a retry on ALL exceptions to capture custom routing and API issues safely
        print(f"⚠️ [API Failure Encountered]: Activating evaluation retry logic window... Details: {e}")
        raise e  

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

def evaluate_match_percentage(generated_answer: str, gold_answer: str) -> float:
    """
    Calculates the percentage of partial token overlap match between the generated answer and the golden answer.
    Returns a score between 0.0 and 100.0.
    """
    def clean_text(text: str) -> list:
        text = text.strip().lower()
        text = re.sub(r'\[[^\]]+\]', '', text)  
        text = text.replace("**", "").replace("__", "").replace("`", "")
        return text.translate(str.maketrans('', '', string.punctuation)).split()

    gen_words = clean_text(generated_answer)
    gold_words = clean_text(gold_answer)
    
    if not gen_words or not gold_words:
        return 0.0

    gen_set, gold_set = set(gen_words), set(gold_words)
    overlap = gen_set.intersection(gold_set)
    
    # Calculate percentage based on matched tokens over the golden baseline reference targets
    match_percentage = (len(overlap) / len(gold_set)) * 100.0
    return match_percentage

def load_evaluation_dataset(dataset_path: str) -> list:
    """
    Loads and returns the golden validation evaluation JSON test suite.
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Evaluation dataset file missing at: {dataset_path}")
        return []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation_suite():
    """
    Main execution orchestration entrypoint loop.
    Iterates test datasets, executes pipelines, runs metrics, and builds reports.
    """
    overall_start_time = time.time()
    print("🚀 Starting RAG Evaluation Engine Pipeline...")
    dataset = load_evaluation_dataset(EVAL_DATASET_PATH)
    
    if not dataset:
        print("Stopping engine execution. No test rows present.")
        return

    results_table = []
    
    total_cases = len(dataset)
    grounded_count = 0
    citation_correct_count = 0
    match_percentages = []
    latencies = []

    for idx, case in enumerate(dataset, start=1):
        case_id = case.get("id", f"case_{idx}")
        question = case.get("question", "")
        gold_answer = case.get("answer", "")
        gold_docs = case.get("document_references", [])
        gold_sections = case.get("section_references", [])

        print(f"\n[Case {idx}/{total_cases}] Processing Case ID: {case_id}...")
        
        # Execute application logic layer pass
        payload = run_llm_application(case_id, question)
        
        gen_answer = payload["generated_answer"]
        context = payload["retrieved_context"]
        cited_docs = payload["cited_documents"]
        cited_secs = payload["cited_sections"]
        latency = payload["latency"]
        
        latencies.append(latency)

        # Evaluate standard metric matrix layers
        is_grounded = evaluate_groundedness(gen_answer, context)
        is_citation_accurate = evaluate_citation_accuracy(
            cited_docs, cited_secs, gold_docs, gold_sections
        )
        match_pct = evaluate_match_percentage(gen_answer, gold_answer)

        # Aggregate metric layers
        if is_grounded: grounded_count += 1
        if is_citation_accurate: citation_correct_count += 1
        match_percentages.append(match_pct)

        # Track results for reporting rows
        results_table.append([
            case_id,
            "PASS" if is_grounded else "FAIL",
            "PASS" if is_citation_accurate else "FAIL",
            f"{match_pct:.1f}%",
            f"{latency:.2f}s"
        ])

    total_batch_duration = time.time() - overall_start_time

    # Compute macro summary system metrics
    avg_groundedness = (grounded_count / total_cases) * 100.0 if total_cases > 0 else 0.0
    avg_citation = (citation_correct_count / total_cases) * 100.0 if total_cases > 0 else 0.0
    avg_match_pct = np.mean(match_percentages) if match_percentages else 0.0

    # Compute advanced latency performance distribution layers
    p50_latency = np.percentile(latencies, 50) if latencies else 0.0
    p95_latency = np.percentile(latencies, 95) if latencies else 0.0
    min_latency = np.min(latencies) if latencies else 0.0
    max_latency = np.max(latencies) if latencies else 0.0

#     from tabulate import tabulate

# # Open the markdown file in write mode ('w')
#     with open("evaluation-matrices.md", "w", encoding="utf-8") as f:
#         # 1. Write the Detailed Case Results Layer
#         f.write("# EVALUATION METRICS REPORT\n\n")
#         f.write("## DETAILED CASE RESULTS LAYER\n")
#         f.write("```text\n")
#         headers = ["Case ID", "Grounded", "Citation Acc.", "Match Pct", "Latency"]
#         f.write(tabulate(results_table, headers=headers, tablefmt="grid"))
#         f.write("\n```\n\n")

#         # 2. Write the Answer Quality Metrics
#         f.write("## ANSWER QUALITY METRICS\n")
#         f.write("```text\n")
#         quality_summary = [
#             ["Groundedness (Fact Consistency)", f"{avg_groundedness:.2f}%"],
#             ["Citation Accuracy (Attribution)", f"{avg_citation:.2f}%"],
#             ["Average Match Percentage", f"{avg_match_pct:.2f}%"],
#         ]
#         f.write(tabulate(quality_summary, headers=["Metric", "Score / Rate"], tablefmt="grid"))
#         f.write("\n```\n\n")

#         # 3. Write the System Performance Metrics
#         f.write("## SYSTEM PERFORMANCE METRICS\n")
#         f.write("```text\n")
#         performance_summary = [
#             ["p50 (Median Latency)", f"{p50_latency:.3f} seconds"],
#             ["p95 (Tail Latency)", f"{p95_latency:.3f} seconds"],
#             ["Min / Max Bounds", f"{min_latency:.3f}s / {max_latency:.3f}s"],
#             ["Total Batch Process Duration", f"{total_batch_duration:.2f} seconds"]
#         ]
#         f.write(tabulate(performance_summary, headers=["Latency Metric", "Value"], tablefmt="grid"))
#         f.write("\n```\n")

#     print("Evaluation results successfully saved to evaluation-matrices.md")




    print("\n" + "="*60)
    print("   DETAILED CASE RESULTS LAYER")
    print("="*60)
    headers = ["Case ID", "Grounded", "Citation Acc.", "Match Pct", "Latency"]
    print(tabulate(results_table, headers=headers, tablefmt="grid"))

    print("### ANSWER QUALITY METRICS")
    quality_summary = [
        ["Groundedness (Fact Consistency)", f"{avg_groundedness:.2f}%"],
        ["Citation Accuracy (Attribution)", f"{avg_citation:.2f}%"],
        [" Average Match Percentage", f"{avg_match_pct:.2f}%"],
    ]
    print(tabulate(quality_summary, headers=["Metric", "Score / Rate"], tablefmt="grid"))
    print("|\n")

    print("### SYSTEM PERFORMANCE METRICS")
    performance_summary = [
        ["p50 (Median Latency)", f"{p50_latency:.3f} seconds"],
        ["p95 (Tail Latency)", f"{p95_latency:.3f} seconds"],
        ["Min / Max Bounds", f"{min_latency:.3f}s / {max_latency:.3f}s"],
        ["Total Batch Process Duration", f"{total_batch_duration:.2f} seconds"]
    ]
    print(tabulate(performance_summary, headers=["Latency Metric", "Value"], tablefmt="grid"))

if __name__ == "__main__":
    run_evaluation_suite()



# import json
# import time
# import os
# import string
# import re
# import numpy as np
# from tabulate import tabulate
# from openai import OpenAI
# from tenacity import retry, stop_after_attempt, wait_exponential
# from .rag_chain import PolicyRAG
# from .utils import (
#     LLM_MODEL,
#     OPENROUTER_API_KEY
# )

# # ---------------------------------------------------------
# # CONSTANTS & CONFIGURATION
# # ---------------------------------------------------------
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# EVAL_DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "eval_dataset.json"))
# OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

# # Initialize application pipeline globally once
# rag_pipeline = PolicyRAG()

# def run_llm_application(case_id: str, question: str) -> dict:
#     """
#     Executes the PolicyRAG pipeline cleanly via a single-pass payload.
#     Leverages the optimized single-pass source_documents return array.
#     """
#     start_time = time.time()
#     result = rag_pipeline.generate(question)
#     latency = time.time() - start_time

#     generated_answer = result.get("answer", "")
#     retrieved_docs = result.get("source_documents", [])

#     if retrieved_docs:
#         retrieved_context = "\n\n".join(
#             doc.page_content if hasattr(doc, 'page_content') else str(doc)
#             for doc in retrieved_docs
#         )
#     else:
#         retrieved_context = "No context retrieved."

#     documents, sections = extract_citations_from_answer(generated_answer)

#     return {
#         "id": case_id,
#         "generated_answer": generated_answer,
#         "retrieved_context": retrieved_context,
#         "cited_documents": documents,
#         "cited_sections": sections,
#         "latency": latency
#     }

# def extract_citations_from_answer(answer: str):
#     """
#     Extracts every cited document and section from the generated answer text layer.
#     """
#     documents = []
#     sections = []

#     if not answer:
#         return documents, sections

#     citations = re.findall(r"\[(.*?)\]", answer)

#     for citation in citations:
#         citation = citation.strip()

#         # Pattern: [Section: xxx, file.md]
#         m1 = re.search(r"Section:\s*(.*?),\s*([A-Za-z0-9_.\-]+\.(?:md|pdf|txt|html))", citation, re.IGNORECASE)
#         if m1:
#             sections.append(m1.group(1).strip())
#             documents.append(m1.group(2).strip())
#             continue

#         # Pattern: [file.md, Section: xxx]
#         m2 = re.search(r"([A-Za-z0-9_.\-]+\.(?:md|pdf|txt|html))\s*,\s*Section:\s*(.*)", citation, re.IGNORECASE)
#         if m2:
#             documents.append(m2.group(1).strip())
#             sections.append(m2.group(2).strip())
#             continue

#         # Pattern: Standalone file marker [file.md]
#         m3 = re.search(r"([A-Za-z0-9_.\-]+\.(?:md|pdf|txt|html))", citation, re.IGNORECASE)
#         if m3:
#             documents.append(m3.group(1).strip())
#             sections.append("General Section")

#     return documents, sections

# # ---------------------------------------------------------
# # FREE-TIER SELF-HEALING GRADING LOGIC (LLM-AS-A-JUDGE)
# # ---------------------------------------------------------
# @retry(
#     stop=stop_after_attempt(5),
#     wait=wait_exponential(multiplier=2, min=4, max=15),
#     reraise=False
# )
# def llm_judge_grade(prompt: str) -> bool:
#     """
#     Queries OpenRouter using exponential backoff to handle 429 rate limits.
#     Intentionally pauses execution to stay safe under the 15 RPM platform threshold.
#     """
#     time.sleep(2.0)
    
#     try:
#         response = client.chat.completions.create(
#             model=LLM_MODEL,
#             messages=[
#                 {"role": "system", "content": "You are a strict QA evaluator. Respond with exactly 'YES' or 'NO'."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.0
#         )
#         verdict = response.choices[0].message.content.strip().upper()
#         return "YES" in verdict
#     except Exception as e:
#         if "429" in str(e) or "rate" in str(e).lower() or "quota" in str(e).lower() or "exhausted" in str(e).lower():
#             print(f"⚠️ [API Blocked - 429 Rate Limit]: Activating exponential backoff retry rule...")
#             raise e  
        
#         print(f"Evaluator API structural error: {e}")
#         return False

# def evaluate_groundedness(answer: str, context: str) -> bool:
#     """Checks if the answer contains any information absent from the retrieved text context."""
#     if not context or "No context retrieved" in context:
#         return False

#     prompt = f"""
#     Context: {context}
#     Answer: {answer}
    
#     Determine whether every factual claim in the answer is supported by at least one statement in the retrieved context.
#     Minor rewording, summarization, or combining multiple retrieved passages should still be considered grounded.
#     Only answer NO if the response introduces unsupported factual claims.
#     Respond only YES or NO.
#     """
#     return llm_judge_grade(prompt)

# def evaluate_citation_accuracy(cited_docs, cited_secs, gold_documents, gold_sections):
#     """
#     Validates generated model citations against expected golden evaluation targets.
#     """
#     def normalize(text):
#         text = str(text).lower()
#         text = text.replace("_", " ").replace("-", " ")
#         text = re.sub(r"\.md|\.pdf|\.txt|\.html", "", text)
#         text = re.sub(r"[^a-z0-9 ]", "", text)
#         return " ".join(text.split())

#     if not isinstance(gold_documents, list): gold_documents = [gold_documents]
#     if not isinstance(gold_sections, list): gold_sections = [gold_sections]

#     gold_documents = [normalize(x) for x in gold_documents]
#     gold_sections = [normalize(x) for x in gold_sections]
#     cited_docs = [normalize(x) for x in cited_docs]
#     cited_secs = [normalize(x) for x in cited_secs]

#     for doc, sec in zip(cited_docs, cited_secs):
#         for gold_doc, gold_sec in zip(gold_documents, gold_sections):
#             doc_match = (doc == gold_doc or doc in gold_doc or gold_doc in doc)
#             sec_match = (sec == gold_sec or sec in gold_sec or gold_sec in sec)
#             if doc_match and sec_match:
#                 return True
#     return False

# def evaluate_match_percentage(generated_answer: str, gold_answer: str) -> float:
#     """
#     Calculates the percentage of partial token overlap match between the generated answer and the golden answer.
#     Returns a score between 0.0 and 100.0.
#     """
#     def clean_text(text: str) -> list:
#         text = text.strip().lower()
#         text = re.sub(r'\[[^\]]+\]', '', text)  
#         text = text.replace("**", "").replace("__", "").replace("`", "")
#         return text.translate(str.maketrans('', '', string.punctuation)).split()

#     gen_words = clean_text(generated_answer)
#     gold_words = clean_text(gold_answer)
    
#     if not gen_words or not gold_words:
#         return 0.0

#     gen_set, gold_set = set(gen_words), set(gold_words)
#     overlap = gen_set.intersection(gold_set)
    
#     # Calculate percentage based on matched tokens over the golden baseline reference targets
#     match_percentage = (len(overlap) / len(gold_set)) * 100.0
#     return match_percentage

# def load_evaluation_dataset(dataset_path: str) -> list:
#     """
#     Loads and returns the golden validation evaluation JSON test suite.
#     """
#     if not os.path.exists(dataset_path):
#         print(f"❌ Evaluation dataset file missing at: {dataset_path}")
#         return []
    
#     with open(dataset_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def run_evaluation_suite():
#     """
#     Main execution orchestration entrypoint loop.
#     Iterates test datasets, executes pipelines, runs metrics, and builds reports.
#     """
#     overall_start_time = time.time()
#     print("🚀 Starting RAG Evaluation Engine Pipeline...")
#     dataset = load_evaluation_dataset(EVAL_DATASET_PATH)
    
#     if not dataset:
#         print("Stopping engine execution. No test rows present.")
#         return

#     results_table = []
    
#     total_cases = len(dataset)
#     grounded_count = 0
#     citation_correct_count = 0
#     match_percentages = []
#     latencies = []

#     for idx, case in enumerate(dataset, start=1):
#         case_id = case.get("id", f"case_{idx}")
#         question = case.get("question", "")
#         gold_answer = case.get("answer", "")
#         gold_docs = case.get("document_references", [])
#         gold_sections = case.get("section_references", [])

#         print(f"\n[Case {idx}/{total_cases}] Processing Case ID: {case_id}...")
        
#         # Execute application logic layer pass
#         payload = run_llm_application(case_id, question)
        
#         gen_answer = payload["generated_answer"]
#         context = payload["retrieved_context"]
#         cited_docs = payload["cited_documents"]
#         cited_secs = payload["cited_sections"]
#         latency = payload["latency"]
        
#         latencies.append(latency)

#         # Evaluate standard metric matrix layers
#         is_grounded = evaluate_groundedness(gen_answer, context)
#         is_citation_accurate = evaluate_citation_accuracy(
#             cited_docs, cited_secs, gold_docs, gold_sections
#         )
#         match_pct = evaluate_match_percentage(gen_answer, gold_answer)

#         # Aggregate metric layers
#         if is_grounded: grounded_count += 1
#         if is_citation_accurate: citation_correct_count += 1
#         match_percentages.append(match_pct)

#         # Track results for reporting rows
#         results_table.append([
#             case_id,
#             "PASS" if is_grounded else "FAIL",
#             "PASS" if is_citation_accurate else "FAIL",
#             f"{match_pct:.1f}%",
#             f"{latency:.2f}s"
#         ])

#     total_batch_duration = time.time() - overall_start_time

#     # Compute macro summary system metrics
#     avg_groundedness = (grounded_count / total_cases) * 100.0 if total_cases > 0 else 0.0
#     avg_citation = (citation_correct_count / total_cases) * 100.0 if total_cases > 0 else 0.0
#     avg_match_pct = np.mean(match_percentages) if match_percentages else 0.0

#     # Compute advanced latency performance distribution layers
#     p50_latency = np.percentile(latencies, 50) if latencies else 0.0
#     p95_latency = np.percentile(latencies, 95) if latencies else 0.0
#     min_latency = np.min(latencies) if latencies else 0.0
#     max_latency = np.max(latencies) if latencies else 0.0

#     print("\n" + "="*60)
#     print("   DETAILED CASE RESULTS LAYER")
#     print("="*60)
#     headers = ["Case ID", "Grounded", "Citation Acc.", "Match Pct", "Latency"]
#     print(tabulate(results_table, headers=headers, tablefmt="grid"))

#     print("### ANSWER QUALITY METRICS")
#     quality_summary = [
#         ["Groundedness (Fact Consistency)", f"{avg_groundedness:.2f}%"],
#         ["Citation Accuracy (Attribution)", f"{avg_citation:.2f}%"],
#         [" Average Match Percentage", f"{avg_match_pct:.2f}%"],
#     ]
#     print(tabulate(quality_summary, headers=["Metric", "Score / Rate"], tablefmt="grid"))
#     print("|\n")

#     print("### SYSTEM PERFORMANCE METRICS")
#     performance_summary = [
#         ["p50 (Median Latency)", f"{p50_latency:.3f} seconds"],
#         ["p95 (Tail Latency)", f"{p95_latency:.3f} seconds"],
#         ["Min / Max Bounds", f"{min_latency:.3f}s / {max_latency:.3f}s"],
#         ["Total Batch Process Duration", f"{total_batch_duration:.2f} seconds"]
#     ]
#     print(tabulate(performance_summary, headers=["Latency Metric", "Value"], tablefmt="grid"))

# if __name__ == "__main__":
#     run_evaluation_suite()

