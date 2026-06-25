# Corporate Policy RAG Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) assistant designed to parse, index, and query structured corporate Markdown policy documents. The system uses hierarchical Markdown chunking, local text embeddings via ChromaDB, and open-source models via OpenRouter to deliver accurate answers with source file citations.

---

## 🏢 Company Context & Scope

This RAG engine is custom-built for **NileTech Solutions Ltd.** to automate employee policy clarification, reduce internal HR overhead, and enforce compliance across teams. 

The engine operates on a fixed corporate knowledge domain and is restricted from discussing non-corporate information. It evaluates internal compliance parameters using data pulled directly from the official corporate knowledge corpus.

### 📄 Documents Handled
The pipeline processes the following core operational documents located in the `/documents` folder:
*   `employee-handbook.md`: Outlines working conditions, shift schedules, standard working hours, and general codes of conduct.
*   `PTO-policy.md`: Details leave scales, annual leave administration rules, postponement limits, and specialized leaves (maternity, paternity, and child treatment leave).
*   `IT_assets_and_acceptable_use_policy.md`: Enforces information security responsibilities, acceptable hardware/software utilization boundaries, password rules, and asset tracking.
*   `security-policy.md`: Details corporate physical security parameters and logical software repository protections.
*   `remote-work-policy.md`: Manages remote work communication standards and core accessibility hours.

---

## 🛠️ Tech Stack & Architecture

Use code with caution.+-------------------+| /documents Folder |+---------+---------+|v+-------------------+             +---------+---------+| HuggingFace       |             | Ingestion Engine  || MiniLM Embeddings +------------>| (Dual-Layer       |+-------------------+             |  Splitter)        |+---------+---------+|+-------------------+                       v| Persistent        |<----------------------+| ChromaDB Store    |+---------+---------+|v+---------+---------+             +---------+---------+| Retrieval Engine  |             | OpenRouter API    || (L2 Distance      +------------>| (Gemma-2-27B-IT   ||  Filter <= 1.25)  |             |  Deterministic)   |+-------------------+             +---------+---------+|v+---------+---------+| Factual Answer    || + Citations       |+-------------------+
*   **Orchestration Framework:** `langchain` (v0.3+) & `langchain-community` — manages modular data extraction pipelines.
*   **Vector Database:** `langchain-chroma` — disk-persistent vector indexing operating locally.
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (via `langchain-huggingface`) — generates 384-dimensional dense vectors locally.
*   **Large Language Model:** `google/gemma-2-27b-it:free` (via OpenRouter API) — utilizes a zero-temperature (`0.0`) configuration for deterministic policy retrieval.

---

## 🚀 Installation & Local Execution

### 1. Environment Setup
Clone the repository to your local machine and initialize an isolated virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the verified production dependency tree
pip install langchain langchain-community langchain-chroma langchain-huggingface markdown openai python-dotenv
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory of the project and populate it with your OpenRouter token and hyperparameters:

```env
OPENROUTER_API_KEY=your_actual_openrouter_token_here
LLM_MODEL=google/gemma-2-27b-it:free
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=data/chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=4
SEED=42
```

### 3. Execution Pipeline

#### Step 1: Run Ingestion
Parses raw policy documents inside `/documents`, triggers the dual-layer splitter, generates vector weights locally, and persists the index to disk:
```bash
python ingestion.py
```

#### Step 2: Run Retrieval & Generation Test Suite
Queries Chroma, applies the mathematical threshold filtering layer, and calls OpenRouter to generate direct answers with citations:
```bash
python src/generation.py
```

---

## 🧪 Testing Suite & Validation Matrix

The application includes an automated validation suite inside `src/generation.py` to evaluate retrieval precision and model grounding across four distinct compliance bounds:

```bash
python src/generation.py
```

### Evaluation Behaviors Matrix

| Test Scenario | Sample Input Query | Expected Pipeline Action & Safeguard |
| :--- | :--- | :--- |
| **1. Missing Operational Context** | *"What is the policy for requesting time off?"* | Chunks are retrieved based on keywords, but since the step-by-step *how-to submission* text is absent from the file, the model successfully catches the grounding gap and flags it without hallucinating. |
| **2. Precise Metric Match** | *"What is the total number of paid time off leaves for paternity leave?"* | Identifies a tight semantic chunk (L2 distance ~`0.64`), extracts the exact metric ("10 working days"), and formats a clear response containing explicit `[PTO-policy.md]` citations. |
| **3. Multi-Document Aggregation** | *"What are the basic security protocols for laptop usage?"* | Scans across sections, executes internal content deduplication, and formats a clean bulleted layout split by responsibilities and restrictions. |
| **4. Out-of-Bounds Query** | *"Who is brad pitt?"* | Chunks return extreme L2 distances (>1.7). The retrieval engine filters out these segments before calling the LLM, triggering the safety fallback response immediately. |

---

## 🔄 CI/CD Pipeline

A continuous integration workflow is configured via **GitHub Actions** (`.github/workflows/ci.yml`) to automatically validate the code structure, enforce formatting rules, and check dependency integrity on every code change.

### GitHub Actions Workflow Configuration
```yaml
name: NileTech RAG CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  validate:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Code Repository
      uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Cache Pip Dependencies
      uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key:  runner.os -pip-{{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          \${{ runner.os }}-pip-

    - name: Install Project Stack
      run: |
        python -m pip install --upgrade pip
        pip install langchain langchain-community langchain-chroma langchain-huggingface markdown openai python-dotenv flake8 pytest

    - name: Lint Code Base (Flake8 Compliance)
      run: |
        # Stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # Exit-zero treats all remaining style warnings as info flags without failing the build
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Run Basic Pipeline Component Verification
      env:
        OPENROUTER_API_KEY: "mock_key_for_structural_test"
        CHROMA_DIR: "data/test_chroma_db"
      run: |
        # Verifies that file system packages compile and imports resolve cleanly
        python -c "from src.retrieval import PolicyRetriever; print('Retrieval Engine Compilation: Pass')"
```