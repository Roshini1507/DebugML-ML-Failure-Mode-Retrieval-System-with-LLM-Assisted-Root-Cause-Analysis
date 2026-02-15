# DebugML: Failure Analysis Assistant

A production-grade RAG (Retrieval-Augmented Generation) system that helps ML engineers debug model failures by retrieving similar past failures and generating root cause analysis and solutions using a local LLM.

---

## Project Overview

DebugML is an AI-powered assistant for diagnosing machine learning model failures. Instead of relying solely on documentation or generic LLM responses, it grounds its analysis in a curated dataset of 15+ real ML failure cases—covering overfitting, exploding/vanishing gradients, data leakage, class imbalance, poor preprocessing, and more.

**How it works:** You describe your failure, the system retrieves similar past failures from a FAISS vector store, and an Ollama-powered LLM synthesizes a root cause analysis and recommended solutions based on those examples.

**Key features:**
-  Semantic search over failure cases (no API keys required)
-  Local LLM inference via Ollama (llama3)
-  Persistent FAISS index (no embedding recomputation on restart)
-  Docker support for one-command deployment
-  Production-ready structure with error handling and logging

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                         │
│                  Failure description → Analyze button                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG Pipeline (rag.py)                            │
│  1. Embed query  2. Retrieve  3. Context  4. LLM  5. Structured out  │
└───────┬─────────────────────────────────────┬───────────────────────┘
        │                                     │
        ▼                                     ▼
┌───────────────────┐               ┌─────────────────────┐
│  VectorDB         │               │  OllamaLLM          │
│  (vector_db.py)   │               │  (llm.py)           │
│  FAISS + metadata │               │  llama3 via Ollama  │
└─────────┬─────────┘               └─────────────────────┘
          │
          ▼
┌───────────────────┐
│  EmbeddingModel   │
│  (embeddings.py)  │
│  all-MiniLM-L6-v2 │
└───────────────────┘
```

| Component | Responsibility |
|-----------|----------------|
| **app.py** | Streamlit interface, input, results display |
| **rag.py** | Orchestrates retrieval + generation, returns structured output |
| **vector_db.py** | Loads failures.json, embeds text, FAISS search, persists index |
| **embeddings.py** | Sentence-transformers (singleton), embed text/batch |
| **llm.py** | Ollama , formats context, generates analysis |
| **data/failures.json** | Curated ML failure dataset with descriptions, root causes, solutions |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **UI** | Streamlit |
| **RAG** | LangChain (optional), custom pipeline |
| **Vector Store** | FAISS (CPU) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **LLM** | Ollama (llama3), local inference |
| **Data** | JSON (failures.json) |
| **Runtime** | Python 3.10, Docker |

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- ~2GB RAM for embeddings model, ~4GB+ for llama3

### 1. Clone and enter the project

```bash
git clone <repository-url>
cd DebugML
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Ollama model

```bash
ollama pull llama3
```

### 5. (Optional) Start Ollama if not already running

```bash
ollama serve
```

---

## Usage

### Run the Streamlit app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

### Workflow

1. Enter your ML failure description in the text area (e.g., *"Model gets 99% on train but 55% on test"*).
2. Click **Analyze Failure**.
3. View:
   - **Similar Past Failures** — Retrieved cases with relevance scores, descriptions, root causes, and solutions.
   - **Root Cause Analysis & Recommended Solutions** — LLM-generated explanation based on those examples.


### Docker

```bash
# Build
docker build -t debugml .

# Run (pull llama3 on first use)
docker run -p 8501:8501 debugml

# In another terminal, pull model if needed
docker exec -it <container_id> ollama pull llama3
```

App: [http://localhost:8501](http://localhost:8501).

---

## Example Screenshots

<!-- Add screenshots here. Example structure: -->

| Main interface | Results view |
|----------------|--------------|
| ![Main interface](data\main.png) | ![Results](data\results.png) |

---

## Future Improvements

- [ ] **Streaming responses** — Stream LLM output token-by-token for faster perceived latency.
- [ ] **Custom dataset** — Allow users to upload or point to their own failure dataset.
- [ ] **Model selection** — UI toggle for different Ollama models (llama3, mistral, etc.).
- [ ] **Confidence scoring** — Surface retrieval confidence and LLM certainty.
- [ ] **Feedback loop** — Let users mark results as helpful or submit new failure cases.
- [ ] **Multi-turn chat** — Follow-up questions without re-running retrieval.
- [ ] **GPU support** — Optional FAISS-GPU and CUDA for sentence-transformers.
- [ ] **Export report** — PDF/Markdown export of analysis for documentation.

---

