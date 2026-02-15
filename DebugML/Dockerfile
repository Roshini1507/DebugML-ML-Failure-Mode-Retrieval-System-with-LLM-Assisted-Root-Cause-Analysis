# DebugML: Production-grade Docker image
# Python 3.10 | Streamlit | Ollama | RAG failure analysis

FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps: curl for Ollama, procps for process management
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (binary to /usr/local/bin)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY rag.py .
COPY llm.py .
COPY vector_db.py .
COPY embeddings.py .
COPY data/ data/

EXPOSE 8501

# Health check (Streamlit built-in endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Ollama in background, then Streamlit (exec so Streamlit is PID 1 for signals)
# --server.address 0.0.0.0 allows external connections
CMD ["sh", "-c", "ollama serve & sleep 10 && exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
