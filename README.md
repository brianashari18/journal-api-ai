# journal-api-ai

FastAPI AI worker (:8000) untuk weekly emotional summary — project mandiri:
embed (bge-m3 via Ollama), simpan/retrieve (Qdrant), narrative LLM (chat model
configurable via `OLLAMA_CHAT_MODEL`, default `qwen2.5:7b`).

## Setup

```bash
make venv    # python3 -m venv .venv + pip install -r requirements.txt
```

## Run

```bash
./start-dev.sh   # Qdrant (:6333) + Ollama (:11434) + FastAPI (:8000)
```

## Test

```bash
make test   # pytest (integration auto-skip kalau Qdrant/Ollama bge-m3 mati)
```

## Endpoint

```
POST /ai/summarize   # HYBRID: evidence deterministik + narasi LLM ter-guardrail + presentation
POST /ai/query       # RAG klasik (similarity search)
GET  /health
```

Model entri: `emotions: [{emotion, intensity 1-10}]` (maks 3, pilihan USER — bukan tebakan mesin);
LLM provider: `OLLAMA_CHAT_MODEL` (lokal) / `LLM_PROVIDER=google` + `GOOGLE_API_KEY` (Gemini).
Evaluasi fidelity: `python evaluate.py [start] [end]` (data sample di `data/journals/`).