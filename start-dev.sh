#!/usr/bin/env bash
# Jalankan stack AI: Qdrant -> Ollama -> FastAPI (port 8000)
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Qdrant (port 6333)"
if curl -sf http://127.0.0.1:6333/healthz >/dev/null 2>&1; then
  echo "    sudah jalan"
else
  qdrant >/tmp/qdrant.log 2>&1 &
  echo "    qdrant started (log: /tmp/qdrant.log)"
fi

echo "==> Ollama (port 11434)"
if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "    sudah jalan"
else
  ollama serve >/tmp/ollama.log 2>&1 &
  echo "    ollama started (log: /tmp/ollama.log)"
fi

echo "==> FastAPI (port 8000)"
if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "    sudah jalan"
else
  .venv/bin/uvicorn app.main:app --port 8000 >/tmp/ai-service.log 2>&1 &
  echo "    fastapi started (log: /tmp/ai-service.log)"
fi

sleep 2
echo
echo "URLs:"
echo "  FastAPI  : http://127.0.0.1:8000/health"
echo "  Qdrant   : http://127.0.0.1:6333/healthz"
echo "  Ollama   : http://127.0.0.1:11434/api/tags"