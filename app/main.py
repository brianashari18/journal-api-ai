from fastapi import FastAPI
from app.api.routers import journal_ai

app = FastAPI(
    title="Journal AI Service",
    description="AI worker: embedding (Gemini/bge-m3), retrieval (Qdrant), narasi summary + Q&A (opencode/Gemini). Internal — dipanggil Go gateway.",
    version="1.0.0",
)

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "journal-ai"}

app.include_router(journal_ai.router)

# ci-test marker
