from fastapi import FastAPI
from app.api.routers import journal_ai

app = FastAPI(title="Journal AI Service")

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "journal-ai"}

app.include_router(journal_ai.router)
