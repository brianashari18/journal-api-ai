"""Journal AI Service — FastAPI. Dipanggil oleh Go server (internal, port 8000)."""

import json

from fastapi import FastAPI, HTTPException

import qdrant_store
import summarizer
from ollama_client import chat_json
from schemas import (
    Entry,
    PeriodSummary,
    QueryRequest,
    QueryResponse,
    SummarizeRequest,
    WeeklySummaryResponse,
)

app = FastAPI(title="Journal AI Service")

# entry JSON: {date, day, emotion, text}
ENTRY_LINE = "{e.date}|{e.day}|{e.text}"


@app.get("/health")
def health():
    return {"status": "ok", "service": "journal-ai"}


@app.post("/ai/summarize", response_model=WeeklySummaryResponse)
def summarize(req: SummarizeRequest):
    """Weekly emotional summary (HYBRID): deterministik (label user, context, kandidat
    pattern/change) + LLM narasi & reflection. LLM tidak menamai ulang emosi."""
    qdrant_store.ensure_collection()
    if req.entries:
        qdrant_store.upsert_entries(req.entries)

    entries = qdrant_store.retrieve_by_date(req.start_date, req.end_date)
    if not entries:
        raise HTTPException(status_code=404, detail="tidak ada jurnal di rentang tersebut")

    structured = summarizer.build_structured(entries)
    patterns, changes, reflection = summarizer.generate_narrative(structured)
    presentation = summarizer.build_presentation(structured, req.start_date, req.end_date, reflection)
    return WeeklySummaryResponse(
        period=PeriodSummary(start=req.start_date, end=req.end_date, entry_count=len(entries)),
        emotion_overview=structured["emotion_overview"],
        timeline=structured["timeline"],
        patterns=patterns,
        changes=changes,
        reflection_question=reflection,
        presentation=presentation,
    )


@app.post("/ai/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """RAG klasik: similarity search -> chunks retrieved -> jawaban grounded + sumber."""
    qdrant_store.ensure_collection()
    hits = qdrant_store.search_similar(req.question, limit=5)
    if not hits:
        raise HTTPException(status_code=404, detail="tidak ada data jurnal untuk dicari")

    context = "\n".join(ENTRY_LINE.format(e=e) for e in hits)
    prompt = (
        "Jawab pertanyaan berikut berdasarkan entri jurnal relevan di bawah. "
        "Sebutkan tanggal sumbernya dan kutip bagian jurnal untuk mendukung jawabanmu. "
        "Jika tidak ada di jurnal, jawab jujur bahwa tidak ada. "
        'Balas HANYA JSON murni: {"answer": str, "sources": [tanggal...]}\n\n'
        f"PERTANYAAN: {req.question}\n\nENTRI RELEVAN:\n{context}"
    )

    raw = chat_json(summarizer.SYSTEM, prompt)
    try:
        data = json.loads(raw)
        answer = str(data.get("answer", "")).strip()
        sources = [str(x) for x in data.get("sources", [])]
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM JSON tidak valid: {raw[:200]}") from exc

    if not answer:
        raise HTTPException(status_code=502, detail="jawaban LLM kosong")
    return QueryResponse(question=req.question, answer=answer, sources=sources)