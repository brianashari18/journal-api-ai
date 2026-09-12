import json
from fastapi import APIRouter, HTTPException

from app.repository import qdrant_store
from app.services import summarizer
from app.services.llm_client import chat_json
from app.schemas.journal import (
    EntriesRequest,
    EntriesResponse,
    PeriodSummary,
    QueryRequest,
    QueryResponse,
    SummarizeRequest,
    WeeklySummaryResponse,
    PromptRequest,
    PromptResponse,
)

router = APIRouter(prefix="/ai", tags=["AI"])

# entry JSON: {date, day, emotion, text}
ENTRY_LINE = "{e.date}|{e.day}|{e.text}"


@router.post("/entries", response_model=EntriesResponse)
def save_entries(req: EntriesRequest):
    """Simpan entri jurnal baru ke Qdrant (idempotent per date|text)."""
    if not req.entries:
        raise HTTPException(status_code=400, detail="entries kosong")
    qdrant_store.ensure_collection()
    qdrant_store.upsert_entries(req.entries)
    return EntriesResponse(
        entry_ids=[summarizer.entry_id_for(e) for e in req.entries],
        count=len(req.entries),
    )


@router.post("/summarize", response_model=WeeklySummaryResponse)
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


@router.post("/query", response_model=QueryResponse)
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


@router.post("/prompt", response_model=PromptResponse)
def generate_prompt(req: PromptRequest):
    """Generate contextual micro-prompt based on user's current writing."""
    if not req.text or len(req.text.strip()) == 0:
        return PromptResponse(
            prompt="Dari mana kamu ingin mulai bercerita hari ini? Tentang sebuah kejadian, perasaan, atau pikiran yang mengganjal?"
        )

    system_prompt = (
        "Kamu adalah fasilitator expressive writing yang sangat empatik dan non-judgmental. "
        "Tugasmu memberikan SATU micro-prompt pendek (maksimal 12 kata) berbentuk pertanyaan eksploratif. "
        "ATURAN UTAMA:\n"
        "1. SEQUENCE: Secara internal, nilai tahap tulisan penulis (Event -> Experience -> Emotion -> Meaning -> Reflection). "
        "Lalu berikan pertanyaan perancah (scaffold) untuk memandu ke langkah selanjutnya. JANGAN melompat ke emosi, refleksi, atau pemecahan masalah (problem-solving) jika mereka baru menceritakan kejadian.\n"
        "2. FOKUS: JANGAN PERNAH menanyakan tentang rencana tindakan, solusi, atau bagaimana menyelesaikan masalah. Fokus murni pada penggalian cerita, pikiran, perasaan, dan pengalaman saat ini.\n"
        "3. EMOTION ACCEPTANCE: Jika penulis menunjukkan emosi negatif, dorong mereka untuk mengobservasi emosi tersebut dengan terbuka dan rasa ingin tahu, tanpa menghakiminya.\n"
        "4. NATURAL TONE: Bersikaplah non-directive, exploratory, accepting, dan curious. Gunakan bahasa Indonesia sehari-hari yang santai dan kasual (seperti 'Gimana', 'Bikin', 'Terasa'). JANGAN kaku atau formal seperti robot.\n"
        "5. OPEN-ENDED: Selalu berikan pertanyaan terbuka yang memancing cerita (seperti 'Apa yang...', 'Bagaimana...'). JANGAN PERNAH memberikan pertanyaan tertutup (Ya/Tidak) seperti 'Apakah...'.\n"
        "6. TANPA LABEL: JANGAN PERNAH menyertakan kata awalan seperti 'Event:', 'Emotion:', atau semacamnya di dalam pertanyaanmu. Langsung berikan pertanyaannya saja.\n"
        "7. AI ROLE: Kamu adalah fasilitator, BUKAN interpreter atau therapist. Jangan pernah menebak-nebak penyebab atau memberi saran/reframing positif di awal.\n"
        "8. UNIT & COGNITIVE LOAD: Berikan HANYA SATU pertanyaan kecil per langkah untuk mengurangi beban kognitif.\n"
        "9. KATA GANTI: Gunakan kata ganti orang kedua (kamu/mu) untuk merujuk pada penulis. JANGAN PERNAH menggunakan kata ganti orang pertama (aku/ku/gue/saya) di dalam pertanyaanmu.\n"
        "10. RAG CONTEXT: Jika ada catatan masa lalu yang relevan, gunakan untuk memperdalam refleksinya (misal: 'Apakah kecemasan ini sama seperti bulan lalu?'). JANGAN paksakan menyebut masa lalu jika tidak relevan dengan cerita utamanya saat ini.\n\n"
        "CONTOH PERTANYAAN IDEAL (CONTOH GAYA BAHASA):\n"
        "- \"Apa yang paling terasa dari kejadian itu?\"\n"
        "- \"Bagian mana yang paling mengganggumu?\"\n"
        "- \"Kalau kamu tidak perlu menilai perasaan itu, seperti apa rasanya?\"\n"
        "- \"Apa yang muncul di pikiranmu ketika mengingatnya?\"\n\n"
        "Kembalikan HANYA JSON murni dengan format: {\"prompt\": \"pertanyaanmu\"}"
    )
    
    user_prompt = f"CERITA PENULIS SAAT INI:\n{req.text}"
    
    # RAG: Cari jurnal masa lalu yang relevan
    try:
        qdrant_store.ensure_collection()
        similar_entries = qdrant_store.search_similar(req.text, limit=2)
        if similar_entries:
            past_context = "\n".join([f"- {e.date}: {e.text}" for e in similar_entries])
            user_prompt += f"\n\nCATATAN MASA LALU YANG MUNGKIN RELEVAN (Hanya gunakan jika nyambung dengan cerita saat ini):\n{past_context}"
    except Exception as e:
        # Abaikan error Qdrant agar prompt tetap berjalan meskipun DB kosong/mati
        print(f"RAG Error: {e}")
    
    # Gunakan temperature yang lebih tinggi agar prompt lebih bervariasi dan natural
    raw = chat_json(system_prompt, user_prompt, temperature=0.9)
    try:
        data = json.loads(raw)
        prompt_str = data.get("prompt", "")
        # Fallback if the LLM returned a list in "prompts"
        if not prompt_str and "prompts" in data and len(data["prompts"]) > 0:
            prompt_str = data["prompts"][0]
            
        prompt_str = str(prompt_str).strip()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM JSON tidak valid: {raw[:200]}") from exc

    if not prompt_str:
        prompt_str = "Coba ceritakan lebih lanjut tentang bagian itu..."
        
    return PromptResponse(prompt=prompt_str)
