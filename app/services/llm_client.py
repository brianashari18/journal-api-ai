"""Klien LLM: embedding (Ollama bge-m3, TETAP LOKAL) + chat JSON
(provider: Ollama lokal ATAU Google Gemini API).

Provider chat dipilih via env:
- `LLM_PROVIDER=google`  -> Gemini API (wajib `GOOGLE_API_KEY`; model via `GEMINI_MODEL`)
- `LLM_PROVIDER=ollama`  -> Ollama lokal (`OLLAMA_CHAT_MODEL`)
- tidak diisi             -> otomatis: google kalau GOOGLE_API_KEY ada, selain itu ollama

CATATAN PRIVASI: provider google mengirim teks jurnal ke server Google.
OK untuk data sample/POC; untuk jurnal emosi asli pakai ollama lokal biar data tetap di Mac.
Embedding (bge-m3) selalu lokal di kedua mode.
"""

import httpx

from app.core.config import settings

def embed(text: str) -> list[float]:
    """Embedding 1 teks -> vektor 1024-d (bge-m3 via Ollama, SELALU lokal)."""
    resp = httpx.post(
        f"{settings.OLLAMA_URL}/api/embeddings",
        json={"model": settings.EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _provider() -> str:
    p = settings.LLM_PROVIDER
    if p in ("google", "ollama"):
        return p
    return "google" if settings.GOOGLE_API_KEY else "ollama"


def chat_json(system: str, user: str, temperature: float = 0.7) -> str:
    """Chat JSON mode. Return raw JSON string dari provider aktif."""
    if _provider() == "google":
        return _google_chat_json(system, user, temperature)
    return _ollama_chat_json(system, user, temperature)


def _ollama_chat_json(system: str, user: str, temperature: float) -> str:
    model = settings.OLLAMA_CHAT_MODEL
    resp = httpx.post(
        f"{settings.OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _google_chat_json(system: str, user: str, temperature: float) -> str:
    api_key = settings.GOOGLE_API_KEY
    if not api_key:
        raise RuntimeError("LLM_PROVIDER=google tapi GOOGLE_API_KEY kosong")
    model = settings.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": temperature,
        },
    }
    resp = httpx.post(url, params={"key": api_key}, json=body, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Gemini response tak dikenal: {data}")