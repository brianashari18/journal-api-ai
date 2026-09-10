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

import os

import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"


def embed(text: str) -> list[float]:
    """Embedding 1 teks -> vektor 1024-d (bge-m3 via Ollama, SELALU lokal)."""
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _provider() -> str:
    p = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if p in ("google", "ollama"):
        return p
    return "google" if os.environ.get("GOOGLE_API_KEY", "") else "ollama"


def chat_json(system: str, user: str) -> str:
    """Chat JSON mode. Return raw JSON string dari provider aktif."""
    if _provider() == "google":
        return _google_chat_json(system, user)
    return _ollama_chat_json(system, user)


def _ollama_chat_json(system: str, user: str) -> str:
    model = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    resp = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _google_chat_json(system: str, user: str) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("LLM_PROVIDER=google tapi GOOGLE_API_KEY kosong")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.7},
    }
    resp = httpx.post(url, params={"key": api_key}, json=body, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Gemini response tak dikenal: {data}")