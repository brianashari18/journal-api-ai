"""Klien LLM: embedding + chat JSON, dua-duanya pluggable per provider.

Chat (`chat_json`):
- `LLM_PROVIDER=google`  -> Gemini API (wajib `GOOGLE_API_KEY`; model via `GEMINI_MODEL`)
- `LLM_PROVIDER=ollama`  -> Ollama lokal (`OLLAMA_CHAT_MODEL`)
- tidak diisi             -> otomatis: google kalau GOOGLE_API_KEY ada, selain itu ollama

Embedding (`embed`):
- `EMBED_PROVIDER=google` -> Gemini `text-embedding-004` (768-d), wajib `GOOGLE_API_KEY`
- `EMBED_PROVIDER=ollama` -> Ollama `bge-m3` (1024-d) lokal
- tidak diisi              -> otomatis seperti chat

CATATAN PRIVASI: provider google mengirim teks jurnal ke server Google.
OK untuk data sample/POC; untuk jurnal emosi asli pakai ollama lokal biar data tetap di Mac.
Dimensi vektor beda per provider (google 768 vs ollama 1024) -> collection Qdrant
dimensinya ikut `EMBED_DIM` dan harus cocok dengan provider yang aktif.
"""

import time

import httpx

from app.core.config import settings


def _post_google(url: str, params: dict, body: dict, timeout: float, attempts: int = 4) -> httpx.Response:
    """POST ke Gemini API dengan retry + backoff buat 429 (free-tier rate limit).

    Kalau quota harian benar-benar habis, 4 percobaan tetap 429 -> return respons 429
    terakhir (raise_for_status di caller yang nanti raise; retry tak bisa menembus quota).
    """
    resp = httpx.post(url, params=params, json=body, timeout=timeout)
    for i in range(1, attempts):
        if resp.status_code != 429:
            return resp
        time.sleep(2 ** (i - 1))  # 1s, 2s, 4s
        resp = httpx.post(url, params=params, json=body, timeout=timeout)
    return resp


def _resolve_provider(explicit: str) -> str:
    p = (explicit or "").strip().lower()
    if p in ("google", "ollama", "opencode"):
        return p
    return "google" if settings.GOOGLE_API_KEY else "ollama"


def _embed_provider() -> str:
    return _resolve_provider(settings.EMBED_PROVIDER)


def _chat_provider() -> str:
    return _resolve_provider(settings.LLM_PROVIDER)


def embed(text: str) -> list[float]:
    """Embedding 1 teks -> vektor (dimensi sesuai provider aktif)."""
    if _embed_provider() == "google":
        return _google_embed(text)
    return _ollama_embed(text)


def _ollama_embed(text: str) -> list[float]:
    resp = httpx.post(
        f"{settings.OLLAMA_URL}/api/embeddings",
        json={"model": settings.EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _google_embed(text: str) -> list[float]:
    api_key = settings.GOOGLE_API_KEY
    if not api_key:
        raise RuntimeError("EMBED_PROVIDER=google tapi GOOGLE_API_KEY kosong")
    model = settings.EMBED_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
    body = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": settings.EMBED_DIM,
    }
    resp = _post_google(url, {"key": api_key}, body, 120)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["embedding"]["values"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Gemini embed response tak dikenal: {data}")


def chat_json(system: str, user: str, temperature: float = 0.7) -> str:
    """Chat JSON mode. Return raw JSON string dari provider aktif."""
    provider = _chat_provider()
    if provider == "google":
        return _google_chat_json(system, user, temperature)
    if provider == "opencode":
        return _opencode_chat_json(system, user, temperature)
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
    resp = _post_google(url, {"key": api_key}, body, 180)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Gemini response tak dikenal: {data}")


def _post_opencode(url: str, headers: dict, body: dict, timeout: float, attempts: int = 4) -> httpx.Response:
    """POST OpenAI-compatible dengan retry + backoff buat 429."""
    resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
    for i in range(1, attempts):
        if resp.status_code != 429:
            return resp
        time.sleep(2 ** (i - 1))
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
    return resp


def _opencode_chat_json(system: str, user: str, temperature: float) -> str:
    api_key = settings.OPENCODE_API_KEY
    if not api_key:
        raise RuntimeError("LLM_PROVIDER=opencode tapi OPENCODE_API_KEY kosong")
    url = f"{settings.OPENCODE_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-opencode-session": "journal-ai-worker",
    }
    body = {
        "model": settings.OPENCODE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    resp = _post_opencode(url, headers, body, 180)
    resp.raise_for_status()
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"opencode response tak dikenal: {data}")
    # Reasoning model (deepseek-v4-pro) bisa return content kosong kalau token habis
    # di reasoning_content — jangan diam-diam return "" ke parser JSON.
    if not content:
        finish = data.get("choices", [{}])[0].get("finish_reason")
        raise RuntimeError(f"opencode content kosong (finish_reason={finish}) — naikkan token?")
    return content