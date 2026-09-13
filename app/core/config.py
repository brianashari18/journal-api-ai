import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "").strip().lower()
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    OLLAMA_CHAT_MODEL: str = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    # Embedding provider: "google" (Gemini gemini-embedding-001, 768-d) | "ollama" (bge-m3, 1024-d).
    # Kosong = auto: google kalau GOOGLE_API_KEY ada, selain itu ollama.
    EMBED_PROVIDER: str = os.environ.get("EMBED_PROVIDER", "").strip().lower()
    EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
    EMBED_DIM: int = int(os.environ.get("EMBED_DIM", "768"))
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    COLLECTION: str = os.environ.get("COLLECTION", "journals")

settings = Settings()