import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "").strip().lower()
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    OLLAMA_CHAT_MODEL: str = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "bge-m3")
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    COLLECTION: str = os.environ.get("COLLECTION", "journals")

settings = Settings()
