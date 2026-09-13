import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "").strip().lower()
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    OLLAMA_CHAT_MODEL: str = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    # OpenAI-compatible chat (opencode.ai subscription — deepseek-v4-pro dkk).
    OPENCODE_API_KEY: str = os.environ.get("OPENCODE_API_KEY", os.environ.get("OPENCODE_GO_API_KEY", ""))
    OPENCODE_BASE_URL: str = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
    OPENCODE_MODEL: str = os.environ.get("OPENCODE_MODEL", "deepseek-v4-pro")
    # Embedding provider: "google" (Gemini gemini-embedding-001, 768-d) | "ollama" (bge-m3, 1024-d).
    # Kosong = auto: google kalau GOOGLE_API_KEY ada, selain itu ollama.
    EMBED_PROVIDER: str = os.environ.get("EMBED_PROVIDER", "").strip().lower()
    EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
    EMBED_DIM: int = int(os.environ.get("EMBED_DIM", "768"))
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    # Qdrant Cloud (kalau diset): endpoint https + api key -> prioritas di atas QDRANT_URL lokal.
    QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")
    QDRANT_CLUSTER_ENDPOINT: str = os.environ.get("QDRANT_CLUSTER_ENDPOINT", "")
    COLLECTION: str = os.environ.get("COLLECTION", "journals")

settings = Settings()