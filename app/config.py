import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

# "hash" = deterministic fake embedder, for tests only
EMBEDDER = os.getenv("EMBEDDER", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# LLM_PROVIDER: "anthropic" (paid, needs ANTHROPIC_API_KEY) or "openai" = any
# OpenAI-compatible endpoint (Ollama, Groq, Gemini, OpenRouter, ...)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")  # Ollama default
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")  # Ollama ignores it

# Rewrite sloppy questions (typos, missing diacritics, slang) before searching. "0" turns it off (for A/B evals).
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "1") == "1"

TOP_K = int(os.getenv("TOP_K", "6"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))  # tune with the eval harness
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
PSEUDO_PAGE_WORDS = 350  # for formats without real pages (docx, txt)

for d in (UPLOAD_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)