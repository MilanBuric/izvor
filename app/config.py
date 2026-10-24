import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

# "hash" = deterministic fake embedder, for tests only
EMBEDDER = os.getenv("EMBEDDER", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

TOP_K = int(os.getenv("TOP_K", "6"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))  # tune with the eval harness
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
PSEUDO_PAGE_WORDS = 350  # for formats without real pages (docx, txt)

for d in (UPLOAD_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)
