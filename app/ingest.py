from pathlib import Path

from docx import Document
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_WORDS, PSEUDO_PAGE_WORDS


def _pseudo_pages(text: str) -> list[tuple[int, str]]:
    words = text.split()
    return [
        (i // PSEUDO_PAGE_WORDS + 1, " ".join(words[i : i + PSEUDO_PAGE_WORDS]))
        for i in range(0, len(words), PSEUDO_PAGE_WORDS)
    ]


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. PDF = real pages; .txt/.md split on form feed
    (\\f) if present; .docx/.txt without breaks = ~350-word pseudo-pages."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    if ext == ".docx":
        text = "\n".join(p.text for p in Document(str(path)).paragraphs)
        return _pseudo_pages(text)
    if ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "\f" in text:
            return [(i + 1, t) for i, t in enumerate(text.split("\f"))]
        return _pseudo_pages(text)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_page(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + CHUNK_WORDS]))
        if i + CHUNK_WORDS >= len(words):
            break
    return chunks


def build_chunks(doc_id: str, filename: str, path: Path) -> list[dict]:
    out = []
    for page, text in extract_pages(path):
        for j, chunk in enumerate(chunk_page(text)):
            out.append(
                {
                    "id": f"{doc_id}:{page}:{j}",
                    "text": chunk,
                    "meta": {"doc_id": doc_id, "file": filename, "page": page},
                }
            )
    return out
