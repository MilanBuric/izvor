import hashlib
import math
from functools import lru_cache

import chromadb

from .config import CHROMA_DIR, EMBEDDER


class HashEmbedder:
    """Deterministic bag-of-words embedder. Tests only, not for real use."""

    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FastEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]


@lru_cache(maxsize=1)
def embedder():
    return HashEmbedder() if EMBEDDER == "hash" else FastEmbedder(EMBEDDER)


@lru_cache(maxsize=1)
def collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def add_chunks(chunks: list[dict]) -> None:
    col = collection()
    for i in range(0, len(chunks), 64):
        batch = chunks[i : i + 64]
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["meta"] for c in batch],
            embeddings=embedder().embed([c["text"] for c in batch]),
        )


def search(question: str, k: int, doc_ids: list[str] | None = None) -> list[dict]:
    col = collection()
    if col.count() == 0:
        return []
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    res = col.query(
        query_embeddings=embedder().embed([question]),
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": d, "file": m["file"], "page": m["page"], "doc_id": m["doc_id"], "score": 1 - dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def list_documents() -> list[dict]:
    got = collection().get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for m in got["metadatas"]:
        d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "file": m["file"], "pages": 0, "chunks": 0})
        d["pages"] = max(d["pages"], m["page"])
        d["chunks"] += 1
    return list(docs.values())


def delete_document(doc_id: str) -> None:
    collection().delete(where={"doc_id": doc_id})
