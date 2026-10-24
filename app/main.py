import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import rag, store
from .config import UPLOAD_DIR
from .ingest import build_chunks

ALLOWED = {".pdf", ".docx", ".txt", ".md"}

app = FastAPI(title="Document Q&A")


class AskRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/documents")
def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(400, f"Allowed types: {', '.join(sorted(ALLOWED))}")
    doc_id = uuid.uuid4().hex[:8]
    dest = UPLOAD_DIR / f"{doc_id}_{name}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        chunks = build_chunks(doc_id, name, dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not read file: {e}")
    if not chunks:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "No extractable text (scanned PDF? OCR is not supported).")
    store.add_chunks(chunks)
    return {"doc_id": doc_id, "file": name, "chunks": len(chunks)}


@app.get("/documents")
def documents():
    return store.list_documents()


@app.delete("/documents/{doc_id}")
def delete(doc_id: str):
    store.delete_document(doc_id)
    for p in UPLOAD_DIR.glob(f"{doc_id}_*"):
        p.unlink(missing_ok=True)
    return {"deleted": doc_id}


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    return rag.ask(req.question, req.doc_ids)
