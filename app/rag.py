import json

import anthropic

from .config import LLM_MODEL, MIN_SCORE, TOP_K
from .store import search

SYSTEM = """You answer questions strictly from the numbered SOURCES provided. Rules:
1. Use only the SOURCES. Never use outside knowledge, never guess.
2. If the SOURCES do not clearly contain the answer, set "answerable" to false and say briefly that the documents do not contain this information.
3. Reply in the language of the QUESTION (Serbian or English). For Serbian keep the script the question uses (Latin or Cyrillic).
4. In "sources" list only the numbers of the SOURCES you actually relied on. Empty list if not answerable.
5. Be concise and exact: keep numbers, dates and names as written in the SOURCES.
Output ONLY a JSON object: {"answerable": true|false, "answer": "...", "sources": [1, 2]}"""

REFUSAL = {
    "sr": "Traženi podatak nije pronađen u otpremljenim dokumentima.",
    "en": "The uploaded documents do not contain this information.",
}

_client: anthropic.Anthropic | None = None


def _llm() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def _parse(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON in model output: {raw[:200]}")
    return json.loads(raw[start : end + 1])


def _format_sources(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (file: {h['file']}, page: {h['page']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )


def ask(question: str, doc_ids: list[str] | None = None) -> dict:
    hits = [h for h in search(question, TOP_K, doc_ids) if h["score"] >= MIN_SCORE]
    if not hits:
        return {"answerable": False, "answer": REFUSAL["en"] + " / " + REFUSAL["sr"], "citations": [], "retrieved": []}

    msg = _llm().messages.create(
        model=LLM_MODEL,
        max_tokens=800,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"SOURCES:\n{_format_sources(hits)}\n\nQUESTION: {question}"}],
    )
    parsed = _parse("".join(b.text for b in msg.content if b.type == "text"))

    answerable = bool(parsed.get("answerable"))
    cited = []
    seen = set()
    for n in parsed.get("sources", []) if answerable else []:
        if isinstance(n, int) and 1 <= n <= len(hits):
            h = hits[n - 1]
            key = (h["file"], h["page"])
            if key not in seen:  # one citation per file+page; ids outside range are dropped
                seen.add(key)
                cited.append({"file": h["file"], "page": h["page"], "snippet": h["text"][:300], "score": round(h["score"], 3)})
    if answerable and not cited:  # answer with no valid citation is not trustworthy
        answerable = False
        parsed["answer"] = REFUSAL["en"] + " / " + REFUSAL["sr"]

    return {
        "answerable": answerable,
        "answer": parsed.get("answer", ""),
        "citations": cited,
        "retrieved": [{"file": h["file"], "page": h["page"], "score": round(h["score"], 3)} for h in hits],
    }
