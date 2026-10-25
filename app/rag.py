import json
import re

import anthropic
import httpx

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER, MIN_SCORE, QUERY_REWRITE, TOP_K
from .store import search

SYSTEM = """You answer questions strictly from the numbered SOURCES provided. Rules:
1. Use only the SOURCES. Never use outside knowledge, never guess.
2. If the SOURCES do not clearly contain the answer, set "answerable" to false and say briefly that the documents do not contain this information.
3. Write the answer in the language of the QUESTION, even when the SOURCES are in another language: translate, do not copy. An English question gets an English answer, a Serbian question a Serbian answer. For Serbian keep the script the question uses (Latin or Cyrillic).
4. In "sources" list only the numbers of the SOURCES you actually relied on. Empty list if not answerable.
5. Answer in one or two complete sentences. Include every condition, exception, deadline, amount and number in the SOURCES that bears on the question. If the SOURCES give different values for different situations (for example during a probation period versus after it), give the value for the exact situation the question asks about and name that situation. If the question does not say which situation it means and the SOURCES give different values for different situations, give each value with its situation. Keep numbers, dates and names exactly as written.
6. The QUESTION may contain typos, missing diacritics or slang. A CLEAR VERSION may follow it: use it to understand what is asked, but write the answer in the language of the original QUESTION.
Output ONLY a JSON object: {"answerable": true|false, "answer": "...", "sources": [1, 2]}"""

REFUSAL = {
    "sr": "Traženi podatak nije pronađen u otpremljenim dokumentima.",
    "en": "The uploaded documents do not contain this information.",
}

_client = None


def _refusal(english: bool) -> str:
    return REFUSAL["en"] if english else REFUSAL["sr"]


def _complete(system: str, user: str) -> str:
    if LLM_PROVIDER == "openai":  # Ollama / Groq / Gemini / OpenRouter (OpenAI-compatible)
        r = httpx.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = _client.messages.create(
        model=LLM_MODEL, max_tokens=800, system=system, messages=[{"role": "user", "content": user}]
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def _parse(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON in model output: {raw[:200]}")
    return json.loads(raw[start : end + 1])


def _format_sources(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (file: {h['file']}, page: {h['page']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )


REWRITE_SYSTEM = """You clean up questions for a document Q&A system. The user is an employee typing quickly: expect typos, missing Serbian diacritics (s/c/z/dj for š/č/ž/đ), slang, abbreviations and vague wording.
Rewrite the question as one clear, complete question in the SAME language and script. Fix typos, restore diacritics, expand abbreviations.
Do NOT answer it. Do NOT add facts. Do NOT change numbers or names. If it is already clear, return it unchanged.
Output ONLY a JSON object: {"query": "..."}"""


_EN_WORDS = {
    "the", "is", "are", "was", "were", "does", "did", "how", "what", "when", "who", "where", "which", "why",
    "can", "could", "should", "must", "many", "much", "any", "for", "of", "in", "at", "by", "with", "from",
    "my", "our", "you", "we", "be", "have", "has", "there", "if", "it", "this", "that", "or", "and",
}  # words that are NOT also common Serbian words (so no "a", "i", "on", "do", "to")


def _looks_english(q: str) -> bool:
    words = re.findall(r"[a-z']+", q.lower())
    hits = sum(w in _EN_WORDS for w in words)
    return hits >= 2 or (bool(words) and hits / len(words) >= 0.3)


def rewrite_query(question: str) -> str:
    """Clear version of a sloppy question. Falls back to the original on any problem."""
    if not QUERY_REWRITE:
        return question
    try:
        q = _parse(_complete(REWRITE_SYSTEM, question)).get("query")
        return q.strip() if isinstance(q, str) and q.strip() else question
    except Exception:
        return question


def _retrieve(queries: list[str], doc_ids: list[str] | None) -> list[dict]:
    """Search with every query, keep each passage's best score, return the top TOP_K."""
    best: dict[tuple, dict] = {}
    for q in queries:
        for h in search(q, TOP_K, doc_ids):
            key = (h["doc_id"], h["page"], h["text"][:80])
            if key not in best or h["score"] > best[key]["score"]:
                best[key] = h
    return sorted(best.values(), key=lambda h: -h["score"])[:TOP_K]


def ask(question: str, doc_ids: list[str] | None = None) -> dict:
    english = _looks_english(question)
    # English questions are not rewritten: the small model translated them into Serbian, and the
    # multilingual search finds the right page from English anyway.
    clear = question if english else rewrite_query(question)
    queries = [question] if clear == question else [question, clear]  # original is always searched too
    hits = [h for h in _retrieve(queries, doc_ids) if h["score"] >= MIN_SCORE]
    if not hits:
        return {"answerable": False, "answer": _refusal(english), "citations": [], "retrieved": []}

    user = f"SOURCES:\n{_format_sources(hits)}\n\nQUESTION: {question}"
    if clear != question:
        user += f"\nCLEAR VERSION OF THE QUESTION: {clear}"
    if english:
        user += "\nANSWER LANGUAGE: English. Translate the facts from the SOURCES into English."
    raw = _complete(SYSTEM, user)
    try:
        parsed = _parse(raw)
    except ValueError:  # small models sometimes break JSON: treat as a refusal, don't crash
        return {"answerable": False, "answer": _refusal(english), "citations": [],
                "retrieved": [{"file": h["file"], "page": h["page"], "score": round(h["score"], 3)} for h in hits]}

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
        parsed["answer"] = _refusal(english)

    # A refusal always shows the fixed text: the model's own words there are unsourced claims.
    answer = (parsed.get("answer") or "").strip() if answerable else _refusal(english)

    return {
        "answerable": answerable,
        "answer": answer,
        "citations": cited,
        "retrieved": [{"file": h["file"], "page": h["page"], "score": round(h["score"], 3)} for h in hits],
        "interpreted_as": clear if clear != question else None,
    }