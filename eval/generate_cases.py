"""Generate eval cases from your own documents with an LLM, so you don't write them by hand.

    python -m eval.generate_cases --docs sample_docs --per-page 3 --variants 2
    python -m eval.run_eval --cases eval/cases.jsonl eval/generated_cases.jsonl

For every page: the LLM writes questions plus the key facts a correct answer must contain
(Serbian and English versions) -> a sanity filter drops questions whose key facts do not
actually appear on the page -> the LLM rewrites each question the way a hurried employee
would type it (typos, no diacritics, slang) -> "messy" cases.

The generator is an LLM too: REVIEW the output file before trusting it, and keep a small
hand-written core (eval/cases.jsonl), especially for questions the documents cannot answer.
Uses the same LLM_PROVIDER / LLM_MODEL settings as the app; a stronger model writes better cases.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEN_SYSTEM = """You write test questions for a document Q&A system. You are given ONE page of a company document.
Write N questions that a normal employee might ask and that this page answers explicitly.
Rules:
- Each question must be self-contained: no "this page", "the document", "above".
- Each question must have ONE clear answer on this page. If a fact depends on a situation (for example during versus after probation), name the situation in the question.
- Never write a question the page does not answer.
- "question" is in the language of the page. "question_en" is the same question in English.
- "key_facts" lists the facts a correct answer must contain. Each fact is a list of accepted spellings, including digits, words, and Serbian and English forms, for example [["30", "trideset", "thirty"]]. ALWAYS include the English forms too, for example [["2", "dva", "two"]] or [["30. jun", "June 30", "30 June", "30th of June"]]. Keep every spelling SHORT: a number, a date, or a phrase of 1-3 words, never a full sentence. Copy numbers exactly as the page writes them. If the correct answer is a negation (not allowed, not reimbursed, none), list the phrases a correct answer would use in Serbian and English, for example [["ne nadoknađuje", "does not cover", "does not reimburse", "not covered"]]; never use a bare "0" or "no".
Output ONLY a JSON object: {"items": [{"question": "...", "question_en": "...", "key_facts": [["..."]]}]}"""

VAR_SYSTEM = """You imitate how a hurried employee types a question into a search box.
Rewrite the given question N different ways, keeping the exact meaning: lowercase, no question mark, typos, missing Serbian diacritics (s for š, c for č or ć, z for ž, dj for đ), slang, vague or shortened wording.
Keep the language of the question. Do not change numbers.
Output ONLY a JSON object: {"variants": ["...", "..."]}"""


def _script_ok(q: str, base: str) -> bool:
    """Reject garbled generator output: words that mix Latin and Cyrillic letters,
    or a variant written in a different script than the question it came from."""
    cyr = lambda s: sum("Ѐ" <= ch <= "ӿ" for ch in s)
    lat = lambda s: sum((ch.isalpha() and ch.isascii()) or ch in "čćžšđČĆŽŠĐ" for ch in s)
    if any(cyr(w) and lat(w) for w in q.split()):
        return False
    return (cyr(q) > lat(q)) == (cyr(base) > lat(base))


def _norm_facts(raw) -> list[list[str]]:
    groups = []
    for g in raw if isinstance(raw, list) else []:
        alts = [g] if isinstance(g, str) else g if isinstance(g, list) else []
        alts = [a.strip() for a in alts if isinstance(a, str) and a.strip()]
        if alts:
            groups.append(alts)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "sample_docs"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "generated_cases.jsonl"))
    ap.add_argument("--per-page", type=int, default=3, help="questions per page")
    ap.add_argument("--variants", type=int, default=2, help="messy rewrites per question")
    args = ap.parse_args()

    os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="docqa_gen_")  # keep the real index untouched
    from app import rag
    from app.ingest import extract_pages

    cases, dropped, pages_done = [], 0, 0
    for path in sorted(Path(args.docs).iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
            continue
        for page, text in extract_pages(path):
            if len(text.split()) < 40:
                continue
            pages_done += 1
            try:
                items = rag._parse(rag._complete(GEN_SYSTEM, f"PAGE TEXT:\n{text}\n\nWrite N={args.per_page} questions.")).get("items", [])
            except Exception as e:
                print(f"skip {path.name} p.{page}: {e}")
                continue
            for it in items:
                facts = _norm_facts(it.get("key_facts"))
                # sanity: every key fact must appear on the page in at least one accepted spelling
                if not facts or not all(any(a.casefold() in text.casefold() for a in g) for g in facts):
                    dropped += 1
                    continue
                for lang_key in ("question", "question_en"):
                    q = it.get(lang_key)
                    if not isinstance(q, str) or not q.strip() or not _script_ok(q, q):
                        continue
                    base = {"answerable": True, "file": path.name, "pages": [page], "must_contain": facts, "source": "generated"}
                    cases.append({"q": q.strip(), "kind": "clean", **base})
                    if args.variants > 0:
                        try:
                            vs = rag._parse(rag._complete(VAR_SYSTEM, f"QUESTION: {q}\n\nWrite N={args.variants} variants.")).get("variants", [])
                        except Exception:
                            vs = []
                        for v in vs[: args.variants]:
                            if (isinstance(v, str) and v.strip() and v.strip().casefold() != q.strip().casefold()
                                    and _script_ok(v, q)):
                                cases.append({"q": v.strip(), "kind": "messy", **base})

    Path(args.out).write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    clean = sum(c["kind"] == "clean" for c in cases)
    print(f"{len(cases)} cases ({clean} clean, {len(cases) - clean} messy) from {pages_done} pages; "
          f"dropped {dropped} questions whose key facts were not on the page -> {args.out}")


if __name__ == "__main__":
    main()