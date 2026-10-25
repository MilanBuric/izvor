"""Evaluate answer quality: retrieval, citation correctness, refusal behaviour.

    python -m eval.run_eval                                   # sample_docs/ + eval/cases.jsonl
    python -m eval.run_eval --cases eval/cases.jsonl eval/generated_cases.jsonl
    python -m eval.run_eval --docs my_docs --cases my_cases.jsonl

Metrics (answerable cases): retrieval_hit, citation_correct, keyword_ok.
Metrics (unanswerable cases): refused_correctly.
Cases may carry "kind": "clean" (default) or "messy" (typos, no diacritics, slang);
the summary reports them separately. A/B the question rewriting with QUERY_REWRITE=0 / 1.
Uses an isolated data dir, so it never touches your real index.
"""
import argparse
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def norm(s: str) -> str:
    """Compare answers loosely: '2,500' = '2.500' = '2500', case and spacing ignored."""
    s = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", s)
    return re.sub(r"\s+", " ", s).casefold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "sample_docs"))
    ap.add_argument("--cases", nargs="+", default=[str(ROOT / "eval" / "cases.jsonl")])
    ap.add_argument("--out", default=str(ROOT / "eval" / "report.json"))
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="docqa_eval_")
    os.environ["DATA_DIR"] = tmp  # must be set before importing app modules

    from app import rag, store
    from app.ingest import build_chunks

    for p in sorted(Path(args.docs).iterdir()):
        if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
            store.add_chunks(build_chunks(uuid.uuid4().hex[:8], p.name, p))

    cases = []
    for f in args.cases:
        cases += [json.loads(l) for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = []
    for c in cases:
        r = rag.ask(c["q"])
        row = {
            "q": c["q"],
            "kind": c.get("kind", "clean"),
            "expected_answerable": c["answerable"],
            "answerable": r["answerable"],
            "answer": r["answer"],
            "interpreted_as": r.get("interpreted_as"),
        }
        if c["answerable"]:
            expected = {(c["file"], p) for p in c["pages"]}
            cited = {(x["file"], x["page"]) for x in r["citations"]}
            retrieved = {(x["file"], x["page"]) for x in r["retrieved"]}
            row["retrieval_hit"] = bool(expected & retrieved)
            row["citation_correct"] = r["answerable"] and bool(cited) and cited <= expected
            kws = c.get("must_contain", [])
            # each entry is a string, or a list of accepted alternatives (any one matches)
            has = lambda k: any(norm(a) in norm(r["answer"]) for a in (k if isinstance(k, list) else [k]))
            row["keyword_ok"] = all(has(k) for k in kws)
            row["refused"] = not r["answerable"]
            row["expected_keywords"] = kws
            row["expected_pages"] = sorted(c["pages"])
            row["cited_pages"] = sorted({x["page"] for x in r["citations"]})
        else:
            row["refused_correctly"] = not r["answerable"] and not r["citations"]
        rows.append(row)

    ans = [r for r in rows if r["expected_answerable"]]
    una = [r for r in rows if not r["expected_answerable"]]
    pct = lambda xs, k: round(100 * sum(1 for x in xs if x[k]) / len(xs), 1) if xs else None
    by_kind = {}
    for kind in sorted({r["kind"] for r in ans}):
        grp = [r for r in ans if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(grp),
            "retrieval_hit_%": pct(grp, "retrieval_hit"),
            "citation_correct_%": pct(grp, "citation_correct"),
            "keyword_ok_%": pct(grp, "keyword_ok"),
        }
    summary = {
        "model": rag.LLM_MODEL,
        "query_rewrite": rag.QUERY_REWRITE,
        "answerable_cases": len(ans),
        "unanswerable_cases": len(una),
        "retrieval_hit_%": pct(ans, "retrieval_hit"),
        "citation_correct_%": pct(ans, "citation_correct"),
        "keyword_ok_%": pct(ans, "keyword_ok"),
        "refused_correctly_%": pct(una, "refused_correctly"),
        "refused_though_answerable_%": pct(ans, "refused"),  # safe misses: no answer given
        "by_kind": by_kind,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in rows:
        ok = r.get("citation_correct", r.get("refused_correctly"))
        tag = "[messy] " if r["kind"] == "messy" else ""
        passed = bool(ok and r.get("keyword_ok", True))
        print(("PASS " if passed else "FAIL ") + tag + r["q"])
        if not passed:  # say WHY, so you don't have to dig through report.json
            why = []
            if r["expected_answerable"]:
                if not r.get("citation_correct"):
                    why.append(f"cited page {r['cited_pages'] or 'none'}, expected {r['expected_pages']}")
                if r.get("refused"):
                    why.append("REFUSED although the document answers it (safe miss)")
                elif not r.get("keyword_ok"):
                    why.append(f"answered, but lacks {r['expected_keywords']}")
            else:
                why.append("should have refused but answered")
            print("      why:    " + "; ".join(why))
            print(f"      answer: {r['answer']}")
            if r.get("interpreted_as"):
                print(f"      read as: {r['interpreted_as']}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()