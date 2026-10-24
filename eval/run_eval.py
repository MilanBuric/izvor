"""Evaluate answer quality: retrieval, citation correctness, refusal behaviour.

    python -m eval.run_eval                 # uses sample_docs/ + eval/cases.jsonl
    python -m eval.run_eval --docs my_docs --cases my_cases.jsonl

Metrics (answerable cases): retrieval_hit, citation_correct, keyword_ok.
Metrics (unanswerable cases): refused_correctly.
Uses an isolated data dir, so it never touches your real index.
"""
import argparse
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "sample_docs"))
    ap.add_argument("--cases", default=str(ROOT / "eval" / "cases.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "report.json"))
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="docqa_eval_")
    os.environ["DATA_DIR"] = tmp  # must be set before importing app modules

    from app import rag, store
    from app.ingest import build_chunks

    for p in sorted(Path(args.docs).iterdir()):
        if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
            store.add_chunks(build_chunks(uuid.uuid4().hex[:8], p.name, p))

    cases = [json.loads(l) for l in Path(args.cases).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for c in cases:
        r = rag.ask(c["q"])
        row = {"q": c["q"], "expected_answerable": c["answerable"], "answerable": r["answerable"], "answer": r["answer"]}
        if c["answerable"]:
            expected = {(c["file"], p) for p in c["pages"]}
            cited = {(x["file"], x["page"]) for x in r["citations"]}
            retrieved = {(x["file"], x["page"]) for x in r["retrieved"]}
            row["retrieval_hit"] = bool(expected & retrieved)
            row["citation_correct"] = r["answerable"] and bool(cited) and cited <= expected
            kws = c.get("must_contain", [])
            row["keyword_ok"] = all(k.casefold() in r["answer"].casefold() for k in kws)
        else:
            row["refused_correctly"] = not r["answerable"] and not r["citations"]
        rows.append(row)

    ans = [r for r in rows if r["expected_answerable"]]
    una = [r for r in rows if not r["expected_answerable"]]
    pct = lambda xs, k: round(100 * sum(1 for x in xs if x[k]) / len(xs), 1) if xs else None
    summary = {
        "model": rag.LLM_MODEL,
        "answerable_cases": len(ans),
        "unanswerable_cases": len(una),
        "retrieval_hit_%": pct(ans, "retrieval_hit"),
        "citation_correct_%": pct(ans, "citation_correct"),
        "keyword_ok_%": pct(ans, "keyword_ok"),
        "refused_correctly_%": pct(una, "refused_correctly"),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in rows:
        ok = r.get("citation_correct", r.get("refused_correctly"))
        print(("PASS " if ok and r.get("keyword_ok", True) else "FAIL ") + r["q"])
    print(json.dumps(summary, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
