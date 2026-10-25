# Document Q&A (RAG) for small businesses

Staff upload contracts, manuals or policies (PDF, DOCX, TXT, MD) and ask questions in Serbian or English. Every answer cites file and page, and the system refuses when the documents don't say.

**Stack:** FastAPI · ChromaDB · multilingual embeddings (fastembed, local, no API key) · Claude for generation.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```

Open http://localhost:8000, upload `sample_docs/pravilnik_o_radu.txt`, ask a question. The first run downloads the embedding model (~220 MB).

API: `POST /documents` (multipart `file`), `GET /documents`, `DELETE /documents/{id}`, `POST /ask` (`{"question": "...", "doc_ids": [...]}`).

## Evaluate

```bash
python -m eval.run_eval
```

Runs `eval/cases.jsonl` against `sample_docs/` in an isolated index and writes `eval/report.json`.

| Metric            | Meaning                                                            |
| ----------------- | ------------------------------------------------------------------ |
| retrieval_hit     | the right page was among the retrieved chunks                      |
| citation_correct  | every cited file+page is a correct one (and there is at least one) |
| keyword_ok        | the answer contains the expected fact                              |
| refused_correctly | for questions the docs can't answer: no answer, no citations       |

Add your own documents and cases (same JSONL format) with `--docs` and `--cases` (several case files are allowed).

**Generated cases.** `python -m eval.generate_cases` writes questions (clean and "messy": typos, no diacritics, slang) from your documents, filtered so every expected fact really appears on the cited page. Review the output, then run `python -m eval.run_eval --cases eval/cases.jsonl eval/generated_cases.jsonl`. The summary reports clean and messy cases separately.

**Question rewriting.** Imprecise questions are rewritten into a clear version before searching (the original is searched too). Compare with `QUERY_REWRITE=0` and `1`. Tune `TOP_K`, `MIN_SCORE`, `CHUNK_WORDS` and `LLM_MODEL` (env vars, see `app/config.py`) and compare reports.

## Design notes

- **Citations are validated in code**, not trusted: the model returns source numbers, the server maps them to real file/page and drops invalid ones. An "answerable" answer with no valid citation is turned into a refusal.
- **Pages:** PDFs use real pages. DOCX and TXT have no pages, so they get ~350-word pseudo-pages (TXT can use form feed `\f` as a page break).
- **Limits:** no OCR (scanned PDFs are rejected), no auth, no per-user document isolation. Fine for a demo; not production.

## Ideas to extend

Hybrid search (BM25 + vectors), reranking, LLM-as-judge faithfulness score in the eval, streaming answers, per-user document scopes.
