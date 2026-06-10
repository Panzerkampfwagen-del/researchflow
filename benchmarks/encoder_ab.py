#!/usr/bin/env python3
"""End-to-end retrieval A/B: base vs SPECTER-fine-tuned encoder, over a real corpus.

This is the *end-to-end* number, not just a triplet score. It:

  1. Builds a real paper corpus from arXiv (title+abstract): the gold papers for
     every benchmark query plus hundreds of topical distractors per query.
  2. Indexes that corpus with a **Faiss** index (exact ``IndexFlatIP`` over
     normalised embeddings = cosine) for each encoder, and also builds a Faiss
     ``IndexHNSWFlat`` ANN index to confirm the approximate index works at scale.
  3. Runs every research query through both encoders and scores retrieval against
     the curated gold arXiv ids: Recall@10, MRR, NDCG@10.
  4. Scores *source relevance* of the retrieved papers with the same NLI model the
     app uses for grounding — entailment(premise=abstract, hypothesis=query) —
     so quality is measured, not eyeballed.

Honesty: the "baseline" is the local base ``all-MiniLM-L6-v2`` the system already
uses — it was never an embeddings *API*. The only variable changed between the two
arms is the encoder, so the delta is attributable to the fine-tune. Corpus is
cached to ``benchmarks/data/corpus.json`` so reruns don't hammer arXiv.

Run (GPU overlay venv):
    python benchmarks/encoder_ab.py --base sentence-transformers/all-MiniLM-L6-v2 \
        --finetuned backend/models/minilm-specter-ft
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

HERE = Path(__file__).resolve().parent
QUERIES_PATH = HERE / "bench_queries.json"
CORPUS_CACHE = HERE / "data" / "corpus.json"
ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
_VER_RE = re.compile(r"v\d+$")
NLI_MODEL = "cross-encoder/nli-roberta-base"
_ENTAIL_IDX = 1  # [contradiction, entailment, neutral]


def _arxiv_get(params: dict) -> str:
    """GET the arXiv API with a polite User-Agent, returning the Atom XML text."""
    url = f"{ARXIV_ENDPOINT}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchFlow-bench/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def _parse(xml_text: str) -> list[dict]:
    """Parse an arXiv Atom feed into ``{arxiv_id, title, abstract}`` records."""
    out: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for entry in root.findall("a:entry", ATOM):
        tid = entry.find("a:id", ATOM)
        title = entry.find("a:title", ATOM)
        summ = entry.find("a:summary", ATOM)
        if tid is None or tid.text is None or title is None or title.text is None:
            continue
        arxiv_id = _VER_RE.sub("", tid.text.strip().rsplit("/", 1)[-1])
        out.append(
            {
                "arxiv_id": arxiv_id,
                "title": " ".join(title.text.split()),
                "abstract": " ".join((summ.text or "").split()) if summ is not None else "",
            }
        )
    return out


def build_corpus(queries: list[dict], per_query: int, sleep: float) -> dict[str, dict]:
    """Fetch gold papers + topical distractors from arXiv into a deduped corpus."""
    corpus: dict[str, dict] = {}

    gold_ids = sorted({g for q in queries for g in q["gold"]})
    for i in range(0, len(gold_ids), 50):
        chunk = gold_ids[i : i + 50]
        for rec in _parse(_arxiv_get({"id_list": ",".join(chunk), "max_results": 50})):
            corpus[rec["arxiv_id"]] = rec
        time.sleep(sleep)
    found = sum(1 for g in gold_ids if g in corpus)
    print(f"  gold papers fetched: {found}/{len(gold_ids)}")

    for q in queries:
        params = {
            "search_query": f"all:{q['query']}",
            "start": 0,
            "max_results": per_query,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        for rec in _parse(_arxiv_get(params)):
            corpus.setdefault(rec["arxiv_id"], rec)
        time.sleep(sleep)
        print(f"  corpus now {len(corpus)} after: {q['query'][:50]}")
    return corpus


def load_or_build_corpus(queries: list[dict], per_query: int, sleep: float) -> dict[str, dict]:
    """Return the cached corpus if present, else build and cache it."""
    if CORPUS_CACHE.is_file():
        corpus = json.loads(CORPUS_CACHE.read_text(encoding="utf-8"))
        print(f"Loaded cached corpus: {len(corpus)} papers")
        return corpus
    print("Building corpus from arXiv ...")
    corpus = build_corpus(queries, per_query, sleep)
    CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_CACHE.write_text(json.dumps(corpus), encoding="utf-8")
    print(f"Cached corpus ({len(corpus)} papers) -> {CORPUS_CACHE}")
    return corpus


def encode(model: SentenceTransformer, texts: list[str], batch: int = 256) -> np.ndarray:
    """Encode texts to L2-normalised float32 embeddings (cosine-ready for Faiss)."""
    emb = model.encode(
        texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(emb.astype("float32"))


def dcg(rels: list[int]) -> float:
    """Discounted cumulative gain of a binary relevance list."""
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def score_query(retrieved_ids: list[str], gold: set[str], k: int) -> tuple[float, float, float]:
    """Return (recall@k, MRR, NDCG@k) for one query."""
    top = retrieved_ids[:k]
    hits = [1 if rid in gold else 0 for rid in top]
    recall = sum(hits) / len(gold) if gold else 0.0
    mrr = 0.0
    for rank, rid in enumerate(top, start=1):
        if rid in gold:
            mrr = 1.0 / rank
            break
    idcg = dcg(sorted([1] * len(gold), reverse=True)[:k])
    ndcg = (dcg(hits) / idcg) if idcg else 0.0
    return recall, mrr, ndcg


def evaluate_encoder(
    name: str, model: SentenceTransformer, corpus_ids: list[str], corpus_texts: list[str],
    queries: list[dict], k: int, nli: CrossEncoder,
) -> dict:
    """Index the corpus with Faiss, retrieve every query, and aggregate metrics."""
    print(f"\n[{name}] encoding {len(corpus_texts)} papers + {len(queries)} queries ...")
    corpus_emb = encode(model, corpus_texts)
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(corpus_emb)
    q_emb = encode(model, [q["query"] for q in queries])
    _, idx = index.search(q_emb, k)

    recalls, mrrs, ndcgs = [], [], []
    nli_pairs: list[tuple[str, str]] = []
    nli_counts: list[int] = []
    per_query = []
    for qi, q in enumerate(queries):
        # Faiss pads unfilled neighbour slots with -1; drop them rather than let
        # Python negative-index into the last corpus paper (a phantom retrieval).
        hits = [int(j) for j in idx[qi] if j >= 0]
        retrieved = [corpus_ids[j] for j in hits]
        r, m, n = score_query(retrieved, set(q["gold"]), k)
        recalls.append(r)
        mrrs.append(m)
        ndcgs.append(n)
        per_query.append({"query": q["query"], "recall": round(r, 3),
                          "mrr": round(m, 3), "ndcg": round(n, 3)})
        for j in hits:
            nli_pairs.append((corpus_texts[j], q["query"]))  # premise=paper, hyp=query
        nli_counts.append(len(hits))

    # NLI source-relevance: mean entailment prob over all retrieved (query, paper) pairs.
    logits = np.asarray(nli.predict(nli_pairs))
    probs = _softmax(logits)[:, _ENTAIL_IDX]
    nli_relevance = float(np.mean(probs))

    return {
        "name": name,
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "mrr": round(float(np.mean(mrrs)), 4),
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4),
        "nli_relevance": round(nli_relevance, 4),
        "per_query": per_query,
        "_corpus_emb": corpus_emb,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over NLI logits."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def faiss_hnsw_recall(corpus_emb: np.ndarray, queries_emb: np.ndarray, k: int) -> float:
    """Build a Faiss HNSW ANN index and report its recall@k vs the exact flat index."""
    flat = faiss.IndexFlatIP(corpus_emb.shape[1])
    flat.add(corpus_emb)
    _, exact = flat.search(queries_emb, k)
    hnsw = faiss.IndexHNSWFlat(corpus_emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 64
    hnsw.add(corpus_emb)
    _, approx = hnsw.search(queries_emb, k)
    hits = sum(len(set(exact[i]) & set(approx[i])) for i in range(len(exact)))
    return hits / (len(exact) * k)


def main() -> None:
    """Run the base-vs-fine-tuned retrieval A/B and write BENCHMARK.md."""
    parser = argparse.ArgumentParser(description="Encoder retrieval A/B benchmark")
    parser.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--finetuned", default="backend/models/minilm-specter-ft")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--per-query", type=int, default=150)
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    corpus = load_or_build_corpus(queries, args.per_query, args.sleep)
    corpus_ids = list(corpus.keys())
    corpus_texts = [f"{corpus[c]['title']}. {corpus[c]['abstract']}".strip() for c in corpus_ids]
    gold_in_corpus = sum(1 for q in queries for g in q["gold"] if g in corpus)
    gold_total = sum(len(q["gold"]) for q in queries)
    print(f"\nCorpus: {len(corpus_ids)} papers | gold present: {gold_in_corpus}/{gold_total} "
          f"| queries: {len(queries)} | k={args.k}")

    nli = CrossEncoder(NLI_MODEL, device=args.device)
    base = SentenceTransformer(args.base, device=args.device)
    ft = SentenceTransformer(args.finetuned, device=args.device)

    base_res = evaluate_encoder("base", base, corpus_ids, corpus_texts, queries, args.k, nli)
    ft_res = evaluate_encoder("fine-tuned", ft, corpus_ids, corpus_texts, queries, args.k, nli)

    q_emb_ft = encode(ft, [q["query"] for q in queries])
    ann_recall = faiss_hnsw_recall(ft_res["_corpus_emb"], q_emb_ft, args.k)

    print("\n================ RETRIEVAL A/B (real corpus) ================")
    hdr = f"{'metric':<16}{'base':>10}{'fine-tuned':>14}{'delta':>10}"
    print(hdr)
    rows = []
    for key, label in [("recall_at_k", f"Recall@{args.k}"), ("mrr", "MRR"),
                       ("ndcg_at_k", f"NDCG@{args.k}"), ("nli_relevance", "NLI relevance")]:
        b, f = base_res[key], ft_res[key]
        print(f"{label:<16}{b:>10.4f}{f:>14.4f}{f - b:>+10.4f}")
        rows.append((label, b, f, f - b))
    print(f"\nFaiss IndexHNSWFlat ANN recall@{args.k} vs exact (fine-tuned): {ann_recall:.4f}")

    _write_md(rows, ann_recall, args, len(corpus_ids), gold_in_corpus, gold_total,
              base_res, ft_res)


def _write_md(rows, ann_recall, args, n_corpus, gold_in, gold_total, base_res, ft_res) -> None:
    """Write the benchmark results to BENCHMARK.md."""
    out = HERE.parent / "BENCHMARK.md"
    lines = [
        "# Encoder Retrieval Benchmark (end-to-end, real corpus)",
        "",
        f"- **Corpus:** {n_corpus} real arXiv papers (title+abstract), built from "
        f"{args.per_query} distractors/query + gold papers.",
        f"- **Gold present in corpus:** {gold_in}/{gold_total}",
        "- **Index:** Faiss `IndexFlatIP` (exact cosine); ANN cross-check via "
        "`IndexHNSWFlat`.",
        "- **Baseline:** local `all-MiniLM-L6-v2` (never an API). **Variable changed:** "
        "the encoder only.",
        "",
        "| Metric | base | fine-tuned | delta |",
        "|--------|------|-----------|-------|",
    ]
    for label, b, f, d in rows:
        lines.append(f"| {label} | {b:.4f} | {f:.4f} | {d:+.4f} |")
    lines += [
        "",
        f"Faiss `IndexHNSWFlat` ANN recall@{args.k} vs exact (fine-tuned encoder): "
        f"**{ann_recall:.4f}**",
        "",
        "## Per-query Recall (fine-tuned)",
        "",
        "| Query | Recall | MRR | NDCG |",
        "|-------|--------|-----|------|",
    ]
    for pq in ft_res["per_query"]:
        lines.append(f"| {pq['query']} | {pq['recall']:.2f} | {pq['mrr']:.2f} | {pq['ndcg']:.2f} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
