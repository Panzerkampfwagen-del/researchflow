# Fine-tuning the retrieval encoder (real run, RTX 3050)

This documents an actual fine-tune + measured end-to-end retrieval gain, run on a
laptop RTX 3050 (6 GB). No numbers here are invented — reproduce with the commands
below. The checkpoint and corpus are gitignored (regenerable, ~88 MB + ~4 MB); the
**scripts** are tracked.

## What was done

`training/finetune_encoder.py` fine-tunes the encoder the system already uses —
`sentence-transformers/all-MiniLM-L6-v2` (384-dim) — on **80,000 SPECTER scientific
citation triplets** `(anchor paper, cited paper, non-cited paper)` with
`MultipleNegativesRankingLoss` (in-batch InfoNCE + the SPECTER hard negative), 1
epoch, batch 96, fp16. Output stays **384-dim**, so it is a **drop-in**: set
`EMBEDDING_MODEL=models/minilm-specter-ft` — no `EMBEDDING_DIM` change, no pgvector
migration.

> The baseline is the **local** base encoder. The system never used an embeddings
> *API*; the only variable changed in the A/B below is the encoder weights.

## Result 1 — SPECTER held-out triplet accuracy (saturated)

| | triplet accuracy |
|---|---|
| base `all-MiniLM-L6-v2` | 0.9357 |
| fine-tuned | 0.9370 (**+0.0013**) |

Near-flat — and honestly so: distinguishing a cited paper title from a random
unrelated title is *easy*, so the base model is already at the ceiling. This metric
is not where the fine-tune earns its keep, which is why the real test is end-to-end.

## Result 2 — end-to-end retrieval over a real corpus (the meaningful one)

**The claim, fully specified (so it can be checked):**

> Fine-tuned `sentence-transformers/all-MiniLM-L6-v2` on 80k SPECTER citation triplets;
> evaluated on a **held-out retrieval benchmark** of **24 research-question queries** over
> a **2,995-paper arXiv corpus** (gold = curated seminal arXiv ids, **none seen in
> training**), Faiss `IndexFlatIP` exact cosine; **Recall@10 0.250 → 0.326 (+31% relative)
> vs the base `all-MiniLM-L6-v2`**.

- **Benchmark:** `benchmarks/encoder_ab.py` + `benchmarks/bench_queries.json` (24 queries,
  each with curated gold arXiv ids). Corpus = those gold papers + 150 topical arXiv
  distractors per query, deduped to 2,995 (title+abstract).
- **Baseline:** the *local* base `all-MiniLM-L6-v2` the system ships with — not an
  embeddings API, and **not** `allenai/specter`. This is a same-architecture A/B: base
  MiniLM vs SPECTER-fine-tuned MiniLM, 384-dim both, identical retrieval code; the only
  variable is the encoder weights.
- **Split:** the SPECTER eval triplets (Result 1) are held out from training; the
  retrieval gold papers (Result 2) are seminal papers that never appear as training
  triplets. So neither result is the training objective measured back to itself.
- **Not SciDocs.** This is a purpose-built arXiv benchmark, not the standard SciDocs/BEIR
  task. Running the fine-tuned MiniLM against `allenai/specter` on SciDocs would be a
  *different* experiment (different baseline, model size, and split) — not claimed here.

Full table in [BENCHMARK.md](BENCHMARK.md).

| Metric | base | fine-tuned | delta | relative |
|--------|------|-----------|-------|----------|
| Recall@10 | 0.2500 | 0.3264 | **+0.0764** | **+31%** |
| MRR | 0.1404 | 0.2543 | **+0.1139** | **+81%** |
| NDCG@10 | 0.1357 | 0.2260 | **+0.0903** | **+67%** |
| NLI relevance | 0.1580 | 0.1249 | −0.0331 | −21% |

Fine-tuning on scientific citation triplets **substantially improves retrieval of the
right sources** — biggest on MRR (a relevant paper reaches the top far more often).
The query distribution here (natural-language research question → paper) differs from
the training distribution (title → title), so this is a genuine transfer result, not
the training objective measured back to itself.

**Two honest caveats:**
- **Absolute recall is modest** (0.25 → 0.33). MiniLM is a tiny encoder and "find one
  specific seminal paper among ~3,000 with title+abstract only" is hard; several
  queries score 0 for *both* encoders. The *relative* gain from fine-tuning is the
  point, and it is real and consistent across recall/MRR/NDCG.
- **NLI relevance went down**, and I don't read much into it: entailment
  `(premise = abstract, hypothesis = short query)` is a weak relevance relation —
  note both arms sit at ~0.13–0.16, near the floor. The gold-anchored metrics are the
  trustworthy signal; the NLI column is the "measure relevance with NLI" idea tried
  honestly, and it turns out to be a poor proxy for *topical* relevance.

## ANN index sanity

The same benchmark builds a Faiss `IndexHNSWFlat` over the corpus and compares it to
exact search: **ANN recall@10 vs exact = 1.0000** on this corpus — the approximate
index loses nothing here while giving the log-time path that matters at corpus scale.

## Reproduce

```bash
# GPU env with: torch(CUDA) + sentence-transformers + datasets + faiss-cpu
python training/finetune_encoder.py --max-train 80000 --epochs 1 --batch-size 96
python benchmarks/encoder_ab.py --k 10 --per-query 150     # writes BENCHMARK.md

# Enable the fine-tuned encoder in the app (drop-in, 384-dim, no migration):
#   backend/.env →  EMBEDDING_MODEL=models/minilm-specter-ft
```
