# Four-Pipeline Reliability Baseline v1

## Status

Locally validated synthetic-data benchmark. This is not a production deployment, customer-data evaluation, or final-answer grounding claim.

## Reproducibility

- **Artifact ID:** `four_pipeline_baseline_v1`
- **Reference run:** `local_four_pipeline_run_v2`
- **Tokenizer:** `tiktoken:cl100k_base`
- **Embedding model:** `sentence-transformers:sentence-transformers/all-MiniLM-L6-v2`
- **Reranker model:** `sentence-transformers-cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Device:** `cpu`
- **Fixed cases:** 18
- **Recall cutoff:** Recall@5
- **Candidate pool:** 8

## Measured results

| Pipeline | Recall@5 | MRR@10 | Evidence inclusion | Context drops |
|---|---:|---:|---:|---:|
| `char_dense_naive` | 77.8% (14/18) | 0.736 | 72.2% (13/18) | 0.0% (0/14) |
| `token_dense_naive` | 94.4% (17/18) | 0.852 | 88.9% (16/18) | 0.0% (0/18) |
| `token_hybrid_naive` | 100.0% (18/18) | 0.870 | 100.0% (18/18) | 0.0% (0/18) |
| `token_hybrid_rerank_budgeted` | 100.0% (18/18) | 0.972 | 100.0% (18/18) | 0.0% (0/18) |

## Evidence-backed interpretation

- **Token-aware chunking:** `token_dense_naive` raised Recall@5 by +16.7 percentage points versus the character+dense baseline and raised evidence inclusion by +16.7 percentage points.
- **Hybrid retrieval:** `token_hybrid_naive` reached 100.0% Recall@5 and 100.0% evidence inclusion, a +22.2 percentage points Recall@5 gain versus baseline.
- **Reranking:** `token_hybrid_rerank_budgeted` preserved the hybrid retrieval result and increased MRR@10 by +0.236 versus baseline (0.972 versus 0.736).
- **Context budgeting:** the standard comparison suite recorded zero context-assembly drops. The separate controlled Phase 6 pressure diagnostic remains the mechanism proof that rendered wrapper overhead can displace otherwise retrieved evidence.

## Baseline failure pattern

The `char_dense_naive` baseline recorded 3 chunking-stage loss(es), 1 retrieval-stage loss(es), and 1 ranking-stage loss(es).

## Deterministic repair sequence

This sequence is derived only from observed baseline failure labels and fixed-pipeline comparison evidence. It does not add speculative repairs.

### 1. Critical — `sentence_aware_token_chunking`

- **Observed labels:** `bad_chunk_boundary`, `gold_evidence_split`
- **Repair:** Replace character-boundary chunking with sentence-aware token chunking and retain bounded overlap only where a clause, table row, or event can cross the boundary.
- **Expected signal:** Watch Recall@5 and evidence inclusion. In this fixed run, the supporting pipeline moved Recall@5 from 77.8% to 94.4% and evidence inclusion from 72.2% to 88.9%.
- **Metrics to watch:** Recall@5, evidence inclusion
- **Trade-off:** Sentence-aware splitting increases implementation and tokenizer-provenance discipline; count final emitted text, including separators, rather than trusting configured unit budgets.
- **Supporting comparison:** `token_dense_naive`

### 2. Critical — `hybrid_retrieval`

- **Observed labels:** `dense_retrieval_miss`
- **Repair:** Add BM25-backed hybrid retrieval so exact terms, identifiers, legal clauses, prices, and error codes can complement dense semantic recall.
- **Expected signal:** Watch Recall@5 before changing ranking. In this fixed run, the supporting pipeline moved Recall@5 from 77.8% to 100.0% and evidence inclusion from 72.2% to 100.0%.
- **Metrics to watch:** Recall@5, evidence inclusion
- **Trade-off:** Hybrid retrieval adds lexical-index and fusion parameters that must remain fixed and attributable across comparisons; do not conceal candidate-depth changes behind a higher headline cutoff.
- **Supporting comparison:** `token_hybrid_naive`

### 3. High — `cross_encoder_reranking`

- **Observed labels:** `relevant_chunk_ranked_too_low`
- **Repair:** Apply cross-encoder reranking after candidate recall succeeds and before final context selection, so the most query-specific evidence reaches the context window.
- **Expected signal:** Watch MRR@10 after candidate recall is already sufficient. In this fixed run, the supporting pipeline moved MRR@10 from 0.736 to 0.972 while preserving Recall@5 at 100.0%.
- **Metrics to watch:** MRR@10, evidence inclusion
- **Trade-off:** Reranking adds per-candidate inference cost and latency; bound candidate depth first, then verify that ordering improved without trading away recall.
- **Supporting comparison:** `token_hybrid_rerank_budgeted`

## Regression gate

A fresh run must preserve the fixed provenance, pipeline definitions, case set, Recall cutoff, and all baseline-included evidence. It may improve metrics, but it fails when Recall@5, MRR@10, or evidence inclusion falls below the reviewed baseline, or when dropped-evidence rate increases.

```powershell
python .\scripts\run_comparison_baseline.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
```

## Non-claims

This benchmark does not evaluate final generated answers, citation correctness in generated output, customer data, production latency, production cost, model-version stability across vendors, or production readiness.
