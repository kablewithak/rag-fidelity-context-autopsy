# Public-Corpus Transfer Review v1

## Status

Reviewed external-validity probe on a fixed public SQuAD v1.1 subset. This report presents the public run beside the controlled synthetic benchmark without averaging their scores or changing the synthetic regression gate.

## Review boundary

- **Synthetic baseline:** `four_pipeline_baseline_v1` (18 fixed synthetic cases, 7 source documents)
- **Public transfer artifact:** `public_transfer_squad_v1_dev_v1_reviewed_v1` (30 fixed public cases, 10 source documents)
- **Public dataset:** `squad_v1.1_dev` (version `1.1`, license `CC BY-SA 4.0`)
- **Public source SHA-256:** `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`
- **Public fixture manifest SHA-256:** `d3b7c5647be5a3f8e8ec338a880acd4b0b690335e3b45608dcdc53666593f75b`

The synthetic benchmark remains the controlled mechanism test. The public fixture is a separate transfer probe. Their rates are displayed side by side for interpretation only and must not be pooled into one headline score.

## Controlled synthetic benchmark

| Pipeline | Recall@5 | MRR@10 | Evidence inclusion | Context drops |
|---|---:|---:|---:|---:|
| `char_dense_naive` | 77.8% (14/18) | 0.736 | 72.2% (13/18) | 0.0% |
| `token_dense_naive` | 94.4% (17/18) | 0.852 | 88.9% (16/18) | 0.0% |
| `token_hybrid_naive` | 100.0% (18/18) | 0.870 | 100.0% (18/18) | 0.0% |
| `token_hybrid_rerank_budgeted` | 100.0% (18/18) | 0.972 | 100.0% (18/18) | 0.0% |

## Public-corpus transfer probe

| Pipeline | Recall@5 | MRR@10 | Evidence inclusion | Context drops |
|---|---:|---:|---:|---:|
| `char_dense_naive` | 100.0% (30/30) | 0.782 | 86.7% (26/30) | 0.0% |
| `token_dense_naive` | 96.7% (29/30) | 0.809 | 90.0% (27/30) | 0.0% |
| `token_hybrid_naive` | 96.7% (29/30) | 0.864 | 93.3% (28/30) | 0.0% |
| `token_hybrid_rerank_budgeted` | 96.7% (29/30) | 0.933 | 96.7% (29/30) | 0.0% |

## Measured transfer findings

- **Public ranking signal:** within the 30-case public fixture, `token_hybrid_rerank_budgeted` changed MRR@10 by +15.2 percentage points relative to `char_dense_naive` (0.933 versus 0.782).
- **Public evidence-survival signal:** within that same fixture, the full pipeline changed evidence inclusion by +10.0 percentage points relative to `char_dense_naive` (96.7% versus 86.7%).
- **Public hybrid signal:** `token_hybrid_naive` changed MRR@10 by +8.2 percentage points relative to `char_dense_naive` (0.864 versus 0.782).
- **Non-uniform chunking result:** on this public fixture, `token_dense_naive` changed Recall@5 by -3.3 percentage points relative to `char_dense_naive` (96.7% versus 100.0%). This is not evidence for a universal chunking rule.

## Interpretation

The public run supports a limited transfer claim: the harness can measure evidence survival on non-authored public prose, and hybrid retrieval plus reranking improved ordering and final evidence inclusion on this fixed probe. It does not support a claim that every synthetic intervention transfers uniformly across corpora.

## Reproducibility

- **Tokenizer:** `tiktoken:cl100k_base`
- **Embedding model:** `sentence-transformers:sentence-transformers/all-MiniLM-L6-v2`
- **Reranker model:** `sentence-transformers-cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Device:** `cpu`
- **Candidate pool:** 8
- **Recall cutoff:** Recall@5

```powershell
python .\scripts\publish_public_transfer_review.py --check
```

## Non-claims

This review does not evaluate final generated answers, citation correctness in generated output, customer data, production latency, production cost, cross-vendor stability, security posture, or production readiness.
