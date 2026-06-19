# ADR-001: Use Reciprocal Rank Fusion for the First Hybrid Retriever

**Status:** Accepted for the local-first diagnostic lab

## Context

The lab compares a BM25 lexical retriever with a cosine-similarity dense retriever. Their raw scores are not calibrated to the same scale:

- BM25 scores depend on corpus term distributions and document length.
- cosine similarity depends on the selected embedding model and vector geometry.

Directly adding or weighting these scores would introduce an arbitrary calibration choice before the eval harness has enough evidence to tune one responsibly.

## Decision

Use **reciprocal rank fusion (RRF)** for the first hybrid retriever.

For every corpus chunk, the hybrid score is:

```text
1 / (rrf_k + bm25_rank) + 1 / (rrf_k + dense_rank)
```

The v1 default is `rrf_k = 60`.

The hybrid trace must retain, for every returned chunk:

- BM25 rank and raw BM25 score
- dense rank and raw cosine score
- RRF parameter
- fused score
- lexical analyzer name
- embedding model name and vector dimension

## Consequences

### Easier now

- No unvalidated score normalization or learned weights
- Deterministic, scale-independent fusion
- Direct inspection of why a hybrid result moved
- A small and portable local-first implementation

### Not claimed

- RRF is not proven optimal for every corpus
- The default `rrf_k` is not data-tuned
- Hybrid fusion does not prove improved performance until aggregate fixed-case comparison exists

## Future extension seam

A later experiment may add calibrated weighted fusion or a learned ranker only when the eval harness contains enough fixed cases and before/after metrics to compare it against RRF. That implementation must preserve the typed `RetrievalTrace` contract and record its fusion method and parameters.
