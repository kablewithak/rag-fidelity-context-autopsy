# ADR-002: Add Cross-Encoder Reranking as a Second-Stage Candidate Rescorer

## Context

The lab now produces typed BM25, dense, and hybrid first-stage retrieval traces. A first-stage
retriever can return the correct evidence but rank it too low for later context packing. That is a
ranking problem, not necessarily a retrieval miss.

A repair must preserve the existing candidate set and expose before/after evidence. It must not
silently retrieve additional chunks, normalize incomparable scores, or hide model identity.

## Decision

Add `CrossEncoderReranker` after first-stage retrieval.

- Input: one typed `RetrievalTrace` with a fixed candidate list.
- Model boundary: `PairScoringModel`.
- Default runtime adapter: `SentenceTransformersCrossEncoderModel` on CPU.
- Default runtime model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Output: one typed `RerankingTrace` containing every original candidate, first-stage rank and
  score, reranker score, reranked rank, model identity, and gold-evidence rank before and after.
- Tie-break order: reranker score descending, then first-stage rank ascending, then `chunk_id`.

Reranking is intentionally constrained to first-stage candidates. A missing gold-evidence chunk
stays missing and must remain diagnosed as a retrieval failure.

## Alternatives considered

### Reuse first-stage scores only

Rejected. It cannot distinguish a retrieval candidate from a query-specific relevance judgment.

### Add cross-encoder scores to BM25, cosine, or RRF scores

Rejected. The score scales are not calibrated or comparable. The reranker owns the final order of
its fixed candidate set and records its own scores separately.

### Send the full corpus to the cross-encoder

Rejected for v1. It defeats the latency and cost purpose of a two-stage retrieval design and hides
first-stage recall failures.

## Consequences

### Easier now

- Detect retrieved-but-ranked-too-low evidence.
- Prove exact before/after rank movement.
- Keep first-stage retrieval provenance available for inspection.
- Test reranking with deterministic fixture scorers without model downloads.

### Harder now

- A real cross-encoder adds CPU latency and a first-run model download.
- Candidate-pool size becomes a measured configuration value.
- Model suitability must be evaluated separately for multilingual and domain-specific content.

## Verification gate

- Fixture tests validate candidate-set parity, score count, finite scores, rank continuity, and
  deterministic tie-breaking.
- The optional smoke script runs hybrid retrieval followed by one real local CPU cross-encoder.
- No claim of aggregate improvement is allowed until the pipeline comparison runner reports fixed
  before/after metrics across the evaluation set.
