# ADR-007: Real Four-Pipeline Execution Runner

- **Status:** Accepted
- **Date:** 2026-06-20
- **Decision owner:** RAG Fidelity & Context Autopsy Lab

## Context

ADR-006 established the schema and metric contract for four fixed ablation pipelines.
A reducer populated by fixtures is not evaluation evidence. The lab now needs a local,
reproducible runtime that derives every normalized outcome from real component traces.

The runtime must run the same fixed corpus and exact fixed evaluation cases through:

1. `char_dense_naive`
2. `token_dense_naive`
3. `token_hybrid_naive`
4. `token_hybrid_rerank_budgeted`

It must preserve the distinction between an evidence failure at chunking, retrieval,
ranking, or context assembly. It must not infer a cause from an eval case's expected
failure label, fabricate a metric delta, or store raw evidence in the comparison report.

## Decision

Introduce `FourPipelineComparisonRunner` as a local in-process execution harness.

The runner:

- materializes character and sentence-aware token chunk corpora with one declared
  tokenizer and fixed chunking configuration;
- builds real dense and hybrid retrievers over those fixed chunks;
- runs every supplied case through every required pipeline exactly once;
- retrieves one fixed shared candidate-pool depth for all pipelines;
- computes Recall@5 by default from that shared candidate pool and serializes the cutoff
  as `retrieval_metric_k`;
- uses cross-encoder reranking only for `token_hybrid_rerank_budgeted`;
- uses measured `ContextAssembler` packing only for the budgeted intervention;
- derives `CasePipelineOutcome` values from typed runtime traces;
- retains raw typed traces only in memory for local inspection;
- passes only trace IDs and SHA-256 fingerprints into the serializable comparison report.

The naive pipelines select the first `final_evidence_chunk_limit` first-stage results.
Their `gold_evidence_included` field means **selected for the final evidence set**, not
that a final prompt was rendered or an LLM generated a grounded answer.

The budgeted pipeline uses the compact citation profile. This is a declared intervention
from Phase 6's context-pressure evidence, not an implicit formatting change.

A generic `retrieval_miss` taxonomy label is added for non-dense retrievers that fail
to return the full evidence-bearing chunk. `dense_retrieval_miss` remains available for
dense-only failures. This prevents the runner from falsely labeling a hybrid failure as
a dense-only failure.

## Consequences

### Positive

- Every report outcome is derived from the real component boundary that produced it.
- Fixed case-set parity and shared candidate-pool depth are enforced before aggregation.
- The runner cannot count an evidence inclusion that lacks a retrieval or reranking rank.
- Rich traces can be inspected locally without exporting synthetic source text or prompt
  content into the comparison artifact.
- The next slice can write a versioned JSON report from a stable runtime result rather
  than directly from model calls.

### Negative

- A full local run requires both Sentence Transformers models and, by default, a cached
  `tiktoken` encoding.
- Dense and cross-encoder scores can change when the declared model version changes.
  The report records model names through component trace fingerprints, but it does not
  yet lock model revisions or hashes.
- The synthetic corpus is small, so observed deltas are diagnostic evidence only, not a
  statistical claim about general production performance.
- Recall@5 and candidate-pool depth are separate controls: a deeper candidate pool can
  support reranking without inflating the reported first-stage recall cutoff.

## Rejected alternatives

### Use fixture outcomes in the final report

Rejected. Fixtures validate contracts but cannot support performance claims.

### Run a different case subset for the slowest pipeline

Rejected. Case-set parity is a hard comparison requirement.

### Serialize full traces with raw chunk text

Rejected. Trace fingerprints, ranks, counts, and failure labels are sufficient for the
comparison report. Full synthetic traces remain inspectable in local runtime memory.

### Treat a token-budgeted pipeline as automatically better

Rejected. Metrics are computed from actual fixed runs. The intervention may regress on
some cases and the report must retain that result.

## Non-claims

This runner does not evaluate generation, answer faithfulness, citations in model output,
customer data, latency, cost at production traffic, security controls, or production
readiness. `gold_evidence_included` proves only that known complete evidence survived to
the pipeline's final evidence boundary.
