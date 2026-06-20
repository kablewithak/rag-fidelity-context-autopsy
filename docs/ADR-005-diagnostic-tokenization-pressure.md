# ADR-005: Add Controlled Boundary and Rendered-Context Pressure Diagnostics

## Status

Accepted.

## Context

The original synthetic corpus was intentionally small. Under a 500-token `cl100k_base` chunk limit, most source documents fit into one chunk. That was sufficient for wiring the retrieval and autopsy boundaries, but it did not visibly demonstrate the project north star:

> Where did the evidence die, and which repair brought it back?

A tokenization reliability lab needs controlled cases where chunking and rendered-context capacity materially change a measurable evidence outcome.

## Decision

Add `tokenization_stress_policy.txt` and two fixed eval cases:

1. `token_boundary_export_017`
   - a character-window baseline is deliberately positioned to cut a complete export obligation;
   - sentence-aware token chunking preserves the same clause as a complete chunk.

2. `token_context_notice_018`
   - a deterministic three-candidate reranking trace places complete notification evidence at rank 3;
   - verbose source/rank/chunk wrappers consume enough real rendered capacity to drop the gold chunk;
   - compact citations restore the evidence under the exact same calibrated context window.

The context-pressure trace is deterministic and explicitly labeled as a diagnostic candidate order. It isolates rendered-context accounting from live embedding or cross-encoder variability.

## Consequences

### Positive

- The lab now proves an actual boundary failure, not merely that chunkers have different implementations.
- The lab now proves a relevant chunk can die after ranking because prompt-rendering tax consumes the remaining window.
- Fixed diagnostics can become regression gates for future chunkers, prompt templates, and context-packing policies.

### Costs

- The cases are synthetic and intentionally constructed; they do not prove prevalence in arbitrary corpora.
- A deterministic reranking trace proves context assembly behavior, not live reranker quality.

## Non-claims

These diagnostics do not prove that every production RAG system has the same failure pattern, that compact citations are universally best, or that a live reranker will produce the exact fixed candidate order.
