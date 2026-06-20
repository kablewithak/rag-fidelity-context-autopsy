# ADR-006: Four-Pipeline Comparison Harness Contract

- **Status:** Accepted
- **Date:** 2026-06-20
- **Decision owner:** RAG Fidelity & Context Autopsy Lab

## Context

The lab now proves individual RAG reliability mechanisms: unsafe chunk boundaries,
lexical versus dense retrieval behavior, hybrid fusion, reranking boundaries,
tokenizer alignment, and rendered-context evidence loss.

Those components are not yet an evaluation product. A buyer needs one reproducible
comparison artifact answering:

> Which pipeline retained evidence, where did the weaker pipeline fail, and what changed?

The project PRD requires four fixed ablations:

1. `char_dense_naive`
2. `token_dense_naive`
3. `token_hybrid_naive`
4. `token_hybrid_rerank_budgeted`

## Decision

Introduce a schema-first comparison layer before building the live execution runner.

The layer defines:

- immutable pipeline definitions for the four fixed ablations;
- one bounded outcome per `(pipeline_id, case_id)`;
- trace references by ID and SHA-256, never raw evidence or prompts;
- deterministic metrics with retained numerators and denominators;
- a report-level retrieval metric cutoff, independent from the candidate-pool depth;
- baseline-to-intervention deltas;
- validation that every pipeline ran the same fixed non-empty case set exactly once; and
- validation that every pipeline uses one shared candidate-pool depth that is at least the
  retrieval metric cutoff.

The first implementation is a pure reducer. It receives already-derived outcomes and
does not load models, invoke retrievers, read corpus files, or write outputs.

## Metric semantics

- **Recall@5 by default:** first-stage retrieval found complete gold evidence at rank five
  or better, divided by all evaluated cases. The runner may retrieve a deeper shared
  candidate pool for hybrid fusion and reranking, but the report serializes
  `retrieval_metric_k` so the recall cutoff is never inferred from that pool depth.
- **MRR@10:** reciprocal rank of the final ranking stage used for context selection; zero when gold evidence is absent or ranked below ten; averaged across all cases.
- **Evidence-inclusion rate:** complete gold evidence reached final context, divided by all evaluated cases.
- **Dropped-evidence rate:** evidence loss at `context_assembly`, divided by cases where first-stage retrieval found the gold evidence.
- **Failure counts:** frequency by standardized failure label and loss stage.

A `MetricRate` stores numerator, denominator, and exact derived value so no percentage is detached from its sample size.

## Consequences

### Positive

- The future runner has a strict output contract before it executes expensive local models.
- Baseline and intervention results cannot silently use different case sets.
- The report can be serialized to JSON without raw synthetic evidence, source text, or rendered prompts.
- Metric deltas are mechanically derived rather than hand-written into a portfolio claim.
- A deeper retrieval pool cannot silently make recall look stronger because the report-level
  cutoff is independently declared and validated.

### Negative

- This ADR does not yet produce actual pipeline results.
- A report built from fixture outcomes is only a contract test, not performance evidence.
- The next slice must derive outcomes from real chunking, retrieval, reranking, and context traces.

## Rejected alternatives

### Start with a dashboard

Rejected. A UI before reproducible metrics would make the comparison visually persuasive
without making it auditable.

### Store only final percentages

Rejected. A percentage without numerator/denominator hides case counts and makes small
sample changes look stronger than they are.

### Let each pipeline choose a different case subset

Rejected. That would destroy causal comparison.

### Export raw evidence in reports

Rejected. The lab's default privacy posture is to retain bounded identifiers, ranks,
counts, hashes, and labels when raw text is unnecessary.

## Non-claims

This contract does not prove any pipeline improves quality, works on customer data,
chooses a universal best retriever, or is production-ready. Those claims require the
next execution slice, fixed real runs, review of failures, and later customer-specific validation.
