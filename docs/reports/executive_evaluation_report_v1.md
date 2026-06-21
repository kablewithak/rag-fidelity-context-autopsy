# Executive Evaluation Report v1

## Decision context

This is a local, fixed synthetic-data evidence-selection evaluation. It shows whether complete known evidence survives chunking, retrieval, ranking, and final context selection. It does not make a customer-data, production-readiness, generated-answer-grounding, or citation-correctness claim.

- **Reviewed artifact:** `four_pipeline_baseline_v1`
- **Reference run:** `local_four_pipeline_run_v2`
- **Fixed evaluation cases:** 18
- **Reported retrieval metric:** Recall@5
- **Tokenizer provenance:** `tiktoken:cl100k_base`

## Executive finding

On the fixed 18-case synthetic benchmark, `char_dense_naive` reached 72.2% evidence inclusion. The strongest reviewed pipeline, `token_hybrid_rerank_budgeted`, reached 100.0%. That is a 27.8-point evidence-inclusion difference, a 22.2-point Recall@5 difference, and a 0.236 MRR@10 difference within this fixed benchmark.

## Four-pipeline scorecard

| Pipeline | Recall@{k} | MRR@10 | Evidence inclusion | Dropped evidence among eligible candidates |
|---|---:|---:|---:|---:|
| Character + dense (`char_dense_naive`) | 77.8% | 0.736 | 72.2% | 0.0% (0/14) |
| Token + dense (`token_dense_naive`) | 94.4% | 0.852 | 88.9% | 0.0% (0/18) |
| Token + hybrid (`token_hybrid_naive`) | 100.0% | 0.870 | 100.0% | 0.0% (0/18) |
| Token + hybrid + rerank + budget (`token_hybrid_rerank_budgeted`) | 100.0% | 0.972 | 100.0% | 0.0% (0/18) |

Dropped-evidence rate uses only cases whose complete gold evidence entered the first-stage candidate set. It is not divided by all fixed evaluation cases, because evidence that never reached candidates cannot be dropped by context packing.

## Where the baseline lost evidence

| Evidence boundary | Affected fixed cases | Observed failure labels |
|---|---:|---|
| Chunking | 3 | `bad_chunk_boundary`, `gold_evidence_split` |
| Retrieval | 1 | `dense_retrieval_miss` |
| Ranking | 1 | `relevant_chunk_ranked_too_low` |

## Controlled context-budget finding

This is a separate local mechanism proof and is not counted as a standard four-pipeline benchmark failure.

- **Fixed pressure case:** `token_context_notice_018`
- **Tokenizer:** `tiktoken:cl100k_base`
- **Same calibrated context window:** 468 tokens
- **Reserved output allowance:** 120 tokens
- **Gold-evidence rank before context:** #3
- **Verbose audit wrappers:** gold evidence dropped by `budget_exhausted`; wrapper tax 77 tokens
- **Compact citation wrappers:** gold evidence retained; wrapper tax 25 tokens

## Ordered repair sequence

### 1. sentence_aware_token_chunking

- **Priority:** critical
- **Observed boundary:** chunking
- **Observed labels:** `bad_chunk_boundary`, `gold_evidence_split`
- **Action:** Replace character-boundary chunking with sentence-aware token chunking and retain bounded overlap only where a clause, table row, or event can cross the boundary.
- **Expected signal:** Watch Recall@5 and evidence inclusion. In this fixed run, the supporting pipeline moved Recall@5 from 77.8% to 94.4% and evidence inclusion from 72.2% to 88.9%.
- **Metrics to watch:** Recall@5, evidence inclusion
- **Trade-off:** Sentence-aware splitting increases implementation and tokenizer-provenance discipline; count final emitted text, including separators, rather than trusting configured unit budgets.
- **Supporting reviewed pipeline:** `token_dense_naive`

### 2. hybrid_retrieval

- **Priority:** critical
- **Observed boundary:** retrieval
- **Observed labels:** `dense_retrieval_miss`
- **Action:** Add BM25-backed hybrid retrieval so exact terms, identifiers, legal clauses, prices, and error codes can complement dense semantic recall.
- **Expected signal:** Watch Recall@5 before changing ranking. In this fixed run, the supporting pipeline moved Recall@5 from 77.8% to 100.0% and evidence inclusion from 72.2% to 100.0%.
- **Metrics to watch:** Recall@5, evidence inclusion
- **Trade-off:** Hybrid retrieval adds lexical-index and fusion parameters that must remain fixed and attributable across comparisons; do not conceal candidate-depth changes behind a higher headline cutoff.
- **Supporting reviewed pipeline:** `token_hybrid_naive`

### 3. cross_encoder_reranking

- **Priority:** high
- **Observed boundary:** ranking
- **Observed labels:** `relevant_chunk_ranked_too_low`
- **Action:** Apply cross-encoder reranking after candidate recall succeeds and before final context selection, so the most query-specific evidence reaches the context window.
- **Expected signal:** Watch MRR@10 after candidate recall is already sufficient. In this fixed run, the supporting pipeline moved MRR@10 from 0.736 to 0.972 while preserving Recall@5 at 100.0%.
- **Metrics to watch:** MRR@10, evidence inclusion
- **Trade-off:** Reranking adds per-candidate inference cost and latency; bound candidate depth first, then verify that ordering improved without trading away recall.
- **Supporting reviewed pipeline:** `token_hybrid_rerank_budgeted`

## Next evaluation gate

Before extending this result to a customer pilot, freeze an approved and privacy-reviewed evaluation set, preserve artifact provenance and trace fields, define acceptance thresholds before intervention, and add final-answer grounding and citation evaluation as a separate boundary.

## Evidence boundary

The report intentionally excludes raw source text, chunks, prompts, candidate scores, rendered context, and generated answers. It is evidence-selection and mechanism proof, not a claim that generated answers are correct.
