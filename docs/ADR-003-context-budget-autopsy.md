# ADR-003: Measure Rendered Context Capacity Before Claiming Evidence Reached Generation

## Status

Accepted.

## Context

A retrieved and reranked chunk does not automatically reach the generation model. It can be displaced by the real token cost of system instructions, query text, response contracts, citation wrappers, evidence separators, and reserved output capacity.

The first implementation represented prompt and schema tax as configured integer fields. That was useful as an early planner, but it was not strong enough for the portfolio claim. A declared number is not the same as counting the actual prompt that will be sent to the model.

## Decision

`ContextAssembler` now renders a deterministic prompt before packing candidates and counts it with the selected context tokenizer.

The measured budget is:

```text
max_context_tokens
- actual_static_prompt_tokens
- reserved_output_tokens
```

For each reranked candidate, the assembler counts:

- `chunk_token_count`: raw source chunk count under the chunking tokenizer;
- `raw_context_token_count`: raw source chunk count under the final context tokenizer;
- `tokenizer_count_delta`: context raw count minus chunk-time count;
- `rendered_context_token_count`: marginal token increase after source/rank/citation wrappers and separators are appended to the actual prompt;
- `rendering_token_tax`: rendered-context count minus raw-context count.

The assembler packs candidates in reranked order, does not silently truncate text, and records an include/drop decision for every candidate.

Two deterministic render profiles are retained for diagnosis:

- `verbose_audit`: explicit source, chunk, and rank XML-like metadata;
- `compact_citation`: a shorter source-and-rank reference.

They are not claimed to be universally optimal. They exist to measure how prompt formatting itself consumes evidence capacity.

## Consequences

### Positive

- Static instructions and response-contract text are measured rather than guessed.
- Citation and wrapper tax is visible at the individual-candidate level.
- The autopsy can prove a relevant chunk was retrieved and reranked but displaced by rendered prompt cost.
- The same final prompt text is available ephemerally for a future generation boundary, while reports retain hashes and bounded metadata only.

### Costs

- Token counts depend on the selected tokenizer and on the precise rendered prompt format.
- Changing prompt wording, citation style, or response contracts changes context capacity and must be re-evaluated.
- Rank-ordered packing is a transparent baseline, not a claim of coverage-optimal packing.

## Non-claims

This decision does not prove answer faithfulness, citation correctness, optimal prompt design, optimal context packing, provider-specific model compatibility, deployment readiness, or customer-data readiness.
