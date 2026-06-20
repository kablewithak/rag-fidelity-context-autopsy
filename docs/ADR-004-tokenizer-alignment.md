# ADR-004: Require Tokenizer Alignment for Raw Context-Budget Decisions

## Status

Accepted.

## Context

A raw chunk token count has no operational meaning unless the tokenizer that produced it is recorded. During the original Phase 6 smoke test, chunks were created with the offline Unicode-codepoint diagnostic counter while final context packing used `tiktoken:cl100k_base`.

The same evidence text therefore produced large negative deltas. That was not a token-budget regression. It was a measurement mismatch: Unicode code points and a model-family tokenizer are different counting systems.

## Decision

1. Each `TextChunk` records `tokenizer_name` alongside `token_count`.
2. Chunkers populate that provenance from their configured `TokenCounter`.
3. Context assembly rejects unattributed chunks.
4. Context assembly requires matching chunking and context tokenizers by default.
5. A mismatch is permitted only through `allow_tokenizer_mismatch=True` for an explicit diagnostic experiment.
6. Reports expose:
   - `context_tokenizer_name`;
   - `chunking_tokenizer_names`;
   - `tokenizer_alignment_status`;
   - `tokenizer_count_delta_detected`;
   - `budget_underestimation_detected` for positive raw-count deltas;
   - `budget_overestimation_detected` for negative raw-count deltas.
7. Rendered wrapper cost is reported separately as `rendering_token_tax`; it is not confused with tokenizer mismatch.

## Consequences

### Aligned runtime mode

The selected tokenizer is used for both raw chunk creation and final context packing. A zero raw tokenizer delta is expected for unmodified chunk text. That proves count alignment, not that wrappers are free.

### Explicit mismatch diagnostic mode

A deliberate mismatch can show how a proxy counter distorts raw budget planning. The report labels it `mismatched` and preserves the signed delta rather than calling it a regression.

## Non-claims

- Alignment does not prove the selected tokenizer is correct for every provider or model.
- A zero raw-count delta does not prove final prompt capacity; rendered wrapper tax still matters.
- Alignment does not prove context packing is optimal or that answers will be grounded.
