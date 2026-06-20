# ADR-008: Versioned comparison baseline artifacts and regression gate

- **Status:** Accepted
- **Date:** 2026-06-20
- **Scope:** Phase 7C, synthetic four-pipeline comparison baseline

## Context

Phase 7B can execute the four fixed RAG pipelines and emit a bounded
`comparison_report_v2` from real local traces. That result was initially visible only
through a local command and temporary output. A future code change could alter retrieval,
ranking, context selection, tokenizer alignment, corpus assets, or evaluation cases without
a reviewed before/after baseline.

A durable reliability asset needs more than an ad hoc successful terminal run. It needs:

1. a committed benchmark artifact;
2. explicit provenance for corpus, evaluation cases, models, tokenizer, and execution settings;
3. a concise human readout generated from the artifact rather than manually authored;
4. a command that regenerates a fresh local result and fails on meaningful regression; and
5. a controlled mechanism for intentionally replacing the baseline.

## Decision

Commit one reviewed synthetic benchmark artifact at:

```text
artifacts/comparisons/four_pipeline_baseline_v1.json
```

The artifact stores:

- the bounded `comparison_report_v2`;
- tokenizer, embedding model, reranker model, device, and execution configuration;
- hashes of the corpus manifest and evaluation-case manifest, not raw source content;
- an explicit zero-tolerance regression policy for this fixed benchmark.

Generate the executive readout at:

```text
docs/reports/four_pipeline_baseline_v1.md
```

The readout is deterministic output of the artifact renderer. It cannot be edited into a
claim not supported by the stored metrics.

Use the following command for one local reproduction plus regression verification:

```powershell
python .\scripts\run_comparison_baseline.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base `
    --verify
```

The command writes fresh, git-ignored artifacts under `outputs/comparisons/` and checks:

- provenance matches the reviewed baseline;
- fixed pipeline definitions and case coverage match;
- baseline-included evidence remains included;
- baseline top-k retrieval remains top-k retrieval;
- Recall@5, MRR@10, and evidence-inclusion rate do not fall;
- dropped-evidence rate does not rise.

A baseline rewrite requires both:

```text
--update-baseline
--confirm-baseline-update
```

That dual opt-in prevents a benchmark regression from being silently normalized as the new
reference.

## Consequences

### Benefits

- The repo now has an inspectable, versioned local benchmark rather than a terminal-only result.
- A future dashboard can consume the artifact without recomputing or inventing metrics.
- Model, tokenizer, corpus, evaluation-case, and configuration drift are visible immediately.
- Privacy is preserved: the committed artifact contains IDs, hashes, ranks, counts, labels, and
  metrics, not raw chunks, prompts, source documents, or generated answers.
- Intentional changes require an explicit review and new baseline decision.

### Costs and trade-offs

- The real verification command loads local Sentence Transformers models and is slower than the
  unit suite.
- Zero tolerance is strict. A deliberate model, tokenizer, corpus, or retrieval behavior change
  will fail the gate until the baseline is reviewed and updated.
- This gate validates the fixed synthetic benchmark only. It is not a customer-data validation,
  a production readiness gate, a latency benchmark, or a final-answer grounding evaluation.

## Non-claims

The baseline does not prove that compact citations always improve results, that the selected
models are best for every domain, that no production regression is possible, or that generated
answers are correct and grounded. It proves only the measured behavior of the declared four
pipelines on the declared fixed synthetic corpus and evaluation set.
