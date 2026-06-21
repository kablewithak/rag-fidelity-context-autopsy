# External Review Surface Contract

## Purpose

Expose the committed public-corpus transfer evidence in the existing Streamlit Executive Report so an external reviewer can inspect the project’s evidence boundary without rerunning models or reading local output folders.

## Data boundary

The Streamlit surface loads only:

- `artifacts/comparisons/four_pipeline_baseline_v1.json`
- `artifacts/public_transfer/public_transfer_squad_v1_dev_v1_reviewed_v1.json`
- `docs/reports/public_transfer_squad_v1_dev_v1_reviewed_v1.md`

The loader fails closed when:

- the public artifact does not have the reviewed identity;
- the public report does not exactly match its committed artifact;
- public and synthetic runs do not use the same fixed pipeline definitions, tokenizer, embedding model, reranker, device, or execution configuration;
- the public fixture is not the fixed 10-document, 30-case SQuAD v1.1 subset.

## Display rules

The Executive Report must display:

1. the synthetic 18-case benchmark and public 30-case transfer probe as separate suites;
2. the four public pipeline metric rows;
3. public fixture size and provenance identifiers;
4. the observed public ranking, evidence-inclusion, and non-uniform chunking findings;
5. an explicit statement that the rates must not be pooled into one headline score.

## Non-negotiables

- No live retrieval, embedding, reranking, or model execution in Streamlit.
- No public source documents, public questions, public answers, chunks, prompts, candidate scores, or generated answers shown in this reviewer route.
- No customer data.
- No final-answer, citation-correctness, latency, cost, security, or production-readiness claim.
- No additional benchmark baseline or public regression gate is created by this surface.

## Acceptance gate

The slice passes only when:

```powershell
python -m pytest .\tests\test_public_transfer_explorer.py
python -m pytest .\tests\test_streamlit_public_transfer_surface.py
python -m pytest
python .\scripts\run_comparison_baseline.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
python .\scripts\publish_public_transfer_review.py --check
python -m py_compile .\app\streamlit_app.py
git diff --check
```

A local Streamlit smoke test must confirm that the Executive Report shows the labelled public-transfer section while the sidebar continues to offer the same five read-only surfaces.
