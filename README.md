# RAG Fidelity & Context Autopsy Lab

A local-first RAG reliability lab for diagnosing how tokenization, retrieval, reranking, and context assembly affect whether evidence reaches the model.

## North star

Show **where evidence dies** in a RAG pipeline:

- chunking splits or damages evidence;
- retrieval misses it;
- ranking leaves it too low;
- context assembly drops it under a token budget; or
- generation produces an answer unsupported by the supplied context.

The lab compares a deliberately weak baseline with stronger interventions and produces traceable before/after evidence on fixed diagnostic cases.

## Current milestone

**Phase 7C — Versioned baseline artifact and regression gate**

The repository now contains a reviewed, committed four-pipeline synthetic benchmark:

```text
artifacts/comparisons/four_pipeline_baseline_v1.json
docs/reports/four_pipeline_baseline_v1.md
```

The artifact captures a bounded comparison report plus the exact benchmark provenance:

- tokenizer, embedding model, reranker model, and device;
- corpus and evaluation-case manifest hashes;
- chunking, retrieval, hybrid fusion, reranking, and context settings;
- Recall@5, MRR@10, evidence-inclusion rate, dropped-evidence rate, failure counts, and per-case trace references.

The report stores IDs, hashes, ranks, counts, and metrics only. It does not serialize raw documents, chunks, prompts, rendered context, or generated answers.

**Status:** production-shaped local evaluation harness over synthetic data. It is not a production deployment, customer-data evaluation, grounded-answer guarantee, or production-readiness claim.

## Four fixed pipelines

1. `char_dense_naive`
2. `token_dense_naive`
3. `token_hybrid_naive`
4. `token_hybrid_rerank_budgeted`

## Versioned baseline result

| Pipeline | Recall@5 | MRR@10 | Evidence inclusion |
|---|---:|---:|---:|
| Character + dense | 77.8% | 0.736 | 72.2% |
| Token + dense | 94.4% | 0.852 | 88.9% |
| Token + hybrid | 100.0% | 0.870 | 100.0% |
| Token + hybrid + rerank + budget | 100.0% | 0.972 | 100.0% |

Read the exact evidence, scope, and non-claims in:

```text
docs/reports/four_pipeline_baseline_v1.md
```

## Repository layout

```text
rag-fidelity-context-autopsy/
├── artifacts/
│   └── comparisons/            # Reviewed bounded benchmark artifacts
├── data/
│   ├── corpus/                 # Synthetic source documents only
│   └── eval_cases.jsonl        # Fixed diagnostic cases
├── docs/
│   ├── reports/                # Generated executive readouts committed for review
│   └── ADR-006...ADR-008       # Comparison, runner, and baseline decisions
├── outputs/                    # Git-ignored fresh local comparison outputs
├── rag_lab/
│   ├── chunkers.py             # Character and sentence-aware token chunking
│   ├── retrievers.py           # BM25, dense, and hybrid retrieval traces
│   ├── rerankers.py            # Cross-encoder reranking traces
│   ├── context_assembly.py     # Measured rendered-context packing
│   ├── comparison.py           # Fixed comparison contracts and metric reducer
│   ├── comparison_runner.py    # Real four-pipeline execution harness
│   ├── comparison_artifacts.py # Artifact, readout, and regression-gate contracts
│   └── schemas.py              # Pydantic boundary contracts
├── scripts/
│   ├── run_four_pipeline_comparison.py
│   └── run_comparison_baseline.py
└── tests/
```

## Local setup

This project supports Python 3.11+ and is currently validated on Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dense,tiktoken]"
```

The real comparison commands require the selected Sentence Transformers models and `tiktoken` encoding to be available locally. The runtime does not silently substitute another model or tokenizer.

## Run tests

```powershell
python -m pytest
```

## Reproduce and verify the baseline

```powershell
python .\scripts\run_comparison_baseline.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base `
    --verify
```

The command:

1. executes all 18 fixed cases across all four pipelines;
2. writes fresh git-ignored JSON and Markdown outputs under `outputs/comparisons/`;
3. compares the fresh result to the reviewed committed baseline; and
4. fails if provenance changes, baseline evidence is lost, Recall@5, MRR@10, or evidence inclusion falls, or dropped-evidence rate rises.

An intentional benchmark update requires both `--update-baseline` and `--confirm-baseline-update` after review.

## Run the raw comparison only

```powershell
python .\scripts\run_four_pipeline_comparison.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base `
    --retrieval-metric-k 5
```

## Data and privacy posture

The corpus and evaluation cases are synthetic. Do not add real customer transcripts, customer support tickets, credentials, or personally identifiable information to this repository.

Keep rich traces local unless an approved review workflow requires more data. The committed comparison artifacts retain identifiers, hashes, ranks, counts, metrics, and failure labels rather than raw chunks, source documents, prompts, or generated answers.

## Planned build order

1. Fixed eval cases and schemas — **complete**
2. Character, token, and sentence-aware chunking — **complete**
3. BM25, dense retrieval, hybrid fusion, and reranking — **complete**
4. Token-budget-aware context autopsy and lost-evidence reports — **complete**
5. Comparison contracts, execution runner, and versioned benchmark — **complete**
6. Streamlit demonstration surface
7. Hugging Face Spaces CPU deployment

## Non-claims

This repository does not claim to eliminate hallucinations, prove all RAG systems improve, operate on customer data, represent production readiness, or validate final generated answers. It is an inspectable diagnostic lab operating on fixed synthetic cases.
