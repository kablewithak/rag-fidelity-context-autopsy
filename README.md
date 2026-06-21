---
title: RAG Fidelity & Context Autopsy
emoji: 🔎
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
python_version: "3.12"
startup_duration_timeout: 30m
short_description: Read-only RAG reliability benchmark demo.
---

# RAG Fidelity & Context Autopsy Lab

A local-first RAG reliability lab for diagnosing how tokenization, chunking, retrieval, reranking, and context assembly affect whether complete known evidence reaches the model.

## North star

Show **where evidence dies** in a RAG pipeline:

- chunking splits or damages evidence;
- retrieval misses it;
- ranking leaves it too low;
- context assembly drops it under a token budget; or
- generation produces an answer unsupported by the supplied context.

The lab compares a deliberately weak baseline with stronger interventions and produces traceable before/after evidence on fixed diagnostic cases.

## Current milestone

**Phase 12 — Hosted read-only evidence demonstration**

The project now has a completed, externally reviewable proof path:

1. A reviewed, committed four-pipeline synthetic benchmark:
   ```text
   artifacts/comparisons/four_pipeline_baseline_v1.json
   docs/reports/four_pipeline_baseline_v1.md
   ```
2. A separate reviewed public-corpus transfer probe:
   ```text
   artifacts/public_transfer/public_transfer_squad_v1_dev_v1_reviewed_v1.json
   docs/reports/public_transfer_squad_v1_dev_v1_reviewed_v1.md
   ```
3. Five read-only Streamlit surfaces: Executive report, Failure case, Chunking, Retrieval, and Context autopsy.
4. A deterministic executive evaluation report that turns reviewed evidence and repairs into a CTO-readable decision surface.
5. A reproducible Docker package for CPU-only hosting.
6. A public Hugging Face Space deployed from GitHub `main` through GitHub Actions.
7. A committed hosted-demo validation record:
   ```text
   docs/hosted_demo_validation_v1.md
   ```

The benchmark artifact retains IDs, hashes, ranks, counts, metrics, and bounded trace references. It does not serialize raw documents, chunks, prompts, rendered context, candidate scores, or generated answers.

**Hosted demo:** `https://huggingface.co/spaces/KaboKableMolefe/rag-fidelity-context-autopsy`

**Status:** production-shaped, synthetic-data validated, hosted read-only demonstration. It is not customer-data tested, a final-answer grounding guarantee, production monitored, load tested, or production ready.

## Four fixed pipelines

1. `char_dense_naive`
2. `token_dense_naive`
3. `token_hybrid_naive`
4. `token_hybrid_rerank_budgeted`

## Versioned synthetic baseline

| Pipeline | Recall@5 | MRR@10 | Evidence inclusion |
|---|---:|---:|---:|
| Character + dense | 77.8% | 0.736 | 72.2% |
| Token + dense | 94.4% | 0.852 | 88.9% |
| Token + hybrid | 100.0% | 0.870 | 100.0% |
| Token + hybrid + rerank + budget | 100.0% | 0.972 | 100.0% |

Read the exact evidence, deterministic repair sequence, public-transfer boundary, and non-claims in:

```text
docs/reports/four_pipeline_baseline_v1.md
docs/reports/executive_evaluation_report_v1.md
docs/reports/public_transfer_squad_v1_dev_v1_reviewed_v1.md
docs/hosted_demo_validation_v1.md
```

Synthetic and public-corpus rates are shown side by side for interpretation only. They must not be pooled into a single headline score.

## Repository layout

```text
rag-fidelity-context-autopsy/
├── .github/
│   └── workflows/
│       └── deploy-huggingface-space.yml        # GitHub main → Space mirror
├── app/
│   └── streamlit_app.py                         # Read-only exploration surfaces
├── artifacts/
│   ├── comparisons/                             # Reviewed synthetic benchmark artifacts
│   └── public_transfer/                         # Reviewed bounded public-transfer artifacts
├── data/
│   ├── corpus/                                  # Synthetic source documents only
│   ├── eval_cases.jsonl                         # Fixed synthetic diagnostic cases
│   └── public_transfer/                         # Fixed public fixture metadata
├── docs/
│   ├── reports/                                 # Committed deterministic readouts
│   └── hosted_demo_validation_v1.md             # Hosted evidence record
├── outputs/                                     # Git-ignored fresh local comparison outputs
├── rag_lab/
│   ├── case_explorer.py                         # Typed synthetic evidence-lifecycle views
│   ├── chunking_explorer.py                     # Typed chunking views
│   ├── retrieval_explorer.py                    # Typed rank and candidate views
│   ├── context_autopsy_explorer.py              # Controlled context-budget accounting
│   ├── public_transfer_explorer.py              # Bounded public-transfer view
│   ├── comparison.py                            # Fixed comparison contracts and metrics
│   ├── comparison_artifacts.py                  # Synthetic baseline contracts
│   ├── public_transfer_artifacts.py             # Public-transfer contracts
│   └── schemas.py                               # Pydantic boundary contracts
├── scripts/
│   ├── run_comparison_baseline.py
│   ├── run_public_transfer_comparison.py
│   └── publish_public_transfer_review.py
└── tests/
```

## Local setup

This project supports Python 3.11+ and is currently validated on Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dense,tiktoken,demo]"
```

The real comparison commands require the selected Sentence Transformers models and `tiktoken` encoding to be available locally. The runtime does not silently substitute another model or tokenizer.

## Run tests

```powershell
python -m pytest
```

## Run the read-only Streamlit explorers

```powershell
python -m streamlit run .\app\streamlit_app.py
```

The app contains five read-only surfaces:

- **Executive report:** reviewed scorecard, failure-stage summary, repair sequence, controlled context proof, and separately labelled public-transfer evidence.
- **Failure case:** a fixed synthetic case's reviewed evidence lifecycle across all four pipelines.
- **Chunking:** standard character and sentence-aware outcomes, plus a separately labelled controlled boundary probe.
- **Retrieval:** committed candidate presence, first-stage rank, reranked rank where available, final evidence selection, loss stage, and bounded trace IDs.
- **Context autopsy:** one fixed controlled pressure case where verbose audit wrappers drop a reranked gold candidate and compact citation wrappers retain it under the same calibrated context window.

The app is a read-only evidence surface. It does not rerun embeddings, retrieval, reranking, baseline generation, or answer generation.

## Reproduce and verify the synthetic baseline

```powershell
python .\scripts\run_comparison_baseline.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
```

The command executes the 18 fixed synthetic cases across all four pipelines, writes fresh ignored output under `outputs/comparisons/`, and compares the run to the committed baseline. It fails if provenance changes, baseline evidence is lost, Recall@5, MRR@10, or evidence inclusion falls, or dropped-evidence rate rises.

## Verify the reviewed public-transfer evidence

```powershell
python .\scripts\publish_public_transfer_review.py --check
```

The public-corpus transfer probe is separate from the synthetic benchmark. It measures evidence survival on a fixed public SQuAD v1.1 fixture and does not establish customer-data performance, final-answer correctness, or production readiness.

## Hosted deployment

GitHub `main` is the source of truth. The `Deploy Hugging Face Space` GitHub Actions workflow mirrors merged `main` to the Hugging Face Space through a secret-only credential boundary and verifies source parity after deployment.

Do not routinely push from a laptop to the `hf` remote. Use GitHub pull requests and merge to `main`. The `hf` remote is retained for source-parity inspection and incident investigation only.

Hosted acceptance is recorded in `docs/hosted_demo_validation_v1.md`. Re-run the hosted smoke check after a material app, Docker, artifact, or deployment-workflow change.

## Data and privacy posture

The synthetic corpus and synthetic evaluation cases are synthetic. The public-transfer fixture is a bounded, reviewed public corpus probe. Do not add customer transcripts, customer support tickets, credentials, or personally identifiable information to this repository.

The public Space loads bounded artifacts and reviewer-facing metrics only. It does not accept customer uploads, store visitor content, call external model APIs at request time, or expose raw public fixture questions, answers, chunks, prompts, retrieval candidates, or generated answers.

## Planned build order

1. Fixed eval cases and schemas — **complete**
2. Character, token, and sentence-aware chunking — **complete**
3. BM25, dense retrieval, hybrid fusion, and reranking — **complete**
4. Token-budget-aware context autopsy and lost-evidence reports — **complete**
5. Comparison contracts, execution runner, and versioned synthetic benchmark — **complete**
6. Deterministic repair recommendations and executive report surfaces — **complete**
7. Read-only Streamlit diagnostic and executive surfaces — **complete**
8. Public-corpus transfer fixture, runtime, and reviewed report — **complete**
9. Hugging Face Docker package and local container smoke — **complete**
10. Public Hugging Face Space deployment and GitHub Actions deployment parity — **complete**

## Non-claims

This repository does not claim to eliminate hallucinations, prove all RAG systems improve, validate customer data, guarantee final generated answers or citations, provide production monitoring or incident response, or represent production readiness. It is an inspectable diagnostic lab operating on fixed synthetic cases and one bounded public-corpus transfer probe.
