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

**Phase 10A — Deterministic Executive Evaluation Report**

The repository now has six layers:

1. A reviewed, committed four-pipeline synthetic benchmark:
   ```text
   artifacts/comparisons/four_pipeline_baseline_v1.json
   docs/reports/four_pipeline_baseline_v1.md
   ```
2. A read-only Failure Case Explorer for inspecting the reviewed evidence lifecycle across all four pipelines.
3. A read-only Chunking Explorer for comparing character boundaries with sentence-aware token chunking over the same fixed synthetic case.
4. A read-only Retrieval Explorer for inspecting committed candidate presence, first-stage rank, reranked rank, final evidence selection, and bounded trace references.
5. A controlled Context Autopsy Explorer for measuring how static prompt tax and evidence wrappers can displace a reranked gold candidate under one calibrated token window.
6. A deterministic executive evaluation report that converts the reviewed baseline, observed repair sequence, and separate controlled context proof into a CTO-readable Markdown decision surface.

The benchmark artifact captures bounded comparison evidence and exact provenance:

- tokenizer, embedding model, reranker model, and device;
- corpus and evaluation-case manifest hashes;
- chunking, retrieval, hybrid fusion, reranking, and context settings;
- Recall@5, MRR@10, evidence-inclusion rate, dropped-evidence rate, failure counts, and per-case trace references.

The artifact stores IDs, hashes, ranks, counts, and metrics only. It does not serialize raw documents, chunks, prompts, rendered context, candidate scores, or generated answers.

**Status:** production-shaped local evaluation harness over synthetic data with locally runnable read-only demo surfaces and a deterministic executive readout. It is not a production deployment, customer-data evaluation, grounded-answer guarantee, or production-readiness claim.

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

Read the exact evidence, deterministic repair sequence, scope, and non-claims in:

```text
docs/reports/four_pipeline_baseline_v1.md
docs/reports/executive_evaluation_report_v1.md
```

## Repository layout

```text
rag-fidelity-context-autopsy/
├── app/
│   └── streamlit_app.py                   # Read-only exploration surfaces
├── artifacts/
│   └── comparisons/                        # Reviewed bounded benchmark artifacts
├── data/
│   ├── corpus/                             # Synthetic source documents only
│   └── eval_cases.jsonl                    # Fixed diagnostic cases
├── docs/
│   ├── reports/                            # Generated executive readouts committed for review
│   └── ADR-006...ADR-008                   # Comparison, runner, and baseline decisions
├── outputs/                                # Git-ignored fresh local comparison outputs
├── rag_lab/
│   ├── case_explorer.py                    # Typed read-only evidence-lifecycle views
│   ├── chunking_explorer.py                # Typed character versus token chunking views
│   ├── retrieval_explorer.py               # Typed reviewed candidate and rank views
│   ├── context_autopsy_explorer.py         # Typed controlled context-budget accounting view
│   ├── executive_evaluation_report.py      # Typed executive report contract and renderer
│   ├── chunkers.py                         # Character and sentence-aware token chunking
│   ├── retrievers.py                       # BM25, dense, and hybrid retrieval traces
│   ├── rerankers.py                        # Cross-encoder reranking traces
│   ├── context_assembly.py                 # Measured rendered-context packing
│   ├── comparison.py                       # Fixed comparison contracts and metric reducer
│   ├── comparison_runner.py                # Real four-pipeline execution harness
│   ├── comparison_artifacts.py             # Artifact, readout, and regression-gate contracts
│   └── schemas.py                          # Pydantic boundary contracts
├── scripts/
│   ├── run_four_pipeline_comparison.py
│   ├── run_comparison_baseline.py
│   ├── run_repair_recommendations.py
│   └── render_executive_evaluation_report.py
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

The app contains four local-only, read-only surfaces:

- **Failure case:** inspect a fixed case's reviewed evidence lifecycle across all four pipelines.
- **Chunking:** inspect actual standard character and sentence-aware outcomes, then open the separate controlled boundary probe for `token_boundary_export_017` to see a deliberately positioned character cut split the same gold clause.
- **Retrieval:** inspect the reviewed top-8 candidate boundary, first-stage rank, reranked rank where available, final evidence selection, loss stage, and bounded trace IDs.
- **Context autopsy:** inspect one fixed controlled pressure case where verbose audit wrappers drop a reranked gold candidate and compact citation wrappers retain it under the same calibrated context window.

```powershell
python -m streamlit run .\app\streamlit_app.py
```

Open the local URL printed by Streamlit. Select a fixed diagnostic case in the sidebar, then change the surface:

- **Failure case** shows query, gold evidence, expected answer, diagnostic note, baseline loss stage, failure labels, and per-pipeline ranks.
- **Chunking** deterministically emits local chunks with `tiktoken:cl100k_base` over the synthetic source document. It shows emitted chunks, measured token counts, source-character spans, and whether one emitted chunk contains the complete gold evidence. The `token_boundary_export_017` case also exposes a clearly labelled controlled boundary probe, which is separate from standard benchmark execution and intentionally places a character boundary inside the known clause.
- **Retrieval** reads the reviewed comparison artifact without rerunning models. It distinguishes evidence unavailable after chunking from a genuine candidate miss, shows candidate-pool depth separately from the Recall@5 reporting cutoff, displays first-stage and reranked ranks, and exposes bounded trace IDs only.
- **Context autopsy** executes only the fixed synthetic context-pressure trace with `tiktoken:cl100k_base`. It measures static prompt tax, per-candidate wrapper tax, include/drop decisions, and gold-evidence retention under two render profiles using the same calibrated context window. It does not rerun embeddings, retrieval, or reranking.

The app does not write benchmark artifacts or change the fixed corpus. The Context Autopsy Explorer is a controlled mechanism proof, not a claim that the standard reviewed benchmark contains a context-budget regression.

## Render and verify the executive evaluation report

The executive report is generated from the reviewed baseline artifact, deterministic observed repair sequence, and the separately labelled controlled context-pressure proof.

```powershell
python .\scripts\render_executive_evaluation_report.py

python .\scripts\render_executive_evaluation_report.py --check
```

The report is deterministic and contains no raw source text, chunks, prompts, candidate scores, rendered context, or generated answers. Its dropped-evidence rate is explicitly reported among cases whose complete gold evidence entered the candidate set, because evidence absent before context packing cannot be dropped by that stage.

## Reproduce and verify the baseline

```powershell
python .\scripts\run_comparison_baseline.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
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

The Chunking Explorer renders emitted chunks from the synthetic corpus only. The Retrieval Explorer displays only reviewed ranks, loss labels, and bounded trace IDs. The Context Autopsy Explorer displays bounded decision metadata only, never raw rendered context or candidate text. The executive report retains metrics, counts, IDs, failure labels, repairs, and controlled diagnostic accounting only. Keep rich traces local unless an approved review workflow requires more data.

## Planned build order

1. Fixed eval cases and schemas — **complete**
2. Character, token, and sentence-aware chunking — **complete**
3. BM25, dense retrieval, hybrid fusion, and reranking — **complete**
4. Token-budget-aware context autopsy and lost-evidence reports — **complete**
5. Comparison contracts, execution runner, and versioned benchmark — **complete**
6. Deterministic repair recommendation and executive report surfaces — **complete**
7. Streamlit Failure Case Explorer — **complete**
8. Streamlit Chunking Explorer — **complete** (standard view plus separate controlled boundary probe)
9. Streamlit Retrieval Explorer — **complete**
10. Controlled Streamlit Context Autopsy Explorer — **complete**
11. Deterministic executive evaluation report and committed Markdown readout — **complete**
12. Streamlit executive report surface and guided demo route
13. Hugging Face Spaces CPU deployment

## Streamlit Executive Report route

The **Executive report** route is the guided CTO entry point. It renders the typed
Phase 10A decision contract without calculating fresh metrics. It shows reviewed
pipeline progression, observed baseline failure stages, observed repair sequence, and
the separate controlled context-pressure proof.

Recommended walkthrough:

1. Start on **Executive report**.
2. Use **Chunking** to inspect evidence survival at segmentation.
3. Use **Retrieval** to separate candidate misses from ranking losses.
4. Use **Context autopsy** to inspect the separate measured wrapper-tax mechanism.
5. Return to the executive repair sequence. Do not infer generated-answer quality from
   evidence-selection metrics.

## Non-claims

This repository does not claim to eliminate hallucinations, prove all RAG systems improve, operate on customer data, represent production readiness, or validate final generated answers. It is an inspectable diagnostic lab operating on fixed synthetic cases.
