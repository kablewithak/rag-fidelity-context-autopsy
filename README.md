# RAG Fidelity & Context Autopsy Lab

A local-first RAG reliability lab for diagnosing how tokenization, retrieval, reranking, and context assembly affect whether evidence reaches the model.

## North star

Show **where evidence dies** in a RAG pipeline:

- chunking splits or damages evidence;
- retrieval misses it;
- ranking leaves it too low;
- context assembly drops it under a token budget; or
- generation produces an answer unsupported by the supplied context.

The lab compares a deliberately weak baseline with a stronger intervention pipeline and produces traceable before/after evidence on fixed diagnostic cases.

## Current milestone

**Phase 7B — Real four-pipeline execution runner**

The repository now has a schema-first comparison contract and a local runner that derives
one normalized outcome per fixed case and pipeline from real typed component traces.

The runner executes the required ablations:

1. `char_dense_naive`
2. `token_dense_naive`
3. `token_hybrid_naive`
4. `token_hybrid_rerank_budgeted`

For each case, it records whether complete gold evidence:

- survived chunking;
- was found in the shared first-stage candidate pool;
- was high enough in the final ordering for selection; and
- for the budgeted intervention, was included or dropped by measured context packing.

The runner retrieves eight first-stage candidates per pipeline and reports **Recall@5**
from that shared candidate pool. The `comparison_report_v2` artifact serializes
`retrieval_metric_k` so the cutoff behind every retrieval-recall value is explicit.

The machine-readable comparison report contains only bounded IDs, hashes, ranks, counts,
metrics, and failure labels. Rich component traces are retained only in local process
memory for inspection and are not serialized into that report.

**Status:** production-shaped local evaluation harness over synthetic data. It is not a
production deployment, a customer-data evaluation, or a grounded-answer guarantee.

## Repository layout

```text
rag-fidelity-context-autopsy/
├── data/
│   ├── corpus/                 # Synthetic source documents only
│   └── eval_cases.jsonl        # Fixed diagnostic cases
├── docs/
│   ├── ADR-006-four-pipeline-comparison-harness.md
│   └── ADR-007-real-four-pipeline-execution.md
├── outputs/                    # Git-ignored generated reports
├── rag_lab/
│   ├── chunkers.py             # Character and sentence-aware token chunking
│   ├── retrievers.py           # BM25, dense, and hybrid retrieval traces
│   ├── rerankers.py            # Cross-encoder reranking traces
│   ├── context_assembly.py     # Measured rendered-context packing
│   ├── comparison.py           # Fixed comparison contracts and metric reducer
│   ├── comparison_runner.py    # Real four-pipeline execution harness
│   └── schemas.py              # Pydantic boundary contracts
├── scripts/
│   └── run_four_pipeline_comparison.py
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

The first real comparison run also requires the selected Sentence Transformers models
and `tiktoken` encoding to be available locally. The runtime does not silently substitute
another model or tokenizer.

## Run tests

```powershell
python -m pytest
```

## Run the real four-pipeline comparison

```powershell
python .\scripts\run_four_pipeline_comparison.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base `
    --retrieval-metric-k 5
```

The command runs every fixed evaluation case through all four pipelines and prints a
privacy-bounded JSON comparison report to stdout. The metric cutoff must not exceed the
shared `retrieval_top_k` candidate depth, and the report records the selected cutoff.

This Phase 7B command does **not** write a report file. Phase 7C will add explicit,
versioned output writing plus the executive markdown report once the runtime result has
been reviewed and stabilized.

## Data and privacy posture

The corpus and evaluation cases are synthetic. Do not add real customer transcripts,
customer support tickets, credentials, or personally identifiable information to this
repository.

The comparison report retains identifiers, hashes, ranks, counts, metrics, and failure
labels rather than raw chunks, source documents, prompts, or generated answers. Keep rich
traces local unless a later approved review workflow requires more data.

## Planned build order

1. Fixed eval cases and schemas — **complete**
2. Character, token, and sentence-aware chunking — **complete**
3. BM25, dense retrieval, hybrid fusion, and reranking — **complete**
4. Token-budget-aware context autopsy and lost-evidence reports — **complete**
5. Comparison contracts and real four-pipeline runner — **in progress**
6. Versioned JSON/markdown comparison report and regression gate
7. Streamlit demonstration surface
8. Hugging Face Spaces CPU deployment

## Non-claims

This repository does not claim to eliminate hallucinations, prove all RAG systems improve,
operate on customer data, represent production readiness, or validate final generated
answers. It is an inspectable diagnostic lab operating on fixed synthetic cases.
