# RAG Fidelity & Context Autopsy Lab

A local-first RAG reliability lab for diagnosing how tokenization, retrieval, reranking, and context assembly affect whether evidence reaches the model.

## North star

Show **where evidence dies** in a RAG pipeline:

- chunking splits or damages evidence;
- retrieval misses it;
- ranking leaves it too low;
- context assembly drops it under a token budget; or
- generation produces an answer unsupported by the supplied context.

The product compares a deliberately weak baseline with a stronger intervention pipeline and produces traceable before/after evidence on fixed diagnostic cases.

## Current milestone

**Phase 0 — Foundation and Eval Case Gate**

This initial commit establishes the non-negotiable harness contracts before retrieval or dashboard work:

- typed Pydantic v2 schemas at the data boundary;
- an explicit failure taxonomy and repair guidance;
- a synthetic, non-sensitive corpus;
- sixteen fixed JSONL evaluation cases;
- validation tests proving every gold-evidence string exists in its declared source document.

**Status:** locally testable foundation. It does not yet implement chunking, retrieval, reranking, context packing, Streamlit, or a production deployment.

## Repository layout

```text
rag-fidelity-context-autopsy/
├── data/
│   ├── corpus/                 # Synthetic source documents only
│   └── eval_cases.jsonl        # Fixed diagnostic cases
├── docs/
│   └── PROJECT_SCOPE.md
├── outputs/                    # Git-ignored generated reports
├── rag_lab/
│   ├── schemas.py              # Pydantic boundary contracts
│   ├── eval_cases.py           # JSONL loading and validation
│   └── failure_taxonomy.py     # Failure labels and repair mapping
└── tests/
```

## Local setup

This project supports Python 3.11+ and is currently validated on Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the foundation tests

```powershell
python -m pytest
```

## Data and privacy posture

The corpus and evaluation cases are synthetic. Do not add real customer transcripts, customer support tickets, credentials, or personally identifiable information to this repository.

Trace and report layers added later must minimize retained text, avoid secret/PII logging, and use identifiers, hashes, and bounded metadata wherever raw content is unnecessary.

## Planned build order

1. Fixed eval cases and schemas — **complete in this milestone**
2. Character, token, and sentence-aware chunking
3. BM25, dense retrieval, hybrid fusion, and reranking
4. Token-budget-aware context autopsy and lost-evidence reports
5. Deterministic pipeline comparison and executive markdown export
6. Streamlit demonstration surface
7. Hugging Face Spaces CPU deployment

## Non-claims

This repository does not claim to eliminate hallucinations, prove all RAG systems improve, operate on customer data, or represent production readiness. It is an inspectable diagnostic lab operating on fixed synthetic cases.
