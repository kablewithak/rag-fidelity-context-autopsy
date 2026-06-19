# RAG Fidelity & Context Autopsy Lab

A local-first RAG reliability lab for diagnosing where evidence dies before generation.

The project compares chunking, retrieval, reranking, and context-assembly choices against fixed diagnostic cases. Its purpose is to show whether evidence was split, missed, ranked too low, or dropped under a tokenizer-specific context budget.

## North star

> Where did the evidence die, and which repair brought it back?

This is not a generic token counter or a prompt-only demo. It is an inspectable RAG reliability harness built around deterministic cases, typed reports, tests, and before/after evidence.

## Current capability

Phase 1 adds the chunking boundary of the harness:

- fixed-character baseline chunking;
- fixed token-window chunking;
- sentence-aware token chunking that preserves sentence, table-row, and log-event units where they fit the configured budget;
- source character spans, token counts, and boundary-quality metadata for every chunk;
- chunking reports showing whether the gold evidence was preserved or split;
- an offline deterministic diagnostic tokenizer for reliable local tests;
- an optional `tiktoken` adapter for later model-specific token-budget comparisons.

**Status:** locally validated on synthetic data. Retrieval, reranking, context assembly, Streamlit, deployment, and customer-data validation are not implemented yet.

## Why tokenization matters here

Tokenization is visible at the engineering boundary where text becomes model capacity:

1. Chunk sizes are measured in explicit token units, not only characters.
2. A different tokenizer can move a sentence across a token limit and change chunk boundaries.
3. Evidence can be retrieved only if a meaningful evidence-bearing chunk survives segmentation.
4. Later context packing will use the same token-counter contract to prove when evidence is dropped before generation.

> Token counts are tokenizer-specific. Recalculate budgets when changing models or tokenizers.

### Tokenizer posture in Phase 1

The default tests use `diagnostic:unicode_codepoint_v1`, an offline deterministic tokenizer. It is deliberately **not** presented as a production model tokenizer. This prevents an external tokenizer-vocabulary download from making the harness flaky.

The project also includes an optional `TiktokenTokenCounter` adapter for `cl100k_base`. When we add a runtime comparison or Streamlit demo, install it and pre-warm its cache deliberately:

```powershell
python -m pip install -e ".[tiktoken]"
```

That separation is intentional: **the harness boundary is proven first; model-specific tokenizer selection is made explicit and traceable later.**

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
│   ├── failure_taxonomy.py     # Failure labels and repair mapping
│   ├── tokenizers.py           # Offline + optional model-tokenizer adapters
│   └── chunkers.py             # Character, token, and sentence-aware chunkers
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

## Run tests

```powershell
python -m pytest
```

## Data and privacy posture

The corpus and evaluation cases are synthetic. Do not add real customer transcripts, customer support tickets, credentials, or personally identifiable information to this repository.

Trace and report layers must minimize retained text, avoid secret/PII logging, and use identifiers, hashes, and bounded metadata wherever raw content is unnecessary.

## Planned build order

1. Fixed eval cases and schemas — **complete**
2. Character, token, and sentence-aware chunking — **complete**
3. BM25, dense retrieval, hybrid fusion, and reranking
4. Token-budget-aware context autopsy and lost-evidence reports
5. Deterministic pipeline comparison and executive markdown export
6. Streamlit demonstration surface
7. Hugging Face Spaces CPU deployment

## Non-claims

This repository does not claim to eliminate hallucinations, prove all RAG systems improve, operate on customer data, or represent production readiness. It is an inspectable diagnostic lab operating on fixed synthetic cases.
