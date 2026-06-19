# RAG Fidelity & Context Autopsy Lab

A local-first RAG reliability lab for diagnosing where evidence dies before generation.

The project compares chunking, retrieval, reranking, and context-assembly choices against fixed diagnostic cases. Its purpose is to show whether evidence was split, missed, ranked too low, or dropped under a tokenizer-specific context budget.

## North star

> Where did the evidence die, and which repair brought it back?

This is not a generic token counter or a prompt-only demo. It is an inspectable RAG reliability harness built around deterministic cases, typed reports, tests, and before/after evidence.

## Current capability

Phase 2 adds the first retrieval boundary of the harness:

- strict loading of the fixed synthetic corpus through a declared manifest;
- deterministic corpus integrity metadata, including source-text SHA-256 hashes;
- chunk preparation through the existing character, token-window, or sentence-aware chunker seam;
- BM25 Okapi lexical retrieval over emitted chunks;
- Unicode-aware lexical normalization that preserves exact terms, identifiers, and multilingual words where possible;
- typed retrieval traces with scores, stable ranks, candidate chunks, and gold-evidence recall;
- deterministic tie-breaking by `chunk_id` so reports do not change when scores tie.

**Status:** locally validated on synthetic data. Dense retrieval, hybrid fusion, cross-encoder reranking, context assembly, Streamlit, deployment, and customer-data validation are not implemented yet.

## Why tokenization matters here

Tokenization now appears at two deliberately separate engineering boundaries:

1. **Model-tokenizer boundary:** chunk windows use the `TokenCounter` contract. A model tokenizer can change a chunk’s size and boundary behaviour.
2. **Lexical retrieval boundary:** BM25 normalizes words for term matching. This is not the same as a model tokenizer and is recorded separately as `lexical_analyzer_name` in every retrieval trace.

That distinction prevents a common diagnostic mistake: treating lexical search terms, embedding inputs, and model context tokens as if they were the same segmentation system.

> Token counts are tokenizer-specific. Recalculate budgets when changing models or tokenizers.

### Tokenizer posture

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
│   ├── corpus/                 # Fixed synthetic source documents only
│   └── eval_cases.jsonl        # Fixed diagnostic cases
├── docs/
│   └── PROJECT_SCOPE.md
├── outputs/                    # Git-ignored generated reports
├── rag_lab/
│   ├── schemas.py              # Pydantic boundary contracts
│   ├── eval_cases.py           # JSONL loading and validation
│   ├── corpus_loader.py        # Strict synthetic corpus manifest and chunk preparation
│   ├── failure_taxonomy.py     # Failure labels and repair mapping
│   ├── tokenizers.py           # Offline + optional model-tokenizer adapters
│   ├── chunkers.py             # Character, token, and sentence-aware chunkers
│   └── retrievers.py           # BM25 lexical retrieval and typed traces
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
3. Corpus manifest and deterministic BM25 retrieval traces — **complete**
4. Dense retrieval, hybrid fusion, and first-stage comparison
5. Cross-encoder reranking and before/after rank evidence
6. Token-budget-aware context autopsy and lost-evidence reports
7. Deterministic pipeline comparison and executive markdown export
8. Streamlit demonstration surface
9. Hugging Face Spaces CPU deployment

## Non-claims

This repository does not claim to eliminate hallucinations, prove all RAG systems improve, operate on customer data, or represent production readiness. It is an inspectable diagnostic lab operating on fixed synthetic cases.
