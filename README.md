# RAG Fidelity & Context Autopsy Lab

A local-first RAG reliability lab for diagnosing where evidence dies before generation.

The project compares chunking, retrieval, reranking, and context-assembly choices against fixed diagnostic cases. Its purpose is to show whether evidence was split, missed, ranked too low, or dropped under a tokenizer-specific context budget.

## North star

> Where did the evidence die, and which repair brought it back?

This is not a generic token counter or a prompt-only demo. It is an inspectable RAG reliability harness built around deterministic cases, typed reports, tests, and before/after evidence.

## Current capability

Phase 3 adds the dense-retrieval boundary while keeping model downloads outside the deterministic test suite:

- strict loading of the fixed synthetic corpus through a declared manifest;
- deterministic corpus integrity metadata, including source-text SHA-256 hashes;
- character, token-window, and sentence-aware token chunking;
- BM25 Okapi lexical retrieval over emitted chunks;
- a provider-neutral dense-embedding contract and cosine-similarity `DenseRetriever`;
- a lazy CPU-first `SentenceTransformerEmbeddingModel` adapter for explicitly selected models;
- typed retrieval traces that record either lexical-analyzer provenance, embedding-model provenance, or both in the future hybrid pipeline;
- deterministic tie-breaking by `chunk_id` so reports do not change when scores tie;
- offline fixture vectors for dense-retrieval unit tests, so CI does not download model weights.

**Status:** locally validated on synthetic data. Hybrid fusion, cross-encoder reranking, context assembly, Streamlit, deployment, and customer-data validation are not implemented yet.

## Why tokenization matters here

Tokenization appears at three deliberately separate engineering boundaries:

1. **Model-tokenizer boundary:** chunk windows use the `TokenCounter` contract. A model tokenizer can change a chunk’s size and boundary behaviour.
2. **Lexical retrieval boundary:** BM25 normalizes words for term matching. This is not the same as a model tokenizer and is recorded as `lexical_analyzer_name` in a BM25 trace.
3. **Embedding-model boundary:** dense retrieval converts an entire chunk and query into fixed-dimension vectors. That semantic representation is recorded through `embedding_model_name` and `embedding_dimension`; it is not a token-count measure.

That distinction prevents a common diagnostic mistake: treating lexical search terms, embedding inputs, and model context tokens as if they were the same segmentation system.

> Token counts are tokenizer-specific. Recalculate budgets when changing models or tokenizers.

### Tokenizer and embedding posture

The default tests use `diagnostic:unicode_codepoint_v1`, an offline deterministic tokenizer. It is deliberately **not** presented as a production model tokenizer.

The project has optional runtime extras for an actual tokenizer and dense model adapter:

```powershell
python -m pip install -e ".[dev,tiktoken,dense]"
```

Run one real Sentence Transformers retrieval trace over the synthetic corpus:

```powershell
python .\scripts\run_dense_smoke.py
```

The first run may download the explicitly selected model (`sentence-transformers/all-MiniLM-L6-v2` by default). Its model identity and vector dimension appear in the output trace. Unit-test success alone does **not** claim the selected real model was downloaded, loaded, or benchmarked.

## Repository layout

```text
rag-fidelity-context-autopsy/
├── data/
│   ├── corpus/                 # Fixed synthetic source documents only
│   └── eval_cases.jsonl        # Fixed diagnostic cases
├── docs/
│   └── PROJECT_SCOPE.md
├── outputs/                    # Git-ignored generated reports
├── scripts/
│   └── run_dense_smoke.py      # Optional real-model local smoke run
├── rag_lab/
│   ├── schemas.py              # Pydantic boundary contracts
│   ├── eval_cases.py           # JSONL loading and validation
│   ├── corpus_loader.py        # Strict synthetic corpus manifest and chunk preparation
│   ├── failure_taxonomy.py     # Failure labels and repair mapping
│   ├── tokenizers.py           # Offline + optional model-tokenizer adapters
│   ├── embedders.py            # Provider-neutral dense embedding boundary
│   ├── chunkers.py             # Character, token, and sentence-aware chunkers
│   └── retrievers.py           # BM25 and cosine dense retrieval with typed traces
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
4. Dense retrieval boundary and Sentence Transformers runtime adapter — **complete**
5. Hybrid fusion and first-stage retrieval comparison
6. Cross-encoder reranking and before/after rank evidence
7. Token-budget-aware context autopsy and lost-evidence reports
8. Deterministic pipeline comparison and executive markdown export
9. Streamlit demonstration surface
10. Hugging Face Spaces CPU deployment

## Non-claims

This repository does not claim to eliminate hallucinations, prove all RAG systems improve, operate on customer data, or represent production readiness. It is an inspectable diagnostic lab operating on fixed synthetic cases.
