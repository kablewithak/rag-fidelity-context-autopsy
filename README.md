# RAG Fidelity & Context Autopsy Lab

A local-first RAG reliability lab for diagnosing where evidence dies before generation.

The project compares chunking, retrieval, reranking, and context-assembly choices against fixed diagnostic cases. Its purpose is to show whether evidence was split, missed, ranked too low, or dropped under a tokenizer-specific context budget.

## North star

> Where did the evidence die, and which repair brought it back?

This is not a generic token counter or a prompt-only demo. It is an inspectable RAG reliability harness built around deterministic cases, typed reports, tests, and before/after evidence.

## Current capability

Phase 5 adds an auditable cross-encoder reranking boundary:

- strict loading of the fixed synthetic corpus through a declared manifest;
- deterministic corpus integrity metadata, including source-text SHA-256 hashes;
- character, token-window, and sentence-aware token chunking;
- BM25 Okapi lexical retrieval over emitted chunks;
- a provider-neutral dense-embedding contract and cosine-similarity `DenseRetriever`;
- `HybridRetriever` using reciprocal rank fusion (RRF) over the full BM25 and dense rankings;
- `CrossEncoderReranker` that rescales only the fixed first-stage candidate set;
- typed reranking traces that retain first-stage rank, first-stage score, reranker score, reranked rank, model identity, and gold-evidence rank before and after reranking;
- deterministic tie-breaking: reranker score, then first-stage rank, then `chunk_id`;
- offline fixture scorers for unit tests, so CI does not download model weights.

**Status:** locally validated on synthetic data. Context assembly, lost-evidence reports, aggregate pipeline comparison, Streamlit, deployment, and customer-data validation are not implemented yet.

## Why tokenization matters here

Tokenization appears at three deliberately separate engineering boundaries:

1. **Model-tokenizer boundary:** chunk windows use the `TokenCounter` contract. A model tokenizer can change a chunk’s size and boundary behaviour.
2. **Lexical retrieval boundary:** BM25 normalizes words for term matching. This is not the same as a model tokenizer and is recorded as `lexical_analyzer_name` in a BM25 or hybrid trace.
3. **Context-budget boundary:** later context packing will count prompt, schema, retrieved evidence, and output-reserve tokens using an explicitly selected tokenizer.

Dense embeddings and cross-encoder rerankers are separate model boundaries. They are recorded with model names in traces but are not token-count measurements.

Hybrid fusion does **not** merge raw BM25 and cosine scores. They have different scales. Instead, RRF combines their positions in the two ranked lists and records the inputs behind the fused result.

> Token counts are tokenizer-specific. Recalculate budgets when changing models or tokenizers.

### Runtime posture

The default tests use `diagnostic:unicode_codepoint_v1`, an offline deterministic tokenizer. It is deliberately **not** presented as a production model tokenizer.

The project has optional runtime extras for actual tokenization, dense retrieval, and cross-encoder reranking:

```powershell
python -m pip install -e ".[dev,tiktoken,dense]"
```

Run one real hybrid-retrieval plus cross-encoder reranking trace over the synthetic corpus:

```powershell
python .\scripts\run_rerank_smoke.py
```

The first run may download the explicitly selected embedding and reranker models. The JSON output records the candidate set and the rank before and after reranking. Unit-test success alone does **not** claim a selected real model was downloaded, loaded, or benchmarked.

## Repository layout

```text
rag-fidelity-context-autopsy/
├── data/
│   ├── corpus/                 # Fixed synthetic source documents only
│   └── eval_cases.jsonl        # Fixed diagnostic cases
├── docs/
│   ├── ADR-001-hybrid-fusion.md
│   ├── ADR-002-cross-encoder-reranking.md
│   └── PROJECT_SCOPE.md
├── outputs/                    # Git-ignored generated reports
├── scripts/
│   ├── run_dense_smoke.py
│   ├── run_hybrid_smoke.py
│   └── run_rerank_smoke.py
├── rag_lab/
│   ├── schemas.py              # Pydantic boundary contracts
│   ├── eval_cases.py           # JSONL loading and validation
│   ├── corpus_loader.py        # Strict synthetic corpus manifest and chunk preparation
│   ├── failure_taxonomy.py     # Failure labels and repair mapping
│   ├── tokenizers.py           # Offline + optional model-tokenizer adapters
│   ├── embedders.py            # Provider-neutral dense embedding boundary
│   ├── chunkers.py             # Character, token, and sentence-aware chunkers
│   ├── retrievers.py           # BM25, dense, and rank-fused hybrid retrieval
│   └── rerankers.py            # Cross-encoder candidate rescoring
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
5. Hybrid fusion and first-stage retrieval comparison — **complete**
6. Cross-encoder reranking and before/after rank evidence — **complete**
7. Token-budget-aware context autopsy and lost-evidence reports
8. Deterministic pipeline comparison and executive markdown export
9. Streamlit demonstration surface
10. Hugging Face Spaces CPU deployment

## Non-claims

This repository does not claim to eliminate hallucinations, prove all RAG systems improve, operate on customer data, or represent production readiness. It is an inspectable diagnostic lab operating on fixed synthetic cases.
