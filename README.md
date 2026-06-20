# RAG Fidelity & Context Autopsy Lab

A local-first RAG reliability lab for diagnosing where evidence dies before generation.

The lab compares chunking, retrieval, reranking, and rendered-context assembly against fixed diagnostic cases. It shows whether evidence was split, missed, ranked too low, or dropped before it reached a generation boundary.

## North star

> Where did the evidence die, and which repair brought it back?

This is not a token-counter toy or a prompt-only demo. It is an inspectable RAG reliability harness with deterministic cases, typed Pydantic contracts, explicit failure labels, traceable reports, and regression tests.

## Current proof assets

The repository currently contains two controlled tokenization diagnostics in addition to the real retrieval pipeline:

1. **Chunk-boundary diagnostic**
   - a character window deliberately cuts a complete export clause;
   - sentence-aware token chunking preserves that same clause;
   - the report proves gold-evidence split versus preservation.

2. **Rendered-context pressure diagnostic**
   - complete gold evidence is present at reranked rank 3 in a fixed candidate trace;
   - verbose source/rank/chunk wrappers consume enough measured capacity to drop it;
   - compact citations retain it under the same calibrated context window;
   - the report separates raw chunk tokens, tokenizer-count alignment, actual static prompt cost, and rendered wrapper tax.

The deterministic scenarios are synthetic by design. They prove a specific mechanism, not universal prevalence.

## Why tokenization matters here

Tokenization appears at three separate engineering boundaries:

1. **Chunking boundary** — tokenizer-specific chunk limits decide where source text is divided. A poor character boundary can split a complete clause before retrieval begins.
2. **Lexical retrieval boundary** — BM25 uses lexical normalization for exact-term matching. This is separate from model-tokenization and is recorded as `lexical_analyzer_name`.
3. **Rendered-context boundary** — the selected tokenizer counts the actual static prompt, query, response contract, evidence separators, citation wrappers, raw evidence, and output reserve.

For an included evidence candidate, the autopsy records:

```text
chunk_token_count
raw_context_token_count
tokenizer_count_delta
rendered_context_token_count
rendering_token_tax
```

A zero `tokenizer_count_delta` in aligned mode is healthy. It does **not** mean the final prompt is free: `rendering_token_tax` exposes the cost of labels, citation metadata, separators, and prompt formatting.

> Token counts are tokenizer-specific. Re-run chunking and context-budget evals when changing models, tokenizer families, prompt templates, response contracts, or citation wrappers.

## Current capability

- fixed synthetic corpus and schema-validated JSONL cases;
- character, token-window, and sentence-aware token chunking;
- BM25 lexical retrieval;
- provider-neutral dense embedding boundary;
- reciprocal-rank fusion;
- cross-encoder reranking constrained to a fixed first-stage candidate set;
- tokenizer-provenance contracts and mismatch diagnostics;
- measured rendered-context packing with duplicate suppression and explicit drop reasons;
- lost-evidence reporting only when complete gold evidence was present after reranking and died during context assembly;
- SHA-256 text hashes and bounded report metadata instead of raw evidence in persistable autopsies.

**Status:** locally validated on synthetic data. Aggregate pipeline comparison, executive markdown export, Streamlit, deployment, customer-data validation, and production monitoring are not implemented yet.

## Local setup

Python 3.11+ is supported. The repository is currently validated on Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional local runtime dependencies only when you need real tokenization and local embedding/reranking smoke runs:

```powershell
python -m pip install -e ".[dev,tiktoken,dense]"
```

## Validation

```powershell
python -m pytest
```

## Run the diagnostic proofs

### 1. Character boundary failure versus sentence-aware token repair

```powershell
python .\scripts\run_chunking_boundary_diagnostic.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
```

Check:

```text
character_baseline.gold_evidence_split: true
sentence_aware_repair.gold_evidence_preserved: true
```

### 2. Rendered-context pressure baseline versus compact-wrapper repair

```powershell
python .\scripts\run_context_pressure_diagnostic.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
```

Check:

```text
baseline_verbose_audit.context_autopsy.gold_evidence_dropped: true
repair_compact_citation.context_autopsy.gold_evidence_included: true
baseline_verbose_audit.lost_evidence.loss_stage: context_assembly
```

### 3. Real local hybrid retrieval, reranking, and context autopsy

```powershell
python .\scripts\run_context_autopsy_smoke.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
```

The first real runtime may download model weights and tokenizer assets. This command proves that the local model path runs. The deterministic diagnostics remain the regression proof for the specific boundary and context-pressure mechanisms.

## Repository layout

```text
rag-fidelity-context-autopsy/
├── data/
│   ├── corpus/                 # Fixed synthetic source documents only
│   └── eval_cases.jsonl        # Fixed diagnostic cases
├── docs/
│   ├── ADR-001-hybrid-fusion.md
│   ├── ADR-002-cross-encoder-reranking.md
│   ├── ADR-003-context-budget-autopsy.md
│   ├── ADR-004-tokenizer-alignment.md
│   ├── ADR-005-diagnostic-tokenization-pressure.md
│   └── PROJECT_SCOPE.md
├── outputs/                    # Git-ignored generated reports
├── scripts/
│   ├── run_chunking_boundary_diagnostic.py
│   ├── run_context_pressure_diagnostic.py
│   ├── run_context_autopsy_smoke.py
│   ├── run_dense_smoke.py
│   ├── run_hybrid_smoke.py
│   └── run_rerank_smoke.py
├── rag_lab/
│   ├── chunkers.py
│   ├── context_assembly.py
│   ├── corpus_loader.py
│   ├── diagnostic_scenarios.py
│   ├── embedders.py
│   ├── eval_cases.py
│   ├── failure_taxonomy.py
│   ├── rerankers.py
│   ├── retrievers.py
│   ├── schemas.py
│   └── tokenizers.py
└── tests/
```

## Data and privacy posture

The corpus and eval cases are synthetic. Do not add customer transcripts, support tickets, credentials, secrets, private documents, or personal data to this repository.

`ContextAssemblyResult.context_text` is an ephemeral rendered prompt handoff. Persist or export the bounded `ContextAutopsyReport`, not raw context text, by default. Reports retain identifiers, tokenizer provenance, ranks, counts, hashes, and drop reasons where raw text is unnecessary.

## Non-claims

This lab does not claim to eliminate hallucinations, prove all RAG systems improve, choose the best model for every language, prove production readiness, or operate safely on customer data. It demonstrates specific RAG failure modes and repair patterns on fixed synthetic diagnostic cases.
