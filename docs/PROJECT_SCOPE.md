# Phase 0 Scope — Foundation and Eval Case Gate

## Objective

Establish the fixed evaluation assets and typed contracts that every later RAG pipeline must respect.

The first implementation gate is deliberately narrow:

- at least 15 fixed diagnostic cases;
- Pydantic validation for every case;
- every case references an existing synthetic source document;
- every gold-evidence string exists exactly in that source document;
- all expected failure labels map to an explicit loss stage and repair recommendation.

## Why this comes before retrieval

A dashboard can make an unmeasured system look convincing. This project will instead build a testable harness first, then compare pipelines against the same cases.

The later baseline and intervention pipelines must be able to answer:

1. Did chunking preserve the evidence?
2. Did retrieval find the evidence?
3. Was the evidence ranked high enough?
4. Did the context budget retain it?
5. Was the final answer supported by the retained context?

## Phase 0 acceptance criteria

```text
python -m pytest
```

must pass with tests that prove case-schema validity, corpus integrity, case uniqueness, exact gold-evidence linkage, and total taxonomy coverage.

## Privacy boundary

`data/corpus/` is synthetic demonstration material only. Do not treat it as a customer-data ingestion path.
