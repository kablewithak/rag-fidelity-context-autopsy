# Public-Corpus Transfer Comparison Contract

## Purpose

This slice measures whether the existing four-pipeline evidence-survival harness
can execute against a small, pinned corpus of real public text.

It does **not** replace, modify, or verify against the fixed synthetic
four-pipeline baseline.

## Inputs

- Fixture: `data/public_transfer/squad_v1_dev_v1`
- Source documents: 10 grouped SQuAD v1.1 development-set article extracts
- Evaluation cases: 30 answerable public questions
- Pipelines: the existing fixed four pipeline definitions
- Runtime: explicit tokenizer, embedding model, reranker model, device, and
  execution configuration

## Adapter boundary

The public fixture is not coerced into the synthetic corpus taxonomy.

`rag_lab.public_transfer_runtime` creates narrow structural runner inputs:

- `PublicTransferRuntimeDocument`: document ID, text, character count, hash
- `PublicTransferRuntimeCase`: case ID, question, exact evidence, source ID
- `PublicTransferCaseReference`: external question ID, source document hash,
  answer offset, and hashes of question, answer, and evidence

The existing runner is reused only for its chunking, retrieval, ranking, context
assembly, and outcome derivation behavior. Public fixture provenance remains in
the separate reference objects and output artifact.

## Output boundary

The command:

```powershell
python .\scripts\run_public_transfer_comparison.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
```

writes only local, git-ignored files:

```text
outputs/public_transfer/squad_v1_dev_v1_current.json
outputs/public_transfer/squad_v1_dev_v1_current.md
```

The JSON artifact contains:

- public fixture identity and source SHA-256;
- bounded per-case provenance references;
- runtime configuration and model identities;
- four-pipeline metric and case-outcome report;
- bounded trace identifiers and hashes.

It contains no raw public document text, chunks, rendered contexts, prompts, or
model answers.

## Acceptance gate

A successful command prints:

```text
PUBLIC TRANSFER RUN: PASS
fixture=squad_v1_dev_v1
documents=10
cases=30
pipelines=4
```

After a public-transfer run, the synthetic guard must still pass independently:

```powershell
python .\scripts\run_comparison_baseline.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
```

Expected result:

```text
BASELINE REGRESSION GATE: PASS (4 pipelines, 18 cases)
```

## Non-claims

This is an external-validity probe, not:

- customer-data validation;
- a production benchmark;
- an answer-generation or grounding evaluation;
- proof that a repair sequence transfers;
- a replacement for the synthetic benchmark;
- a production-readiness claim.

The next slice reviews the measured transfer artifact, compares its failure
distribution with the synthetic benchmark, and records only findings supported by
those results.
