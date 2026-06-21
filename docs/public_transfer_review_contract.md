# Public-Transfer Review Contract

## Purpose

This contract freezes one measured SQuAD v1.1 public-corpus transfer run as an externally reviewable portfolio evidence asset. It is deliberately separate from the controlled 18-case synthetic baseline.

## Inputs

The review requires the deterministic `squad_v1_dev_v1` fixture, exactly 10 public source documents and 30 public cases, the same four fixed pipeline definitions, and the same tokenizer, embedding model, reranker, device, and execution configuration as the reviewed synthetic baseline.

## Outputs

Publication creates exactly two reviewed files:

```text
artifacts/public_transfer/public_transfer_squad_v1_dev_v1_reviewed_v1.json
docs/reports/public_transfer_squad_v1_dev_v1_reviewed_v1.md
```

The artifact contains bounded provenance, case references, metrics, ranks, loss labels, and trace hashes. It contains no raw public documents, questions, answers, chunks, prompts, or generated answers.

The Markdown report is rendered deterministically from the reviewed public artifact and the synthetic baseline. `--check` fails when the report drifts.

## Review boundary

| Suite | Purpose | Case count | Policy |
|---|---|---:|---|
| Synthetic baseline | Controlled mechanism diagnosis | 18 | Reviewed regression gate |
| Public SQuAD transfer | External-validity probe on public prose | 30 | Reviewed measurement artifact; no regression gate |

Do not average or pool rates across these suites. Side-by-side presentation is allowed only when the report labels the boundary clearly.

## Publication gate

```powershell
python .\scripts\publish_public_transfer_review.py `
    --publish `
    --confirm-public-transfer-review `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base

python .\scripts\publish_public_transfer_review.py --check
```

The synthetic baseline must still pass independently:

```powershell
python .\scripts\run_comparison_baseline.py `
    --tokenizer tiktoken `
    --tiktoken-encoding cl100k_base
```

## Allowed claim

> The RAG evidence-survival harness was measured on a separate fixed public SQuAD v1.1 probe. The public run retained fixture provenance and showed how chunking, retrieval, ranking, and final context selection behaved on non-authored public prose.

## Non-claims

This review does not establish final-answer correctness, citation correctness in generated answers, customer-data performance, production latency, cost, cross-vendor stability, security controls, or production readiness.
