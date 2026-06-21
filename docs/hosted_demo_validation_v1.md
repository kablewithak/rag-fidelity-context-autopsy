# Hosted Demo Validation v1

## Status

**Hosted read-only demonstration: validated**

This record freezes the first externally reachable deployment proof for the RAG Fidelity & Context Autopsy lab. It records deployment source, smoke scope, security boundary, and explicit non-claims. It is not a benchmark update, customer-data evaluation, production SLO, or production-readiness certification.

## Deployment identity

| Field | Value |
|---|---|
| Hosted platform | Hugging Face Spaces |
| Space | `KaboKableMolefe/rag-fidelity-context-autopsy` |
| Space repository | `https://huggingface.co/spaces/KaboKableMolefe/rag-fidelity-context-autopsy` |
| Runtime | Docker Space, CPU Basic |
| Deployment source of truth | GitHub `main` |
| Deployment automation | GitHub Actions workflow `Deploy Hugging Face Space` |
| Deployment source merge | `7cea06e` — PR #25, GitHub Actions to Hugging Face deployment path |

## Deployment boundary

The hosted Space is a read-only diagnostic and reviewer surface.

It loads committed synthetic baseline, committed bounded public-transfer artifacts, and committed deterministic report material. It does not accept uploads, store visitor content, invoke external LLM APIs at request time, regenerate benchmark results, rerun dense retrieval, rerank candidates, or generate answers.

The public-transfer view remains a separate external-validity probe. Its rates are displayed beside the synthetic benchmark for interpretation only and are not pooled into a headline metric.

## Validation evidence

### Source integrity

- GitHub `main` was clean after PR #25 merge.
- The deployment workflow completed successfully as reported by the GitHub Actions green run.
- The hosted Space became running after the workflow deployment.
- The reviewer manually navigated the hosted application after deployment.

### Hosted smoke scope

The hosted browser check confirmed:

1. The application loaded without a visible traceback.
2. The sidebar exposed exactly five read-only surfaces:
   - Executive report
   - Failure case
   - Chunking
   - Retrieval
   - Context autopsy
3. Retrieval Autopsy rendered a fixed diagnostic case with reviewed candidate-state, rank, and final-evidence fields.
4. Context Autopsy rendered the fixed wrapper-tax mechanism proof.
5. The Executive Report rendered the separately labelled public-corpus transfer probe.
6. The public probe reported the fixed 10-document and 30-case fixture.
7. The UI retained the no-pooling boundary between synthetic and public rates.
8. The hosted UI did not expose raw public fixture questions, answers, chunks, prompts, rendered context, customer data, or generated answers.

## Deployed maturity

```text
synthetic-data validated
public-corpus transfer measured
locally Docker-validated
hosted read-only demonstration
```

## Non-claims

This validation does not establish:

- customer-data testing;
- final-answer correctness or citation correctness;
- production latency, availability, capacity, or cost behaviour;
- security penetration testing, incident response, monitoring, or alerting;
- multi-tenant isolation;
- production readiness.

## Revalidation triggers

Re-run hosted validation after any material change to:

- `Dockerfile` or `.dockerignore`;
- Streamlit entrypoint or public-review loader;
- reviewed benchmark or public-transfer artifacts;
- deployment workflow or deployment secret boundary;
- Hugging Face Space runtime configuration.

## Operator rule

GitHub `main` is the deployment source of truth. Deploy through the GitHub Actions workflow. Do not routinely push local branches or uncommitted changes to the Hugging Face Space remote.
