# Public Corpus Transfer Evaluation Contract

## Purpose

This workstream tests whether the repository's evidence-fidelity harness can be applied to a small, independently sourced corpus of real public prose. It is intentionally separate from the fixed synthetic benchmark.

```text
synthetic baseline = controlled mechanism benchmark
public transfer fixture = external-validity probe
customer corpus = later scoped pilot only
```

The transfer fixture does **not** replace, merge with, or rewrite the synthetic 18-case / four-pipeline baseline. It must never overwrite the reviewed baseline artifact, baseline regression policy, synthetic cases, or current Streamlit surfaces.

## Selected source

The initial transfer fixture is materialized from the SQuAD v1.1 development set.

```text
external_dataset_id=squad_v1.1_dev
dataset_version=1.1
source_url=https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json
license=CC BY-SA 4.0
```

SQuAD v1.1 contains question-answer pairs associated with Wikipedia-derived passages, where the answer is a text span in the passage. The materializer records a SHA-256 digest of the raw downloaded source, source-document identifiers, question identifiers, answer spans, answer-bearing evidence sentences, and per-document text hashes.

## Fixed selection policy

The first fixture is intentionally small and deterministic:

```text
10 grouped source documents
8 answer-bearing source paragraphs per document
3 answerable cases per document
30 total public transfer cases
```

Selection runs in SQuAD source order. It retains the first ten articles that contain at least eight paragraphs with valid answer spans, then retains the first three answerable questions found across those retained paragraphs. The algorithm is code, not manual curation.

Artifacts are written to:

```text
data/public_transfer/squad_v1_dev_v1/
├── manifest.json
├── corpus.jsonl
├── cases.jsonl
└── ATTRIBUTION.md
```

`manifest.json` is the dataset contract. `corpus.jsonl` contains the grouped real-source documents. `cases.jsonl` contains the question, answer span, answer-bearing evidence sentence, document hash, and external question identifier. `ATTRIBUTION.md` preserves source and licensing context.

## Materialization command

```powershell
python .\scripts\materialize_squad_v1_transfer_fixture.py
```

Re-run only with `--overwrite` when deliberately replacing a reviewed fixture:

```powershell
python .\scripts\materialize_squad_v1_transfer_fixture.py --overwrite
```

Validate a materialized fixture without downloading anything:

```powershell
python .\scripts\materialize_squad_v1_transfer_fixture.py --check
```

## Evaluation boundary for the next slice

The next implementation slice may adapt the existing four pipelines to consume this fixture and produce a separate transfer report. It must keep these boundaries:

```text
- no synthetic baseline changes
- no synthetic and public metric blending
- no fresh answer-generation claims
- score evidence containment/retrieval before answer quality
- retain source ID, source hash, question ID, and failure labels per outcome
- report outcomes as transfer evidence, not production or customer validation
```

## Acceptance criteria

This phase is accepted only when:

```text
- source JSON is downloaded over HTTPS
- the raw source SHA-256 is recorded in the manifest
- exactly 10 documents and 30 cases are materialized
- every case answer span matches its retained real-text document
- every case carries external dataset and question provenance
- attribution is written beside the derived artifacts
- synthetic baseline files and regression policy remain untouched
```

## Non-claims

A 30-case public transfer fixture is not a representative benchmark, customer-data validation, a safety certification, or production readiness. It is an auditable, reproducible external-text probe that makes the next benchmark claim narrower and more credible.
