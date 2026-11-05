/* GENERATED from a separately reviewed public transfer artifact.
 * No public fixture questions, answers, chunks, prompts or model calls.
 * Regenerate: python -m scripts.export_research_publication_figure05 --write */
const FIGURE05_PUBLIC = Object.freeze({
  "schemaVersion": "figure05_public_transfer_v1",
  "scope": "separate_reviewed_public_transfer_not_pooled",
  "artifactId": "public_transfer_squad_v1_dev_v1_reviewed_v1",
  "runId": "public_transfer_squad_v1_dev_v1_reviewed_v1",
  "datasetId": "squad_v1.1_dev",
  "datasetVersion": "1.1",
  "license": "CC BY-SA 4.0",
  "sourceSha256": "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
  "fixtureManifestSha256": "d3b7c5647be5a3f8e8ec338a880acd4b0b690335e3b45608dcdc53666593f75b",
  "sourceDocumentCount": 10,
  "publicCaseCount": 30,
  "syntheticCaseCount": 18,
  "recallCutoff": 5,
  "rows": [
    {
      "id": "char_dense_naive",
      "inclusion": {
        "numerator": 26,
        "denominator": 30,
        "value": 0.8666666666666667
      },
      "recallAt5": {
        "numerator": 30,
        "denominator": 30,
        "value": 1.0
      },
      "mrrAt10": 0.7816666666666666,
      "lossStages": {
        "chunking": 0,
        "retrieval": 0,
        "ranking": 4,
        "context_assembly": 0,
        "generation": 0
      }
    },
    {
      "id": "token_dense_naive",
      "inclusion": {
        "numerator": 27,
        "denominator": 30,
        "value": 0.9
      },
      "recallAt5": {
        "numerator": 29,
        "denominator": 30,
        "value": 0.9666666666666667
      },
      "mrrAt10": 0.8086507936507936,
      "lossStages": {
        "chunking": 0,
        "retrieval": 0,
        "ranking": 3,
        "context_assembly": 0,
        "generation": 0
      }
    },
    {
      "id": "token_hybrid_naive",
      "inclusion": {
        "numerator": 28,
        "denominator": 30,
        "value": 0.9333333333333333
      },
      "recallAt5": {
        "numerator": 29,
        "denominator": 30,
        "value": 0.9666666666666667
      },
      "mrrAt10": 0.8638888888888889,
      "lossStages": {
        "chunking": 0,
        "retrieval": 1,
        "ranking": 1,
        "context_assembly": 0,
        "generation": 0
      }
    },
    {
      "id": "token_hybrid_rerank_budgeted",
      "inclusion": {
        "numerator": 29,
        "denominator": 30,
        "value": 0.9666666666666667
      },
      "recallAt5": {
        "numerator": 29,
        "denominator": 30,
        "value": 0.9666666666666667
      },
      "mrrAt10": 0.9333333333333333,
      "lossStages": {
        "chunking": 0,
        "retrieval": 1,
        "ranking": 0,
        "context_assembly": 0,
        "generation": 0
      }
    }
  ]
});
