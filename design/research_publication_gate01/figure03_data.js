/* AUTO-GENERATED — reviewed synthetic metric summary for Figure 03.
 * Regenerate: python scripts/export_research_publication_figure03.py --write
 * Figures 01–02 remain on the separate fixed-case projection. */
const FIGURE03_METRICS = Object.freeze({
  "artifactId": "four_pipeline_baseline_v1",
  "runId": "local_four_pipeline_run_v2",
  "retrievalMetricK": 5,
  "rows": [
    {
      "id": "char_dense_naive",
      "inclusion": {
        "numerator": 13,
        "denominator": 18,
        "value": 0.7222222222222222
      },
      "recallAt5": {
        "numerator": 14,
        "denominator": 18,
        "value": 0.7777777777777778
      },
      "mrrAt10": 0.7361111111111112
    },
    {
      "id": "token_dense_naive",
      "inclusion": {
        "numerator": 16,
        "denominator": 18,
        "value": 0.8888888888888888
      },
      "recallAt5": {
        "numerator": 17,
        "denominator": 18,
        "value": 0.9444444444444444
      },
      "mrrAt10": 0.8523809523809525
    },
    {
      "id": "token_hybrid_naive",
      "inclusion": {
        "numerator": 18,
        "denominator": 18,
        "value": 1.0
      },
      "recallAt5": {
        "numerator": 18,
        "denominator": 18,
        "value": 1.0
      },
      "mrrAt10": 0.8703703703703703
    },
    {
      "id": "token_hybrid_rerank_budgeted",
      "inclusion": {
        "numerator": 18,
        "denominator": 18,
        "value": 1.0
      },
      "recallAt5": {
        "numerator": 18,
        "denominator": 18,
        "value": 1.0
      },
      "mrrAt10": 0.9722222222222222
    }
  ]
});
