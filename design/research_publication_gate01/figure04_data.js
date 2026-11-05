/* GENERATED from the existing deterministic controlled context-pressure view.
 * No raw text, chunks, prompts, answers, or model calls.
 * Regenerate: python scripts/export_research_publication_figure04.py --write */
const FIGURE04_PROOF = Object.freeze({
  "schemaVersion": "figure04_context_budget_v1",
  "status": "measured",
  "caseId": "token_context_notice_018",
  "tokenizer": "tiktoken:cl100k_base",
  "windowTokens": 468,
  "reservedOutputTokens": 120,
  "goldRank": 3,
  "profiles": [
    {
      "id": "verbose_audit",
      "staticPromptTokens": 78,
      "rawEvidenceTokens": 172,
      "wrapperTaxTokens": 77,
      "remainingTokens": 21,
      "includedCount": 2,
      "droppedCount": 1,
      "goldIncluded": false,
      "goldDropReason": "budget_exhausted",
      "decisions": [
        {
          "rank": 1,
          "firstStageRank": 1,
          "rawTokens": 93,
          "renderedTokens": 131,
          "wrapperTokens": 38,
          "isGold": false,
          "included": true,
          "dropReason": null
        },
        {
          "rank": 2,
          "firstStageRank": 2,
          "rawTokens": 79,
          "renderedTokens": 118,
          "wrapperTokens": 39,
          "isGold": false,
          "included": true,
          "dropReason": null
        },
        {
          "rank": 3,
          "firstStageRank": 3,
          "rawTokens": 73,
          "renderedTokens": 112,
          "wrapperTokens": 39,
          "isGold": true,
          "included": false,
          "dropReason": "budget_exhausted"
        }
      ]
    },
    {
      "id": "compact_citation",
      "staticPromptTokens": 78,
      "rawEvidenceTokens": 245,
      "wrapperTaxTokens": 25,
      "remainingTokens": 0,
      "includedCount": 3,
      "droppedCount": 0,
      "goldIncluded": true,
      "goldDropReason": null,
      "decisions": [
        {
          "rank": 1,
          "firstStageRank": 1,
          "rawTokens": 93,
          "renderedTokens": 101,
          "wrapperTokens": 8,
          "isGold": false,
          "included": true,
          "dropReason": null
        },
        {
          "rank": 2,
          "firstStageRank": 2,
          "rawTokens": 79,
          "renderedTokens": 87,
          "wrapperTokens": 8,
          "isGold": false,
          "included": true,
          "dropReason": null
        },
        {
          "rank": 3,
          "firstStageRank": 3,
          "rawTokens": 73,
          "renderedTokens": 82,
          "wrapperTokens": 9,
          "isGold": true,
          "included": true,
          "dropReason": null
        }
      ]
    }
  ],
  "scope": "controlled_context_pressure_not_four_pipeline_benchmark"
});
