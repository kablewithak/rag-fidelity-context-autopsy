/* AUTO-GENERATED — do not edit. Source: reviewed Python evaluation artifacts.
 * Regenerate: python scripts/export_research_publication_gate01.py --write
 * Figure 01 only. Synthetic evidence selection; no model reruns. */
const PIPELINES = Object.freeze([
  {
    "id": "char_dense_naive",
    "short": "C + D",
    "name": "Character + dense",
    "description": "Character chunking, dense retrieval"
  },
  {
    "id": "token_dense_naive",
    "short": "T + D",
    "name": "Token + dense",
    "description": "Sentence-aware token chunking, dense retrieval"
  },
  {
    "id": "token_hybrid_naive",
    "short": "T + H",
    "name": "Token + hybrid",
    "description": "Sentence-aware token chunking, hybrid retrieval"
  },
  {
    "id": "token_hybrid_rerank_budgeted",
    "short": "T + H + R",
    "name": "Hybrid + rerank",
    "description": "Sentence-aware token chunking, hybrid retrieval, reranking and budget-aware selection"
  }
]);
const CASES = Object.freeze([
  {
    "id": "legal_termination_001",
    "label": "Material breach",
    "group": "Legal clause",
    "query": "When can the customer terminate the agreement after a material breach?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "legal_payment_002",
    "label": "Invoice payment",
    "group": "Legal clause",
    "query": "How many days does a customer have to pay an undisputed invoice?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "legal_confidentiality_003",
    "label": "Confidentiality care",
    "group": "Legal clause",
    "query": "What level of care must each party use for confidential information?",
    "states": [
      [
        0,
        null,
        null,
        "chunking"
      ],
      [
        0,
        5,
        null,
        "ranking"
      ],
      [
        1,
        3,
        null,
        null
      ],
      [
        1,
        3,
        1,
        null
      ]
    ]
  },
  {
    "id": "legal_governing_law_004",
    "label": "Governing law",
    "group": "Legal clause",
    "query": "Which state's law governs the agreement?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "support_refund_005",
    "label": "Refund eligibility",
    "group": "Support policy",
    "query": "When is a customer eligible to request a refund?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "support_enterprise_s1_006",
    "label": "Enterprise severity 1",
    "group": "Support policy",
    "query": "What is the enterprise initial response target for a Severity 1 incident?",
    "states": [
      [
        0,
        4,
        null,
        "ranking"
      ],
      [
        1,
        2,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "support_security_notice_007",
    "label": "Security notification",
    "group": "Support policy",
    "query": "How quickly must Northstar notify a customer after confirmed unauthorized access?",
    "states": [
      [
        0,
        null,
        null,
        "chunking"
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        2,
        null
      ]
    ]
  },
  {
    "id": "pricing_team_plan_008",
    "label": "Team plan price",
    "group": "Pricing table",
    "query": "What is the monthly price and included seat count for the Team Plan?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        2,
        null,
        null
      ],
      [
        1,
        2,
        1,
        null
      ]
    ]
  },
  {
    "id": "pricing_annual_discount_009",
    "label": "Annual discount",
    "group": "Pricing table",
    "query": "What annual billing discount applies to the Business Plan?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "pricing_procurement_010",
    "label": "Procurement threshold",
    "group": "Pricing table",
    "query": "What happens when annual purchases exceed $250,000?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "multilingual_password_011",
    "label": "Password reset · FR",
    "group": "Multilingual",
    "query": "Comment réinitialiser un mot de passe et combien de temps le lien reste-t-il valide ?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "multilingual_security_012",
    "label": "Security notice · DE",
    "group": "Multilingual",
    "query": "Wann informiert Northstar betroffene Unternehmenskunden nach einem bestätigten Sicherheitsvorfall?",
    "states": [
      [
        0,
        null,
        null,
        "retrieval"
      ],
      [
        0,
        7,
        null,
        "ranking"
      ],
      [
        1,
        3,
        null,
        null
      ],
      [
        1,
        3,
        1,
        null
      ]
    ]
  },
  {
    "id": "code_sso_error_013",
    "label": "SSO error code",
    "group": "Incident log",
    "query": "Which error code appeared during the login failure spike?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "code_root_cause_014",
    "label": "SSO root cause",
    "group": "Incident log",
    "query": "What was identified as the root cause of the SSO incident?",
    "states": [
      [
        0,
        null,
        null,
        "chunking"
      ],
      [
        1,
        2,
        null,
        null
      ],
      [
        1,
        2,
        null,
        null
      ],
      [
        1,
        2,
        1,
        null
      ]
    ]
  },
  {
    "id": "faq_export_015",
    "label": "Workspace export",
    "group": "FAQ",
    "query": "Where can a workspace owner export workspace data?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "faq_delete_016",
    "label": "Workspace deletion",
    "group": "FAQ",
    "query": "Who can request workspace deletion and what happens after the request?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "token_boundary_export_017",
    "label": "Continuity export",
    "group": "Support policy",
    "query": "When must the provider deliver a recoverable customer data export after a Severity 1 continuity event?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  },
  {
    "id": "token_context_notice_018",
    "label": "Restricted-record notice",
    "group": "Support policy",
    "query": "How quickly must Northstar notify designated incident contacts after a confirmed restricted-record disclosure?",
    "states": [
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        null,
        null
      ],
      [
        1,
        1,
        1,
        null
      ]
    ]
  }
]);
const DATA_PROVENANCE = Object.freeze({
  "artifactId": "four_pipeline_baseline_v1",
  "runId": "local_four_pipeline_run_v2",
  "artifactSha256": "ab94200dfba7ae2aefdba6d04a2490183700a2fb65cd12c2ba15058a1a7938f6",
  "casesSha256": "9c2c1992a658a167ffc648ae530d8e4c63071cd885634247d42c957d336e9832",
  "url": "https://github.com/kablewithak/rag-fidelity-context-autopsy"
});
