"""Failure labels mapped to evidence-loss stages and consulting-oriented repair guidance."""

from __future__ import annotations

from dataclasses import dataclass

from rag_lab.schemas import EvidenceLossStage, FailureLabel


@dataclass(frozen=True, slots=True)
class FailureTaxonomyEntry:
    """A deterministic explanation and default repair for one failure label."""

    label: FailureLabel
    loss_stage: EvidenceLossStage
    definition: str
    repair_recommendation: str


FAILURE_TAXONOMY: dict[FailureLabel, FailureTaxonomyEntry] = {
    FailureLabel.BAD_CHUNK_BOUNDARY: FailureTaxonomyEntry(
        label=FailureLabel.BAD_CHUNK_BOUNDARY,
        loss_stage=EvidenceLossStage.CHUNKING,
        definition="A chunk boundary breaks a meaningful sentence, clause, table row, or code event.",
        repair_recommendation="Use sentence-aware token chunking and preserve clause, row, or event boundaries.",
    ),
    FailureLabel.GOLD_EVIDENCE_SPLIT: FailureTaxonomyEntry(
        label=FailureLabel.GOLD_EVIDENCE_SPLIT,
        loss_stage=EvidenceLossStage.CHUNKING,
        definition="Gold evidence is divided across chunks so no individual chunk carries enough meaning.",
        repair_recommendation="Use sentence-aware token chunking with bounded overlap around evidence-bearing boundaries.",
    ),
    FailureLabel.DENSE_RETRIEVAL_MISS: FailureTaxonomyEntry(
        label=FailureLabel.DENSE_RETRIEVAL_MISS,
        loss_stage=EvidenceLossStage.RETRIEVAL,
        definition="Dense retrieval does not return the evidence-bearing chunk among the candidate set.",
        repair_recommendation="Inspect embeddings and add a complementary lexical or hybrid retrieval path.",
    ),
    FailureLabel.KEYWORD_RETRIEVAL_NEEDED: FailureTaxonomyEntry(
        label=FailureLabel.KEYWORD_RETRIEVAL_NEEDED,
        loss_stage=EvidenceLossStage.RETRIEVAL,
        definition="Exact identifiers, legal terms, prices, or error codes require lexical recall.",
        repair_recommendation="Add BM25 or hybrid retrieval for exact legal, entity, table, and identifier terms.",
    ),
    FailureLabel.RERANKER_NEEDED: FailureTaxonomyEntry(
        label=FailureLabel.RERANKER_NEEDED,
        loss_stage=EvidenceLossStage.RANKING,
        definition="The candidate set contains evidence, but first-stage retrieval scoring is too coarse.",
        repair_recommendation="Use a cross-encoder reranker to rescore query-chunk relevance before packing context.",
    ),
    FailureLabel.RELEVANT_CHUNK_RANKED_TOO_LOW: FailureTaxonomyEntry(
        label=FailureLabel.RELEVANT_CHUNK_RANKED_TOO_LOW,
        loss_stage=EvidenceLossStage.RANKING,
        definition="Relevant evidence is retrieved but falls below the rank needed for final context inclusion.",
        repair_recommendation="Increase candidate recall and apply reranking before selecting final context chunks.",
    ),
    FailureLabel.RELEVANT_CHUNK_DROPPED_BY_BUDGET: FailureTaxonomyEntry(
        label=FailureLabel.RELEVANT_CHUNK_DROPPED_BY_BUDGET,
        loss_stage=EvidenceLossStage.CONTEXT_ASSEMBLY,
        definition="Retrieved evidence is excluded because the context pack exhausts its token budget.",
        repair_recommendation="Use budget-aware packing that reserves capacity for high-relevance evidence.",
    ),
    FailureLabel.CONTEXT_BUDGET_EXCEEDED: FailureTaxonomyEntry(
        label=FailureLabel.CONTEXT_BUDGET_EXCEEDED,
        loss_stage=EvidenceLossStage.CONTEXT_ASSEMBLY,
        definition="Prompt overhead, chunk tokens, or output reserve exceed available model context.",
        repair_recommendation="Reduce prompt and schema overhead, shrink chunks, and enforce a pre-generation budget check.",
    ),
    FailureLabel.DUPLICATE_CONTEXT_WASTE: FailureTaxonomyEntry(
        label=FailureLabel.DUPLICATE_CONTEXT_WASTE,
        loss_stage=EvidenceLossStage.CONTEXT_ASSEMBLY,
        definition="Near-duplicate chunks consume capacity without increasing evidence coverage.",
        repair_recommendation="Deduplicate similar chunks before final context packing.",
    ),
    FailureLabel.ANSWER_UNSUPPORTED_BY_CONTEXT: FailureTaxonomyEntry(
        label=FailureLabel.ANSWER_UNSUPPORTED_BY_CONTEXT,
        loss_stage=EvidenceLossStage.GENERATION,
        definition="The final answer states a claim not supported by the context delivered to the model.",
        repair_recommendation="Require evidence-linked answer checks and an unsupported-answer fallback.",
    ),
    FailureLabel.CITATION_MISSING_OR_WRONG: FailureTaxonomyEntry(
        label=FailureLabel.CITATION_MISSING_OR_WRONG,
        loss_stage=EvidenceLossStage.GENERATION,
        definition="A citation is absent, points to the wrong chunk, or does not support the attached claim.",
        repair_recommendation="Validate citation-to-evidence alignment before presenting an answer as grounded.",
    ),
    FailureLabel.TOKEN_BUDGET_REGRESSION: FailureTaxonomyEntry(
        label=FailureLabel.TOKEN_BUDGET_REGRESSION,
        loss_stage=EvidenceLossStage.CONTEXT_ASSEMBLY,
        definition="A prompt, tokenizer, model, or schema change reduces evidence capacity unexpectedly.",
        repair_recommendation="Track tokenizer-specific budget use and gate changes on retained gold evidence.",
    ),
}


def get_failure_taxonomy_entry(label: FailureLabel) -> FailureTaxonomyEntry:
    """Return the deterministic taxonomy entry for a validated failure label."""

    return FAILURE_TAXONOMY[label]
