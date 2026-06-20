"""Prove a controlled character-boundary failure against a token-aware repair."""
from __future__ import annotations

import argparse
import json

from rag_lab.chunkers import CharacterChunker, SentenceAwareTokenChunker, build_chunking_report
from rag_lab.diagnostic_scenarios import (
    BOUNDARY_CASE_ID,
    load_diagnostic_case,
    load_stress_source_text,
)
from rag_lab.tokenizers import TiktokenTokenCounter, TokenCounter, UnicodeCodePointTokenCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a deliberately bad character boundary with sentence-aware token chunking."
    )
    parser.add_argument("--tokenizer", choices=("diagnostic", "tiktoken"), default="tiktoken")
    parser.add_argument("--tiktoken-encoding", default="cl100k_base")
    parser.add_argument("--sentence-max-tokens", type=int, default=200)
    return parser.parse_args()


def build_token_counter(*, kind: str, encoding: str) -> TokenCounter:
    if kind == "diagnostic":
        return UnicodeCodePointTokenCounter()
    return TiktokenTokenCounter(encoding_name=encoding)


def report_summary(report: object) -> dict[str, object]:
    # `report` is a ChunkingReport; keeping this presentation helper duck-typed avoids
    # exposing raw synthetic source text in JSON output.
    return {
        "chunker_name": report.chunker_name.value,
        "tokenizer_name": report.tokenizer_name,
        "chunk_count": report.chunk_count,
        "avg_token_count": report.avg_token_count,
        "max_token_count": report.max_token_count,
        "gold_evidence_preserved": report.gold_evidence_preserved,
        "gold_evidence_split": report.gold_evidence_split,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "token_count": chunk.token_count,
                "source_char_start": chunk.source_char_start,
                "source_char_end": chunk.source_char_end,
                "boundary_quality": chunk.boundary_quality.value,
            }
            for chunk in report.chunks
        ],
    }


def main() -> None:
    args = parse_args()
    if args.sentence_max_tokens < 1:
        raise SystemExit("--sentence-max-tokens must be at least 1")

    counter = build_token_counter(kind=args.tokenizer, encoding=args.tiktoken_encoding)
    case = load_diagnostic_case(BOUNDARY_CASE_ID)
    source_text = load_stress_source_text()
    evidence_start = source_text.find(case.gold_evidence_text)
    if evidence_start < 0:
        raise SystemExit("configured boundary evidence is absent from the stress source")
    # The first character window ends inside the known evidence sentence. This is an
    # explicit baseline failure fixture, not an accidental corpus property.
    character_window = evidence_start + 42

    character_chunks = CharacterChunker(
        token_counter=counter,
        max_characters=character_window,
    ).chunk(text=source_text, source_doc_id=case.source_doc_id)
    sentence_chunks = SentenceAwareTokenChunker(
        token_counter=counter,
        max_tokens=args.sentence_max_tokens,
    ).chunk(text=source_text, source_doc_id=case.source_doc_id)

    character_report = build_chunking_report(
        source_text=source_text,
        source_doc_id=case.source_doc_id,
        chunker_name=CharacterChunker.strategy,
        tokenizer_name=counter.name,
        chunks=character_chunks,
        gold_evidence_text=case.gold_evidence_text,
    )
    sentence_report = build_chunking_report(
        source_text=source_text,
        source_doc_id=case.source_doc_id,
        chunker_name=SentenceAwareTokenChunker.strategy,
        tokenizer_name=counter.name,
        chunks=sentence_chunks,
        gold_evidence_text=case.gold_evidence_text,
    )
    if not character_report.gold_evidence_split:
        raise SystemExit("baseline character fixture did not split the configured gold evidence")
    if not sentence_report.gold_evidence_preserved:
        raise SystemExit("sentence-aware token repair did not preserve the configured gold evidence")

    print(
        json.dumps(
            {
                "scenario": "controlled_chunk_boundary",
                "case_id": case.case_id,
                "tokenizer_name": counter.name,
                "character_baseline_window_characters": character_window,
                "sentence_aware_max_tokens": args.sentence_max_tokens,
                "character_baseline": report_summary(character_report),
                "sentence_aware_repair": report_summary(sentence_report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
