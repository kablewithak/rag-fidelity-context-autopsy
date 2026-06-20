"""Prove how rendered prompt and citation tax can displace retrieved evidence."""
from __future__ import annotations

import argparse
import json

from rag_lab.context_assembly import (
    ContextAssembler,
    ContextAssemblyConfig,
    ContextRenderConfig,
    ContextRenderProfile,
    build_lost_evidence_report,
)
from rag_lab.diagnostic_scenarios import (
    DEFAULT_STRESS_CHUNK_MAX_TOKENS,
    build_context_pressure_trace,
    build_stress_chunks,
    load_stress_source_text,
)
from rag_lab.tokenizers import TiktokenTokenCounter, TokenCounter, UnicodeCodePointTokenCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic context-pressure autopsy where a verbose rendered prompt "
            "drops gold evidence and a compact citation wrapper restores it."
        )
    )
    parser.add_argument("--tokenizer", choices=("diagnostic", "tiktoken"), default="tiktoken")
    parser.add_argument("--tiktoken-encoding", default="cl100k_base")
    parser.add_argument("--reserved-output-tokens", type=int, default=120)
    parser.add_argument(
        "--stress-chunk-max-tokens",
        type=int,
        default=DEFAULT_STRESS_CHUNK_MAX_TOKENS,
        help=(
            "Token limit used to create the controlled stress chunks. The default is calibrated "
            "to produce at least four chunks under the selected tokenizer."
        ),
    )
    return parser.parse_args()


def build_token_counter(*, kind: str, encoding: str) -> TokenCounter:
    if kind == "diagnostic":
        return UnicodeCodePointTokenCounter()
    return TiktokenTokenCounter(encoding_name=encoding)


def assemble_full(*, trace: object, counter: TokenCounter, profile: ContextRenderProfile, reserve: int):
    return ContextAssembler(
        token_counter=counter,
        config=ContextAssemblyConfig(
            max_context_tokens=50_000,
            reserved_output_tokens=reserve,
            render_config=ContextRenderConfig(profile=profile),
        ),
    ).assemble(reranking_trace=trace)


def total_prompt_tokens(report: object) -> int:
    return (
        report.actual_static_prompt_tokens
        + report.used_rendered_evidence_tokens
        + report.reserved_output_tokens
    )


def main() -> None:
    args = parse_args()
    if args.reserved_output_tokens < 0:
        raise SystemExit("--reserved-output-tokens must be non-negative")
    if args.stress_chunk_max_tokens < 1:
        raise SystemExit("--stress-chunk-max-tokens must be at least 1")

    counter = build_token_counter(kind=args.tokenizer, encoding=args.tiktoken_encoding)
    source_text = load_stress_source_text()
    stress_chunks = build_stress_chunks(
        token_counter=counter,
        max_tokens=args.stress_chunk_max_tokens,
    )
    trace = build_context_pressure_trace(
        token_counter=counter,
        max_tokens=args.stress_chunk_max_tokens,
    )

    verbose_full = assemble_full(
        trace=trace,
        counter=counter,
        profile=ContextRenderProfile.VERBOSE_AUDIT,
        reserve=args.reserved_output_tokens,
    )
    compact_full = assemble_full(
        trace=trace,
        counter=counter,
        profile=ContextRenderProfile.COMPACT_CITATION,
        reserve=args.reserved_output_tokens,
    )
    verbose_decisions = verbose_full.report.decisions
    if len(verbose_decisions) != 3:
        raise SystemExit("context-pressure fixture requires exactly three ranked candidates")

    verbose_prefix_before_gold = (
        verbose_full.report.actual_static_prompt_tokens
        + verbose_decisions[0].rendered_context_token_count
        + verbose_decisions[1].rendered_context_token_count
        + args.reserved_output_tokens
    )
    compact_total = total_prompt_tokens(compact_full.report)
    verbose_total = total_prompt_tokens(verbose_full.report)
    calibrated_max_context_tokens = max(verbose_prefix_before_gold, compact_total)
    if calibrated_max_context_tokens >= verbose_total:
        raise SystemExit(
            "stress fixture did not create enough rendered-wrapper tax to demonstrate a repair"
        )

    baseline = ContextAssembler(
        token_counter=counter,
        config=ContextAssemblyConfig(
            max_context_tokens=calibrated_max_context_tokens,
            reserved_output_tokens=args.reserved_output_tokens,
            render_config=ContextRenderConfig(profile=ContextRenderProfile.VERBOSE_AUDIT),
        ),
    ).assemble(reranking_trace=trace)
    repair = ContextAssembler(
        token_counter=counter,
        config=ContextAssemblyConfig(
            max_context_tokens=calibrated_max_context_tokens,
            reserved_output_tokens=args.reserved_output_tokens,
            render_config=ContextRenderConfig(profile=ContextRenderProfile.COMPACT_CITATION),
        ),
    ).assemble(reranking_trace=trace)

    baseline_lost = build_lost_evidence_report(
        reranking_trace=trace,
        autopsy_report=baseline.report,
    )
    repair_lost = build_lost_evidence_report(
        reranking_trace=trace,
        autopsy_report=repair.report,
    )
    if not baseline.report.gold_evidence_dropped or baseline_lost is None:
        raise SystemExit("baseline fixture did not drop gold evidence during context assembly")
    if not repair.report.gold_evidence_included or repair_lost is not None:
        raise SystemExit("compact rendered-context repair did not retain gold evidence")

    print(
        json.dumps(
            {
                "scenario": "rendered_context_pressure",
                "case_id": trace.case_id,
                "tokenizer_name": counter.name,
                "stress_chunking": {
                    "source_token_count": counter.count(source_text),
                    "configured_chunk_max_tokens": args.stress_chunk_max_tokens,
                    "actual_chunk_count": len(stress_chunks),
                    "candidate_chunk_count": trace.candidate_count,
                },
                "calibration": {
                    "reserved_output_tokens": args.reserved_output_tokens,
                    "verbose_prompt_tokens_before_gold": verbose_prefix_before_gold,
                    "verbose_full_prompt_tokens": verbose_total,
                    "compact_full_prompt_tokens": compact_total,
                    "calibrated_max_context_tokens": calibrated_max_context_tokens,
                },
                "baseline_verbose_audit": {
                    "context_autopsy": baseline.report.model_dump(mode="json"),
                    "lost_evidence": baseline_lost.model_dump(mode="json"),
                },
                "repair_compact_citation": {
                    "context_autopsy": repair.report.model_dump(mode="json"),
                    "lost_evidence": None,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
