"""Render a deterministic repair-recommendation surface from the reviewed baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

from rag_lab.comparison_artifacts import (
    DEFAULT_BASELINE_ARTIFACT_PATH,
    load_baseline_artifact,
)
from rag_lab.repair_recommendations import (
    build_repair_recommendation_report,
    write_repair_recommendation_json,
    write_repair_recommendations_markdown,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = Path("outputs/reports/four_pipeline_repair_recommendations_v1.json")
DEFAULT_OUTPUT_MARKDOWN = Path("outputs/reports/four_pipeline_repair_recommendations_v1.md")


def parse_args() -> argparse.Namespace:
    """Parse only paths; benchmark execution is intentionally outside this script."""

    parser = argparse.ArgumentParser(
        description=(
            "Render deterministic repair recommendations from the committed comparison baseline."
        )
    )
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        default=DEFAULT_BASELINE_ARTIFACT_PATH,
        help="Reviewed comparison baseline artifact, relative to the repository root.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Local JSON output path, relative to the repository root.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MARKDOWN,
        help="Local Markdown output path, relative to the repository root.",
    )
    return parser.parse_args()


def main() -> int:
    """Load the reviewed artifact and write local recommendation outputs."""

    args = parse_args()
    baseline_path = _resolve_project_path(args.baseline_artifact)
    output_json_path = _resolve_project_path(args.output_json)
    output_markdown_path = _resolve_project_path(args.output_markdown)

    artifact = load_baseline_artifact(baseline_path)
    report = build_repair_recommendation_report(comparison_report=artifact.report)
    write_repair_recommendation_json(report=report, path=output_json_path)
    write_repair_recommendations_markdown(report=report, path=output_markdown_path)

    print(f"REPAIR RECOMMENDATION JSON WRITTEN: {_display_path(output_json_path)}")
    print(
        "REPAIR RECOMMENDATION READOUT WRITTEN: "
        f"{_display_path(output_markdown_path)}"
    )
    print(
        "REPAIR RECOMMENDATION REPORT: "
        f"{len(report.recommendations)} recommendation(s), "
        f"{report.evaluated_case_count} fixed cases"
    )
    return 0


def _resolve_project_path(path: Path) -> Path:
    """Resolve repository-relative defaults while preserving explicit absolute paths."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> Path:
    """Prefer repository-relative output in normal runs without rejecting absolute paths."""

    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main())
