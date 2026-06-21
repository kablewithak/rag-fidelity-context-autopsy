"""Render or verify the deterministic executive evaluation report."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_lab.executive_evaluation_report import (
    DEFAULT_EXECUTIVE_EVALUATION_REPORT_PATH,
    load_executive_evaluation_report,
    render_executive_evaluation_report_markdown,
    write_executive_evaluation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render or verify the artifact-backed executive evaluation report."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed Markdown report differs from the deterministic render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    report = load_executive_evaluation_report(project_root=project_root)
    output_path = project_root / DEFAULT_EXECUTIVE_EVALUATION_REPORT_PATH
    expected = render_executive_evaluation_report_markdown(report=report)

    if args.check:
        if not output_path.exists():
            raise SystemExit(
                f"EXECUTIVE REPORT GATE: FAIL (missing {DEFAULT_EXECUTIVE_EVALUATION_REPORT_PATH})"
            )
        actual = output_path.read_text(encoding="utf-8")
        if actual != expected:
            raise SystemExit(
                "EXECUTIVE REPORT GATE: FAIL "
                "(committed Markdown differs from deterministic executive report render)"
            )
        print(
            "EXECUTIVE REPORT GATE: PASS "
            f"({DEFAULT_EXECUTIVE_EVALUATION_REPORT_PATH})"
        )
        return

    write_executive_evaluation_report(report=report, path=output_path)
    print(f"EXECUTIVE REPORT WRITTEN: {DEFAULT_EXECUTIVE_EVALUATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
