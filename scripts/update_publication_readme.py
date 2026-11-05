"""Insert a bounded, idempotent hosted-publication note without overwriting README.

Usage: python -m scripts.update_publication_readme --write
       python -m scripts.update_publication_readme --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ANCHOR = "## Hosted deployment\n"
SECTION = """## Hosted publication — Gate 07

The public Hugging Face Docker Space serves the **read-only research publication** at
`design/research_publication_gate01/`, with five evidence figures backed by reviewed,
versioned exports. The Streamlit explorer remains available locally via
`python -m streamlit run app/streamlit_app.py`; it is no longer the hosted entrypoint.

Before merging the serving cutover, GitHub pull-request checks verify the frozen
synthetic/public evidence exports, the measured context-budget fixture, frontend
JavaScript, and an actual Docker HTTP smoke. The `main` deployment retains the
existing non-force Git mirror and commit-parity check. A failed hosted release
must be rolled back through a new GitHub revert commit, not a force-push to Space.

This hosting change does **not** establish production readiness, customer-data
validity, monitored service-level objectives, or generated-answer grounding.

"""


class ReadmeMigrationError(ValueError):
    """README structure differs from the reviewed insertion boundary."""


def updated_readme(text: str) -> str:
    if text.count(ANCHOR) != 1:
        raise ReadmeMigrationError("Expected exactly one 'Hosted deployment' heading")
    if SECTION in text:
        return text
    if "## Hosted publication — Gate 07" in text:
        raise ReadmeMigrationError("An existing Gate 07 section has unexpected content")
    return text.replace(ANCHOR, SECTION + ANCHOR, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update or check the hosted-publication README note")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    before = README.read_text(encoding="utf-8")
    after = updated_readme(before)
    if args.check:
        if before != after:
            parser.exit(1, "STALE: run python -m scripts.update_publication_readme --write\n")
        print("PASS: hosted-publication README note is present")
    else:
        if before != after:
            README.write_text(after, encoding="utf-8")
        print("WROTE: README.md" if before != after else "UNCHANGED: README.md")


if __name__ == "__main__":
    main()
