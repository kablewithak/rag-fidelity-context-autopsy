"""Materialize the fixed public SQuAD v1.1 transfer probe.

This command downloads the official SQuAD v1.1 development JSON, derives a
small deterministic corpus/case fixture, records the exact source digest, and
writes only the reviewed subset to data/public_transfer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_lab.public_transfer import (  # noqa: E402
    DEFAULT_CASES_PER_DOCUMENT,
    DEFAULT_DOCUMENT_COUNT,
    DEFAULT_PARAGRAPHS_PER_DOCUMENT,
    SQUAD_V1_SOURCE_URL,
    PublicTransferError,
    build_squad_v1_transfer_fixture_from_bytes,
    load_public_transfer_fixture,
    write_public_transfer_fixture,
)

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "public_transfer" / "squad_v1_dev_v1"
MAX_SOURCE_BYTES = 10_000_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--source-url", default=SQUAD_V1_SOURCE_URL)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--document-count", type=int, default=DEFAULT_DOCUMENT_COUNT)
    parser.add_argument(
        "--paragraphs-per-document",
        type=int,
        default=DEFAULT_PARAGRAPHS_PER_DOCUMENT,
    )
    parser.add_argument("--cases-per-document", type=int, default=DEFAULT_CASES_PER_DOCUMENT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate an already materialized fixture without downloading source data",
    )
    return parser.parse_args(argv)


def download_source_bytes(*, source_url: str, timeout_seconds: float) -> bytes:
    if timeout_seconds <= 0:
        raise PublicTransferError("timeout_seconds must be greater than zero")

    request = Request(
        source_url,
        headers={"User-Agent": "rag-fidelity-context-autopsy-transfer-fixture/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310: fixed HTTPS source
            raw_payload = response.read(MAX_SOURCE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as error:
        raise PublicTransferError(f"unable to download public SQuAD source: {error}") from error

    if len(raw_payload) > MAX_SOURCE_BYTES:
        raise PublicTransferError("downloaded public SQuAD source exceeds the size limit")
    return raw_payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_directory = args.output_directory.resolve()

    try:
        if args.check:
            fixture = load_public_transfer_fixture(output_directory)
            print(
                "PUBLIC TRANSFER FIXTURE CHECK: PASS "
                f"({fixture.manifest.document_count} documents, {len(fixture.cases)} cases)"
            )
            return 0

        raw_payload = download_source_bytes(
            source_url=args.source_url,
            timeout_seconds=args.timeout_seconds,
        )
        fixture = build_squad_v1_transfer_fixture_from_bytes(
            raw_payload,
            source_url=args.source_url,
            document_count=args.document_count,
            paragraphs_per_document=args.paragraphs_per_document,
            cases_per_document=args.cases_per_document,
        )
        write_public_transfer_fixture(
            fixture,
            output_directory,
            overwrite=args.overwrite,
        )
        print(
            "PUBLIC TRANSFER FIXTURE MATERIALIZED: "
            f"{output_directory} "
            f"({fixture.manifest.document_count} documents, {len(fixture.cases)} cases, "
            f"source_sha256={fixture.manifest.source_sha256})"
        )
        return 0
    except PublicTransferError as error:
        print(f"PUBLIC TRANSFER FIXTURE: FAIL ({error})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
