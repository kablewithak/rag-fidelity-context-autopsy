"""Gate 07: offline HTTP contract, asset allowlist and README migration."""
from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from app.publication_server import (
    CONTENT_TYPES,
    PUBLICATION_DIRECTORY,
    PublicationAssetError,
    make_handler,
    validate_publication_assets,
)
from scripts.smoke_publication_host import read_response, smoke
from scripts.update_publication_readme import ANCHOR, SECTION, ReadmeMigrationError, updated_readme


def _valid_fixture(tmp_path: Path) -> Path:
    directory = tmp_path / "publication"
    directory.mkdir()
    for filename in CONTENT_TYPES:
        (directory / filename).write_text("fixture\n", encoding="utf-8")
    references = "".join(f'<script src="{x}"></script>' for x in CONTENT_TYPES if x != "index.html")
    anchors = "".join(f'<section id="figure-{x}"></section>' for x in ("one", "two", "three", "four", "five"))
    (directory / "index.html").write_text("Where evidence" + references + anchors, encoding="utf-8")
    (directory / "figure04_data.js").write_text(
        '/* GENERATED from the existing deterministic controlled context-pressure view. */\n'
        'const FIGURE04_PROOF = {"status": "measured"};\n',
        encoding="utf-8",
    )
    return directory


def test_runtime_requires_exact_static_assets_and_all_five_figures(tmp_path: Path) -> None:
    directory = _valid_fixture(tmp_path)
    validate_publication_assets(directory)
    (directory / "figure05_data.js").unlink()
    with pytest.raises(PublicationAssetError, match="Missing or empty publication asset"):
        validate_publication_assets(directory)
    (directory / "figure05_data.js").write_text("fixture", encoding="utf-8")
    (directory / "index.html").write_text("no figures", encoding="utf-8")
    with pytest.raises(PublicationAssetError, match="does not reference required asset"):
        validate_publication_assets(directory)


def test_preview_only_context_proof_fails_closed(tmp_path: Path) -> None:
    directory = _valid_fixture(tmp_path)
    (directory / "figure04_data.js").write_text(
        '/* PREVIEW ONLY */\nconst FIGURE04_PROOF = {"status": "preview"};', encoding="utf-8"
    )
    with pytest.raises(PublicationAssetError, match="generated measured asset"):
        validate_publication_assets(directory)


def test_real_publication_asset_contract() -> None:
    # On the user's repo this requires --write for all reviewed exports first.
    validate_publication_assets(PUBLICATION_DIRECTORY)


def test_http_server_serves_only_approved_assets_and_no_directory_listing(tmp_path: Path) -> None:
    directory = _valid_fixture(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(directory))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        smoke(base)
        status, body, headers = read_response(base, "?case=legal_confidentiality_003")
        assert status == 200 and b"Where evidence" in body
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert headers["Cache-Control"] == "no-store"
        status, _, _ = read_response(base, ".github/workflows/deploy-huggingface-space.yml")
        assert status == 404
        status, _, _ = read_response(base, "artifacts/comparisons/four_pipeline_baseline_v1.json")
        assert status == 404
        status, _, _ = read_response(base, "%2e%2e/README.md")
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=4)


def test_readme_migration_is_idempotent_and_rejects_changed_anchor() -> None:
    source = "---\nsdk: docker\n---\n\n" + ANCHOR + "The old hosted deployment route.\n"
    revised = updated_readme(source)
    assert SECTION in revised
    assert updated_readme(revised) == revised
    assert "sdk: docker" in revised
    with pytest.raises(ReadmeMigrationError, match="exactly one"):
        updated_readme("## No matching heading\n")
