from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rag_lab.public_transfer import PublicTransferError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "materialize_squad_v1_transfer_fixture.py"
SPEC = importlib.util.spec_from_file_location("materialize_squad_fixture", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_download_rejects_non_positive_timeout() -> None:
    with pytest.raises(PublicTransferError, match="timeout_seconds"):
        MODULE.download_source_bytes(
            source_url="https://example.invalid/source.json",
            timeout_seconds=0,
        )


def test_check_mode_validates_existing_fixture(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_directory = tmp_path / "missing"

    exit_code = MODULE.main(["--output-directory", str(output_directory), "--check"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "PUBLIC TRANSFER FIXTURE: FAIL" in captured.err
