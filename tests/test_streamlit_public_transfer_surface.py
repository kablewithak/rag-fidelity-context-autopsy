from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"


def test_executive_surface_exposes_reviewed_public_transfer_without_pooling() -> None:
    source = STREAMLIT_APP.read_text(encoding="utf-8")

    assert "load_public_transfer_review_view" in source
    assert "_render_public_transfer_review" in source
    assert "side by side" in source.lower()
    assert "must not be pooled" in source.lower()
    assert "public-corpus transfer probe" in source.lower()
