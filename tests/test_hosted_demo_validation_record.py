from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
HOSTED_VALIDATION_PATH = PROJECT_ROOT / "docs" / "hosted_demo_validation_v1.md"


def test_hosted_demo_validation_record_matches_deployment_boundary() -> None:
    source = HOSTED_VALIDATION_PATH.read_text(encoding="utf-8")

    assert "# Hosted Demo Validation v1" in source
    assert "KaboKableMolefe/rag-fidelity-context-autopsy" in source
    assert "`7cea06e`" in source
    assert "GitHub `main`" in source
    assert "GitHub Actions workflow `Deploy Hugging Face Space`" in source
    assert "exactly five read-only surfaces" in source
    assert "10-document and 30-case fixture" in source
    assert "not pooled" in source
    assert "customer-data testing" in source
    assert "production readiness" in source


def test_readme_exposes_hosted_status_without_overclaiming() -> None:
    source = README_PATH.read_text(encoding="utf-8")

    assert "Phase 12 — Hosted read-only evidence demonstration" in source
    assert "https://huggingface.co/spaces/KaboKableMolefe/rag-fidelity-context-autopsy" in source
    assert "GitHub `main` is the source of truth" in source
    assert "hosted read-only demonstration" in source
    assert "not customer-data tested" in source
    assert "production ready" in source
