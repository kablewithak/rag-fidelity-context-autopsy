"""The verified GitHub→Hugging Face mirror must remain non-destructive."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-huggingface-space.yml"


def test_pull_requests_verify_without_deploying_or_accessing_hf_secrets() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request:" in source
    assert "push:" in source
    assert "workflow_dispatch:" in source
    assert "branches:" in source
    assert "- main" in source
    assert "contents: read" in source
    assert "verify-deployment-contract:" in source
    assert "verify-container-runtime:" in source
    assert "needs: verify-deployment-contract" in source
    assert "needs: [verify-deployment-contract, verify-container-runtime]" in source
    assert "if: github.event_name != 'pull_request' && github.ref == 'refs/heads/main'" in source
    assert 'test "$GITHUB_REF" = "refs/heads/main"' in source
    assert "group: huggingface-space-deploy" in source
    assert "cancel-in-progress: false" in source


def test_container_smoke_and_reviewed_export_gates_run_before_mirror() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert 'python-version: "3.12"' in source
    assert 'node-version: "22"' in source
    assert 'python -m pip install ".[dev]"' in source
    assert 'python -m pip install "tiktoken>=0.8,<1.0"' in source
    assert "tests/test_hugging_face_spaces_package.py" in source
    assert "tests/test_hugging_face_deploy_workflow.py" in source
    assert "tests/test_publication_hosting.py" in source
    assert "python -m scripts.update_publication_readme --check" in source
    assert "python scripts/export_research_publication_gate01.py --check" in source
    assert "python scripts/export_research_publication_figure03.py --check" in source
    assert "python -m scripts.export_research_publication_figure04 --check" in source
    assert "python -m scripts.export_research_publication_figure05 --check" in source
    assert "tests/test_research_publication_figure02.py" in source
    assert "tests/test_research_publication_figure04.py" in source
    assert "tests/test_research_publication_figure05.py" in source
    assert "node --check design/research_publication_gate01/app.js" in source
    assert "docker build --tag rag-fidelity-publication:gate07 ." in source
    assert "python -m scripts.smoke_publication_host --base-url http://127.0.0.1:17860" in source


def test_credential_boundary_and_non_force_commit_parity_remain_unchanged() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "HF_SPACE_DEPLOY_TOKEN: ${{ secrets.HF_SPACE_DEPLOY_TOKEN }}" in source
    assert "KaboKableMolefe/rag-fidelity-context-autopsy" in source
    assert "GIT_ASKPASS=" in source
    assert "GIT_TERMINAL_PROMPT=0" in source
    assert "git push --porcelain hf HEAD:main" in source
    assert "--force" not in source
    assert 'test "$local_commit" = "$space_commit"' in source
