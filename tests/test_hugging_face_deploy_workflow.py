from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "deploy-huggingface-space.yml"


def test_hugging_face_deployment_workflow_is_main_only_and_non_destructive() -> None:
    source = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "name: Deploy Hugging Face Space" in source
    assert "workflow_dispatch:" in source
    assert "branches:" in source
    assert "- main" in source
    assert "group: huggingface-space-deploy" in source
    assert "cancel-in-progress: false" in source
    assert "contents: read" in source

    assert "verify-deployment-contract:" in source
    assert "needs: verify-deployment-contract" in source
    assert 'python-version: "3.12"' in source
    assert 'python -m pip install ".[dev]"' in source
    assert "tests/test_hugging_face_spaces_package.py" in source
    assert "tests/test_hugging_face_deploy_workflow.py" in source

    assert "HF_SPACE_DEPLOY_TOKEN: ${{ secrets.HF_SPACE_DEPLOY_TOKEN }}" in source
    assert "KaboKableMolefe/rag-fidelity-context-autopsy" in source
    assert "GIT_ASKPASS=" in source
    assert "GIT_TERMINAL_PROMPT=0" in source

    assert 'git push --porcelain hf HEAD:main' in source
    assert "--force" not in source
    assert 'test "$local_commit" = "$space_commit"' in source
