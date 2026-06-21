from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"


def _front_matter(source: str) -> str:
    assert source.startswith("---\n")
    _, front_matter, _ = source.split("---\n", maxsplit=2)
    return front_matter


def test_readme_declares_a_docker_space_for_cpu_demo_hosting() -> None:
    front_matter = _front_matter(README_PATH.read_text(encoding="utf-8"))

    assert "sdk: docker" in front_matter
    assert "app_port: 7860" in front_matter
    assert "suggested_hardware: cpu-basic" in front_matter
    assert 'python_version: "3.12"' in front_matter
    assert "startup_duration_timeout: 30m" in front_matter


def test_dockerfile_runs_only_the_read_only_demo_runtime() -> None:
    source = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in source
    assert "RUN useradd --create-home --uid 1000 user" in source
    assert "USER user" in source
    assert "WORKDIR $HOME/app" in source
    assert 'COPY --chown=user app ./app' in source
    assert 'COPY --chown=user rag_lab ./rag_lab' in source
    assert 'COPY --chown=user artifacts ./artifacts' in source
    assert 'COPY --chown=user data ./data' in source
    assert 'COPY --chown=user docs/reports ./docs/reports' in source
    assert '".[tiktoken,demo]"' in source
    assert "sentence-transformers" not in source
    assert "streamlit run app/streamlit_app.py" in source
    assert "--server.address=0.0.0.0" in source
    assert "--server.port=${PORT:-7860}" in source
    assert "--server.headless=true" in source


def test_dockerignore_excludes_local_outputs_and_development_state_but_keeps_review_reports() -> None:
    ignored = set(DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines())

    assert {".git/", ".venv/", ".pytest_cache/", "__pycache__/", "outputs/", "tests/"} <= ignored
    assert "docs/" not in ignored
    assert {"docs/*", "!docs/reports/", "!docs/reports/**"} <= ignored
