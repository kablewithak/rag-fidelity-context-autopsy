"""Hugging Face Docker publication packaging, without importing the RAG runtime."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"
ASSETS = (
    "index.html", "styles.css", "app.js", "data.js", "figure03_data.js",
    "figure04_data.js", "figure05_data.js",
)


def _front_matter(source: str) -> str:
    assert source.startswith("---\n")
    _, front_matter, _ = source.split("---\n", maxsplit=2)
    return front_matter


def test_readme_preserves_hugging_face_docker_space_metadata() -> None:
    source = README_PATH.read_text(encoding="utf-8")
    front_matter = _front_matter(source)
    assert "sdk: docker" in front_matter
    assert "app_port: 7860" in front_matter
    assert "suggested_hardware: cpu-basic" in front_matter
    assert 'python_version: "3.12"' in front_matter
    assert "startup_duration_timeout: 30m" in front_matter
    assert "## Hosted publication — Gate 07" in source
    assert "Streamlit explorer remains available locally" in source


def test_dockerfile_serves_only_allowlisted_static_publication() -> None:
    source = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in source
    assert "RUN useradd --create-home --uid 1000 user" in source
    assert "USER user" in source
    assert "WORKDIR $HOME/app" in source
    assert "EXPOSE 7860" in source
    assert "COPY --chown=user app/publication_server.py ./app/publication_server.py" in source
    for filename in ASSETS:
        assert (
            f"COPY --chown=user design/research_publication_gate01/{filename} "
            f"./design/research_publication_gate01/{filename}"
        ) in source
    assert "exec python app/publication_server.py --host 0.0.0.0" in source
    assert "${PORT:-7860}" in source
    for forbidden in (
        'COPY --chown=user rag_lab',
        'COPY --chown=user artifacts',
        'COPY --chown=user data ./data',
        'COPY --chown=user docs/reports',
        'streamlit run app/streamlit_app.py',
        'pip install',
        'sentence-transformers',
    ):
        assert forbidden not in source


def test_dockerignore_keeps_runtime_context_free_of_local_outputs() -> None:
    ignored = set(DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines())
    assert {".git/", ".venv/", ".pytest_cache/", "__pycache__/", "outputs/", "tests/"} <= ignored
    assert "docs/" not in ignored
    assert {"docs/*", "!docs/reports/", "!docs/reports/**"} <= ignored
    assert "design/" not in ignored
