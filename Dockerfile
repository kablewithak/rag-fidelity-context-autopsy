# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 user

USER user
WORKDIR $HOME/app

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user rag_lab ./rag_lab
COPY --chown=user app ./app
COPY --chown=user artifacts ./artifacts
COPY --chown=user data ./data
COPY --chown=user docs/reports ./docs/reports

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir ".[tiktoken,demo]"

EXPOSE 7860

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-7860} --server.headless=true --browser.gatherUsageStats=false"]
