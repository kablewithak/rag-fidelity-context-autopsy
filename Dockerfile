# syntax=docker/dockerfile:1
# Gate 07: serve the reviewed static research publication; no inference at request time.
FROM python:3.12-slim

ENV HOME=/home/user \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 user
USER user
WORKDIR $HOME/app

COPY --chown=user app/publication_server.py ./app/publication_server.py
COPY --chown=user design/research_publication_gate01/index.html ./design/research_publication_gate01/index.html
COPY --chown=user design/research_publication_gate01/styles.css ./design/research_publication_gate01/styles.css
COPY --chown=user design/research_publication_gate01/app.js ./design/research_publication_gate01/app.js
COPY --chown=user design/research_publication_gate01/data.js ./design/research_publication_gate01/data.js
COPY --chown=user design/research_publication_gate01/figure03_data.js ./design/research_publication_gate01/figure03_data.js
COPY --chown=user design/research_publication_gate01/figure04_data.js ./design/research_publication_gate01/figure04_data.js
COPY --chown=user design/research_publication_gate01/figure05_data.js ./design/research_publication_gate01/figure05_data.js

EXPOSE 7860
CMD ["sh", "-c", "exec python app/publication_server.py --host 0.0.0.0 --port ${PORT:-7860}"]
