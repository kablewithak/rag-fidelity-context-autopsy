"""Minimal allowlisted HTTP server for the public, read-only research publication.

The evaluation engine and its raw artifacts are never served by this process.
Only explicitly approved, generated browser assets are exposed. This is a
small demonstration server behind Hugging Face's HTTPS gateway, not a general
production HTTP stack.
"""

from __future__ import annotations

import argparse
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIRECTORY = ROOT / "design" / "research_publication_gate01"
CONTENT_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "data.js": "text/javascript; charset=utf-8",
    "figure03_data.js": "text/javascript; charset=utf-8",
    "figure04_data.js": "text/javascript; charset=utf-8",
    "figure05_data.js": "text/javascript; charset=utf-8",
}


class PublicationAssetError(ValueError):
    """Static publication assets are incomplete or include preview-only data."""


def validate_publication_assets(directory: Path) -> None:
    """Refuse to serve incomplete assets or the unpublished Figure 04 preview."""

    for filename in CONTENT_TYPES:
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise PublicationAssetError(f"Missing or empty publication asset: {filename}")

    figure04 = (directory / "figure04_data.js").read_text(encoding="utf-8")
    if not figure04.startswith("/* GENERATED from the existing deterministic"):
        raise PublicationAssetError("Figure 04 must be the generated measured asset")
    if '"status": "measured"' not in figure04:
        raise PublicationAssetError("Figure 04 must have measured status")

    html = (directory / "index.html").read_text(encoding="utf-8")
    for filename in ("styles.css", "data.js", "figure03_data.js", "figure04_data.js", "figure05_data.js", "app.js"):
        if f'"{filename}"' not in html:
            raise PublicationAssetError(f"HTML does not reference required asset: {filename}")
    for anchor in ('id="figure-one"', 'id="figure-two"', 'id="figure-three"', 'id="figure-four"', 'id="figure-five"'):
        if anchor not in html:
            raise PublicationAssetError(f"Missing research figure: {anchor}")


def make_handler(directory: Path) -> type[BaseHTTPRequestHandler]:
    """Create a handler with a closed set of paths and no filesystem listing."""

    class PublicationHandler(BaseHTTPRequestHandler):
        server_version = "RAGFidelity"
        sys_version = ""

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            self._respond(include_body=True)

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            self._respond(include_body=False)

        def _respond(self, *, include_body: bool) -> None:
            pathname = urlsplit(self.path).path
            if pathname == "/healthz":
                self._send_bytes(b"ok\n", "text/plain; charset=utf-8", include_body)
                return
            filename = "index.html" if pathname == "/" else pathname.removeprefix("/")
            if pathname not in ("/", *tuple(f"/{x}" for x in CONTENT_TYPES)):
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                body = (directory / filename).read_bytes()
            except OSError:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Asset unavailable")
                return
            self._send_bytes(body, CONTENT_TYPES[filename], include_body)

        def _send_bytes(self, body: bytes, content_type: str, include_body: bool) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            """Do not log URL paths or user-supplied query strings."""

    return PublicationHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve only the reviewed RAG research publication")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument("--directory", type=Path, default=PUBLICATION_DIRECTORY)
    args = parser.parse_args()
    if not (1 <= args.port <= 65535):
        parser.error("port must be 1..65535")
    directory = args.directory.resolve()
    validate_publication_assets(directory)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(directory))
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
