"""Dependency-free, bounded HTTP smoke for the published research page.

Usage: python -m scripts.smoke_publication_host --base-url http://127.0.0.1:17860
"""

from __future__ import annotations

import argparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ASSETS = (
    "styles.css", "data.js", "figure03_data.js", "figure04_data.js", "figure05_data.js", "app.js",
)


def read_response(base_url: str, path: str, method: str = "GET") -> tuple[int, bytes, dict[str, str]]:
    request = Request(f"{base_url.rstrip('/')}/{path.lstrip('/')}", method=method)
    try:
        with urlopen(request, timeout=4) as response:
            return response.status, response.read(), dict(response.headers.items())
    except HTTPError as error:
        return error.code, error.read(), dict(error.headers.items())


def smoke(base_url: str) -> None:
    status, body, _ = read_response(base_url, "healthz")
    assert status == 200 and body == b"ok\n", "Health endpoint failed"
    status, html, headers = read_response(base_url, "")
    assert status == 200 and b"Where evidence" in html, "Article not served"
    assert headers.get("X-Content-Type-Options") == "nosniff", "Missing nosniff header"
    for anchor in (b'id="figure-one"', b'id="figure-two"', b'id="figure-three"', b'id="figure-four"', b'id="figure-five"'):
        assert anchor in html, f"Missing figure: {anchor.decode('ascii')}"
    for filename in ASSETS:
        status, content, headers = read_response(base_url, filename)
        assert status == 200 and content, f"Asset missing: {filename}"
        assert headers.get("Cache-Control") == "no-store", f"Asset cache policy missing: {filename}"
        head_status, head_body, _ = read_response(base_url, filename, method="HEAD")
        assert head_status == 200 and not head_body, f"HEAD failed: {filename}"
        if filename == "figure04_data.js":
            assert b'"status": "measured"' in content and b"PREVIEW ONLY" not in content, "Preview figure leaked"
    status, _, _ = read_response(base_url, "README.md")
    assert status == 404, "Unexpected file exposure"
    status, _, _ = read_response(base_url, "design/research_publication_gate01/")
    assert status == 404, "Directory listing must be disabled"
    print("PUBLICATION_HOST_SMOKE: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a running publication HTTP server")
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    try:
        smoke(args.base_url)
    except (AssertionError, URLError, TimeoutError) as error:
        parser.exit(1, f"PUBLICATION_HOST_SMOKE: FAIL: {error}\n")


if __name__ == "__main__":
    main()
