"""Local web interface for inspecting the Pando connectivity engine."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
ALGORITHM_DIR = ROOT / "algorithm"
WEB_DIR = ROOT / "web"
MAX_REQUEST_BYTES = 64 * 1024
MODEL_NAME = "word2vec-google-news-300"
DEFAULT_MAX_HOPS = 8

sys.path.insert(0, str(ALGORITHM_DIR))

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


class RequestError(ValueError):
    """An invalid API request that should be returned as HTTP 400."""


def _load_seed_puzzles() -> list[dict[str, str]]:
    source = ALGORITHM_DIR / "seed_words.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    return data["endpoint_pairs"]


SEED_PUZZLES = _load_seed_puzzles()
DEFAULT_THRESHOLD = json.loads(
    (ALGORITHM_DIR / "calibration.json").read_text(encoding="utf-8")
)["threshold"]


def get_engine():
    """Load and prepare the engine once, leaving static page startup fast."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    with _ENGINE_LOCK:
        if _ENGINE is None:
            engine = importlib.import_module("connectivity")
            engine.get_pool()
            _ENGINE = engine
    return _ENGINE


def _require_object(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RequestError("The request body must be a JSON object.")
    return payload


def _require_word(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"'{key}' must be a non-empty string.")
    if len(value.strip()) > 100:
        raise RequestError(f"'{key}' must be 100 characters or fewer.")
    return value.strip()


def _require_words(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("words")
    if not isinstance(values, list):
        raise RequestError("'words' must be an array of connector words.")

    words = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise RequestError(f"Connector word {index + 1} must not be blank.")
        words.append(value.strip())
    return words


def _max_hops(payload: Mapping[str, Any]) -> int:
    value = payload.get("maxHops", DEFAULT_MAX_HOPS)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= DEFAULT_MAX_HOPS
    ):
        raise RequestError("'maxHops' must be an integer from 0 through 8.")
    return value


def serialize_link(link: Any) -> dict[str, Any]:
    """Convert an engine Link into the stable browser-facing shape."""
    cosine = None if link.cosine is None else round(float(link.cosine), 6)
    return {
        "left": link.left,
        "right": link.right,
        "cosine": cosine,
        "verified": bool(link.verified),
        "reason": link.reason,
    }


def verify_request(payload: Any, engine: Any = None) -> dict[str, Any]:
    request = _require_object(payload)
    start = _require_word(request, "start")
    end = _require_word(request, "end")
    words = _require_words(request)
    engine = get_engine() if engine is None else engine

    result = engine.verify_sequence(start, end, words)
    return {
        "start": result.start,
        "end": result.end,
        "words": list(result.words),
        "verified": bool(result.verified),
        "length": int(result.length),
        "links": [serialize_link(link) for link in result.links],
    }


def benchmark_request(payload: Any, engine: Any = None) -> dict[str, Any]:
    request = _require_object(payload)
    start = _require_word(request, "start")
    end = _require_word(request, "end")
    max_hops = _max_hops(request)
    engine = get_engine() if engine is None else engine

    route = engine.shortest_route(start, end, max_hops=max_hops)
    if route is None:
        return {
            "start": start,
            "end": end,
            "maxHops": max_hops,
            "found": False,
            "route": None,
        }

    return {
        "start": route.start,
        "end": route.end,
        "maxHops": max_hops,
        "found": True,
        "route": {
            "words": list(route.words),
            "length": int(route.length),
            "totalCosine": round(float(route.total_cosine), 6),
            "links": [serialize_link(link) for link in route.links],
        },
    }


def puzzles_response() -> dict[str, Any]:
    return {
        "puzzles": SEED_PUZZLES,
        "engine": {
            "model": MODEL_NAME,
            "threshold": DEFAULT_THRESHOLD,
            "maxHops": DEFAULT_MAX_HOPS,
        },
    }


class PandoHandler(BaseHTTPRequestHandler):
    static_files = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    }

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/puzzles":
            self._send_json(HTTPStatus.OK, puzzles_response())
            return
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if path not in self.static_files:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": {"code": "not-found", "message": "Not found."}},
            )
            return

        filename, content_type = self.static_files[path]
        body = (WEB_DIR / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        operations = {
            "/api/verify": verify_request,
            "/api/benchmark": benchmark_request,
        }
        operation = operations.get(path)
        if operation is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": {"code": "not-found", "message": "Not found."}},
            )
            return

        try:
            payload = self._read_json()
            response = operation(payload)
        except RequestError as error:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid-request", str(error))
            return
        except RuntimeError as error:
            self._send_error(
                HTTPStatus.SERVICE_UNAVAILABLE, "engine-unavailable", str(error)
            )
            return
        except Exception:
            self.log_error("Unhandled engine error")
            traceback.print_exc()
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "engine-error",
                "The engine could not complete this request.",
            )
            return

        self._send_json(HTTPStatus.OK, response)

    def _read_json(self) -> Any:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise RequestError("A JSON request body is required.")
        try:
            length = int(content_length)
        except ValueError as error:
            raise RequestError("Invalid Content-Length header.") from error
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise RequestError("The request body must be between 1 byte and 64 KB.")
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RequestError("The request body must contain valid JSON.") from error

    def _send_error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def make_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), PandoHandler)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local Pando inspector.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args(argv)

    server = make_server(args.host, args.port)
    print(f"Pando inspector running at http://{args.host}:{server.server_port}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Pando inspector.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
