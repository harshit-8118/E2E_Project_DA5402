#!/usr/bin/env python3
# serve_frontend.py
# Serves the frontend HTML on a configurable port
# uvicorn src.api.main:app --port 8000
# python src/api/serve_frontend.py --port 7500 --api-url http://127.0.0.1:8000

"""Serve the frontend with a runtime-configured API URL."""

import argparse
import http.server
import os
import socketserver
from pathlib import Path

import yaml
from dotenv import load_dotenv


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
CONFIG_PATH = os.path.join(FRONTEND_DIR, "config.js")
PARAMS_PATH = Path(__file__).resolve().parents[2] / "params.yaml"


def load_frontend_defaults() -> dict:
    if not PARAMS_PATH.exists():
        return {}
    with open(PARAMS_PATH, "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    return params.get("frontend", {})


def write_runtime_config(api_url: str, user_manual_url: str) -> None:
    config = (
        "window.DERMAI_CONFIG = window.DERMAI_CONFIG || {};\n"
        f"window.DERMAI_CONFIG.API_URL = '{api_url.rstrip('/')}';\n"
        f"window.DERMAI_CONFIG.USER_MANUAL_URL = '{user_manual_url}';\n"
    )
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        handle.write(config)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    frontend_defaults = load_frontend_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("PUBLIC_API_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--user-manual-url",
        type=str,
        default=os.getenv("USER_MANUAL_URL", frontend_defaults.get("user_manual_url", "")),
    )
    args = parser.parse_args()

    write_runtime_config(args.api_url, args.user_manual_url)
    os.chdir(FRONTEND_DIR)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.path = "/auth.html"
            super().do_GET()

        def log_message(self, format, *args):
            return

    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        print(f"Frontend running at http://127.0.0.1:{args.port}")
        print(f"API URL configured: {args.api_url}")
        print(f"User manual URL configured: {args.user_manual_url}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
