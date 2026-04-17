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


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
CONFIG_PATH = os.path.join(FRONTEND_DIR, "config.js")


def write_runtime_config(api_url: str) -> None:
    config = (
        "window.DERMAI_CONFIG = window.DERMAI_CONFIG || {};\n"
        f"window.DERMAI_CONFIG.API_URL = '{api_url.rstrip('/')}';\n"
    )
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        handle.write(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000")
    args = parser.parse_args()

    write_runtime_config(args.api_url)
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
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()