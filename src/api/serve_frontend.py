#!/usr/bin/env python3
# serve_frontend.py
# Serves the frontend HTML on a configurable port
# Usage: python serve_frontend.py --port 3000
# uvicorn src.api.main:app --port 5000 --reload
# python src/api/serve_frontend.py --port 3000 --api-url http://127.0.0.1:5000

import argparse
import http.server
import os
import socketserver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    type=int, default=3000)
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000")
    args = parser.parse_args()

    # inject API URL into HTML before serving
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    with open(html_path, "r") as f:
        html = f.read()

    # write a temp version with the API URL injected
    injected = html.replace(
        "window.API_URL || 'http://127.0.0.1:8000'",
        f"window.API_URL || '{args.api_url}'"
    )

    serve_dir = os.path.join(os.path.dirname(__file__), "frontend")

    # write the injected version temporarily
    tmp_path = os.path.join(serve_dir, "_index.html")
    with open(tmp_path, "w") as f:
        f.write(injected)

    os.chdir(serve_dir)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.path = "/_index.html"
            super().do_GET()

        def log_message(self, format, *args):
            pass  # suppress access logs

    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        print(f"Frontend running at http://127.0.0.1:{args.port}")
        print(f"API URL configured: {args.api_url}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            os.remove(tmp_path)

if __name__ == "__main__":
    main()