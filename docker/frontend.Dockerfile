# docker/frontend.Dockerfile
# Lightweight Python HTTP server serving static frontend files
# serve_frontend.py writes config.js with runtime API_URL then serves on --port

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# only stdlib needed — no pip installs required
COPY src/api/serve_frontend.py ./src/api/serve_frontend.py
COPY src/api/frontend/         ./src/api/frontend/

EXPOSE 7500

CMD ["python", "src/api/serve_frontend.py", "--port", "7500", "--api-url", "http://localhost:8000"]
