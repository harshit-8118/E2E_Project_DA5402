# docker/backend.Dockerfile
# CPU-only FastAPI backend for DermAI
# uv for fast pip installs with build cache

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system libs for opencv + pillow + healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── layer 1: requirements (cached unless docker-requirements.txt changes) ──────
COPY docker-requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-strategy unsafe-best-match -r docker-requirements.txt

# install grad-cam separately (not in requirements)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "grad-cam==1.5.4"

# ── layer 2: package setup (cached unless setup.py/params.yaml changes) ────────
COPY setup.py params.yaml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e . --no-deps

# ── layer 3: source code (changes most often — last for cache efficiency) ───────
COPY src/ ./src/
COPY params.yaml .
COPY setup.py .

# create runtime directories
RUN mkdir -p /app/outputs/models /app/logs

EXPOSE 8000

# healthcheck uses curl installed above
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
