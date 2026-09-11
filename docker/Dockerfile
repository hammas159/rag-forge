# API image. CPU torch here on purpose: the container is for "clone it and it runs",
# while the GPU path is the host venv (make api). Keeps the image ~1 GB instead of ~6 GB.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so a source edit does not invalidate the layer.
COPY pyproject.toml README.md ./
RUN uv pip install --system --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r pyproject.toml

COPY src/ ./src/
COPY ui/ ./ui/
COPY eval/ ./eval/
RUN uv pip install --system --no-deps -e .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "ragforge.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
