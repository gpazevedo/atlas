# ATLAS Counsel runtime — FastAPI service + MCP server in one image.
FROM python:3.13-slim
WORKDIR /app

# Install uv from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install deps first (layer cache when deps don't change).
COPY uv.lock pyproject.toml README.md ./
RUN uv sync --extra service --frozen --no-dev

# Copy source and a default checkpoint db location.
COPY src ./src
RUN mkdir -p /data
ENV COUNSEL_CHECKPOINT_DB=/data/checkpoints.db

EXPOSE 8000

# Default: HTTP API. Override CMD to run the MCP stdio server instead.
CMD ["uv", "run", "uvicorn", "atlas_counsel.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
