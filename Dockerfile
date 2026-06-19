# ATLAS Counsel runtime — FastAPI + MCP server in one image.
FROM python:3.13-slim
WORKDIR /app

# Install uv from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install deps first (layer cache when deps don't change).
COPY uv.lock pyproject.toml README.md ./
RUN uv sync --extra service --extra qdrant --extra otel --frozen --no-dev

# Copy source.
COPY src ./src
RUN mkdir -p /data
ENV CHECKPOINT_DIR=/data

EXPOSE 8000

# HTTP + MCP on the same port (MCP mounted at /mcp).
CMD ["uv", "run", "uvicorn", "atlas_counsel.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
