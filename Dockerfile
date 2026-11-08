FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8000

WORKDIR /app

# onnxruntime（Chroma 依赖）需要 libgomp。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

COPY --from=uv /uv /uvx /bin/

# 先安装依赖，利用 Docker layer cache。
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY data/docs ./data/docs
COPY langgraph.json ./
RUN uv sync --frozen --no-dev

RUN mkdir -p /app/data/chroma /app/data/reports \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["python", "-m", "app.web.server"]
