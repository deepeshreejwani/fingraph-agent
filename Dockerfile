FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --locked --no-dev \
    && useradd \
        --create-home \
        --uid 10001 \
        --shell /usr/sbin/nologin \
        fingraph \
    && chown -R fingraph:fingraph /app

ENV PATH="/app/.venv/bin:$PATH"

USER fingraph

EXPOSE 8000

CMD [ "uvicorn", "fingraph.api.app:app", "--host", "0.0.0.0", "--port", "8000" ]