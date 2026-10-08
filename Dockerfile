FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY pyproject.toml ./
COPY app ./app
# The injection classifier is served as an int8 ONNX model, so the image needs ONNX Runtime
# but not PyTorch.
RUN pip install ".[onnx]" \
 && python -m spacy download en_core_web_sm

COPY policies ./policies

# A fixed non-root uid, which most container hosts expect.
RUN useradd --create-home --uid 1000 safegate
USER safegate

# Without Postgres (e.g. the public demo) the decision log falls back to SQLite in the user's home.
ENV SAFEGATE_DATABASE_URL=sqlite+aiosqlite:////home/safegate/safegate.db \
    MALLOC_ARENA_MAX=2 \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/healthz')"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
