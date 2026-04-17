FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
RUN pip install --no-cache-dir ".[postgres]" psycopg2-binary


COPY alembic/ alembic/
COPY alembic.ini .

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head || true; uvicorn genesys.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
