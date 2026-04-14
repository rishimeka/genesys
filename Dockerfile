FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir ".[postgres]"

COPY alembic/ alembic/
COPY alembic.ini .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn genesys.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
