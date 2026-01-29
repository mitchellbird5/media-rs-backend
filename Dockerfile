# =====================================================
# Base image (shared)
# =====================================================
FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="/root/.local/bin:$PATH"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
ENV PYTHONPATH=/app

# Copy dependency files first (better cache)
COPY pyproject.toml poetry.lock* README.md ./

# =====================================================
# Dev stage
# =====================================================
FROM base AS dev

# Install all deps (including dev)
RUN poetry install --no-interaction --no-ansi --with dev

# Copy app code
COPY . .

# Dev server with debugger
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "manage.py", "runserver", "0.0.0.0:8000"]

# =====================================================
# Prod stage
# =====================================================
FROM base AS prod

# Copy dependency files first
COPY pyproject.toml poetry.lock* README.md requirements.txt ./

# Copy app code (so Poetry can find the package)
COPY media_rs ./media_rs

# Install only production deps
RUN poetry install --no-interaction --no-ansi --only main

COPY api ./api

# Gunicorn WSGI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]