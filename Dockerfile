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

# Install only production deps
RUN poetry install --no-interaction --no-ansi --only main

# Copy app code
COPY . .

# Gunicorn (WSGI)
CMD ["gunicorn", "api_project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--threads", "2"]
