# -----------------------------
# Base image
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Poetry
# -----------------------------
ENV POETRY_VERSION=2.2.1
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app
ENV PYTHONPATH=/app

# -----------------------------
# Copy pyproject.toml and poetry.lock first (to leverage Docker cache)
# -----------------------------
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root --no-interaction --no-ansi

# -----------------------------
# Copy application code
# -----------------------------
COPY . /app

# -----------------------------
# Entrypoint script
# -----------------------------
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set default behavior: can override in docker-compose
ENTRYPOINT ["/app/entrypoint.sh"]
