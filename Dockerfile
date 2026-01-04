FROM python:3.14-slim

# -----------------------------
# System deps
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Poetry
# -----------------------------
ENV POETRY_VERSION=2.2.1
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

# -----------------------------
# App setup
# -----------------------------
WORKDIR /app
ENV PYTHONPATH=/app
ENV POETRY_VIRTUALENVS_CREATE=false


# -----------------------------
# App code
# -----------------------------
COPY . .

# -----------------------------
# Install deps (cached layer)
# -----------------------------
COPY pyproject.toml poetry.lock* README.md ./
RUN poetry install --no-interaction --no-ansi --with dev

CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "manage.py", "runserver", "0.0.0.0:8000", "--noreload"]