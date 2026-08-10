# Multi-stage production Dockerfile for DriftGuard
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# Final minimal runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/driftuser/.local/bin:${PATH}" \
    PYTHONPATH="/app"

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root user
RUN useradd -m -u 1000 driftuser
USER driftuser

# Copy installed dependencies from builder
COPY --from=builder --chown=driftuser:driftuser /root/.local /home/driftuser/.local

# Copy application source code
COPY --chown=driftuser:driftuser . .

# Ensure artifacts and reports directories exist with appropriate permissions
RUN mkdir -p training/artifacts reports data/processed data/raw

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
