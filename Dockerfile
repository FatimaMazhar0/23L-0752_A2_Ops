# Slim base, same Python minor version as my venv (3.11)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Non-root user
RUN useradd --create-home --uid 1000 appuser

# 1) Dependencies first: this layer is cached until requirements-docker.txt changes,
#    so editing app code does not trigger a new TensorFlow download.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 2) Source code and the single model the app uses (baked in)
COPY visionguard/ visionguard/
COPY models/champion.keras models/champion.keras

RUN mkdir -p instance && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check: the container is "healthy" only when the app answers over HTTP
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# No secrets here: DATABASE_URL etc. are passed at run time (.env / Compose)
CMD ["waitress-serve", "--listen=0.0.0.0:8000", "--call", "visionguard:create_app"]
