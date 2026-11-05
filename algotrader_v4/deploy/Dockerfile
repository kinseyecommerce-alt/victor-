# AlgoTrader Pro — one image: Python backend + built React dashboard (+ Kronos deps).
# Build context = algotrader_v4/ (Railway: root directory "algotrader_v4", this Dockerfile).
# Persistent state (vault, ledger, swing positions, logs, model cache) lives under /data —
# mount a volume there. Secrets come from environment variables, never from the image.

# ── 1. dashboard ──────────────────────────────────────────────────────────────
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ── 2. python deps ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS py
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-kronos.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && pip install --no-cache-dir --prefix=/install uvloop httptools
ARG KRONOS=1
RUN if [ "$KRONOS" = "1" ]; then \
      pip install --no-cache-dir --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir --prefix=/install -r requirements-kronos.txt; fi

# ── 3. runtime ────────────────────────────────────────────────────────────────
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 algotrader
WORKDIR /app
COPY --from=py /install /usr/local
COPY . .
COPY --from=web /web/dist ./frontend/dist
RUN rm -rf .env .secrets data/*.json frontend/node_modules && chmod +x deploy/entrypoint.sh \
    && mkdir -p /data && chown -R algotrader:algotrader /app /data
# The entrypoint starts as root only to take ownership of the mounted volume (Railway/Docker
# mount volumes root-owned), then drops to the unprivileged user before running the app.
ENV PYTHONUNBUFFERED=1 \
    SECRETS_DIR=/data/secrets \
    PERFORMANCE_FILE=/data/app/performance.json \
    AI_SWING_FILE=/data/app/ai_swing_positions.json \
    STRATEGY_PARAMS_FILE=/data/app/strategy_params.json \
    RESEARCH_LOG_FILE=/data/app/research_log.json \
    HF_HOME=/data/hf \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD curl -sf http://localhost:${PORT}/health || exit 1
ENTRYPOINT ["deploy/entrypoint.sh"]
