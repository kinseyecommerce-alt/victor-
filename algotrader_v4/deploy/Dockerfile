FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install uvloop httptools

# ── Runtime ────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN useradd -m algotrader
WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

# Don't bake secrets into the image — mount .env at runtime
RUN rm -f .env

USER algotrader
EXPOSE 8000

# Single worker — in-memory state (kill-switch, positions, tick engine)
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-level", "warning", \
     "--access-log"]
