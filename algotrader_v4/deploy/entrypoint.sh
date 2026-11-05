#!/usr/bin/env bash
# Container entrypoint: put everything that must survive a redeploy under /data (the volume),
# then run a single uvicorn worker (positions, kill switch and the tick engine live in memory).
set -euo pipefail
mkdir -p /data/app /data/logs /data/secrets /data/hf
chmod 700 /data/secrets
rm -rf /app/logs && ln -s /data/logs /app/logs          # sebi audit + trade memory write to logs/
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 \
     --loop uvloop --http httptools --log-level warning --access-log --no-use-colors
