#!/usr/bin/env bash
# Container entrypoint: put everything that must survive a redeploy under /data (the volume),
# then run a single uvicorn worker (positions, kill switch and the tick engine live in memory).
# Volumes are mounted root-owned, so when started as root this script fixes ownership and
# re-executes itself as the unprivileged "algotrader" user (uid 10001).
set -euo pipefail
APP_USER=algotrader
if [ "$(id -u)" = "0" ]; then
    mkdir -p /data/app /data/logs /data/secrets /data/hf
    chown -R "$APP_USER:$APP_USER" /data
    exec setpriv --reuid="$APP_USER" --regid="$APP_USER" --init-groups "$0" "$@"
fi
mkdir -p /data/app /data/logs /data/secrets /data/hf
chmod 700 /data/secrets
rm -rf /app/logs && ln -s /data/logs /app/logs          # sebi audit + trade memory write to logs/
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 \
     --loop uvloop --http httptools --log-level warning --access-log --no-use-colors
