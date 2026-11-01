#!/usr/bin/env bash
# Update a running AlgoTrader Pro server: pull, install, build, TEST, then restart.
# The service is restarted only if the test suite passes, so a bad push can't take the
# trading server down. Run as root:  sudo bash /opt/algotrader/deploy/update.sh [branch]
set -euo pipefail
APP_USER="algotrader"
REPO_DIR="/opt/algotrader-repo"
APP_DIR="/opt/algotrader"
VENV="$APP_DIR/venv"
BRANCH="${1:-$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)}"

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)."; exit 1; }

echo "▶ Pulling $BRANCH"
sudo -u "$APP_USER" git -C "$REPO_DIR" fetch --quiet origin
sudo -u "$APP_USER" git -C "$REPO_DIR" checkout --quiet "$BRANCH"
sudo -u "$APP_USER" git -C "$REPO_DIR" pull --quiet --ff-only origin "$BRANCH"
echo "▶ Now at $(git -C "$REPO_DIR" log --oneline -1)"

echo "▶ Dependencies"
sudo -u "$APP_USER" "$VENV/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
sudo -u "$APP_USER" bash -c "cd '$APP_DIR/frontend' && npm ci --silent --no-audit --no-fund && npm run build --silent"

echo "▶ Tests (the service keeps running the old code meanwhile)"
TMP="$(mktemp -d)"; chown "$APP_USER" "$TMP"
if ! sudo -u "$APP_USER" env PERFORMANCE_FILE="$TMP/perf.json" AI_SWING_FILE="$TMP/swing.json" SECRETS_DIR="$TMP/sec" \
        TRADING_MODE=PAPER PAPER_DATA_SOURCE=SIM DEMO_MODE=false \
        "$VENV/bin/python" "$APP_DIR/test_pipeline.py" > "$TMP/tests.log" 2>&1; then
    grep -E "RESULTS|❌" "$TMP/tests.log" | tail -20
    echo "✗ Tests failed — NOT restarting. Old version still running. Full log: $TMP/tests.log"
    exit 1
fi
grep RESULTS "$TMP/tests.log"; rm -rf "$TMP"

echo "▶ Restarting"
systemctl restart algotrader
for _ in $(seq 1 30); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:8000/health >/dev/null && echo "▶ Running: $(curl -s http://127.0.0.1:8000/health | head -c 120)" \
    || { echo "✗ app did not come back — tail /var/log/algotrader/app.log"; exit 1; }
