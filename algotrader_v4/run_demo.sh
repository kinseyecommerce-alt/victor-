#!/usr/bin/env bash
# run_demo.sh — start AlgoTrader Pro in DEMO mode and open the animated dashboard.
#
# Demo = paper trading on the built-in price simulator, with a clock that starts at
# 10:00 IST so the "market" is open whenever you test. No Zerodha or Claude keys needed,
# no real orders. Ctrl+C stops everything.
#
#   ./run_demo.sh            # backend :8000 + dashboard :5000, bot auto-started
#   NO_AUTOSTART=1 ./run_demo.sh   # start the bot yourself from the dashboard
set -euo pipefail
cd "$(dirname "$0")"

export TRADING_MODE=PAPER
export PAPER_DATA_SOURCE=SIM
export DEMO_MODE=true
export SKIP_STARTUP_BACKTEST=true     # demo prices are synthetic — don't backtest on them
export USE_MULTI_TIMEFRAME=false      # let the demo trade within minutes (keep it ON for real use)
export USE_CLAUDE_TRADE_GATE="${USE_CLAUDE_TRADE_GATE:-false}"

WATCHLIST='[{"symbol":"RELIANCE","exchange":"NSE"},{"symbol":"TCS","exchange":"NSE"},
{"symbol":"INFY","exchange":"NSE"},{"symbol":"HDFCBANK","exchange":"NSE"},
{"symbol":"ICICIBANK","exchange":"NSE"},{"symbol":"SBIN","exchange":"NSE"},
{"symbol":"ITC","exchange":"NSE"},{"symbol":"LT","exchange":"NSE"},
{"symbol":"AXISBANK","exchange":"NSE"},{"symbol":"MARUTI","exchange":"NSE"}]'

pids=()
cleanup() { echo; echo "Stopping demo…"; kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "▶ Backend  http://localhost:8000  (DEMO, paper simulator)"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level warning &
pids+=($!)

if [ ! -d frontend/node_modules ]; then
  echo "▶ Installing dashboard dependencies (first run only)…"
  (cd frontend && npm install --no-audit --no-fund)
fi
echo "▶ Dashboard http://localhost:5000"
(cd frontend && npx vite --port 5000 --strictPort) &
pids+=($!)

for _ in $(seq 1 60); do
  curl -sf http://localhost:8000/health >/dev/null && break
  sleep 1
done
curl -sf http://localhost:8000/health >/dev/null || { echo "Backend did not start"; exit 1; }

if [ -z "${NO_AUTOSTART:-}" ]; then
  KEY_HEADER=()
  [ -n "${API_KEY:-}" ] && KEY_HEADER=(-H "X-API-Key: ${API_KEY}")
  curl -sf -X POST http://localhost:8000/bot/start "${KEY_HEADER[@]}" \
       -H 'Content-Type: application/json' \
       -d "{\"strategies\":[\"intraday\",\"scalping\"],\"watchlist\":${WATCHLIST}}" >/dev/null \
    && echo "▶ Bot started (intraday + scalping on 10 symbols) — first trades usually within a minute"
fi

echo
echo "Open http://localhost:5000 in your browser.   Ctrl+C to stop."
wait
