---
name: backtest
description: >
  Run AlgoTrader Pro backtests through its API and explain the results: one
  symbol across all strategies (ranked by Sharpe), a batch of symbols for one
  strategy against the live-trading gate, or the currently approved list. Shows
  win rate, Sharpe, drawdown, profit factor and walk-forward out-of-sample
  figures, and why a symbol failed the gate. Use when the user asks to
  "backtest RELIANCE", "which strategy suits TCS", "re-qualify the watchlist",
  or "/backtest intraday RELIANCE,TCS".
---

# Backtest — AlgoTrader Pro

Arguments (all optional): a strategy (`intraday`, `scalping`, `swing`, `fno`) and
a comma-separated symbol list. Backtests use yfinance history, so they need
internet access from the server.

## Setup

```bash
ALGO_URL="${ALGO_URL:-http://127.0.0.1:8000}"
# ALGO_API_KEY must be set. Never echo, log or paste it.
H="X-API-Key: ${ALGO_API_KEY}"
j()    { curl -sS -m 30 -H "$H" "$ALGO_URL$1"; }
post() { curl -sS -m 600 -H "$H" -H "Content-Type: application/json" -X POST "$ALGO_URL$1" -d "$2"; }
```

## Pick the mode

| Request | Call |
|---------|------|
| One symbol, "which strategy?" | `post /backtest/compare '{"symbol":"RELIANCE","exchange":"NSE"}'` |
| One symbol + strategy | `post /backtest/run '{"symbol":"RELIANCE","strategy":"intraday"}'` |
| Many symbols, one strategy | `post /backtest/batch '{"strategy":"intraday","symbols":[{"symbol":"RELIANCE","exchange":"NSE"},{"symbol":"TCS","exchange":"NSE"}]}'` |
| What is approved now | `j /backtest/approved/intraday` |
| Trades / equity curve (after a `/backtest/run` for that pair, else 404) | `curl -sS -H "$H" -o trades.csv "$ALGO_URL/backtest/trades/RELIANCE/intraday"` (CSV) · same with `/backtest/equity/...` `-o equity.png` (PNG chart) |
| Full weekly re-qualification | `post /backtest/weekly '{}'` — slow; ask before running |
| **Replay through the live agent code** (preferred) | `post /backtest/replay '{"agent":"intraday","symbols":["RELIANCE","TCS"],"days":30}'` — real minute bars → the agent's own evaluate_tick, risk sizing, trailing stop, 5 bps slippage, Zerodha costs; `source` says kite / yfinance / synthetic; `benchmark` compares with NIFTY. Add `"params":{"MIN_SCORE":5}` to try a parameter. |
| Momentum rotation vs NIFTY | `j "/backtest/momentum?months=24"` — monthly 12-1 rotation of Nifty 100, equal weight, 200-DMA filter; report `strategy_return_pct` vs `nifty_return_pct`, CAGR, drawdown, `beats_index` |
| Live scoreboard (not a backtest) | `j /performance/scoreboard` — the account vs NIFTY after costs; `j /performance/strategies` — which strategies are pruned |

Every result also carries `oos_return_pct`, `oos_hold_return_pct` (buy-and-hold of the symbol)
and `oos_nifty_return_pct` over the same out-of-sample windows, and `beats_index`. A symbol
that does not beat NIFTY out of sample fails the gate (`BT_REQUIRE_BEATS_INDEX`).

`walk_forward` defaults to `true` (in-sample fit + out-of-sample check). Keep it on
unless the user asks otherwise. `lookback_days` defaults to `BT_LOOKBACK_DAYS` (180).

Batch runs can take minutes; run them in the background and report when done.

## Read the result

Each result has: `passed`, `total_trades`, `win_rate`, `sharpe_ratio`,
`max_drawdown_pct`, `profit_factor`, `total_pnl`, `oos_win_rate`, `oos_trades`,
`oos_total_pnl`, `fail_reasons`.

The gate these endpoints apply (from `.env`, defaults) is the same for every
strategy: win rate ≥ `BT_MIN_WIN_RATE` (55%), Sharpe ≥ `BT_MIN_SHARPE` (1.0),
drawdown ≤ `BT_MAX_DRAWDOWN_PCT` (15%), trades ≥ `BT_MIN_TRADES` (20).
(`strategy_signals.GATE_THRESHOLDS` holds per-strategy gates, but only the
separate `auto_backtest_runner.py` uses those.)

## Report

A table per request: symbol, strategy, PASS/FAIL, trades, win %, Sharpe, max DD %,
PF, net P&L, OOS win % (OOS trades). Then:
- For each FAIL, its `fail_reasons` in plain words.
- **Overfitting check:** flag any symbol whose OOS win rate is more than 10 points
  below its in-sample win rate, or with fewer than 5 OOS trades. Those results
  are not trustworthy.
- For `compare`, name the best strategy and say if none pass.

**Caveat to state once:** these endpoints run `backtest_engine._generate_signals`,
one simplified rule per strategy (e.g. "intraday" = VWAP cross with EMA9 > EMA21
and RSI 45-65). They do **not** replay the live agents' multi-pattern scoring
(10 intraday patterns, 8 F&O patterns, context bonuses, Claude gate), and fills
ignore slippage and brokerage. A pass says the symbol suits the strategy's core
idea; it is not a backtest of what the live agent will do. Confirm with a
real-price paper session (`/paper-session`).
