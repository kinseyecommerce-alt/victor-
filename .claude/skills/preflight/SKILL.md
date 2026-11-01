---
name: preflight
description: >
  Read-only GO / NO-GO check of AlgoTrader Pro before a trading session — and
  mandatory before switching TRADING_MODE to LIVE. Verifies server health, Kite
  session and data feed, SEBI kill-switch state, risk limits, capital
  allocation, enabled agents, backtest approvals and the test suite, then
  reports a table with a single verdict. Changes nothing. Use when the user
  asks "are we ready", "preflight", "can I go live", or before /paper-session.
---

# Preflight — AlgoTrader Pro GO / NO-GO

You are checking whether AlgoTrader Pro is safe to trade **today**. This skill is
**read-only**: never start the bot, place or cancel orders, change settings,
reset the kill switch or edit `.env`. If something fails, report it and say how
to fix it — the user decides.

## Setup

```bash
ALGO_URL="${ALGO_URL:-http://127.0.0.1:8000}"
# ALGO_API_KEY must be set in the environment. Never echo, log or paste it.
H="X-API-Key: ${ALGO_API_KEY}"
j() { curl -sS -m 10 -H "$H" "$ALGO_URL$1"; }
```

If `ALGO_API_KEY` is unset, say so and run only the public checks (1, 2).

## Checks — run all, collect results, then report

| # | Check | Command | PASS when |
|---|-------|---------|-----------|
| 1 | Server up | `j /health` | `"status":"ok"`, `tick_engine` = `running` |
| 2 | Credentials & feed | `j /config/validate` | see rules below |
| 3 | Kite session | `j /auth/kite/status` | `connected: true` (needed for LIVE and `PAPER_DATA_SOURCE=KITE`) |
| 4 | SEBI state | `j /sebi/status` | `state` = `ACTIVE` (not `KILLED`/`PAUSED`) |
| 5 | Risk limits | `j /risk/status` | `is_halted` false; `max_daily_loss` > 0 and ≤ 5% of capital |
| 6 | Capital split | `j /settings/capital-allocation` | percentages sum ≤ 100 |
| 7 | Agents | `j /settings/agent-enables` | at least one enabled; list which |
| 8 | Market | `j /market/status` | report open/closed (closed is a WARN, not a FAIL) |
| 9 | Backtest approvals | `j /backtest/approved/<s>` for each enabled agent (`intraday`, `scalping`, `swing`, `fno`) | ≥ 1 approved symbol per enabled agent, else WARN |
| 10 | Tests | `cd algotrader_v4 && python3 test_pipeline.py 2>&1 \| tail -3` | `0 failed` |

**Check 2 rules** (`/config/validate`):
- `trading_mode` = `PAPER`: PASS. Note `paper_data_source` (`SIM` = simulated prices, `KITE` = real prices). If `KITE`, `kite_access_token`/`kite_initialised` must be true.
- `trading_mode` = `LIVE`: FAIL unless `ready_to_trade` is true **and** `ticker_source` is `KITE_WS` or `KITE_REST`.
- `api_key_set` false → FAIL (all mutating endpoints would be open).
- `anthropic_api_key` false → WARN: the Claude trade gate is off and every signal is auto-approved at confidence 70.

The test suite takes a few minutes; run it in the background while doing the
HTTP checks. Test section 13 needs a live Anthropic key — if only that section
fails, report WARN, not FAIL.

## Going LIVE

If the user is preparing to go LIVE, also confirm — and list as explicit items:
- They have run at least one full real-price paper session (`TRADING_MODE=PAPER`,
  `PAPER_DATA_SOURCE=KITE`) since the last code change, and reviewed it with `/trade-review`.
- `KILL_SWITCH_RESET_SECRET` is set and differs from `API_KEY`.
- The server's public IP is whitelisted with Zerodha (static IP rule for algos).

Never edit `TRADING_MODE` yourself. Switching to LIVE is the user's action.

## Report format

A table `# | Check | Result (PASS/WARN/FAIL) | Detail`, then one line:

- **GO** — no FAIL.
- **GO with warnings** — WARNs only; list them.
- **NO-GO** — any FAIL; list each with the fix.

Never include the API key, tokens or secrets in the report.
