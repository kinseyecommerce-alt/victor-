---
name: preflight
description: >
  Read-only GO / NO-GO check of AlgoTrader Pro (core rebuild, PAPER only) before a
  trading day: server health, PAPER mode, Kite session and price feed, SEBI kill
  switch, global / per-strategy trading states and the drawdown brake, approval
  mode and pending tickets, the four strategies' entry blocks, drift warnings,
  scheduled jobs, the LIVE-readiness report and the test suite (API_V2.md).
  Changes nothing. Use when the user asks "are we ready", "preflight", "can I go
  live", or before /paper-session.
---

# Preflight — AlgoTrader Pro GO / NO-GO

Checks whether the app is ready for **today's paper session**. Read-only: never start / stop the bot, change a
trading state, approve a ticket, reset the brake, touch the kill switch or edit `.env`. Report problems and how to
fix them — the user decides (the dashboard's Risk and System pages have the controls).

Since the core rebuild (Nov 2026, `algotrader_v4/REBUILD.md`) the app runs four lab-validated **PAPER-only**
strategies driven by the scheduler: `auc_fade` (AUC-FADE), `attn_on` (SWING-ATTN-ON-3X), `vrp_winged`
(VRP-STRANGLE-016-WINGED) and the candidate `pead_h60` (EV-PEAD-LONG-H60). There are no tick agents, no AI trader,
no backtest gate. API reference: `algotrader_v4/API_V2.md`.

## Setup

```bash
ALGO_URL="${ALGO_URL:-http://127.0.0.1:8000}"
# ALGO_API_KEY must be set in the environment. Never echo, log or paste it.
H="X-API-Key: ${ALGO_API_KEY}"
j() { curl -sS -m 15 -H "$H" "$ALGO_URL$1"; }
```
If `ALGO_API_KEY` is unset, say so and run only the public checks (1, 2). Save JSON to the scratchpad and read fields
with a short `python3 -I` script rather than by eye.

## Checks — run all, collect results, then report

| # | Check | Command | PASS when |
|---|-------|---------|-----------|
| 1 | Server up | `j /api/v2/health` | `status` = `ok`, `mode` = `PAPER` (anything else → FAIL: the strategies refuse LIVE) |
| 2 | Config | `j /config/validate` | `api_key_set` true; `paper_data_source` = `KITE` (with `SIM` the agents take no entries → WARN) |
| 3 | Kite + market | `j /api/v2/overview` → `kite`, `market` | `kite.connected` true on a trading day (else FAIL: every agent shows `no_kite_data`); market closed / holiday = INFO |
| 4 | Engine running | overview `bot_running` | true (false = operator Stop, persists until `POST /bot/start` → WARN) |
| 5 | Trading states | `j /api/v2/trading-state` | `global.state` and `brake.state` `ACTIVE`; list any strategy not `ACTIVE` with its reason / by / at (WARN) |
| 6 | Drawdown brake | `j /api/v2/risk-budget` → `brake` | `state` `ACTIVE`; report `drawdown`, `drawdown_pct` vs `reduce_pct` / `halt_pct` |
| 7 | Approvals | `j /api/v2/approvals` | `mode` false, or (mode on) a pending ticket per enabled strategy before its deadline — list them |
| 8 | Strategies | `j /api/v2/strategies` | per card: `enabled.setting` / `enabled.toggle`, `trading_state.effective`, `entry_block` (null = ready). Explain each block with the codes in API_V2.md (`no_kite_data`, `kill_switch`, `drawdown_kill`, `disabled`, `stopped`, `daily_loss_stop`, `awaiting_approval`, `no_data:store_stale` …). `pead_h60` off by default is INFO |
| 9 | SEBI | `j /sebi/status` | `state` = `ACTIVE` (not `KILLED` / `PAUSED`) |
| 10 | Drift | `j /api/v2/drift` | every `status` `ok`; quote `warnings` / `alerts` (warnings only — they never change anything) |
| 11 | Jobs | `j /api/v2/system` → `jobs` | today's entry / exit jobs present (`validated_open`, `validated_auc_exit`, `validated_vrp_entry`, `validated_attn_entry`, `squareoff`) |
| 12 | Leftovers | `j /api/v2/positions` → `orphan_rows` | empty (orphans = rows of the removed engine → WARN) |
| 13 | Tests | `cd algotrader_v4 && python3 test_pipeline.py 2>&1 \| tail -3` | 0 failures (includes the lab-parity tests and `test_rebuild.py`) |

The test suite takes minutes: run it in the background while doing the HTTP checks.

## "Can I go LIVE?"

No. `j /api/v2/readiness` always reports `ready_for_live: false` for the kept strategies — every validated agent
refuses LIVE (`live_mode_refused`). List its `blockers` and failing `checks` (static outbound IP in `KITE_STATIC_IP`
registered with Kite — required for API orders since 1 April 2026 —, kill-switch reset secret, Kite keys …) as the
road map, and say clearly that LIVE is not possible in this build. Never edit `TRADING_MODE`.

## Report format

A table `# | Check | Result (PASS/WARN/FAIL/INFO) | Detail`, then one line:

- **GO** — no FAIL.
- **GO with warnings** — WARNs only; list them.
- **NO-GO** — any FAIL; list each with the fix.

Never include the API key, tokens or secrets in the report.
