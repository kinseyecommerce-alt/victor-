---
name: paper-session
description: >
  Monitor an AlgoTrader Pro paper-trading day (core rebuild): confirm PAPER mode and a
  Kite session, check that the four scheduled lab strategies (auc_fade, attn_on,
  vrp_winged, pead_h60) are ready, then follow the day's scheduled stages through the
  v2 API — plans, entries, positions, exits, trading states, drawdown brake,
  approvals, drift — flag anything abnormal and summarise at the end. The strategies
  run by themselves on the scheduler; this skill never places orders. Use when the
  user says "start a paper session", "watch today's paper trading", or "/paper-session".
---

# Paper Session — AlgoTrader Pro (v2 API)

The strategies are **scheduler-driven** (IST): `auc_fade` enters at the 09:15 auction and exits 09:20; `attn_on`
buys at 15:21 and sells at the next 09:15 auction; `vrp_winged` opens a NIFTY winged strangle at 10:20 and settles
on expiry day; `pead_h60` (candidate, off by default) enters at 09:15 after results and holds ~60 sessions. There is
nothing to "start" per strategy — the 08:55 preflight sets them running unless the operator stopped the bot.
API reference: `algotrader_v4/API_V2.md`. Optional argument: strategies to focus on.

## Setup

```bash
ALGO_URL="${ALGO_URL:-http://127.0.0.1:8000}"
# ALGO_API_KEY must be set. Never echo, log or paste it.
H="X-API-Key: ${ALGO_API_KEY}"
j()    { curl -sS -m 15 -H "$H" "$ALGO_URL$1"; }
post() { curl -sS -m 60 -H "$H" -H "Content-Type: application/json" -X POST "$ALGO_URL$1" -d "${2:-{\}}"; }
```

## 1. Safety gate — do not skip

```bash
j /api/v2/health          # mode must be PAPER — anything else: stop and tell the user
j /api/v2/overview        # kite, market, bot_running, trading_state, approval_mode, strategies.*.entry_block
```
- `paper_data_source` must be `KITE` with `kite.connected` true; otherwise every agent shows `no_kite_data` and
  takes no entries. Ask the user to log in (dashboard → System → Connect Kite, or the 08:45 automatic login).
- `bot_running` false → the operator Stop persists; only the user decides to `post /bot/start`.
- Market closed / holiday → say so; nothing will trade today.

Run `/preflight` first if the user hasn't today.

## 2. Follow the day

Poll at the scheduled stages (use the session's scheduling tools; never a tight loop — the API suggests 10–15 s
for the dashboard, 5–15 min is plenty here):

```bash
j /api/v2/strategies              # per strategy: today.plan, today.positions, today.trades, entry_block, drift.status
j /api/v2/strategies/<name>       # adds day_log (every skip / decision of the day)
j /api/v2/positions               # all open positions, broker rows, orphan rows
j /api/v2/trades?since=<YYYY-MM-DD>
j /api/v2/fills?n=50              # quote vs fill, slip_bps
j /api/v2/trading-state ; j /api/v2/risk-budget ; j /api/v2/approvals ; j /api/v2/drift
```
Useful times: 09:16 (auction entries), 09:21 (auction exits), 10:21 (strangle), 15:16 (square-off),
15:22 (attention entries), 15:41 (store refresh), 17:16 (drift check).

On each poll report only what changed: new plans / entries (symbol, side, qty, price), exits (net P&L), skips from
`day_log`, state changes.

**Raise immediately:**
- `trading_state.global` or `brake` not `ACTIVE`, or a strategy's effective state changed.
- `approval_mode` on with a **pending** ticket close to its deadline (`/api/v2/approvals` → `deadlines`).
- An `entry_block` that is not expected (`daily_loss_stop`, `drawdown_kill`, `no_data:store_stale`, `stopped`).
- `orphan_rows` non-empty; a position still open after its scheduled exit; `errors` in a strategy card.
- Drift `status` `warn` / `alert` (quote the message — warnings never change anything by themselves).
- SEBI `j /sebi/status` not `ACTIVE`.

Never place, modify or cancel orders, change trading states, approve tickets or reset the brake yourself. If something
looks wrong, tell the user and offer the control (`POST /api/v2/trading-state`, `POST /api/v2/approvals/{id}/approve`,
`POST /bot/stop`, `POST /orders/squareoff`); run it only when the user says so.

## 3. End of day

```bash
j /api/v2/positions        # only overnight holds should remain (attn_on CNC, vrp_winged NRML legs, pead_h60)
j /api/v2/trades?since=<today>
```
Then run `/trade-review` for the summary.
