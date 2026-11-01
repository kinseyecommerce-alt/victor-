---
name: trade-review
description: >
  Review the day's AlgoTrader Pro trading (paper or live): rebuild every
  round-trip trade from the order book, compute P&L, win rate and profit factor
  per agent, show how each trade exited (stop-loss, trailing SL, target,
  strategy rule), summarise Claude gate approvals and skips, and flag risk or
  compliance problems. Read-only. Use when the user asks "how did we do today",
  "review trades", "P&L by strategy", or at the end of /paper-session.
---

# Trade Review — AlgoTrader Pro

Read-only analysis. Never place or cancel orders or change settings.

## Setup

```bash
ALGO_URL="${ALGO_URL:-http://127.0.0.1:8000}"
# ALGO_API_KEY must be set. Never echo, log or paste it.
H="X-API-Key: ${ALGO_API_KEY}"
j() { curl -sS -m 15 -H "$H" "$ALGO_URL$1"; }
```

## 1. Collect

```bash
j /config/validate        # trading_mode + paper_data_source — say which in the report
j /portfolio/orders       # source of truth for trades
j /portfolio/positions    # anything still open
j /risk/status            # daily_pnl, trades_today, is_halted
j /bot/status             # per-agent trades_today / pnl_today / last_signal
j /gate/log?n=200         # Claude gate decisions
j /sebi/status            # kill-switch state, order-to-trade ratio
```

Also read `algotrader_v4/logs/trade_memory.jsonl` if it exists (one JSON object
per closed strategy-exit trade, with Claude Haiku's lesson). It does not cover
every exit, so use it for lessons only, not for P&L.

Write the JSON to files in the scratchpad and analyse them with a short Python
script (`python3 -I script.py file.json`) rather than by eye.

## 2. Rebuild round trips from `/portfolio/orders`

Order `tag` (Kite truncates it to 20 characters, so match on prefixes) identifies the leg:

| Tag prefix | Meaning |
|------------|---------|
| `Agent-<agent>` (exact) | entry |
| `Agent-<agent>-SL` | protective SL-M (rests until cancelled; `CANCELLED` after a normal exit) |
| `Agent-<agent>-EXIT` | strategy-rule exit (`should_exit_position`) |
| `Agent-<agent>-NOSL…` | entry flattened because its SL-M was rejected — **always flag** |
| `TSL-HIT-<agent>` | trailing stop-loss exit |
| `TSL-T2-<agent>` | final target exit |
| `TSL-<agent>` | replacement SL-M after a trail move |
| `BRK-…` | manual bracket orders |
| `SquareOff` | end-of-day square-off |

Pair each entry with the next opposite-side `COMPLETE` order on the same
`tradingsymbol` (F&O legs are NFO option symbols). P&L = (exit − entry) × qty
for a long, mirrored for a short. Use `average_price`, falling back to `price`.

## 3. Report

1. **Header:** date, mode (PAPER-SIM / PAPER-KITE / LIVE), net P&L, trades, win rate,
   profit factor (gross win / gross loss), max single loss, `risk/status.daily_pnl`
   (and the gap to your rebuilt P&L, if any).
2. **Per agent table:** trades, wins, losses, win rate, net P&L, average win, average loss.
3. **Exit mix:** count by exit type (TSL-HIT, TSL-T2, EXIT, SquareOff, NOSL).
4. **Pattern view:** for entries whose pattern is known (`/bot/status`
   `last_signal.trigger` holds it as `[PATTERN]` for the latest signal; the gate log
   holds symbol + strategy + confidence), group by pattern where you can. Say
   plainly when a pattern can't be attributed.
5. **Claude gate:** ENTER vs SKIP counts, average confidence, the top 3 SKIP reasons.
6. **Flags** (only if present): any `NOSL` exit; a position still open; an SL-M still
   `TRIGGER PENDING` for a symbol that is flat (leftover stop); a reversed position;
   `is_halted`; SEBI state not `ACTIVE`.
7. **Lessons:** 2-4 bullets from the data (and `trade_memory.jsonl`) — e.g. an agent
   losing on most trades, exits concentrated in TSL-HIT within minutes (stops too
   tight), or the gate skipping one agent's signals repeatedly.

If the mode is PAPER-SIM, end with: "Prices were simulated — these results do
not reflect real market behaviour."
