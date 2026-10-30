---
name: paper-session
description: >
  Run and monitor an AlgoTrader Pro paper-trading session: confirm the server is
  in PAPER mode (refuses LIVE), start the chosen strategy agents, poll
  positions, orders, risk, the Claude gate and open brackets, flag anything
  abnormal, then stop the bot and summarise. Best with PAPER_DATA_SOURCE=KITE
  (real prices, simulated orders). Use when the user says "start a paper
  session", "paper trade today", "run the bot in paper", or "/paper-session
  intraday,scalping".
---

# Paper Session — AlgoTrader Pro

Drive one paper-trading session end to end. Arguments (optional): a
comma-separated strategy list (`intraday`, `scalping`, `swing`, `fno`) and/or a
comma-separated watchlist. Default strategies: every enabled agent. Default
watchlist: none (the symbol scanner picks).

## Setup

```bash
ALGO_URL="${ALGO_URL:-http://127.0.0.1:8000}"
# ALGO_API_KEY must be set. Never echo, log or paste it.
H="X-API-Key: ${ALGO_API_KEY}"
j()    { curl -sS -m 15 -H "$H" "$ALGO_URL$1"; }
post() { curl -sS -m 120 -H "$H" -H "Content-Type: application/json" -X POST "$ALGO_URL$1" -d "${2:-{\}}"; }
```

## 1. Safety gate — do not skip

```bash
j /config/validate
```

- If `trading_mode` is **not** `PAPER`: **stop**. Tell the user the server is in
  LIVE mode and this skill only runs paper sessions. Do not start the bot.
- Report `paper_data_source`:
  - `KITE` → real prices. Also require `kite_initialised` true, or `j /auth/kite/status`
    returns `connected: true`. If not, ask the user to log in to Kite through the app
    (`/auth/kite/callback` flow) before continuing.
  - `SIM` → tell the user prices are **simulated** (random walk) and results say
    nothing about real market behaviour. Ask whether to continue or switch the
    server to `PAPER_DATA_SOURCE=KITE` first.
- `j /market/status` — if the market is closed, say so: with Kite data the bot idles
  outside market hours.

Optionally run `/preflight` first if the user hasn't today.

## 2. Start

```bash
post /bot/start '{"strategies": ["intraday","scalping"]}'
# with a fixed watchlist:
post /bot/start '{"strategies": ["intraday"], "watchlist": [{"symbol":"RELIANCE","exchange":"NSE"}]}'
```

`400 Already running` → the bot is already on; just monitor. Report the returned
`watchlist` and, per strategy, how many symbols were approved
(`report.<strategy>.approved / total`). If an agent has 0 approved symbols, it
will not trade — say so.

## 3. Monitor

Poll every ~5 minutes (use the session's scheduling tools if available; never a
tight loop):

```bash
j /bot/status            # agents running, trades_today, pnl_today, last_signal
j /portfolio/positions   # open paper positions (quantity ≠ 0)
j /portfolio/orders      # every paper order — tags show agent and exit type
j /risk/status           # daily_pnl, open_positions, is_halted
j /gate/log?n=20         # Claude gate ENTER/SKIP decisions with reasons
j /brackets              # active brackets
j /sebi/status           # kill switch state
```

On each poll report only what changed: new entries (symbol, side, qty, price,
pattern from `last_signal.trigger`), exits (reason, P&L), and any alert below.

**Raise immediately:**
- `risk/status.is_halted` true (daily loss limit hit) or `sebi/status.state` not `ACTIVE`.
- An open position with no matching `TRIGGER PENDING` SL-M order in `/portfolio/orders`.
- A position whose quantity sign is opposite to its entry (an unintended reverse position).
- No ticks: `/market/live` empty or unchanged for 2 polls while the market is open.
- Several consecutive gate SKIPs with the same reason — summarise the reason.

Never place, modify or cancel orders yourself during the session. If something
looks wrong, tell the user and offer `POST /bot/stop` or `POST /orders/squareoff`;
run them only when the user says so.

## 4. End

When the user asks, or at their requested end time (the bot also auto-squares-off
at `SQUAREOFF_TIME`, default 15:10 IST):

```bash
post /bot/stop
j /portfolio/positions    # confirm everything is flat
```

Then run `/trade-review` for the summary, or give a short one: trades, win rate,
net P&L, best/worst trade, P&L per agent.
