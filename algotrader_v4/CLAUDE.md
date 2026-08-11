# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**AlgoTrader Pro** — a tick-driven algorithmic trading system for NSE/BSE Indian equity and
options markets. FastAPI + asyncio backend, React SPA frontend, Zerodha Kite for order
execution, Claude for per-trade assessment and market-regime review.

## Repository Layout

The git repo root is one level above this file. Almost all code lives in `algotrader_v4/`.

```
.                            repo root
├── README.md                short project overview
├── memory.md                long-form project notes (PARTLY STALE — see below)
├── .replit                  Replit workflow + deployment config
├── .claude/skills/abi/      "Abi" autonomous self-improvement agent skill (/abi)
└── algotrader_v4/           the application
    ├── CLAUDE.md            this file
    ├── main.py              FastAPI app — every REST route, /ws, /webhooks/n8n
    ├── agents/              BaseAgent + the four strategy agents
    ├── strategies/          empty package (placeholder — strategy logic lives in agents/)
    ├── frontend/            React 18 + TypeScript + Vite SPA
    ├── deploy/              Dockerfile, docker-compose, nginx, EC2 setup
    ├── static/              server-rendered dashboard.html + login.html
    ├── requirements.txt     pinned Python deps
    └── *.py                 ~50 flat modules (see Module Map)
```

Version strings are inconsistent across the tree (`main.py` says v4, `README.md` and the
Abi skill say v5, `startup.py` prints v2). Treat them as cosmetic; there is one codebase.

### Branch and history

The repo's **default branch is `claude/create-nirma-trade-repo-7rMKX`**, not `main` —
check your base branch before opening a PR. The most recent commit (2026-07-01) was made
by a Replit agent; commits before that came from Claude Code sessions.

## Common Commands

All backend commands must run from `algotrader_v4/` — modules import each other by bare
name (`from config import settings`), so a different working directory breaks imports.

```bash
# Install dependencies
pip install -r algotrader_v4/requirements.txt

# One-time credential/connectivity check (validates .env, pings Anthropic + Kite)
cd algotrader_v4 && python startup.py

# Start the server (development)
cd algotrader_v4 && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start the server (production)
cd algotrader_v4 && uvicorn main:app --host 0.0.0.0 --port 8000 \
    --workers 1 --loop uvloop --http httptools

# Pre-learn approved symbols before first run (replaces the startup backtest)
cd algotrader_v4 && python historical_learner.py
cd algotrader_v4 && python historical_learner.py --resume   # continue interrupted run
```

Copy `.env.example` to `.env` and fill in credentials before running.

**Always use one worker.** All state — kill switch, open positions, tick engine, order
guard, paper orders — is in-memory per process. A second worker silently gets its own
copy and the two disagree about live positions.

### Tests

There is **no pytest and no unittest**. Each suite is a plain script with a custom
`run(name, fn)` / `ok()` / `fail()` harness that executes at module level, prints a
summary, and exits non-zero on failure. There are no test classes, so **you cannot select
a single test from the command line** — arguments are ignored. To run one test, call its
`t_*` function directly or comment out the `run(...)` lines you don't want.

```bash
cd algotrader_v4

# Full unit/integration suite — ~263 checks across 16 sections
python test_pipeline.py

# Section 13 (SIGNAL ENGINE) calls the live Anthropic API and hangs offline.
# Standard workaround — execute the file only up to that section:
python3 -c "
src = open('test_pipeline.py').read()
cut = src.find('# 13. SIGNAL ENGINE')
exec(compile(src[:cut if cut > 0 else len(src)], 'test_pipeline.py', 'exec'))
"

# Paper-mode order lifecycle + tick pipeline (13 checks, no Kite token needed)
python test_sim_orders_flow.py

# API/browser E2E — requires the server already running on :8000
python playwright_e2e.py

# UI button coverage across every dashboard tab
python test_buttons.py
```

`phase4_trade_lifecycle.py`, `phase5_intelligence.py`, and `phase6_resilience.py` are
additional standalone scenario scripts (trade lifecycle, intelligence modules, error
handling) that run the same way.

### Frontend

```bash
cd algotrader_v4/frontend
npm install
npm run dev       # Vite dev server on :5000
npm run build     # tsc && vite build → dist/
npm run preview   # serve the built dist/
```

## Architecture

### Tick pipeline (core data flow)

```
KiteConnect WebSocket (LIVE) │ kite.quote() REST batch fallback (LIVE) │ GBM simulator (PAPER)
  → tick_engine.py            builds 1-min candles, computes ~25 indicators
  → asyncio.Queue per agent   tick_engine.add_subscriber("agent_<name>")
  → BaseAgent._run_loop()     IntradayAgent / ScalpingAgent / SwingAgent / FnOAgent
  → evaluate_tick()           pattern detection + context scoring
  → claude_trade_gate.py      per-trade Sonnet assessment (skippable)
  → risk_manager + order_guard + sebi_compliance   pre-order gates
  → atomic_bracket.py         entry + SL-M + target placed as a unit
  → kite_client.py            LIVE: real Kite API │ PAPER: _paper_orders dict
```

In LIVE mode with a valid access token, `kite_ticker.py` (KiteConnect WebSocket) replaces
REST polling and falls back automatically on disconnect. `truedata_client.py` can replace
the Kite feed entirely when `USE_TRUEDATA_WEBSOCKET=true`.

### Signal scoring

Agents score setups rather than firing on a single condition:

```
base_score (from the matched pattern) + ctx_bonus (EMA/VWAP/RSI/volume/MACD/flow alignment)
    ≥ MIN_SCORE (4)  → signal fires
```

Base scores by convention: `3` = weak, `4` = solid, `5` = strong, `6+` = very strong.
`IntradayAgent` implements patterns as `_pat_*` methods wired into a tuple loop in
`evaluate_tick()`; `ScalpingAgent` and `SwingAgent` use a single `_detect_pattern()` /
`_score_setup()` pair instead. Adding a pattern means touching both the detector and
`_update_state()` (which maintains the `_prev_*` dicts that cross-detection uses).

### Capital calculation

```
total_capital × intraday_capital_pct% = intraday bucket
intraday bucket ÷ max_intraday_positions = per-symbol capital
int(per_symbol_capital // ltp) = quantity
```

`risk_manager.max_capital_for_agent(agent_name)` returns the per-symbol capital.
`intraday` and `scalping` share the MIS equity bucket; `swing` uses CNC delivery; `fno`
uses the options NRML bucket (lot-based, full bucket, no per-symbol division).

### Scheduled jobs

Two independent APScheduler instances, **both pinned to `Asia/Kolkata`** — never rely on
server local time for market decisions.

| Scheduler | Job | When (IST) |
|---|---|---|
| `platform_scheduler.py` (starts with FastAPI, runs even when the bot is stopped) | Kite token auto-refresh | 08:50 Mon–Fri |
| | Pre-market report | 09:00 Mon–Fri |
| | Morning data refresh | 09:10 Mon–Fri |
| | Auto-start bot | 09:16 Mon–Fri |
| | Options cache refresh | every 5 min |
| `master_agent_v5.py` (starts with the bot) | Claude regime review | every 60 s |
| | Auto squareoff | `SQUAREOFF_TIME` Mon–Fri |
| | Daily reset | 09:15 Mon–Fri |
| | Nightly adaptive review | 21:00 Mon–Fri |
| | Weekly backtest | 20:00 Sun |
| | Weekly memory synthesis | 21:00 Sun |

`platform_scheduler` silently skips everything if `KITE_API_KEY` is unset.

### Module map

**Core loop**

| Module | Role |
|---|---|
| `main.py` | FastAPI app, ~75 routes, `/ws`, `/webhooks/n8n`; startup launches tick engine + schedulers |
| `config.py` | Pydantic `Settings` from `.env`; single mutable `settings` singleton |
| `tick_engine.py` | `Tick`/`Candle`/`LiveIndicators`/`MarketSnapshot`, `TickBuffer`, `IndicatorCalc`, `TickEngine` |
| `agents/base_agent.py` | Async tick consumer; trade lifecycle; `_try_enter()` wires TSL callbacks |
| `agents/strategy_agents.py` | `IntradayAgent`, `FnOAgent`, `SwingAgent`, `ScalpingAgent`; exported as `ALL_AGENTS` |
| `agents/master_agent.py` | 7-line back-compat shim re-exporting `master_agent_v5` |
| `master_agent_v5.py` | Regime gating, Claude directives, scheduler jobs, `MASTER_PROMPT` |
| `bot_state.py` | Agent enable/disable flags — exists solely to break the `main ↔ master_agent` import cycle |
| `ist_clock.py` | `now_ist()`, `is_market_open()`, `minutes_to_squareoff()` — **use these, never `datetime.now()`** |

**Execution and risk**

| Module | Role |
|---|---|
| `kite_client.py` | Kite REST wrapper; PAPER mode stores orders in `_paper_orders` |
| `kite_ticker.py` | KiteConnect WebSocket; symbol → instrument-token mapping |
| `kite_auto_login.py` | Headless Playwright + TOTP for the daily Kite OAuth refresh |
| `atomic_bracket.py` | Entry + SL-M + target as one unit; rolls back on partial failure |
| `trailing_sl_engine.py` | Per-position trailing SL; module-level callbacks `on_sl_hit`, `on_sl_moved`, `on_target_hit` |
| `risk_manager.py` | `check_before_order()`, `calculate_quantity()`, `max_capital_for_agent()` |
| `order_guard.py` | Duplicate blocking, per-strategy daily trade caps, post-loss cooldown |
| `sebi_compliance.py` | 10 SEBI regulations: kill switch, audit log, IP whitelist, algo IDs |
| `correlation_guard.py` | Rolling 20-day correlation matrix; blocks over-concentration |
| `event_calendar.py` | Blocks/downsizes entries around earnings, ex-dividend, splits |

**Market data**

| Module | Role |
|---|---|
| `market_data.py` | NSE India + yfinance clients; `NSE_HEADERS` browser spoofing; 8 req/s limiter |
| `truedata_client.py` | Optional TrueData ticker / history / option-chain clients |
| `symbol_scanner.py` | Picks each morning's tradable symbols from the Nifty 100 universe |
| `nifty100.py` | Nifty 100 constituent list |
| `levels_engine.py` | Prior-day H/L/C, pivots, weekly levels, VWAP ± ATR bands |
| `multi_timeframe.py` | Derives 5m/15m candles from 1m history — no extra API calls |
| `market_regime.py` | 6-regime detector + `REGIME_PLANS` strategy gating |
| `institutional_flow.py` | Delivery %, block/bulk deals |

**Options**

| Module | Role |
|---|---|
| `greeks_engine.py` | Black-Scholes pricing, Greeks, Newton-Raphson IV solver (pure math, no I/O) |
| `iv_surface.py` | IV smile, skew, term structure |
| `options_intelligence.py` | IV rank/percentile, max pain, PCR, OI buildup |
| `options_flow.py` | Unusual volume, block trades, sweep detection |
| `gamma_scalp.py` | Gamma exposure (GEX), gamma walls, pin risk |

**Intelligence and learning**

| Module | Role |
|---|---|
| `claude_trade_gate.py` | Per-trade Sonnet assessment; prompt-cached system prompt |
| `signal_engine.py` | AI and technical signal generation for `/signals/generate` |
| `strategy_signals.py` | Registry of signal functions + per-strategy backtest thresholds |
| `backtest_engine.py` | Walk-forward backtest; the gate every symbol must pass |
| `auto_backtest_runner.py` | 08:45 IST automated backtest pipeline |
| `historical_learner.py` | One-time pre-learning over Nifty 100 × 4 strategies |
| `adaptive_engine.py` | Nightly parameter re-tuning from live trade outcomes |
| `trade_memory.py` | Post-trade Haiku analysis → rolling knowledge base |
| `pre_market_report.py` | 09:00 IST Claude briefing from global cues + FII/DII |
| `paper_trade_sim.py` | Full offline pipeline simulation across 5 symbols × 4 agents |

**Integration**

| Module | Role |
|---|---|
| `auth.py` | JWT login (PyJWT, HS256) + bcrypt password hashing + Kite OAuth helpers |
| `n8n_bridge.py` | Fire-and-forget outbound webhooks — always via `asyncio.create_task()` |
| `platform_scheduler.py` | Server-level scheduler, independent of bot start/stop |

### Indicators

`LiveIndicators` (a dataclass in `tick_engine.py`) carries ~40 fields: EMA 9/21/50/200,
VWAP + 2σ/3σ bands, RSI 14/7, MACD, Bollinger Bands, ATR-14, OBV, volume ratio,
Supertrend (+direction), Hull MA (+direction), TTM Squeeze (`squeeze_on`,
`squeeze_momentum`), Stochastic RSI, Williams %R, plus derived `trend` / `momentum` /
`volatility` labels.

Adding an indicator means four coordinated edits — miss one and it silently never reaches
the UI:

1. Add the field(s) to the `LiveIndicators` dataclass.
2. Add a `_<indicator>()` helper above `class IndicatorCalc`.
3. Compute it inside `IndicatorCalc.compute()`, guarded by a minimum bar count.
4. Add it to `TickEngine.all_latest()` **and** to the `ws_broadcast` payload in
   `TickEngine._process_tick()`, or the dashboard never sees it.

Older notes in `memory.md` and the Abi skill tell you to update "both `ws_broadcast`
blocks" — that is out of date. Kite-WS ticks and polled ticks both funnel through
`_process_tick()`, so there is now exactly one broadcast site.

`IndicatorCalc.compute(symbol, tick, df)` is a **static** method taking an OHLCV
DataFrame — there is no `update()`/`current()` instance API. `TickBuffer` requires a
resolution: `TickBuffer(60)`, and `push(ltp, volume, timestamp)` takes scalars, not a
`Tick`.

## Conventions

### Settings

All config is read from `.env` at startup via `pydantic-settings` into a single `settings`
singleton. Runtime changes (risk limits, capital allocation, agent enables, Claude gate
threshold) mutate that object **in memory only** — they never get written back to `.env`
and do not survive a restart. New config knobs go in `config.py` *and* `.env.example`.

### Agent enable/disable

Enabled state lives in `bot_state.py`, not `main.py`, purely to avoid a circular import
between `main.py` and `master_agent_v5.py`. `/settings/agent-enables` reads and writes it;
`master_agent_v5._apply_directives()` checks it before resuming any strategy.

### Time

Every market-timing decision goes through `ist_clock.py`. Production runs in UTC, so
`datetime.now()` in trading logic is a bug. Both schedulers pass `timezone="Asia/Kolkata"`
explicitly — keep that when adding jobs.

### n8n events

`n8n_bridge.notify()` never raises, never blocks, and no-ops when `N8N_WEBHOOK_URL` is
unset. Call it only as `asyncio.create_task(_n8n(...))`, and import it *inside* the
function (all nine call sites do this to avoid import cycles). Five event types are
emitted: `trade_entry`, `trade_exit`, `signal`, `regime_change`, `system`. Setting
`N8N_WEBHOOK_SECRET` adds an `X-AlgoTrader-Signature: sha256=<hex>` HMAC header.

Inbound: `POST /webhooks/n8n` with `{"action": ..., "payload": {...}}` supports
`get_status`, `start_bot`, `stop_bot`, `squareoff`, `place_order`.

### Persisted state

Everything lands under `algotrader_v4/logs/` (gitignored): `approved_symbols.json`,
`learning_progress.json`, `trade_memory.jsonl`, `adaptive/`, `backtests/`. There is no
database.

## Security model

- All mutating routes (POST/PUT/PATCH/DELETE) and sensitive GETs require `X-API-Key` or
  `Authorization: Bearer <JWT>`.
- Public, no auth: `/health`, `/openapi.json`, `/auth/login-url`, `/config/validate`.
- Sensitive GETs behind the gate: `/portfolio/positions`, `/portfolio/orders`,
  `/sebi/audit-log`, `/docs`, `/redoc`.
- IP whitelist on `/orders/*` and `/sebi/*`.
- Rate limits: 30 orders/min and 10 signals/min per IP.
- WebSocket auth via query param: `ws://host/ws?token=<API_KEY>`; rejects with code 4001.
- `KILL_SWITCH_RESET_SECRET` is deliberately separate from `API_KEY`.
- CORS is restricted to explicit methods and headers — no wildcards. Set
  `ALLOWED_ORIGINS` to the deployed frontend origin.

## Frontend

`algotrader_v4/frontend/` — React 18 + TypeScript + Vite 5 + TailwindCSS 3 + Zustand +
TradingView Lightweight Charts v4 + Recharts.

Layout: `Header` (mode badge, market status, IST clock, start/stop), `Watchlist`,
`MainChart`, `OrderPanel`, and six bottom tabs — Positions, Orders, Brackets, Risk,
Agents, SEBI.

**API base resolution** (`store/index.ts`): `localStorage('api_base')` →
`VITE_API_BASE_URL` → `'/api'`. The final fallback is the **relative** `/api`, which the
Vite dev proxy rewrites to `http://localhost:8000`. A deployed build with no
`VITE_API_BASE_URL` set will therefore call its own origin and fail — set the env var for
any non-Replit deployment.

**WebSocket** (`ws/websocket.ts`): absolute `apiBase` → `ws(s)://<host>/ws`; relative
`apiBase` → `/ws-proxy/ws` through the Vite proxy. Reconnects every 3 s.

**lightweight-charts is v4** — use `chart.addCandlestickSeries()` and
`chart.addLineSeries()`. The `chart.addSeries(SeriesType, opts)` form is v5 and will throw.

## Deployment

| Target | Notes |
|---|---|
| Replit | `.replit` runs frontend (:5000) and backend (:8000) in parallel; deploy target `autoscale` |
| Railway | `railway.toml`, NIXPACKS, healthcheck `/health`, root dir `algotrader_v4` |
| Docker / EC2 | `deploy/` — multi-stage Python 3.11 build, non-root user, `.env` mounted at runtime, single worker |
| Vercel | **Frontend only.** Root `algotrader_v4/frontend`, output `dist`, `vercel.json` has SPA rewrites |

The FastAPI backend **cannot** run on Vercel — it needs WebSockets, background threads,
and persistent in-process state.

## Gotchas

- **Working directory** — backend commands must run from `algotrader_v4/`; bare-name
  imports break anywhere else.
- **PyJWT is undeclared.** `auth.py` imports `jwt` (PyJWT), but `requirements.txt` still
  pins only `python-jose`. The 2026-07-01 commit swapped the library without updating
  the pin. A clean `pip install -r requirements.txt` produces an environment where
  `auth.py` fails to import — install `pyjwt` explicitly, or add it to the file.
  `test_buttons.py` still signs tokens with `jose`; both are HS256, so they interoperate.
- **`/gate/log` is registered twice** in `main.py` (under tags `Intelligence` and
  `AI Signal`). FastAPI keeps the first; the second is dead code.
- **Python 3.11** (`.python-version`) — no backslashes inside f-string `{}` braces.
- `anthropic>=0.50.0` is required when `httpx>=0.28` is installed (the `proxies` param
  was removed).
- `python main.py` has been observed exiting with code 144 in sandboxed environments —
  use `uvicorn main:app` directly.
- **`memory.md` is stale** in places: it predates the 2026-07-01 commit, cites the old
  `spbtextile/JAG` repo, says the frontend dev server is on :5173 (it is :5000), and says
  the API base falls back to `http://localhost:8000` (it is now `/api`). Prefer this file.
- The **Abi skill** (`.claude/skills/abi/SKILL.md`) hardcodes `/home/user/JAG` paths, a
  stale push branch, and a Playwright Chromium path that disagrees with `memory.md`.
  Verify paths against the current environment before running `/abi`.
- Claude model IDs currently referenced: `claude-sonnet-4-6` (trade gate, regime review)
  and `claude-haiku-4-5-20251001` (trade memory, startup check).

## Trading modes

`TRADING_MODE=PAPER` (default) simulates fills in `kite_client._paper_orders` and places
no real orders. `TRADING_MODE=LIVE` sends real Zerodha orders and requires a
`KITE_ACCESS_TOKEN` that expires daily (refreshed at 08:50 IST by `kite_auto_login.py`
when TOTP credentials are configured). Verify the mode before testing anything that
places orders — `GET /health` reports it.
