# AlgoTrader Pro — Project Memory

> Last updated: 2026-05-23 (n8n integration added)  
> Branch: `claude/create-nirma-trade-repo-7rMKX`  
> Repo: `spbtextile/JAG`

---

## What This Project Is

**AlgoTrader Pro v4** — a production-grade Indian stock market algo-trading system for NSE/BSE.

- **Backend:** FastAPI + asyncio, Python 3.11, Zerodha Kite API
- **Frontend:** React 18 + TypeScript + Vite + TailwindCSS (light theme)
- **Trading:** 4 strategy agents (Intraday, Scalping, Swing, FnO), SEBI-compliant, kill-switch
- **Data:** KiteConnect WebSocket (LIVE) or GBM simulator (PAPER)
- **AI gate:** Claude Sonnet assesses each trade before execution

---

## Current Branch State

All work lives on `claude/create-nirma-trade-repo-7rMKX`. Key recent commits:

| Commit | What |
|--------|------|
| `292dd87` | feat(n8n): bidirectional n8n webhook integration |
| `c117978` | docs: add memory.md |
| `2bab959` | chore: frontend .gitignore + package-lock.json |
| `fea8041` | abi(run-3): Williams %R indicator + sim test fixes |
| `bf3ded4` | fix(frontend): lightweight-charts v4 API + vite-env.d.ts |

---

## Architecture

### Tick Pipeline
```
KiteConnect WebSocket (LIVE) or GBM simulator (PAPER)
  → tick_engine.py  → LiveIndicators (16+ indicators)
  → asyncio Queue per agent
  → agent._tick_loop() → _check_entry() / _check_exit()
  → claude_trade_gate.py (Sonnet assessment)
  → atomic_bracket.py (entry + SL-M + target)
  → kite_client.py (LIVE: real Kite; PAPER: _paper_orders dict)
```

### Key Files

| File | Role |
|------|------|
| `algotrader_v4/main.py` | FastAPI app, all REST endpoints, WebSocket `/ws`, `/webhooks/n8n` |
| `algotrader_v4/kite_client.py` | Kite REST + KiteTicker; paper orders in `_paper_orders` |
| `algotrader_v4/tick_engine.py` | LiveIndicators dataclass, IndicatorCalc, TickBuffer, TickEngine |
| `algotrader_v4/agents/strategy_agents.py` | IntradayAgent, ScalpingAgent, SwingAgent, FnOAgent |
| `algotrader_v4/risk_manager.py` | Pre-order checks, position sizing, daily loss limits |
| `algotrader_v4/sebi_compliance.py` | Kill-switch, audit log, IP whitelist, algo IDs |
| `algotrader_v4/config.py` | Pydantic Settings from `.env`; all runtime config |
| `algotrader_v4/n8n_bridge.py` | Async fire-and-forget outbound webhook to n8n |
| `algotrader_v4/frontend/` | React + Vite SPA (Watchlist, Chart, OrderPanel, 6 tabs) |
| `algotrader_v4/deploy/` | Dockerfile, docker-compose, nginx, EC2 setup scripts |

---

## Indicators in tick_engine.py

`LiveIndicators` dataclass fields (all in `all_latest()` dict):

| Field | Description |
|-------|-------------|
| `ema9`, `ema21`, `ema50`, `ema200` | Exponential Moving Averages |
| `vwap` | Volume Weighted Average Price |
| `rsi_14`, `rsi_7` | Relative Strength Index |
| `macd`, `macd_signal`, `macd_hist` | MACD (12,26,9) |
| `bb_upper`, `bb_mid`, `bb_lower` | Bollinger Bands (20, 2σ) |
| `atr_14` | Average True Range |
| `obv` | On-Balance Volume |
| `supertrend`, `supertrend_dir` | Supertrend (10, 3.0) |
| `hma`, `hma_dir` | Hull Moving Average (20) |
| `squeeze_on`, `squeeze_momentum` | TTM Squeeze |
| `vwap_upper2/3`, `vwap_lower2/3` | VWAP Bands (2σ/3σ) |
| `stoch_rsi_k`, `stoch_rsi_d` | Stochastic RSI (14,3,3) |
| `williams_r` | Williams %R (14) — added run-3 |
| `trend`, `momentum`, `volatility` | Derived labels |

### IndicatorCalc API
```python
# Static method — pass a DataFrame of OHLCV candles:
ind = IndicatorCalc.compute(symbol, tick, df)

# TickBuffer — 60-second resolution:
buf = TickBuffer(60)
buf.push(ltp, volume, timestamp)
candles = buf.candles()
```

---

## Strategy Agents

### IntradayAgent patterns
`VWAP_TREND`, `EMA_PULLBACK`, `ORB_BREAK`, `BREAKOUT`, `VWAP_RECLAIM`, `TTM_SQUEEZE`, `VWAP_BAND_REVERT`

### ScalpingAgent patterns
`EMA9X`, `EMA921X`, `VWAP_BOUNCE`, `SURGE`, `ORB`, `SUPERTREND_FLIP`

Pattern scoring: base_score + context bonus ≥ MIN_SCORE (4) to fire. Base scores: 3=weak, 4=solid, 5=strong, 6+=very strong.

---

## Security Model

- All mutating routes + sensitive GETs require `X-API-Key` or `Authorization: Bearer <JWT>`
- `/config/validate`, `/health`, `/openapi.json`, `/auth/login-url` are public (no auth)
- IP whitelist enforced on `/orders/*`, `/sebi/*` endpoints
- Rate limits: 30 orders/min per IP, 10 signals/min per IP
- WebSocket auth: `ws://host/ws?token=<API_KEY>`; rejects with code 4001 if invalid
- Max 50 concurrent WebSocket connections
- `KILL_SWITCH_RESET_SECRET` separate from `API_KEY` for SEBI reset

---

## Tests

| Suite | File | Count | Status |
|-------|------|-------|--------|
| Unit tests | `test_pipeline.py` | ~417 | ✅ pass (skip section 13 — needs Anthropic key) |
| Simulation orders | `test_sim_orders_flow.py` | 13 | ✅ 13/13 pass |
| E2E API | `playwright_e2e.py` | 59 | ✅ 59/59 pass |
| UI buttons | `test_buttons.py` | varies | ✅ pass |

### Run tests
```bash
cd algotrader_v4

# Unit tests (skip Anthropic section)
python3 -c "
src = open('test_pipeline.py').read()
cut = src.find('# 13. SIGNAL ENGINE')
exec(compile(src[:cut if cut > 0 else len(src)], 'test_pipeline.py', 'exec'))
"

# Simulation orders (paper mode, no Kite token needed)
python3 test_sim_orders_flow.py

# E2E (requires server running on :8000)
python3 playwright_e2e.py
```

---

## Frontend

**Location:** `algotrader_v4/frontend/`  
**Stack:** React 18 + TypeScript + Vite 5 + TailwindCSS 3 + TradingView Lightweight Charts v4 + Recharts + Zustand

### Key components
- `Header` — logo, PAPER/LIVE badge, market status, IST clock, Start/Stop Bot
- `Watchlist` — live sparklines, LTP, % change per symbol
- `MainChart` — TradingView OHLCV chart with EMA9/21 overlays
- `OrderPanel` — BUY/SELL form, qty, market/limit, MIS/CNC, place + squareoff
- Bottom tabs: **Positions | Orders | Brackets | Risk | Agents | SEBI**

### Build
```bash
cd algotrader_v4/frontend
npm install
npm run build    # outputs to dist/
npm run dev      # dev server on :5173
npm run preview  # preview built dist on :4173
```

### API base URL priority
`localStorage('api_base')` → `VITE_API_BASE_URL` env var → `http://localhost:8000`

### Known issue: lightweight-charts
Library is **v4.2.3**. Use v4 API:
- `chart.addCandlestickSeries()` — NOT `chart.addSeries(CandlestickSeries, ...)`
- `chart.addLineSeries()` — NOT `chart.addSeries(LineSeries, ...)`

---

## n8n Bidirectional Integration

### Outbound (AlgoTrader → n8n)

Set `N8N_WEBHOOK_URL` in `.env`. AlgoTrader POSTs these 5 event types automatically:

| Event | Fired from | Key data fields |
|-------|-----------|-----------------|
| `trade_entry` | `base_agent.py` `_try_enter()` | agent, symbol, action, price, qty, sl, target, order_id, rsi, trend |
| `trade_exit` | `base_agent.py` exit handler | agent, symbol, reason, pnl, side |
| `signal` | `main.py` `gen_signal()` | symbol, strategy, action, price, confidence, pattern |
| `regime_change` | `master_agent_v5.py` regime loop | regime, active, paused, size_factor, reasoning |
| `system` | `master_agent_v5.py` + `main.py` | type: bot_started/bot_stopped/squareoff/daily_reset/kill_switch |

All events include top-level `event`, `timestamp` (IST ISO), `trading_mode`.

**Optional HMAC signing:** Set `N8N_WEBHOOK_SECRET` → every outbound POST gets  
`X-AlgoTrader-Signature: sha256=<hex>` header. Set same secret in n8n Credential.

**Implementation:** `n8n_bridge.py` — shared `httpx.AsyncClient`, 1 retry on network error, silent no-op if URL not set. Always called via `asyncio.create_task()` — never blocks trades.

### Inbound (n8n → AlgoTrader)

`POST /webhooks/n8n` — body: `{"action": "<action>", "payload": {...}}`

Supported actions:

| Action | Maps to |
|--------|---------|
| `get_status` | returns bot running state + daily P&L |
| `start_bot` | starts all agents (POST `/bot/start` equivalent) |
| `stop_bot` | stops all agents |
| `squareoff` | squares off all open positions |
| `place_order` | full order with guard/risk/SEBI checks |

Auth: requires `X-API-Key` header OR, if `N8N_WEBHOOK_SECRET` is set, validates `X-AlgoTrader-Signature` HMAC instead.

### .env keys
```
N8N_WEBHOOK_URL=https://your-n8n.com/webhook/algotrader
N8N_WEBHOOK_SECRET=     # optional; generate: python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### n8n setup steps
1. Create **Webhook node** in n8n → copy URL → paste as `N8N_WEBHOOK_URL`
2. To call AlgoTrader from n8n: **HTTP Request node** → POST `/webhooks/n8n` with `X-API-Key` header
3. Filter events by `{{$json.event}}` in n8n Switch node

---

## Start the Server

```bash
cd algotrader_v4
cp .env.example .env   # fill in credentials first

# Dev
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production
python3 main.py

# Health check
curl http://localhost:8000/health
curl http://localhost:8000/config/validate
```

---

## Deployment

### Frontend → Vercel

| Setting | Value |
|---------|-------|
| Root Directory | `algotrader_v4/frontend` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Install Command | `npm install` |
| Env Var | `VITE_API_BASE_URL=https://your-backend.railway.app` |

`vercel.json` is already committed with SPA rewrites.

### Backend → Railway / Render / EC2

```bash
# Railway: set root dir = algotrader_v4, start command:
uvicorn main:app --host 0.0.0.0 --port $PORT

# Docker (EC2/VPS):
cd algotrader_v4/deploy
docker-compose up -d
```

**FastAPI CANNOT run on Vercel** — WebSockets, background threads, and persistent state are not supported by serverless.

### Required env vars for backend
```
TRADING_MODE=PAPER           # or LIVE
API_KEY=<secret>
JWT_SECRET_KEY=<32-char-random>
KITE_API_KEY=<zerodha>
KITE_API_SECRET=<zerodha>
KITE_ACCESS_TOKEN=<zerodha>  # rotates daily in LIVE mode
ANTHROPIC_API_KEY=<anthropic>
FRONTEND_ORIGIN=https://your-app.vercel.app   # for CORS
N8N_WEBHOOK_URL=                               # optional — n8n outbound events
N8N_WEBHOOK_SECRET=                            # optional — HMAC signing
```

---

## Abi — Autonomous Agent

Run `/abi` to trigger the self-improvement loop:
- Boots server → runs all 3 test suites → searches web for improvements
- Adds new indicators/strategies → updates UI → commits & pushes

Abi's skill + learned knowledge: `.claude/skills/abi/SKILL.md`

Playwright Chromium path (this environment):
```
/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell
```

---

## NSE API Notes

- Endpoints `/api/quote-equity` and `/api/allIndices` — working as of 2026-05
- Rate limit: 8 req/s (125ms interval) — safe under NSE's 10 OPS hard cap
- NSE May 2025 circular: OAuth + 2FA + static IP required for registered algos
- `NSE_HEADERS` in `market_data.py` mimics browser headers to avoid bot detection

---

## Known Constraints

- `test_pipeline.py` section 13 requires live Anthropic key — skip with `src[:cut]` trick
- `python3 main.py` exit code 144 in this sandbox — use `uvicorn main:app` directly instead
- Python 3.11 — no backslash inside f-string `{}` braces
- `anthropic>=0.50.0` required when `httpx>=0.28.0` is installed
- Server must start from `algotrader_v4/` directory (relative imports)
