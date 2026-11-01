# Testing AlgoTrader Pro

Four levels, from "see it work in two minutes" to "trust it with money". Run them in order.

| Level | What it proves | Needs | Time |
|---|---|---|---|
| 1. Demo | The dashboard and the whole pipeline work end to end | Python 3.11, Node 18+ | 2 min |
| 2. Automated tests | Logic, limits and the dashboard behave as designed | same | 5 min |
| 3. Real-price paper | Strategies on **real** market prices, no real orders | Kite Connect app + daily login | 2–4 weeks |
| 4. Small live | Real orders, tiny size | everything above, passed | ongoing |

---

## 1. Demo — see it running (any time, no keys)

```bash
cd algotrader_v4
pip install -r requirements.txt          # first time
./run_demo.sh
```

Open **http://localhost:5000**. You land on the **Overview** dashboard.

Demo mode = paper trading on the built-in price simulator, with the IST clock starting at
10:00 so the "market" is open whenever you test. Nothing touches Zerodha, nothing needs keys.
It only activates with `TRADING_MODE=PAPER` + `PAPER_DATA_SOURCE=SIM`.

**What you should see within about a minute**

| Area | Look for |
|---|---|
| Top bar | `PAPER-SIM` and `DEMO` badges, green "Market open", clock ticking |
| Ticker tape | Symbols scrolling, cells flashing green/red as prices move (hover pauses it) |
| KPI cards | Numbers counting up/down smoothly; Day P&L card glows green or red |
| Intraday P&L | Curve drawing itself as the session runs |
| Agents | Intraday and Scalping orbs pulsing ("running") |
| Open positions | Cards slide in; the dot moves along **SL → entry → TGT**; the SL label pops and "moves" counts up when the stop trails |
| Live activity | Orders and stops sliding in at the top (IST times) |
| Risk | Gauge fills as realised losses use the daily loss limit |

Try: **Stop bot / Start bot** in the top bar, **Trading view** (the original chart + order panel)
and **✨ Overview** to come back. The page also works at phone width.

`NO_AUTOSTART=1 ./run_demo.sh` starts without the bot, so you can press **Start bot** yourself.
Ctrl+C stops everything.

> The demo turns off the multi-timeframe filter (`USE_MULTI_TIMEFRAME=false`) so trades appear
> quickly on simulated prices. Keep it **on** for real use.

### Demo with the AI trader
Add your Claude key to try the AI on simulated prices (this costs real API money — watch
"AI API cost" on the dashboard):

```bash
ANTHROPIC_API_KEY=sk-ant-... ./run_demo.sh
```
Then press **Start AI** in the top bar, or use the **AI Trader** tab in Trading view to chat
("what are you watching?"). AI decisions, alerts and trail moves appear in Live activity.

---

## 2. Automated tests

With the demo still running (it provides the backend and dashboard):

```bash
cd algotrader_v4

# Dashboard in a real browser (27 checks, saves screenshots to e2e_screenshots/)
pip install playwright==1.52.0 && python3 -m playwright install chromium   # first time
python3 dashboard_e2e.py
```

Without anything running:

```bash
python3 test_pipeline.py          # ~340 unit/integration tests, includes the AI trader with a
                                  # fake Claude client (no API spend)
python3 test_sim_orders_flow.py   # paper order flow
cd frontend && npx tsc --noEmit && npm run build    # dashboard type-check + production build
```

All must pass before you move on. Section 13 of `test_pipeline.py` needs a real Anthropic key;
if only that section fails, that is expected without one.

---

## 3. Real-price paper trading (the important one)

Real Kite prices, simulated orders. This is where you find out whether the strategies and the
AI actually make money.

`.env`:
```
TRADING_MODE=PAPER
PAPER_DATA_SOURCE=KITE
DEMO_MODE=false
API_KEY=<long random>       # protects the dashboard/API — set this before saving any keys
```

**Your keys — enter them in the app, not in chat or files you share.** Trading view → ⚙ Settings →
**API Keys**: paste the Kite API key and secret (from developers.kite.trade) and, for the AI trader,
the Anthropic key. With "Remember on this server" ticked they are saved **encrypted** in
`.secrets/vault.enc` (permissions 600, git-ignored), reloaded at startup, and never shown again —
only the last 4 characters. **Remove** deletes one from the vault and the running server.
`python3 secret_store.py` prints what is stored (no values).

In the Kite developer console set the redirect URL to `http://<your-server>:8000/auth/kite/callback`.

**Your Zerodha user ID, password and TOTP are never entered into this app.** Kite Connect requires
you to log in yourself on Zerodha's own page once a day; the app only receives the day's access token,
which it keeps (encrypted) until Kite expires it the next morning.

Each trading day:
1. Settings → API Keys → **Log in with Zerodha** and log in on Zerodha's page.
2. In Claude Code run `/preflight` — it must say **GO**.
3. Start the bot and/or the AI from the Overview top bar. Use `/paper-session` to have Claude
   watch it for you.
4. After the close run `/trade-review`.

Move on only after 2–4 weeks that are profitable **after brokerage and taxes**, with no flags in
`/trade-review` (no position without a stop, no leftover stop orders, no reversed positions).

### What the AI can trade

| Type | Product | Direction | Held | Stop | On by default |
|---|---|---|---|---|---|
| Intraday stocks | MIS | long / short | closed 15:00 | SL-M order | yes |
| Index options | MIS (NFO) | buy CE / PE only | closed 15:00 | SL-limit order | yes |
| Index futures | MIS (NFO) | long / short | closed 15:00 | SL-limit order | **no** — `AI_AGENT_FUTURES_ENABLED=true` |
| Swing stocks | CNC delivery | long only | up to 15 days | **GTT** at Zerodha | yes |
| Bearish swing (puts) | NRML long put, stock or index | bearish | up to 15 days | **GTT** at Zerodha | yes |
| Option selling | MIS credit spreads (sell 1 + buy 1 further OTM) | bull put / bear call | closed 15:00 | code: % of credit | **no** — `AI_AGENT_OPTION_SELLING_ENABLED=true` |

- **Swing:** a daily scan at 14:50 shortlists uptrending stocks (breakout or pullback to the
  20-day EMA) and Claude decides on the daily chart. The stop is a GTT, so it works overnight
  and if the server is off. Open swing positions are saved to `data/ai_swing_positions.json`
  and picked up again on restart; code sells on the last day of the holding period. Check in the
  Kite app (Orders → GTT) that every swing position has its GTT.
- **Bearish swing:** delivery shares can't be shorted overnight, so the AI buys a put (F&O stock
  or index) expiring at least a week after its planned exit. The most it can lose is the premium.
- **Option selling:** only as credit spreads. The protective option is bought first and the short
  option is sold only after that fills; on exit the short is bought back first and the
  protective leg is sold only after that fills. Worst case = (strike width − credit) × quantity,
  capped at ₹10,000 by default. Spreads are not saved across a server restart — if the server
  stops mid-day, close them in Kite (Zerodha also squares off MIS positions near 15:20).
- **Futures:** one lot of NIFTY is ~₹18 lakh of contract value, so a 1% move is ~₹18,000. The
  code caps the loss at the stop (₹5,000 by default), the margin per trade and the lots. Your
  account must have the F&O segment active and enough margin.

---

### Is it beating the index?

The Overview dashboard has an **"Are we beating the index?"** panel (`GET /performance/scoreboard`).
It compares the account's return **after brokerage, taxes and the Claude API bill** with
NIFTY 50 (what a Zerodha account could simply hold instead) and, for reference, Nasdaq-100,
for today, the week, the month, the year and since tracking began. A snapshot is stored every
day at 15:35 (`data/performance.json`). After 20+ trading days the verdict line tells you
plainly whether an index fund would have done better.

**Strategy health** (`GET /performance/strategies`) shows each strategy's trades, win rate, net
P&L and profit factor. Once a strategy has 30 closed trades, if its last 50 are net losers
(negative P&L after costs and profit factor < 1) it is **switched off automatically** and you
are told on Telegram. The dashboard button re-enables it; the judgement then restarts.

**Trend filter:** when NIFTY 50 is below its 200-day EMA the AI halves its intraday size and
makes no new swing or momentum buys (bearish trades are unaffected). Shown on the scoreboard.

**Momentum rotation** (`MOMENTUM_ENABLED`): on the first trading day of each month the top 5
Nifty 100 stocks by 12-month momentum (skipping the latest month, 12-1) are bought as delivery
with a 12% GTT stop and names that dropped out are sold — code only, no Claude call. Check the
plan first: `POST /momentum/rebalance?execute=false`; backtest it against NIFTY with
`GET /backtest/momentum?months=24` (real data only once the server has Kite/yfinance access).

**Backtest gate:** every symbol/strategy backtest now also compares its out-of-sample return
with buying and holding NIFTY 50 over the same windows, and fails if it does not beat it.

## Deploying to a server

`deploy/AWS.md` walks through a Mumbai EC2 server: `deploy/user-data.sh` installs everything
on first boot, `deploy/setup-ec2.sh` is the idempotent installer (systemd + nginx + HTTPS +
dashboard build + generated secrets, PAPER on real prices), `deploy/update.sh` pulls new code
and restarts **only if the test suite passes**.

## 4. Small live trading

```
TRADING_MODE=LIVE
AI_AGENT_ALLOW_LIVE=true          # only if the AI trader should place live orders
AI_AGENT_MAX_POSITION_VALUE=10000
AI_AGENT_MAX_TRADES_PER_DAY=2
MAX_DAILY_LOSS=1000
```
Needs a static IP whitelisted in your Kite developer console. Start tiny, raise limits slowly.
Emergency controls: **Stop bot / Stop AI** in the dashboard, **SEBI kill switch** in Trading
view, and the Kite app itself (always works).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard says "Cannot reach the backend" | Is `uvicorn` running on :8000? `curl localhost:8000/health` |
| Dashboard says "needs your API key" | Trading view → ⚙ Settings → paste the `API_KEY` from `.env` |
| No trades in demo after 2 min | Check the bot is running (Agents orbs pulse); restart `./run_demo.sh` |
| No trades in real-price paper | Normal outside 09:15–15:30 IST; strategies only fire on real setups |
| "Market closed" during market hours | Server clock/timezone — the app uses IST internally; check `date` |
| `dashboard_e2e.py` can't launch a browser | `python3 -m playwright install chromium`, or set `CHROMIUM_PATH` |
