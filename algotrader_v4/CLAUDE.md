# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Market: MCX commodity futures

The platform has been **restructured for MCX (Multi Commodity Exchange of India)**
commodity-futures trading (bullion, energy, base metals). The NSE/BSE equity
agents are retired (kept in `agents/strategy_agents.py` only for their unit-test
coverage of the shared framework). The live registry is `agents/mcx_agents.py`,
exported as `ALL_AGENTS`, keyed by stable names so risk buckets / scheduler /
dashboard keep working:

| Registry key | MCX agent | Product |
|--------------|-----------|---------|
| `intraday` | MCX Intraday | MIS |
| `scalping` | MCX Scalping | MIS |
| `swing`    | MCX Positional | NRML |
| `fno`      | MCX Options / Spread | NRML |

Each agent runs a registry of **20 strategies** (`agents/mcx_strategies.py`,
80 total) on every tick and takes the best-scoring signal; the score drives the
size factor. Strategies are pure functions over an `SCtx` (indicators + the
symbol's rolling previous-tick state) returning `(action, score)` or `None`.
The registry is keyed by agent name in `STRATEGY_REGISTRY`; agents keep
per-symbol prev state in `self._pstate`. Inspect via `GET /agents/strategies`.

Contracts, lot sizes, tick sizes, margins and sessions live in `mcx_universe.py`.
MCX sizing is **lot-based and margin-aware** (`risk_manager.calculate_quantity`
takes a `symbol=` and returns whole-lot quantities; position-size checks cap on
margin, not notional).

### Market data — broker only

Market data (quotes, ticks, historical bars) comes from the **broker (Zerodha
Kite) in both paper and live modes** — yfinance/NSE-India are no longer used as
live sources. `trading_mode` (PAPER/LIVE) governs **order execution only**, not
the data feed.

- `kite_client.is_connected()` gates all data calls (`quote_kite`, `ltp_kite`,
  `historical_data`, `get_instruments`) — they return empty when no broker
  session, never based on PAPER/LIVE.
- `mcx_instruments.py` resolves each base name (CRUDEOIL) to its live near-month
  futures contract (tradingsymbol + instrument_token) from the broker's MCX
  instrument dump. Used by the WebSocket ticker, REST quote batch and historical.
- `tick_engine` starts the Kite WebSocket whenever the broker is connected (both
  modes) and polls Kite REST as fallback; `get_historical` pulls Kite bars.
- The FastAPI startup establishes the broker session from `KITE_ACCESS_TOKEN`.
- Offline dev fallback: `settings.use_paper_simulator=True` re-enables the GBM
  tick simulator (default False → broker feed).

### Inter-agent communication (agents talk to each other)

- `agent_bus.py` — a shared blackboard. Every agent publishes its SIGNAL / INTENT
  / FILL / EXIT (keyed by symbol) and can read peers' latest messages.
- `agent_coordinator.py` — arbitrates every entry before an order is placed:
  blocks opposite-direction conflicts and duplicate contracts, enforces the
  correlated-group margin cap, applies a global concurrency cap, and boosts or
  damps size based on peer conviction read off the bus.
- Wiring lives in `agents/base_agent.py`: it publishes the SIGNAL, calls
  `agent_coordinator.evaluate()`/`request()` inside `_try_enter`, and releases the
  reservation on exit / SL. Both are gated by `settings.use_agent_bus` and
  `settings.use_agent_coordinator`.
- Read endpoints: `GET /agents/bus`, `GET /agents/coordinator`.

## Common Commands

```bash
# Install dependencies
pip install -r algotrader_v4/requirements.txt

# Start the server (development)
cd algotrader_v4 && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start the server (production, via main)
cd algotrader_v4 && python main.py

# Run the test suites
cd algotrader_v4 && python test_pipeline.py          # core framework (263 tests)
cd algotrader_v4 && python test_sim_orders_flow.py   # paper order lifecycle (13 tests)
cd algotrader_v4 && python test_mcx.py               # MCX universe/bus/coordinator/agents (35 tests)
cd algotrader_v4 && python test_broker_data.py       # broker-only market data (15 tests)
cd algotrader_v4 && python test_mcx_strategies.py    # 20 strategies per agent (16 tests)

# Run a single test class or method
cd algotrader_v4 && python test_pipeline.py TestRiskManager
cd algotrader_v4 && python test_pipeline.py TestRiskManager.test_calculate_quantity

# Pre-learn approved symbols before first run (replaces startup backtest)
cd algotrader_v4 && python historical_learner.py
cd algotrader_v4 && python historical_learner.py --resume   # continue interrupted run
```

Copy `.env.example` to `.env` and fill in credentials before running.

## Architecture

### Tick Pipeline (core data flow)

```
KiteConnect WebSocket (LIVE mode) or kite.quote() REST batch fallback (LIVE) or GBM simulator (PAPER)
  → tick_engine.py  (computes 15+ indicators: EMA, VWAP, RSI, MACD, BB, ATR)
  → asyncio Queue per agent subscriber
  → agent._tick_loop()  (IntradayAgent / ScalpingAgent / SwingAgent / FnOAgent)
  → _check_entry() + _check_exit()
  → claude_trade_gate.py  (per-trade Sonnet assessment, skipped in PAPER if disabled)
  → atomic_bracket.py  (entry + SL-M + target placed atomically)
  → kite_client.py  (LIVE: real Kite API; PAPER: _paper_orders dict, no API calls)
```

In LIVE mode with a valid access token, `kite_ticker.py` (KiteConnect WebSocket) replaces NSE polling. On disconnect it falls back automatically.

### Capital Calculation

```
total_capital × intraday_capital_pct% = intraday bucket
intraday bucket ÷ max_intraday_positions = per-symbol capital
int(per_symbol_capital // ltp) = quantity
```

`risk_manager.max_capital_for_agent(agent_name)` returns the per-symbol capital. `intraday` and `scalping` agents share the same MIS equity bucket; `swing` uses CNC delivery; `fno` uses options NRML (lot-based, full bucket, no per-symbol division).

### Key Modules

| Module | Role |
|--------|------|
| `config.py` | Pydantic `Settings` from `.env`; single `settings` singleton; all runtime config is mutable in-memory (no .env writes) |
| `main.py` | FastAPI app, all REST endpoints; startup launches `tick_engine` and `symbol_scanner` |
| `agents/strategy_agents.py` | All four strategy agents (`IntradayAgent`, `ScalpingAgent`, `SwingAgent`, `FnOAgent`); exported as `ALL_AGENTS` dict |
| `agents/base_agent.py` | Async tick consumer base; trade lifecycle; TSL callback wiring in `_try_enter()` |
| `master_agent_v5.py` | APScheduler jobs: 1-min Claude regime review, daily reset, nightly adaptive review, weekly backtest + memory synthesis |
| `risk_manager.py` | Pre-order checks (daily loss, position count, size); `calculate_quantity(price, agent=)`; `max_capital_for_agent()` |
| `trailing_sl_engine.py` | Tracks trailing SL per position; module-level optional callbacks `on_sl_hit`, `on_sl_moved`, `on_target_hit` |
| `bot_state.py` | Shared enabled/disabled state for agents (avoids `main.py ↔ master_agent` circular imports) |
| `kite_client.py` | Zerodha KiteConnect wrapper; PAPER mode stores orders in `_paper_orders` dict |
| `sebi_compliance.py` | Audit log, kill switch (`KillSwitchState`), IP whitelist, approved algo IDs |
| `symbol_scanner.py` | Scans Nifty 100 universe for liquid, volatile symbols meeting entry criteria |
| `atomic_bracket.py` | Places entry + SL-M + target orders as a unit; rolls back on partial failure |

### Agent Enable/Disable

Agent enabled state lives in `bot_state.py` (not `main.py`) to avoid circular imports. The `/settings/agent-enables` endpoint reads/writes this. `master_agent_v5._apply_directives()` checks enabled state before resuming any strategy.

### Settings Pattern

All settings are read from `.env` at startup via `pydantic-settings`. Runtime changes (risk limits, capital allocation, agent enables, Claude gate threshold) are applied by mutating the `settings` object directly — they survive until process restart but are never written back to `.env`.
