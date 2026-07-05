# Nirma Trade — AlgoTrader Pro (MCX)

Algorithmic trading system for **MCX (Multi Commodity Exchange of India)**
commodity futures — bullion, energy and base metals.

## Features
- Tick-driven architecture (1-second cadence)
- **4 MCX trading-type agents, each running 20 strategies (80 total), that
  communicate with each other:**
  - `intraday` → **MCX Intraday (MIS)** — 20 trend/momentum/breakout strategies
  - `scalping` → **MCX Scalping (MIS)** — 20 fast micro-momentum / mean-reversion strategies
  - `swing`    → **MCX Positional (NRML)** — 20 multi-day trend-following strategies
  - `fno`      → **MCX Options / Spread (NRML)** — 20 volatility/directional strategies + inter-commodity spread overlay
- Each agent evaluates all 20 strategies per tick and takes the best-scoring
  signal (`agents/mcx_strategies.py`); score drives the position size factor.
  Inspect them at `GET /agents/strategies`.
- **Shared agent bus + coordinator** — agents broadcast signals/fills/exposure on a
  blackboard; a coordinator arbitrates every entry (blocks conflicting/duplicate
  contracts, enforces correlated-group margin caps, boosts/damps size on peer conviction)
- **Market data from the broker only** — quotes, ticks and historical bars all
  come from Zerodha Kite (broker WebSocket + REST) in **both paper and live modes**;
  no yfinance / NSE-India feed. MCX base names are resolved to live near-month
  futures contracts from the broker's instrument dump (`mcx_instruments.py`).
- Lot-based, margin-aware position sizing (MCX contracts trade in whole lots)
- MCX session handling (09:00–23:30 IST normal, 09:00–21:00 agri)
- Atomic bracket orders (entry + SL placed atomically)
- Adaptive learning engine, market-regime detection, SEBI compliance module

## Setup
```bash
cd algotrader_v4
cp .env.example .env
# Fill in your API keys
pip install -r requirements.txt
python startup.py
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Data vs. execution
Market **data** always comes from the broker (Kite); **execution** is what
`TRADING_MODE` switches:
- **PAPER** (default): real broker market data, simulated order fills — no money at risk
- **LIVE**: real broker market data, real Zerodha Kite orders on the MCX segment

A valid `KITE_ACCESS_TOKEN` is required for any market data. For fully offline
development without a broker session, set `USE_PAPER_SIMULATOR=true` to fall back
to the built-in GBM tick simulator.

## Architecture
- `main.py` — FastAPI server + REST/WebSocket endpoints
- `mcx_universe.py` — MCX contract universe (lot sizes, tick sizes, margins, sessions)
- `mcx_instruments.py` — resolves base names → live near-month futures from the broker
- `kite_client.py` / `kite_ticker.py` — broker session, quotes, historical, WebSocket ticks
- `agent_bus.py` — inter-agent pub/sub blackboard
- `agent_coordinator.py` — arbitrates entries across agents
- `agents/mcx_agents.py` — the 4 MCX trading-type agents (live registry)
- `agents/mcx_strategies.py` — 80 signal strategies (20 per agent), best-score selection
- `agents/base_agent.py` — tick consumer; publishes to the bus, consults the coordinator
- `tick_engine.py` — real-time / simulated tick pipeline + indicators
- `risk_manager.py` — lot/margin-based sizing + daily loss limits
- `order_guard.py` — duplicate order prevention
- `sebi_compliance.py` — regulatory compliance

### Inspecting agent communication
- `GET /agents` — per-agent status and approved contracts
- `GET /agents/bus` — recent inter-agent bus messages + stats
- `GET /agents/coordinator` — reserved book, per-group exposure, caps

## Tests
```bash
cd algotrader_v4
python test_pipeline.py          # core framework (263 tests)
python test_sim_orders_flow.py   # paper order lifecycle (13 tests)
python test_mcx.py               # MCX restructure: universe, bus, coordinator, agents (35 tests)
python test_broker_data.py       # broker-only market data feed (15 tests)
python test_mcx_strategies.py    # 20 strategies per agent (16 tests)
```
