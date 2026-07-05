# Nirma Trade — AlgoTrader Pro (MCX)

Algorithmic trading system for **MCX (Multi Commodity Exchange of India)**
commodity futures — bullion, energy and base metals.

## Features
- Tick-driven architecture (1-second cadence)
- **4 MCX trading-type agents that communicate with each other:**
  - `intraday` → **MCX Intraday (MIS)** — fast momentum, square off intraday
  - `scalping` → **MCX Scalping (MIS)** — high-frequency small moves
  - `swing`    → **MCX Positional (NRML)** — multi-day trend, carries overnight
  - `fno`      → **MCX Options / Spread (NRML)** — directional options + inter-commodity spreads
- **Shared agent bus + coordinator** — agents broadcast signals/fills/exposure on a
  blackboard; a coordinator arbitrates every entry (blocks conflicting/duplicate
  contracts, enforces correlated-group margin caps, boosts/damps size on peer conviction)
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

## Trading Mode
- **PAPER** (default): Simulates orders, no real money at risk (MCX contracts
  seeded from reference prices)
- **LIVE**: Real Zerodha Kite orders on the MCX segment

## Architecture
- `main.py` — FastAPI server + REST/WebSocket endpoints
- `mcx_universe.py` — MCX contract universe (lot sizes, tick sizes, margins, sessions)
- `agent_bus.py` — inter-agent pub/sub blackboard
- `agent_coordinator.py` — arbitrates entries across agents
- `agents/mcx_agents.py` — the 4 MCX trading-type agents (live registry)
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
```
