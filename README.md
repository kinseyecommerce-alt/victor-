# Nirma Trade — AlgoTrader Pro v5

Algorithmic trading system for NSE/BSE Indian equity markets.

## Features
- Tick-driven architecture (1-second cadence via NSE India API)
- 4 strategy agents: Intraday, F&O, Swing, Scalping
- 25 proven signal generators
- Atomic bracket orders (entry + SL placed atomically)
- Adaptive learning engine (auto-tunes parameters from live trades)
- Market regime detection (6 regimes, auto strategy selection)
- SEBI compliance module (10 regulations implemented)
- Backtest gate (every symbol must pass before live trading, and beat NIFTY out of sample)
- Scoreboard vs NIFTY 50 / Nasdaq-100 after costs; losing strategies pruned automatically
- Claude AI trader: intraday equity, index options (buy + credit spreads), index futures,
  swing (delivery) and momentum rotation, with a 200-day trend filter

## Try it in 2 minutes
```bash
cd algotrader_v4 && pip install -r requirements.txt && ./run_demo.sh
```
Open http://localhost:5000 — the animated Overview dashboard runs on simulated prices (demo mode,
no keys, no real orders). See [algotrader_v4/TESTING.md](algotrader_v4/TESTING.md) for the full
test path from demo to live.

## Deploy to AWS
See [algotrader_v4/deploy/AWS.md](algotrader_v4/deploy/AWS.md): one t3.small in Mumbai, set up by
`deploy/user-data.sh` on first boot; `deploy/update.sh` for updates.

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
- **PAPER** (default): Simulates orders, no real money at risk
  - `PAPER_DATA_SOURCE=SIM` (default): prices from a random simulator
  - `PAPER_DATA_SOURCE=KITE`: real Kite market prices, orders still simulated (needs a Kite access token)
- **LIVE**: Real Zerodha Kite orders

## Architecture
- `main.py` — FastAPI server + REST/WebSocket endpoints
- `tick_engine.py` — Real-time tick data from NSE India API
- `agents/` — Strategy agents (intraday, fno, swing, scalping)
- `backtest_engine.py` — Pre-trade symbol qualification
- `risk_manager.py` — Position sizing + daily loss limits
- `order_guard.py` — Duplicate order prevention
- `sebi_compliance.py` — Regulatory compliance
