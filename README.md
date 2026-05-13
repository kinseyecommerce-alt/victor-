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
- Backtest gate (every symbol must pass before live trading)

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
- **LIVE**: Real Zerodha Kite orders

## Architecture
- `main.py` — FastAPI server + REST/WebSocket endpoints
- `tick_engine.py` — Real-time tick data from NSE India API
- `agents/` — Strategy agents (intraday, fno, swing, scalping)
- `backtest_engine.py` — Pre-trade symbol qualification
- `risk_manager.py` — Position sizing + daily loss limits
- `order_guard.py` — Duplicate order prevention
- `sebi_compliance.py` — Regulatory compliance
