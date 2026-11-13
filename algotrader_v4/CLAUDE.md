# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies
pip install -r algotrader_v4/requirements.txt

# Start the server (development)
cd algotrader_v4 && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start the server (production, via main)
cd algotrader_v4 && python main.py

# Run all tests
cd algotrader_v4 && python test_pipeline.py

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
| `agents/strategy_agents.py` | `SwingAgent`, `FnOAgent`, the legacy pattern agents (`IntradayAgent`, `ScalpingAgent`, research/AI-scout only) and the `ALL_AGENTS` registry |
| `agents/intraday_v2.py` | The live `intraday` (opening-range breakout, 20-day trend filter, 2.5R) and `scalping` (opening-range retest, 1.5R) agents, validated on real Kite 1-minute bars via `replay_backtest.py` |
| `agents/base_agent.py` | Async tick consumer base; trade lifecycle; TSL callback wiring in `_try_enter()` |
| `master_agent_v5.py` | APScheduler jobs: 1-min Claude regime review, daily reset, nightly adaptive review, weekly backtest + memory synthesis |
| `risk_manager.py` | Pre-order checks (daily loss, position count, size); `calculate_quantity(price, agent=)`; `max_capital_for_agent()` |
| `trailing_sl_engine.py` | Tracks trailing SL per position; module-level optional callbacks `on_sl_hit`, `on_sl_moved`, `on_target_hit` |
| `bot_state.py` | Shared enabled/disabled state for agents (avoids `main.py ↔ master_agent` circular imports) |
| `kite_client.py` | Zerodha KiteConnect wrapper; PAPER mode stores orders in `_paper_orders` dict |
| `sebi_compliance.py` | Audit log, kill switch (`KillSwitchState`), IP whitelist, approved algo IDs |
| `symbol_scanner.py` | Scans Nifty 100 universe for liquid, volatile symbols meeting entry criteria |
| `atomic_bracket.py` | Places entry + SL-M + target orders as a unit; rolls back on partial failure |
| `paper_book.py` | PAPER only: snapshots the paper book (orders, positions, trailing stops, agent/AI bookkeeping, risk counters) to `PAPER_BOOK_FILE` and restores it after a same-day restart (an earlier session's snapshot: CNC / swing holdings restored, intraday MIS / NRML positions closed at their last price into the ledger); a watchdog manages restored positions until their agent does; resets the day counters at 09:15 when master is not running and squares off restored intraday/NRML positions at `squareoff_time` (`PERSIST_PAPER_BOOK`) |

### Live / replay parity (read before changing the entry or exit path)

The agents' only evidence is `replay_backtest.py` on real Kite minute bars. Any check the live path applies that the
replay does not is an unvalidated change to the strategy, and several such checks silently stopped every live trade
before (Nov 2026): the trade-gate context crashed on every entry, the multi-timeframe veto rejected 100 % of signals,
the ₹50k order cap rejected ₹1L positions, the ledger pruned v2 agents on v1 losses, swing needed 200 candles from a
60-candle snapshot, the legacy per-symbol backtest approved 0 symbols, and "trend failed" rule exits turned every agent
into a loser. Rules that keep them in step:

- Exits: `BaseAgent.rule_exit()` is the single exit-rule policy for both the live loop and the replay. Validated agents
  (`strategy_rule_exits = False`) exit only by stop, target, trailing stop, the MIS square-off at `squareoff_time`
  and the delivery (CNC) holding period `base_agent.MAX_CNC_SESSIONS` (the replay imports the same constant).
- Filters: `replay_backtest.LIVE_FILTERS` lets the replay apply live-only filters (MTF, strategy exits) so their effect
  can be measured before they are switched on.
- Versions: bump an agent's `VERSION` when its logic changes; the performance ledger, pruning and the allocator's live
  evidence count only the current version.
- Every early return in `BaseAgent._try_enter` goes through `_skip()` (logged, counted in `get_status()`); never return
  silently.
- Verify end to end with `test_pipeline.py`'s live-path test (a validated signal must reach the paper broker at full
  size) after touching sizing, risk or filters.

#### Validated agents (lab-protocol) — `agents/validated_*.py`

Strategies that passed stratlab's pre-registered protocol run as PAPER-only, scheduler-driven agents
(`scheduled = True`, never started on the tick engine). So far: `auc_fade` (AUC-FADE-3PCT, `stratlab/auction.py`).

- Imports, not copies: rule functions and constants come from the lab module; `LAB_FINGERPRINT` (sha1 of their
  source + constants) is pinned in `test_pipeline.py`, so a lab change fails the test until `VERSION` is bumped.
- Guards bypassed (each would resize, delay or drop a tested trade): Claude/debate gate, MTF, Kelly, event sizing,
  correlation guard, allocator/bucket room, portfolio VaR, regime pauses, order_guard, risk_manager counters and
  halts, pruning, trailing SL / SL-M / GTT. Kept: the SEBI kill switch / pause (`sebi_compliance.pre_order_check`).
  Own overlays instead: kill switch (`VALIDATED_AUCTION_ENABLED` / dashboard toggle; entries only), daily loss stop,
  drawdown kill (disables the agent until a manual re-enable). Ledger rows use `charge_risk=False`.
- Quote ↔ bar mapping (parity test `t_va_lab_parity`): auction price = Kite `ohlc.open` at 09:15:05 = the 09:15 bar
  open; `prev_close` = last 1-minute close of the previous valid (≥ 200-bar) session from `validated_data`'s store
  (Kite minute candles through `summarize_sessions`, the same function the parity test uses), not `ohlc.close`.
- `VALIDATED_ONLY` (default true) gates the validated agents' entries; false = no validated entries (exits still run).
- Every early return logs through `_note()` → `_skip()` + the per-day `day_log` in `/agents` status.

### Agent Enable/Disable

Agent enabled state lives in `bot_state.py` (not `main.py`) to avoid circular imports. The `/settings/agent-enables` endpoint reads/writes this. `master_agent_v5._apply_directives()` checks enabled state before resuming any strategy.

### Settings Pattern

All settings are read from `.env` at startup via `pydantic-settings`. Runtime changes (risk limits, capital allocation, agent enables, Claude gate threshold) are applied by mutating the `settings` object directly — they survive until process restart but are never written back to `.env`.
