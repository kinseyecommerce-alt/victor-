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
(`scheduled = True`, never started on the tick engine): `auc_fade` (AUC-FADE-3PCT, `stratlab/auction.py`),
`attn_on` (SWING-ATTN-ON-3X, `stratlab/attention.py`) and `vrp_winged` (VRP-STRANGLE-016-WINGED,
`stratlab/options_vrp.py`). All three share `agents/validated_base.py` and `validated_data.py`.

- Imports, not copies: rule functions and constants come from the lab module; `LAB_FINGERPRINT` (sha1 of their
  source + constants) is pinned in `test_pipeline.py`, so a lab change fails the test until `VERSION` is bumped.
- Guards bypassed (each would resize, delay or drop a tested trade): Claude/debate gate, MTF, Kelly, event sizing,
  correlation guard, allocator/bucket room, portfolio VaR, regime pauses, order_guard, risk_manager counters and
  halts, pruning, trailing SL / SL-M / GTT. Kept: the SEBI kill switch / pause (`sebi_compliance.pre_order_check`).
  Own overlays instead: kill switch (`VALIDATED_<X>_ENABLED` / dashboard toggle; entries only), daily loss stop,
  drawdown kill (disables the agent until a manual re-enable). Ledger rows use `charge_risk=False`.
- Quote ↔ bar mapping (parity test `t_va_lab_parity`): auction price = Kite `ohlc.open` at 09:15:05 = the 09:15 bar
  open; `prev_close` = last 1-minute close of the previous valid (≥ 200-bar) session from `validated_data`'s store
  (Kite minute candles through `summarize_sessions`, the same function the parity test uses), not `ohlc.close`.
- `VALIDATED_ONLY` (default true) is the switch between the two books:
  - true: `master.start()` drops the legacy tick agents (`bot_state.LEGACY_AGENTS`: intraday, scalping, swing, fno)
    with report `{"disabled": "VALIDATED_ONLY"}`, `_apply_regime_plan` / `_apply_directives` never start them, the
    monthly momentum rotation (unvalidated, order-placing) is skipped, `/settings/agent-enables` refuses to enable a
    legacy agent (400), and forward_test's 09:20 Telegram summary is muted (`auc_fade` is the official record).
    The AI trader is unchanged: it is the benchmark ("benchmark — no validated edge" in its status).
  - false: today's behaviour returns (legacy agents start as before); validated agents take no new entries but
    their open positions still exit / settle by rule.
  Restored legacy positions from before the switch are still managed by the paper_book watchdog until they exit.
- Guard interplay: the validated agents never call allocator / risk_manager / portfolio_risk / order_guard, and their
  P&L does not feed risk_manager (`charge_risk=False`), so neither book can halt or resize the other.
- Every early return logs through `_note()` / `_skip()` + the per-day `day_log`; `GET /validated` shows all three
  (status, live stats vs the lab expectation, live-readiness criteria — read-only, nothing switches to LIVE).
- One stage at a time: every job coroutine (entry, exit, settlement, `catch_up`) is `@serialized` on a per-agent
  asyncio lock (`validated_base.serialized`, re-entrant within a task). The scheduled job and the restart catch-up
  call the same coroutines; without the lock both pass the done-check before the first Kite await and double every
  order. Exit / settlement loops also re-check `_still_open(pos)` right before each paper fill.
- Running state: the 08:55 preflight and each agent's `catch_up` call `auto_start()` on a trading day, unless the
  operator stopped the bot: `/bot/stop` persists `bot_state.operator_stopped()` (`bot_stopped.json` next to the
  ledger) until the next `/bot/start`, so a Stop never silently expires. `/agents/{name}/pause` lasts the day; the
  enable toggle is the persistent per-agent switch.
- Restart catch-up (`platform_scheduler._validated_catch_up`, start-up + 60 s): refreshes the session store when it
  misses the previous trading day and loads the corporate-action calendar (in memory only) before the agents'
  `catch_up`. `auc_fade` and `attn_on` refuse entries on a stale store (`no_data:store_stale`).
- Calendars: `ist_clock.EXTRA_TRADING_DAYS` (+ `MARKET_EXTRA_SESSIONS`) lists full weekend sessions (Budget days) the
  lab's calendar traded; validated jobs run every day and gate on `is_trading_day`. `ist_clock.SPECIAL_SESSIONS`
  (+ `VALIDATED_SPECIAL_SESSIONS`) lists Muhurat / DR-drill sessions (the lab's special days and announced ones):
  add each year's Muhurat date. The parity tests also run on the live calendar (2026): decisions identical; the lab's
  bars lack sessions on 2026-03-12, 05-12 and 09-11, which ist_clock trades (one strangle expiry and three attention
  exits differ by a day there).
- Live-only guards (not in the tested rule, listed in each `LAB_EXPECTATION["live_only_guards"]`): the
  corporate-action exclusion beyond the lab's `CA_GAPS` and the stale-store refusal.

#### Validated agents (lab protocol): SWING-ATTN-ON-3X (`agents/validated_attention.py`)

- Scheduled (`platform_scheduler` jobs `validated_*`, no tick loop), PAPER only. The rule function and constants are
  imported from `stratlab.attention`; `LAB_FINGERPRINT` (pinned in the tests) changes with them, which forces a
  `VERSION` bump.
- None of the legacy guards applies (trade gate, debate, MTF, allocator, risk_manager limits, order_guard, pruning,
  trailing stops, GTT): none was in the tested rule. SEBI pre-order checks do. Per-agent daily loss stop, drawdown
  kill and kill switch block entries only; open positions always exit by the rule.
- Carrier: a CNC delivery long (the app has no stock futures), booked at the lab's futures costs
  (`performance.record_trade(cost_kind="future", charge_risk=False)`); each row also carries `cost_cnc`.
- Quote ↔ bar mapping: 15:21 quote volume = Σ minute volumes 09:15..15:20, LTP = 15:20 close, ask = 15:21 open, next
  session's official open = 09:15 bar open. Session history comes from `validated_data` (built by
  `summarize_sessions` with the lab's cleaning and 200-bar rule). `test_pipeline.py`'s lab-parity test replays every
  day of the 3-year data set through this code (774/774 lab trades matched).
- Overnight positions are listed in `base_agent.overnight_hold_qty()` and persisted in the paper book's
  `validated` block.

#### Validated agents (lab protocol): VRP-STRANGLE-016-WINGED (`agents/validated_strangle.py`, key `vrp_winged`)

- Only the WINGED (defined-risk) variant trades; NAKED was also confirmed but is not traded, VRP-WINGED-HIGHVIX failed
  the screen. The decision is the lab engine itself: `stratlab.options_vrp.BY_ID[...].strategy()` (an
  `options_replay.ShortStrangle`), `next_expiry`, `MarketView`, `bs`, `_max_risk`, `lot_size_on`; `LAB_FINGERPRINT`
  is pinned in the tests.
- Quote ↔ bar mapping: spot = NIFTY 50 LTP at 10:20 (lab: open of the 10:20 bar), vol = INDIA VIX quote
  `prev_close` (lab: previous VIX day close; fallback: Kite day candle), decision instant = 10:20:00 (lab: the bar
  time), settlement S = mean of the expiry day's 15:00-15:29 Kite minute closes (lab: same bars).
- Contracts must match exactly (`options_resolver.resolve_exact`; Kite's nearest listed expiry must equal the lab's,
  else `expiry_mismatch`). Wings are bought first, then shorts sold; a missing leg quote means no order at all and a
  failed short unwinds the wings (`partial_structure`). NRML legs are listed in `overnight_hold_qty()`.
- Settlement: `kite_client.paper_fill_at(..., fill_kind="SETTLEMENT")` at intrinsic per leg and one ledger row per leg
  at intrinsic (an option settled at ₹0 books exit 0, not the entry). Never from a guessed price; pending
  settlements retry every 10 min to 16:51 and at restart (`catch_up`). On the expiry day only the full 30-close window
  15:00-15:29 settles (Kite's feed can lag); a later day settles on what Kite has, flagged `partial_settle_window`.
  A structure restored in status `opening` (the process died between legs) is unwound at restart
  (`unwind_orphans`, deviation `partial_structure`); the day's entry is then done.
- `test_pipeline.py`'s lab-parity test runs all 726 sessions: 152/152 lab trades matched (legs, fills, credit,
  max risk, margin, VIX, settlement), no extra live plans.

### Agent Enable/Disable

Agent enabled state lives in `bot_state.py` (not `main.py`) to avoid circular imports. The `/settings/agent-enables` endpoint reads/writes this. `master_agent_v5._apply_directives()` checks enabled state before resuming any strategy.

### Settings Pattern

All settings are read from `.env` at startup via `pydantic-settings`. Runtime changes (risk limits, capital allocation, agent enables, Claude gate threshold) are applied by mutating the `settings` object directly — they survive until process restart but are never written back to `.env`.
