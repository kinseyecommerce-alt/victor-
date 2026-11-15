# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
pip install -r algotrader_v4/requirements.txt
cd algotrader_v4 && uvicorn main:app --host 0.0.0.0 --port 8000 --reload     # dev server
cd algotrader_v4 && python test_pipeline.py                                   # full suite (includes test_rebuild.py, test_validated_pead.py)
cd algotrader_v4 && python test_shadow_registry.py && python test_option_bar_archiver.py && python test_rebuild.py
cd algotrader_v4 && python test_forward_lab.py                                # forward lab (parity part needs the lab scratch)
cd algotrader_v4 && python forward_lab_build.py baseline|parity|bench|warmup --scratch <lab scratch>   # offline tools
cd algotrader_v4 && python lab_risk_build.py --scratch <lab scratch dir> --auc AUC-FADE-2PCT   # rebuild lab_risk.json
```

Copy `.env.example` to `.env` (only the variables `config.py` declares are read; others are ignored).

## Architecture (after the core rebuild — see REBUILD.md)

Everything is PAPER. The kept strategies are four lab-validated agents driven by `platform_scheduler` jobs at the
tested IST times; there is no tick-driven strategy engine any more.

```
platform_scheduler (APScheduler, IST)
  → agents/validated_*.py  (rule imported from stratlab/, entry gate = ValidatedAgent.entry_block_reason)
      reads validated_data (Kite quotes + stored minute sessions), pead_data, options_resolver
      orders: validated_base.paper_marketable / paper_fill_at → kite_client paper broker
  → performance.record_trade (ledger) · paper_book (snapshot / restore) · shadow_registry (observer)
tick_engine: price feed only (newest Kite tick per symbol; the paper broker's touch and position marks)
```

| Module | Role |
|---|---|
| `config.py` | pydantic `Settings` (unknown env vars ignored); mutable at runtime, never written back |
| `main.py` | FastAPI app: auth gate (X-API-Key / JWT), Kite login, vault, bot start/stop, SEBI, ledger, research reads |
| `api_v2.py` | the dashboard API `/api/v2` (documented in `API_V2.md`) |
| `platform_scheduler.py` | every job (Kite login, validated jobs, forward test, daily reset, square-off, scoreboard, shadow registry, archiver, risk brake, drift monitor); bot start/stop |
| `agents/strategy_agents.py` | `ALL_AGENTS` registry of the four agents and `PROVEN` |
| `agents/validated_base.py` | shared base: entry gates (settings, trading state, drawdown kill, daily loss stop, approvals), paper order helpers (state guard, quote-vs-fill log), persistence, status |
| `trading_state.py` | ACTIVE / REDUCE_ONLY / HALTED, global + per strategy + drawdown brake |
| `approvals.py` | semi-auto approval mode (default off) |
| `risk_budget.py` | size factor per strategy (default 1.0 = lab sizing) and the latching drawdown brake; reads `lab_risk.json` |
| `drift_monitor.py` | live-vs-lab drift and fill quality, warnings only |
| `live_readiness.py` | LIVE prerequisites (static IP …); LIVE stays impossible for the kept strategies |
| `kite_client.py` | Kite wrapper; PAPER broker (`_paper_orders` / `_paper_positions`, `paper_fill_at`); LIVE MARKET orders become price-protected LIMITs |
| `paper_book.py` | PAPER book snapshot / restore (old snapshots load; removed engines' blocks ignored) |
| `performance.py` | ledger after costs, per-strategy stats, daily scoreboard vs NIFTY 50 |
| `sebi_compliance.py` | pre-order checks, kill switch / pause, audit log |
| `forward_test.py`, `shadow_registry.py`, `option_bar_archiver.py` | observers (no orders) |
| `forward_lab.py` (+ `forward_lab_store.py`, `forward_lab_engine.py`, `forward_lab_build.py`) | forward lab: every lab strategy (608 catalogue + 668 tfgrid + the 4 live rules) on each new session after the close, engine in its own process, hash-chained ledgers, frozen rule `forward_lab_rule.json` (sha256 pinned in `forward_lab.RULE_SHA256`), lab references `forward_lab_baseline.json`; `FORWARD_LAB_ENABLED` default false; `GET /api/v2/forward-lab` |
| `bot_state.py` | per-agent enable toggles and the persisted operator Stop |

### Rules that keep paper comparable with the lab
- The agents' parity tests (`test_pipeline.py`, `test_validated_pead.py`) are the proof that a live decision equals
  the lab decision; they must pass unchanged after any change to an agent, the base or the data layer.
- New overlays must default to "no change": trading states ACTIVE, approvals off, allocator scale exactly 1.0, drift
  Telegram off. An overlay may block entries; it must never block an exit the rule schedules.
- No path may place a real-money order for these strategies: they refuse LIVE (`live_mode_refused`), the paper helpers
  raise in LIVE, `kite_client.paper_fill_at` raises in LIVE. Never weaken these guards.

#### Validated agents (lab-protocol) — `agents/validated_*.py`

Strategies that passed stratlab's pre-registered protocol run as PAPER-only, scheduler-driven agents
(`scheduled = True`): `auc_fade` (AUC-FADE-3PCT / 2PCT by `VALIDATED_AUCTION_VARIANT`, `stratlab/auction.py`),
`attn_on` (SWING-ATTN-ON-3X, `stratlab/attention.py`) and `vrp_winged` (VRP-STRANGLE-016-WINGED,
`stratlab/options_vrp.py`) and the candidate `pead_h60` (EV-PEAD-LONG-H60, off by default). They share
`agents/validated_base.py`; the first three also `validated_data.py`.

- Imports, not copies: rule functions and constants come from the lab module; `LAB_FINGERPRINT` (sha1 of their
  source + constants) is pinned in `test_pipeline.py`, so a lab change fails the test until `VERSION` is bumped.
- No legacy guard applies (those modules were removed). Kept: the SEBI kill switch / pause
  (`sebi_compliance.pre_order_check`), the trading states and the semi-auto approval gate (both no-ops by default).
  Own overlays instead: kill switch (`VALIDATED_<X>_ENABLED` / dashboard toggle; entries only), daily loss stop,
  drawdown kill (disables the agent until a manual re-enable). Ledger rows use `charge_risk=False`.
- Quote ↔ bar mapping (parity test `t_va_lab_parity`): auction price = Kite `ohlc.open` at 09:15:05 = the 09:15 bar
  open; `prev_close` = last 1-minute close of the previous valid (≥ 200-bar) session from `validated_data`'s store
  (Kite minute candles through `summarize_sessions`, the same function the parity test uses), not `ohlc.close`.
- `VALIDATED_ONLY` (default true): false blocks every validated entry (`validated_only_off`); open positions still
  exit / settle by rule. The legacy book it used to switch between was removed in the core rebuild.
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

Per-agent toggles live in `bot_state.py` (persisted `agent_enables.json`; keys of removed agents are kept on disk and
ignored). `POST /settings/agent-enables` or `POST /api/v2/strategies/{name}/enable` set them; the server flags
`VALIDATED_<X>_ENABLED` are the master switches.

### Settings Pattern

All settings are read from the environment / `.env` at startup via `pydantic-settings`; runtime changes mutate the
`settings` object and are never written back. Trading states and the approval mode are persisted in their own files
next to the ledger.
