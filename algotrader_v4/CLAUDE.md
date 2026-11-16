# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
pip install -r algotrader_v4/requirements.txt
cd algotrader_v4 && uvicorn main:app --host 0.0.0.0 --port 8000 --reload     # dev server
cd algotrader_v4 && python test_pipeline.py                                   # full suite (includes test_rebuild.py, test_validated_pead.py)
cd algotrader_v4 && python test_shadow_registry.py && python test_option_bar_archiver.py && python test_rebuild.py
cd algotrader_v4 && python test_forward_lab.py                                # forward lab (parity part needs the lab scratch)
cd algotrader_v4 && python test_builder_data.py                               # builder data layer (layout / parity parts need the lab scratch)
cd algotrader_v4 && python test_builder.py                                    # automatic strategy builder (synthetic data, no scratch)
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
| `builder.py` (+ `builder_run.py`, `builder_forward.py`, `stratlab/builder_*.py`) | AUTOMATIC STRATEGY BUILDER, PAPER research only, `BUILDER_ENABLED` default false: every night 02:30 IST (builder/v3; Saturday 06:00 under v1 / v2) one batch of 400 candidates from our own data (5 agents × 4 market types × 20, `stratlab/builder_grammar.py`, deterministic, never re-drawn), frozen in the hash-chained registry `<volume>/builder/registry.jsonl` BEFORE evaluation, evaluated in its own process (nice 10, ≤ 2 workers, `BUILDER_MEM_MB`, `BUILDER_TIME_LIMIT_S`, resumable) under the frozen rule `builder_rule.json` (sha256 pinned in `builder.RULE_SHA256`; DISCOVERY / VALIDATION / one FINAL look, shuffle / placebo / planted controls, DSR ≥ 0.95 at an N that never shrinks); proven → generated `<volume>/builder/catalog/builder_gen.py` + the forward lab's separate family `builder/v1` (`builder_forward_rule.json`, pinned in `builder_forward.RULE_SHA256`; `forward_lab_rule.json` untouched); never changes a rule or a batch after the fact (defects are `erratum` records); the sandbox's batches 0001-0006 are imported on first use, in order, from `builder_seed_0001.json` and `builder_seed_0002_0006.json` (sha256 pinned in `builder.SEED_SHA256` / `SEED2_SHA256`, `builder.seed_files()`; every file verified before anything is written; skipped in the sandbox registry they came from): N starts at 4,979, their 2,400 keys are never re-drawn, the server's first own batch is 0007, and batch 0003's proven strategy (`bld0003_int_hig_02_0ea14e`) is imported as a `promoted` record (`seeded: true`) that `reconcile_forward` registers with the builder/v1 forward family on the next builder run; every batch is evaluated on IMMUTABLE inputs frozen with it (`stratlab/builder_bundle.py`: panels + features + option tables, ~160 MB at full scale, and hard links to the data layer's minute files) under `<volume>/builder/inputs/`, so a timed-out / crashed run resumes on identical data; after 3 failed attempts, or when the frozen inputs are gone and the live data changed, the batch is recorded `failed` (its candidates stay in N and in the never-redraw set) and the queue moves on; the builder/v1 forward family runs on its own job (07:05 IST, `BUILDER_ENABLED`, independent of `FORWARD_LAB_ENABLED`, under the data layer's lock), which also retries a failed forward registration of a promoted strategy (`forward_registered` / `forward_register_failed` records); `BUILDER_MAX_MB` (400) disk cap (a hard-linked file counts once the data layer replaced it), caches on ephemeral `BUILDER_CACHE_DIR`; `GET /api/v2/builder`, `/api/v2/builder/batches/{id}`, dashboard page Builder. **builder/v2** (`builder_rule_v2.json`, pinned in `builder.RULE_SHA256_V2`; v1 stays valid, never rewritten): grammar bg2 = bg1 × the stock UNIVERSE (`NIFTY100` = the 97, `NIFTY200` = the 97 + the data layer's added NIFTY 200 names with complete minute data over the split, `TOP500` = daily-grid stock blocks, `INDEX`); a bg2 sentence whose universe is bg1's implicit one is its bg1 twin and is never drawn once that bg1 key was drawn; one registry and one N across v1 and v2 (record `rule_upgrade`); a pending v1 batch is evaluated under v1 (`builder.rule_by_sha`); a new v2 batch waits (`waiting_data`) until the NIFTY 200 minute backfill is complete; a promoted strategy carries its universe's frozen names (`universe_names`, traded by its builder/v1 forward entry from the data layer's store). **builder/v3** (`builder_rule_v3.json`, pinned in `builder.RULE_SHA256_V3`; v1 / v2 stay valid, never rewritten; the CURRENT rule): trade size from Rs 10 lakh capital per agent - grammar bg3 = bg2 under the size regime `c10` (`Candidate.size`, `builder_grammar.size_regime`): MIS stock trades (intraday, scalping) and stock trades at futures costs (the futures agent's stock sentences, swing shorts) are Rs 5 lakh per trade; OPTION lots follow the capital, decided at entry from the entry fills only (`stratlab/builder_options.option_lots` / `OPTION_SIZES`: long ATM and debit spreads Rs 1 lakh of net premium, iron condors Rs 1 lakh of maximum loss, naked strangles Rs 5 lakh of margin at the lab's 12 % of spot notional; capped at 1,800 NIFTY / 600 BANKNIFTY units per leg = one un-sliced order; the simulator `nsedata` is not edited: `nsedata.Position` is swapped for a sizing subclass while a re-sized option job runs); CNC delivery longs stay Rs 1 lakh and index futures 1 real lot - their earlier results ARE their Rs 10 lakh results (`stratlab/builder_engine.SIZE_TABLES` + `OPTION_SIZES` = the rule's `costs.size_rs`; `Engine.apply_rule` refuses a mismatch); a re-sized sentence has a NEW key (`['bg3', …, universe, 'c10']`: drawn again, every draw a new trial in N), a sentence of unchanged size (swing long-only, index futures) IS its bg2 sentence and is never re-drawn; the candidate carries its regime, so a pending / seeded v1 / v2 batch and a v1 / v2 promoted strategy (the seeded `bld0003_int_hig_02_0ea14e`) keep Rs 1 lakh, in the batch run and in the builder/v1 forward family; gates, splits, slippage, fees, controls, DSR unchanged; limitation recorded in the rule (`costs.capital`): for multi-session holds concurrent positions are not capped by the capital; schedule: EVERY NIGHT from 02:30 IST (job `builder_nightly` -> `builder.nightly_job_locked`, replaces Saturday 06:00; a start later than 04:30 is skipped) batch after batch, each frozen, evaluated and ingested before the next is frozen (`weekly_job` = one batch; N grows batch by batch), until the grammar has no undrawn sentence left (status `grammar_exhausted`; `builder.grammar_progress`, `undrawn` in `GET /api/v2/builder`), `BUILDER_NIGHT_BATCHES` (12) ran, 05:15 IST passed (no batch starts 05:15-15:45; an evaluation's limit ends 08:30) or a batch did not complete; data gates of a NEW v3 batch (`schedule.data_gates`, `data_wait_reason`): the NIFTY 200 minute universe AND the data layer's NSE archive backfill 2012 -> (`builder.nse_archive_status`, read-only on `builder_data`'s state with its own finality rule: every calendar day up to 10 days ago final - a day in an error status blocks and is named - and every past feed window fetched once; windows in error are reported, not waited for); the registry moves v1 -> v2 -> v3 with one `rule_upgrade` record per step |
| `builder_data.py` (+ `builder_minute.py`) | strategy-builder DATA layer, PAPER research only, `BUILDER_DATA_ENABLED` default false: NSE end-of-day archives 2012 → in `stratlab/nsedata.py`'s layout (`<volume>/builder_data/nse`; delivery, F&O parts, participant OI / volume, ban, FII stats, index closes) + flowdata / events layouts (deals, corporate actions, index membership, results, board meetings), Kite 1-minute bars 2016 → for the forward lab's universe (a sibling `forward_lab_store.Store`) plus, with `BUILDER_MINUTE_UNIVERSE=nifty200` (default; `builder_universe.py`), the added NIFTY 200 names (current list + historical NIFTY 100 members since 2016 still on Kite; `<root>/minute/universe.json`; the forward lab's own store keeps its 97), hard cap `BUILDER_DATA_MAX_MB` (2,600; worst-case volume 500 + 1,024 + 400 + 2,600 = 4,524 MB), per-run budgets, market-hours / PEAD-window guards, NSE refusal fail-safe; `nse_view` (last 400 sessions) is what the forward lab's engine reads (`builder_data.engine_env()` → `STRATLAB_NSE_DIR`); `GET /api/v2/builder/data` |

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
- Size per agent: `VALIDATED_<AUCTION|ATTENTION|PEAD>_NOTIONAL_SCALE` (default 1.0 = the lab's ₹1 L per trade) multiplies
  the global `VALIDATED_NOTIONAL_SCALE` in `ValidatedAgent._scale()`; quantities, loss stops, drawdown kills and gross
  caps of that agent scale with it, decisions never change, observers normalise back to lab size. Production gives
  each agent ₹10 L of capital through Railway variables: auction 5.0 (₹5 L per MIS trade, ≤ 10 trades at 20 % margin),
  PEAD 0.1 (₹10,000 per position, lab peak 88 open), attention 1.0, strangle 2 lots; `RISK_CAPITAL` 40 L.
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
