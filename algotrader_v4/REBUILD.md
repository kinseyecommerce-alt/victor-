# Core rebuild (Nov 2026)

Decision: "Re-create entire app with new strategies and trading engine and remove old" → **clean rebuild on core**,
strategies **"proven + candidates"**, everything **PAPER only**.

**22 Nov 2026: one definition of "proven"** (`agents/strategy_agents.PROVEN` / `WHY_NOT_PROVEN`): Deflated Sharpe
>= 0.95 at the current trial count N, executable as tested, and positive at the costs of the instrument actually
held. None of the four qualifies, so all four are candidates (reasons in the API's `why_not_proven`). The same day:
paper fills pay `PAPER_SLIPPAGE_BPS` (production 5) except settlements / reconciliations / corporate actions; the
attention swing books CNC delivery costs; every agent's readiness also needs the powered gate
(`ValidatedAgent.powered_gate`: n, one-sided t >= 2.33, gross >= 2 x cost; the strangle counts cycles).

| Strategy | Key | Status | Default |
|---|---|---|---|
| AUC-FADE-2PCT (opening-auction fade; variant via `VALIDATED_AUCTION_VARIANT`, code default 3PCT) | `auc_fade` | candidate (proven before 22 Nov 2026) | on |
| SWING-ATTN-ON-3X (attention overnight long) | `attn_on` | candidate (proven before 22 Nov 2026) | on |
| VRP-STRANGLE-016-WINGED (NIFTY weekly winged strangle) | `vrp_winged` | candidate | on |
| EV-PEAD-LONG-H60 (post-earnings drift) | `pead_h60` | candidate | **off** (`VALIDATED_PEAD_ENABLED=false`) |

The four agents' rule implementations are unchanged (their lab-parity tests pass unchanged). The only edits in
`agents/validated_*.py`: the entry size is multiplied by the risk-budget factor (`alloc_scale()`, exactly 1.0 by
default; the strangle's lots go through `alloc_lots()`, 2 → 2), and PEAD's evening *planning* step calls
`entry_block_reason(approval=False)` (the semi-auto approval is checked at its 09:15 entry instead).

## 1. Removed

### Files (62)
Legacy strategies and their engine: `agents/intraday_v2.py`, `agents/research_agents.py`, `agents/master_agent.py`,
`master_agent_v5.py`, the legacy agents inside `agents/strategy_agents.py` (now only the registry), `strategies/`,
`strategy_signals.py`, `signal_engine.py`, `gamma_scalp.py`, `momentum_rotation.py`, `market_filter.py`,
`multi_timeframe.py`, `opportunity_scanner.py`, `symbol_scanner.py`, `universe.py`, `nifty100.py`.
AI / LLM: `ai_trader.py`, `ai_replay.py`, `llm_provider.py`, `claude_trade_gate.py`, `debate_gate.py`,
`allocator.py` (LLM allocator), `adaptive_engine.py`, `historical_learner.py`, `trade_memory.py`,
`research_loop.py`, `optimizer.py`, `regime_v2.py`, `market_regime.py`, `market_context.py`.
Old execution / risk of the tick book: `atomic_bracket.py`, `trailing_sl_engine.py`, `order_guard.py`,
`risk_manager.py`, `portfolio_risk.py`, `correlation_guard.py`, `execution.py`.
Backtest / replay engines: `backtest_engine.py`, `replay_backtest.py`, `auto_backtest_runner.py`, `bg_compute.py`,
`bg_worker.py`, `daily_data.py`.
Data / intelligence: `market_data.py` (NSE quote/yfinance/GBM simulator — its NSE JSON client moved to
`nse_http.py`), `truedata_client.py`, `levels_engine.py`, `institutional_flow.py`, `iv_surface.py`,
`options_flow.py`, `options_intelligence.py`, `pre_market_report.py`.
Other: `n8n_bridge.py`, `phase4_trade_lifecycle.py`, `phase5_intelligence.py`, `phase6_resilience.py`,
`paper_trade_sim.py`, `startup.py`, `run_demo.sh`, `static/dashboard.html`, `dashboard_e2e.py`, `playwright_e2e.py`,
`test_buttons.py`, `test_sim_orders_flow.py`.
Slimmed (kept file, legacy parts removed): `main.py` (rewritten), `platform_scheduler.py`, `agents/base_agent.py`,
`agents/strategy_agents.py`, `paper_book.py`, `performance.py` (no pruning, no yfinance), `tick_engine.py` (price feed
only: no candles/indicators/strategy queues/simulator), `kite_client.py` (GTT and ticker helpers), `config.py`,
`ist_clock.py` (demo clock), `bot_state.py`, `event_calendar.py`, `secret_store.py`, `test_pipeline.py`.

### Endpoints (80 removed; 52 old paths kept + the v2 API)
`POST /adaptive/review`, `GET /adaptive/status`, `POST /agent/chat|close-all|cycle|start|stop`,
`GET /agent/log|status`, `POST /ai/providers/test`, `GET /allocator`, `POST /allocator/run`,
`GET|POST /backtest/ai-replay`, `GET /backtest/approved/{strategy}`, `GET /backtest/bars`, `POST /backtest/batch`,
`POST /backtest/compare`, `GET /backtest/equity/{symbol}/{strategy}`, `GET|POST /backtest/gate-replay`,
`GET /backtest/momentum`, `POST /backtest/replay`, `GET /backtest/replay/last`, `POST /backtest/run`,
`GET /backtest/trades/{symbol}/{strategy}`, `POST /backtest/weekly`, `GET /bot/directives`, `GET /brackets`,
`POST /brackets/manual`, `GET /brackets/{id}`, `GET /dashboard/summary`, `GET /gate/log`, `GET /market-type`,
`POST /market-type/refresh`, `PUT /market-type/research`, `GET /market/context/{symbol}`,
`GET /market/live/{symbol}`, `GET /market/opportunities`, `GET /market/option-chain/{symbol}`, `GET /market/status`,
`GET /market/trend-filter`, `GET /momentum/rankings`, `POST /momentum/rebalance`,
`POST /optimize/params/{agent}/reset`, `POST /optimize/run`, `GET /optimize/status`, `POST /orders/place`,
`DELETE /orders/{order_id}`, `POST /performance/strategies/{name}/enable|disable`, `GET /regime/history|plans|status`,
`POST /regime/refresh`, `GET /research/evidence|log`, `POST /research/run`, `GET /risk/portfolio`, `GET /risk/status`,
`PATCH /risk/update`, `GET|PATCH /settings/capital-allocation`, `GET|PATCH /settings/trading-limits`,
`POST /signals/generate`, `POST /simulate/price-tick`, `GET /symbols/criteria|selected|universe`,
`POST /symbols/scan`, `GET /symbols/scores/{strategy}`, `GET /trailing-sl/configs|status`,
`PATCH /trailing-sl/config`, `GET /universe`, `POST /universe/refresh`, `POST /webhooks/n8n`, `WS /ws`,
`WS /ws-proxy/ws`.

Kept (scripts and the operator still use them): `/health`, `/validated`, `/shadow-registry`, `/forward-test`,
`/backtest/history-bars` (stratlab data-fetch scripts), auth + Kite login + vault + app password, `/bot/start|stop|status`
(no body needed now), `/agents…`, `/settings/agent-enables`, `/portfolio/positions|orders`, `/orders/squareoff`,
`/paper-book/status`, `/market/live`, `/sebi/*`, `/performance/scoreboard|strategies|execution|trades|snapshot`,
`/status/connections`, `/config/validate`, `/docs`, `/redoc`, `/login`, `/dashboard`.

### Config fields
Removed (≈140): the AI trader (`AI_AGENT_*`, `AI_PROVIDER_*`, `ANTHROPIC_API_KEY`, `OPENAI_*`), trade gates
(`USE_CLAUDE_TRADE_GATE`, `CLAUDE_GATE_*`, `GATE_*`, debate), regime (`USE_REGIME_ROUTING`, `REGIME_*`), MTF/Kelly,
allocator, master review, backtest gate (`BT_*`), trade limits / capital buckets of the tick agents
(`MAX_TRADES_*`, `*_CAPITAL_PCT`, `MAX_*_POSITIONS`, `INTRADAY_LEVERAGE`, `STOP_LOSS_PCT`, `TARGET_PCT`, …),
portfolio risk (`RISK_PER_TRADE_PCT`, `PORTFOLIO_VAR_*`, `VAR_*`, `FNO_*`, `MAX_PORTFOLIO_HEAT`), pruning
(`PRUNE_*`), trend filter / momentum (`TREND_FILTER_*`, `MOMENTUM_*`), auto-start of the tick agents
(`AUTO_START_STRATEGIES`, `USE_NIFTY100_WATCHLIST`, `USE_ROLLING_UNIVERSE`, `UNIVERSE_*`, `AI_AGENT_AUTO_START`),
TrueData, n8n, `DEMO_MODE`, `INDICATOR_MIN_INTERVAL_S`, `USE_LIMIT_ENTRIES`, `ENTRY_LIMIT_BUFFER_PCT`,
`SQUAREOFF_CLOSES_TRACKED`, `USD_INR`, …. **Unknown variables are ignored** (`extra="ignore"`), so a deployment that
still sets them starts normally. `AUTO_START_WATCHLIST` stays (price-feed warm list).
Added: `RISK_BUDGET_DAILY_VOL`, `RISK_BUDGET_ALLOW_UPSCALE`, `RISK_BUDGET_MAX_SCALE`, `RISK_CAPITAL`,
`RISK_BRAKE_REDUCE_PCT`, `RISK_BRAKE_HALT_PCT`, `APPROVAL_MODE`, `DRIFT_TELEGRAM`, `ORDER_PROTECTION_BAND_PCT`,
`KITE_STATIC_IP`.

### Scheduler jobs and Telegram
Removed: `pre_market` (09:00 report), `morning_data` (09:10 levels/events/institutional/correlation/opportunity
warm-up), `allocate_morning|midday` (LLM allocator), `auto_start` + `auto_start_catch_up` (09:16 start of the tick
agents / AI trader), `regime_preopen|first_hour`, `universe_refresh`, `options_cache`, master's
`master_review` (1-min), `nightly_adaptive`, `weekly_backtest|optimise|research|memory_synthesis`,
`monthly_momentum`. Their Telegram messages (pre-market report, regime changes, universe, AI start, momentum,
"bot auto-started") go with them; the master's Telegram commands never existed beyond these notifications.
Kept / moved into `platform_scheduler`: Kite login 08:45/08:55/09:05 (+ startup), validated jobs (unchanged times),
forward test, 09:15 daily reset (was master's), square-off at `SQUAREOFF_TIME` (was master's, PAPER), 15:35
scoreboard (+ its Telegram), shadow registry, option-bar archiver. Added: `risk_brake` (every minute),
`drift_monitor` (17:15).

### Requirements
Dropped: `yfinance`, `truedata-ws`, `slowapi`, `python-jose`, `passlib`, `anthropic`, `openai`, `ta`, `matplotlib`,
`aiofiles`, `playwright`. Added explicitly (were transitive): `cryptography` (vault), `requests` (PEAD / stratlab
NSE fetches). The Dockerfile is unchanged apart from three env lines of removed files.

## 2. Kept core

Kite (`kite_client.py`, `kite_auto_login.py`, `kite_ticker.py`, `tick_engine.py` price feed), paper broker and
`paper_book.py`, `sebi_compliance.py` (pre-order checks, kill switch), `performance.py` ledger, `costs.py`,
`forward_test.py`, `shadow_registry.py`, `option_bar_archiver.py`, `validated_data.py`, `validated_parity.py`,
`pead_data.py`, `pead_parity.py`, `event_calendar.py` (+ `nse_http.py`), `options_resolver.py`, `options_replay.py`,
`greeks_engine.py` (the strangle's lab engine), `ist_clock.py`, `auth.py`, `secret_store.py`, `bot_state.py`,
`agents/validated_*.py`, `stratlab/` (untouched).

## 3. Persistent state on the volume (loads unchanged; nothing deleted or rewritten)

| File | Old content | Behaviour |
|---|---|---|
| `performance.json` | trades of every strategy, days, pruned, enabled_since | loaded as is; rows of removed strategies stay (shown with `kept: false`); unknown top-level keys and `pruned` / `enabled_since` are written back unchanged |
| `agent_enables.json` | intraday / fno / swing / scalping + validated keys | legacy keys ignored but written back unchanged on the next toggle |
| `bot_stopped.json` | operator Stop | unchanged semantics (persists until `/bot/start`) |
| `paper_book.json` | schema 1 with `tracked`, `tsl`, `ai`, `risk`, `guard`, legacy agents' day state | kept blocks restored (paper rows, GTT records, validated books, kept agents' day state); removed blocks ignored; the first such restore writes a byte-identical copy `paper_book.pre-rebuild.json`; legacy paper rows are listed as `orphan_rows` (an MIS/NRML orphan is flattened by the 15:15 square-off as before; a CNC orphan stays) |
| shadow ledgers, PEAD store, option-bar archive, session store, forward test ledger, vault | — | untouched; the vault may hold keys of removed features (Anthropic/OpenAI/TrueData): they stay in the vault and are simply not applied |
| new: `trading_state.json`, `approvals.json`, `drift_state.json`, `fill_log.jsonl` | — | created next to the ledger on first use |

Tested in `test_rebuild.py` (old-format paper book same-day and earlier-session, corrupt / foreign snapshot, legacy
enables, old ledger, vault with removed keys).

## 4. Engine upgrades

| Module | What | Default (Monday unchanged because…) |
|---|---|---|
| `risk_budget.py` | per-strategy size factor = budget share ÷ lab daily P&L std (`lab_risk.json`), capped at 1.0 (or `RISK_BUDGET_MAX_SCALE` if `RISK_BUDGET_ALLOW_UPSCALE`); agents' own hard caps stay upper bounds; latching portfolio drawdown brake on realised net of the kept strategies: ≥ `RISK_BRAKE_REDUCE_PCT` % of `RISK_CAPITAL` → REDUCE_ONLY, ≥ `RISK_BRAKE_HALT_PCT` % → HALTED; reset via API | `RISK_BUDGET_DAILY_VOL=0` → scale exactly 1.0 = lab sizing; brake 16 % / 24 % of ₹25 L = ₹4 L / ₹6 L, above the lab's combined block-bootstrap p99 drawdown (₹3.25 L), so a lab-like path never trips it |
| `trading_state.py` | global + per-strategy `ACTIVE / REDUCE_ONLY / HALTED`, persisted, logged (who / why), most restrictive of global, strategy and brake; enforced in `ValidatedAgent.entry_block_reason()` (every entry stage) and in the paper order helpers (an order that opens / enlarges a position is refused when not ACTIVE; reducing orders, settlements, reconciliations and corporate-action bookings always pass) | everything ACTIVE |
| `approvals.py` | semi-auto mode: a strategy's entries of the day need an approved ticket (opened at 08:55 or on demand; deadline = its entry time); HALTED cancels pending tickets; exits never need approval | `APPROVAL_MODE=false` |
| `drift_monitor.py` | per strategy: trade frequency vs lab, entry slippage vs the lab fill model, quote-vs-fill log, last-20-trade mean vs the lab bootstrap band, live drawdown vs block-bootstrap p95/p99; warnings only; Telegram at most once per strategy per day | `DRIFT_TELEGRAM=false` (no new messages) |
| `kite_client.py` LIVE path | a MARKET order goes out as a marketable LIMIT within `ORDER_PROTECTION_BAND_PCT` (0.5 %) of the touch; quote vs fill logged | PAPER path untouched; the paper helpers only log quote vs fill |
| `live_readiness.py` | LIVE prerequisites report (static IP in `KITE_STATIC_IP`, API key, JWT secret, kill-switch secret, Kite keys, …); logged at startup; `/api/v2/readiness` | always `ready_for_live: false`: the kept strategies refuse LIVE |
| `api_v2.py` | the dashboard API (`API_V2.md`) | read-only except the documented write endpoints |

### Trading states — precise definition
* **ACTIVE**: entries and exits by the rule.
* **REDUCE_ONLY**: entry stages skip (`trading_state:REDUCE_ONLY`); any order that would open or enlarge a position
  is refused (`TradingStateRefused`); every exit, stop, settlement, square-off and reconciliation runs by the rule.
* **HALTED**: as REDUCE_ONLY, and pending approvals are cancelled; the order path accepts **risk exits only** (orders
  that reduce an open position). All exits of the four kept rules are risk exits (time stops, stop-losses, expiry
  settlement, the MIS square-off), so positions still leave at their tested exit — holding them past it would add
  risk the lab never measured.

### Streaming half-candle hazard
`stratlab.indicators.build_sessions` treats the last candle of the data it is given as complete. **No kept live path
uses it** (checked by `test_rebuild.test_streaming_half_candle_hazard_not_on_a_live_path`, which greps every kept
module). The only live minute-bar store, `validated_data.refresh_store`, drops today's session before 15:35 (tested
at 14:00 vs 15:40); the strangle settles only on the full 15:00-15:29 window; PEAD reads daily bars of completed
sessions; the tick feed no longer builds candles at all.

### Static IP for LIVE (Kite, from 1 Apr 2026)
Kite accepts API orders only from a static outbound IP registered in the developer console. Railway's default egress
is not static: LIVE needs Railway's static-egress add-on or a fixed-IP host (AWS Mumbai + Elastic IP, `deploy/AWS.md`).
Put the IP in `KITE_STATIC_IP`; `/api/v2/readiness` and the startup log report it. Infrastructure is not set up by
this rebuild, and LIVE remains impossible for these strategies (their guards refuse LIVE).

## 5. Behaviour differences the operator should know

* The price feed subscribes only the warm watchlist (first 20 NIFTY 50 names + the two indices, or
  `AUTO_START_WATCHLIST`). The old 09:16 auto-start additionally subscribed the legacy agents' watchlist; for symbols
  that are no longer streamed, a paper marketable order fills at the quote's touch (the price the rule saw) instead of
  the better of quote and stream touch. Auction-fade entries (official open) and settlements are unaffected.
* An operator Stop now persists until `/bot/start` (the old 09:16 auto-start of the tick agents silently cleared it).
* The 15:15 square-off runs every trading day in PAPER (before: only when the old master had been started).
* The dashboard (`frontend/`) was rebuilt on the v2 API (mobile-first; Overview, Strategies, Positions & trades, Risk & controls, Forward lab, Builder, Shadow registry, System). It polls instead of using the removed `/ws`; see TESTING.md.

## 6. Strategy-builder data layer (Part 1, `builder_data.py` + `builder_minute.py`) — server paths for the builder engine

Flag `BUILDER_DATA_ENABLED` (default false: no job, no network). PAPER only, read-only towards trading. Root
`<root>` = `BUILDER_DATA_DIR`, else `<dir of PERFORMANCE_FILE>/builder_data` (the Railway volume). Everything is read
through the lab's EXISTING readers — no new format:

| Data | Server path | How lab code reads it |
|---|---|---|
| NSE EOD archives 2012 → (delivery, F&O parts fut / optidx / optstk aggregates, expiry-eve `optstk_full`, participant OI / volume, ban, FII stats, index closes) | `<root>/nse` | `STRATLAB_NSE_DIR=<root>/nse STRATLAB_BARS_DIR=<root>/nse/bars` → `stratlab.nsedata` unchanged (or `stratlab.oldera_fo.use(<root>/nse)`) |
| the last 400 sessions of the same (what the forward lab's engine reads) | `<root>/nse_view` | `builder_data.engine_env()` sets `STRATLAB_NSE_DIR` / `STRATLAB_BARS_DIR` for the engine process unless the operator set them |
| deals, corporate actions, index membership (NIFTY 50 / NEXT 50 / 100), FII stats | `<root>/flow` | `STRATLAB_FLOW_DIR=<root>/flow` → `stratlab.flowdata` (or `flowdata.use(path)`) |
| results / board-meeting calendars | `<root>/events` | `STRATLAB_EVENTS_DIR=<root>/events` → `stratlab.events_data` (or `events_data.use(path)`) |
| 1-minute bars 2016 → of `forward_test.UNIVERSE` + `_NIFTY50_min`, `_NIFTYBANK_min`; daily `_NIFTY50_day`, `_VIX_day` | `<root>/minute` | `forward_lab_store.Store(<root>/minute).load_symbol(sym, upto=day)` → the lab's `SymbolData`; `load_aux(name)`; `write_lab_cache(dir, upto)` for the daily series |

Caveats a reader must know: nsedata builds an uncompressed `.npz` cache inside the NSE dir it reads (`_cache/`); for the
whole 2012 → archive that would be ~2 GB, so on the server read `<root>/nse_view` (cache ~0.2 GB) or build the cache of
a copy on ephemeral disk. Per-strike stock-option chains exist only on the 1-4 calendar days before a stock expiry
(nearest expiry). Delivery `bars_symbol` maps the 97 lab names (+ ZOMATO / MCDOWELL-N / MACROTECH) exactly like the
lab's 3-year file. F&O `lot` is NaN before 2024 (the old bhavcopy has none). Index membership has month-end precision
before the server's first daily list (no press-release dates: `announce_source = "none"`). Corporate-action
`known_date` = the session before the ex-date unless the server saw the action earlier. Status:
`GET /api/v2/builder/data` (API_V2.md).

Minute UNIVERSE (`BUILDER_MINUTE_UNIVERSE`, `builder_universe.py`): `nifty100` = the forward lab's 97 + 2 indices
(as before); `nifty200` (default) = those, first and unchanged, then the added names: the current NIFTY 200 list
(`<root>/flow/raw/index/daily/NIFTY200/`, nsearchives `ind_nifty200list.csv`, fetched daily from the first run on; a
committed snapshot of 2026-11-16 until then) + every historical NIFTY 100 member since 2016 (`index_members.csv.gz`).
`<root>/minute/universe.json` records base / extra / missing-on-Kite; the builder reads it. NIFTY 200 membership is NOT
point-in-time (NSE Indices' month-end files cover NIFTY 50 / NEXT 50 only) and delisted names are not on Kite:
survivorship bias of the added names, recorded with every batch. Disk: ~51 KB per instrument-month (measured; re-measured
live in the API) → ≤ ~0.9 GB for ~137 added names with full history; cap 2,600 MB.

## 7. Automatic strategy builder (Part 2, `builder.py` + `builder_run.py` + `builder_forward.py` + `stratlab/builder_*.py`)

Flag `BUILDER_ENABLED` (default false: no job). Every night 02:30 IST (builder/v3; Saturday 06:00 before): freeze 400 candidates (registry
`<volume>/builder/registry.jsonl`), evaluate them in `python -m builder_run` (nice 10, `BUILDER_JOBS` <= 2,
`BUILDER_MEM_MB` 4096 address space per evaluation process, `BUILDER_SNAPSHOT_MEM_MB` 6144 for the snapshot
process that builds the full daily panel (one thread per numeric library, two malloc arenas), `BUILDER_TIME_LIMIT_S` 7200, resumable), ingest; proven strategies go
to the forward lab's separate family builder/v1 (processed at the end of `forward_lab.daily_job`). Reads the data layer
of section 6 through `stratlab/builder_server_panels.ServerSource` (`BUILDER_DATA_DIR`); its derived panels and the
compact option tables are cached on ephemeral disk (`BUILDER_CACHE_DIR`, default `/tmp/builder_cache`), never on the
volume; the builder directory is capped at `BUILDER_MAX_MB` (400). Each frozen batch keeps its own immutable inputs under `<volume>/builder/inputs/<id>/` (panels, features, option tables ~160 MB at full scale, plus hard links to the data layer's minute files: free until the data layer replaces a file) and is always evaluated on them, so a retry sees identical data; they are deleted when the batch is ingested or failed. A batch that fails 3 times, or whose frozen inputs are lost while the live data changed, is closed `failed` (registry record `batch_failed`; its candidates stay counted in N and are never re-drawn). The builder/v1 forward family has its own job, `builder_forward`, 07:05 IST daily under `BUILDER_ENABLED` (it does not need `FORWARD_LAB_ENABLED`), holding the data layer's lock; it and the weekly job register any promoted strategy whose forward registration failed. The weekly job holds the data layer's lock while it
freezes / evaluates. Rules: `builder_rule.json` (pinned in `builder.RULE_SHA256`), `builder_forward_rule.json`
(pinned in `builder_forward.RULE_SHA256`). Results / design: `stratlab/README.md` (last section), API: API_V2.md.

builder/v2 (`builder_rule_v2.json`, pinned in `builder.RULE_SHA256_V2`, written next to v1, which is never rewritten):
grammar bg2 adds the stock UNIVERSE (NIFTY100 / NIFTY200 / TOP500 / INDEX) to every sentence; gates, controls, splits,
costs and DSR are v1's. On the first weekly run after the deploy the registry appends `rule_upgrade` (N, candidates and
FINAL looks at that moment; N continues from there). A v1 batch still pending is resumed under v1; new batches are v2,
never re-draw a v1 key or its bg2 twin, and wait (`waiting_data`) until the NIFTY 200 minute backfill is complete.

builder/v3 (`builder_rule_v3.json`, pinned in `builder.RULE_SHA256_V3`, the current rule; v1 / v2 never rewritten):
the test trade size follows the owner's capital of Rs 10 lakh per agent (grammar bg3 = bg2 under the size regime
`c10`): Rs 5 lakh per MIS stock trade and per stock trade at futures costs; option lots from the capital at entry
(Rs 1 lakh of net premium for long options / debit spreads, Rs 1 lakh of maximum loss for iron condors, Rs 5 lakh of
margin at 12 % of spot notional for naked strangles, at most one un-sliced order per leg); Rs 1 lakh per CNC delivery
long and 1 lot per index future (unchanged: their earlier results are their Rs 10 lakh results); every re-sized
sentence is a new candidate counted in N, a sentence of unchanged size is never re-drawn; a v1 / v2 batch or promoted strategy keeps Rs 1 lakh. The job now runs EVERY NIGHT from
02:30 IST (`builder_nightly` -> `builder.nightly_job_locked`; Saturday 06:00 before), still only with
`BUILDER_ENABLED` and outside market hours: batch after batch (each frozen, evaluated and ingested before the next),
until the grammar has no undrawn sentence left, `BUILDER_NIGHT_BATCHES` (12) ran, 05:15 IST passed or a batch did not
complete. A NEW v3 batch waits (`waiting_data`) for the NIFTY 200 minute universe and for the data layer's NSE archive
backfill 2012 -> (`builder.nse_archive_status`: every calendar day up to 10 days ago final, every past feed window
fetched once). `GET /api/v2/builder` shows `undrawn` (sentences still to test, total and per agent) and the last
night's record. A registry at v1 or v2 moves forward with one `rule_upgrade` record per step.
