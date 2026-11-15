# AlgoTrader Pro — dashboard API v2

The read/write API the new dashboard is built on. Implementation: `api_v2.py` (router mounted at `/v2`; the
dashboard calls `/api/v2/...`, the `/api` prefix is stripped by the server). API version `2.0`.

Everything here is **PAPER**. No endpoint places an order or switches to LIVE.

## Conventions

| Item | Rule |
|---|---|
| Base path | `/api/v2` (same origin as the dashboard; `/v2/...` also works) |
| Auth | With `API_KEY` set on the server, **every** `/api/v2` path except `/api/v2/health` needs `X-API-Key: <key>` **or** `Authorization: Bearer <jwt>`. Get a JWT with `POST /auth/login` (form fields `username`, `password`) → `{"access_token", "token_type": "bearer", "expires_in"}`. Without `API_KEY` (local dev) nothing is checked. |
| Errors | `401 {"detail": "Unauthorized: provide X-API-Key or Bearer token"}`, `404` unknown strategy / ticket, `409` ticket not pending, `422` invalid body, `429` rate limit (writes) |
| Times | IST ISO-8601 with offset, e.g. `2026-11-16T09:15:05+05:30`; dates `YYYY-MM-DD` |
| Money | ₹, floats, after costs unless named `gross` |
| Numbers | never NaN/Infinity (they are sent as `null`) |
| Strategy names | `auc_fade`, `attn_on`, `vrp_winged`, `pead_h60` (fixed order in lists) |
| Polling | the dashboard should poll; suggested: overview 10 s, strategies 15 s, positions 10 s, the rest 60 s |

Trading states: `ACTIVE` · `REDUCE_ONLY` (no new entries, exits by rule) · `HALTED` (risk exits only; every
exit of the kept rules counts as a risk exit). The effective state of a strategy is the most restrictive of the
global state, its own state and the drawdown brake.

Entry block reasons (`entry_block`, `null` = entries allowed): `live_mode_refused`, `no_kite_data`,
`validated_only_off`, `kill_switch` (setting off), `trading_state:REDUCE_ONLY|HALTED`, `drawdown_kill`, `disabled`
(toggle off), `stopped` (bot / agent stopped), `daily_loss_stop`, `awaiting_approval`, `approval_rejected`,
`approval_expired`, `approval_cancelled`.

---

## Read endpoints

### GET /api/v2/health  (no auth)
```json
{"status": "ok", "api_version": "2.0", "mode": "PAPER", "time": "2026-11-16T09:20:01+05:30"}
```

### GET /api/v2/overview
Top bar + portfolio tiles.
```json
{
 "api_version": "2.0", "time": "2026-11-16T09:20:01+05:30", "mode": "PAPER", "paper_data_source": "KITE",
 "kite": {"connected": true, "keys": true, "auto_login": true, "account": "AB1234"},
 "market": {"open": true, "trading_day": true, "holiday": null, "next_session": "2026-11-16"},
 "bot_running": true,
 "trading_state": {"global": "ACTIVE", "brake": "ACTIVE"},
 "approval_mode": false,
 "portfolio": {"realised_today": 782.3, "unrealised": -120.5, "open_positions": 3,
               "realised_since_brake_reset": 15230.0, "drawdown": 4100.0, "drawdown_pct": 0.164,
               "brake_level": "ACTIVE", "risk_capital": 2500000.0},
 "strategies": {
   "auc_fade":   {"lab_id": "AUC-FADE-2PCT", "running": true, "entry_block": null, "state": "ACTIVE", "positions": 0},
   "attn_on":    {"lab_id": "SWING-ATTN-ON-3X", "running": true, "entry_block": null, "state": "ACTIVE", "positions": 3},
   "vrp_winged": {"lab_id": "VRP-STRANGLE-016-WINGED", "running": true, "entry_block": null, "state": "ACTIVE", "positions": 0},
   "pead_h60":   {"lab_id": "EV-PEAD-LONG-H60", "running": false, "entry_block": "kill_switch", "state": "ACTIVE", "positions": 0}
 },
 "drift_warnings": 0
}
```
`kite.detail` / `kite.error` appear when not connected. `drift_warnings` = total warnings + alerts of all strategies
(`null` if the monitor failed).

### GET /api/v2/strategies
`{"strategies": [card, card, card, card]}` — one card per strategy, same shape as below.

### GET /api/v2/strategies/{name}
A strategy card (plus `day_log`, the day's decision log, only on this endpoint).
```json
{
 "name": "auc_fade", "label": "Opening-auction fade", "lab_id": "AUC-FADE-2PCT",
 "version": "AUC-FADE-2PCT/auction1/v1", "status": "proven", "proven": true,
 "enabled": {"setting": true, "toggle": true}, "running": true, "entry_block": null,
 "trading_state": {"own": "ACTIVE", "effective": "ACTIVE"},
 "allocator_scale": 1.0,
 "today": {
   "plan": {"date": "2026-11-16", "entry_done": true, "candidates": [{"symbol": "ABB", "prev_close": 100.0,
             "auction_px": 104.0, "gap_pct": 4.0}], "planned": [["ABB", -1, 961]]},
   "positions": [{"oid": "PAPER-1A2B3C4D", "agent": "auc_fade", "symbol": "ABB", "exchange": "NSE", "product": "MIS",
                  "side": "SELL", "qty": 961, "entry_px": 104.0, "entry_ref": 104.0,
                  "opened": "2026-11-16T09:15:05+05:30", "overnight": false, "group": "", "meta": {},
                  "unrealised": 211.4}],
   "trades": [{"t": "2026-11-16T09:20:01+05:30", "strategy": "auc_fade", "version": "AUC-FADE-2PCT/auction1/v1",
               "symbol": "ABB", "side": "SELL", "qty": 961, "entry": 104.0, "exit": 103.1, "gross": 864.9,
               "cost": 82.6, "net": 782.3, "product": "MIS", "exchange": "NSE", "note": "09:20 exit",
               "oid": "PAPER-1A2B3C4D", "signal": 104.0, "slip_bps": 0.0, "lab_id": "AUC-FADE-2PCT",
               "exit_kind": "normal"}],
   "realised": 782.3, "unrealised": 0.0,
   "approval": null
 },
 "performance": {
   "live": {"n": 1, "trades": 1, "net": 782.3, "pf": null, "win_pct": 100.0, "net_bps": 78.27},
   "drawdown": 0.0,
   "lab_expectation": {"DISC": {"trades": 1090, "net_bps": 23.69, "pf": 1.43, "win_pct": 56.4},
                       "FINAL": {"trades": 525, "net_bps": 4.27, "pf": 1.08, "win_pct": 53.7}},
   "lab_reference": {"net_mean_per_trade": 141.2, "win_pct": 55.5, "trades_per_session": 2.2245,
                     "observed_max_dd": 29535.33},
   "live_mean_per_trade": 782.3
 },
 "readiness": {"ready": false, "trades": 1, "target_trades": 150, "progress": 0.007,
               "criterion": ">= 150 trades, net > 0, PF >= 1.15 on paper fills AND forward_test ...",
               "detail": {"...": "the agent's own live_readiness() block"}},
 "drift": {"status": "ok", "warnings": [], "alerts": [], "...": "see /api/v2/drift"},
 "limits": {"daily_loss_stop": 16000.0, "dd_kill": 37000.0},
 "deviations": {}, "skip_counts": {"no_signal": 87}, "flags": {}, "errors": [],
 "next_jobs": [{"id": "validated_open", "next_run": "2026-11-17T09:15:05+05:30"},
               {"id": "validated_auc_exit", "next_run": "2026-11-17T09:20:00+05:30"}]
}
```
Notes: `status` is `proven` (auc_fade, attn_on) or `candidate` (vrp_winged, pead_h60). `today.plan` is the agent's
working state and differs per strategy (show it as key/value). `lab_reference` money figures are scaled to the
strategy's current size. `readiness.progress` = trades / target (null when the target is not a trade count).
`today.approval` is today's semi-auto ticket (see approvals) or null.

### GET /api/v2/positions
```json
{"positions": [{"strategy": "attn_on", "symbol": "NHPC", "exchange": "NSE", "product": "CNC", "side": "BUY",
                "qty": 1250, "entry_px": 80.05, "entry_ref": 80.05, "opened": "2026-11-16T15:21:02+05:30",
                "overnight": true, "group": "", "unrealised": -62.5}],
 "broker_rows": [{"tradingsymbol": "NHPC", "exchange": "NSE", "product": "CNC", "quantity": 1250,
                  "average_price": 80.05, "last_price": 80.0, "pnl": -62.5, "buy_value": 100062.5, "sell_value": 0.0}],
 "orphan_rows": []}
```
`broker_rows` = the paper broker's net rows; `orphan_rows` = open rows no kept strategy owns (left by the removed
engine) — show them as a warning.

### GET /api/v2/trades?strategy=&since=&n=200
Newest first. `since` = ISO time prefix (e.g. `2026-11-01`). `n` ≤ 2000.
```json
{"count": 42, "trades": [{"t": "...", "strategy": "auc_fade", "symbol": "ABB", "side": "SELL", "qty": 961,
                          "entry": 104.0, "exit": 103.1, "gross": 864.9, "cost": 82.6, "net": 782.3, "...": "..."}],
 "stats": {"auc_fade": {"trades": 42, "wins": 24, "losses": 18, "win_rate": 57.1, "gross_pnl": 9800.0,
                        "costs": 3460.0, "net_pnl": 6340.0, "avg_net_per_trade": 150.95, "profit_factor": 1.31,
                        "max_drawdown": 2900.0, "strategy": "auc_fade", "label": "Auction fade (validated)",
                        "version": "AUC-FADE-2PCT/auction1/v1", "kept": true, "enabled": true}}}
```
Ledger rows of removed strategies may appear in `stats` with `"kept": false`.

### GET /api/v2/risk-budget
```json
{"mode": "lab sizing (scale 1.0)", "daily_vol_budget": 0.0, "cap": 1.0, "upscale_allowed": false,
 "strategies": {"auc_fade": {"lab_id": "AUC-FADE-2PCT", "active": true, "scale": 1.0, "lab_daily_std": 3897.48,
                             "lab_max_dd": 29535.33, "lab_block_dd_p99": 88051.6,
                             "hard_caps": {"daily_loss_stop": 16000.0, "dd_kill": 37000.0, "max_gross": 1000000.0}}},
 "brake": {"state": "ACTIVE", "since": null, "net": 782.3, "peak": 782.3, "drawdown": 0.0, "drawdown_pct": 0.0,
           "capital": 2500000.0, "reduce_pct": 16.0, "halt_pct": 24.0, "would_be": "ACTIVE"},
 "lab_combined": {"strategies": ["AUC-FADE-2PCT", "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED"],
                  "daily_std": 8207.68, "observed_max_dd": 132043.15,
                  "block_dd": {"p5": 73521.45, "p50": 135651.73, "p95": 251641.71, "p99": 325240.36}},
 "lab_built": "2026-11-15"}
```

### GET /api/v2/drift
Warnings only (never trades). Per strategy:
```json
{"strategies": {"attn_on": {
   "lab_id": "SWING-ATTN-ON-3X", "trades": 25, "size_factor": 1.0, "lab_available": true, "status": "warn",
   "warnings": ["last-20-trade mean ₹-2,000 below the lab band p5 ₹-180"], "alerts": [],
   "frequency": {"live_sessions": 12, "live_per_session": 2.08, "lab_per_session": 1.066, "ratio": 1.95},
   "fill_quality": {"lab_fill_model": "entry = 15:21 bar open ...", "lab_slippage_bps": 2.0,
                    "entry_slip_bps_mean": 3.1, "entry_slip_n": 25, "quote_vs_fill_bps_mean": 0.0, "quote_vs_fill_n": 25},
   "expectancy": {"last20_mean": -2000.0, "n": 20, "lab_band": {"p5": -180.46, "p50": 185.0, "p95": 731.63, "p99": 1107.91},
                  "lab_mean": 202.4},
   "drawdown": {"live": 50000.0, "lab_block_dd": {"p5": 8671.86, "p50": 13268.48, "p95": 22714.31, "p99": 27933.85},
                "lab_observed_max_dd": 12951.31}}},
 "telegram": false, "last_sent": {}, "note": "warnings only — never trades, never changes a state"}
```
`status`: `ok` | `warn` | `alert` | `error`.

### GET /api/v2/fills?strategy=&n=200
Quote-vs-fill log (paper helper fills; LIVE protected orders when they ever exist). Newest first.
```json
{"fills": [{"t": "2026-11-16T15:21:02+05:30", "agent": "attn_on", "symbol": "NHPC", "side": "BUY", "qty": 1250,
            "quote": 80.05, "fill": 80.05, "kind": "MARKETABLE", "slip_bps": 0.0, "mode": "PAPER"}]}
```
`kind`: `MARKETABLE` (limit at the touch), `AUCTION`, `SETTLEMENT`, `RECONCILE`, `CORPORATE_ACTION`, `LIVE_PROTECTED`.

### GET /api/v2/trading-state
```json
{"states": ["ACTIVE", "REDUCE_ONLY", "HALTED"],
 "global": {"state": "ACTIVE"},
 "brake": {"state": "ACTIVE"},
 "strategies": {"auc_fade": {"state": "ACTIVE", "effective": "ACTIVE"},
                "attn_on": {"state": "REDUCE_ONLY", "reason": "earnings week", "by": "user:admin",
                            "at": "2026-11-16T10:02:11+05:30", "effective": "REDUCE_ONLY"},
                "vrp_winged": {"state": "ACTIVE", "effective": "ACTIVE"},
                "pead_h60": {"state": "ACTIVE", "effective": "ACTIVE"}},
 "history": [{"at": "2026-11-16T10:02:11+05:30", "scope": "strategy:attn_on", "from": "ACTIVE",
              "to": "REDUCE_ONLY", "reason": "earnings week", "by": "user:admin"}]}
```

### GET /api/v2/approvals
```json
{"mode": false, "granularity": "one entry stage of one strategy per day",
 "deadlines": {"auc_fade": "09:16", "pead_h60": "09:16", "vrp_winged": "10:21", "attn_on": "15:24"},
 "pending": [],
 "recent": [{"id": "3f2a9c1b7d4e", "strategy": "auc_fade", "day": "2026-11-16", "stage": "entries",
             "status": "approved", "created": "2026-11-16T08:55:00+05:30", "deadline": "2026-11-16T09:16:00+05:30",
             "requested_by": "preflight", "note": "", "decided_at": "2026-11-16T09:01:40+05:30",
             "decided_by": "user:admin"}]}
```
Ticket `status`: `pending` | `approved` | `rejected` | `expired` | `cancelled`.

### GET /api/v2/shadow-registry
```json
{"enabled": false, "rule_hash": "f7d06030…", "n_active": 0, "n_family": 0, "chain_ok": true,
 "shadows": {"AUC-FADE-2PCT@auc_fade": {"family": "auction", "strategy_ref": "…", "source": "agent:auc_fade",
             "config_hash": "…", "registered_on": "2026-11-12", "forward_from": "2026-11-13", "status": "active",
             "status_reason": null, "looks": [], "trades": 12, "net": 2310.5}}}
```

### GET /api/v2/forward-lab
The forward lab (`forward_lab.py`, rule `forward_lab_rule.json`): every lab strategy run on each new session after
the close (PAPER research, no orders), judged by the pre-registered rule. Read-only; reads the status file the
daily job writes (cheap to poll, suggested 60 s).
```json
{"enabled": false, "as_of": "2026-11-16",
 "rule": {"version": "forward-lab/v1", "family_size": 1276, "alpha": 0.05, "z_required": 4.5084,
          "min_trades": 50, "min_days": 40},
 "last_run": {"date": "2026-11-16", "status": "complete", "strategies_run": 1280, "errors": 0},
 "summary": {"n_strategies": 1280, "qualified": 0, "demoted": 3, "consistency_fail": 0},
 "strategies": [
   {"id": "AUC-FADE-2PCT", "family": "opening_auction_fade", "group": "proven", "trades": 14, "days": 6,
    "net": 3120.5, "pf": 1.41, "z": null, "status": "consistent"},
   {"id": "EV-PEAD-LONG-H60", "family": "post_earnings_drift", "group": "candidate", "trades": 0, "days": 6,
    "net": 0.0, "pf": null, "z": null, "status": "consistent"},
   {"id": "tfg_ema_9_21@5m:stocks", "family": "ma_cross", "group": "forward", "trades": 41, "days": 6,
    "net": -8812.0, "pf": 0.74, "z": null, "status": "watching"}]}
```
| Field | Meaning |
|---|---|
| `enabled` | `FORWARD_LAB_ENABLED` (and PAPER) |
| `as_of` | the last session that was run COMPLETE (`null` before the first) |
| `rule.family_size` | the forward group's Bonferroni family: every forward strategy ever registered (never decreases); the rule file's initial size before the first run |
| `rule.alpha` | the family-wise level (0.05); the per-test level is `alpha / (family_size × 12 looks)` |
| `rule.z_required` | the bootstrap z a forward strategy needs at a look (`Φ⁻¹(1 − alpha/(family_size × 12))`) |
| `rule.min_trades`, `rule.min_days` | a qualification look needs ≥ 50 closed trades and ≥ 40 forward sessions |
| `last_run.status` | `never`, `complete`, `pending` (incomplete data, retried at the next session), `incomplete` (skipped for every strategy), `error` (engine failed twice; skipped), `disk_cap`, `tamper` (a ledger / registry chain or trade file failed verification), `rule_changed` (the rule file's sha256 differs from the pinned one: nothing runs) |
| `strategies[]` | one row per registered strategy: proven (AUC-FADE-2PCT, SWING-ATTN-ON-3X, VRP-STRANGLE-016-WINGED), candidate (EV-PEAD-LONG-H60), then the forward group (608 catalogue ids + 668 tfgrid combos), sorted by id |
| `trades`, `days`, `net`, `pf` | closed virtual trades, COMPLETE forward sessions since registration, net ₹ after the lab's costs and slippage at lab size, profit factor (`null` without a losing trade) |
| `z` | the z of the strategy's latest TESTED look (`null` before one) |
| `status` | forward group: `watching` · `qualified` (a look passed z ≥ z_required) · `demoted` (forward max drawdown > 1.5 × the lab's 2023-26 max drawdown; permanent). Proven / candidate: `consistent` · `inconsistent` (live daily mean below the lab's 1 % block-bootstrap band at a look with ≥ 30 trades; latched) · `demoted`. None of these switches anything on. |

### GET /api/v2/forward-lab/{id}
`404` for an unregistered id. `daily` lists every COMPLETE forward session since the strategy's registration
(zero-trade days included).
```json
{"id": "tfg_ema_9_21@5m:stocks",
 "daily": [{"date": "2026-11-16", "trades": 7, "net": -1290.4}],
 "rule_status": {"group": "forward", "family": "ma_cross", "kind": "tfgrid:5m:stocks:scalping",
                 "registered_on": "2026-11-16", "status": "watching", "status_reason": null,
                 "looks": [{"look": 1, "day": 20261211, "days": 20, "trades": 133, "net": -5120.0,
                            "mean_daily": -256.0, "tested": false, "verdict": "insufficient", "family_size": 1276,
                            "alpha": 3.27e-06, "z_required": 4.5084}],
                 "next_look_at_days": 40, "days": 21, "trades": 140, "net": -5410.2, "forward_max_dd": 9120.0,
                 "lab": {"trades": 9120, "net": -310220.0, "mean_daily": -427.3, "max_dd": 351000.0, "pf": 0.82},
                 "demotion_limit": 526500.0, "family_size": 1276, "z_required": 4.5084,
                 "chain_ok": true, "chain_error": null, "config_hash": "…", "rule_sha256": "…"}}
```
A tested forward look adds `z`, `p`, `p_boot`, `sd_boot` and `verdict` `qualified` / `not_qualified`; a consistency
look adds `band`, `lab_mean_daily` and `verdict` `consistent` / `inconsistent`. `chain_ok` = this strategy's ledger
verifies against its registry checkpoint.

### GET /api/v2/builder/data
The strategy builder's data layer (`builder_data.py`, `builder_minute.py`; `BUILDER_DATA_ENABLED`, default false →
nothing scheduled, no network call; the endpoint still reports what is on disk). One entry per dataset: `from` / `to`
(yyyymmdd, or the window tag `2024Q3` / `202410` for windowed feeds), `rows` (when known), `files`, `mb` (allocated
on disk), `last_update` / `last_error`, `progress_pct` (share of the 2012 → (2016 → for minute bars) backfill that is
final), `source`. `disk.cap_mb` = `BUILDER_DATA_MAX_MB` (default 1,800) for the whole builder directory; at the cap
nothing more is written (`cap_reached: true`, runs stop with `disk_cap`), nothing is deleted. `refused_until` is set
for 24 h after NSE refused three requests in a row (no new data meanwhile; never worked around).
Server paths / env vars for the lab readers (nsedata, flowdata, events_data, the forward-lab minute store): REBUILD.md
section 6 — `<root>/nse` = `STRATLAB_NSE_DIR` (+ `STRATLAB_BARS_DIR=<root>/nse/bars`), `<root>/nse_view` = its last 400
sessions (the forward lab's engine), `<root>/flow` = `STRATLAB_FLOW_DIR`, `<root>/events` = `STRATLAB_EVENTS_DIR`,
`<root>/minute` = a `forward_lab_store.Store` (`load_symbol` → `SymbolData`).
```json
{"enabled": false, "root": "/data/app/builder_data",
 "schedule": {"nse": "23:05 and 05:30 IST daily, 10:05 / 15:05 Sat-Sun",
              "minute": "18:40 and 06:10 IST daily, 11:40 / 14:40 Sat-Sun"},
 "budgets": {"kite_calls_per_run": 1500, "nse_requests_per_run": 2500},
 "disk": {"used_mb": 412.6, "cap_mb": 1800.0, "cap_reached": false},
 "refused_until": null, "refused_at": null,
 "datasets": {
   "delivery": {"from": 20120102, "to": 20261113, "rows": 6012345, "files": 3690, "mb": 236.1, "progress_pct": 100.0,
                "source": "nsearchives sec_bhavdata_full (2020 ->), CM bhavcopy + MTO (2012-2019)",
                "last_update": "2026-11-14T00:07:12", "last_error": null},
   "fo_parts": {"from": 20120102, "to": 20261113, "rows": null, "files": 3690, "mb": 262.0, "progress_pct": 100.0,
                "sessions_expected": 3690, "source": "…", "last_update": "…", "last_error": null},
   "participant_oi": {…}, "participant_vol": {…}, "ban": {…}, "fii_stats": {…}, "index_spot": {…},
   "deals": {"from": "2012Q1", "to": "2026Q4", "files": 120, "mb": 4.6, "progress_pct": 100.0, …},
   "corpactions": {…}, "index_membership": {…}, "events": {…},
   "minute": {"from": 20160101, "to": 20261113, "files": 12969, "mb": 655.0, "progress_pct": 100.0,
              "windows_final": 6633, "windows_total": 6633, "missing_symbols": [], "source": "Kite historical_data …",
              "last_update": "2026-11-14T06:21:40", "last_error": null},
   "nse_view": {"from": 20250402, "to": 20261113, "files": 1625, "mb": 210.4, "progress_pct": null, …}},
 "last_run": {"started": "…", "requests": 2500, "days": 301, "stopped": "budget: request budget 2500 reached",
              "http": {"ok": 2210, "missing": 290}, "view": {"sessions": 400, "from": 20250402, "to": 20261113}},
 "minute_last_run": {"calls": 1500, "windows": 1500, "sessions": 61000, "stopped": "call budget", …}}
```

### GET /api/v2/builder
The AUTOMATIC STRATEGY BUILDER (`builder.py`, rule `builder_rule.json` builder/v1, sha256 pinned in
`builder.RULE_SHA256`; `BUILDER_ENABLED`, default false → nothing scheduled; the endpoint still reports what is on
disk). Every Saturday 06:00 IST it builds one batch of 400 candidates from our own data (5 agents × 4 market types ×
20, `stratlab/builder_grammar.py`), freezes the list in a hash-chained registry before any return is computed,
evaluates it in its own process (DISCOVERY → ≤ 3 per cell to VALIDATION → one FINAL look → proof) and promotes only
proven strategies to the forward lab's separate `builder/v1` family. `n_used` = the trial count the Deflated Sharpe
Ratio uses (lab trials 2,552 + 16 development checks + every builder candidate ever evaluated + every FINAL look ever;
it never decreases). Batch rows: `discovery_passers` vs `shuffle_expected` (passers of the same candidates with their
signal dates shuffled), `validation_passers`, `final_confirmed`, `proven`, `errata` (recorded defects; a batch is never
re-evaluated). `grid`: one entry per agent × market type with `proven`, `in_forward_lab`, `qualified` (builder/v1
qualification) and `ready` (qualified, consistent, not demoted — a flag for a human; nothing trades on it).
`chain_ok: false` = the registry chain failed verification.
```json
{"enabled": false, "rule_version": "builder/v1", "rule_sha256": "a455009021b9…", "rule_error": null,
 "agents": ["intraday", "scalping", "swing", "options", "futures"],
 "market_types": ["RISING", "FALLING", "SIDEWAYS", "HIGH_VOL"],
 "grammar_sizes": {"intraday": 1176, "scalping": 414, "swing": 685, "options": 576, "futures": 394},
 "n_used": 2970, "lab_trials_count": 2552, "evaluated": 400, "final_looks": 2, "promoted": 0, "chain_ok": true,
 "disk_mb": 0.9, "last_run": {"status": "complete", "batch": "0001", "proven": [], "seconds": 81.6},
 "batches": [{"id": "0001", "created": "2026-11-15T08:17:11", "hash": "f506bc18…", "candidates_sha256": "a3875563…",
              "snapshot_end": 20261109, "status": "evaluated", "candidates": 400, "discovery_passers": 27,
              "shuffle_passers": 6, "shuffle_expected": 6, "validation_tested": 21, "validation_passers": 2,
              "final_confirmed": 0, "proven": 0, "errata": 1, "n_used": 2970,
              "best_discovery_dsr": {"id": "bld0001_sca_ris_02_3e21a1", "dsr": 0.622}}],
 "grid": [{"agent": "intraday", "mtype": "RISING", "proven": 0, "in_forward_lab": 0, "qualified": 0, "ready": 0}, …]}
```
`status` of a batch: `frozen` (registered, not evaluated), `final_open` (FINAL looks recorded, result not ingested),
`evaluated`.

### GET /api/v2/builder/batches/{id}
One batch (404 if unknown): registry hash, candidate-file sha256, data snapshot sha256 and end, generator (grammar
version + sha256), `n_before`, windows (`discovery` / `validation` / `final`, yyyymmdd), counts, `n_disc` / `n_proof`,
`v` (cross-trial variance of the DISCOVERY daily Sharpe used by the DSR), `planted` (per agent: the planted-edge
control at p = 0.65 and the coin flip, `gate_<agent>`), `errata`, and every candidate: plain-English rule `text`
(generated from the grammar), `stage` (`discovery` = failed discovery, `discovery_passed`, `validation_failed`,
`final_rejected`, `final_confirmed`, `proven`), `discovery` {trades, pf, net, h1, h2, dsr, passed, shuffle_passed,
why}, `validation` {passed, trades, pf, net, reasons} | null, `final` {confirmed, trades, pf, net} | null, `proof`
{dsr, gates, proven} | null.
```json
{"id": "0001", "status": "evaluated", "windows": {"discovery": [0, 20241109], "validation": [20241110, 20251109],
 "final": [20251110, 20261109], "T": 20261109}, "discovery_passers": 27, "shuffle_passers": 6, "validation_tested": 21,
 "validation_passers": 2, "final_confirmed": 0, "proven": [], "n_disc": 2968, "n_proof": 2970,
 "errata": [{"batch": "0001", "agent": "options", "issue": "…", "effect": "…", "action": "…"}],
 "candidates": [{"id": "bld0001_sca_sid_10_ab194a", "agent": "scalping", "mtype": "SIDEWAYS", "block": "gap_fade",
   "text": "Scalping in a SIDEWAYS market (neither rising nor falling): when the stock opens >= 1% away from yesterday's close (call auction), trade the stock against the move at 09:15 (MIS); exit 10 minutes after entry.",
   "stage": "final_rejected",
   "discovery": {"trades": 2109, "pf": 1.286, "net": 238777.0, "h1": …, "h2": …, "dsr": 0.0136, "passed": true,
                 "shuffle_passed": false, "why": []},
   "validation": {"passed": true, "trades": 222, "pf": 1.655, "net": 65227.36, "reasons": []},
   "final": {"confirmed": false, "trades": 288, "pf": 0.947, "net": -7590.51}, "proof": null}, …]}
```

### GET /api/v2/archiver
```json
{"enabled": false, "root": "/data/app/option_bars", "archive_mb": 0.0, "max_mb": 1500.0, "estimate_mb_per_week": 25}
```

### GET /api/v2/readiness
LIVE prerequisites (read-only; `ready_for_live` is always false for the kept strategies).
```json
{"mode": "PAPER", "ready_for_live": false, "blockers": ["static_ip", "strategies_live_capable"],
 "checks": [{"check": "static_ip", "ok": false, "blocker": true,
             "detail": "KITE_STATIC_IP=(unset): Kite requires a whitelisted static outbound IP for API orders (from 1 Apr 2026); register it in the Kite developer console"},
            {"check": "api_key", "ok": true, "blocker": true, "detail": "API_KEY set (protects every write endpoint)"}],
 "strategies": {"auc_fade": {"live_capable": false, "why": "PAPER-only validated agent: refuses LIVE ..."}},
 "note": "read-only; nothing here switches to LIVE. LIVE stays impossible for the kept strategies."}
```

### GET /api/v2/system
Health and jobs.
```json
{"version": "5.0.0", "api_version": "2.0", "mode": "PAPER", "time": "2026-11-16T09:20:01+05:30",
 "feed": {"running": true, "source": "KiteConnect REST+WS (paper orders)", "symbols": 22, "quoted": 22,
          "websocket": true, "last_tick_age_s": 0.4, "unknown_symbols": []},
 "kite": {"connected": true, "keys": true, "auto_login": true, "account": "AB1234"},
 "jobs": [{"id": "validated_auc_exit", "next_run": "2026-11-16T09:20:00+05:30"},
          {"id": "validated_vrp_entry", "next_run": "2026-11-16T10:20:00+05:30"}],
 "paper_book": {"enabled": true, "file": "/data/app/paper_book.json", "exists": true, "book_day": "2026-11-16",
                "last_restore": {"restored": true, "stale": true, "...": "..."}, "orphan_rows": []},
 "sebi": {"state": "ACTIVE", "kill_reason": "", "pause_reason": ""},
 "live_readiness": {"...": "as /api/v2/readiness"}}
```
`jobs` is sorted by next run (soonest first, at most 40).

---

## Write endpoints (auth always required when `API_KEY` is set)

### POST /api/v2/trading-state
Body: `{"scope": "global" | "strategy", "strategy": "attn_on" (scope strategy only), "state": "ACTIVE" | "REDUCE_ONLY" | "HALTED", "reason": "text ≤ 200"}`
```json
{"changed": {"at": "2026-11-16T10:02:11+05:30", "scope": "strategy:attn_on", "from": "ACTIVE",
             "to": "REDUCE_ONLY", "reason": "earnings week", "by": "user:admin"},
 "...": "the full GET /api/v2/trading-state body"}
```
`422` for an unknown strategy or state. HALTED also cancels pending approvals of its scope.

### POST /api/v2/strategies/{name}/enable
Body `{"enabled": true|false}` → `{"name": "vrp_winged", "enabled": false, "setting": true, "running": false}`.
`enabled` = the persisted dashboard toggle; `setting` = the server flag (`VALIDATED_<X>_ENABLED`, not changeable here).
Disabling stops new entries; open positions still exit by rule.

### POST /api/v2/approvals
Open (or return) today's ticket. Body `{"strategy": "auc_fade", "day": "2026-11-17" (optional), "deadline": "2026-11-17T09:16:00+05:30" (optional), "note": ""}` → `{"ticket": {...}, ...GET /api/v2/approvals}`.

### POST /api/v2/approvals/{id}/approve · POST /api/v2/approvals/{id}/reject
Body `{"note": ""}` (optional) → `{"ticket": {..., "status": "approved"}, ...}`. `404` unknown id, `409` not pending
(already decided / expired / cancelled).

### POST /api/v2/approvals/mode
Body `{"on": true|false}` → the GET /api/v2/approvals body. Persisted (overrides `APPROVAL_MODE`). Default OFF.

### POST /api/v2/risk/brake/reset
Body `{"reason": "manual reset"}` (optional) → the GET /api/v2/risk-budget body with `brake.state` `ACTIVE` and the
drawdown measured afresh from now.

---

## Other endpoints the dashboard may use (unchanged, not versioned)

| Method / path | Use |
|---|---|
| `POST /auth/login` | JWT login (form `username`, `password`) |
| `GET /auth/me` | who am I |
| `GET /auth/login-url`, `GET /auth/kite/callback`, `POST /auth/kite/auto-login`, `GET /auth/kite/auto-login/status`, `POST /auth/token` | Kite login |
| `GET/POST/DELETE /settings/credentials…`, `POST /settings/app-password` | key vault, dashboard password |
| `POST /bot/start`, `POST /bot/stop`, `GET /bot/status` | operator Start / Stop (Stop persists until Start) |
| `POST /agents/{name}/pause`, `POST /agents/{name}/resume` | pause for the day |
| `POST /orders/squareoff` | flatten leftover MIS / NRML paper rows now |
| `GET /sebi/status`, `POST /sebi/kill-switch`, `POST /sebi/pause`, `POST /sebi/resume`, `POST /sebi/reset-kill-switch` | SEBI kill switch |
| `GET /health`, `GET /validated`, `GET /forward-test`, `GET /shadow-registry`, `GET /performance/*`, `GET /paper-book/status`, `GET /market/live` | legacy reads kept for scripts |
