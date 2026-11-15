---
name: trade-review
description: >
  Review AlgoTrader Pro paper trading (core rebuild, v2 API): today's or a period's
  closed trades from the ledger per strategy (auc_fade, attn_on, vrp_winged,
  pead_h60), net P&L after costs, win rate, profit factor, paper result vs the lab
  expectation, fill quality vs the lab fill model, drift warnings, live-readiness
  progress, trading-state / approval / brake events, and flags. Read-only. Use when
  the user asks "how did we do today", "review trades", "P&L by strategy", or at the
  end of /paper-session.
---

# Trade Review — AlgoTrader Pro (v2 API)

Read-only analysis. Never place or cancel orders or change settings. API reference: `algotrader_v4/API_V2.md`.

## Setup

```bash
ALGO_URL="${ALGO_URL:-http://127.0.0.1:8000}"
# ALGO_API_KEY must be set. Never echo, log or paste it.
H="X-API-Key: ${ALGO_API_KEY}"
j() { curl -sS -m 15 -H "$H" "$ALGO_URL$1"; }
```

## 1. Collect

```bash
j /api/v2/overview                         # mode, paper_data_source, portfolio, brake
j "/api/v2/trades?since=<YYYY-MM-DD>&n=2000"   # ledger rows after costs (+ all-time stats per strategy)
j /api/v2/strategies                       # per strategy: performance.live vs lab_expectation / lab_reference, readiness
j /api/v2/drift                            # frequency, fill quality, last-20 mean vs lab band, drawdown vs lab
j "/api/v2/fills?n=500"                    # quote vs fill per order (slip_bps, kind)
j /api/v2/positions                        # still open (overnight holds are normal for attn_on / vrp_winged / pead_h60)
j /api/v2/trading-state                    # history: who changed what and why
j /api/v2/approvals                        # recent tickets (if the semi-auto mode was used)
```
Save the JSON to the scratchpad and analyse with a short `python3 -I script.py file.json`, not by eye.

The ledger (`/api/v2/trades`) is the source of truth for P&L: every row is one closed trade with `gross`, `cost`,
`net` (after costs), `exit_kind`, `note`, `slip_bps`, `lab_id`. A strangle books one row per leg — group its legs by
`oid` / `group` / expiry into one structure. `stats` rows with `kept: false` belong to strategies removed in the
rebuild: show them separately, never mix them into the kept totals.

## 2. Report

1. **Header:** period, mode (PAPER with KITE or SIM prices), net P&L after costs, trades, win rate, profit factor.
2. **Per strategy:** trades, wins, net, average per trade vs the lab's `lab_reference.net_mean_per_trade`, win % vs
   lab, drawdown vs `lab_reference.observed_max_dd`, readiness `trades / target_trades`.
3. **Fill quality:** mean entry slippage (bps) vs `lab_slippage_bps` per strategy; any fill kind other than the usual
   (`RECONCILE`, `CORPORATE_ACTION`).
4. **Drift:** each strategy's `status` and its warnings / alerts verbatim.
5. **Controls used:** trading-state changes, approvals (approved / rejected / expired), brake level changes.
6. **Flags** (only if present): open position past its scheduled exit, `orphan_rows`, strategy `errors`,
   `deviations`, `entry_block` codes that stopped entries (`daily_loss_stop`, `drawdown_kill`, `no_data:store_stale`).
7. **Reading the result:** a handful of trades says almost nothing — compare with the lab band
   (`drift.expectancy.lab_band` p5…p95) and say plainly when the sample is too small to judge.

If prices were simulated (`paper_data_source` = `SIM`) say so: the agents take no entries without Kite data.
