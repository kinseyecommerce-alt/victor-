# Shadow-incubation registry (`shadow_registry.py`)

Forward testing that cannot be mined. A *shadow* is a strategy that is frozen, registered, and then only watched
on live market data, in PAPER. The registry does not trade: it copies paper trades that already exist (the
opening-fade forward test and the three validated agents) into tamper-evident ledgers and applies a promotion
rule that was written down before any forward result existed.

Written 2026-11-14, rule version `shadow-promotion/v1`. At the time of writing no shadow had a forward trade.

## What a shadow is

| field | meaning |
|---|---|
| `id` | unique; `forward:<variant>/<rule>` or `<agent>@<VERSION>` |
| `strategy_ref` | what is watched (`forward_test:AUC-FADE-3PCT/live`, `agents.validated_auction:AuctionFadeAgent`) |
| `config` + `config_hash` | the frozen parameters (sha256 of canonical JSON); validated agents include `LAB_FINGERPRINT` |
| `registered_on`, `forward_from` | registration date; trades count only when their exit date is ≥ registration + 1 + `max_hold_days`, so no trade decided before registration can enter (no backfill) |
| ledger | `data/shadow/ledgers/<id>.jsonl`: append-only JSONL; every record has `seq`, `prev_hash`, `hash` (sha256 of the record without `hash`). The first record is a `genesis` that carries the config hash |

`data/shadow/registry.jsonl` is a second hash-chained file holding registrations, retirements, look results,
status changes (promotion / demotion) and checkpoints of each ledger's head. `verify_chain(path)` detects an
edited field, a deleted / inserted / reordered line or a re-hashed edit; the checkpoints also catch a truncated
tail. A ledger whose chain fails is not evaluated.

Re-registering an id with a different config raises `FrozenConfigError`: a changed strategy is a new shadow (new
id, new trials-ledger row, one more test in the Bonferroni family). A validated agent whose lab code changed
without a `VERSION` bump is reported as `config_drift` and its trades are no longer copied.

## Pre-registered promotion rule (`PROMOTION_RULE`, do not tune)

- Looks at **50, 100 and 200** forward trades (a strangle structure's four legs count as one trade). No promotion
  is possible before 50 trades, and a look sample must also span **≥ 20 distinct trading days**. The look sample
  is every trade up to and including the first day on which both are met (deterministic given the ledger).
- Each look runs **exactly once**; its result is written to the registry and never recomputed.
- Test: one-sided **bootstrap** of the mean **daily** forward net P&L (trading-day totals, so several trades on one
  day are not treated as independent). H0: mean ≤ 0. Resample the centred daily totals with replacement,
  B = 20 000, seed = first 8 bytes of sha256(`<id>|look<k>`); p = (1 + #{boot mean ≥ observed}) / (B + 1).
- Promote when the mean is positive and **p ≤ 0.05 / (n_active × 3)** — **Bonferroni** over the shadows active
  at the look (incubating or promotion-eligible; not demoted or retired) and over the three looks.
  There is no profit-factor threshold.
- `promotion_eligible` is a recommendation recorded in the registry. It never switches anything on; changing what
  trades (or going LIVE) stays a human decision.

## Demotion rule

At every evaluation: if the forward maximum drawdown of daily net totals exceeds **1.5 × the backtest maximum
drawdown** (daily totals over the lab period, at lab size), the shadow is `demoted` (permanent; a revised strategy
needs a new id). Forward P&L is normalised to lab size when the config gives `per_trade_notional` (₹1 L stock
trades). References: AUC-FADE-3PCT ₹24,605, AUC-FADE-2PCT ₹29,535, SWING-ATTN-ON-3X ₹12,951,
VRP-STRANGLE-016-WINGED ₹129,760 (2 lots, raw rupees). AUC-FADE-1TO3PCT was never run in the lab: no reference,
so its demotion cannot fire (the evaluation says so).

Retiring a shadow removes it from the active set; the reason is recorded in the registry chain, so retiring losers
to shrink the Bonferroni family is visible.

## Enrolled shadows

- Forward test (`forward_test.py`): `forward:AUC-FADE-{3PCT,2PCT,1TO3PCT}/{final,live}` (6 shadows). Their trades
  are copied as forward_test built them (auction ± 2.1 bps, 09:20 LTP ± 2.1 bps, 2026 MIS costs).
- Validated agents: `auc_fade@<VERSION>` (the configured variant), `attn_on@<VERSION>`, `vrp_winged@<VERSION>`,
  copied from the performance ledger rows of that `VERSION` (the agents' own paper fills: marketable LIMIT at the
  quote touch, LTP without one).

`paper_fill()` / `record_paper_trade()` give a new shadow the same fill model: BUY at the ask / SELL at the bid,
else the reference ± 2.1 bps half-spread, with `costs.trade_cost`.

## Trials ledger

On registration a row `id,family,protocol=shadow,kind,registered,config_hash,source,note` is appended to the lab's
trials ledger **if the file exists** (`SHADOW_TRIALS_CSV`, else `$STRATLAB_CACHE/trials.csv`, else the lab scratch
`stratlab_cache/trials.csv`); columns of an existing header are respected, the file is never created.

## Running it

Off by default. `SHADOW_REGISTRY_ENABLED=true` (PAPER only) adds the `shadow_registry` job to
`platform_scheduler` at 17:05 IST every day (after the strangle's last settlement retry): enrol, copy, evaluate,
checkpoint. The job re-checks the flag and the mode. `GET /shadow-registry` shows the state (read-only). Nothing
in the module places, modifies or cancels an order (`test_shadow_registry.py` checks the source and runs the job
with every order entry point booby-trapped).

```
cd algotrader_v4 && python test_shadow_registry.py
```
