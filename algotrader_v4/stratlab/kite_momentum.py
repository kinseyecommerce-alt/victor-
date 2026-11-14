"""
stratlab/kite_momentum.py — Zerodha Varsity's price-momentum portfolio (Trading Systems module, chapter
"momentum-portfolios"), exactly as published, on the same data, costs, metrics and gates as the pre-registered
factor-momentum test (stratlab/factor_momentum.py, prereg/momentum_v1.json), which it reuses. Research only.

Rule (Varsity): at the close of the last trading day of each month rank the universe by 12-month return
(end / start - 1 = C[t] / C[t-252] - 1, no skip month, no volatility adjustment), buy the top 12 equal weight on the
first trading day of the next month (here: at its OPEN, as factor_momentum executes), and at each monthly rebalance
sell the names that dropped out of the top 12 and buy the new entrants. Defaults fixed in the pre-registration
(stratlab/prereg/kite_ecosystem_v1.json) before any run: names that stay in the top 12 are NOT re-weighted; the cash
from the sales is split equally among the entrants (at the start: 1/12 of capital each); no trend filter, no stop.

Universe, data, costs and gates are factor_momentum's: era 1 = nse_old 2013-01-31..2023-11-09 point-in-time
top-100 by turnover (delisted names kept), era 2 = bars3y 2023-11-30..2026-11 on the current ~97 NIFTY-100 names
(survivorship-biased, history spliced from nse_old); CNC statutory costs + 10 bps slippage per side; PASS = each
era-1 half CAGR > 0 and > NIFTY, era-1 Sharpe >= 0.5, era-2 CAGR > 0 and > NIFTY. One look: a ledger refuses to
re-run the strategy.

    python -m stratlab.kite_momentum --prereg <PREREGISTERED_KITE.json> --ledger <ledger.json> --out <dir>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import inspect
import json
import os
import sys

import numpy as np

from . import factor_momentum as FM

SID = "r5_kite_vars_momentum12"
TOP_N = 12
LOOKBACK = 252
PARAMS = {"top_n": TOP_N, "lookback": LOOKBACK, "rebalance": "monthly", "reweight_kept": False,
          "entrants_share_cash": True, "slippage_bps": FM.SLIP_BPS}


def scores(p: FM.Panel, t: int, exec_month: int) -> np.ndarray:
    """12-month return C[t] / C[t-252] - 1 (forward-filled closes), NaN where ineligible (factor_momentum.eligible)."""
    ok = FM.eligible(p, t, exec_month)
    out = np.full(len(p.symbols), np.nan)
    if t < LOOKBACK:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        r = p.cf[t] / p.cf[t - LOOKBACK] - 1.0
    ok &= np.isfinite(r)
    out[ok] = r[ok]
    return out


def pick(p: FM.Panel, sc: np.ndarray) -> list:
    idx = [j for j in range(len(sc)) if np.isfinite(sc[j])]
    idx.sort(key=lambda j: (-sc[j], p.symbols[j]))
    return idx[:TOP_N]


def backtest(p: FM.Panel, first_decision: int, end: int | None = None, capital: float = FM.CAPITAL,
             cost_fn=FM.trade_costs) -> dict:
    """Same output layout as factor_momentum.backtest (so its segment_metrics / judge apply unchanged)."""
    dates = p.dates
    last = len(dates) - 1 if end is None else int(np.searchsorted(dates, end, side="right")) - 1
    decs = [t for t in FM.decision_sessions(dates[: last + 1]) if dates[t] >= first_decision]
    t0 = decs[0]
    decset = set(decs)
    M = len(p.symbols)
    sh = np.zeros(M)
    cash = capital
    eq_dates, eq = [int(dates[t0])], [capital]
    stat_tot = slip_tot = traded = 0.0
    stale_exits, n_rebal = 0, 0
    log = []
    pending = pick(p, scores(p, t0, int(dates[t0 + 1]) // 100))
    for i in range(t0 + 1, last + 1):
        if pending is not None:
            px = p.open[i].copy()
            stale = ~np.isfinite(px)
            px[stale] = p.cf[i - 1][stale]
            want = set(pending)
            held = {j for j in range(M) if sh[j] > 0}
            for j in sorted(held - want):                      # sell the names that dropped out
                v = sh[j] * px[j]
                stat, slip = cost_fn(v, "SELL")
                cash += v - stat - slip
                stat_tot += stat
                slip_tot += slip
                traded += v
                stale_exits += int(stale[j])
                log.append((int(dates[i]), p.symbols[j], "SELL", round(v, 2), round(stat + slip, 2), bool(stale[j])))
                sh[j] = 0.0
            new = [j for j in pending if j not in held and not stale[j]]
            if new and cash > 0:
                budget = cash / len(new)                       # entrants share the available cash equally
                for j in new:
                    stat, slip = cost_fn(budget, "BUY")
                    v = budget - stat - slip                   # costs come out of the entrant's share
                    if v <= 0:
                        continue
                    stat, slip = cost_fn(v, "BUY")
                    sh[j] = v / px[j]
                    cash -= v + stat + slip
                    stat_tot += stat
                    slip_tot += slip
                    traded += v
                    log.append((int(dates[i]), p.symbols[j], "BUY", round(v, 2), round(stat + slip, 2), False))
            n_rebal += 1
            pending = None
        e = cash + float((sh * np.nan_to_num(p.cf[i])).sum())
        eq_dates.append(int(dates[i]))
        eq.append(e)
        if i in decset and i < last:
            pending = pick(p, scores(p, i, int(dates[i + 1]) // 100))
    return {"dates": np.array(eq_dates), "equity": np.array(eq), "statutory": stat_tot, "slippage": slip_tot,
            "traded": traded, "stale_exits": stale_exits, "rebalances": n_rebal, "cash_switches": 0,
            "trades": log, "first_decision": int(dates[t0]), "last": int(dates[last])}


def fingerprint() -> str:
    src = "".join(inspect.getsource(f) for f in (scores, pick, backtest, FM.eligible, FM.trade_costs,
                                                 FM.segment_metrics, FM.judge))
    return hashlib.sha1(repr((SID, sorted(PARAMS.items()), src)).encode()).hexdigest()[:12]


def main(argv=None) -> int:
    from .data import assert_outside_repo
    ap = argparse.ArgumentParser(prog="python -m stratlab.kite_momentum")
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from .kite_spreads import OLD_DIR, NSE_DIR, BARS3Y, WORK
    pr = json.load(open(a.prereg))
    reg = pr.get("strategies", {}).get(SID)
    if reg is None or reg.get("fingerprint") != fingerprint():
        raise SystemExit(f"{SID} not pre-registered with fingerprint {fingerprint()}: refusing to run")
    ledger_path = assert_outside_repo(a.ledger)
    out = assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    led = FM.load_ledger(ledger_path)
    if SID in led:
        print(f"REFUSED: {SID} already judged {led[SID]['when']} (one look). Stored verdict: {led[SID]['result']['verdict']}")
        return 0
    os.makedirs(WORK, exist_ok=True)
    p1, n1, eq = FM.load_oldera(OLD_DIR, cache=os.path.join(WORK, "_eqadj_cache.pkl"))
    p2, n2, _notes = FM.load_era2(BARS3Y, eq, OLD_DIR, NSE_DIR)
    r1 = backtest(p1, FM.ERA1["first_decision"], FM.ERA1["end"])
    h1_end = int(r1["dates"][r1["dates"] <= FM.ERA1["h1_end"]][-1])
    era1 = {"h1": FM.segment_metrics(r1, n1, r1["dates"][0], h1_end),
            "h2": FM.segment_metrics(r1, n1, h1_end, r1["dates"][-1]),
            "whole": FM.segment_metrics(r1, n1, r1["dates"][0], r1["dates"][-1]),
            "stale_exits": r1["stale_exits"], "rebalances": r1["rebalances"]}
    r2 = backtest(p2, FM.ERA2["first_decision"])
    era2 = {"whole": FM.segment_metrics(r2, n2, r2["dates"][0], r2["dates"][-1]), "stale_exits": r2["stale_exits"],
            "rebalances": r2["rebalances"], "label": p2.label}
    res = {"era1": era1, "era2": era2, **FM.judge(era1, era2)}
    led[SID] = {"when": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "fingerprint": fingerprint(),
                "params": PARAMS, "result": res}
    FM.save_ledger(ledger_path, led)
    for name, r in (("era1", r1), ("era2", r2)):
        with open(os.path.join(out, f"{SID}_{name}_equity.csv"), "w") as f:
            f.write("date,equity\n")
            for d_, e_ in zip(r["dates"], r["equity"]):
                f.write(f"{d_},{e_:.2f}\n")
        with open(os.path.join(out, f"{SID}_{name}_trades.csv"), "w") as f:
            f.write("date,symbol,side,value,cost,stale_exit\n")
            for t in r["trades"]:
                f.write(",".join(str(x) for x in t) + "\n")
    lines = [f"{SID}  ->  {res['verdict']}  {('failed: ' + ', '.join(res['failed'])) if res['failed'] else ''}"]
    for k in ("h1", "h2", "whole"):
        lines.append(f"  era1 {k:5s} {era1[k]['from']}..{era1[k]['to']}  {FM._fmt(era1[k])}")
    lines.append(f"  era2 whole {era2['whole']['from']}..{era2['whole']['to']}  {FM._fmt(era2['whole'])}  "
                 "[survivorship-biased]")
    lines.append(f"  stale exits era1 {era1['stale_exits']}, rebalances {era1['rebalances']}/{era2['rebalances']}")
    txt = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
