"""
lab_risk_build.py — builds lab_risk.json, the lab reference the engine's risk budget (risk_budget.py) and the drift
monitor (drift_monitor.py) read. Research-data step, run by hand; the app only reads the JSON.

    python lab_risk_build.py --scratch <lab scratch dir> [--out lab_risk.json] [--mc 2000]

From each kept strategy's 2023-26 lab trade file (DISCOVERY + FINAL, at lab size and lab costs; the same files as
stratlab.robustness.named_sources) it records:
  * trades, sessions of the 3-year calendar (stratlab.trials.calendar_3y), trades per session;
  * net per trade (mean, std), win %, the daily net P&L series' mean and std over every session (zero days included),
    the worst day and the observed max drawdown;
  * stratlab.robustness.mc_trades: trade bootstrap and stationary block bootstrap of the daily P&L (drawdown and
    total percentiles) — the lab bands;
  * a bootstrap band of the mean of 20 consecutive trades (p5 / p50 / p95) for the rolling-expectancy check;
  * the combined daily P&L of the strategies the engine runs by default (auc 2PCT/3PCT as configured is the operator's
    choice, so both variants are listed; the combined book uses the variant passed with --auc).
Nothing here changes a lab verdict or touches the trials ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import numpy as np

from stratlab import robustness as R
from stratlab.trials import calendar_3y, daily_pnl, load_trades

FILES = {
    "AUC-FADE-2PCT": ("dd", "auction_final", "trades_AUC-FADE-2PCT.csv"),
    "AUC-FADE-3PCT": ("dd", "auction_final", "trades_AUC-FADE-3PCT.csv"),
    "SWING-ATTN-ON-3X": ("dd", "r2_swing_out", "trades_SWING-ATTN-ON-3X.csv"),
    "VRP-STRANGLE-016-WINGED": ("dd", "rev18", "repro", "trades", "VRP-STRANGLE-016-WINGED.csv"),
    "EV-PEAD-LONG-H60": ("events", "run1", "trades", "ev_pead_long_h60_2023-26.csv"),
}
FILL_MODEL = {
    "AUC-FADE-2PCT": "entry = the official 09:15 auction open (lab: 09:15 bar open, 2.1 bps/side); exit 09:20 bar open",
    "AUC-FADE-3PCT": "entry = the official 09:15 auction open (lab: 09:15 bar open, 2.1 bps/side); exit 09:20 bar open",
    "SWING-ATTN-ON-3X": "entry = 15:21 bar open (live: the 15:21 ask); exit = next session's 09:15 open (auction)",
    "VRP-STRANGLE-016-WINGED": "legs at the 10:20 Black-Scholes credit (flat VIX) less a haircut; settlement at "
                               "intrinsic from the expiry day's 15:00-15:29 mean close",
    "EV-PEAD-LONG-H60": "entry = next session's 09:15 open; exit = stop / 60-session time exit at the close",
}
REF_SLIPPAGE_BPS = {"AUC-FADE-2PCT": 2.1, "AUC-FADE-3PCT": 2.1, "SWING-ATTN-ON-3X": 2.0,
                    "VRP-STRANGLE-016-WINGED": None, "EV-PEAD-LONG-H60": 2.0}


def _pct(a, ps=(5, 50, 95, 99)):
    return {f"p{p}": round(float(np.percentile(a, p)), 2) for p in ps}


def build(scratch: str, n_mc: int = 2000, auc: str = "AUC-FADE-3PCT", log=print) -> dict:
    cal = calendar_3y(scratch)
    lo, hi = cal[0], cal[-1]
    out = {"built": date.today().isoformat(), "source": "lab 2023-26 trade files (DISCOVERY + FINAL), lab size",
           "calendar": {"first": lo, "last": hi, "sessions": len(cal)}, "strategies": {}}
    daily_by = {}
    for lab_id, parts in FILES.items():
        path = os.path.join(scratch, *parts)
        if not os.path.exists(path):
            log(f"missing {path} — {lab_id} skipped")
            continue
        tr = load_trades(path)
        nets = np.array([t["net"] for t in tr], float)
        daily = np.array(daily_pnl(tr, cal, lo, hi), float)
        daily_by[lab_id] = daily
        mc = R.mc_trades(nets, daily, n=n_mc, seed=0)
        rng = np.random.default_rng(1)
        k = min(20, len(nets))
        means20 = nets[rng.integers(0, len(nets), (n_mc, k))].mean(axis=1)
        out["strategies"][lab_id] = {
            "trades": int(len(nets)), "trades_per_session": round(len(nets) / len(cal), 4),
            "net_mean": round(float(nets.mean()), 2), "net_std": round(float(nets.std(ddof=1)), 2),
            "win_pct": round(100 * float((nets > 0).mean()), 1), "net_total": round(float(nets.sum()), 2),
            "daily_mean": round(float(daily.mean()), 2), "daily_std": round(float(daily.std(ddof=1)), 2),
            "worst_day": round(float(daily.min()), 2), "observed_max_dd": round(R.max_drawdown(daily), 2),
            "block_dd": mc.get("block_dd"), "boot_dd": mc["boot_dd"], "block_total": mc.get("block_total"),
            "mean20_band": _pct(means20), "fill_model": FILL_MODEL[lab_id], "ref_slippage_bps": REF_SLIPPAGE_BPS[lab_id],
        }
        log(f"{lab_id}: {len(nets)} trades, daily std {daily.std(ddof=1):,.0f}, block dd p99 "
            f"{(mc.get('block_dd') or {}).get('p99')}")
    book = [x for x in (auc, "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED") if x in daily_by]
    if book:
        comb = np.sum([daily_by[x] for x in book], axis=0)
        idx = R.stationary_bootstrap_indices(len(comb), n_mc, 10.0, np.random.default_rng(2))
        dds = np.array([R.max_drawdown(comb[r]) for r in idx])
        out["combined_default_book"] = {"strategies": book, "daily_std": round(float(comb.std(ddof=1)), 2),
                                        "observed_max_dd": round(R.max_drawdown(comb), 2), "block_dd": _pct(dds)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_risk.json"))
    ap.add_argument("--mc", type=int, default=2000)
    ap.add_argument("--auc", default="AUC-FADE-3PCT")
    a = ap.parse_args(argv)
    res = build(a.scratch, a.mc, a.auc)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("written", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
