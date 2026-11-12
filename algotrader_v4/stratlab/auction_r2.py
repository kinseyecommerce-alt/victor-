"""
stratlab/auction_r2.py — round-2 SCALPING strategy SCALP-AUC-FADE-1TO3-0925 (pre-registered; do not tune).

A parameter variant of the opening-auction fade (stratlab/auction.py): same fill model (enter AT the 09:15 call-
auction print, which the intraday simulator cannot express), same cost function (costs.round_trip_cost,
equity_mis) and the SAME protocol.evaluate (DISCOVERY / FINAL split of the session calendar, H1 / H2 screen,
PF >= 1.15, >= 150 trades, no symbol > 35 % of net, FINAL once via the data cache's final_ledger.json).

Rules (fixed before any run, copied from the pre-registration)
  * universe: every stock in the data directory (index files excluded), MIS
  * skip a symbol-day if its previous session had fewer than 200 one-minute bars (a dropped short session lies
    between the symbol's last full session and today) or if there is no previous session. Corporate actions /
    ex-dates cannot be filtered in the backtest (kept); live uses the exchange calendar.
  * gap = 09:15 bar open / close of the last 1-minute bar of the previous session - 1 (live: the 09:07-09:08
    pre-open equilibrium price). Eligible if 1.0 % <= |gap| <= 3.0 % (both inclusive).
  * side = SHORT if gap > 0, LONG if gap < 0
  * per day: rank qualifying symbols by |gap| (ties by symbol), take at most the 10 largest; one trade per symbol
  * entry at the auction price (09:15 open) -/+ slippage (default 2 bps per side; 5 bps stress reported)
  * exit: market order at the open of the 09:25 bar (+/- slippage). If the 09:25 bar is missing, the open of the
    first later bar before 09:35 (same 10-minute fallback as auction.py); else no trade.
  * qty = floor(1,00,000 / auction price); no stop, no target, no trailing
Look-ahead: the signal for day d reads only bars of earlier sessions and the 09:15 bar of d; the exit reads one
bar at 09:25. Remaining caveat (as auction.py): the backtest uses the FINAL auction print, live uses the indicative
price at 09:07-09:08.

    cd algotrader_v4
    python -m stratlab.auction_r2 --data <bars3y dir> --out <scratch dir> [--screen-only]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402

from . import auction as A  # noqa: E402
from . import data as D  # noqa: E402
from . import protocol as P  # noqa: E402
from . import sim  # noqa: E402
from .api import Spec  # noqa: E402

AUCTION_T = A.AUCTION_T            # 09:15
EXIT_T = 9 * 60 + 25               # 09:25 bar open
EXIT_LATEST = EXIT_T + 10          # fallback window if the 09:25 bar is missing
MIN_GAP = 0.010
MAX_GAP = 0.030
MAX_PER_DAY = 10
NOTIONAL = 100_000.0
DEFAULT_SLIPPAGE_BPS = 2.0         # pre-registered for this strategy (per side)
STRESS_SLIPPAGE_BPS = 5.0
FILL_MODEL = "auction1"            # identical fill rules to auction.py (exit time is a parameter)
COST_MODEL = f"{sim.COST_MODEL}+{FILL_MODEL}"
EPS = 1e-12


def fade_rule(prev_close: float, auction_px: float, min_gap: float = MIN_GAP, max_gap: float = MAX_GAP):
    """(side, gap) or None. side -1 (short) for an up-gap, +1 (long) for a down-gap; band inclusive."""
    if not (prev_close > 0 and auction_px > 0):
        return None
    gap = auction_px / prev_close - 1.0
    if abs(gap) < min_gap - EPS or abs(gap) > max_gap + EPS:
        return None
    return (-1 if gap > 0 else 1), gap


SPECS = [
    Spec("SCALP-AUC-FADE-1TO3-0925", "scalping", "opening-auction fade",
         "Fade the 09:15 auction print when 1 % <= |gap vs previous close| <= 3 %: enter at the auction, exit at the "
         "09:25 open; top 10 |gap| per day, Rs 1 lakh each, no stop", fade_rule,
         {"min_gap": MIN_GAP, "max_gap": MAX_GAP, "max_per_day": MAX_PER_DAY, "exit": "09:25"}),
]


def session_features(sd, exit_t: int = EXIT_T, exit_latest: int = EXIT_LATEST) -> list:
    """Per session: (date, prev_close, auction_px, exit_t, exit_px, prev_short). prev_short = True when a session
    dropped for being short (< 200 bars) lies between the previous full session and this one, or there is none."""
    dropped = np.array(sorted(int(x) for x in (sd.dropped or [])), dtype=np.int64)
    out = []
    for k in range(sd.n_sessions):
        a, b = int(sd.starts[k]), int(sd.ends[k])
        day = int(sd.days[k])
        if k > 0:
            pc = float(sd.c[int(sd.ends[k - 1]) - 1])
            pday = int(sd.days[k - 1])
            short = bool(len(dropped) and ((dropped > pday) & (dropped < day)).any())
        else:
            pc, short = float("nan"), True
        apx = float(sd.o[a]) if int(sd.t[a]) == AUCTION_T else float("nan")
        xt, xp = -1, float("nan")
        tt = sd.t[a:b]
        j = int(np.searchsorted(tt, exit_t))
        if j < len(tt) and int(tt[j]) < exit_latest:
            xt, xp = int(tt[j]), float(sd.o[a + j])
        out.append((day, pc, apx, xt, xp, short))
    return out


def candidates_for(symbol: str, feats: list) -> list:
    """Eligible symbol-days: (date, symbol, side, gap, auction_px, exit_t, exit_px) — auction.py's row layout."""
    rows = []
    for date, pc, apx, xt, xp, short in feats:
        if short or not (np.isfinite(apx) and np.isfinite(xp)):
            continue
        sig = fade_rule(pc, apx)
        if sig is None:
            continue
        rows.append((date, symbol, sig[0], sig[1], apx, xt, xp))
    return rows


def select(rows, spec: Spec, lo: int, hi: int) -> list:
    p = spec.params
    by_day = defaultdict(dict)
    for r in rows:
        if lo <= r[0] <= hi and fade_rule(1.0, 1.0 + r[3], p["min_gap"], p["max_gap"]) is not None:
            by_day[r[0]].setdefault(r[1], r)            # one position per symbol per day
    picked = []
    for d in sorted(by_day):
        picked.extend(sorted(by_day[d].values(), key=lambda r: (-abs(r[3]), r[1]))[: p["max_per_day"]])
    return picked


def settle(spec_id: str, row, slip: float) -> tuple:
    return A.settle(spec_id, row, slip)        # same fill / qty / cost rules as auction.py


def make_simulator(rows, slippage_bps: float = DEFAULT_SLIPPAGE_BPS):
    def simulate(specs, lo, hi, slip_bps=None):
        slip = (slippage_bps if slip_bps is None else slip_bps) / 1e4
        out = {}
        for sp in specs:
            tr = [settle(sp.id, r, slip) for r in select(rows, sp, lo, hi)]
            tr.sort(key=lambda t: (t[2], t[1]))
            out[sp.id] = {"trades": tr, "skipped": 0, "lookahead": None, "error": None}
        return out
    return simulate


def load_universe(data_dir: str, cache: str | None = None, symbols=None, verbose: bool = True):
    man = D.prepare(data_dir, cache or None, verbose=verbose)
    cdir = man["cache_dir"]
    syms = sorted(man["symbols"])
    if symbols:
        syms = syms[:symbols]
    rows, cal, n = [], set(), 0
    for s in syms:
        sd = D.load_symbol(cdir, s)
        if sd is None:
            continue
        n += 1
        cal.update(int(x) for x in sd.days)
        rows.extend(candidates_for(s, session_features(sd)))
        del sd
    return rows, sorted(cal), n, cdir


def extra(tr) -> dict:
    """Long / short split, top-symbol share of net, date-clustered t on net."""
    if not tr:
        return {}
    by_sym = defaultdict(float)
    for t in tr:
        by_sym[t[1]] += t[12]
    net = sum(t[12] for t in tr)
    top = max(by_sym.items(), key=lambda kv: kv[1])
    side = {}
    for nm, sg in (("long", 1), ("short", -1)):
        s = [t for t in tr if t[3] == sg]
        side[nm] = {"trades": len(s), "net": round(sum(t[12] for t in s), 2),
                    "net_bps": round(float(np.mean(A._bps(s, 12))), 2) if s else 0.0,
                    "pf": P.stats(s)["pf"] if s else 0.0}
    return {"top_symbol": top[0], "top_share": round(top[1] / net, 3) if net > 0 else None,
            "t_net": round(A.tstat_by_date(tr, 12), 2), **side}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SCALP-AUC-FADE-1TO3-0925 through the lab protocol")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--symbols", type=int, default=0)
    ap.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    ap.add_argument("--stress-bps", type=float, default=STRESS_SLIPPAGE_BPS)
    ap.add_argument("--screen-only", action="store_true", help="never simulate FINAL (no ledger entry)")
    a = ap.parse_args(argv)                      # deliberately no --peek-final
    out = D.assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    rows, calendar, n_sym, cdir = load_universe(a.data, a.cache, a.symbols)
    simulate = make_simulator(rows, a.slippage_bps)
    results, trades = P.evaluate(SPECS, calendar, simulate, n_symbols=n_sym,
                                 ledger_path=os.path.join(cdir, "final_ledger.json"),
                                 dataset=os.path.realpath(a.data), screen_only=a.screen_only, cost_model=COST_MODEL)
    split = results[0]["split"] if "split" in results[0] else P.split_sessions(calendar)
    stress = make_simulator(rows, a.stress_bps)
    report = {"split": {"h1": split.h1, "h2": split.h2, "final": split.final, "n_sessions": split.n},
              "n_symbols": n_sym, "slippage_bps": a.slippage_bps, "stress_bps": a.stress_bps,
              "cost_model": COST_MODEL, "strategies": {}}
    lines = [f"Round-2 scalping: auction fade 1-3 % / 09:25 | {n_sym} symbols, {split.n} sessions | H1 {split.h1} "
             f"H2 {split.h2} FINAL {split.final} | slippage {a.slippage_bps} bps/side (stress {a.stress_bps}) | "
             f"cost model {COST_MODEL}"]
    for r in results:
        sid = r["id"]
        tr = trades[sid]
        disc = [t for t in tr if split.discovery[0] <= t[2] <= split.discovery[1]]
        h1 = [t for t in disc if t[2] <= split.h1[1]]
        h2 = [t for t in disc if t[2] >= split.h2[0]]
        fin = [t for t in tr if t[2] >= split.final[0]]
        sd5 = stress([r["spec"]], split.discovery[0], split.discovery[1])[sid]["trades"]
        sc5 = P.screen(sd5, split, "intraday")
        rec = {"verdict": r["verdict"], "why": r["why"], "final_note": r.get("final_note", ""),
               "top_symbol": r["screen"]["top_symbol"], "top_share": round(float(r["screen"]["top_share"]), 3),
               "H1": A.block(h1), "H2": A.block(h2), "DISC": A.block(disc), "DISC_extra": extra(disc),
               "DISC_quarters": A.quarters(disc),
               "DISC_stress": {**A.block(sd5), "screen_passed": sc5["passed"], "reasons": sc5["reasons"],
                               "H1_net": sc5["h1"]["net"], "H2_net": sc5["h2"]["net"]}}
        if r["final"] is not None:
            f5 = stress([r["spec"]], split.final[0], split.final[1])[sid]["trades"]
            rec["FINAL"] = A.block(fin)
            rec["FINAL_extra"] = extra(fin)
            rec["FINAL_quarters"] = A.quarters(fin)
            rec["FINAL_stress"] = A.block(f5)
        report["strategies"][sid] = rec
        with open(os.path.join(out, f"trades_{sid}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(sim.TRADE_FIELDS)
            w.writerows(tr)
        lines.append(f"\n{sid}: {r['verdict']} ({r['why']}) {rec['final_note']}")
        for k in ("H1", "H2", "DISC", "DISC_stress", "FINAL", "FINAL_stress"):
            if k in rec:
                b = rec[k]
                lines.append(f"  {k:12s} n={b['trades']:5d} net={b['net']:>11,.0f} PF={b['pf']:.2f} win={b['win_pct']}% "
                             f"gross={b['gross_bps']}bps net={b['net_bps']}bps t_gross={b['t_gross']} "
                             f"net_ex_best5d={b['net_ex_best5_dates']:,.0f}")
        for k in ("DISC_extra", "FINAL_extra"):
            if k in rec:
                lines.append(f"  {k}: {rec[k]}")
        lines.append(f"  top symbol {rec['top_symbol']} {rec['top_share']:.1%} of DISCOVERY net; "
                     f"stress screen {'PASS' if sc5['passed'] else 'FAIL ' + ', '.join(sc5['reasons'])}")
        lines.append(f"  DISC quarters: {rec['DISC_quarters']}")
        if "FINAL_quarters" in rec:
            lines.append(f"  FINAL quarters: {rec['FINAL_quarters']}")
    lines.append("\n" + P.luck_reminder(results))
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
