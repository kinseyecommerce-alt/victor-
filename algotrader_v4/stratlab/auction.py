"""
stratlab/auction.py — opening-auction liquidity provision (NIFTY-100 stocks, MIS, event-driven), pre-registered.

The intraday simulator cannot express these strategies: they are filled AT the 09:15 open (the pre-open call
auction print), while every `Entry` in the lab is filled at the open of the bar after its signal bar. So this
module runs its own tiny fill model through the SAME protocol (protocol.evaluate: same DISCOVERY / FINAL split
of the session calendar, H1 / H2 screen, PF >= 1.15, >= 150 trades, no symbol > 35 % of net, FINAL once through
the same final_ledger.json in the data cache) and the same cost function (costs.round_trip_cost, equity_mis).

Rules (fixed before any run; do not change them after seeing results)
  * universe: every stock in the data directory (index files excluded); sessions < 200 bars dropped
    (data.MIN_SESSION_BARS), so "previous session" is the previous FULL session of that symbol
  * prev_close = close of the last 1-minute bar of the symbol's previous valid session
  * auction_px = open of the 09:15 bar (a session whose first bar is not 09:15 gives no signal)
  * gap = auction_px / prev_close - 1; eligible if MIN_GAP < |gap| <= 20 % (larger = corporate action / circuit)
  * side = SHORT if gap > 0, LONG if gap < 0 (fade the auction)
  * entry at auction_px (+/- slippage); exit at the open of the 09:20 bar (- / + slippage), a market order.
    If the 09:20 bar is missing, the open of the first bar after it (before 09:30) is used; else no trade.
  * qty = floor(1,00,000 / auction_px); at most 10 trades per day, the 10 largest |gap| (ties by symbol)
  * no stop, no target (5-minute hold)
Look-ahead: the signal for day d reads only bars of earlier sessions and the 09:15 bar of day d; the ranking reads
the 09:15 gaps of all symbols on day d (all printed at the same time). Caveat that remains: live, the order is
placed at 09:07 on the INDICATIVE price; the backtest uses the final auction price, which is a mild look-ahead.

    cd algotrader_v4
    python -m stratlab.auction --data <bars3y dir> --out <scratch dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402

from . import data as D  # noqa: E402
from . import protocol as P  # noqa: E402
from . import sim  # noqa: E402
from .api import Spec  # noqa: E402

AUCTION_T = 9 * 60 + 15          # 09:15 bar = the call-auction print
EXIT_T = 9 * 60 + 20             # 09:20 bar open
EXIT_LATEST = 9 * 60 + 30        # fallback window if the 09:20 bar is missing
MAX_ABS_GAP = 0.20               # |gap| above this = corporate action / circuit: excluded
MAX_PER_DAY = 10
NOTIONAL = 100_000.0
DEFAULT_SLIPPAGE_BPS = 2.1       # pre-registered for this family (per side)
STRESS_SLIPPAGE_BPS = 5.0
FILL_MODEL = "auction1"          # bump if a fill rule here changes (part of the FINAL-ledger key)
COST_MODEL = f"{sim.COST_MODEL}+{FILL_MODEL}"


def auction_rule(prev_close: float, auction_px: float, min_gap: float, max_gap: float = MAX_ABS_GAP):
    """The signal: (side, gap) or None. side = -1 (short) for an up-gap, +1 (long) for a down-gap."""
    if not (prev_close > 0 and auction_px > 0):
        return None
    gap = auction_px / prev_close - 1.0
    if abs(gap) <= min_gap + 1e-9 or abs(gap) > max_gap:      # 1e-9: an exact 2.00 % gap is not "> 2 %"
        return None
    return (-1 if gap > 0 else 1), gap


SPECS = [
    Spec("AUC-FADE-2PCT", "intraday", "opening-auction fade",
         "Fade the 09:15 auction print when |gap vs previous close| > 2 %: enter at the auction, exit at the "
         "09:20 open; top 10 |gap| per day, Rs 1 lakh each", auction_rule,
         {"min_gap": 0.02, "max_gap": MAX_ABS_GAP, "max_per_day": MAX_PER_DAY, "exit": "09:20"}),
    Spec("AUC-FADE-3PCT", "intraday", "opening-auction fade",
         "As AUC-FADE-2PCT with |gap| > 3 % (nested subset; one effect, not independent evidence)", auction_rule,
         {"min_gap": 0.03, "max_gap": MAX_ABS_GAP, "max_per_day": MAX_PER_DAY, "exit": "09:20"}),
]


def session_features(sd) -> list:
    """Per session of one symbol: (date, prev_close, auction_px, exit_t, exit_px). prev_close is the close of the
    last bar of the previous (valid) session; auction_px / exit are NaN when the 09:15 / 09:20 bar is unusable.
    Reads, for day d, only bars of sessions before d and bars of d up to the exit bar."""
    out = []
    for k in range(sd.n_sessions):
        a, b = int(sd.starts[k]), int(sd.ends[k])
        pc = float(sd.c[int(sd.ends[k - 1]) - 1]) if k > 0 else float("nan")
        ap_ = float(sd.o[a]) if int(sd.t[a]) == AUCTION_T else float("nan")
        xt, xp = -1, float("nan")
        tt = sd.t[a:b]
        j = int(np.searchsorted(tt, EXIT_T))
        if j < len(tt) and int(tt[j]) < EXIT_LATEST:
            xt, xp = int(tt[j]), float(sd.o[a + j])
        out.append((int(sd.days[k]), pc, ap_, xt, xp))
    return out


def candidates_for(symbol: str, feats: list, min_gap: float) -> list:
    """Eligible symbol-days at the loosest threshold: (date, symbol, side, gap, auction_px, exit_t, exit_px)."""
    rows = []
    for date, pc, apx, xt, xp in feats:
        if not (np.isfinite(apx) and np.isfinite(xp)):
            continue
        sig = auction_rule(pc, apx, min_gap)
        if sig is None:
            continue
        rows.append((date, symbol, sig[0], sig[1], apx, xt, xp))
    return rows


def settle(spec_id: str, row, slip: float) -> tuple:
    """One trade in sim.TRADE_FIELDS layout. qty from the auction price; slippage against us on both fills."""
    date, symbol, side, _gap, apx, xt, xp = row
    qty = max(1, int(NOTIONAL // apx))
    fill = apx * (1.0 + side * slip)
    exit_px = xp * (1.0 - side * slip)
    gross = side * (exit_px - fill) * qty
    buy_v, sell_v = (fill * qty, exit_px * qty) if side > 0 else (exit_px * qty, fill * qty)
    cost = round_trip_cost(buy_v, sell_v, "equity_mis")
    return (spec_id, symbol, int(date), int(side), sim._hm(AUCTION_T), round(fill, 4), int(date), sim._hm(xt),
            round(exit_px, 4), qty, round(gross, 2), cost, round(gross - cost, 2), "time")


def select(rows, spec: Spec, lo: int, hi: int) -> list:
    """Per day in [lo, hi]: the spec's eligible rows, the `max_per_day` largest |gap| (ties by symbol)."""
    p = spec.params
    by_day = defaultdict(list)
    for r in rows:
        if lo <= r[0] <= hi and p["min_gap"] < abs(r[3]) <= p["max_gap"]:
            by_day[r[0]].append(r)
    picked = []
    for d in sorted(by_day):
        picked.extend(sorted(by_day[d], key=lambda r: (-abs(r[3]), r[1]))[: p["max_per_day"]])
    return picked


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
    """Symbol by symbol (low memory): candidate rows at the loosest threshold, the session calendar, symbol count."""
    man = D.prepare(data_dir, cache or None, verbose=verbose)
    cdir = man["cache_dir"]
    syms = sorted(man["symbols"])
    if symbols:
        syms = syms[:symbols]
    loosest = min(sp.params["min_gap"] for sp in SPECS)
    rows, cal, n = [], set(), 0
    for s in syms:
        sd = D.load_symbol(cdir, s)
        if sd is None:
            continue
        n += 1
        cal.update(int(x) for x in sd.days)
        rows.extend(candidates_for(s, session_features(sd), loosest))
        del sd
    return rows, sorted(cal), n, cdir


# ── reporting helpers ───────────────────────────────────────────────────────
def _bps(tr, field):
    return [t[field] / (t[5] * t[9]) * 1e4 for t in tr]


def tstat_by_date(tr, field=10) -> float:
    """Mean gross (or net) return per trade over its date-clustered standard error."""
    if len(tr) < 3:
        return float("nan")
    x = np.array(_bps(tr, field))
    d = np.array([t[2] for t in tr])
    mu = x.mean()
    cs = np.array([(x[d == u] - mu).sum() for u in np.unique(d)])
    g = len(cs)
    if g < 2:
        return float("nan")
    se = math.sqrt((cs ** 2).sum() * g / (g - 1)) / len(x)
    return float(mu / se) if se > 0 else float("nan")


def block(tr) -> dict:
    st = P.stats(tr)
    if not tr:
        return {**st, "gross_bps": 0.0, "net_bps": 0.0, "t_gross": float("nan"), "net_ex_best5_dates": 0.0}
    by_date = defaultdict(float)
    for t in tr:
        by_date[t[2]] += t[12]
    best5 = set(sorted(by_date, key=lambda d: -by_date[d])[:5])
    return {**st, "gross_bps": round(float(np.mean(_bps(tr, 10))), 2), "net_bps": round(float(np.mean(_bps(tr, 12))), 2),
            "t_gross": round(tstat_by_date(tr, 10), 2),
            "net_ex_best5_dates": round(sum(t[12] for t in tr if t[2] not in best5), 2),
            "days": len(by_date), "max_day_trades": max(sum(1 for t in tr if t[2] == d) for d in by_date)}


def quarters(tr) -> dict:
    q = defaultdict(float)
    for t in tr:
        y, m = divmod(t[2] // 100, 100)
        q[f"{y}Q{(m - 1) // 3 + 1}"] += t[12]
    return {k: round(v, 0) for k, v in sorted(q.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Opening-auction fade strategies through the lab protocol")
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
    split = results[0]["split"]
    stress = make_simulator(rows, a.stress_bps)
    report = {"split": {"h1": split.h1, "h2": split.h2, "final": split.final, "n_sessions": split.n},
              "n_symbols": n_sym, "slippage_bps": a.slippage_bps, "stress_bps": a.stress_bps,
              "cost_model": COST_MODEL, "strategies": {}}
    lines = [f"Opening-auction fade | {n_sym} symbols, {split.n} sessions | H1 {split.h1} H2 {split.h2} "
             f"FINAL {split.final} | slippage {a.slippage_bps} bps/side (stress {a.stress_bps}) | cost model {COST_MODEL}"]
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
               "H1": block(h1), "H2": block(h2), "DISC": block(disc), "DISC_quarters": quarters(disc),
               "DISC_stress": {**block(sd5), "screen_passed": sc5["passed"], "reasons": sc5["reasons"],
                               "H1_net": sc5["h1"]["net"], "H2_net": sc5["h2"]["net"]}}
        if r["final"] is not None:          # FINAL legitimately evaluated (screen passer): same trades re-costed at stress
            f5 = stress([r["spec"]], split.final[0], split.final[1])[sid]["trades"]
            rec["FINAL"] = block(fin)
            rec["FINAL_quarters"] = quarters(fin)
            rec["FINAL_stress"] = block(f5)
        report["strategies"][sid] = rec
        with open(os.path.join(out, f"trades_{sid}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(sim.TRADE_FIELDS)
            w.writerows(tr)
        lines.append(f"\n{sid}: {r['verdict']} ({r['why']}) {rec['final_note']}")
        for k in ("H1", "H2", "DISC", "DISC_stress", "FINAL", "FINAL_stress"):
            if k in rec:
                b = rec[k]
                lines.append(f"  {k:12s} n={b['trades']:4d} net={b['net']:>11,.0f} PF={b['pf']:.2f} win={b['win_pct']}% "
                             f"gross={b['gross_bps']}bps net={b['net_bps']}bps t_gross={b['t_gross']} "
                             f"net_ex_best5d={b['net_ex_best5_dates']:,.0f}")
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
