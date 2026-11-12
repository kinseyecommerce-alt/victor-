"""
stratlab/attention.py — attention-driven overnight long (NIFTY-100 stock futures), pre-registered (swing round 2).

SWING-ATTN-ON-3X. Mechanism: stocks with abnormal volume and a rising price draw retail attention; that demand is
filled at the next morning's opening call auction, so the overnight return is high (and reverses intraday). The
trade provides liquidity to the next morning's buyers: buy late on day t, sell into the 09:15 auction of day t+1.

The positional simulator exits at session closes, so this module has its own fill model (`FILL_MODEL`, part of the
FINAL-ledger key) and goes through `protocol.evaluate` unchanged (same 70/30 split, H1 / H2 screen, PF >= 1.15,
>= 60 trades for a positional futures strategy, no symbol > 35 % of net, FINAL once via final_ledger.json).

Rules (fixed before any run; do not change them after seeing results)
  * universe: every stock in the data directory (the 97 NIFTY-100 F&O stocks; index files excluded); sessions with
    < 200 bars are dropped by data.py, so "previous session" is the previous FULL session of that symbol
  * decision bar = the 15:20 one-minute bar of day t (the last bar at or before 15:20, not earlier than 15:15)
  * signal: (a) cumulative volume 09:15..15:20 (decision bar included) > 3.0 x the mean FULL-DAY volume of the
    symbol's previous 20 sessions, and (b) close of the decision bar > the previous session's last close
  * exclusions: day t or t+1 a special / short session (median minute-bar count across symbols < 300, or a
    session dropped as < 200 bars between t and the symbol's next session); known corporate-action gap days
    (CA_GAPS, as in ltr.py) on t or t+1; a signal whose next session is after the end of the simulated period
  * entry: LONG at the open of the first bar after the decision bar (the 15:21 bar; must start by 15:24),
    + slippage
  * exit: the next session's 09:15 open (pre-open auction print, a pre-open order), - slippage. If the 09:15 bar
    is missing, the open of the first bar before 09:30. Sensitivity: the 09:16 bar open
  * qty = floor(1,00,000 / entry); at most 10 per day, the 10 largest volume ratios (ties by symbol); one position
    per symbol (it is always flat again by the next 15:21); no stop, no target, one night
  * costs: costs.round_trip_cost(..., "future") (stock futures simulated on the cash price: no basis, lot size or
    roll); 2 bps / side slippage, 5 bps reported. Sensitivity: CNC delivery costs ("equity_cnc")
Look-ahead: the signal for day t reads bars of earlier sessions and day t up to 15:20 only; the per-day ranking
reads the same-time volume ratios of all symbols; the exit reads only the first bar(s) of the next session.

    cd algotrader_v4
    python -m stratlab.attention --data <bars3y dir> --out <scratch dir> [--screen-only]
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

from . import data as D  # noqa: E402
from . import protocol as P  # noqa: E402
from . import sim  # noqa: E402
from .api import Spec  # noqa: E402
from .auction import block, quarters, tstat_by_date  # noqa: E402
from .ltr import CA_GAPS  # noqa: E402

DECISION_T = 15 * 60 + 20        # 15:20 bar
DECISION_EARLIEST = 15 * 60 + 15
ENTRY_LATEST = 15 * 60 + 24
OPEN_T = 9 * 60 + 15
OPEN2_T = 9 * 60 + 16
EXIT_LATEST = 9 * 60 + 30
LOOKBACK = 20
VOL_MULT = 3.0
SPECIAL_MEDIAN_BARS = 300
MAX_PER_DAY = 10
NOTIONAL = 100_000.0
DEFAULT_SLIPPAGE_BPS = 2.0
STRESS_SLIPPAGE_BPS = 5.0
FILL_MODEL = "attn1"             # bump if a fill rule here changes (part of the FINAL-ledger key)
COST_MODEL = f"{sim.COST_MODEL}+{FILL_MODEL}"


def attention_rule(cum_vol_1520: float, prev20_full_vols, close_1520: float, prev_close: float,
                   mult: float = VOL_MULT):
    """The signal: volume ratio (> mult) or None. prev20_full_vols = full-day volumes of the previous 20 sessions."""
    pv = np.asarray(prev20_full_vols, float)
    if len(pv) < LOOKBACK or not (prev_close > 0 and close_1520 > 0):
        return None
    base = float(pv[-LOOKBACK:].mean())
    if not base > 0:
        return None
    ratio = float(cum_vol_1520) / base
    if ratio > mult + 1e-12 and close_1520 > prev_close:
        return ratio
    return None


SPECS = [
    Spec("SWING-ATTN-ON-3X", "futures", "attention overnight",
         "Volume 09:15-15:20 > 3x the 20-session mean daily volume and 15:20 close > previous close: buy the stock "
         "future at the 15:21 open, sell at the next 09:15 auction; top 10 volume ratios per day, Rs 1 lakh each",
         attention_rule,
         {"vol_mult": VOL_MULT, "lookback": LOOKBACK, "max_per_day": MAX_PER_DAY, "exit": "09:15",
          "special_median_bars": SPECIAL_MEDIAN_BARS, "costs": "future"},
         horizon="positional", decision_time=DECISION_T),
]


def session_features(sd) -> list:
    """Per session k of one symbol where the signal fires: dict with the decision, entry and both exit prints.
    Day-t fields read sessions <= k up to the bar after 15:20; exit fields read only the first bars of k+1."""
    out = []
    dv = np.add.reduceat(sd.v, sd.starts) if sd.n_sessions else np.zeros(0)
    for k in range(LOOKBACK, sd.n_sessions):
        a, b = int(sd.starts[k]), int(sd.ends[k])
        tt = sd.t[a:b]
        j = int(np.searchsorted(tt, DECISION_T, side="right")) - 1      # last bar <= 15:20
        if j < 0 or int(tt[j]) < DECISION_EARLIEST or j + 1 >= len(tt) or int(tt[j + 1]) > ENTRY_LATEST:
            continue
        pc = float(sd.c[int(sd.ends[k - 1]) - 1])
        ratio = attention_rule(float(sd.v[a:a + j + 1].sum()), dv[k - LOOKBACK:k], float(sd.c[a + j]), pc)
        if ratio is None:
            continue
        row = {"date": int(sd.days[k]), "symbol": sd.symbol, "ratio": ratio, "entry_t": int(tt[j + 1]),
               "entry_px": float(sd.o[a + j + 1]), "c1520": float(sd.c[a + j]), "prev_close": pc,
               "exit_date": None, "exit_t": -1, "exit_px": float("nan"), "exit2_t": -1, "exit2_px": float("nan")}
        if k + 1 < sd.n_sessions:
            a2, b2 = int(sd.starts[k + 1]), int(sd.ends[k + 1])
            t2 = sd.t[a2:b2]
            row["exit_date"] = int(sd.days[k + 1])
            if int(t2[0]) < EXIT_LATEST:
                row["exit_t"], row["exit_px"] = int(t2[0]), float(sd.o[a2])
            i2 = int(np.searchsorted(t2, OPEN2_T))
            if i2 < len(t2) and int(t2[i2]) < EXIT_LATEST:
                row["exit2_t"], row["exit2_px"] = int(t2[i2]), float(sd.o[a2 + i2])
        out.append(row)
    return out


def special_days(bar_counts: dict, all_dates) -> set:
    """Dates whose median minute-bar count across symbols is < SPECIAL_MEDIAN_BARS (a dropped session counts 0)."""
    return {d for d in all_dates if float(np.median(bar_counts.get(d, [0]))) < SPECIAL_MEDIAN_BARS}


def eligible(row, all_dates_sorted, special: set, ca=CA_GAPS) -> bool:
    """Exclusions: no next session; next session not the next market date; special t / t+1; CA gap on t / t+1."""
    d, nd = row["date"], row["exit_date"]
    if nd is None or not np.isfinite(row["exit_px"]):
        return False
    i = int(np.searchsorted(all_dates_sorted, d, side="right"))
    if i >= len(all_dates_sorted) or int(all_dates_sorted[i]) != nd:
        return False                       # a dropped / missing session in between: not a one-night hold
    if d in special or nd in special:
        return False
    return (row["symbol"], d) not in ca and (row["symbol"], nd) not in ca


def settle(spec_id: str, row, slip: float, exit_key: str = "exit", kind: str = "future") -> tuple:
    """One LONG trade in sim.TRADE_FIELDS layout; slippage against us on both fills."""
    qty = max(1, int(NOTIONAL // row["entry_px"]))
    fill = row["entry_px"] * (1.0 + slip)
    xp = row[f"{exit_key}_px"] * (1.0 - slip)
    gross = (xp - fill) * qty
    cost = round_trip_cost(fill * qty, xp * qty, kind)
    return (spec_id, row["symbol"], int(row["date"]), 1, sim._hm(row["entry_t"]), round(fill, 4),
            int(row["exit_date"]), sim._hm(row[f"{exit_key}_t"]), round(xp, 4), qty, round(gross, 2), cost,
            round(gross - cost, 2), "time")


def select(rows, spec: Spec, lo: int, hi: int) -> list:
    """Per day in [lo, hi] whose exit is also <= hi: the `max_per_day` largest volume ratios (ties by symbol)."""
    by_day = defaultdict(list)
    for r in rows:
        if lo <= r["date"] <= hi and r["exit_date"] <= hi:
            by_day[r["date"]].append(r)
    picked = []
    for d in sorted(by_day):
        picked.extend(sorted(by_day[d], key=lambda r: (-r["ratio"], r["symbol"]))[: spec.params["max_per_day"]])
    return picked


def make_simulator(rows, slippage_bps: float = DEFAULT_SLIPPAGE_BPS, exit_key: str = "exit", kind: str = "future"):
    def simulate(specs, lo, hi, slip_bps=None):
        slip = (slippage_bps if slip_bps is None else slip_bps) / 1e4
        out = {}
        for sp in specs:
            sel = [r for r in select(rows, sp, lo, hi) if np.isfinite(r[f"{exit_key}_px"])]
            tr = sorted((settle(sp.id, r, slip, exit_key, kind) for r in sel), key=lambda t: (t[2], t[1]))
            out[sp.id] = {"trades": tr, "skipped": 0, "lookahead": None, "error": None}
        return out
    return simulate


def load_universe(data_dir: str, cache: str | None = None, symbols=None, verbose: bool = True):
    """Symbol by symbol: signal rows (before exclusions), the session calendar, bar counts per date."""
    man = D.prepare(data_dir, cache or None, verbose=verbose)
    cdir = man["cache_dir"]
    syms = sorted(man["symbols"])
    if symbols:
        syms = syms[:symbols]
    rows, cal, counts, n = [], set(), defaultdict(list), 0
    for s in syms:
        sd = D.load_symbol(cdir, s)
        if sd is None:
            continue
        n += 1
        cal.update(int(x) for x in sd.days)
        for d, nb in zip(sd.days, sd.ends - sd.starts):
            counts[int(d)].append(int(nb))
        for d in sd.dropped or []:
            counts[int(d)].append(0)
        rows.extend(session_features(sd))
        del sd
    return rows, sorted(cal), counts, n, cdir


def filter_rows(rows, counts) -> tuple[list, set]:
    all_dates = np.array(sorted(counts), dtype=np.int64)
    special = special_days(counts, all_dates.tolist())
    return [r for r in rows if eligible(r, all_dates, special)], special


# ── reporting ───────────────────────────────────────────────────────────────
def extra(tr) -> dict:
    """Top-symbol share of net, long/short split, date-clustered t of net and gross."""
    if not tr:
        return {}
    by = defaultdict(float)
    for t in tr:
        by[t[1]] += t[12]
    net = sum(by.values())
    top = max(by.items(), key=lambda kv: kv[1])
    ls = {s: {"n": sum(1 for t in tr if t[3] == s), "net": round(sum(t[12] for t in tr if t[3] == s), 0)}
          for s in (1, -1)}
    return {"top_symbol": top[0], "top_share": round(top[1] / net, 3) if net > 0 else None,
            "long": ls[1], "short": ls[-1], "t_net_by_date": round(tstat_by_date(tr, 12), 2),
            "t_gross_by_date": round(tstat_by_date(tr, 10), 2)}


def _fmt(k, b):
    return (f"  {k:14s} n={b['trades']:4d} net={b['net']:>10,.0f} PF={b['pf']:.2f} win={b['win_pct']}% "
            f"gross={b['gross_bps']}bps net={b['net_bps']}bps t_gross={b['t_gross']} "
            f"net_ex_best5d={b['net_ex_best5_dates']:,.0f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Attention overnight long through the lab protocol")
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
    raw, calendar, counts, n_sym, cdir = load_universe(a.data, a.cache, a.symbols)
    rows, special = filter_rows(raw, counts)
    simulate = make_simulator(rows, a.slippage_bps)
    results, trades = P.evaluate(SPECS, calendar, simulate, n_symbols=n_sym,
                                 ledger_path=os.path.join(cdir, "final_ledger.json"),
                                 dataset=os.path.realpath(a.data), screen_only=a.screen_only, cost_model=COST_MODEL)
    split = results[0]["split"]
    sens = {"stress_5bps": make_simulator(rows, a.stress_bps),
            "exit_0916": make_simulator(rows, a.slippage_bps, exit_key="exit2"),
            "cnc_costs": make_simulator(rows, a.slippage_bps, kind="equity_cnc")}
    report = {"split": {"h1": split.h1, "h2": split.h2, "final": split.final, "n_sessions": split.n},
              "n_symbols": n_sym, "slippage_bps": a.slippage_bps, "stress_bps": a.stress_bps,
              "cost_model": COST_MODEL, "special_days": sorted(special),
              "signals_before_exclusions": len(raw), "signals_after_exclusions": len(rows), "strategies": {}}
    lines = [f"Attention overnight long | {n_sym} symbols, {split.n} sessions | H1 {split.h1} H2 {split.h2} "
             f"FINAL {split.final} | slippage {a.slippage_bps} bps/side (stress {a.stress_bps}) | cost model "
             f"{COST_MODEL} | special days {len(special)}"]
    for r in results:
        sid, sp = r["id"], r["spec"]
        tr = trades[sid]
        disc = [t for t in tr if split.discovery[0] <= t[2] <= split.discovery[1]]
        h1 = [t for t in disc if t[2] <= split.h1[1]]
        h2 = [t for t in disc if t[2] >= split.h2[0]]
        fin = [t for t in tr if t[2] >= split.final[0]]
        rec = {"verdict": r["verdict"], "why": r["why"], "final_note": r.get("final_note", ""),
               "top_symbol": r["screen"]["top_symbol"], "top_share": round(float(r["screen"]["top_share"]), 3),
               "H1": block(h1), "H2": block(h2), "DISC": block(disc), "DISC_extra": extra(disc),
               "DISC_quarters": quarters(disc)}
        for name, s in sens.items():
            st = s([sp], split.discovery[0], split.discovery[1])[sid]["trades"]
            sc = P.screen(st, split, sp.horizon)
            rec[f"DISC_{name}"] = {**block(st), "screen_passed": sc["passed"], "reasons": sc["reasons"],
                                   "H1_net": sc["h1"]["net"], "H2_net": sc["h2"]["net"]}
        if r["final"] is not None:           # FINAL legitimately evaluated (screen passer); sensitivities re-cost it
            rec["FINAL"] = block(fin)
            rec["FINAL_extra"] = extra(fin)
            rec["FINAL_quarters"] = quarters(fin)
            for name, s in sens.items():
                rec[f"FINAL_{name}"] = block(s([sp], split.final[0], split.final[1])[sid]["trades"])
        report["strategies"][sid] = rec
        with open(os.path.join(out, f"trades_{sid}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(sim.TRADE_FIELDS)
            w.writerows(tr)
        lines.append(f"\n{sid}: {r['verdict']} ({r['why']}) {rec['final_note']}")
        for k in ("H1", "H2", "DISC", "DISC_stress_5bps", "DISC_exit_0916", "DISC_cnc_costs",
                  "FINAL", "FINAL_stress_5bps", "FINAL_exit_0916", "FINAL_cnc_costs"):
            if k in rec:
                lines.append(_fmt(k, rec[k]) + (f" screen={'PASS' if rec[k]['screen_passed'] else 'FAIL'}"
                                                if "screen_passed" in rec[k] else ""))
        lines.append(f"  top symbol {rec['top_symbol']} {rec['top_share']:.1%} of DISCOVERY net; "
                     f"DISC extra {rec['DISC_extra']}")
        lines.append(f"  DISC quarters: {rec['DISC_quarters']}")
        if "FINAL" in rec:
            lines.append(f"  FINAL extra: {rec['FINAL_extra']}")
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
