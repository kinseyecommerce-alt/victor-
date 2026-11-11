"""
stratlab/run.py — command line.

    cd algotrader_v4
    python -m stratlab.run --data <bars dir> --kind intraday --out <scratch dir>
    python -m stratlab.run --data <bars dir> --kind scalping --ids scalp_vwap_touch,scalp_orb5 --symbols 20 --out <dir>

Writes into --out (which must be outside the repository):
    results.csv        one row per strategy
    summary.txt        the readable table, the protocol and the luck reminder
    data_quality.json  per-symbol checks of the input data
    trades/<id>.csv    every simulated trade (FINAL trades only where FINAL was evaluated)
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

from . import catalog  # noqa: F401  (imports every strategy module)
from . import data as D
from . import protocol as P
from . import sim
from .api import KINDS, REGISTRY

_CTX: dict = {}


def _scan(symbol):
    sd = D.load_symbol(_CTX["cache"], symbol)
    if sd is None:
        return symbol, None, []
    return symbol, D.quality_report(sd), [int(x) for x in sd.days]


def _work(args):
    symbol, ids, lo, hi = args
    return symbol, sim.run_symbol(_CTX["cache"], symbol, [REGISTRY[i] for i in ids], lo, hi)


def make_simulator(pool, symbols, timing: dict):
    def simulate(specs, lo, hi):
        ids = [sp.id for sp in specs]
        merged = {i: {"trades": [], "skipped": 0, "lookahead": None, "error": None, "sessions": 0} for i in ids}
        t0 = time.time()
        for _sym, res in pool.imap_unordered(_work, [(s, ids, lo, hi) for s in symbols]):
            for i, r in res.items():
                m = merged[i]
                m["trades"].extend(r["trades"])
                m["skipped"] += r["skipped"]
                m["sessions"] += r["sessions"]
                m["lookahead"] = m["lookahead"] or r["lookahead"]
                m["error"] = m["error"] or r["error"]
                timing[i] = timing.get(i, 0.0) + r["secs"]
                timing.setdefault("_sessions", {})[i] = timing.get("_sessions", {}).get(i, 0) + r["sessions"]
        for m in merged.values():
            m["trades"].sort(key=lambda t: (t[2], t[4], t[1]))
        timing["_wall"] = timing.get("_wall", 0.0) + time.time() - t0
        return merged
    return simulate


def _f(x, nd=0):
    if x is None:
        return ""
    if x == float("inf"):
        return "inf"
    return f"{x:,.{nd}f}"


def write_outputs(out, results, trades_by_id, timing, meta, quality, problems):
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    rows = []
    for r in results:
        sp, sc, fv = r["spec"], r["screen"], r["final"]
        d = sc["disc"] if sc else {}
        rows.append({
            "id": sp.id, "kind": sp.kind, "horizon": sp.horizon, "family": sp.family, "description": sp.desc,
            "params": json.dumps(sp.params, sort_keys=True),
            "disc_trades": d.get("trades", 0), "disc_win_pct": d.get("win_pct", ""), "disc_pf": d.get("pf", ""),
            "disc_net": d.get("net", ""), "h1_net": sc["h1"]["net"] if sc else "",
            "h2_net": sc["h2"]["net"] if sc else "",
            "top_symbol": sc["top_symbol"] if sc else "",
            "top_symbol_share": round(sc["top_share"], 3) if sc and sc["top_share"] == sc["top_share"] else "",
            "skipped_entries": r["skipped"], "verdict": r["verdict"], "why": r["why"],
            "final_trades": fv["trades"] if fv else "", "final_win_pct": fv["win_pct"] if fv else "",
            "final_pf": fv["pf"] if fv else "", "final_net": fv["net"] if fv else "",
            "final_note": r.get("final_note", ""),
            "cpu_secs": round(timing.get(sp.id, 0.0), 2),
        })
        with open(os.path.join(out, "trades", sp.id + ".csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(sim.TRADE_FIELDS)
            w.writerows(trades_by_id.get(sp.id, []))
    if rows:
        with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out, "data_quality.json"), "w") as fh:
        json.dump({"problems": problems, "symbols": quality}, fh, indent=1)

    L = []
    sp_ = results[0].get("split") if results else None
    L.append(f"STRATLAB  kind={meta['kind']}  data={meta['data']}")
    L.append(f"symbols={meta['n_symbols']}  sessions={meta['n_sessions']} "
             f"({meta['first']}..{meta['last']})  strategies={len(results)}")
    if sp_:
        L.append(f"DISCOVERY {sp_.discovery[0]}..{sp_.discovery[1]} ({sp_.n_discovery} sessions; "
                 f"H1 to {sp_.h1[1]}, H2 from {sp_.h2[0]})   FINAL {sp_.final[0]}..{sp_.final[1]} "
                 f"({sp_.n_final} sessions)")
    L.append("All figures are after costs and 5 bps slippage per market fill, ₹1,00,000 notional per trade.")
    L.append("")
    hdr = (f"{'id':<26}{'trades':>7}{'win%':>6}{'PF':>6}{'net H1':>11}{'net H2':>11}{'net disc':>11}"
           f"  {'verdict':<18}{'FINAL tr':>9}{'PF':>6}{'net':>11}  why")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in results:
        sc, fv = r["screen"], r["final"]
        if sc:
            a = (f"{r['id']:<26}{sc['disc']['trades']:>7}{sc['disc']['win_pct']:>6.1f}{_f(sc['disc']['pf'], 2):>6}"
                 f"{_f(sc['h1']['net']):>11}{_f(sc['h2']['net']):>11}{_f(sc['disc']['net']):>11}")
        else:
            a = f"{r['id']:<26}{'':>52}"
        b = (f"{fv['trades']:>9}{_f(fv['pf'], 2):>6}{_f(fv['net']):>11}" if fv else f"{'hidden':>9}{'':>17}")
        note = r["why"] + (f" | {r['final_note']}" if r.get("final_note") else "")
        L.append(f"{a}  {r['verdict']:<18}{b}  {note}")
    L.append("")
    L.append(P.luck_reminder(results))
    L.append("")
    L.append("Protocol (pre-registered, see stratlab/protocol.py): screen on DISCOVERY = net > 0 in H1 and H2, "
             f"PF >= {P.SCREEN_MIN_PF}, trades >= {P.SCREEN_MIN_TRADES['intraday']} intraday / "
             f"{P.SCREEN_MIN_TRADES['positional']} positional, top symbol <= {P.SCREEN_MAX_SYMBOL_SHARE:.0%} of "
             f"net. FINAL only for screen passers: CONFIRMED = net > 0 and PF >= {P.FINAL_MIN_PF}. "
             f"No verdict below {P.MIN_SESSIONS} sessions.")
    L.append("")
    L.append("Runtime (CPU seconds summed over symbols, strategy + simulator, excluding data loading):")
    for r in results:
        i = r["id"]
        ns = timing.get("_sessions", {}).get(i, 0)
        L.append(f"  {i:<26}{timing.get(i, 0.0):>8.1f}s  {ns:>7} symbol-sessions"
                 + (f"  {1e6 * timing.get(i, 0.0) / ns:>7.0f} us/session" if ns else ""))
    L.append(f"  wall clock for simulation: {timing.get('_wall', 0.0):.1f}s on {meta['jobs']} processes; "
             f"data scan/parse {meta['prep_secs']:.1f}s")
    if problems or meta.get("quality_notes"):
        L.append("")
        L.append("Data notes:")
        for x in (problems + meta.get("quality_notes", []))[:40]:
            L.append("  " + x)
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text)
    return text


def quality_notes(quality: list, calendar: list) -> list:
    notes = []
    if not quality:
        return notes
    ncal = len(calendar)
    short = sorted({d for q in quality for d in q["short_sessions"]})
    if short:
        notes.append(f"short sessions (< {D.MIN_SESSION_BARS} bars, dropped): {short[:10]}")
    early = sum(1 for q in quality if q["ends_before_1515"])
    if early:
        mx = max(q["ends_before_1515"] for q in quality)
        notes.append(f"{early} symbols have sessions whose bars stop before 15:15 (up to {mx} sessions each): "
                     f"those days are squared off at the last available close")
    few = [f"{q['symbol']}({q['sessions']})" for q in quality if q["sessions"] < 0.9 * ncal]
    if few:
        notes.append(f"symbols with < 90% of the {ncal} sessions: {', '.join(few[:15])}")
    bad = [f"{q['symbol']}({q['bad_ohlc_bars']})" for q in quality if q["bad_ohlc_bars"]]
    if bad:
        notes.append(f"bars with inconsistent OHLC: {', '.join(bad[:15])}")
    jumps = [f"{q['symbol']}@{q['overnight_jumps_gt_15pct']}" for q in quality if q["overnight_jumps_gt_15pct"]]
    if jumps:
        notes.append(f"overnight moves > 15% (possible unadjusted split/bonus/demerger): {', '.join(jumps[:15])}")
    miss = sorted(((q["missing_minutes"], q["symbol"]) for q in quality if q["missing_minutes"] > 50), reverse=True)
    if miss:
        notes.append("missing minutes inside sessions: " + ", ".join(f"{s}({m})" for m, s in miss[:10]))
    zv = sorted(((q["zero_volume_bars"] / max(q["bars"], 1), q["symbol"]) for q in quality), reverse=True)
    if zv and zv[0][0] > 0.02:
        notes.append("zero-volume bars: " + ", ".join(f"{s}({100 * f:.1f}%)" for f, s in zv[:6] if f > 0.02))
    return notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="", help="directory of bar JSON files (either layout)")
    ap.add_argument("--kind", required=True, choices=KINDS + ("all",))
    ap.add_argument("--ids", default="", help="comma-separated strategy ids (default: every strategy of the kind)")
    ap.add_argument("--symbols", type=int, default=0, help="use only the first N symbols (alphabetical)")
    ap.add_argument("--out", default="", help="output directory (outside the repository)")
    ap.add_argument("--cache", default="", help="cache directory for parsed bars (outside the repository)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--peek-final", action="store_true",
                    help="ALSO evaluate FINAL for strategies that failed the screen. This burns the hold-out: "
                         "the peek is recorded and those strategies can never be CONFIRMED on this data.")
    ap.add_argument("--list", action="store_true", help="list the strategies of the kind and exit")
    a = ap.parse_args(argv)

    specs = [sp for sp in REGISTRY.values() if a.kind in ("all", sp.kind)]
    if a.ids:
        want = [x.strip() for x in a.ids.split(",") if x.strip()]
        unknown = [x for x in want if x not in REGISTRY]
        if unknown:
            print(f"unknown strategy ids: {unknown}", file=sys.stderr)
            return 2
        specs = [sp for sp in specs if sp.id in want]
    specs.sort(key=lambda sp: (sp.kind, sp.id))
    if a.list:
        for sp in specs:
            print(f"{sp.id:<26}{sp.kind:<10}{sp.family:<16}{sp.desc}")
        return 0
    if not specs:
        print("no strategies selected", file=sys.stderr)
        return 2

    if not a.data or not a.out:
        ap.error("--data and --out are required")
    out = D.assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    man = D.prepare(a.data, a.cache or None, verbose=True)
    cache = man["cache_dir"]
    symbols = sorted(man["symbols"])
    if a.symbols:
        symbols = symbols[:a.symbols]
    if not symbols:
        print("no symbols found in --data", file=sys.stderr)
        return 2
    _CTX["cache"] = cache
    with mp.get_context("fork").Pool(a.jobs) as pool:
        quality, cal = [], set()
        for _sym, q, days in pool.imap(_scan, symbols):
            if q:
                quality.append(q)
                cal.update(days)
        calendar = sorted(cal)
        prep = time.time() - t0
        timing: dict = {}
        simulate = make_simulator(pool, symbols, timing)
        results, trades_by_id = P.evaluate(specs, calendar, simulate, n_symbols=len(symbols),
                                           ledger_path=os.path.join(cache, "final_ledger.json"),
                                           peek_final=a.peek_final)
    has_index = "_NIFTY50_min" in man["aux"]
    for r in results:
        if r["spec"].needs_index and not has_index:
            r["why"] = "needs NIFTY 50 minute bars, which this data does not have; " + r["why"]
    problems = list(man["problems"])
    if not has_index:
        problems.append("no _NIFTY50_min.json in this data: strategies that need the index made no trades")
    meta = {"kind": a.kind, "data": os.path.realpath(a.data), "n_symbols": len(symbols),
            "n_sessions": len(calendar), "first": calendar[0] if calendar else "", "jobs": a.jobs,
            "last": calendar[-1] if calendar else "", "prep_secs": prep,
            "quality_notes": quality_notes(quality, calendar)}
    text = write_outputs(out, results, trades_by_id, timing, meta, quality, problems)
    print(text)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
