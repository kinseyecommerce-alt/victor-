"""
stratlab/audit_run.py — drivers of the Nov 2026 simulator audit and corrected-data re-run (research only; every
output goes to <scratchpad>/audit, never into the repository).

    python -m stratlab.audit_run compare   [--symbols N]      # refsim vs sim.py, trade by trade (DISCOVERY only)
    python -m stratlab.audit_run handcheck                    # 30 random trades against the raw JSON rows
    python -m stratlab.audit_run lookahead                    # truncated-view + one-bar-shift tests
    python -m stratlab.audit_run bias                         # tie / gap / slippage accounting of sim.py fills
    python -m stratlab.audit_run catalog --mode net|gross|mirror [--data orig|corrected] [--period disc|final]
    python -m stratlab.audit_run benchmarks
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np

from . import data as D
from . import refsim as R

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
SCRATCH = os.environ.get("STRATLAB_SCRATCH", _SCRATCH)
AUDIT = os.path.join(SCRATCH, "audit")
BARS3Y = os.path.join(SCRATCH, "bars3y")
CACHE3Y = os.path.join(SCRATCH, "stratlab_cache", "bars3y_fb3d4d")
DISC = (20231110, 20251216)          # protocol DISCOVERY of the 726-session 3-year calendar (= tfgrid SPLIT3)
FINAL = (20251217, 20261109)
H1 = (20231110, 20241129)
H2 = (20241202, 20251216)
FIELDS = ("id", "symbol", "date", "side", "entry_t", "entry", "exit_date", "exit_t", "exit", "qty", "gross",
          "cost", "net", "reason")


def log(msg: str, fn: str = "progress.txt"):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    os.makedirs(AUDIT, exist_ok=True)
    with open(os.path.join(AUDIT, fn), "a") as fh:
        fh.write(line + "\n")


def stock_symbols(cache=CACHE3Y) -> list:
    with open(os.path.join(cache, "_manifest.json")) as fh:
        return sorted(s for s in json.load(fh)["symbols"] if not s.startswith("_"))


# ══════════════════════════════════════════════════════════════════════════
# 1. refsim vs sim.py
# ══════════════════════════════════════════════════════════════════════════
REF_IDS = list(R.INTRADAY) + list(R.POSITIONAL)
SIM_IDS = sorted({i.split("@")[0] for i in REF_IDS})
_NIFTY: dict = {}


def _nifty_map():
    if not _NIFTY:
        df = R.read_bars(os.path.join(BARS3Y, "_NIFTY50_min.json"))
        for d, g in df.groupby("date"):
            _NIFTY[int(d)] = (g["t"].to_numpy(int), g["c"].to_numpy(float))
    return _NIFTY


def _cmp_work(sym):
    from . import catalog  # noqa: F401
    from . import sim
    from .api import REGISTRY
    df = R.read_bars(os.path.join(BARS3Y, sym + ".json"))
    ss = R.sessions(df)
    ref = R.run_symbol(sym, ss, REF_IDS, DISC[0], DISC[1], nifty=_nifty_map())
    res = sim.run_symbol(CACHE3Y, sym, [REGISTRY[i] for i in SIM_IDS], DISC[0], DISC[1], audit=False)
    simt = {}
    for i in REF_IDS:                       # an "@code" reference variant is compared with its base id's sim trades
        simt[i] = [dict(zip(FIELDS, (i,) + tuple(t[1:]))) for t in res[i.split("@")[0]]["trades"]]
    errs = {i: r["error"] for i, r in res.items() if r["error"]}
    counts = defaultdict(int)
    for d, n in df.groupby("date").size().items():
        counts[int(d)] = int(n)
    return sym, ref, simt, errs, dict(counts), R.auction_rows(sym, ss), R.attention_rows(sym, ss)


def _key(t):
    return (t["id"], t["symbol"], int(t["date"]), int(t["side"]), str(t["entry_t"]))


def classify(a: dict, b: dict) -> list:
    """Field-level differences between a refsim trade and a sim trade with the same key."""
    diff = []
    if int(a["exit_date"]) != int(b["exit_date"]):
        diff.append("exit_date")
    if a.get("exit_t") and b.get("exit_t") and a["exit_t"] != b["exit_t"]:
        diff.append("exit_t")
    if abs(a["entry"] - b["entry"]) > 1e-6 * max(1.0, abs(b["entry"])) + 1e-4:
        diff.append("entry")
    if abs(a["exit"] - b["exit"]) > 1e-6 * max(1.0, abs(b["exit"])) + 1e-4:
        diff.append("exit")
    if int(a["qty"]) != int(b["qty"]):
        diff.append("qty")
    for f in ("gross", "cost", "net"):
        if abs(a[f] - b[f]) > 0.011:
            diff.append(f)
    return diff


def cmd_compare(a):
    from . import auction, attention
    syms = stock_symbols()[: a.symbols or None]
    out = os.path.join(AUDIT, "refsim")
    os.makedirs(out, exist_ok=True)
    ref, simt, errs, counts, arows, trows = defaultdict(list), defaultdict(list), {}, defaultdict(list), [], []
    t0 = time.time()
    with mp.get_context("fork").Pool(4, maxtasksperchild=1) as pool:
        for sym, r, s, e, cnt, ar, tr in pool.imap_unordered(_cmp_work, syms):
            for i in REF_IDS:
                ref[i].extend(r[i])
                simt[i].extend(s[i])
            errs.update({f"{sym}:{k}": v for k, v in e.items()})
            for d, n in cnt.items():
                counts[d].append(n)
            arows.extend(ar)
            trows.extend(tr)
    log(f"compare: {len(syms)} symbols simulated by both engines in {time.time() - t0:.0f}s")
    # module strategies through their own modules
    rows, cal, n, cdir = auction.load_universe(BARS3Y, CACHE3Y, symbols=a.symbols or None, verbose=False)
    asim = auction.make_simulator(rows)(auction.SPECS, DISC[0], DISC[1])
    for sp in auction.SPECS:
        mg = sp.params["min_gap"]
        simt[sp.id] = [dict(zip(FIELDS, t)) for t in asim[sp.id]["trades"]]
        ref[sp.id] = R.auction_trades(arows, sp.id, mg, DISC[0], DISC[1])
    trow_m, cal2, cnts, n2, _ = attention.load_universe(BARS3Y, CACHE3Y, symbols=a.symbols or None, verbose=False)
    keep, _sp = attention.filter_rows(trow_m, cnts)
    tsim = attention.make_simulator(keep)(attention.SPECS, DISC[0], DISC[1])
    simt["SWING-ATTN-ON-3X"] = [dict(zip(FIELDS, t)) for t in tsim["SWING-ATTN-ON-3X"]["trades"]]
    ref["SWING-ATTN-ON-3X"] = R.attention_trades(trows, counts, DISC[0], DISC[1])
    report, samples = {}, {}
    tot = defaultdict(int)
    for sid in list(simt):
        A = {_key(t): t for t in ref[sid]}
        B = {_key(t): t for t in simt[sid]}
        both = set(A) & set(B)
        exact, fdiff = 0, defaultdict(int)
        bad = []
        for k in sorted(both):
            d = classify(A[k], B[k])
            if not d:
                exact += 1
            else:
                for f in d:
                    fdiff[f] += 1
                bad.append({"key": k, "diff": d, "ref": A[k], "sim": B[k]})
        only_ref = sorted(set(A) - set(B))
        only_sim = sorted(set(B) - set(A))
        n_union = len(set(A) | set(B))
        report[sid] = {"ref": len(A), "sim": len(B), "matched_key": len(both), "exact": exact,
                       "field_diffs": dict(fdiff), "only_ref": len(only_ref), "only_sim": len(only_sim),
                       "match_rate": round(exact / n_union, 4) if n_union else None,
                       "sim_net": round(sum(t["net"] for t in B.values()), 2),
                       "ref_net": round(sum(t["net"] for t in A.values()), 2),
                       "sim_gross": round(sum(t["gross"] for t in B.values()), 2),
                       "ref_gross": round(sum(t["gross"] for t in A.values()), 2)}
        samples[sid] = {"field": bad[:8], "only_ref": [A[k] for k in only_ref[:6]],
                        "only_sim": [B[k] for k in only_sim[:6]]}
        tot["union"] += n_union
        tot["exact"] += exact
    report["_total"] = {"trades_union": tot["union"], "exact": tot["exact"],
                        "match_rate": round(tot["exact"] / max(1, tot["union"]), 4), "sim_errors": errs,
                        "symbols": len(syms), "period": DISC}
    with open(os.path.join(out, "compare.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    with open(os.path.join(out, "mismatch_samples.json"), "w") as fh:
        json.dump(samples, fh, indent=1, default=str)
    with gzip.open(os.path.join(out, "trades.pkl.gz"), "wb") as fh:
        pickle.dump({"ref": dict(ref), "sim": dict(simt)}, fh)
    for sid, r in report.items():
        if sid != "_total":
            print(f"{sid:28} ref {r['ref']:6} sim {r['sim']:6} exact {r['exact']:6} rate {r['match_rate']} "
                  f"onlyR {r['only_ref']} onlyS {r['only_sim']} {r['field_diffs']}")
    log(f"compare total: {report['_total']['exact']}/{report['_total']['trades_union']} trades identical "
        f"({report['_total']['match_rate']:.2%})")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# 2. hand checks against the raw JSON rows
# ══════════════════════════════════════════════════════════════════════════
def cost_breakdown(bv: float, sv: float, product: str) -> tuple:
    """Zerodha-calculator style line items (own arithmetic, zerodha.com/charges 2026-11-15)."""
    if product == "MIS":
        brok = min(bv * 0.0003, 20) + min(sv * 0.0003, 20)
        stt, ex, stamp, dp = sv * 0.00025, (bv + sv) * 0.0000307, bv * 0.00003, 0.0
    elif product == "FUT":
        brok = min(bv * 0.0003, 20) + min(sv * 0.0003, 20)
        stt, ex, stamp, dp = sv * 0.0005, (bv + sv) * 0.0000183, bv * 0.00002, 0.0
    else:
        brok = 0.0
        stt, ex, stamp, dp = (bv + sv) * 0.001, (bv + sv) * 0.0000307, bv * 0.00015, 15.34
    sebi = (bv + sv) * 1e-6
    gst = 0.18 * (brok + ex + sebi)
    items = {"brokerage": brok, "STT": stt, "exchange": ex, "SEBI": sebi, "stamp": stamp, "GST": gst, "DP": dp}
    return items, round(sum(items.values()), 2)


def cmd_handcheck(a):
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    with gzip.open(os.path.join(AUDIT, "refsim", "trades.pkl.gz"), "rb") as fh:
        T = pickle.load(fh)["sim"]
    pool = [t for sid, tr in T.items() if "@" not in sid for t in tr]
    rng = random.Random(20261115)
    by = defaultdict(list)
    for t in pool:
        by[t["id"]].append(t)
    ids = sorted(by)
    pick = []
    while len(pick) < 30:                       # spread over strategies: round-robin over a shuffled id list
        rng.shuffle(ids)
        for i in ids:
            if len(pick) < 30:
                pick.append(rng.choice(by[i]))
    raw_cache = {}
    lines, ok_all = [], 0
    for n, t in enumerate(pick, 1):
        sym = t["symbol"]
        if sym not in raw_cache:
            raw_cache.clear()
            with open(os.path.join(BARS3Y, sym + ".json")) as fh:
                raw_cache[sym] = json.load(fh)["bars"]
        bars = raw_cache[sym]
        idx = {b[0][:16]: b for b in bars}
        sid = t["id"]
        if sid.startswith("AUC"):
            prod, slip = "MIS", 2.1e-4
        elif sid.startswith("SWING-ATTN"):
            prod, slip = "FUT", 2e-4
        else:
            sp = REGISTRY[sid]
            prod = {"swing": "CNC", "futures": "FUT"}.get(sp.kind, "MIS")
            slip = 1e-4 if sp.underlying else 2e-4
        d0, d1 = str(t["date"]), str(t["exit_date"])

        def iso(d, hhmm):
            return f"{d[:4]}-{d[4:6]}-{d[6:]}T{hhmm}"
        et = t["entry_t"] if t["entry_t"] not in ("order", "") else "09:15"
        eb = idx.get(iso(d0, et))
        lines.append(f"#{n} {sid} {sym} side {t['side']:+d} qty {t['qty']} entry {d0} {t['entry_t']} @ {t['entry']:.4f}"
                     f" -> exit {d1} {t['exit_t']} @ {t['exit']:.4f} ({t.get('reason', '')})")
        checks = []
        if eb:
            lines.append(f"    raw entry bar  {eb}")
            exp = eb[1] * (1 + t["side"] * slip)
            checks.append(("entry = raw open x (1 %+.4f %%)" % (t["side"] * slip * 100), abs(exp - t["entry"]) < 1e-3))
        r = t.get("reason", "")
        sgn = t["side"]
        day = [b for b in bars if b[0][:10] == iso(d1, "")[:10]]
        xb = idx.get(iso(d1, t["exit_t"])) if t["exit_t"] else None
        if xb:
            lines.append(f"    raw exit bar   {xb}")
        if r in ("squareoff", "time", "signal") and xb:
            exp = xb[1] * (1 - sgn * slip)
            checks.append(("exit = raw open x (1 -/+ slip) (market order)", abs(exp - t["exit"]) < 1e-3))
        elif r in ("max_hold", "period_end", "data_end") and day:
            lines.append(f"    raw last bar   {day[-1]}")
            exp = day[-1][4] * (1 - sgn * slip)
            checks.append(("exit = that day's last 1-min close x (1 -/+ slip)", abs(exp - t["exit"]) < 1e-3))
        elif r == "stop":
            src = [xb] if xb else day
            lo_, hi_ = min(b[3] for b in src), max(b[2] for b in src)
            raw_x = t["exit"] / (1 - sgn * slip)
            if not xb:
                lines.append(f"    raw exit day   first {day[0]}  low {lo_} high {hi_}")
            checks.append(("stop price (before slippage) inside the exit bar / day range",
                           lo_ - 1e-6 <= raw_x <= hi_ + 1e-6))
        elif r == "target":
            src = [xb] if xb else day
            thr = max(b[2] for b in src) if sgn > 0 else min(b[3] for b in src)
            if not xb:
                lines.append(f"    raw exit day   first {day[0]}  extreme {thr}")
            checks.append(("target traded through by >= 1 tick on the exit bar / day", (thr - t["exit"]) * sgn >= 0.01 - 1e-9))
        elif sid.startswith(("AUC", "SWING")) and xb:
            exp = xb[1] * (1 - sgn * slip)
            checks.append(("exit = raw open x (1 -/+ slip)", abs(exp - t["exit"]) < 1e-3))
        q = int(100000 // (eb[1] if (eb and sid.startswith(("AUC", "SWING"))) else t["entry"]))
        checks.append((f"qty = floor(100000 / price) = {q}", q == t["qty"]))
        g = t["side"] * (t["exit"] - t["entry"]) * t["qty"]
        checks.append((f"gross = side x (exit - entry) x qty = {g:.2f} (prices in the record are rounded to 4 dp)", abs(g - t["gross"]) <= t["qty"] * 1e-4 + 0.011))
        bv, sv = ((t["entry"] * t["qty"], t["exit"] * t["qty"]) if t["side"] > 0
                  else (t["exit"] * t["qty"], t["entry"] * t["qty"]))
        items, tot = cost_breakdown(bv, sv, prod)
        lines.append("    cost (" + prod + "): " + ", ".join(f"{k} {v:.2f}" for k, v in items.items() if v) +
                     f" = {tot:.2f}   lab {t['cost']:.2f}")
        checks.append(("cost = manual Zerodha computation (+-1 paisa)", abs(tot - t["cost"]) <= 0.011))
        allok = all(c[1] for c in checks)
        ok_all += allok
        for c in checks:
            lines.append(f"      [{'ok' if c[1] else 'FAIL'}] {c[0]}")
    lines.insert(0, f"HAND CHECKS: {ok_all}/30 trades pass every check (sampled with seed 20261115 from the sim.py "
                    f"trades of the refsim comparison, DISCOVERY 2023-11-10..2025-12-16; raw rows = bars3y JSON "
                    f"[ts, open, high, low, close, volume])\n")
    with open(os.path.join(AUDIT, "hand_checks.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    log(f"handcheck: {ok_all}/30 trades pass every check -> audit/hand_checks.txt")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# 3. look-ahead: truncated views and the one-bar shift
# ══════════════════════════════════════════════════════════════════════════
LA_INTRADAY = ["orb15_break", "vwap_reclaim", "scalp_vwap_bounce", "r3_tf5m_supertrend_flip", "r3_tf15m_st_flip",
               "r3_tf60m_macd_signal_trend", "cs_bull_engulfing_15m", "r5_kite_ab_vwap_60m", "r7_gh2_oops_stock",
               "r4_futoi_oi_buildup_long" ]
LA_POSITIONAL = ["swing_20d_breakout", "r3_swing_vix_capitulation", "r3_tfweekly_wk_loser_rev", "cs_tweezer_top_d",
                 "r5_kite_vars_macd_d", "r6_gh_wvf_stock", "r4_delivery_brk_fake_short", "r3_tfdaily_aroon_cross"]
CUT_DATE = 20250303


def _cheat_intraday(s, p):
    """PLANTED LOOK-AHEAD (control): at 10:30 go the way the price will be 30 bars later. Must be caught."""
    from .api import Entry
    i = s.idx(10 * 60 + 30)
    if i + 30 >= len(s.t):
        return []
    side = 1 if s.c[i + 30] > s.c[i] else -1
    return [Entry(i, side, stop=float(s.c[i]) * (1 - side * 0.02), time_stop=30)]


def _cheat_next_bar(s, p):
    """PLANTED ONE-BAR LOOK-AHEAD (shift-test control): at every 30th bar from 09:30 take the side of the NEXT bar's
    close, out one minute later. One bar of lag must destroy it."""
    from .api import Entry
    out = []
    for i in range(s.idx(9 * 60 + 30), min(len(s.t) - 2, s.idx(14 * 60 + 30)), 30):
        side = 1 if s.c[i + 1] > s.c[i] else -1
        out.append(Entry(i, side, stop=float(s.c[i]) * (1 - side * 0.02), time_stop=1))
    return out


def _la_specs():
    from . import catalog  # noqa: F401
    from .api import REGISTRY, Spec
    ids = [i for i in LA_INTRADAY if i in REGISTRY]
    for i in sorted(REGISTRY):                          # fill up to 10 intraday ids if a name above does not exist
        if len(ids) >= 10:
            break
        sp = REGISTRY[i]
        if sp.horizon == "intraday" and i not in ids and not sp.underlying and i.startswith(("r3_tf", "r5_kite_ab")):
            ids.append(i)
    cheat = Spec("CHEAT_LOOKAHEAD_30BARS", "intraday", "control", "planted look-ahead", _cheat_intraday, {})
    return [REGISTRY[i] for i in ids], [REGISTRY[i] for i in LA_POSITIONAL if i in REGISTRY], cheat


def _cheat1():
    from .api import Spec
    return Spec("CHEAT_NEXT_BAR", "intraday", "control", "planted one-bar look-ahead", _cheat_next_bar, {},
                max_per_day=20)


def _truncate(sd, before: int):
    keep = sd.date < before
    z = {"date": sd.date[keep], "t": sd.t[keep], "o": sd.o[keep], "h": sd.h[keep], "l": sd.l[keep],
         "c": sd.c[keep], "v": sd.v[keep]}
    return D._sessionize(sd.symbol, z)


def _la_work(sym):
    from . import sim
    from .indicators import View, build_sessions
    intr, pos, cheat = _la_specs()
    sd = D.load_symbol(CACHE3Y, sym)
    idx = D.load_symbol(CACHE3Y, "_NIFTY50_min")
    out = {"view": defaultdict(lambda: [0, 0, None]), "cut": defaultdict(lambda: [0, 0, None])}
    sessions = build_sessions(sd, idx)
    # (a) dense truncated views inside sessions: every 15th bar of every 7th session
    for sp in intr + [cheat]:
        for s in sessions[::7]:
            if not DISC[0] <= s.date <= DISC[1]:
                continue
            full = sp.fn(s, sp.p) or []
            n = len(s.t)
            for m in range(15, n, 15):
                a, b = sim._canon(full, m - 1), sim._canon(sp.fn(View(s, m), sp.p) or [], m - 1)
                r = out["view"][sp.id]
                r[0] += 1
                if a != b:
                    r[1] += 1
                    r[2] = r[2] or f"{sym} {s.date} cut {m}"
    # (b) the whole future removed: data cut before CUT_DATE; every trade that is complete before the cut must be
    #     identical with and without the later data
    cut_sd = _truncate(sd, CUT_DATE)
    full_res = sim.run_symbol(CACHE3Y, sym, intr + pos + [cheat], DISC[0], CUT_DATE - 1, audit=False, sd=sd, index=idx)
    cut_res = sim.run_symbol(CACHE3Y, sym, intr + pos + [cheat], DISC[0], CUT_DATE - 1, audit=False, sd=cut_sd,
                             index=idx)
    last = int(cut_sd.days[-1])
    for sp in intr + pos + [cheat]:
        A = [t for t in full_res[sp.id]["trades"] if t[6] < last]
        B = [t for t in cut_res[sp.id]["trades"] if t[6] < last]
        r = out["cut"][sp.id]
        r[0] += len(A)
        if A != B:
            r[1] += len(set(A) ^ set(B))
            r[2] = r[2] or f"{sym}: {sorted(set(A) ^ set(B))[:1]}"
    return sym, {k: dict(v) for k, v in out.items()}


def shift_spec(sp):
    """The same rule one bar (intraday: one minute) / one session (positional) LATE."""
    import dataclasses
    from .sim import DailyView
    fn0 = sp.fn
    if sp.horizon == "intraday":
        def fn(s, p):
            r = fn0(s, p) or []
            out = []
            for e in r:
                e2 = dataclasses.replace(e, i=int(e.i) + 1)
                out.append(e2)
            return out
    else:
        def fn(d, p):
            if len(d) < 3:
                return None
            v = DailyView(d.date[:-1], d.o[:-1], d.h[:-1], d.l[:-1], d.c[:-1], d.v[:-1])
            v.symbol = getattr(d, "symbol", None)
            v.nifty = None if d.nifty is None else d.nifty[:-1]
            v.vix = None if d.vix is None else d.vix[:-1]
            return fn0(v, p)
    return dataclasses.replace(sp, fn=fn)


def _shift_work(args):
    sym, ids, shifted = args
    from . import catalog  # noqa: F401
    from . import sim
    from .api import REGISTRY
    sim.round_trip_cost = lambda *a, **k: 0.0
    _i, _p, cheat = _la_specs()
    c1 = _cheat1()
    specs = [cheat if i == cheat.id else (c1 if i == c1.id else REGISTRY[i]) for i in ids]
    if shifted:
        specs = [shift_spec(sp) for sp in specs]
    res = sim.run_symbol(CACHE3Y, sym, specs, DISC[0], DISC[1], audit=False, slippage=0.0, index_slippage=0.0)
    return sym, {i: [t[12] for t in r["trades"]] for i, r in res.items()}


def _pf_of(x):
    x = np.asarray(x, float)
    gp, gl = x[x > 0].sum(), -x[x < 0].sum()
    return round(float(gp / gl), 3) if gl > 0 else None


def cmd_shiftcontrol(a):
    """the one-bar shift test of the planted one-bar look-ahead (added to lookahead.json)"""
    c1 = _cheat1()
    gross = {}
    with mp.get_context("fork").Pool(a.jobs, maxtasksperchild=1) as pool:
        for shifted in (False, True):
            acc = []
            for sym, o in pool.imap_unordered(_shift_work, [(s, [c1.id], shifted) for s in stock_symbols()]):
                acc.extend(o.get(c1.id, []))
            gross[shifted] = acc
    p = os.path.join(AUDIT, "lookahead.json")
    rep = json.load(open(p))
    g0, g1 = gross[False], gross[True]
    rep["shift"][c1.id] = {"trades": len(g0), "gross_pf": _pf_of(g0), "gross_per_trade": round(float(np.mean(g0)), 2),
                           "shifted_trades": len(g1), "shifted_pf": _pf_of(g1),
                           "shifted_per_trade": round(float(np.mean(g1)), 2)}
    json.dump(rep, open(p, "w"), indent=1)
    log(f"shift control {c1.id}: {rep['shift'][c1.id]}")
    return 0


def cmd_lookahead(a):
    syms = stock_symbols()[: a.symbols or 20]
    intr, pos, cheat = _la_specs()
    view, cut = defaultdict(lambda: [0, 0, None]), defaultdict(lambda: [0, 0, None])
    with mp.get_context("fork").Pool(a.jobs, maxtasksperchild=1) as pool:
        for sym, o in pool.imap_unordered(_la_work, syms):
            for k, acc in (("view", view), ("cut", cut)):
                for sid, (n, bad, ex) in o[k].items():
                    acc[sid][0] += n
                    acc[sid][1] += bad
                    acc[sid][2] = acc[sid][2] or ex
    # shift test on the full universe, zero cost / zero slippage
    ids = [sp.id for sp in intr + pos] + [cheat.id]
    gross = {}
    with mp.get_context("fork").Pool(a.jobs, maxtasksperchild=1) as pool:
        for shifted in (False, True):
            acc = defaultdict(list)
            for sym, o in pool.imap_unordered(_shift_work, [(s, ids, shifted) for s in stock_symbols()]):
                for i, x in o.items():
                    acc[i].extend(x)
            gross[shifted] = acc
    rep = {"truncated_view": {k: {"checks": v[0], "differences": v[1], "example": v[2]} for k, v in view.items()},
           "data_cut_before": CUT_DATE,
           "truncated_data": {k: {"trades_compared": v[0], "differences": v[1], "example": v[2]} for k, v in cut.items()},
           "shift": {}}
    for i in ids:
        g0, g1 = gross[False].get(i, []), gross[True].get(i, [])
        rep["shift"][i] = {"trades": len(g0), "gross_pf": _pf_of(g0), "gross_per_trade": round(float(np.mean(g0)), 2) if g0 else None,
                           "shifted_trades": len(g1), "shifted_pf": _pf_of(g1),
                           "shifted_per_trade": round(float(np.mean(g1)), 2) if g1 else None}
    with open(os.path.join(AUDIT, "lookahead.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    for k in ("truncated_view", "truncated_data"):
        for i, r in rep[k].items():
            print(k, i, r)
    for i, r in rep["shift"].items():
        print("shift", i, r)
    log("lookahead: written audit/lookahead.json")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# 3b. pessimism / optimism of the fill rules, measured with refsim variants
# ══════════════════════════════════════════════════════════════════════════
def _bias_work(sym):
    df = R.read_bars(os.path.join(BARS3Y, sym + ".json"))
    ss = R.sessions(df)
    ctx = R.context(ss, _nifty_map())
    out = defaultdict(lambda: defaultdict(float))
    ids = [i for i in R.INTRADAY if "@" not in i]
    for sid in ids:
        fn, mpd, prod = R.INTRADAY[sid]
        for s in ss:
            if not DISC[0] <= s.date <= DISC[1]:
                continue
            raw = fn(ctx, s)
            ents = [e[1] if isinstance(e, tuple) else e for e in raw]
            for e in raw:
                if isinstance(e, tuple):
                    _, ee, dist = e
                    if ee.i + 1 < len(s.t):
                        ee.target = s.o[ee.i + 1] + ee.side * dist
            for variant in ("lab", "target_first", "touch_fills", "no_gap"):
                tr = _variant_run(s, ents, mpd, variant)
                o = out[(sid, variant)]
                for g, why, extra in tr:
                    o["n"] += 1
                    o["gross"] += g
                    o["gp"] += max(g, 0)
                    o["gl"] += max(-g, 0)
                    o[why] += 1
                    o["extra"] += extra
    return sym, {f"{k[0]}|{k[1]}": dict(v) for k, v in out.items()}


def _variant_run(s, ents, mpd, variant):
    """Zero cost, zero slippage. lab = the documented rules; target_first = a bar touching both exits at the target;
    touch_fills = a target fills on a bare touch; no_gap = a stop gapped through fills at the stop, not the open.
    Returns [(gross per Rs 1 lakh, reason, 0)]."""
    out, free = [], 0
    for e in sorted(ents, key=lambda x: x.i):
        if len(out) >= mpd or e.i < free:
            continue
        j = e.i + 1
        n = len(s.t)
        if j >= n or s.t[j] >= R.SQ or e.stop is None or e.side * (s.o[j] - e.stop) <= 0:
            continue
        side, fill, stop = e.side, s.o[j], e.stop
        tgt = e.target if e.target is not None else (fill + side * e.rr * abs(fill - stop) if e.rr else None)
        if tgt is not None and side * (tgt - fill) <= 0:
            continue
        lim = s.idx(R.SQ)
        if e.hold:
            lim = min(lim, s.idx(int(s.t[j]) + int(e.hold)))
        tk = 0.0 if (variant == "touch_fills" or tgt is None) else R.tick_for(tgt, s.date)
        res = None
        for k in range(j, lim):
            hs = (s.l[k] <= stop) if side > 0 else (s.h[k] >= stop)
            ht = tgt is not None and ((s.h[k] >= tgt + tk - 1e-9) if side > 0 else (s.l[k] <= tgt - tk + 1e-9))
            if hs and ht and variant == "target_first":
                hs = False
            if hs:
                px = stop if variant == "no_gap" else (min(s.o[k], stop) if side > 0 else max(s.o[k], stop))
                why = "stop_tie" if ht else ("stop_gap" if px != stop else "stop")
                res = (k, px, why)
                break
            if ht:
                px = s.o[k] if side * (s.o[k] - tgt) >= tk - 1e-9 and tk > 0 else tgt
                res = (k, px, "target")
                break
        if res is None:
            k = lim if lim < n else n - 1
            res = (k, s.o[k] if lim < n else s.c[n - 1], "close")
        k, px, why = res
        out.append((side * (px - fill) / fill * 1e5, why, 0.0))
        free = k
    return out


def cmd_bias(a):
    syms = stock_symbols()[: a.symbols or None]
    acc = defaultdict(lambda: defaultdict(float))
    with mp.get_context("fork").Pool(a.jobs, maxtasksperchild=1) as pool:
        for sym, o in pool.imap_unordered(_bias_work, syms):
            for k, d in o.items():
                for f, v in d.items():
                    acc[k][f] += v
    rep = {}
    for k, d in sorted(acc.items()):
        sid, var = k.split("|")
        rep.setdefault(sid, {})[var] = {"trades": int(d["n"]), "gross_per_trade_rs": round(d["gross"] / max(d["n"], 1), 2),
                                        "pf": round(d["gp"] / d["gl"], 3) if d["gl"] else None,
                                        "stop_ties": int(d.get("stop_tie", 0)), "stop_gaps": int(d.get("stop_gap", 0))}
    tot = defaultdict(lambda: [0.0, 0])
    for sid, v in rep.items():
        for var, r in v.items():
            tot[var][0] += r["gross_per_trade_rs"] * r["trades"]
            tot[var][1] += r["trades"]
    rep["_all"] = {var: {"trades": n, "gross_per_trade_rs": round(g / max(n, 1), 2)} for var, (g, n) in tot.items()}
    with open(os.path.join(AUDIT, "fill_bias.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep["_all"], indent=1))
    log("bias: written audit/fill_bias.json")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# 5. benchmarks and random-entry controls
# ══════════════════════════════════════════════════════════════════════════
BH_STOCKS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "SBIN"]
SEEDS = (1, 2, 3, 4, 5)


def _rng(seed, sym, date):
    import zlib
    return np.random.default_rng([seed, zlib.crc32(sym.encode()), int(date)])


def _control_specs():
    from .api import Entry, Spec, Swing
    specs = []

    def bh(d, p):
        return Swing(1, max_hold=100000)
    specs.append(Spec("BH_STOCK", "swing", "control", "buy at the first open, hold to the period end", bh, {},
                      horizon="positional"))
    specs.append(Spec("BH_NIFTY", "futures", "control", "NIFTY future bought and held", bh, {},
                      horizon="positional", underlying="NIFTY"))
    for seed in SEEDS:
        def rt(s, p, seed=seed):
            r = _rng(seed, s.symbol, s.date)
            a, b = s.idx(9 * 60 + 30), s.idx(14 * 60 + 30)
            if b - a < 2 or r.random() > 0.3:
                return []
            i = int(r.integers(a, b))
            side = 1 if r.random() < 0.5 else -1
            return [Entry(i, side, stop=float(s.c[i]) * (1 - side * 0.05), time_stop=30)]

        def rb(s, p, seed=seed):
            r = _rng(seed + 100, s.symbol, s.date)
            a, b = s.idx(9 * 60 + 30), s.idx(14 * 60 + 30)
            if b - a < 2 or r.random() > 0.3:
                return []
            i = int(r.integers(a, b))
            side = 1 if r.random() < 0.5 else -1
            c = float(s.c[i])
            return [Entry(i, side, stop=c * (1 - side * 0.005), target=c * (1 + side * 0.005))]

        def rp(d, p, seed=seed):
            r = _rng(seed + 200, d.symbol, d.date[-1])
            return Swing(1, max_hold=5) if r.random() < 0.05 else None
        specs.append(Spec(f"RAND_TIME30_s{seed}", "intraday", "control", "random entry, 30-minute time exit", rt, {}))
        specs.append(Spec(f"RAND_BRACKET05_s{seed}", "intraday", "control", "random entry, +-0.5 % bracket", rb, {}))
        specs.append(Spec(f"RAND_SWING5_s{seed}", "swing", "control", "random entry, hold 5 sessions", rp, {},
                          horizon="positional"))
    return specs


def _bench_work(args):
    sym, zero = args
    from . import sim
    if zero:
        sim.round_trip_cost = lambda *a, **k: 0.0
    specs = [sp for sp in _control_specs() if (sp.id != "BH_STOCK" or sym in BH_STOCKS)]
    slip = 0.0 if zero else None
    res = sim.run_symbol(CACHE3Y, sym, specs, DISC[0], DISC[1], audit=False, slippage=slip, index_slippage=slip)
    out = {}
    for i, r in res.items():
        if r["trades"]:
            out[i] = [(t[1], t[2], t[5], t[6], t[8], t[9], t[10], t[11], t[12]) for t in r["trades"]]
        if r["error"]:
            out[i + "_error"] = r["error"]
    return sym, zero, out


def cmd_benchmarks(a):
    syms = stock_symbols() + ["_NIFTY50_min"]
    acc = {True: defaultdict(list), False: defaultdict(list)}
    with mp.get_context("fork").Pool(a.jobs, maxtasksperchild=1) as pool:
        for sym, zero, o in pool.imap_unordered(_bench_work, [(s, z) for s in syms for z in (True, False)]):
            for i, tr in o.items():
                acc[zero][i].extend(tr if isinstance(tr, list) else [tr])
    rep = {"period": DISC, "buy_and_hold": {}, "random": {}}
    # buy and hold vs the raw bars
    for t in acc[True]["BH_STOCK"] + acc[True]["BH_NIFTY"]:
        sym, d0, fill, d1, xp, qty, gross, cost, net = t
        df = R.read_bars(os.path.join(BARS3Y, sym + ".json"))
        ss = [s for s in R.sessions(df) if DISC[0] <= s.date <= DISC[1]]
        o0 = [s for s in ss if s.date == d0][0].o[0]
        c1 = [s for s in ss if s.date == d1][0].c[-1]
        raw = c1 / o0 - 1
        real = [x for x in acc[False]["BH_STOCK"] + acc[False]["BH_NIFTY"] if x[0] == sym][0]
        rep["buy_and_hold"][sym] = {"entry": d0, "exit": d1, "raw_return_pct": round(100 * raw, 3),
                                    "sim_gross_return_pct": round(100 * gross / (fill * qty), 3),
                                    "sim_net_return_pct_with_costs": round(100 * real[8] / (real[2] * real[5]), 3),
                                    "rs10L_equivalent_gross": round(raw * 1e6, 0)}
    nd = os.path.join(BARS3Y, "_NIFTY50_day.json")
    if os.path.exists(nd):
        with open(nd) as fh:
            js = json.load(fh)
        rows = {int(b[0][:4] + b[0][5:7] + b[0][8:10]): b for b in js["bars"]}
        ks = sorted(k for k in rows if DISC[0] <= k <= DISC[1])
        rep["buy_and_hold"]["NIFTY50_official_daily"] = {
            "first_close": rows[ks[0]][4], "last_close": rows[ks[-1]][4],
            "close_to_close_pct": round(100 * (rows[ks[-1]][4] / rows[ks[0]][4] - 1), 3),
            "open_day2_to_last_close_pct": round(100 * (rows[ks[-1]][4] / rows[ks[1]][1] - 1), 3)}
    # random controls
    for fam in ("RAND_TIME30", "RAND_BRACKET05", "RAND_SWING5"):
        for seed in SEEDS:
            i = f"{fam}_s{seed}"
            z, r = acc[True].get(i, []), acc[False].get(i, [])
            gz = np.array([t[6] for t in z])
            nr = np.array([t[8] for t in r])
            cr = np.array([t[7] for t in r])
            gr = np.array([t[6] for t in r])
            drag = (gz.mean() - nr.mean()) if len(gz) and len(nr) else None
            exp_pf = _pf_of(gz - (cr.mean() + (gz.mean() - gr.mean()))) if len(gz) and len(cr) else None
            rep["random"][i] = {"trades_zero_cost": int(len(gz)), "pf_zero_cost": _pf_of(gz),
                                "mean_gross_zero_cost": round(float(gz.mean()), 2) if len(gz) else None,
                                "trades_real_cost": int(len(nr)), "pf_real_cost": _pf_of(nr),
                                "mean_net_real_cost": round(float(nr.mean()), 2) if len(nr) else None,
                                "mean_cost_rs": round(float(cr.mean()), 2) if len(cr) else None,
                                "mean_slippage_rs": round(float(gz.mean() - gr.mean()), 2) if len(gr) else None,
                                "observed_drag_per_trade_rs": round(float(drag), 2) if drag is not None else None,
                                "expected_pf_real_cost": exp_pf}
    with open(os.path.join(AUDIT, "benchmarks.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str)[:6000])
    log("benchmarks: written audit/benchmarks.json")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# 4. gross vs net, mirror rules
# ══════════════════════════════════════════════════════════════════════════
def _tf_rows(stage1_dir: str) -> dict:
    out = {}
    if not os.path.isdir(stage1_dir):
        return out
    for f in sorted(os.listdir(stage1_dir)):
        if f.endswith(".json.gz") and (f.startswith("results_") or f.startswith("natives_")):
            with gzip.open(os.path.join(stage1_dir, f), "rt") as fh:
                out.update(json.load(fh))
    return out


def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return {str(q): round(float(np.quantile(x, q)), 3) for q in qs} if len(x) else {}


def cmd_grossnet(a):
    cat = os.path.join(AUDIT, "catalog")
    net = json.load(open(os.path.join(cat, "net_orig_disc.json")))
    gro = json.load(open(os.path.join(cat, "gross_orig_disc.json")))
    mir = json.load(open(os.path.join(cat, "mirror_orig_disc.json")))
    rows = []
    for i, r in net.items():
        g, m = gro.get(i), mir.get(i)
        n_tr = r["disc"]["trades"]
        rows.append({"id": i, "kind": r["kind"], "horizon": r["horizon"], "index": bool(r["underlying"]),
                     "trades": n_tr, "net": r["disc"]["net"], "net_pf": r["disc"]["pf"], "passed": r["passed"],
                     "gross": g["disc"]["net"] if g else None, "gross_pf": g["disc"]["pf"] if g else None,
                     "gross_trades": g["disc"]["trades"] if g else None,
                     "gross_screen_pass": g["passed"] if g else None,
                     "mirror_gross": m["disc"]["net"] if m else None, "mirror_pf": m["disc"]["pf"] if m else None,
                     "mirror_trades": m["disc"]["trades"] if m else None})
    traded = [r for r in rows if r["trades"] > 0 and r["gross"] is not None]

    def share(f, pool):
        return {"n": sum(1 for r in pool if f(r)), "of": len(pool),
                "pct": round(100.0 * sum(1 for r in pool if f(r)) / max(1, len(pool)), 1)}
    rep = {"catalogue": {
        "strategies": len(rows), "with_trades": len(traded),
        "lose_before_costs (gross <= 0)": share(lambda r: r["gross"] <= 0, traded),
        "gross_pf_below_1.15": share(lambda r: (r["gross_pf"] or 0) < 1.15, traded),
        "killed_only_by_costs (gross > 0, net <= 0)": share(lambda r: r["gross"] > 0 and r["net"] <= 0, traded),
        "gross screen pass but net screen fail": share(lambda r: bool(r["gross_screen_pass"]) and not r["passed"],
                                                       traded),
        "net > 0": share(lambda r: r["net"] > 0, traded),
        "net screen pass": share(lambda r: r["passed"], traded),
        "gross_pf_quantiles": _q([r["gross_pf"] for r in traded]),
        "net_pf_quantiles": _q([r["net_pf"] for r in traded])}}
    for hz in ("intraday", "positional"):
        pool = [r for r in traded if r["horizon"] == hz]
        rep["catalogue"][hz] = {"n": len(pool),
                                "lose_before_costs": share(lambda r: r["gross"] <= 0, pool),
                                "killed_only_by_costs": share(lambda r: r["gross"] > 0 and r["net"] <= 0, pool),
                                "gross_pf_quantiles": _q([r["gross_pf"] for r in pool]),
                                "gross_per_trade_rs_quantiles": _q([r["gross"] / r["gross_trades"] for r in pool
                                                                    if r["gross_trades"]])}
    # mirror: a rule and its opposite at zero cost
    mm = [r for r in traded if r["mirror_gross"] is not None and r["mirror_trades"]]
    both_lose = [r for r in mm if r["gross"] < 0 and r["mirror_gross"] < 0]
    per = [(r["gross"] / r["gross_trades"] + r["mirror_gross"] / r["mirror_trades"]) for r in mm if r["gross_trades"]]
    rep["mirror"] = {"pairs": len(mm), "both_lose_before_costs": share(lambda r: r in both_lose, mm),
                     "rule_wins_mirror_loses": share(lambda r: r["gross"] > 0 > r["mirror_gross"], mm),
                     "rule_loses_mirror_wins": share(lambda r: r["gross"] < 0 < r["mirror_gross"], mm),
                     "both_win": share(lambda r: r["gross"] > 0 and r["mirror_gross"] > 0, mm),
                     "sum_of_per_trade_gross_rs_quantiles (rule + mirror, Rs per Rs 1 lakh trade)": _q(per),
                     "intraday_pairs": share(lambda r: r in both_lose, [r for r in mm if r["horizon"] == "intraday"]),
                     "positional_pairs": share(lambda r: r in both_lose, [r for r in mm if r["horizon"] == "positional"])}
    # tfgrid: original net stage 1 vs zero-cost stage 1
    tn = _tf_rows(os.path.join(SCRATCH, "tfgrid", "stage1"))
    tg = _tf_rows(os.path.join(AUDIT, "tfgrid_gross", "stage1"))
    tf = []
    for cid, r in tn.items():
        g = tg.get(cid)
        if not g or "screen" not in r or "screen" not in g:
            continue
        tf.append({"id": cid, "tf": r.get("tf"), "net": r["screen"]["disc"]["net"], "gross": g["screen"]["disc"]["net"],
                   "trades": r["screen"]["disc"]["trades"], "gross_pf": g["screen"]["disc"]["pf"],
                   "net_pass": r["screen"]["passed"], "gross_pass": g["screen"]["passed"]})
    tft = [x for x in tf if x["trades"] > 0]
    rep["tfgrid"] = {"combos": len(tf), "with_trades": len(tft),
                     "lose_before_costs": share(lambda x: x["gross"] <= 0, tft),
                     "killed_only_by_costs": share(lambda x: x["gross"] > 0 and x["net"] <= 0, tft),
                     "gross_screen_pass": share(lambda x: x["gross_pass"], tft),
                     "net_screen_pass": share(lambda x: x["net_pass"], tft),
                     "by_tf": {}}
    for t in ("1m", "3m", "5m", "15m", "30m", "60m", "1d", "1w", "native"):
        pool = [x for x in tft if x["tf"] == t]
        if pool:
            rep["tfgrid"]["by_tf"][t] = {"n": len(pool), "gross_median_rs": round(float(np.median([x["gross"] for x in pool])), 0),
                                         "net_median_rs": round(float(np.median([x["net"] for x in pool])), 0),
                                         "lose_before_costs_pct": share(lambda x: x["gross"] <= 0, pool)["pct"],
                                         "killed_only_by_costs_pct": share(lambda x: x["gross"] > 0 and x["net"] <= 0, pool)["pct"]}
    with open(os.path.join(AUDIT, "gross_net.json"), "w") as fh:
        json.dump({"report": rep, "rows": rows, "tfgrid_rows": tf}, fh, indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
    log("grossnet: written audit/gross_net.json")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# catalogue runs: every native id, aggregated per (strategy, date) inside the worker (low memory)
# ══════════════════════════════════════════════════════════════════════════
def mirror_spec(sp):
    """The opposite-side twin of a strategy: same decision bars, side flipped, stop / target reflected around the
    decision close (fill-relative brackets and % stops are symmetric already). Swing becomes a positional future
    so that shorts are allowed (run at zero cost, so the cost class does not matter)."""
    import dataclasses
    from .api import Entry, Swing
    fn0 = sp.fn

    def flip(px, ref):
        return None if px is None else 2.0 * ref - float(px)

    if sp.horizon == "intraday":
        def fn(s, p):
            r = fn0(s, p)
            if r is None:
                return None
            many = isinstance(r, (list, tuple))
            out = []
            for e in (r if many else [r]):
                i = int(e.i) if many else len(s.t) - 1
                ref = float(s.c[min(i, len(s.t) - 1)])
                out.append(Entry(e.i, -e.side, flip(e.stop, ref), flip(e.target, ref), e.rr, e.time_stop, e.trail,
                                 exit_sig=e.exit_sig, bracket=e.bracket))
            return out if many else out[0]
        kind = sp.kind
    else:
        def fn(d, p):
            r = fn0(d, p)
            if r is None:
                return None
            ref = float(d.c[-1])
            ent = r.entry
            if ent is not None:
                ent = (ent[0], flip(ent[1], ref)) + ((flip(ent[2], ref),) if len(ent) > 2 and ent[2] is not None
                                                     else tuple(ent[2:]))
            return dataclasses.replace(r, side=-r.side, stop=flip(r.stop, ref), target=flip(r.target, ref),
                                       entry=ent)
        kind = "futures" if sp.kind == "swing" else sp.kind
    return dataclasses.replace(sp, fn=fn, kind=kind, id=sp.id)


_CW: dict = {}


def _cat_work(args):
    sym, ids, lo, hi, mode, data, cache = args
    from . import catalog  # noqa: F401
    from . import sim
    from .api import REGISTRY
    if data == "corrected":
        from . import corrected
        corrected.install()
    if mode in ("gross", "mirror"):
        sim.round_trip_cost = lambda *a, **k: 0.0
    specs = [REGISTRY[i] for i in ids]
    if mode == "mirror":
        specs = [mirror_spec(sp) for sp in specs]
    slip = 0.0 if mode in ("gross", "mirror") else None
    t0 = time.time()
    res = sim.run_symbol(cache, sym, specs, lo, hi, audit=False, slippage=slip, index_slippage=slip)
    out = {}
    for sid, r in res.items():
        tr = r["trades"]
        if not tr and not r["error"]:
            continue
        if tr:
            dts = np.array([t[2] for t in tr], np.int64)
            net = np.array([t[12] for t in tr], float)
            gro = np.array([t[10] for t in tr], float)
            cst = np.array([t[11] for t in tr], float)
            u, inv = np.unique(dts, return_inverse=True)
            agg = np.zeros((len(u), 6))
            np.add.at(agg[:, 0], inv, 1)
            np.add.at(agg[:, 1], inv, net)
            np.add.at(agg[:, 2], inv, np.where(net > 0, net, 0))
            np.add.at(agg[:, 3], inv, np.where(net < 0, net, 0))
            np.add.at(agg[:, 4], inv, gro)
            np.add.at(agg[:, 5], inv, cst)
            out[sid] = (u, agg, r["error"], r["skipped"])
        else:
            out[sid] = (np.zeros(0, np.int64), np.zeros((0, 6)), r["error"], r["skipped"])
    return sym, out, time.time() - t0


def screen_from(dates, A, sym_disc_net: dict, horizon: str, single: bool) -> dict:
    """protocol.screen re-expressed on per-date aggregates (same gates, same constants)."""
    from . import protocol as Pr

    def part(rng):
        m = (dates >= rng[0]) & (dates <= rng[1])
        x = A[m].sum(axis=0) if m.any() else np.zeros(6)
        gl = -x[3]
        pf = (x[2] / gl) if gl > 0 else (float("inf") if x[2] > 0 else 0.0)
        return {"trades": int(x[0]), "net": round(float(x[1]), 2), "pf": round(float(pf), 3) if pf != float("inf") else pf,
                "gross": round(float(x[4]), 2), "cost": round(float(x[5]), 2), "gp": float(x[2]), "gl": float(gl)}
    h1, h2, dsc = part(H1), part(H2), part(DISC)
    top_sym, top = max(sym_disc_net.items(), key=lambda kv: kv[1]) if sym_disc_net else ("", 0.0)
    share = top / dsc["net"] if dsc["net"] > 0 else float("nan")
    need = Pr.SCREEN_MIN_TRADES[horizon]
    why = []
    if h1["net"] <= 0:
        why.append("H1 net <= 0")
    if h2["net"] <= 0:
        why.append("H2 net <= 0")
    if dsc["pf"] < Pr.SCREEN_MIN_PF:
        why.append(f"PF {dsc['pf']:.3f} < {Pr.SCREEN_MIN_PF}")
    if dsc["trades"] < need:
        why.append(f"{dsc['trades']} trades < {need}")
    if dsc["net"] > 0 and share > Pr.SCREEN_MAX_SYMBOL_SHARE and not single:
        why.append(f"{top_sym} is {share:.0%} of net (> {Pr.SCREEN_MAX_SYMBOL_SHARE:.0%})")
    return {"passed": not why, "reasons": why, "h1": h1, "h2": h2, "disc": dsc, "top_symbol": top_sym,
            "top_share": None if share != share else round(share, 4), "final": part(FINAL)}


def run_catalog(ids, mode: str, data: str, lo: int, hi: int, jobs: int = 4, cache: str = CACHE3Y, tag: str = ""):
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    syms = stock_symbols(cache) + ["_NIFTY50_min", "_NIFTYBANK_min"]
    acc = {}
    persym = defaultdict(dict)
    errs, skipped = {}, defaultdict(int)
    t0 = time.time()
    args = [(s, list(ids), lo, hi, mode, data, cache) for s in syms]
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=1) as pool:
        for n, (sym, out, secs) in enumerate(pool.imap_unordered(_cat_work, args), 1):
            for sid, (u, agg, err, sk) in out.items():
                d = acc.setdefault(sid, {})
                for dt, row in zip(u.tolist(), agg):
                    if dt in d:
                        d[dt] = d[dt] + row
                    else:
                        d[dt] = row.copy()
                m = (u >= DISC[0]) & (u <= DISC[1])
                persym[sid][sym] = float(agg[m, 1].sum())
                skipped[sid] += sk
                if err and sid not in errs:
                    errs[sid] = err
            if n % 20 == 0:
                log(f"  catalog {tag} {mode}/{data}: {n}/{len(syms)} symbols ({time.time() - t0:.0f}s)")
    res = {}
    daily = {}
    for sid in ids:
        sp = REGISTRY[sid]
        d = acc.get(sid, {})
        dates = np.array(sorted(d), np.int64)
        A = np.array([d[x] for x in dates]) if len(dates) else np.zeros((0, 6))
        sc = screen_from(dates, A, persym.get(sid, {}), sp.horizon, bool(sp.underlying))
        res[sid] = {"kind": sp.kind, "horizon": sp.horizon, "underlying": sp.underlying, "family": sp.family,
                    "error": errs.get(sid), "skipped": skipped[sid], **sc}
        daily[sid] = (dates, A[:, 1] if len(A) else np.zeros(0))
    log(f"catalog {tag} {mode}/{data} {lo}-{hi}: {len(ids)} ids in {time.time() - t0:.0f}s")
    return res, daily


def cmd_catalog(a):
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    ids = sorted(REGISTRY) if not a.ids else [x for x in a.ids.split(",") if x]
    lo, hi = (DISC if a.period == "disc" else (DISC[0], FINAL[1]))
    res, daily = run_catalog(ids, a.mode, a.data, lo, hi, a.jobs, tag=a.tag)
    out = os.path.join(AUDIT, "catalog")
    os.makedirs(out, exist_ok=True)
    name = f"{a.mode}_{a.data}_{a.period}{('_' + a.tag) if a.tag else ''}"
    with open(os.path.join(out, name + ".json"), "w") as fh:
        json.dump(res, fh, indent=0, default=str)
    with gzip.open(os.path.join(out, name + "_daily.pkl.gz"), "wb") as fh:
        pickle.dump(daily, fh)
    n_pass = sum(1 for r in res.values() if r["passed"])
    log(f"catalog {name}: {n_pass}/{len(res)} pass the DISCOVERY screen")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--symbols", type=int, default=0)
    ap.add_argument("--mode", default="net")
    ap.add_argument("--data", default="orig")
    ap.add_argument("--period", default="disc")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ids", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    return {"compare": cmd_compare, "catalog": cmd_catalog, "handcheck": cmd_handcheck, "lookahead": cmd_lookahead, "bias": cmd_bias, "benchmarks": cmd_benchmarks, "grossnet": cmd_grossnet, "shiftcontrol": cmd_shiftcontrol}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
