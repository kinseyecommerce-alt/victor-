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
    return {"compare": cmd_compare, "catalog": cmd_catalog, "handcheck": cmd_handcheck}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
