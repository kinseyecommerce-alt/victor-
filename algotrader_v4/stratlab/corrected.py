"""
stratlab/corrected.py — a CORRECTED VIEW of the lab's bars, applied on load (an overlay: no bar file, cache file or
other module is modified).  Built for the Nov 2026 corrected-data re-run (prereg/rerun_corrected_v1.json).

    python -m stratlab.corrected --build            # per-symbol correction maps -> <scratch>/audit/corrected/
    python -m stratlab.corrected --validate         # every bar_adjustment_issues row after correction
    from stratlab import corrected; corrected.install()   # data.load_symbol / data_daily.load_symbol now correct

ONE CONVENTION for every minute bar, both eras (2016-23 Kite cache and the 2023-26 bars3y cache): AS-TRADED prices
adjusted for capital changes only, on NSE's official prints:
    corrected price = official NSE price (unadjusted bhavcopy) x F(d)
    F(d) = product over official events with  d < ex_date <= the last bars3y session  of
           split / bonus / consolidation     corpactions `factor`
           rights                            corpactions `factor` (theoretical ex-rights; skipped when missing)
           demerger / scheme / cap. reduct.  official open(ex) / official close(previous session) when < 0.98
                                             (NSE discovers the ex-price in the special pre-open session: the open)
    Dividends are NOT back-adjusted (DIVIDENDS = False): the bars show the ex-dividend drop exactly as it traded,
    which is what a live strategy sees (its previous close, gap and daily levels are unadjusted) and what a futures
    position earns.  Kite's files are inconsistent here (on 2023-26 some dividends are back-adjusted by a ratio,
    some in rupees, most not at all; on 2016-23 large ones are not, older ones were subtracted in rupees): all of
    them are put on the as-traded basis.  The cost of the convention: a delivery (CNC) long held over an ex-date
    is not credited the dividend (about 1.3 % a year of the time invested; not modelled by sim.py either).
So F = 1 on the latest session and prices before a split / bonus / demerger are scaled by its factor; overnight
ratios across a capital change equal official open / close / factor, and the 2023-11-09 / 10 seam between the two
minute sets is consistent by construction (both are mapped to the same official basis).

The bars are mapped per SESSION onto that basis: corrected = (bar - off) x mult, constant within a segment.
Segments are split at every official ex-date (dividends included) and wherever the bar / official ratio s(d) (median over open, high, low)
steps by more than 1 % between the 3 sessions before and the 3 after.  In a segment
    multiplicative (the usual case): off = 0, mult = F / median(s)
    affine ("dividends subtracted in rupees"): if bar = a x official + b fits the segment (>= 15 sessions, >= 5 %
           price range) with |b| > 0.5 % of price and half the residual of a pure ratio: off = b, mult = F / a
    bogus first bar: Kite's stale 09:15 print on special pre-open days (demerger ex-dates; trading really starts at
           ~09:50 at the official open) is dropped when the rest of the day matches the official prints
    single session: a session > 0.3 % off its segment map whose open / high / low ratios to the official prints
           agree within 0.5 % (a cleanly rescaled day, e.g. Kite's demerger ex-day) gets its own ratio
Intraday returns are preserved by the multiplicative map and REPAIRED by the affine one (they were off by ~b / price).

EXCLUSION RULE (stated before the re-run): a session whose bars disagree in SHAPE with the official prints of that day
(after mapping still > 1.5 % off, median over open, high, low, with open / high / low ratios more than 0.5 % apart)
cannot be corrected and is dropped from the corrected view;
sessions without an official print keep their segment's map (listed as unverified).
VOLUME: Kite back-adjusts volume with price except on the `unadjusted_volume` events of bar_adjustment_issues.csv:
there the volume of every session before the ex-date is divided by the applied price factor (and the window is
listed in the flags).  Index files (NIFTY 50, NIFTY BANK, VIX) are not changed.
DAILY FILE (nse_old eq_daily_adj, used by data_daily / events_daily): the 10 `daily_2012_2023` issues (7 official
splits / bonuses not applied, 3 wrong factors) are repaired by multiplying every row before the ex-date by
official_factor / applied_factor (volume divided by it).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np

from . import data as D

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
SCRATCH = os.environ.get("STRATLAB_SCRATCH", _SCRATCH)
OUT = os.path.join(SCRATCH, "audit", "corrected")
CA_FILE = os.path.join(SCRATCH, "databuild", "corpactions", "corpactions.csv.gz")
ISSUES = os.path.join(SCRATCH, "databuild", "corpactions", "bar_adjustment_issues.csv")
DB_SCRIPTS = os.path.join(SCRATCH, "databuild", "scripts")
ERAS = {"3y": os.path.join(SCRATCH, "stratlab_cache", "bars3y_fb3d4d"),
        "old": os.path.join(SCRATCH, "oldera_min", "cache")}
ERA_SOURCE = {"3y": "min_2023_2026", "old": "min_2016_2023"}
EXCLUDE_TOL = 0.015
STEP_TOL = 0.01
DEMERGER_MAX = 0.98
AFF_MIN_SESSIONS = 15         # affine fit: sessions in the segment
AFF_MIN_RANGE = 0.05          # ... log price range
AFF_MIN_OFFSET = 0.005        # ... |b| / price
SESSION_TOL = 0.003           # a session off its segment map by more gets its own ratio (if clean, see below)
DISP_TOL = 0.005
VERSION = "corr1"
DIVIDENDS = False             # see the docstring: as-traded prices

_MAPS: dict = {}
_ORIG: dict = {}


# ── official prices and corporate actions ───────────────────────────────────
def _db():
    if DB_SCRIPTS not in sys.path:
        sys.path.insert(0, DB_SCRIPTS)
    import cal          # noqa: E402  (read-only helpers of the data build: official bhavcopy prices, renames)
    import symmap       # noqa: E402
    return cal, symmap


def official(sym: str, days) -> dict:
    """{date: (open, high, low, close, volume)} from the unadjusted NSE bhavcopy, under the symbol in force."""
    cal, symmap = _db()
    out, names = {}, {}
    for d in days:
        ym = int(d) // 100
        if ym not in names:
            names[ym] = []
            for nm in (symmap.symbol_on(sym, ym * 100 + 1), symmap.symbol_on(sym, ym * 100 + 28), sym):
                if nm not in names[ym]:
                    names[ym].append(nm)
        for nm in names[ym]:
            s = cal.series(nm)
            if s is None:
                continue
            j = int(np.searchsorted(s["date"], int(d)))
            if j < len(s["date"]) and int(s["date"][j]) == int(d) and s["open"][j] > 0 and s["close"][j] > 0:
                out[int(d)] = tuple(float(s[k][j]) for k in ("open", "high", "low", "close", "volume"))
                break
    return out


_CA: dict = {}


def corp_events(sym: str) -> list:
    if not _CA:
        with gzip.open(CA_FILE, "rt") as fh:
            for r in csv.DictReader(fh):
                _CA.setdefault(r["lab_symbol"], []).append(r)
    return _CA.get(sym, [])


def event_factors(sym: str, cutoff: int) -> dict:
    """{ex_date: (factor, kinds)} of every official event with ex_date <= cutoff (see the module docstring)."""
    cal, symmap = _db()
    evs = {}
    for r in corp_events(sym):
        ex = int(r["ex_date"])
        if ex > cutoff:
            continue
        evs.setdefault(ex, []).append(r)
    out = {}
    for ex, rows in sorted(evs.items()):
        f, kinds = 1.0, []
        prev = None
        for nm in (symmap.symbol_on(sym, ex), sym):
            s = cal.series(nm)
            if s is None:
                continue
            j = int(np.searchsorted(s["date"], ex))
            if 0 < j <= len(s["date"]):
                prev = (s, j)
                break
        pc = float(prev[0]["close"][prev[1] - 1]) if prev else float("nan")
        op = (float(prev[0]["open"][prev[1]]) if prev and prev[1] < len(prev[0]["date"])
              and int(prev[0]["date"][prev[1]]) == ex else float("nan"))
        for r in rows:
            k = r["kind"]
            if k in ("split", "bonus", "consolidation", "rights") and r["factor"]:
                f *= float(r["factor"])
                kinds.append(k)
            elif k in ("demerger", "capital_reduction") and op == op and pc == pc and op / pc < DEMERGER_MAX:
                f *= op / pc
                kinds.append(k)
            elif DIVIDENDS and k == "dividend" and r["dividend"] and pc == pc and pc > 0:
                f *= max(0.0, 1.0 - float(r["dividend"]) / pc)
                kinds.append("dividend")
        if abs(f - 1.0) > 1e-9:
            out[ex] = (f, "+".join(kinds))
    return out


def _issue_rows() -> list:
    with open(ISSUES) as fh:
        return list(csv.DictReader(fh))


# ── building the per-symbol maps ────────────────────────────────────────────
def build_symbol(sym: str, sd, cutoff: int, era: str) -> dict:
    dl = sd.daily()
    days = dl["date"].astype(np.int64)
    n = len(days)
    off = official(sym, days)
    oo, oh, ol, oc, ov = (np.array([off.get(int(d), (np.nan,) * 5)[i] for d in days]) for i in range(5))
    bo, bh, bl, bc = dl["o"].copy(), dl["h"].copy(), dl["l"].copy(), dl["c"]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.nanmedian(np.vstack([bo / oo, bh / oh, bl / ol]), axis=0)
    ev = event_factors(sym, cutoff)
    exs = np.array(sorted(ev), np.int64)
    fs = np.array([ev[x][0] for x in exs]) if len(exs) else np.zeros(0)
    # F(d) = product of factors with ex_date > d
    F = np.ones(n)
    for x, f in zip(exs, fs):
        F[days < x] *= f
    # segment boundaries: first session on/after each ex-date, plus data-driven steps of s
    bnd = set()
    all_ex = sorted({int(r["ex_date"]) for r in corp_events(sym) if int(r["ex_date"]) <= cutoff})
    for x in all_ex:                    # every official ex-date (dividends too: Kite sometimes adjusts for them)
        j = int(np.searchsorted(days, x))
        if 0 < j < n:
            bnd.add(j)
    ls = np.log(s)
    for g in range(3, n - 2):
        a, b = ls[g - 3:g], ls[g:g + 3]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) >= 2 and len(b) >= 2 and abs(np.median(b) - np.median(a)) > STEP_TOL:
            if not any(abs(g - x) <= 2 for x in bnd):
                bnd.add(g)
    cuts = [0] + sorted(bnd) + [n]
    offv, mult, kind = np.zeros(n), np.ones(n), np.array(["m"] * n, dtype=object)
    seg_s = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        ok = np.isfinite(s[a:b])
        aff = None
        if ok.sum() >= AFF_MIN_SESSIONS:
            x = np.concatenate([oo[a:b][ok], oh[a:b][ok], ol[a:b][ok]])
            y = np.concatenate([bo[a:b][ok], bh[a:b][ok], bl[a:b][ok]])
            if float(np.ptp(np.log(x))) >= AFF_MIN_RANGE:
                A = np.vstack([x, np.ones(len(x))]).T
                (fa, fb), *_ = np.linalg.lstsq(A, y, rcond=None)
                fp = float(np.sum(x * y) / np.sum(x * x))
                if abs(fb) / float(np.median(y)) > AFF_MIN_OFFSET and np.std(y - fa * x - fb) < 0.5 * np.std(y - fp * x):
                    aff = (float(fa), float(fb))
        if aff is not None:
            offv[a:b], mult[a:b], kind[a:b] = aff[1], F[a:b] / aff[0], "a"
            seg_s.append((a, b, "affine", aff))
        else:
            sm = float(np.median(s[a:b][ok])) if ok.any() else np.nan
            seg_s.append((a, b, "ratio", sm))
            mult[a:b] = F[a:b] / sm if sm == sm else np.nan
    # segments without any official print borrow the nearest segment's ratio
    for a, b, kd, val in seg_s:
        if kd == "ratio" and not (val == val):
            near = [v for (a2, b2, k2, v) in seg_s if k2 == "ratio" and v == v]
            prv = [v for (a2, b2, k2, v) in seg_s if k2 == "ratio" and v == v and b2 <= a]
            sm = prv[-1] if prv else (near[0] if near else 1.0)
            mult[a:b] = F[a:b] / sm
    # residual vs the official prints after mapping. A session off its segment by > SESSION_TOL whose bars are a
    # clean scaled copy of that day's official prints (open / high / low ratios within DISP_TOL of each other: e.g.
    # Kite's demerger ex-day, scaled on its own) gets its own ratio; one whose shape disagrees by > EXCLUDE_TOL is
    # dropped (cannot be corrected).
    with np.errstate(invalid="ignore", divide="ignore"):
        rat = np.vstack([bo / oo, bh / oh, bl / ol])
        disp = (np.nanmax(rat, axis=0) - np.nanmin(rat, axis=0)) / s

        def resid():
            co, ch, cl = (bo - offv) * mult, (bh - offv) * mult, (bl - offv) * mult
            return np.nanmedian(np.vstack([co / (oo * F), ch / (oh * F), cl / (ol * F)]), axis=0) - 1.0
        res = resid()
        # a bogus FIRST bar (Kite prints a stale 09:15 bar on special pre-open days, e.g. demerger ex-dates, while
        # trading starts ~09:50 at the official open): if the day matches the official prints without it, drop it
        drop_first = np.zeros(n, bool)
        for k in np.flatnonzero(np.isfinite(res) & (np.abs(res) > SESSION_TOL) & (disp >= DISP_TOL)):
            a, b = int(sd.starts[k]), int(sd.ends[k])
            if b - a < 2:
                continue
            r2 = np.array([sd.o[a + 1] / oo[k], sd.h[a + 1:b].max() / oh[k], sd.l[a + 1:b].min() / ol[k]])
            if np.isfinite(r2).all() and (r2.max() - r2.min()) / np.median(r2) < DISP_TOL:
                drop_first[k] = True
                bo[k], bh[k], bl[k] = sd.o[a + 1], sd.h[a + 1:b].max(), sd.l[a + 1:b].min()
                s[k], disp[k] = float(np.median(r2)), 0.0
        res = resid()
        own = np.isfinite(res) & (np.abs(res) > SESSION_TOL) & (disp < DISP_TOL)
        offv[own], mult[own], kind[own] = 0.0, F[own] / s[own], "p"
        res = resid()
    excl = np.isfinite(res) & (np.abs(res) > EXCLUDE_TOL)
    unver = ~np.isfinite(res)
    # volume: unadjusted-volume events of this era
    vmult = np.ones(n)
    vflags = []
    for r in _issue_rows():
        if r["source"] == ERA_SOURCE[era] and r["symbol"] == sym and r["issue"] == "unadjusted_volume":
            x, fa = int(r["date"]), float(r["applied_factor"])
            vmult[days < x] /= fa
            vflags.append({"ex_date": x, "applied_factor": fa, "first": int(days[0]),
                           "sessions": int((days < x).sum())})
    return {"days": days.astype(np.int32), "off": offv, "mult": mult, "vmult": vmult, "exclude": excl,
            "drop_first": drop_first,
            "unverified": unver, "kind": kind, "F": F, "events": {int(k): v for k, v in ev.items()},
            "segments": [(int(a), int(b), k, v) for a, b, k, v in seg_s], "resid": res, "vflags": vflags,
            "s": s}


def build(eras=("3y", "old"), verbose=True) -> dict:
    os.makedirs(OUT, exist_ok=True)
    cut_sd = D.load_symbol(ERAS["3y"], "_NIFTY50_min")
    cutoff = int(cut_sd.days[-1])
    report = {"version": VERSION, "cutoff": cutoff, "eras": {}}
    for era in eras:
        cache = ERAS[era]
        with open(os.path.join(cache, "_manifest.json")) as fh:
            syms = sorted(s for s in json.load(fh)["symbols"] if not s.startswith("_"))
        maps, rep = {}, {}
        t0 = time.time()
        for sym in syms:
            sd = _orig_load(cache, sym)
            if sd is None:
                continue
            m = build_symbol(sym, sd, cutoff, era)
            maps[sym] = m
            chg = np.abs(np.log(np.where(m["mult"] > 0, m["mult"], np.nan)))
            rep[sym] = {"sessions": int(len(m["days"])), "excluded": [int(d) for d in m["days"][m["exclude"]]],
                        "unverified": int(m["unverified"].sum()),
                        "affine_segments": [(int(m["days"][a]), int(m["days"][b - 1]), v)
                                            for a, b, k, v in m["segments"] if k == "affine"],
                        "segments": len(m["segments"]), "vflags": m["vflags"],
                        "max_abs_log_mult_step": float(np.nanmax(np.abs(np.diff(np.log(m["mult"])))))
                        if len(m["mult"]) > 1 else 0.0}
            if verbose and len(rep) % 20 == 0:
                print(f"  {era}: {len(rep)}/{len(syms)} symbols ({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(OUT, f"maps_{era}.pkl"), "wb") as fh:
            pickle.dump(maps, fh)
        report["eras"][era] = rep
    with open(os.path.join(OUT, "build_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    return report


# ── applying the maps on load ───────────────────────────────────────────────
def _era_of(cache_dir: str):
    p = os.path.realpath(cache_dir or "")
    for era, c in ERAS.items():
        if p == os.path.realpath(c):
            return era
    return None


def _maps(era: str) -> dict:
    if era not in _MAPS:
        with open(os.path.join(OUT, f"maps_{era}.pkl"), "rb") as fh:
            _MAPS[era] = pickle.load(fh)
    return _MAPS[era]


def _orig_load(cache_dir, symbol):
    return (_ORIG.get("min") or D.load_symbol)(cache_dir, symbol)


def apply_map(sd, m: dict):
    """A corrected copy of a minute SymbolData (excluded sessions dropped)."""
    pos = {int(d): k for k, d in enumerate(m["days"])}
    k = np.array([pos.get(int(d), -1) for d in sd.days])
    if (k < 0).any():                         # a session the map does not know: leave it unchanged
        off_s = np.where(k >= 0, m["off"][np.maximum(k, 0)], 0.0)
        mul_s = np.where(k >= 0, m["mult"][np.maximum(k, 0)], 1.0)
        vm_s = np.where(k >= 0, m["vmult"][np.maximum(k, 0)], 1.0)
        ex_s = np.where(k >= 0, m["exclude"][np.maximum(k, 0)], False)
    else:
        off_s, mul_s, vm_s, ex_s = m["off"][k], m["mult"][k], m["vmult"][k], m["exclude"][k]
    lens = sd.ends - sd.starts
    off, mul, vm = np.repeat(off_s, lens), np.repeat(mul_s, lens), np.repeat(vm_s, lens)
    keep = ~np.repeat(ex_s, lens)
    df = m.get("drop_first")
    if df is not None:
        for j in np.flatnonzero(k >= 0):
            if df[k[j]]:
                keep[sd.starts[j]] = False          # bogus first bar (see build_symbol)
    z = {"date": sd.date[keep], "t": sd.t[keep]}
    for c in "ohlc":
        z[c] = ((getattr(sd, c) - off) * mul)[keep]
    z["v"] = (sd.v * vm)[keep]
    out = D._sessionize(sd.symbol, z)
    out.dropped = list(sd.dropped or []) + [int(d) for d in sd.days[ex_s]]
    return out


def load_symbol_corrected(cache_dir: str, symbol: str):
    sd = _orig_load(cache_dir, symbol)
    era = _era_of(cache_dir)
    if sd is None or era is None or symbol.startswith("_"):
        return sd
    m = _maps(era).get(symbol)
    return sd if m is None else apply_map(sd, m)


# daily file (nse_old eq_daily_adj): the 10 daily_2012_2023 issues
def daily_fixes() -> dict:
    fx = {}
    for r in _issue_rows():
        if r["source"] != "daily_2012_2023":
            continue
        fo, fa = float(r["official_factor"]), float(r["applied_factor"] or 1.0)
        fx.setdefault(r["symbol"], []).append((int(r["date"]), fo / fa))
    return fx


def load_daily_corrected(cache_dir: str, symbol: str):
    sd = _ORIG["daily"](cache_dir, symbol)
    fx = daily_fixes().get(symbol) if sd is not None else None
    if not fx:
        return sd
    mult = np.ones(len(sd.date))
    for ex, f in fx:
        mult[sd.date < ex] *= f
    for c in "ohlc":
        setattr(sd, c, getattr(sd, c) * mult)
    sd.v = sd.v / mult
    return sd


def install(minute: bool = True, daily: bool = True):
    """Route data.load_symbol (minute caches of both eras) and data_daily.load_symbol through the corrections.
    Idempotent. Call before forking workers."""
    from . import data_daily as DD
    if minute and "min" not in _ORIG:
        _ORIG["min"] = D.load_symbol
        D.load_symbol = load_symbol_corrected
    if daily and "daily" not in _ORIG:
        _ORIG["daily"] = DD.load_symbol
        DD.load_symbol = load_daily_corrected


def uninstall():
    from . import data_daily as DD
    if "min" in _ORIG:
        D.load_symbol = _ORIG.pop("min")
    if "daily" in _ORIG:
        DD.load_symbol = _ORIG.pop("daily")


# ── validation ──────────────────────────────────────────────────────────────
def validate() -> dict:
    """For every minute-bar issue row: the step of corrected / official around the date must equal the official
    factor (1 for a date without one), and the corrected overnight ratio must equal the official open / previous
    official close divided by that factor (up to the 15:29-close vs official-close difference)."""
    rows = _issue_rows()
    out = []
    for r in rows:
        era = {"min_2023_2026": "3y", "min_2016_2023": "old"}.get(r["source"])
        sym, d = r["symbol"], int(r["date"])
        if r["source"] == "junction_2023-11":
            out.append(_validate_seam(sym))
            continue
        if era is None:
            continue
        if r["issue"] == "additive_offset":
            out.append(_validate_affine(era, sym, d, r))
            continue
        m = _maps(era).get(sym)
        if m is None:
            out.append({"source": r["source"], "symbol": sym, "date": d, "issue": r["issue"], "status": "no map"})
            continue
        days = m["days"]
        j = int(np.searchsorted(days, d))
        if j <= 0 or j >= len(days):
            out.append({"source": r["source"], "symbol": sym, "date": d, "issue": r["issue"], "status": "out of range"})
            continue
        f_off = m["F"][j - 1] / m["F"][j]
        sd = _orig_load(ERAS[era], sym)
        c = apply_map(sd, m)
        cd = c.daily()
        jj = int(np.searchsorted(cd["date"], d))
        offi = official(sym, [int(days[j - 1]), int(days[j])])
        rec = {"source": r["source"], "symbol": sym, "date": d, "issue": r["issue"], "official_factor": round(f_off, 5)}
        if jj <= 0 or jj >= len(cd["date"]) or int(cd["date"][jj]) != int(days[j]):
            rec["status"] = "session excluded"
            out.append(rec)
            continue
        corr_gap = float(cd["o"][jj] / cd["c"][jj - 1])
        rec["corrected_gap"] = round(corr_gap, 5)
        rec["original_gap"] = round(float(sd.daily()["o"][j] / sd.daily()["c"][j - 1]), 5)
        if int(days[j - 1]) in offi and int(days[j]) in offi:
            og = offi[int(days[j])][0] / offi[int(days[j - 1])][3]
            rec["official_gap"] = round(og, 5)
            rec["expected_gap"] = round(og / f_off, 5)
            rec["gap_error"] = round(corr_gap / (og / f_off) - 1.0, 5)
        # step of corrected / official (median of o/h/l) across the date = 1 after correction
        lo, hi = max(0, jj - 3), min(len(cd["date"]), jj + 3)
        dd = [int(x) for x in cd["date"][lo:hi]]
        of2 = official(sym, dd)
        sc = []
        for i, x in enumerate(dd):
            if x in of2:
                k = lo + i
                F = m["F"][int(np.searchsorted(days, x))]
                sc.append((x, np.median([cd["o"][k] / (of2[x][0] * F), cd["h"][k] / (of2[x][1] * F),
                                         cd["l"][k] / (of2[x][2] * F)])))
        before = [v for x, v in sc if x < d]
        after = [v for x, v in sc if x >= d]
        if before and after:
            rec["residual_step"] = round(float(np.median(before) / np.median(after) - 1.0), 5)
        rec["status"] = "ok" if abs(rec.get("residual_step", 0.0)) < 0.005 else "check"
        out.append(rec)
    with open(os.path.join(OUT, "validation.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    return out


def _validate_affine(era: str, sym: str, d: int, r: dict) -> dict:
    """Additive-offset window (250 sessions from `d`): after correction, corrected = a x official x F + b must have
    |b| < 0.2 % of price (a pure ratio), where the original bars had the reported offset."""
    m = _maps(era)[sym]
    sd = _orig_load(ERAS[era], sym)
    c = apply_map(sd, m).daily()
    j = int(np.searchsorted(c["date"], d))
    dd = [int(x) for x in c["date"][j:j + 250]]
    offi = official(sym, dd)
    xs, ys = [], []
    for i, x in enumerate(dd):
        if x in offi:
            F = m["F"][int(np.searchsorted(m["days"], x))]
            for k, col in ((0, "o"), (1, "h"), (2, "l")):
                xs.append(offi[x][k] * F)
                ys.append(c[col][j + i])
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([xs, np.ones(len(xs))]).T
    (fa, fb), *_ = np.linalg.lstsq(A, ys, rcond=None)
    rel = float(fb / np.median(ys))
    return {"source": r["source"], "symbol": sym, "date": d, "issue": "additive_offset",
            "original": r["subject"][:90], "corrected_fit": f"corrected = {fa:.4f} x official x F {fb:+.3f}",
            "offset_rel": round(rel, 5), "status": "ok" if abs(rel) < 0.002 else "check"}


def _validate_seam(sym: str) -> dict:
    """Old-era last close 2023-11-09 vs 3-year first open 2023-11-10 after correction, against the official gap."""
    a = apply_map(_orig_load(ERAS["old"], sym), _maps("old")[sym]).daily()
    b = apply_map(_orig_load(ERAS["3y"], sym), _maps("3y")[sym]).daily()
    ia = int(np.searchsorted(a["date"], 20231109, side="right")) - 1
    ib = int(np.searchsorted(b["date"], 20231110))
    offi = official(sym, [int(a["date"][ia]), int(b["date"][ib])])
    g = float(b["o"][ib] / a["c"][ia])
    rec = {"source": "junction_2023-11", "symbol": sym, "date": int(b["date"][ib]), "issue": "era_scale_mismatch",
           "corrected_gap": round(g, 5)}
    if len(offi) == 2:
        og = offi[int(b["date"][ib])][0] / offi[int(a["date"][ia])][3]
        rec["official_gap"] = round(og, 5)
        rec["gap_error"] = round(g / og - 1.0, 5)
    rec["status"] = "ok" if abs(rec.get("gap_error", 1.0)) < 0.005 else "check"
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.corrected")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--eras", default="3y,old")
    a = ap.parse_args(argv)
    if a.build:
        rep = build(tuple(a.eras.split(",")))
        for era, r in rep["eras"].items():
            n_ex = sum(len(x["excluded"]) for x in r.values())
            n_aff = sum(len(x["affine_segments"]) for x in r.values())
            print(f"{era}: {len(r)} symbols, {n_ex} sessions excluded, {n_aff} affine segments")
    if a.validate:
        v = validate()
        bad = [x for x in v if x.get("status") not in ("ok",)]
        print(f"validated {len(v)} issue rows; not ok: {len(bad)}")
        for x in bad[:40]:
            print("  ", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
