"""
stratlab/protocol_min_oos.py — the OLD-ERA MINUTE-BAR OUT-OF-SAMPLE judgement (2016-01-01 .. 2023-11-09).

The 2012-2023 daily round (protocol_oos.py) could not judge anything that needs 1-minute bars: the three live
PAPER strategies (AUC-FADE-3PCT, SWING-ATTN-ON-3X, VRP-STRANGLE-016-WINGED), the other minute-bar strategies
FINAL-confirmed on 2023-26, and every intraday / scalping / decision-time catalogue spec. Kite 1-minute bars for
2016-01-01 .. 2023-11-09 (today's NIFTY-100 names + NIFTY 50, NIFTY BANK, INDIA VIX) make that possible. None of these
strategies was developed on these years. This module runs each of them ONCE on them, with no tuning.

GATES (frozen; GATES_FINGERPRINT is recorded with every judgement and in the pre-registration file). They are
protocol_oos's gates (its constants are imported, so they cannot drift) on this era's halves:
  halves                H1 = 2016-01-01 .. 2019-12-31, H2 = 2020-01-01 .. 2023-11-09 (a trade belongs to the half of
                        its ENTRY date; the era is one block, so a trade entered in H1 may exit in H2)
  PASS (OOS-CONFIRMED)  net after costs > 0 in H1 AND in H2
                        profit factor >= 1.15 over the whole era
                        trades >= protocol_oos.MIN_TRADES_PER_HALF[horizon] in EACH half (intraday 150, positional 60)
                        top symbol <= 35 % of the era's net (skipped for one-instrument strategies)
  REPORTED, NOT GATES   the same strategy at 5 bps slippage per side (index 2.5 bps; options: haircut x2.5) and
                        whether it would pass there; whether an intraday strategy would pass with 60 trades per half;
                        net minus the same notional's NIFTY move over each trade, per half ("excess vs NIFTY"), the
                        relevant line for long-biased strategies (long share of era trades >= 70 %)
  COMBINED              "proven in both eras" = FINAL-CONFIRMED on 2023-26 AND OOS-CONFIRMED here
Costs: costs.py as it is (2026 rates) for every year. Fills, slippage and costs are each runner's own: sim.py for
catalogue specs (2 bps stocks / 1 bps index), auction.py (2.1 bps, its pre-registered value), auction_r2.py,
attention.py (2 bps), options_vrp / options_vrp_vix / options_r3 (Black-Scholes MODEL premiums on the NIFTY / NIFTY BANK
minute bars with the previous India VIX close as one flat vol: not option prices), r3_crosssec.py, ltr.py.

Data: <scratch>/bars_min_old/<SYMBOL>_<YEAR>.json.gz (Kite, split / bonus adjusted), parsed by data.py's own
_parse_bars / _clean_minute into ONE compressed .npz per symbol in <scratch>/oldera_min/cache (same arrays, same names
as data.prepare writes, read by data.load_symbol unchanged). Index files become _NIFTY50_min / _NIFTYBANK_min; daily
NIFTY 50 and India VIX closes (_NIFTY50_day / _VIX_day) come from <scratch>/nse_old/index_daily.csv.gz (official).
Universe: today's NIFTY-100 names only (SURVIVORSHIP-BIASED in favour of long strategies; names listed later simply
start later). nsedata is pointed at the old-era view (<scratch>/nse_old_view) for the r4_* specs.

ONE LOOK: every judgement goes to <scratch>/oldera_min/ledger.json keyed by id + cost model. A run never recomputes a
judged strategy: it reports the first result ("first judged <when>"), and a changed strategy / changed gates is refused.
A strategy is judged only if it is in the pre-registration (--prereg) with the same fingerprint, the gates are the
registered ones and the pre-registration file has the sha256 recorded in PREREG_SHA256_FILE.

    cd algotrader_v4
    python -m stratlab.protocol_min_oos --build-cache                 # bars_min_old -> oldera_min/cache
    python -m stratlab.protocol_min_oos --data-check                  # coverage, gaps, adjustment, seam (no P&L)
    python -m stratlab.protocol_min_oos --list                        # groups a / b / c and the ineligible
    python -m stratlab.protocol_min_oos --preregister <file>          # freeze ids + fingerprints + gates
    python -m stratlab.protocol_min_oos --run a|b|c --prereg <file> [--jobs 4] [--trials <ledger>]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np

from . import data as D
from . import protocol as P
from . import protocol_oos as PO

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
SCRATCH = os.environ.get("STRATLAB_SCRATCH", _SCRATCH)
SRC_DIR = os.path.join(SCRATCH, "bars_min_old")
NSE_OLD = os.path.join(SCRATCH, "nse_old")
ROOT = os.path.join(SCRATCH, "oldera_min")
CACHE = os.path.join(ROOT, "cache")
LEDGER = os.path.join(ROOT, "ledger.json")
PREREG = os.path.join(ROOT, "PREREGISTERED_MIN_OOS.json")
PREREG_SHA256_FILE = os.path.join(ROOT, "PREREGISTERED_MIN_OOS.sha256")
REPO_PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prereg", "oldera_min_v1.json")

ERA = (20160101, 20231109)
H1 = (20160101, 20191231)
H2 = (20200101, 20231109)
MIN_PF = PO.MIN_PF
MIN_TRADES_PER_HALF = dict(PO.MIN_TRADES_PER_HALF)
MAX_SYMBOL_SHARE = PO.MAX_SYMBOL_SHARE
SENS_BPS = PO.SENS_BPS
REPORT_MIN_TRADES_INTRADAY = 60          # report-only: would an intraday strategy pass with 60 trades per half?
LONG_BIASED_SHARE = 0.70                 # report-only: long share of era trades that makes the excess line the relevant one
GATES = {"era": ERA, "h1": H1, "h2": H2, "min_pf": MIN_PF, "min_trades_per_half": MIN_TRADES_PER_HALF,
         "max_symbol_share": MAX_SYMBOL_SHARE, "net_gt_0": ["h1", "h2"], "sens_bps": SENS_BPS,
         "universe": "today's NIFTY-100 names with Kite minute bars 2016-01-01..2023-11-09 (survivorship-biased); "
                     "no point-in-time gate",
         "costs": "costs.py 2026 rates for every year"}
GATES_FINGERPRINT = hashlib.sha1(json.dumps(GATES, sort_keys=True).encode()).hexdigest()[:12]
ERA_TAG = "2016-23"
PROTOCOL_TAG = "2016-23 minute OOS"
_NET, _SYM, _DATE = 12, 1, 2
INDEX_SRC = {"NIFTY_50": "_NIFTY50_min", "NIFTY_BANK": "_NIFTYBANK_min"}
VIX_SRC = "INDIA_VIX"


# ══════════════════════════════════════════════════════════════════════════
# data: cache build and data check
# ══════════════════════════════════════════════════════════════════════════
def _src_files(src: str = SRC_DIR) -> dict:
    """{symbol stem: [paths by year]} for <SYMBOL>_<YEAR>.json.gz."""
    out = defaultdict(list)
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".json.gz"):
            continue
        stem, year = fn[:-8].rsplit("_", 1)
        if year.isdigit():
            out[stem].append(os.path.join(src, fn))
    return {k: sorted(v) for k, v in out.items()}


def _read_bars(paths) -> tuple[list, dict]:
    bars, per_year = [], {}
    for p in paths:
        with gzip.open(p, "rt") as fh:
            d = json.load(fh)
        b = d.get("bars") or []
        per_year[os.path.basename(p)[:-8].rsplit("_", 1)[1]] = len(b)
        bars.extend(b)
    return bars, per_year


# Data-cleaning rules found by the data check BEFORE any strategy was run (pre-registered):
#   IRCTC   bars before its listing day 2019-10-14 belong to another instrument quoted under the same Kite name
#           (price ~37, a different series), so they are dropped;
#   PIIND   2016-01 .. 2020-01 carries prints of another instrument (196 sessions with bars 20 % or more away from the
#           session median, e.g. a 09:15 open of 138 against closes of 650-700, whole days at ~95 in Jan 2019): bars
#           before 2020-02-01 are dropped.
DROP_BEFORE = {"IRCTC": 20191014, "PIIND": 20200201}
NSE_ALIAS = {"ETERNAL": "ZOMATO", "UNITDSPR": "MCDOWELL-N"}      # renamed after 2023-11-09 (nse_old ends there)
NOCMP_TOL = 0.10                 # no NSE row to compare with: |log(Kite overnight ratio)| above this = dirty link
DIRTY_TOL = 0.05                 # |log(Kite overnight ratio / NSE adjusted overnight ratio)| above this = dirty link


def _build_one(args):
    stem, paths, cache = args
    bars, per_year = _read_bars(paths)
    name = INDEX_SRC.get(stem, stem)
    if stem == VIX_SRC:
        name = "_VIX_min"
    p = D._parse_bars(bars)
    if p is not None and stem in DROP_BEFORE:
        keep = p["date"] >= DROP_BEFORE[stem]
        p = {k: v[keep] for k, v in p.items()}
    out = D._clean_minute(p) if p is not None else None
    if out is None:
        return stem, name, per_year, 0
    np.savez_compressed(os.path.join(cache, D._cache_name(name)), **out)
    return stem, name, per_year, int(len(out["date"]))


def _nse_index_daily(nse_old: str = NSE_OLD) -> dict:
    """{"NIFTY": {...}, "BANKNIFTY": {...}, "INDIAVIX": {...}} official daily OHLC from index_daily.csv.gz."""
    rows = defaultdict(list)
    with gzip.open(os.path.join(nse_old, "index_daily.csv.gz"), "rt") as fh:
        for r in csv.DictReader(fh):
            rows[r["index"]].append(r)
    out = {}
    for k, rs in rows.items():
        rs.sort(key=lambda r: r["date"])
        f = lambda c: np.array([float(r[c]) if r[c] else np.nan for r in rs])  # noqa: E731
        out[k] = {"date": np.array([int(r["date"]) for r in rs], np.int32), "o": f("open"), "h": f("high"),
                  "l": f("low"), "c": f("close"), "v": np.zeros(len(rs))}
    return out


def build_cache(src: str = SRC_DIR, cache: str = CACHE, jobs: int = 4, nse_old: str = NSE_OLD) -> dict:
    """bars_min_old -> one compressed npz per symbol (data.py's parser and cleaner), daily NIFTY / VIX from nse_old."""
    cache = D.assert_outside_repo(cache)
    os.makedirs(cache, exist_ok=True)
    files = _src_files(src)
    man = {"built": time.strftime("%Y-%m-%d %H:%M"), "src": src, "symbols": {}, "aux": {}, "per_year": {},
           "empty": []}
    with mp.get_context("fork").Pool(jobs) as pool:
        for stem, name, per_year, n in pool.imap_unordered(_build_one, [(s, p, cache) for s, p in files.items()]):
            man["per_year"][stem] = per_year
            if n == 0:
                man["empty"].append(stem)
                continue
            (man["aux"] if name.startswith("_") else man["symbols"])[name] = n
    idx = _nse_index_daily(nse_old)
    for key, name in (("NIFTY", "_NIFTY50_day"), ("INDIAVIX", "_VIX_day"), ("BANKNIFTY", "_NIFTYBANK_day")):
        z = idx[key]
        keep = np.isfinite(z["c"])
        np.savez_compressed(os.path.join(cache, D._cache_name(name)), **{k: v[keep] for k, v in z.items()})
        man["aux"][name] = int(keep.sum())
    with open(os.path.join(cache, "_manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    return man


def nse_calendar(nse_old: str = NSE_OLD) -> list:
    """NSE regular sessions of the era (NIFTY rows of index_daily) minus the Muhurat / special sessions the lab drops
    anyway (< 200 one-minute bars): weekday Muhurat days are recognised by their missing / tiny NIFTY range."""
    idx = _nse_index_daily(nse_old)["NIFTY"]
    spec = set()
    with open(os.path.join(nse_old, "special_sessions.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["weekday"] in ("Sat", "Sun") and int(r["date"]) not in (20200201,):   # 2020-02-01 = full Budget day
                spec.add(int(r["date"]))
    out = []
    for d, h, lo, c in zip(idx["date"], idx["h"], idx["l"], idx["c"]):
        d = int(d)
        if ERA[0] <= d <= ERA[1] and d not in spec and d not in MUHURAT_WEEKDAY:
            out.append(d)
    return out


# Muhurat-only weekday sessions (the regular market was closed for Diwali; NSE publishes a one-hour session)
MUHURAT_WEEKDAY = (20171019, 20181107, 20211104, 20221024)


def dirty_links(cache: str = CACHE, nse_old: str = NSE_OLD) -> dict:
    """{symbol: sorted dates d} where the overnight link previous-session -> d in the minute data cannot be trusted:
      * the previous NSE regular session is missing from the symbol's minute data (download holes; the lab's
        "previous session" would silently be two sessions back), or the session itself was dropped (< 200 bars):
        d = the first available session after the hole;
      * the Kite overnight ratio (09:15 open / previous 15:29 close) differs from NSE's adjusted ratio (official
        open / previous close, back-adjusted by nse_old's factors incl. demergers) by more than DIRTY_TOL in log
        terms: an unadjusted demerger (ADANIENT 2018-09-06, GRASIM 2017-07-19), a wrong Kite adjustment (NMDC
        2022-10-27) or a bad opening print. Real market moves agree in both sources;
      * NSE has no EQ row for d or for the previous session (no comparison possible) and the Kite overnight move
        exceeds NOCMP_TOL in log terms (ABB 2019-12-20, demerger ex-day mis-adjusted by Kite).
    A trade whose entry .. exit dates contain a dirty date of its symbol is excluded (DIRTY RULE, pre-registered);
    for index instruments only the first rule applies."""
    cal = nse_calendar(nse_old)
    calset = set(cal)
    syms = symbols(cache)
    nse_names = {s: NSE_ALIAS.get(s, s) for s in syms}
    want = set(nse_names.values())
    with open(os.path.join(nse_old, "symbol_changes.csv")) as fh:
        ch = list(csv.DictReader(fh))
    back = defaultdict(set)
    for r in ch:
        back[r["new_symbol"]].add(r["old_symbol"])
    for s in list(want):
        stack = [s]
        while stack:
            x = stack.pop()
            for o in back.get(x, ()):
                if o not in want:
                    want.add(o)
                    stack.append(o)
    rows = defaultdict(dict)                      # nse symbol -> date -> (open*af, close*af)
    with gzip.open(os.path.join(nse_old, "eq_daily_adj.csv.gz"), "rt") as fh:
        for r in csv.DictReader(fh):
            if r["series"] != "EQ" or r["symbol"] not in want:
                continue
            d = int(r["date"])
            if d < ERA[0] - 10000:
                continue
            rows[r["symbol"]][d] = (float(r["open"]), float(r["close"]))
    out = {}
    for s in syms + ["_NIFTY50_min", "_NIFTYBANK_min"]:
        sd = D.load_symbol(cache, s)
        if sd is None:
            continue
        days = [int(x) for x in sd.days]
        have = set(days)
        first = days[0]
        dirty = set()
        # holes: every NSE session after the first one present, missing from the data
        for i, d in enumerate(cal):
            if d > first and d not in have:
                j = i + 1
                while j < len(cal) and cal[j] not in have:
                    j += 1
                if j < len(cal):
                    dirty.add(cal[j])
        if not s.startswith("_"):
            names = {nse_names[s]} | {o for o in want if o in back.get(nse_names[s], set())}
            nse = {}
            for nm in names:
                for d, v in rows.get(nm, {}).items():
                    nse.setdefault(d, v)
            nd = sorted(nse)
            prev_nse = {nd[i]: nd[i - 1] for i in range(1, len(nd))}
            dd = sd.daily()
            for i in range(1, len(days)):
                d, pd_ = days[i], days[i - 1]
                if d in dirty or d not in calset:
                    continue
                kr = float(dd["o"][i] / dd["c"][i - 1])
                if d not in nse or prev_nse.get(d) != pd_:
                    # no NSE comparison possible (no EQ row that day, e.g. ABB's 2019-12-20 demerger ex-day):
                    # an overnight move beyond NOCMP_TOL is treated as dirty
                    if kr > 0 and abs(math.log(kr)) > NOCMP_TOL:
                        dirty.add(d)
                    continue
                o, _c = nse[d]
                nr = o / nse[pd_][1]
                if kr > 0 and nr > 0 and abs(math.log(kr / nr)) > DIRTY_TOL:
                    dirty.add(d)
        out[s] = sorted(dirty)
    return out


def symbols(cache: str = CACHE) -> list:
    with open(os.path.join(cache, "_manifest.json")) as fh:
        return sorted(json.load(fh)["symbols"])


def data_check(cache: str = CACHE, out: str = ROOT, nse_old: str = NSE_OLD, bars3y: str | None = None) -> dict:
    """Coverage per symbol-year, calendar gaps vs NSE, the split / bonus adjustment (vs nse_old's corporate actions),
    volume adjustment, overnight jumps, index / VIX closes vs NSE, and the seam with bars3y (2023-11-09 -> 11-10).
    No strategy is run."""
    bars3y = bars3y or os.path.join(SCRATCH, "bars3y")
    with open(os.path.join(cache, "_manifest.json")) as fh:
        man = json.load(fh)
    rep: dict = {"universe_note": "today's NIFTY-100 names only: survivorship-biased in favour of long strategies"}
    idx = _nse_index_daily(nse_old)
    nse_days = [int(x) for x in idx["NIFTY"]["date"] if ERA[0] <= x <= ERA[1]]
    nifty = D.load_symbol(cache, "_NIFTY50_min")
    kd = set(int(x) for x in nifty.days)
    special = {}
    with open(os.path.join(nse_old, "special_sessions.csv")) as fh:
        for r in csv.DictReader(fh):
            special[int(r["date"])] = r["weekday"]
    rep["calendar"] = {"nse_sessions": len(nse_days), "kite_nifty_sessions_kept": len(kd),
                       "kite_nifty_dropped_short": nifty.dropped,
                       "nse_sessions_missing_in_kite": [d for d in nse_days if d not in kd and d not in nifty.dropped],
                       "kite_sessions_not_in_nse": sorted(d for d in kd if d not in set(nse_days)),
                       "special_sessions_in_era": {str(k): v for k, v in special.items() if ERA[0] <= k <= ERA[1]}}
    years = [str(y) for y in range(2016, 2024)]
    cov = {}
    for stem, py in sorted(man["per_year"].items()):
        cov[stem] = {y: py.get(y, 0) for y in years}
    rep["coverage_bars_per_year"] = cov
    rep["empty_files"] = sorted(f"{s}_{y}" for s, py in man["per_year"].items() for y, n in py.items() if n == 0)
    # corporate actions of our symbols inside the era (nse_old, unadjusted NSE data)
    syms = sorted(man["symbols"])
    acts = []
    with open(os.path.join(nse_old, "adj_factors.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["symbol"] in syms and ERA[0] <= int(r["ex_date"]) <= ERA[1] and r["status"].startswith("applied"):
                acts.append(r)
    nse_vol = defaultdict(dict)
    want = {(r["symbol"], int(r["prev_session"] or 0)) for r in acts} | {(r["symbol"], int(r["first_session"] or 0))
                                                                         for r in acts}
    with gzip.open(os.path.join(nse_old, "eq_daily.csv.gz"), "rt") as fh:
        for r in csv.DictReader(fh):
            if r["series"] == "EQ" and (r["symbol"], int(r["date"])) in want:
                nse_vol[r["symbol"]][int(r["date"])] = (float(r["close"]), float(r["volume"]))
    checks, quality, jumps = [], {}, {}
    seam = {}
    for s in syms:
        sd = D.load_symbol(cache, s)
        if sd is None:
            continue
        q = D.quality_report(sd)
        quality[s] = {k: q[k] for k in ("sessions", "first", "last", "short_sessions", "ends_before_1515",
                                        "missing_minutes", "zero_volume_bars", "bad_ohlc_bars")}
        dd = sd.daily()
        pos = {int(d): i for i, d in enumerate(dd["date"])}
        if q["overnight_jumps_gt_15pct"]:
            jumps[s] = q["overnight_jumps_gt_15pct"]
        for r in [a for a in acts if a["symbol"] == s]:
            a, b = int(r["prev_session"] or 0), int(r["first_session"] or 0)
            if a not in pos or b not in pos:
                checks.append({"symbol": s, "ex_date": r["ex_date"], "kind": r["kind"], "factor": r["factor"],
                               "status": "not in kite data"})
                continue
            kgap = float(dd["o"][pos[b]] / dd["c"][pos[a]])
            ngap = float(r["gap_open_ratio"]) if r["gap_open_ratio"] else float("nan")
            nv = nse_vol[s]
            vr = None
            if a in nv and b in nv and nv[a][1] > 0 and nv[b][1] > 0:
                # Kite volume / NSE unadjusted volume on the day before and the ex-day: equal ratios = Kite volume
                # NOT adjusted; ratios differing by 1/factor = adjusted
                vr = [round(float(dd["v"][pos[a]] / nv[a][1]), 3), round(float(dd["v"][pos[b]] / nv[b][1]), 3)]
            checks.append({"symbol": s, "ex_date": r["ex_date"], "kind": r["kind"], "factor": float(r["factor"]),
                           "nse_unadjusted_gap": ngap, "kite_gap": round(kgap, 4),
                           "kite_adjusted": bool(abs(kgap - 1.0) < abs(kgap - float(r["factor"]))),
                           "kite_vol_over_nse_vol_prev_ex": vr})
        p3 = os.path.join(bars3y, f"{s}.json")
        if os.path.exists(p3):
            with open(p3) as fh:
                head = fh.read(400)
            try:
                first = head.split('"bars": [[', 1)[1].split("]", 1)[0].split(",")
                o3 = float(first[1])
                seam[s] = {"old_last_date": int(sd.days[-1]), "old_last_close": float(dd["c"][-1]),
                           "bars3y_first": first[0].strip('"')[:10], "bars3y_first_open": o3,
                           "gap": round(o3 / float(dd["c"][-1]) - 1.0, 4)}
            except Exception as exc:          # pragma: no cover - layout surprise
                seam[s] = {"error": str(exc)[:80]}
        del sd
    rep["corporate_action_check"] = checks
    rep["quality"] = quality
    rep["overnight_jumps_gt_15pct"] = jumps
    g = sorted((abs(v["gap"]), k) for k, v in seam.items() if "gap" in v)
    rep["seam_2023_11_09_to_10"] = {"n": len(g), "median_abs_gap": float(np.median([x for x, _ in g])) if g else None,
                                    "largest": [(k, seam[k]["gap"]) for _x, k in g[-8:]], "per_symbol": seam,
                                    "note": "no overlapping dates: bars_min_old ends 2023-11-09, bars3y starts 2023-11-10"}
    # index / VIX closes vs NSE official closes
    for name, key in (("_NIFTY50_min", "NIFTY"), ("_NIFTYBANK_min", "BANKNIFTY"), ("_VIX_min", "INDIAVIX")):
        sd = D.load_symbol(cache, name)
        if sd is None:
            rep[f"{name}_vs_nse"] = "missing"
            continue
        dd = sd.daily()
        z = idx[key]
        m = {int(d): c for d, c in zip(z["date"], z["c"])}
        diffs = [abs(float(c) / m[int(d)] - 1.0) for d, c in zip(dd["date"], dd["c"]) if int(d) in m and m[int(d)] > 0]
        rep[f"{name}_vs_nse"] = {"days": len(diffs), "median_abs_rel_diff_last_minute_close_vs_official":
                                 float(np.median(diffs)), "p99": float(np.quantile(diffs, 0.99)),
                                 "max": float(max(diffs))}
    dl = dirty_links(cache, nse_old)
    with open(os.path.join(cache, "dirty_links.json"), "w") as fh:
        json.dump(dl, fh, indent=0, sort_keys=True)
    rep["dirty_links"] = {"rule": dirty_links.__doc__.split("\n    A trade")[0].strip(),
                          "n_symbol_dates": sum(len(v) for v in dl.values()),
                          "per_symbol": {k: len(v) for k, v in dl.items()},
                          "examples": {k: v[:6] for k, v in dl.items() if v and not k.startswith("_")}}
    rep["nse_calendar_sessions"] = len(nse_calendar(nse_old))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "data_check.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    return rep


# ══════════════════════════════════════════════════════════════════════════
# trade aggregation (workers never ship millions of trade tuples to the parent)
# ══════════════════════════════════════════════════════════════════════════
def _calendar_days(cache: str = CACHE) -> np.ndarray:
    """The data's session calendar (NIFTY 50 minute sessions kept by the loader) inside the era."""
    sd = D.load_symbol(cache, "_NIFTY50_min")
    d = np.asarray(sd.days, np.int64)
    return d[(d >= ERA[0]) & (d <= ERA[1])]


class IndexRef:
    """Index minute prices for the report-only "excess vs NIFTY" line: price_at(date, "HH:MM", close=False)."""

    def __init__(self, sd):
        self.key = sd.date.astype(np.int64) * 10000 + sd.t.astype(np.int64)
        self.o, self.c, self.date = sd.o, sd.c, sd.date

    def price_at(self, date: int, hm: str, close: bool = False):
        if hm in ("", "order", None):
            hm, close = "15:29", True
        try:
            m = int(hm[:2]) * 60 + int(hm[3:5])
        except Exception:
            return None
        k = int(date) * 10000 + m
        j = int(np.searchsorted(self.key, k, side="right")) - 1
        if j < 0 or int(self.date[j]) != int(date):
            return None
        if int(self.key[j]) == k:
            return float(self.c[j] if close else self.o[j])
        return float(self.c[j])                    # last price before that minute on the same day


def new_agg(ndays: int) -> dict:
    return {"h1": [0, 0.0, 0.0, 0.0, 0], "h2": [0, 0.0, 0.0, 0.0, 0],    # n, net, gross profit, gross loss, wins
            "by_sym": defaultdict(float), "by_year": defaultdict(float), "daily": np.zeros(ndays),
            "s1": 0.0, "s2": 0.0, "gross": 0.0, "cost": 0.0, "n": 0, "long": 0,
            "ex": {"h1": [0, 0.0], "h2": [0, 0.0]}, "dirty_excluded": 0, "outside_era": 0}


def _dirty_hit(dirty: dict, sym: str, a: int, b: int, nxt: dict) -> bool:
    """A dirty date of `sym` in [a, b] or equal to the session right after b (DIRTY RULE)."""
    dl = dirty.get(sym)
    if dl is None or len(dl) == 0:
        return False
    j = int(np.searchsorted(dl, a))
    if j < len(dl) and dl[j] <= b:
        return True
    n = nxt.get(sym, {}).get(b)
    return n is not None and j < len(dl) and int(dl[j]) == n


def add_trades(agg: dict, trades, cal: np.ndarray, dirty: dict | None = None, nxt: dict | None = None,
               refs: dict | None = None, sym_map: dict | None = None, apply_dirty: bool = True) -> dict:
    """Fold lab trade tuples (sim.TRADE_FIELDS) into `agg`. Trades entered outside the era, or caught by the dirty
    rule, are counted and left out. refs = {ref name: IndexRef}; sym_map maps a trade symbol to (dirty key, ref)."""
    dirty, nxt, refs, sym_map = dirty or {}, nxt or {}, refs or {}, sym_map or {}
    for t in trades:
        e = int(t[_DATE])
        if not ERA[0] <= e <= ERA[1]:
            agg["outside_era"] += 1
            continue
        x = int(t[6]) if t[6] else e
        x = max(x, e)
        sym = t[_SYM]
        dkey, rname = sym_map.get(sym, (sym, "NIFTY"))
        if apply_dirty and _dirty_hit(dirty, dkey, e, x, nxt):
            agg["dirty_excluded"] += 1
            continue
        net, gross, cost = float(t[_NET]), float(t[10]), float(t[11])
        h = agg["h1"] if e <= H1[1] else agg["h2"]
        h[0] += 1
        h[1] += net
        if net > 0:
            h[2] += net
            h[4] += 1
        elif net < 0:
            h[3] -= net
        agg["by_sym"][sym] += net
        agg["by_year"][e // 10000] += net
        i = min(int(np.searchsorted(cal, x)), len(cal) - 1)
        agg["daily"][i] += net
        agg["s1"] += net
        agg["s2"] += net * net
        agg["gross"] += gross
        agg["cost"] += cost
        agg["n"] += 1
        side = int(t[3])
        agg["long"] += side > 0
        ref = refs.get(rname)
        if ref is not None:
            p0 = ref.price_at(e, str(t[4]))
            p1 = ref.price_at(x, str(t[7]), close=str(t[7]) in ("15:29", ""))
            if p0 and p1:
                notional = float(t[5]) * float(t[9])
                k = "h1" if e <= H1[1] else "h2"
                agg["ex"][k][0] += 1
                agg["ex"][k][1] += net - side * notional * (p1 / p0 - 1.0)
    return agg


def merge_agg(a: dict, b: dict) -> dict:
    for k in ("h1", "h2"):
        a[k] = [x + y for x, y in zip(a[k], b[k])]
        a["ex"][k] = [x + y for x, y in zip(a["ex"][k], b["ex"][k])]
    for k in ("by_sym", "by_year"):
        for kk, v in b[k].items():
            a[k][kk] += v
    a["daily"] = a["daily"] + b["daily"]
    for k in ("s1", "s2", "gross", "cost", "n", "long", "dirty_excluded", "outside_era"):
        a[k] += b[k]
    return a


def _pf(gp, gl):
    return round(gp / gl, 3) if gl > 0 else (float("inf") if gp > 0 else 0.0)


def _stats(v) -> dict:
    n, net, gp, gl, w = v
    return {"trades": int(n), "win_pct": round(100.0 * w / n, 1) if n else 0.0, "pf": _pf(gp, gl),
            "net": round(net, 2)}


def judge(agg: dict, horizon: str, single_underlying: bool = False, min_trades: dict | None = None) -> dict:
    """protocol_oos.judge's gates on this era's halves, from an aggregate (identical arithmetic to P.stats)."""
    need = (min_trades or MIN_TRADES_PER_HALF)[horizon]
    era = [x + y for x, y in zip(agg["h1"], agg["h2"])]
    h1, h2, a = _stats(agg["h1"]), _stats(agg["h2"]), _stats(era)
    by = dict(agg["by_sym"])
    top, top_net = max(by.items(), key=lambda kv: kv[1]) if by else ("", 0.0)
    share = top_net / a["net"] if a["net"] > 0 else float("nan")
    reasons = []
    if h1["net"] <= 0:
        reasons.append("H1 net <= 0")
    if h2["net"] <= 0:
        reasons.append("H2 net <= 0")
    if a["pf"] < MIN_PF:
        reasons.append(f"PF {a['pf']:.3f} < {MIN_PF}")
    if h1["trades"] < need:
        reasons.append(f"H1 {h1['trades']} trades < {need}")
    if h2["trades"] < need:
        reasons.append(f"H2 {h2['trades']} trades < {need}")
    if a["net"] > 0 and share > MAX_SYMBOL_SHARE and not single_underlying:
        reasons.append(f"{top} is {share:.0%} of net (> {MAX_SYMBOL_SHARE:.0%})")
    return {"passed": not reasons, "reasons": reasons, "h1": h1, "h2": h2, "era": a, "top_symbol": top,
            "top_share": share}


def agg_from_trades(trades, cal, dirty=None, nxt=None, refs=None, sym_map=None, apply_dirty=True) -> dict:
    return add_trades(new_agg(len(cal)), trades, cal, dirty, nxt, refs, sym_map, apply_dirty)


# ══════════════════════════════════════════════════════════════════════════
# the strategy list: groups a (live PAPER), b (other FINAL-confirmed minute-bar), c (every other minute-bar id)
# ══════════════════════════════════════════════════════════════════════════
LIVE = ("AUC-FADE-3PCT", "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED")
CONFIRMED_MINUTE = ("AUC-FADE-2PCT", "VRP-STRANGLE-016-NAKED", "r3_tfweekly_wk_loser_rev", "r3_options_ic_25_10",
                    "r3_options_wc_fri")
CALENDAR_BOUND = {
    "r4_mechanics_budget_mom": "r4_mechanics.BUDGET lists 2024-2026 Budget days only: never fires in 2016-2023",
    "r4_mechanics_mpc_all_mom": "r4_mechanics.MPC lists 2023-12..2026 RBI MPC days only: never fires in 2016-2023",
    "r4_mechanics_mpc_ratesens_mom": "r4_mechanics.MPC lists 2023-12..2026 RBI MPC days only",
    "r4_mechanics_preholiday_nifty_1520": "r4_mechanics.HOLIDAYS lists 2023-11..2026-11 holidays only",
    "r4_mechanics_preholiday_nifty_id": "r4_mechanics.HOLIDAYS lists 2023-11..2026-11 holidays only",
    "r4_mechanics_preholiday_stocks_id": "r4_mechanics.HOLIDAYS lists 2023-11..2026-11 holidays only",
    "r4_mechanics_sip_bnf_1520": "month's last regular session from r4_mechanics.HOLIDAYS (2023-11..2026-11 only)",
    "r4_mechanics_sip_nifty_1520": "month's last regular session from r4_mechanics.HOLIDAYS (2023-11..2026-11 only)",
}
NOT_MINUTE_CONFIRMED = {
    "r4_realopt_nf_wk_strangle_3pct": "FINAL-confirmed on 2023-26, but a daily real-price (EOD bhavcopy) option spec, "
                                      "not a minute-bar one; NIFTY weeklies exist only from 2019-02-11, so H1 "
                                      "(2016-2019) holds ~45 weeks < 60 by construction (excluded from the 2012-23 "
                                      "round for the same reason)",
}
C_ORDER_SEED = 20160101          # pre-registered execution order of group c (a fixed random permutation)
C_CHUNK = 100


def module_specs() -> dict:
    """{id: (runner name, spec)} for the module-level runners that need minute bars."""
    from . import auction as A, auction_r2 as A2, attention as T, options_vrp as V, options_vrp_vix as VX
    from . import options_r3 as R3, r3_crosssec as X, ltr as L
    out = {}
    for sp in A.SPECS:
        out[sp.id] = ("auction", sp)
    for sp in A2.SPECS:
        out[sp.id] = ("auction_r2", sp)
    for sp in T.SPECS:
        out[sp.id] = ("attention", sp)
    for sp in V.SPECS:
        out[sp.id] = ("options_vrp", sp)
    for sp in VX.SPECS:
        out[sp.id] = ("options_vrp_vix", sp)
    for sp in R3.SPECS:
        out[sp.id] = ("options_r3", sp)
    for sp in X.SPECS:
        out[sp.id] = ("r3_crosssec", sp)
    out[L.STRATEGY_ID] = ("ltr", L.PortfolioSpec())
    return out


def catalogue_minute_specs() -> dict:
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    return {sp.id: sp for sp in REGISTRY.values() if sp.horizon == "intraday" or sp.decision_time is not None}


def status_2023_26(scratch: str = SCRATCH) -> dict:
    """{id: status on the 3-year 2023-26 data}: CONFIRMED / REJECTED ON FINAL from the one-look FINAL ledgers (3-year
    dataset only), else the latest bars3y screen verdict in trials_v2.csv, else 'not screened on bars3y'."""
    st = {}
    for f in (os.path.join(scratch, "stratlab_cache", "final_ledger.json"),
              os.path.join(scratch, "stratlab_cache", "bars3y_fb3d4d", "final_ledger.json")):
        try:
            with open(f) as fh:
                L = json.load(fh)
        except Exception:
            continue
        for k, v in L.items():
            if "|726s|" not in k or v.get("peeked"):
                continue
            st[k.split("|")[0]] = "CONFIRMED" if v["final"].get("confirmed") else "REJECTED ON FINAL"
    scr = {}
    try:
        with open(os.path.join(scratch, "stratlab_cache", "trials_v2.csv"), newline="") as fh:
            for r in csv.DictReader(fh):
                if r["era"] == "2023-26" and r["stage"] == "screen" and r["dataset"] == "bars3y":
                    scr[r["strategy_id"]] = r["verdict"]
    except FileNotFoundError:
        pass
    allids = set(module_specs()) | set(catalogue_minute_specs())
    for i in allids:
        if i not in st:
            st[i] = scr.get(i, "not screened on bars3y")
    return st


def groups() -> dict:
    mods, cat = module_specs(), catalogue_minute_specs()
    a = list(LIVE)
    b = list(CONFIRMED_MINUTE)
    inel = [(i, "calendar-bound: " + w) for i, w in CALENDAR_BOUND.items()]
    inel += [(i, w) for i, w in NOT_MINUTE_CONFIRMED.items()]
    c = sorted(i for i in list(mods) + list(cat) if i not in a and i not in b and i not in CALENDAR_BOUND)
    rnd = random.Random(C_ORDER_SEED)
    rnd.shuffle(c)
    return {"a": a, "b": b, "c": c, "ineligible": inel}


def spec_of(i: str):
    mods = module_specs()
    if i in mods:
        return mods[i]
    return ("catalogue", catalogue_minute_specs()[i])


def runner_cost_model(runner: str) -> str:
    from . import sim
    if runner == "catalogue":
        return sim.COST_MODEL
    if runner in ("auction", "auction_r2"):
        from . import auction as A
        return A.COST_MODEL
    if runner == "attention":
        from . import attention as T
        return T.COST_MODEL
    if runner in ("options_vrp", "options_vrp_vix"):
        from . import options_vrp as V
        return V.COST_MODEL
    if runner == "options_r3":
        from . import options_r3 as R3
        return R3.COST_MODEL
    if runner == "r3_crosssec":
        from . import r3_crosssec as X
        return X.COST_MODEL
    if runner == "ltr":
        from . import ltr as L
        return L.COST_MODEL
    raise KeyError(runner)


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


CODE_FILES = ("protocol_min_oos.py", "protocol_oos.py", "protocol.py", "sim.py", "data.py", "auction.py",
              "auction_r2.py", "attention.py", "options_vrp.py", "options_vrp_vix.py", "options_r3.py",
              "r3_crosssec.py", "ltr.py", "nsedata.py", "oldera_fo.py")


def code_sha() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    out = {f: _sha(os.path.join(here, f)) for f in CODE_FILES}
    out["../costs.py"] = _sha(os.path.join(os.path.dirname(here), "costs.py"))
    out["../options_replay.py"] = _sha(os.path.join(os.path.dirname(here), "options_replay.py"))
    return out


def live_agent_fingerprints() -> dict:
    """The validated agents' pinned LAB_FINGERPRINT / VERSION (read-only import, nothing is started)."""
    out = {}
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if here not in sys.path:
            sys.path.insert(0, here)
        import agents.validated_attention as VA
        import agents.validated_auction as VU
        import agents.validated_strangle as VS
        out["AUC-FADE-3PCT"] = {"agent": "agents/validated_auction.py", "LAB_FINGERPRINT": VU.LAB_FINGERPRINT,
                                "VERSION": getattr(getattr(VU, "ValidatedAuctionAgent", None), "VERSION", "")}
        out["SWING-ATTN-ON-3X"] = {"agent": "agents/validated_attention.py", "LAB_FINGERPRINT": VA.LAB_FINGERPRINT}
        out["VRP-STRANGLE-016-WINGED"] = {"agent": "agents/validated_strangle.py",
                                          "LAB_FINGERPRINT": VS.LAB_FINGERPRINT}
    except Exception as exc:                       # pragma: no cover - environment without app deps
        out["error"] = repr(exc)[:200]
    return out


def preregister(path: str, repo_copy: str = REPO_PREREG) -> dict:
    """Write the pre-registration ONCE (refuses to overwrite), its byte-identical repo copy and its sha256."""
    if os.path.exists(path) or os.path.exists(repo_copy):
        raise SystemExit(f"{path} or {repo_copy} exists: a pre-registration is written once")
    g = groups()
    st = status_2023_26()
    strategies = {}
    for grp in ("a", "b", "c"):
        for k, i in enumerate(g[grp]):
            runner, sp = spec_of(i)
            strategies[i] = {"group": grp, "order": k, "runner": runner, "fingerprint": sp.fingerprint,
                             "kind": getattr(sp, "kind", ""), "horizon": sp.horizon,
                             "single_underlying": bool(getattr(sp, "underlying", None)),
                             "params": json.loads(json.dumps(getattr(sp, "params", {}), default=str)),
                             "cost_model": runner_cost_model(runner), "status_2023_26": st.get(i, "")}
    with open(os.path.join(CACHE, "_manifest.json"), "rb") as fh:
        man_sha = hashlib.sha256(fh.read()).hexdigest()
    doc = {
        "title": "Old-era minute-bar out-of-sample round (2016-01-01 .. 2023-11-09): one look, no tuning",
        "written": time.strftime("%Y-%m-%d %H:%M"),
        "era": ERA, "h1": H1, "h2": H2, "gates": GATES, "gates_fingerprint": GATES_FINGERPRINT,
        "report_only": {
            "sens": "same strategy at 5 bps slippage per side (index 2.5 bps; auction family 5 bps vs its 2.1; options "
                    "haircut x2.5) and whether it would pass there",
            "min_trades_60_intraday": "whether an intraday strategy would pass with 60 trades per half instead of 150",
            "excess_vs_nifty": "net minus side x the trade's notional x the NIFTY 50 (BANKNIFTY futures: NIFTY BANK) "
                               "move from the entry minute to the exit minute, per half; the relevant line for "
                               f"long-biased strategies (long share of era trades >= {LONG_BIASED_SHARE:.0%})",
            "listed_weeklies_only": "NIFTY weekly option strategies: the same trades restricted to entries on or "
                                    "after 2019-02-11 (first listed NIFTY weekly expiry 2019-02-14); before that "
                                    "the weekly contracts did not exist and the modelled trades are synthetic",
            "attention": "SWING-ATTN-ON-3X also with the 09:16-open exit and with CNC costs (as attention.py reports)",
        },
        "combined_verdict": "proven in both eras = FINAL-CONFIRMED on the 2023-26 3-year data AND OOS-CONFIRMED here",
        "luck_rate": "share of group-c ids with 2023-26 status FAILED SCREEN that are OOS-CONFIRMED here, against the "
                     "share of 2023-26 FINAL-CONFIRMED ids (groups a+b) that are OOS-CONFIRMED here; ids "
                     "'REJECTED ON FINAL' / 'PASSED SCREEN' / 'not screened on bars3y' are reported separately",
        "cost_model_note": "costs.py 2026 rates (COST_MODEL_VERSION) for every year; each runner's own fill model; "
                           "NIFTY / BANKNIFTY lot sizes as each module pre-registered (2026 lots for index futures "
                           "65 / 30; options 75 NIFTY, BANKNIFTY 30) for every date",
        "slippage_bps": {"stocks": 2.0, "index": 1.0, "auction_family": 2.1, "ltr": 2.1, "sens_stocks": SENS_BPS,
                         "sens_index": 2.5, "options_haircut": "max(1 pt, 2 %) per leg per fill, sens x2.5"},
        "options_model": "options_vrp / options_vrp_vix / options_r3: Black-Scholes MODEL premiums on the NIFTY 50 / "
                         "NIFTY BANK minute bars with the previous India VIX official close (nse_old) as one flat vol, "
                         "r 6.5 %, no skew; NOT option prices. NIFTY weekly options were listed only from 2019-02 "
                         "(BANKNIFTY weeklies from 2016-05), so the weekly strangles' 2016-01..2019-02 trades are on "
                         "contracts that did not exist. The verdict uses the whole era as modelled; the listed-weeklies "
                         "block is reported next to it.",
        "data": {"src": SRC_DIR, "cache": CACHE, "cache_manifest_sha256": man_sha,
                 "dirty_links_sha256": _sha(os.path.join(CACHE, "dirty_links.json")),
                 "dirty_rule": dirty_links.__doc__.strip(),
                 "cleaning": {"drop_before": DROP_BEFORE,
                              "why": {"IRCTC": "another instrument under the same Kite name before the 2019-10-14 "
                                               "listing", "PIIND": "prints of another instrument mixed in "
                                                                  "2016-01..2020-01"}},
                 "daily_aux": "_NIFTY50_day / _VIX_day / _NIFTYBANK_day = nse_old index_daily official OHLC",
                 "lookbacks": "no splice of daily history before 2016-01-01: lookbacks warm up inside 2016 as they did "
                              "at the start of the 2023-26 data (LTR's 252-session signal starts 2017)",
                 "nsedata": "r4_* specs read nsedata through oldera_fo's old-era view (nse_old_view)",
                 "universe": "today's NIFTY-100 names (97 stocks with data) - SURVIVORSHIP-BIASED in favour of long "
                             "strategies"},
        "execution": {"order": "a, then b, then c; c in a fixed random permutation (seed C_ORDER_SEED) in chunks of "
                               f"{C_CHUNK}: if time or disk runs out, the judged part of c is the completed prefix of "
                               "the permutation (a random subset by construction) and is reported as such",
                      "seed": C_ORDER_SEED, "chunk": C_CHUNK},
        "live_agents": live_agent_fingerprints(),
        "code_sha256": code_sha(),
        "strategies": strategies,
        "ineligible": [{"id": i, "reason": w} for i, w in g["ineligible"]],
        "counts": {k: len(g[k]) for k in ("a", "b", "c")},
        "note": "written before any 2016-2023 strategy result was computed (the data check ran no strategy)",
    }
    blob = (json.dumps(doc, indent=1, default=str) + "\n").encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(os.path.dirname(repo_copy), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    with open(repo_copy, "wb") as fh:
        fh.write(blob)
    sha = hashlib.sha256(blob).hexdigest()
    with open(PREREG_SHA256_FILE if path == PREREG else path + ".sha256", "w") as fh:
        fh.write(sha + "\n")
    return {"sha256": sha, "counts": doc["counts"], "path": path, "repo_copy": repo_copy}


def load_prereg(path: str, expect_sha: str | None = None) -> dict:
    with open(path, "rb") as fh:
        blob = fh.read()
    sha = hashlib.sha256(blob).hexdigest()
    if expect_sha is None and os.path.exists(path + ".sha256"):
        expect_sha = open(path + ".sha256").read().strip()
    if expect_sha is None and path == PREREG and os.path.exists(PREREG_SHA256_FILE):
        expect_sha = open(PREREG_SHA256_FILE).read().strip()
    if expect_sha and sha != expect_sha:
        raise SystemExit(f"pre-registration {path} sha256 {sha} != recorded {expect_sha}: refused")
    if os.path.exists(REPO_PREREG) and path != REPO_PREREG and _sha(REPO_PREREG) != sha:
        raise SystemExit(f"{path} differs from the repository copy {REPO_PREREG}: refused")
    doc = json.loads(blob)
    doc["_sha256"] = sha
    return doc


# ══════════════════════════════════════════════════════════════════════════
# the one-look ledger
# ══════════════════════════════════════════════════════════════════════════
def _load_json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return {}


def ledger_key(sid: str, cost_model: str) -> str:
    return f"{sid}|oldera_min|{cost_model}"


def partition(specs: list, cost_model_of, prereg: dict | None, ledger_path: str = LEDGER):
    """(already judged / refused results, todo specs). A judged strategy is NEVER recomputed: its first result is
    returned; a changed one is refused; one outside the pre-registration is refused."""
    ledger = _load_json(ledger_path)
    done, todo = [], []
    for sp in specs:
        cm = cost_model_of(sp)
        old = ledger.get(ledger_key(sp.id, cm))
        why = None
        if prereg is not None:
            reg = prereg.get("strategies", {}).get(sp.id)
            if prereg.get("gates_fingerprint") != GATES_FINGERPRINT:
                why = "gates differ from the pre-registration"
            elif reg is None:
                why = "not in the pre-registration"
            elif reg["fingerprint"] != sp.fingerprint:
                why = "code / parameters changed since the pre-registration"
        if why is None and old is not None:
            if old["fingerprint"] != sp.fingerprint or old.get("gates") != GATES_FINGERPRINT:
                why = f"already judged on {old['when']} with different code or gates: no second look"
            else:
                r = dict(old["result"])
                r.update({"id": sp.id, "note": f"first judged {old['when']} (ledger; not recomputed)",
                          "from_ledger": True})
                done.append(r)
                continue
        if why:
            done.append({"id": sp.id, "verdict": "NOT JUDGED", "why": why})
            continue
        todo.append(sp)
    return done, todo


def verdict_of(sp, res: dict) -> dict:
    """Judgement record from a runner result {"agg", "sens_agg", "error", "lookahead", "skipped", "extra"}."""
    single = bool(getattr(sp, "underlying", None))
    agg = res.get("agg")
    if agg is None:
        agg = new_agg(1)
    j = judge(agg, sp.horizon, single)
    j60 = judge(agg, sp.horizon, single, {"intraday": REPORT_MIN_TRADES_INTRADAY, "positional": 60})
    s = res.get("sens_agg")
    js = judge(s, sp.horizon, single) if s is not None else None
    if res.get("error"):
        verdict, why = "INVALID", "strategy raised: " + str(res["error"])[:300]
    elif res.get("lookahead"):
        verdict, why = "INVALID", "LOOK-AHEAD: " + str(res["lookahead"])[:300]
    else:
        verdict = "OOS-CONFIRMED" if j["passed"] else "OOS-FAILED"
        why = "all gates met" if j["passed"] else ", ".join(j["reasons"])
    n = agg["n"]
    mean = agg["s1"] / n if n else float("nan")
    var = (agg["s2"] - n * mean * mean) / (n - 1) if n > 1 else float("nan")
    out = {"verdict": verdict, "why": why, "h1": j["h1"], "h2": j["h2"], "era": j["era"],
           "top_symbol": j["top_symbol"],
           "top_share": None if j["top_share"] != j["top_share"] else round(j["top_share"], 4),
           "skipped": int(res.get("skipped", 0) or 0), "dirty_excluded": int(agg["dirty_excluded"]),
           "long_share": round(agg["long"] / n, 4) if n else None,
           "long_biased": bool(n and agg["long"] / n >= LONG_BIASED_SHARE),
           "excess_vs_index": {k: {"trades": int(v[0]), "excess": round(v[1], 2)} for k, v in agg["ex"].items()},
           "by_year": {str(k): round(v, 2) for k, v in sorted(agg["by_year"].items())},
           "mean_net": mean, "std_net": math.sqrt(var) if var == var and var > 0 else float("nan"),
           "mean_gross": agg["gross"] / n if n else float("nan"), "mean_cost": agg["cost"] / n if n else float("nan"),
           "pass_with_60_intraday": bool(j60["passed"]) if sp.horizon == "intraday" else None,
           "sens": None if js is None else {"era": js["era"], "h1": js["h1"], "h2": js["h2"], "passed": js["passed"],
                                            "reasons": js["reasons"]},
           "extra": res.get("extra") or {}}
    return out


def record(specs_res: list, cost_model_of, dataset: str = "bars_min_old", ledger_path: str = LEDGER,
           prereg_sha: str = "") -> list:
    """Write the first judgement of each (spec, result) to the ledger (first write wins) and return the records."""
    import fcntl
    now = time.strftime("%Y-%m-%d %H:%M")
    harness = _sha(os.path.abspath(__file__))
    new, out = {}, []
    for sp, res in specs_res:
        cm = cost_model_of(sp)
        v = verdict_of(sp, res)
        new[ledger_key(sp.id, cm)] = {"fingerprint": sp.fingerprint, "gates": GATES_FINGERPRINT, "when": now,
                                      "cost_model": cm, "dataset": dataset, "prereg_sha256": prereg_sha,
                                      "harness_sha256": harness, "result": v}
        out.append({**v, "id": sp.id, "note": "", "cost_model": cm})
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        ledger = _load_json(ledger_path)
        for k, v in new.items():
            ledger.setdefault(k, v)
        tmp = ledger_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(ledger, fh, indent=1, default=str)
        os.replace(tmp, ledger_path)
    return out


# ══════════════════════════════════════════════════════════════════════════
# runners: each returns {id: {"agg", "sens_agg", "error", "lookahead", "skipped", "trades", "extra"}}
# ══════════════════════════════════════════════════════════════════════════
_CTX: dict = {}
INDEX_SYMS = ("_NIFTY50_min", "_NIFTYBANK_min")
SYM_MAP = {"_NIFTY50_min": ("_NIFTY50_min", "NIFTY"), "_NIFTYBANK_min": ("_NIFTYBANK_min", "BANKNIFTY"),
           "NIFTY": ("_NIFTY50_min", None), "BANKNIFTY": ("_NIFTYBANK_min", None)}


def setup(cache: str = CACHE, nsedata_view: bool = True) -> dict:
    """Calendar, dirty links, next-session maps, index references; nsedata on the old-era view. Before forking."""
    _CTX["cache"] = cache
    _CTX["cal"] = _calendar_days(cache)
    with open(os.path.join(cache, "dirty_links.json")) as fh:
        _CTX["dirty"] = {k: np.asarray(v, np.int64) for k, v in json.load(fh).items()}
    nxt = {}
    for s in symbols(cache) + list(INDEX_SYMS):
        sd = D.load_symbol(cache, s)
        if sd is None:
            continue
        d = [int(x) for x in sd.days]
        nxt[s] = dict(zip(d[:-1], d[1:]))
        if s in INDEX_SYMS:
            _CTX.setdefault("index_sd", {})[s] = sd
    _CTX["nxt"] = nxt
    _CTX["refs"] = {"NIFTY": IndexRef(_CTX["index_sd"]["_NIFTY50_min"]),
                    "BANKNIFTY": IndexRef(_CTX["index_sd"]["_NIFTYBANK_min"])}
    if nsedata_view:
        from . import oldera_fo as OF
        if os.path.exists(os.path.join(OF.VIEW_DIR, "delivery.csv.gz")):
            N = OF.use(OF.VIEW_DIR)
            N._load()
            _CTX["nsedata"] = OF.VIEW_DIR
    return _CTX


def _agg(trades, apply_dirty=True):
    return agg_from_trades(trades, _CTX["cal"], _CTX["dirty"], _CTX["nxt"], _CTX["refs"], SYM_MAP, apply_dirty)


def _clean_row(sym, a, b) -> bool:
    return not _dirty_hit(_CTX["dirty"], sym, int(a), int(b), _CTX["nxt"])


def _cat_work(args):
    symbol, ids, slip, islip, audit, keep = args
    from . import sim
    from .api import REGISTRY
    specs = [REGISTRY[i] for i in ids]
    sd = D.load_symbol(_CTX["cache"], symbol)
    out = {}
    if sd is None:
        return symbol, out
    idx = _CTX["index_sd"]["_NIFTY50_min"]

    def fold(res):
        for i, r in res.items():
            if not r["trades"] and not r["error"] and not r["lookahead"] and not r["skipped"]:
                continue
            o = out.setdefault(i, {"agg": new_agg(len(_CTX["cal"])), "skipped": 0, "lookahead": None, "error": None,
                                   "trades": [] if i in keep else None})
            merge_agg(o["agg"], _agg(r["trades"]))
            o["skipped"] += r["skipped"]
            o["lookahead"] = o["lookahead"] or r["lookahead"]
            o["error"] = o["error"] or r["error"]
            if o["trades"] is not None:
                o["trades"].extend(r["trades"])
    # positional specs: one call over the whole era (their exits must not be cut at a period end)
    pos = [sp for sp in specs if sp.horizon == "positional"]
    if pos:
        fold(sim.run_symbol(_CTX["cache"], symbol, pos, ERA[0], ERA[1], audit=audit, slippage=slip,
                            index_slippage=islip, sd=sd, index=idx))
    # intraday specs: one call per calendar year. Every intraday trade opens and closes inside its session, so the
    # trades are identical to one call over the era; the sessions (and their cached indicators) of a year are freed
    # before the next (one call over 8 years needs > 4 GB per worker).
    intr = [sp for sp in specs if sp.horizon == "intraday"]
    if intr:
        for y in range(ERA[0] // 10000, ERA[1] // 10000 + 1):
            lo, hi = max(ERA[0], y * 10000 + 101), min(ERA[1], y * 10000 + 1231)
            fold(sim.run_symbol(_CTX["cache"], symbol, intr, lo, hi, audit=audit, slippage=slip,
                                index_slippage=islip, sd=sd, index=idx))
    return symbol, out


def _cat_pass(pool, ids, slip_bps, islip_bps, audit, keep=(), log=None):
    from .api import REGISTRY
    run_syms = _stock_syms()
    if any(getattr(REGISTRY[i], "underlying", None) for i in ids):
        run_syms = run_syms + list(INDEX_SYMS)
    merged = {i: {"agg": new_agg(len(_CTX["cal"])), "skipped": 0, "lookahead": None, "error": None,
                  "trades": [] if i in keep else None} for i in ids}
    t0 = time.time()
    jobs = [(s, ids, slip_bps / 1e4, islip_bps / 1e4, audit, set(keep)) for s in run_syms]
    for k, (sym, out) in enumerate(pool.imap_unordered(_cat_work, jobs), 1):
        for i, r in out.items():
            m = merged[i]
            merge_agg(m["agg"], r["agg"])
            m["skipped"] += r["skipped"]
            m["lookahead"] = m["lookahead"] or r["lookahead"]
            m["error"] = m["error"] or r["error"]
            if r["trades"] is not None:
                m["trades"].extend(r["trades"])
        if log and (k % 10 == 0 or k == len(jobs)):
            log(f"    {k}/{len(jobs)} symbols ({time.time() - t0:.0f}s, slippage {slip_bps:g} bps)")
    return merged


def run_catalogue(specs, jobs: int, keep=(), log=print) -> dict:
    from . import sim
    ids = [sp.id for sp in specs]
    # maxtasksperchild=1: strategy modules keep per-symbol caches (candles, features); a fresh worker per symbol keeps
    # memory at about one symbol's worth (a long-lived worker grew past 4 GB on these 8 years)
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=1) as pool:
        base = _cat_pass(pool, ids, sim.DEFAULT_SLIPPAGE_BPS, sim.INDEX_SLIPPAGE_BPS, True, keep, log)
        sens = _cat_pass(pool, ids, SENS_BPS, sim.INDEX_SLIPPAGE_BPS * SENS_BPS / sim.DEFAULT_SLIPPAGE_BPS, False,
                         (), log)
    out = {}
    for i in ids:
        b = base[i]
        out[i] = {"agg": b["agg"], "sens_agg": sens[i]["agg"], "error": b["error"], "lookahead": b["lookahead"],
                  "skipped": b["skipped"], "trades": b["trades"]}
    return out


def _stock_syms() -> list:
    syms = symbols(_CTX["cache"])
    lim = _CTX.get("symbols_limit")
    return syms[:lim] if lim else syms


def _stock_sds():
    for s in _stock_syms():
        sd = D.load_symbol(_CTX["cache"], s)
        if sd is not None:
            yield s, sd


def _block(trades) -> dict:
    a = _agg(trades, apply_dirty=False)
    return {"era": _stats([x + y for x, y in zip(a["h1"], a["h2"])]), "h1": _stats(a["h1"]), "h2": _stats(a["h2"])}


def _module_result(specs, sim_, sens_, extra_fn=None) -> dict:
    out = {}
    res = sim_(specs, ERA[0], ERA[1])
    sres = sens_(specs, ERA[0], ERA[1]) if sens_ else {}
    for sp in specs:
        r = res.get(sp.id, {"trades": [], "error": "no result"})
        s = sres.get(sp.id)
        out[sp.id] = {"agg": _agg(r["trades"]), "sens_agg": _agg(s["trades"]) if s else None,
                      "error": r.get("error"), "lookahead": r.get("lookahead"), "skipped": r.get("skipped", 0),
                      "trades": list(r["trades"]), "extra": extra_fn(sp, r["trades"]) if extra_fn else {}}
    return out


def run_auction(specs, which: str = "auction") -> dict:
    from . import auction as A, auction_r2 as A2
    rows = []
    for s, sd in _stock_sds():
        if which == "auction":
            rows.extend(A.candidates_for(s, A.session_features(sd), min(sp.params["min_gap"] for sp in A.SPECS)))
        else:
            rows.extend(A2.candidates_for(s, A2.session_features(sd)))
    n0 = len(rows)
    rows = [r for r in rows if _clean_row(r[1], r[0], r[0])]
    mod = A if which == "auction" else A2
    slip = A.DEFAULT_SLIPPAGE_BPS if which == "auction" else A2.DEFAULT_SLIPPAGE_BPS
    out = _module_result(specs, mod.make_simulator(rows, slip), mod.make_simulator(rows, SENS_BPS))
    for v in out.values():
        v["extra"]["candidate_rows_before_after_dirty_rule"] = [n0, len(rows)]
        v["extra"]["slippage_bps"] = slip
    return out


def run_attention(specs) -> dict:
    from . import attention as T
    raw, counts = [], defaultdict(list)
    for s, sd in _stock_sds():
        for d, nb in zip(sd.days, sd.ends - sd.starts):
            counts[int(d)].append(int(nb))
        for d in sd.dropped or []:
            counts[int(d)].append(0)
        raw.extend(T.session_features(sd))
    rows, special = T.filter_rows(raw, counts)
    n1 = len(rows)
    rows = [r for r in rows if _clean_row(r["symbol"], r["date"], r["exit_date"])]

    def extra(sp, trades):
        e = {"signals_before_exclusions": len(raw), "after_lab_exclusions": n1, "after_dirty_rule": len(rows),
             "special_days": sorted(special)}
        for name, sim_ in (("exit_0916", T.make_simulator(rows, T.DEFAULT_SLIPPAGE_BPS, exit_key="exit2")),
                           ("cnc_costs", T.make_simulator(rows, T.DEFAULT_SLIPPAGE_BPS, kind="equity_cnc"))):
            e[name] = _block(sim_([sp], ERA[0], ERA[1])[sp.id]["trades"])
        return e
    return _module_result(specs, T.make_simulator(rows, T.DEFAULT_SLIPPAGE_BPS), T.make_simulator(rows, SENS_BPS),
                          extra)


def _or_bars(name: str) -> list:
    """options_replay Bar tuples (datetime, o, h, l, c) from the cache; short sessions already dropped by data.py."""
    from datetime import datetime
    sd = D.load_symbol(_CTX["cache"], name)
    out = []
    for d, t, o, h, l, c in zip(sd.date, sd.t, sd.o, sd.h, sd.l, sd.c):
        d, t = int(d), int(t)
        out.append((datetime(d // 10000, d // 100 % 100, d % 100, t // 60, t % 60), float(o), float(h), float(l),
                    float(c)))
    return out


def _vix() -> dict:
    from datetime import date
    z = D.load_daily_aux(_CTX["cache"], "_VIX_day")
    return {date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100): float(c) for d, c in zip(z["date"], z["c"])}


LISTED_NIFTY_WEEKLIES = 20190211


def _listed_block(sp, trades) -> dict:
    if getattr(sp, "underlying", "") != "NIFTY":
        return {}
    tr = [t for t in trades if int(t[2]) >= LISTED_NIFTY_WEEKLIES]
    a = _agg(tr)
    return {"listed_weeklies_only_from_20190211": {
        "era": _stats([x + y for x, y in zip(a["h1"], a["h2"])]), "h1_part": _stats(a["h1"]), "h2": _stats(a["h2"])}}


def _patch_trading_days():
    """options_replay treats a weekday without bars inside the data range as a holiday and asks ist_clock outside it.
    ist_clock does not list 2016-2023 holidays, so the look-ahead audit's truncated replays (which end at the entry
    bar) would see a different calendar after the cut than the full replay and report false look-ahead. Inside the era
    the trading-day function is therefore the FULL data's session set (exactly what the full replay uses there);
    outside it the module's own rule. Returns the original function (restore it afterwards)."""
    import options_replay as OR
    from datetime import date
    orig = OR.make_trading_day_fn
    full = set()
    for name in INDEX_SYMS:
        full |= {date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100) for d in _CTX["index_sd"][name].days}
    lo, hi = date(ERA[0] // 10000, ERA[0] // 100 % 100, ERA[0] % 100), max(full)

    def make(sessions):
        base = orig(sessions)

        def fn(d):
            if lo <= d <= hi:
                return d in full
            return base(d)
        return fn
    OR.make_trading_day_fn = make
    return orig


def run_options(specs, which: str, jobs: int = 4) -> dict:
    import options_replay as OR
    orig = _patch_trading_days()
    try:
        return _run_options(specs, which, jobs)
    finally:
        OR.make_trading_day_fn = orig


def _run_options(specs, which: str, jobs: int = 4) -> dict:
    from . import options_vrp as V, options_vrp_vix as VX, options_r3 as R3
    vix = _vix()
    if which == "options_r3":
        R3._G["bars"] = {"NIFTY": _or_bars("_NIFTY50_min"), "BANKNIFTY": _or_bars("_NIFTYBANK_min")}
        R3._G["vix"] = vix
        sim_ = R3.make_simulator(jobs)
        sens_ = R3.make_simulator(jobs, haircut_mult=R3.SENS_MULT, tag="sens", audit=False)
    else:
        bars = _or_bars("_NIFTY50_min")
        if which == "options_vrp_vix":
            VX.HIGHVIX.vix = vix
        sim_ = V.make_simulator(bars, vix)
        sens_ = V.make_simulator(bars, vix, haircut_mult=V.SENS_MULT, tag="sens", audit=False)
    out = _module_result(specs, sim_, sens_, _listed_block)
    for v in out.values():
        v["extra"]["modelled"] = "Black-Scholes model premiums (flat previous-day India VIX), not option prices"
    return out


def run_crosssec(specs) -> dict:
    from . import r3_crosssec as X
    feats, audit = {}, None
    for s, sd in _stock_sds():
        feats[s] = X.symbol_features(sd)
        audit = audit or X.audit_features(sd)
    nifty = X.symbol_features(_CTX["index_sd"]["_NIFTY50_min"])
    panel = X.build_panel(feats, nifty)
    out = _module_result(specs, X.make_simulator(panel, X.DEFAULT_SLIPPAGE_BPS, audit_every=7),
                         X.make_simulator(panel, SENS_BPS))
    for v in out.values():
        if audit:
            v["lookahead"] = v["lookahead"] or ("feature audit: " + str(audit))
    return out


def run_ltr(specs) -> dict:
    from . import ltr as L
    per = {s: L.minute_daily(sd) for s, sd in _stock_sds()}
    panel = L.build_panel(per, {s: {} for s in per}, None)
    sp = specs[0]

    def sim_(specs_, lo, hi, bps=L.SLIPPAGE_BPS):
        r = L.simulate_period(panel, lo, hi, bps)
        return {sp.id: {"trades": r["excess"], "skipped": 0, "lookahead": None, "error": None}}
    out = _module_result(specs, sim_, lambda s_, lo, hi: sim_(s_, lo, hi, SENS_BPS))
    out[sp.id]["extra"]["note"] = "trades are LTR position-trades with net = EXCESS over the equal-weight universe " \
                                  "(ltr.py's protocol unit); no history splice, so the first signal is in 2017"
    return out


def run_runner(runner: str, specs, jobs: int, keep=(), log=print) -> dict:
    if runner == "catalogue":
        return run_catalogue(specs, jobs, keep, log)
    if runner in ("auction", "auction_r2"):
        return run_auction(specs, runner)
    if runner == "attention":
        return run_attention(specs)
    if runner in ("options_vrp", "options_vrp_vix", "options_r3"):
        return run_options(specs, runner, jobs)
    if runner == "r3_crosssec":
        return run_crosssec(specs)
    if runner == "ltr":
        return run_ltr(specs)
    raise KeyError(runner)


# ══════════════════════════════════════════════════════════════════════════
# orchestration, outputs, trials ledger
# ══════════════════════════════════════════════════════════════════════════
def trials_rows(sp, res: dict, rec: dict, source: str, trades_file: str = "") -> dict:
    from . import trials as TR
    agg = res["agg"]
    cal = _CTX["cal"]
    dm = TR.moments(list(agg["daily"])) if len(cal) else TR.moments([])
    n = agg["n"]
    std = rec["std_net"]
    return {"strategy_id": sp.id, "family": TR.family_of(sp.id), "group": getattr(sp, "family", ""),
            "params_fp": sp.fingerprint, "era": ERA_TAG, "protocol": PROTOCOL_TAG, "stage": "oos",
            "dataset": "bars_min_old", "period": f"{ERA[0]}-{ERA[1]}", "cost_model": rec["cost_model"],
            "trades": n, "net": rec["era"]["net"], "pf": rec["era"]["pf"], "mean_net": rec["mean_net"],
            "std_net": std, "sr_trade": rec["mean_net"] / std if std and std == std and std > 0 else None,
            "mean_gross": rec["mean_gross"], "mean_cost": rec["mean_cost"], "n_days": dm["n"], "sr_daily": dm["sr"],
            "skew_daily": dm["skew"], "kurt_daily": dm["kurt"], "verdict": rec["verdict"], "source": source,
            "trades_file": trades_file, "note": rec["why"]}


def _fmt(x):
    if x is None or x == "":
        return ""
    if x == float("inf"):
        return "inf"
    return f"{x:,.0f}"


def write_outputs(out: str, records: list, meta: str) -> str:
    os.makedirs(out, exist_ok=True)
    L = [f"OLD-ERA MINUTE OUT-OF-SAMPLE  {ERA[0]}..{ERA[1]}  H1 {H1[0]}..{H1[1]}  H2 {H2[0]}..{H2[1]}  "
         f"gates {GATES_FINGERPRINT}", meta,
         "SURVIVORSHIP: today's NIFTY-100 names only (biased in favour of long strategies). Costs: 2026 rates on "
         "every year. Options: modelled premiums, not prices.", "",
         f"{'id':<40}{'grp':>4}{'tr H1':>7}{'tr H2':>7}{'PF':>6}{'net H1':>11}{'net H2':>11}{'net era':>11}"
         f"{'net@5':>11}{'s5':>3}{'top%':>6}{'xsH1':>10}{'xsH2':>10}  verdict        why"]
    rows = []
    for r in records:
        if r.get("verdict") == "NOT JUDGED":
            L.append(f"{r['id']:<40}{r.get('group', ''):>4}  NOT JUDGED  {r['why']}")
            rows.append({"id": r["id"], "group": r.get("group", ""), "verdict": "NOT JUDGED", "why": r["why"]})
            continue
        s = r.get("sens") or {}
        ex = r.get("excess_vs_index") or {}
        L.append(f"{r['id']:<40}{r.get('group', ''):>4}{r['h1']['trades']:>7}{r['h2']['trades']:>7}"
                 f"{('inf' if r['era']['pf'] == float('inf') else format(r['era']['pf'], '.2f')):>6}"
                 f"{_fmt(r['h1']['net']):>11}{_fmt(r['h2']['net']):>11}{_fmt(r['era']['net']):>11}"
                 f"{_fmt(s.get('era', {}).get('net')) if s else '':>11}{('Y' if s.get('passed') else 'N') if s else '':>3}"
                 f"{(format(100 * r['top_share'], '.0f') if r.get('top_share') is not None else ''):>6}"
                 f"{_fmt(ex.get('h1', {}).get('excess')) if ex.get('h1', {}).get('trades') else '':>10}"
                 f"{_fmt(ex.get('h2', {}).get('excess')) if ex.get('h2', {}).get('trades') else '':>10}  "
                 f"{r['verdict']:<15}{r['why']}" + (f" | {r['note']}" if r.get("note") else ""))
        rows.append({"id": r["id"], "group": r.get("group", ""), "runner": r.get("runner", ""),
                     "status_2023_26": r.get("status_2023_26", ""), "verdict": r["verdict"], "why": r["why"],
                     "h1_trades": r["h1"]["trades"], "h2_trades": r["h2"]["trades"], "h1_net": r["h1"]["net"],
                     "h2_net": r["h2"]["net"], "era_trades": r["era"]["trades"], "era_pf": r["era"]["pf"],
                     "era_net": r["era"]["net"], "era_win_pct": r["era"]["win_pct"],
                     "top_symbol": r.get("top_symbol"), "top_share": r.get("top_share"),
                     "sens_era_net": s.get("era", {}).get("net", "") if s else "",
                     "sens_era_pf": s.get("era", {}).get("pf", "") if s else "",
                     "sens_passed": s.get("passed", "") if s else "",
                     "pass_with_60_intraday": r.get("pass_with_60_intraday"),
                     "long_share": r.get("long_share"), "long_biased": r.get("long_biased"),
                     "excess_h1": ex.get("h1", {}).get("excess"), "excess_h2": ex.get("h2", {}).get("excess"),
                     "dirty_excluded": r.get("dirty_excluded"), "skipped": r.get("skipped"),
                     "mean_gross": r.get("mean_gross"), "mean_cost": r.get("mean_cost"),
                     "cost_model": r.get("cost_model", ""), "note": r.get("note", "")})
    judged = [r for r in records if r.get("verdict") in ("OOS-CONFIRMED", "OOS-FAILED")]
    conf = [r for r in judged if r["verdict"] == "OOS-CONFIRMED"]
    L += ["", f"{len(records)} strategies: {len(judged)} judged, {len(conf)} OOS-CONFIRMED. About {P.GATE_LUCK:.0%} of "
          "useless strategies pass each coin-flip-like gate; a confirmation earns a paper-trading trial, nothing more."]
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(records, fh, indent=1, default=str)
    return text


def run_group(grp: str, prereg_path: str, jobs: int, trials_path: str = "", ids_filter=None, log=print,
              max_chunks: int | None = None) -> list:
    """Judge group a / b / c once. c runs in its pre-registered random order, chunk by chunk (module runners in the
    order they appear, catalogue ids in chunks of C_CHUNK); each chunk is written to the ledger as it completes."""
    prereg = load_prereg(prereg_path)
    reg = prereg["strategies"]
    ids = [i for i, v in sorted(reg.items(), key=lambda kv: kv[1]["order"]) if v["group"] == grp]
    if ids_filter:
        ids = [i for i in ids if i in ids_filter]
    setup()
    st = status_2023_26()
    keep = set(LIVE) | set(CONFIRMED_MINUTE)
    by_runner = defaultdict(list)
    for i in ids:
        runner, sp = spec_of(i)
        by_runner[runner].append(sp)
    batches = []
    for runner, sps in by_runner.items():
        if runner == "catalogue":
            for k in range(0, len(sps), C_CHUNK):
                batches.append((runner, sps[k:k + C_CHUNK]))
        else:
            batches.append((runner, sps))
    # c: the catalogue chunks follow the permutation order; module runners first (few ids, fast)
    batches.sort(key=lambda b: (b[0] == "catalogue", min(reg[s.id]["order"] for s in b[1])))
    out_dir = os.path.join(ROOT, f"run_{grp}")
    os.makedirs(os.path.join(out_dir, "trades"), exist_ok=True)
    all_records = []
    for bi, (runner, sps) in enumerate(batches):
        if max_chunks is not None and bi >= max_chunks:
            log(f"  stopping after {max_chunks} batches (pre-registered prefix rule)")
            break
        cm_of = lambda sp, _r=runner: runner_cost_model(_r)  # noqa: E731
        done, todo = partition(sps, cm_of, prereg)
        log(f"[{grp}] batch {bi + 1}/{len(batches)} {runner}: {len(sps)} ids, {len(todo)} to judge, "
            f"{len(done)} already judged / refused")
        recs = []
        if todo:
            t0 = time.time()
            res = run_runner(runner, todo, jobs, keep, log)
            recs = record([(sp, res[sp.id]) for sp in todo], cm_of, prereg_sha=prereg["_sha256"])
            rows = []
            for sp, rec in zip(todo, recs):
                r = res[sp.id]
                tf = ""
                if r.get("trades"):
                    from . import sim
                    tf = os.path.join(out_dir, "trades", sp.id + ".csv.gz")
                    with gzip.open(tf, "wt", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(sim.TRADE_FIELDS)
                        w.writerows(sorted(r["trades"], key=lambda t: (int(t[2]), str(t[4]), str(t[1]))))
                rows.append(trials_rows(sp, r, rec, source=f"oldera_min/ledger.json#{ledger_key(sp.id, rec['cost_model'])}",
                                        trades_file=tf))
            if trials_path:
                from . import trials as TR
                n = TR.TrialsLedger(trials_path).append(rows)
                log(f"  trials ledger: {n} row(s) appended to {trials_path}")
            log(f"  batch done in {time.time() - t0:.0f}s")
        for r in done + recs:
            r["group"], r["runner"], r["status_2023_26"] = grp, runner, st.get(r["id"], "")
        all_records.extend(done + recs)
        write_outputs(out_dir, all_records, f"group {grp}; prereg {prereg_path} sha256 {prereg['_sha256'][:16]}; "
                                            f"jobs {jobs}")
    return all_records


def smoke(jobs: int = 4, n_symbols: int = 8, lo: int = 20160101, hi: int = 20160331) -> dict:
    """Plumbing check of every runner on a short old-era slice and a few symbols: reports ONLY whether each id
    raised / failed its look-ahead audit. No P&L, no trade counts, nothing written to the ledger or the trials file."""
    global ERA
    saved = ERA
    setup()
    _CTX["symbols_limit"] = n_symbols
    g = groups()
    by_runner = defaultdict(list)
    for i in g["a"] + g["b"] + g["c"]:
        r, sp = spec_of(i)
        by_runner[r].append(sp)
    bad = {}
    try:
        ERA = (lo, hi)
        for runner, sps in by_runner.items():
            try:
                res = run_runner(runner, sps, jobs, (), log=lambda *_: None)
            except Exception as exc:
                import traceback
                for sp in sps:
                    bad[sp.id] = f"runner {runner} raised: {exc!r} {traceback.format_exc()[-300:]}"
                continue
            for sp in sps:
                r = res.get(sp.id, {})
                if r.get("error") or r.get("lookahead"):
                    bad[sp.id] = str(r.get("error") or r.get("lookahead"))[:200]
    finally:
        ERA = saved
        _CTX.pop("symbols_limit", None)
    return {"runners": {k: len(v) for k, v in by_runner.items()}, "bad": bad}


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.protocol_min_oos", description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--data-check", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preregister", default="")
    ap.add_argument("--prereg", default=PREREG)
    ap.add_argument("--run", default="", choices=("", "a", "b", "c"))
    ap.add_argument("--ids", default="", help="restrict the run to these ids of the group")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--trials", default=os.environ.get("STRATLAB_TRIALS", ""))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--smoke", action="store_true", help="errors / look-ahead only, 2016-Q1, 8 symbols, no P&L")
    a = ap.parse_args(argv)
    if a.smoke:
        r = smoke(a.jobs)
        print(f"smoke: runners {r['runners']}; {len(r['bad'])} ids raised / failed the audit")
        for i, w in r["bad"].items():
            print(f"  {i}: {w}")
        return 1 if r["bad"] else 0
    if a.build_cache:
        man = build_cache(jobs=a.jobs)
        print(f"cache {CACHE}: {len(man['symbols'])} symbols, aux {sorted(man['aux'])}, empty {man['empty']}")
        return 0
    if a.data_check:
        rep = data_check()
        print(json.dumps({k: v for k, v in rep.items() if k not in ("quality", "coverage_bars_per_year",
                                                                      "seam_2023_11_09_to_10", "dirty_links")},
                         indent=1, default=str)[:6000])
        return 0
    if a.list:
        g = groups()
        st = status_2023_26()
        for k in ("a", "b", "c"):
            print(f"GROUP {k}: {len(g[k])}")
            for i in g[k]:
                print(f"  {i:<44}{spec_of(i)[0]:<16}{st.get(i, '')}")
        print(f"INELIGIBLE {len(g['ineligible'])}")
        for i, w in g["ineligible"]:
            print(f"  {i:<44}{w}")
        return 0
    if a.preregister:
        r = preregister(a.preregister)
        print(json.dumps(r, indent=1))
        return 0
    if a.run:
        ids = {x.strip() for x in a.ids.split(",") if x.strip()} or None
        recs = run_group(a.run, a.prereg, a.jobs, a.trials, ids, max_chunks=a.max_chunks)
        print(open(os.path.join(ROOT, f"run_{a.run}", "summary.txt")).read())
        print(f"{len(recs)} records")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
