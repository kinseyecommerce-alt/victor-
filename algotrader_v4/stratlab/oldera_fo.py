"""
stratlab/oldera_fo.py — the old era (2012-01-01 .. 2023-11-09) for nsedata-based strategies: a directory switch,
the expiry calendar and historical lot sizes. nsedata.py itself is NOT changed.

DIRECTORY SWITCH
  build_view() assembles <scratchpad>/nse_old_view (env STRATLAB_OLD_VIEW) in exactly the layout nsedata reads, from
  the read-only fetch output <scratchpad>/nse_old (nothing there is modified):
    delivery.csv.gz       from eq_daily.csv.gz (series EQ, NSE official, UNADJUSTED, as in the 3-year delivery file);
                          bars_symbol = the canonical old-era symbol (renames resolved like data_daily), so nsedata's
                          alias maps every old F&O name onto the daily lab's symbol; only stocks that are in the
                          point-in-time universe at least once (347) are kept: they are "the lab's symbols" for
                          nsedata (r4_delivery's market breadth ranks over them, as over the 97 bars3y names)
    fo_parts/fut, optidx  per-day bhavcopy parts with the `lot` column filled from nse_old/lot_sizes.csv (the old
                          format has no lot column; lot of a contract = the lot of the period whose expiry range holds
                          its expiry); optstk, optstk_full: symlinks; status.csv copied
    participant_oi.csv.gz participant-wise OI, fetched for this round (raw files in raw_part/oi, scripts/)
    ban.csv.gz            F&O ban list, raw files in raw_part/ban (nsedata reads both)
    bars/                 _NIFTY50_day.json, _NIFTYBANK_min.json (one bar per session, 15:29), _VIX_day.json from
                          index_daily.csv.gz: nsedata's index spot (STRATLAB_BARS_DIR equivalent)
  use() points the imported nsedata module at the view (NSE_DIR, BARS_DIR) and drops its in-memory tables; the
  look-ahead rule is unchanged (every accessor still returns only rows dated strictly before `before`; EOD of D is
  usable from D+1's open). nsedata builds its own .npz cache in <view>/_cache.

EXPIRY CALENDAR (rule, checked against the bhavcopy in tests_oldera.py)
  monthly: last Thursday of the month, moved to the previous trading day if it is a holiday (sessions = the daily
  lab calendar, i.e. without Muhurat / special sessions: 2017-10-19, 2018-11-07, 2021-11-04 expiries moved a day earlier)
  NIFTY weeklies: every Thursday from 2019-02-14 (first listed 2019-02-11); BANKNIFTY weeklies: Thursdays from
  2016-06-02 (first listed 2016-05-27), Wednesdays from 2023-09-06 (NSE moved BANKNIFTY to Wednesday; the
  2023-09-07/14/21 and 10-05 Thursday labels were relabelled on 2023-09-01); holiday -> previous trading day; no
  separate weekly in the week of the monthly expiry. Known exception: April 2014 (election holiday on the 24th; the
  contract keeps its 2014-04-24 label in the bhavcopy, the last trading day was the 23rd).
  listed_expiries() gives the bhavcopy's own list (what nsedata.expiries() sees), the ground truth.

LOT SIZES: lot_size(sym, trade_date, expiry=None) from lot_sizes.csv (implied from turnover; NIFTY 50 / 25 (Nov 2014
contract) / 75 (Nov 2015) / 50 (Jul 2021); BANKNIFTY 25 / 30 / 40 / 20 / 25 / 15). Only the historical-lot
SENSITIVITY of option runs uses exchange lots; the judged option P&L uses options_vrp.LOT_SCHEDULE as pre-registered.

    cd algotrader_v4
    python -m stratlab.oldera_fo --build-view          # after the participant / ban fetch has finished
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import gzip
import io
import json
import os
import re
import shutil
import sys

import numpy as np

from . import data_daily as DD

_SCRATCH = DD._SCRATCH
OLD_DIR = DD.OLD_DIR
VIEW_DIR = os.environ.get("STRATLAB_OLD_VIEW", os.path.join(_SCRATCH, "nse_old_view"))
NIFTY_WEEKLY_FROM = 20190214
BNF_WEEKLY_FROM = 20160602
BNF_WEDNESDAY_FROM = 20230906


# ══════════════════════════════════════════════════════════════════════════
# lot sizes
# ══════════════════════════════════════════════════════════════════════════
_LOTS: dict = {}


def _i(s: str) -> int:
    return int(s.replace("-", ""))


def _lots(old_dir: str = OLD_DIR) -> dict:
    if old_dir in _LOTS:
        return _LOTS[old_dir]
    out: dict = {}
    with open(os.path.join(old_dir, "lot_sizes.csv")) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["symbol"], []).append((_i(r["expiry_from"]), _i(r["expiry_to"]), _i(r["from_date"]),
                                                    _i(r["to_date"]), int(r["lot"])))
    for k in out:
        out[k].sort()
    _LOTS[old_dir] = out
    return out


def lot_size(sym: str, trade_date: int, expiry: int | None = None, old_dir: str = OLD_DIR) -> int | None:
    """Exchange lot of `sym`'s contract expiring `expiry` (default: the period in force on trade_date)."""
    rows = _lots(old_dir).get(sym)
    if not rows:
        return None
    if expiry is not None:
        best = rows[0][4]
        for ef, et, _f, _t, lot in rows:
            if ef <= int(expiry):
                best = lot
        return best
    best = rows[0][4]
    for _ef, _et, f, _t, lot in rows:
        if f <= int(trade_date):
            best = lot
    return best


def _lot_col(df, old_dir: str):
    """Vectorised lot per row (symbol, expiry) for a parts DataFrame."""
    lots = _lots(old_dir)
    ex = df["expiry"].str.replace("-", "").astype(int).to_numpy()
    out = np.full(len(df), np.nan)
    syms = df["symbol"].to_numpy()
    for s in np.unique(syms):
        rows = lots.get(s)
        if not rows:
            continue
        m = syms == s
        ef = np.array([r[0] for r in rows])
        lv = np.array([r[4] for r in rows], float)
        k = np.searchsorted(ef, ex[m], side="right") - 1
        out[m] = lv[np.maximum(k, 0)]
    return out


# ══════════════════════════════════════════════════════════════════════════
# expiry calendar
# ══════════════════════════════════════════════════════════════════════════
def _d(x: int) -> _dt.date:
    return _dt.date(x // 10000, x // 100 % 100, x % 100)


def _n(x: _dt.date) -> int:
    return x.year * 10000 + x.month * 100 + x.day


def _shift_back(day: _dt.date, sessions: set) -> int:
    while _n(day) not in sessions:
        day -= _dt.timedelta(1)
    return _n(day)


def monthly_expiry(year: int, month: int, sessions: set) -> int:
    """Last Thursday of the month, or the previous trading session if that Thursday is a holiday."""
    nxt = _dt.date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - _dt.timedelta(1)
    while d.weekday() != 3:
        d -= _dt.timedelta(1)
    return _shift_back(d, sessions)


def rule_expiries(sym: str, sessions, lo: int = DD.ERA[0], hi: int = DD.ERA[1]) -> list:
    """Rule-based expiry dates in [lo, hi] for NIFTY / BANKNIFTY: monthlies + weeklies (see the module docstring)."""
    ss = set(int(x) for x in sessions)
    out = set()
    y, m = lo // 10000, lo // 100 % 100
    last = max(ss)
    while y * 100 + m <= hi // 100:
        nxt = _dt.date(y + (m == 12), m % 12 + 1, 1)
        if _n(nxt) - 100 <= last or _n(nxt - _dt.timedelta(1)) <= last:     # month inside the calendar
            e = monthly_expiry(y, m, ss)
            th = nxt - _dt.timedelta(1)
            while th.weekday() != 3:
                th -= _dt.timedelta(1)
            if _n(th) <= last:                                               # never shift past the calendar's end
                out.add(e)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    monthly = sorted(out)
    wk_from = {"NIFTY": NIFTY_WEEKLY_FROM, "BANKNIFTY": BNF_WEEKLY_FROM}.get(sym)
    if wk_from:
        d = _d(wk_from)
        while _n(d) <= hi:
            wd = d
            if sym == "BANKNIFTY" and _n(d) >= BNF_WEDNESDAY_FROM - 1:
                wd = d - _dt.timedelta(1)                     # Wednesday expiries from Sep 2023
            e = _shift_back(wd, ss)
            # no separate weekly in the monthly expiry's own week (the monthly contract is that week's expiry)
            if not any(_d(m).isocalendar()[:2] == _d(e).isocalendar()[:2] for m in monthly):
                out.add(e)
            d += _dt.timedelta(7)
    return sorted(x for x in out if lo <= x <= hi)


def listed_expiries(sym: str, view_dir: str = VIEW_DIR) -> list:
    """Expiries that actually reached their expiry day in the option bhavcopy (a row dated on the expiry itself)."""
    import pandas as pd
    t = os.path.join(view_dir, "_cache", f"opt_{sym}.npz")
    if os.path.exists(t):
        with np.load(t) as z:
            d, e = z["date"], z["expiry"]
        return sorted(int(x) for x in np.unique(e[d == e]))
    out = set()
    for f in sorted(glob.glob(os.path.join(view_dir, "fo_parts", "optidx", "*.csv.gz"))):
        df = pd.read_csv(f, usecols=["date", "symbol", "expiry"])
        df = df[(df.symbol == sym) & (df.date == df.expiry)]
        out.update(_i(x) for x in df.expiry.unique())
    return sorted(out)


# ══════════════════════════════════════════════════════════════════════════
# the view
# ══════════════════════════════════════════════════════════════════════════
COLMAP = {
    "future index long": "fut_idx_long", "future index short": "fut_idx_short",
    "future stock long": "fut_stk_long", "future stock short": "fut_stk_short",
    "option index call long": "opt_idx_call_long", "option index put long": "opt_idx_put_long",
    "option index call short": "opt_idx_call_short", "option index put short": "opt_idx_put_short",
    "option stock call long": "opt_stk_call_long", "option stock put long": "opt_stk_put_long",
    "option stock call short": "opt_stk_call_short", "option stock put short": "opt_stk_put_short",
    "total long contracts": "total_long", "total short contracts": "total_short",
}
CT = {"client": "Client", "dii": "DII", "fii": "FII", "pro": "Pro", "total": "TOTAL"}


def _parse_participant(path: str, key: str, problems: list) -> list:
    txt = open(path, "rb").read().decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [l for l in txt.split("\n") if l.strip()]
    hi = next((i for i, l in enumerate(lines) if l.lower().replace("_", " ").startswith("client type")), None)
    if hi is None:                                       # 2014-02-24: the header row is missing; standard order
        hi = next(i for i, l in enumerate(lines) if l.lower().startswith("client,")) - 1
        hdr = ["client type"] + list(COLMAP)
        problems.append(f"{path}: no header row; standard column order assumed")
    else:
        hdr = [h.strip().lower() for h in next(csv.reader([lines[hi]]))]
    shifted = None
    if len(hdr) > 1 and hdr[1].replace(",", "").isdigit():
        # 2013-08-22: the column names sit on the title row and every label is one row too high ("Client Type" holds
        # Client's numbers ... "Pro" holds TOTAL's): relabel the numeric rows in the standard order
        hi -= 1
        hdr = ["client type"] + list(COLMAP)
        shifted = iter(["client", "dii", "fii", "pro", "total"])
        problems.append(f"{path}: row labels shifted by one in the source file; relabelled")
    out = []
    for row in csv.reader(lines[hi + 1:]):
        if not row or not row[0].strip():
            continue
        if shifted is not None:
            if not any(x.strip() for x in row[1:]):
                continue
            row = [next(shifted)] + row[1:]
        ct = CT.get(row[0].strip().lower())
        if ct is None:
            problems.append(f"{path}: unknown client type {row[0]!r}")
            continue
        rec = {"date": key, "client_type": ct}
        for h, v in zip(hdr[1:], row[1:]):
            if h in COLMAP:
                x = v.strip().replace(",", "")
                try:
                    rec[COLMAP[h]] = int(float(x)) if x else None
                except ValueError:                        # "NA" (old files: DII option writing not allowed)
                    rec[COLMAP[h]] = None
        miss = [c for c in COLMAP.values() if c not in rec]
        if miss:
            problems.append(f"{path}: missing {miss}")
        out.append(rec)
    by = {r["client_type"]: r for r in out}
    if "FII" in by and "DII" in by:
        # 2012-01-02's file labels the FII and DII rows the wrong way round (the "FII" row has no option writing,
        # which only DIIs lacked): swap them back
        if not by["FII"].get("opt_idx_call_short") and by["DII"].get("opt_idx_call_short"):
            by["FII"]["client_type"], by["DII"]["client_type"] = "DII", "FII"
            problems.append(f"{path}: FII / DII labels swapped in the source file; corrected")
    if sorted(r["client_type"] for r in out) != sorted(CT.values()):
        problems.append(f"{path}: client types {[r['client_type'] for r in out]}")
    return out


def _parse_ban(path: str, key: str, problems: list) -> list:
    lines = [l.strip() for l in open(path, "rb").read().decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n").split("\n")
             if l.strip()]
    m = re.search(r"(\d{2})-([A-Za-z]+)-(\d{4})", lines[0])
    if m:
        try:
            hd = _dt.datetime.strptime(f"{m.group(1)}-{m.group(2)[:3].title()}-{m.group(3)}", "%d-%b-%Y").date()
            if _n(hd) != int(key):
                problems.append(f"ban {key}: header date {hd} != file date")
        except ValueError:
            problems.append(f"ban {key}: header {lines[0][:60]!r}")
    out = []
    for l in lines[1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1]:
            out.append({"date": f"{key[:4]}-{key[4:6]}-{key[6:]}", "symbol": parts[1]})
        elif l.strip(",").upper() not in ("NIL", "NONE", ""):
            problems.append(f"ban {key}: odd line {l!r}")
    return out


def build_participants(view_dir: str = VIEW_DIR) -> dict:
    import pandas as pd
    problems: list = []
    recs = []
    d = os.path.join(view_dir, "raw_part", "oi")
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv"):
            k = f[:8]
            recs += _parse_participant(os.path.join(d, f), f"{k[:4]}-{k[4:6]}-{k[6:]}", problems)
    oi = pd.DataFrame(recs, columns=["date", "client_type"] + list(COLMAP.values()))
    oi.to_csv(os.path.join(view_dir, "participant_oi.csv.gz"), index=False, compression="gzip")
    recs = []
    d = os.path.join(view_dir, "raw_part", "ban")
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv"):
            recs += _parse_ban(os.path.join(d, f), f[:8], problems)
    ban = pd.DataFrame(recs, columns=["date", "symbol"])
    ban.to_csv(os.path.join(view_dir, "ban.csv.gz"), index=False, compression="gzip")
    return {"participant_days": int(oi.date.nunique()), "ban_rows": int(len(ban)),
            "ban_files": len([f for f in os.listdir(d) if f.endswith(".csv")]), "problems": problems}


def build_view(old_dir: str = OLD_DIR, view_dir: str = VIEW_DIR, verbose: bool = True) -> dict:
    import pandas as pd
    os.makedirs(view_dir, exist_ok=True)
    rep: dict = {}
    # ── delivery (EQ, unadjusted) with canonical bars_symbol
    dp = os.path.join(view_dir, "delivery.csv.gz")
    srcs = [os.path.join(old_dir, f) for f in ("eq_daily.csv.gz", "symbol_changes.csv", "universe_monthly.csv")]
    fresh = os.path.exists(dp) and all(os.path.getmtime(dp) > os.path.getmtime(x) for x in srcs) and \
        os.path.getmtime(dp) > os.path.getmtime(__file__)
    if not fresh:
        _build_delivery(old_dir, view_dir, rep, verbose)
    _rest_of_view(old_dir, view_dir, rep, verbose)
    return rep


def _build_delivery(old_dir, view_dir, rep, verbose):
    import pandas as pd
    eq = pd.read_csv(os.path.join(old_dir, "eq_daily.csv.gz"), dtype={"symbol": str, "series": str, "isin": str},
                     keep_default_na=False, na_values=[""])
    eq = eq[eq.series == "EQ"].copy()
    ch = pd.read_csv(os.path.join(old_dir, "symbol_changes.csv"), dtype={"old_symbol": str, "new_symbol": str},
                     keep_default_na=False)
    canon = eq[["date", "symbol"]].copy()
    canon, applied = DD.apply_renames(canon, ch)
    eq["bars_symbol"] = canon["symbol"].to_numpy()
    # the delivery universe = every stock that is in the point-in-time universe at least once in the era (nsedata
    # treats bars_symbol rows as "the lab's symbols"; r4_delivery's market breadth ranks over them)
    uni = pd.read_csv(os.path.join(old_dir, "universe_monthly.csv"), dtype={"symbol": str}, keep_default_na=False)
    uni["date"] = uni["effective_from"].astype(int)
    uni, _ = DD.apply_renames(uni, ch)
    ever = set(uni["symbol"])
    eq = eq[eq["bars_symbol"].isin(ever)].copy()
    eq["avg_price"] = np.where(eq.volume.astype(float) > 0, eq.turnover / eq.volume.astype(float), np.nan)
    eq["turnover_lacs"] = eq.turnover / 1e5
    cols = ["date", "symbol", "open", "high", "low", "close", "prev_close", "avg_price", "volume", "turnover_lacs",
            "trades", "deliv_qty", "deliv_pct", "bars_symbol"]
    eq[cols].to_csv(os.path.join(view_dir, "delivery.csv.gz"), index=False, compression="gzip")
    rep["delivery_rows"], rep["renames"] = int(len(eq)), len(applied)
    if verbose:
        print(f"delivery.csv.gz: {len(eq)} rows, {len(applied)} renames", flush=True)


def _rest_of_view(old_dir, view_dir, rep, verbose):
    import pandas as pd
    # ── F&O parts
    src, dst = os.path.join(old_dir, "fo_parts"), os.path.join(view_dir, "fo_parts")
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, "status.csv"), os.path.join(dst, "status.csv"))
    for sub in ("optstk", "optstk_full"):
        lk = os.path.join(dst, sub)
        if not os.path.islink(lk):
            if os.path.exists(lk):
                shutil.rmtree(lk)
            os.symlink(os.path.join(src, sub), lk)
    for sub in ("fut", "optidx"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
        n = 0
        for f in sorted(glob.glob(os.path.join(src, sub, "*.csv.gz"))):
            out = os.path.join(dst, sub, os.path.basename(f))
            if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(f):
                continue
            df = pd.read_csv(f, dtype={"symbol": str, "type": str} if sub == "optidx" else {"symbol": str},
                             keep_default_na=False, na_values=[""])
            if len(df):
                df["lot"] = _lot_col(df, old_dir)
            df.to_csv(out + ".tmp", index=False, compression="gzip")
            os.replace(out + ".tmp", out)
            n += 1
        rep[f"parts_{sub}_rewritten"] = n
        if verbose:
            print(f"fo_parts/{sub}: {n} files rewritten with lots", flush=True)
    # ── participant OI + ban
    rep.update(build_participants(view_dir))
    # ── index spot files for nsedata (BARS_DIR)
    bd = os.path.join(view_dir, "bars")
    os.makedirs(bd, exist_ok=True)
    idx = pd.read_csv(os.path.join(old_dir, "index_daily.csv.gz"))
    for name, fn, t in (("NIFTY", "_NIFTY50_day.json", "00:00:00"), ("INDIAVIX", "_VIX_day.json", "00:00:00"),
                        ("BANKNIFTY", "_NIFTYBANK_min.json", "15:29:00")):
        g = idx[idx["index"] == name].sort_values("date")
        bars = [[f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}T{t}", o, h, l, c, 0]
                for d, o, h, l, c in zip(g.date, g.open, g.high, g.low, g.close)]
        with open(os.path.join(bd, fn), "w") as fh:
            json.dump({"symbol": fn[:-5], "interval": "day" if "day" in fn else "minute", "bars": bars}, fh)
    with open(os.path.join(view_dir, "VIEW_MANIFEST.json"), "w") as fh:
        json.dump({k: v for k, v in rep.items() if k != "problems"} | {"n_problems": len(rep.get("problems", [])),
                                                                         "problems": rep.get("problems", [])[:50]},
                  fh, indent=1)
    if verbose:
        print(f"participant days {rep['participant_days']}, ban files {rep['ban_files']} ({rep['ban_rows']} rows), "
              f"{len(rep['problems'])} parse problems", flush=True)


def use(view_dir: str = VIEW_DIR):
    """Point nsedata (already imported or not) at the old-era view. Returns the nsedata module."""
    os.environ["STRATLAB_NSE_DIR"] = view_dir
    os.environ["STRATLAB_BARS_DIR"] = os.path.join(view_dir, "bars")
    from . import nsedata as N
    N.NSE_DIR = view_dir
    N.BARS_DIR = os.path.join(view_dir, "bars")
    N._S.clear()
    return N


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.oldera_fo")
    ap.add_argument("--build-view", action="store_true")
    ap.add_argument("--old", default=OLD_DIR)
    ap.add_argument("--view", default=VIEW_DIR)
    a = ap.parse_args(argv)
    if a.build_view:
        build_view(a.old, a.view)
        N = use(a.view)
        print(N.coverage())
    return 0


if __name__ == "__main__":
    sys.exit(main())
