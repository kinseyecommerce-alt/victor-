"""
stratlab/events_daily.py — ONE continuous daily dataset 2012-01-02 .. 2026-11-09 for the event study (events_run.py),
with a POINT-IN-TIME universe in both eras. Research only; writes outside the repository.

Why not the lab's two existing datasets: the 3-year minute data holds today's 97 NIFTY-100 names (survivorship-
biased, and promoters of the largest companies rarely buy in the market), and the old-era cache gates on the top
100 by turnover. Insider buying and post-earnings drift are published on broad universes (NIFTY 500), so this module
builds the same kind of cache as data_daily.py (same file format, same loader, sim.sim_positional runs on it
unchanged) with:

  prices    2012-01-02 .. 2023-11-09: <nse_old>/eq_daily_adj.csv.gz (data_daily._read_old, unchanged)
            2023-11-10 .. 2026-11-09: NSE sec_bhavdata_full (<nse>/raw_cm), official open / high / low / volume,
            close = LAST traded price (as in data_daily), EQ preferred over BE
  renames   NSE symbolchange.csv, every rename followed to the CURRENT symbol (so the event tables, which NSE serves
            under current symbols, join by symbol)
  adjust    corporate actions with ex-date after 2023-11-09 from NSE's corporate-actions feed: bonus a:b -> b/(a+b),
            face-value split X -> Y -> Y/X, consolidation; each factor must agree with the ex-day gap
            (open / previous close) or it is rejected; a large gap (< 0.85) on a demerger / scheme / rights ex-date is
            adjusted by the observed gap (same rules as the old-era build, <nse_old>/scripts/adjust_old.py).
            The factor multiplies every EARLIER row (old era included): ratios and returns are unchanged by it.
            Unexplained overnight gaps (open / previous close < 0.6 or > 1.67 with no applied action) are listed in
            the manifest (`bad_gaps`); events_run drops every trade whose holding window contains one.
  universe  each month, the top `TOP_N` (500) equity shares (series EQ; ETFs / fund units excluded) by average daily
            turnover over the 60 sessions strictly before the month's first session (non-trading days count 0), traded
            on >= 75 % of those sessions and on the last one — the rule of <nse_old>/universe_monthly.csv with 100 ->
            500. Known at the decision: a position may be OPENED only on a member day (data_daily.gate).
  index     NIFTY 50 / India VIX official daily closes: <nse_old>/index_daily.csv.gz, then the lab's day files.
  sessions  special sessions (Muhurat, DR drills) dropped with data_daily's causal turnover rule.

    cd algotrader_v4 && python -m stratlab.events_daily --build [--top 500]
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import re
import sys

import numpy as np

from . import data_daily as DD

SCRATCH = DD._SCRATCH
NSE_DIR = os.environ.get("STRATLAB_NSE_DIR", os.path.join(SCRATCH, "nse"))
EVENTS_DIR = os.environ.get("STRATLAB_EVENTS_DIR", os.path.join(SCRATCH, "events"))
BARS3Y = os.environ.get("STRATLAB_BARS3Y", os.path.join(SCRATCH, "bars3y"))
CACHE = os.environ.get("STRATLAB_EVENTS_CACHE", os.path.join(SCRATCH, "stratlab_cache", "events_daily"))
TOP_N = 500
OLD_END = DD.ERA[1]                 # 20231109
NEW_END = 20261109
VERSION = "ed1"

NUM = r"(\d+(?:\.\d+)?)"
CUR = r"(?:(?:RS|RE|R)\.?\s*)?"
SPLIT_KEY = r"(?:FACE\s*VALUE\s*SPLIT|FV\s*S\w*|(?:STOCK\s*)?SPLIT|SPLT|SUB[- ]?DIV\w*)"


def parse_action(subject: str):
    """NSE corporate-action subject -> ([(kind, factor)], kind_of_other). factor < 1 for bonus / split."""
    s = (subject or "").upper().replace("/-", " ")
    out = []
    for m in re.finditer(r"BON(?:US)?\s*(?:ISSUE)?\s*" + NUM + r"\s*:\s*" + NUM, s):
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and b > 0:
            out.append(("bonus", b / (a + b)))
    for m in re.finditer(SPLIT_KEY + r"\D{0,60}?" + CUR + NUM + r"\D{0,30}?(?:TO|-)\s*" + CUR + NUM, s):
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and 0 < b < a:
            out.append(("split", b / a))
    for m in re.finditer(r"CONSOL\w*\D{0,60}?" + CUR + NUM + r"\D{0,30}?(?:TO|-)\s*" + CUR + NUM, s):
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and b > a:
            out.append(("consolidation", b / a))
    other = None
    if "RIGHT" in s:
        other = "rights"
    if any(k in s for k in ("DEMERG", "ARRANG", "AMALG", "CAPITAL REDUC", "REDUCTION")):
        other = "scheme"
    return out, other


def decide_factor(parsed: list, other, gap_open: float, gap_close: float):
    """(factor | None, status) for one (symbol, ex-session) from the parsed actions and the observed ex-day gap."""
    if not (gap_open > 0 and gap_close > 0):
        return None, "no_price"
    raw = min(abs(math.log(gap_open)), abs(math.log(gap_close)))
    if parsed:
        kinds = {}
        for k, f in parsed:
            kinds[k] = f                                   # one per kind (the same action announced twice)
        f = float(np.prod(list(kinds.values())))
        err = min(abs(math.log(gap_open / f)), abs(math.log(gap_close / f)))
        ok = (err < math.log(1.30) and err < raw) if abs(math.log(f)) >= math.log(1.3) else err < math.log(1.20)
        return (f, "applied") if ok else (None, "rejected_price_check")
    if other is not None:
        g = gap_close if abs(math.log(gap_close / gap_open)) > math.log(1.5) else gap_open
        if g < 0.85:
            return float(g), "applied_gap_" + other
        return None, "not_applied_" + other
    return None, "no_share_action"


def _d8(s: str) -> int | None:
    from .events_parse import parse_date
    d = parse_date(s)
    return None if d is None else int(d.strftime("%Y%m%d"))


def load_actions(raw_dir: str) -> list:
    """[(symbol, ex yyyymmdd, subject)] from the corporate-actions feed, series EQ / BE."""
    out = []
    for fn in sorted(glob.glob(os.path.join(raw_dir, "ca", "*.json.gz"))):
        try:
            with gzip.open(fn, "rb") as f:
                rows = json.loads(f.read())
        except (OSError, ValueError):
            continue
        for r in rows if isinstance(rows, list) else rows.get("data", []):
            if not isinstance(r, dict) or str(r.get("series", "")).strip() not in ("EQ", "BE", ""):
                continue
            ex = _d8(str(r.get("exDate") or ""))
            if ex is None:
                continue
            out.append((str(r.get("symbol", "")).strip().upper(), ex, " ".join(str(r.get("subject") or "").split())))
    return sorted(set(out))


def read_new(raw_cm: str, lo: int = OLD_END, hi: int = NEW_END):
    """sec_bhavdata_full files dated (lo, hi] -> DataFrame date symbol series open high low close last prev_close
    volume turnover (Rs)."""
    import pandas as pd
    parts = []
    for f in sorted(glob.glob(os.path.join(raw_cm, "sec_bhavdata_full_*.csv.gz"))):
        day = int(os.path.basename(f)[-15:-7])
        if not (lo < day <= hi):
            continue
        x = pd.read_csv(f, skipinitialspace=True, dtype=str, keep_default_na=False)
        x.columns = [c.strip() for c in x.columns]
        # NSE serves the PREVIOUS session's file on holidays / Sundays: trust the date inside the file
        inside = {_d8(v.strip()) for v in x["DATE1"].unique()}
        if inside != {day}:
            continue
        x = x[x["SERIES"].str.strip().isin(["EQ", "BE"])]
        g = pd.DataFrame({"date": day, "symbol": x["SYMBOL"].str.strip(), "series": x["SERIES"].str.strip()})
        for a, b in (("open", "OPEN_PRICE"), ("high", "HIGH_PRICE"), ("low", "LOW_PRICE"), ("close", "CLOSE_PRICE"),
                     ("last", "LAST_PRICE"), ("prev_close", "PREV_CLOSE"), ("volume", "TTL_TRD_QNTY"),
                     ("turnover", "TURNOVER_LACS")):
            g[a] = pd.to_numeric(x[b].str.strip(), errors="coerce").to_numpy()
        g["turnover"] *= 1e5
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def apply_renames_fast(df, changes: list):
    """data_daily.apply_renames (same rule, same result; checked by tests_events) without a full-frame scan per
    rename: rows of OLD dated before the change date become NEW when OLD's rows end before NEW's begin.
    changes: [(date, old, new)] in date order. Returns (df, applied)."""
    sym = df["symbol"].to_numpy(dtype=object).copy()
    date = df["date"].to_numpy()
    rows: dict = {}
    order = np.argsort(sym.astype(str), kind="stable")
    ss = sym[order].astype(str)
    cut = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])
    for a, b in zip(cut[:-1], cut[1:]):
        rows[ss[a]] = order[a:b]
    applied = []
    for day, old, new in sorted(changes):
        io = rows.get(old)
        if io is None or len(io) == 0:
            continue
        m = date[io] < day
        if not m.any():
            continue
        inew = rows.get(new)
        if inew is not None and len(inew) and int(date[io[m]].max()) >= int(date[inew].min()):
            continue
        sym[io[m]] = new
        rows[new] = io[m] if inew is None or len(inew) == 0 else np.concatenate([inew, io[m]])
        rows[old] = io[~m]
        applied.append((int(day), old, new))
    df = df.copy()
    df["symbol"] = sym
    return df, applied


def monthly_universe(eq, equity_ok, top: int = TOP_N):
    """The turnover universe (see the module docstring). eq: date symbol series turnover (canonical symbols)."""
    import pandas as pd
    e = eq[(eq["series"] == "EQ") & eq["symbol"].isin(equity_ok)][["date", "symbol", "turnover"]]
    e = e.drop_duplicates(["date", "symbol"])
    piv = e.pivot(index="date", columns="symbol", values="turnover").sort_index()
    dates = piv.index.values
    rows = []
    for mth in pd.Series(dates // 100).drop_duplicates().values:
        first = dates[dates // 100 == mth][0]
        hist = piv[piv.index < first].tail(60)
        if len(hist) < 20:
            continue
        avg = hist.fillna(0).mean()
        nd = hist.notna().sum()
        ok = avg[(nd >= 0.75 * len(hist)) & (piv.loc[hist.index[-1]].notna())]
        for r, (s, v) in enumerate(ok.sort_values(ascending=False).head(top).items(), 1):
            rows.append([int(mth), int(first), r, s, round(v / 1e7, 3)])
    return pd.DataFrame(rows, columns=["month", "effective_from", "rank", "symbol", "avg_turnover_cr"])


def new_factors(eq, actions: list, lo: int = OLD_END):
    """Corporate-action decisions for ex-dates after `lo`. eq: canonical, sorted by symbol/date, unadjusted in the
    new era. Returns (applied [(symbol, first_session, factor, status, subject)], log rows)."""
    by = {s: (g["date"].to_numpy(), g["open"].to_numpy(float), g["close"].to_numpy(float))
          for s, g in eq[["symbol", "date", "open", "close"]].groupby("symbol", sort=False)}
    grouped: dict = {}
    for sym, ex, subj in actions:
        if ex <= lo:
            continue
        t = by.get(sym)
        if t is None:
            continue
        d, o, c = t
        i = int(np.searchsorted(d, ex))
        if i == 0 or i >= len(d):
            continue
        grouped.setdefault((sym, int(d[i])), {"i": i, "subj": []})["subj"].append(subj)
    applied, log = [], []
    for (sym, fs), g in sorted(grouped.items()):
        d, o, c = by[sym]
        i = g["i"]
        parsed, other = [], None
        for s in g["subj"]:
            p, ot = parse_action(s)
            parsed += p
            other = other or ot
        if not parsed and other is None:
            continue
        f, status = decide_factor(parsed, other, o[i] / c[i - 1], c[i] / c[i - 1])
        log.append({"symbol": sym, "first_session": fs, "factor": f, "status": status,
                    "gap_open": round(float(o[i] / c[i - 1]), 4), "subject": " | ".join(sorted(set(g["subj"])))[:200]})
        if f is not None:
            applied.append((sym, fs, float(f), status, log[-1]["subject"]))
    return applied, log


def build(top: int = TOP_N, cache_dir: str = CACHE, verbose: bool = True) -> dict:
    import pandas as pd
    from .events_parse import load_changes, canon
    raw = os.path.join(EVENTS_DIR, "raw")
    eq_old, idx_old, _uni, _ch = DD._read_old(DD.OLD_DIR)
    eq_old = eq_old[["date", "symbol", "series", "open", "high", "low", "close", "volume", "turnover", "last"]]
    eq_new = read_new(os.path.join(NSE_DIR, "raw_cm"))
    eq = pd.concat([eq_old, eq_new.drop(columns=["prev_close"])], ignore_index=True)
    del eq_old
    # renames: every NSE rename since 2012 followed to the current symbol
    chl = [(d, o, n) for d, o, n in load_changes(os.path.join(raw, "ref", "symbolchange.csv")) if d >= 20120101]
    eq, renamed = apply_renames_fast(eq, chl)
    eq["volume"] = eq["volume"].astype(float)
    if verbose:
        print(f"events_daily: {len(eq):,} rows, {len(renamed)} renames followed", flush=True)
    eq = eq.sort_values(["symbol", "date"]).reset_index(drop=True)
    eq["_pri"] = (eq["series"] != "EQ").astype(int)
    eq = eq.sort_values(["symbol", "date", "_pri"]).drop_duplicates(["symbol", "date"], keep="first")
    eq = eq.drop(columns="_pri").reset_index(drop=True)
    special = DD.special_sessions(eq)

    # corporate actions after the old era
    applied, calog = new_factors(eq, load_actions(raw))
    date_arr = eq["date"].to_numpy()
    pos = eq.groupby("symbol", sort=False).indices
    cols = {c: eq[c].to_numpy(float).copy() for c in ("open", "high", "low", "close", "last", "volume")}
    for sym, fs, f, _st, _sb in applied:
        ii = pos[sym]
        ii = ii[date_arr[ii] < fs]
        for c in ("open", "high", "low", "close", "last"):
            cols[c][ii] *= f
        cols["volume"][ii] /= f
    for c, v in cols.items():
        eq[c] = v

    # equity shares only: an INE ISIN in the old era, or on NSE's current equity list; never an ETF
    isin = pd.read_csv(os.path.join(DD.OLD_DIR, "eq_daily.csv.gz"), usecols=["symbol", "isin", "date"], dtype=str)
    isin = isin[isin["isin"].str.startswith("INE", na=False)]
    last_seen = isin.groupby("symbol")["date"].max().astype(int)
    equity_ok = {canon(s, int(d), chl) for s, d in last_seen.items()}
    with open(os.path.join(raw, "ref", "EQUITY_L.csv"), newline="") as f:
        equity_ok |= {r["SYMBOL"].strip().upper() for r in csv.DictReader(f)}
    with open(os.path.join(raw, "ref", "eq_etfseclist.csv"), newline="", encoding="latin-1") as f:
        etf = {r[0].strip().upper() for r in csv.reader(f) if r}
    equity_ok -= etf
    uni = monthly_universe(eq, equity_ok, top)
    members = set(uni["symbol"])

    # unexplained overnight gaps among universe members (after adjustment)
    same = eq["symbol"].eq(eq["symbol"].shift())
    g = (eq["open"] / eq["close"].shift()).where(same)
    big = eq[((g < 0.6) | (g > 1.67)) & eq["symbol"].isin(members)]
    bad_gaps = sorted((str(s), int(d)) for s, d in zip(big["symbol"], big["date"]))

    # ex-days whose adjustment is a judgement (demerger / scheme / large rights), both eras: trades across them
    # are dropped by events_run together with the unexplained gaps
    amb = {(r["symbol"], int(r["first_session"])) for r in calog
           if r["status"] in ("applied_gap_scheme", "applied_gap_rights", "not_applied_scheme")}
    af = pd.read_csv(os.path.join(DD.OLD_DIR, "adj_factors.csv"), dtype=str, keep_default_na=False)
    af = af[af["status"].isin(["applied_gap_scheme", "applied_gap_rights", "not_applied_scheme"])
            & af["first_session"].str.fullmatch(r"\d{8}")]
    for sym, fs in zip(af["symbol"], af["first_session"]):
        amb.add((canon(sym, int(fs), chl), int(fs)))
    ca_ambiguous = sorted((s_, d_) for s_, d_ in amb if s_ in members)

    # index: NIFTY / VIX
    rows = []
    for name, fn in (("NIFTY", "_NIFTY50_day.json"), ("INDIAVIX", "_VIX_day.json")):
        with open(os.path.join(BARS3Y, fn)) as f:
            for x in json.load(f)["bars"]:
                day = int(x[0][:10].replace("-", ""))
                if OLD_END < day <= NEW_END:
                    rows.append((day, name, x[1], x[2], x[3], x[4]))
    idx = pd.concat([idx_old[idx_old["index"].isin(["NIFTY", "INDIAVIX"])],
                     pd.DataFrame(rows, columns=["date", "index", "open", "high", "low", "close"])], ignore_index=True)
    idx = idx[idx["date"].isin(set(eq["date"]))]
    eq = eq[eq["date"].isin(set(idx.loc[idx["index"] == "NIFTY", "date"]))]     # stock rows only on index sessions

    man = DD.build_cache(eq[eq["symbol"].isin(members)], idx, uni, None, cache_dir, verbose=verbose, special=special)
    man.update({"events_daily": VERSION, "top_n": top, "bad_gaps": bad_gaps, "ca_ambiguous": [list(x) for x in ca_ambiguous],
                "renames_followed": len(renamed),
                "ca_applied": [list(a) for a in applied], "ca_log": calog,
                "universe_months": int(uni["month"].nunique()), "universe_symbols": len(members),
                "era_new": [OLD_END + 1, NEW_END]})
    with open(os.path.join(cache_dir, "_manifest.json"), "w") as fh:
        json.dump(man, fh)
    uni.to_csv(os.path.join(cache_dir, "universe_monthly.csv"), index=False)
    if verbose:
        st = {}
        for r in calog:
            st[r["status"]] = st.get(r["status"], 0) + 1
        print(f"events_daily: top {top}, {len(members)} symbols ever members, {len(bad_gaps)} unexplained gaps, "
              f"corporate actions 2023-26: {st}", flush=True)
    return man


def manifest(cache_dir: str = CACHE) -> dict:
    return DD.manifest(cache_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args(argv)
    if a.build:
        build(a.top, a.cache)
    m = manifest(a.cache)
    print(f"{a.cache}: {len(m.get('symbols', {}))} symbols, {len(m.get('calendar', []))} sessions "
          f"{m['calendar'][0] if m.get('calendar') else ''}..{m['calendar'][-1] if m.get('calendar') else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
