"""
stratlab/data_daily.py — DAILY-ONLY lab mode: the lab's positional simulator (sim.sim_positional) and the Swing /
futures-positional strategies on NSE official end-of-day bars, without minute data.

Built for the old-era out-of-sample round (2012-01-01 .. 2023-11-09, see protocol_oos.py), and usable on the
3-year era for the parity test (NSE sec_bhavdata_full in <nse>/delivery.csv.gz).  data.py is NOT changed: this
module produces data.SymbolData objects with ONE bar per session (t = 09:15), whose .daily() returns exactly the
NSE daily open / high / low / close / volume, and daily aux series in data.load_daily_aux's format.

SOURCES (old era, read-only; <scratchpad>/nse_old, env STRATLAB_OLD_DIR)
  eq_daily_adj.csv.gz   NSE cm bhavcopy, EQ + BE series, every symbol incl. later delisted ones, back-adjusted for
                        splits / bonuses / large demergers (prices x adj_factor, volume / adj_factor)
  index_daily.csv.gz    NIFTY, BANKNIFTY, INDIAVIX official OHLC
  universe_monthly.csv  point-in-time universe: each month's top 100 equity shares by average daily turnover over the
                        60 sessions STRICTLY BEFORE the month's first session (built by the fetch agent; checked by
                        tests_oldera.py against eq_daily.csv.gz)
  symbol_changes.csv    NSE renames: rows of OLD dated before the change date are renamed to NEW (chains resolved in
                        date order), so a renamed company keeps one continuous history (MUNDRAPORT -> ADANIPORTS).

WHAT A STRATEGY SEES (identical to the minute-bar lab, sim.sim_positional is reused unchanged)
  d.date d.o d.h d.l d.c d.v   NSE daily bars up to the decision day (EQ preferred over BE on a date): official
                               open / high / low / volume; close = NSE's LAST traded price (CLOSE="last", default:
                               the minute lab's daily close is the 15:29 minute's close, not the official 30-minute
                               VWAP close; set STRATLAB_DAILY_CLOSE=close for the official close)
  d.nifty d.vix                NIFTY 50 / India VIX official closes aligned to the same days (NaN where missing)
  d.symbol                     the canonical (latest old-era) NSE symbol
Index futures (underlying NIFTY / BANKNIFTY) run on the index's official daily bars under the lab's symbol names
"_NIFTY50_min" / "_NIFTYBANK_min" (volume 0, as in the lab's index files).

POINT-IN-TIME UNIVERSE: a stock strategy may OPEN a position on decision day D only if the stock is in the universe
list of D's month (computed from data dated before that month's first session, i.e. known at the decision). Its
history before / outside membership is still visible (it was public), and an open position is managed to its exit
even if the stock leaves the list or is delisted (data_end exit at the last official close). This is done by
`gate()`, which wraps the strategy function; the strategy's code and parameters are untouched.

SESSIONS: special sessions are dropped exactly like the minute lab drops sessions with < 200 bars: a date whose
total EQ turnover is below SPECIAL_FRAC x the median of the previous 20 sessions (Muhurat hours, the 2012-2014
Saturday DR-test sessions). Full-day weekend sessions (Budget 2015-02-28, 2020-02-01) are kept. The dropped dates
are listed in the manifest.

LOOK-AHEAD: arrays are cut at the decision day by the simulator itself. Back-adjustment scales a symbol's whole
history before a split by a constant; returns, ratios and indicator signals on ratios are unchanged by it, and
tests_oldera.py checks that a cache built from data truncated at any date differs from the full one only by such a
per-symbol constant (no information from after the cut can change a ratio before it).

    cd algotrader_v4
    python -m stratlab.data_daily --build                 # old-era cache (idempotent; rebuilt when sources change)
    python -m stratlab.data_daily --info
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from . import data as _data

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
OLD_DIR = os.environ.get("STRATLAB_OLD_DIR", os.path.join(_SCRATCH, "nse_old"))
OLD_CACHE = os.environ.get("STRATLAB_OLD_CACHE", os.path.join(_SCRATCH, "stratlab_cache", "oldera_daily"))
ERA = (20120101, 20231109)
SPECIAL_FRAC = 0.30             # session turnover < 30 % of the local median -> special session, dropped
BAR_T = 9 * 60 + 15             # the single daily "bar" is stamped 09:15
INDEX_NAMES = {"NIFTY": "_NIFTY50_min", "BANKNIFTY": "_NIFTYBANK_min"}
AUX_NAMES = {"NIFTY": "_NIFTY50_day", "INDIAVIX": "_VIX_day", "BANKNIFTY": "_NIFTYBANK_day"}
CACHE_VERSION = "dd3"
# Daily close used by the strategies: "last" = NSE's LAST traded price (what the minute lab's daily bars use: the 15:29
# minute's close), "close" = NSE's official close (30-minute VWAP). Chosen by the parity test (tests_oldera.py).
CLOSE = os.environ.get("STRATLAB_DAILY_CLOSE", "last")


# ══════════════════════════════════════════════════════════════════════════
# building
# ══════════════════════════════════════════════════════════════════════════
def _sig(paths) -> str:
    h = hashlib.sha1(CACHE_VERSION.encode())
    for p in paths:
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def apply_renames(df, changes, until: int | None = None):
    """Rename rows of OLD dated before each change date to NEW, in date order (chains resolve). A change is applied
    only if OLD's rows end before NEW's begin (no overlap = a pure rename, not a reused ticker). `until`: ignore
    changes dated after it (truncation tests). Returns (df, applied list)."""
    applied = []
    for r in changes.sort_values("date").itertuples(index=False):
        if until is not None and int(r.date) > until:
            continue
        old, new, day = str(r.old_symbol), str(r.new_symbol), int(r.date)
        mo = (df["symbol"] == old) & (df["date"] < day)
        if not mo.any():
            continue
        mn = df["symbol"] == new
        if mn.any() and int(df.loc[mo, "date"].max()) >= int(df.loc[mn, "date"].min()):
            continue
        df.loc[mo, "symbol"] = new
        applied.append((day, old, new))
    return df, applied


def special_sessions(df) -> list:
    """Dates whose total EQ turnover is < SPECIAL_FRAC x the median of the PREVIOUS 20 sessions (causal: a
    session's status never depends on later data). Use the whole market (not a symbol subset) as `df`."""
    t = df[df["series"] == "EQ"].groupby("date")["turnover"].sum().sort_index()
    med = t.shift(1).rolling(20, min_periods=5).median()
    return [int(x) for x in t.index[(t < SPECIAL_FRAC * med).to_numpy()]]


def _rows_to_arrays(g) -> dict:
    c = g["close"].to_numpy(float)
    if "last" in g:
        lt = g["last"].to_numpy(float)
        c = np.where(np.isfinite(lt) & (lt > 0), lt, c)
    return {"date": g["date"].to_numpy(np.int32), "o": g["open"].to_numpy(float), "h": g["high"].to_numpy(float),
            "l": g["low"].to_numpy(float), "c": c,
            "v": g["volume"].astype(float).fillna(0.0).to_numpy(float)}


def build_cache(eq, idx, universe, changes, cache_dir: str, until: int | None = None, verbose: bool = False,
                lo: int = ERA[0], special: list | None = None) -> dict:
    """Write one .npz per symbol (date o h l c v + member flag per session), the index / VIX series and a manifest.
    eq: DataFrame with date symbol series open high low close volume turnover (adjusted prices); idx: DataFrame with
    date index open high low close; universe: DataFrame month effective_from symbol (or None = every symbol is a
    member); changes: symbol_changes DataFrame (or None). `until`: use only rows dated <= until (tests)."""
    import pandas as pd
    _data.assert_outside_repo(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    for f in os.listdir(cache_dir):
        if f.endswith(".npz"):
            os.remove(os.path.join(cache_dir, f))
    eq = eq[eq["date"] >= lo]
    idx = idx[idx["date"] >= lo]
    if until is not None:
        eq = eq[eq["date"] <= until]
        idx = idx[idx["date"] <= until]
    eq = eq[eq["series"].isin(["EQ", "BE"])].copy()
    applied = []
    if changes is not None:
        eq, applied = apply_renames(eq, changes, until)
    special = special_sessions(eq) if special is None else [int(x) for x in special]
    eq = eq[~eq["date"].isin(special)]
    idx = idx[~idx["date"].isin(special)]
    # EQ preferred over BE when both exist on a date
    eq["_pri"] = (eq["series"] != "EQ").astype(int)
    eq = eq.sort_values(["symbol", "date", "_pri"]).drop_duplicates(["symbol", "date"], keep="first")
    ok = (eq[["open", "high", "low", "close"]] > 0).all(axis=1) & np.isfinite(eq[["open", "high", "low", "close"]]).all(axis=1)
    eq = eq[ok]
    # membership: (canonical symbol, month) pairs
    mem = None
    if universe is not None:
        u = universe.copy()
        u["date"] = u["effective_from"].astype(int)
        u["series"] = "EQ"
        if changes is not None:
            # a universe row names the symbol as of its month: rename it the same way as the price rows
            u, _ = apply_renames(u, changes, until)
        mem = set(zip(u["symbol"].astype(str), u["month"].astype(int)))
    calendar = sorted(set(int(x) for x in idx.loc[idx["index"] == "NIFTY", "date"]))
    syms = {}
    for sym, g in eq.groupby("symbol", sort=True):
        a = _rows_to_arrays(g)
        months = a["date"] // 100
        a["member"] = (np.array([(sym, int(m)) in mem for m in months], bool) if mem is not None
                       else np.ones(len(months), bool))
        fn = _data._cache_name(sym)
        np.savez(os.path.join(cache_dir, fn), **a)
        syms[sym] = {"file": fn, "n": int(len(g)), "first": int(a["date"][0]), "last": int(a["date"][-1]),
                     "member_days": int(a["member"].sum())}
    aux = {}
    for name, g in idx.groupby("index"):
        g = g.sort_values("date")
        a = {"date": g["date"].to_numpy(np.int32), "o": g["open"].to_numpy(float), "h": g["high"].to_numpy(float),
             "l": g["low"].to_numpy(float), "c": g["close"].to_numpy(float), "v": np.zeros(len(g))}
        if name in AUX_NAMES:
            np.savez(os.path.join(cache_dir, _data._cache_name(AUX_NAMES[name])), **a)
            aux[AUX_NAMES[name]] = int(len(g))
        if name in INDEX_NAMES:
            a = dict(a)
            a["member"] = np.ones(len(g), bool)
            np.savez(os.path.join(cache_dir, _data._cache_name(INDEX_NAMES[name])), **a)
            aux[INDEX_NAMES[name]] = int(len(g))
    man = {"version": CACHE_VERSION, "symbols": syms, "aux": aux, "calendar": calendar, "special_dropped": special,
           "renames_applied": applied, "until": until}
    with open(os.path.join(cache_dir, "_manifest.json"), "w") as fh:
        json.dump(man, fh)
    if verbose:
        print(f"daily cache {cache_dir}: {len(syms)} symbols, {len(calendar)} sessions, "
              f"{len(special)} special sessions dropped, {len(applied)} renames applied", flush=True)
    return man


def _read_old(old_dir: str):
    import pandas as pd
    eq = pd.read_csv(os.path.join(old_dir, "eq_daily_adj.csv.gz"),
                     usecols=["date", "symbol", "series", "open", "high", "low", "close", "volume", "turnover",
                              "adj_factor"],
                     dtype={"symbol": str, "series": str}, keep_default_na=False, na_values=[""])
    if CLOSE == "last":
        lt = old_last(old_dir)
        eq = eq.merge(lt, on=["date", "symbol", "series"], how="left")
        eq["last"] = eq["last"] * eq["adj_factor"]
    idx = pd.read_csv(os.path.join(old_dir, "index_daily.csv.gz"), usecols=["date", "index", "open", "high", "low",
                                                                            "close"])
    uni = pd.read_csv(os.path.join(old_dir, "universe_monthly.csv"), dtype={"symbol": str}, keep_default_na=False)
    ch = pd.read_csv(os.path.join(old_dir, "symbol_changes.csv"), dtype={"old_symbol": str, "new_symbol": str},
                     keep_default_na=False)
    return eq, idx, uni, ch


def old_last(old_dir: str = OLD_DIR):
    """LAST traded price per (date, symbol, series) from the per-day cm bhavcopy parts (unadjusted), cached."""
    import glob
    import pandas as pd
    out = os.path.join(os.path.dirname(OLD_CACHE), "oldera_last.pkl")
    files = sorted(glob.glob(os.path.join(old_dir, "parts", "cm", "*.csv.gz")))
    sig = f"{len(files)}:{files[-1] if files else ''}"
    if os.path.exists(out):
        df = pd.read_pickle(out)
        if df.attrs.get("sig") == sig:
            return df
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=["symbol", "series", "last"], dtype={"symbol": str, "series": str},
                        keep_default_na=False, na_values=[""])
        d = d[d.series.isin(["EQ", "BE"])]
        d.insert(0, "date", int(os.path.basename(f)[:8]))
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df.attrs["sig"] = sig
    df.to_pickle(out)
    return df


def ensure_old_cache(old_dir: str = OLD_DIR, cache_dir: str = OLD_CACHE, verbose: bool = True) -> str:
    """Build the old-era daily cache if missing or stale. Returns the cache dir."""
    srcs = [os.path.join(old_dir, f) for f in ("eq_daily_adj.csv.gz", "index_daily.csv.gz", "universe_monthly.csv",
                                                "symbol_changes.csv")]
    import fcntl
    sig = _sig(srcs) + CLOSE
    sp = os.path.join(cache_dir, "_signature.txt")
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)                   # one builder at a time; readers wait for it
        if os.path.exists(sp) and open(sp).read().strip() == sig and \
                os.path.exists(os.path.join(cache_dir, "_manifest.json")):
            return cache_dir
        if os.path.exists(sp):
            os.remove(sp)
        eq, idx, uni, ch = _read_old(old_dir)
        build_cache(eq, idx, uni, ch, cache_dir, verbose=verbose)
        with open(sp, "w") as fh:
            fh.write(sig)
    return cache_dir


def build_from_delivery(delivery_csv: str, bars_dir: str, cache_dir: str, symbols=None, verbose: bool = False,
                        ref_cache: str | None = None) -> dict:
    """3-year-era daily cache from NSE sec_bhavdata_full (<nse>/delivery.csv.gz, EQ, official, UNADJUSTED) for the
    lab's bars3y symbols (bars_symbol column), every symbol a member every day, index / VIX from the bars directory's
    daily series (NIFTY day file; NIFTY BANK last minute close; India VIX day file). For the parity test only."""
    import pandas as pd
    dv = pd.read_csv(delivery_csv, usecols=["date", "symbol", "open", "high", "low", "close", "volume", "turnover_lacs",
                                            "bars_symbol"], dtype={"symbol": str, "bars_symbol": str})
    dv = dv[dv.bars_symbol.notna()].copy()
    if CLOSE == "last":
        import glob
        raw = sorted(glob.glob(os.path.join(os.path.dirname(delivery_csv), "raw_cm", "sec_bhavdata_full_*.csv.gz")))
        lt = []
        want = set(dv.symbol)
        for f in raw:
            x = pd.read_csv(f, skipinitialspace=True, usecols=["SYMBOL", "SERIES", "LAST_PRICE"], dtype=str)
            x = x[(x.SERIES.str.strip() == "EQ") & x.SYMBOL.isin(want)]
            day = int(os.path.basename(f)[-15:-7])
            lt.append(pd.DataFrame({"date": day, "symbol": x.SYMBOL.str.strip(),
                                    "last": pd.to_numeric(x.LAST_PRICE, errors="coerce")}))
        dv = dv.merge(pd.concat(lt, ignore_index=True), on=["date", "symbol"], how="left")
    if symbols is not None:
        dv = dv[dv.bars_symbol.isin(set(symbols))]
    dv["symbol"] = dv["bars_symbol"]
    dv["series"] = "EQ"
    dv["turnover"] = dv["turnover_lacs"] * 1e5
    rows = []
    for name, fn in (("NIFTY", "_NIFTY50_day.json"), ("INDIAVIX", "_VIX_day.json")):
        b = json.load(open(os.path.join(bars_dir, fn)))["bars"]
        for x in b:
            rows.append((int(x[0][:10].replace("-", "")), name, x[1], x[2], x[3], x[4]))
    b = json.load(open(os.path.join(bars_dir, "_NIFTYBANK_min.json")))["bars"]
    day: dict = {}
    for x in b:
        k = int(x[0][:10].replace("-", ""))
        if k not in day:
            day[k] = [x[1], x[2], x[3], x[4]]
        else:
            day[k][1] = max(day[k][1], x[2]); day[k][2] = min(day[k][2], x[3]); day[k][3] = x[4]
    for k, v in day.items():
        rows.append((k, "BANKNIFTY", *v))
    idx = pd.DataFrame(rows, columns=["date", "index", "open", "high", "low", "close"])
    dv = dv[dv.date >= int(idx.date.min())]
    if ref_cache is not None:
        dv, idx = _align_to_lab(dv, idx, ref_cache)
    return build_cache(dv, idx, None, None, cache_dir, verbose=verbose, lo=0, special=[])


def _align_to_lab(dv, idx, ref_cache: str):
    """Parity only: put the NSE rows on the minute lab's calendar and adjustment basis. bars3y (Kite) is
    back-adjusted for corporate actions (RELIANCE 1:1 bonus 2024-10, ...), sec_bhavdata_full is not: the NSE prices of
    each symbol are scaled by the lab/NSE close ratio, taken as a piecewise constant (median between jumps > 1 %), and
    rows on sessions the minute lab does not have (dropped short sessions, days before its first session) removed."""
    import pandas as pd
    cal = None
    out = []
    for sym, g in dv.groupby("symbol"):
        sd = _data.load_symbol(ref_cache, sym)
        if sd is None:
            continue
        ref = sd.daily()
        cal = set(int(x) for x in ref["date"]) if cal is None else cal | set(int(x) for x in ref["date"])
        g = g[g.date.isin(set(int(x) for x in ref["date"]))].sort_values("date").copy()
        if g.empty:
            continue
        rc = pd.Series(ref["c"], index=ref["date"]).reindex(g.date).to_numpy(float)
        nc = g["last"].to_numpy(float) if "last" in g else g["close"].to_numpy(float)
        r = rc / nc
        cut = np.r_[0, np.flatnonzero(np.abs(r[1:] / r[:-1] - 1) > 0.01) + 1, len(r)]
        f = np.empty(len(r))
        for a, b in zip(cut[:-1], cut[1:]):
            f[a:b] = np.nanmedian(r[a:b])
        f = np.where(np.abs(f - 1) < 0.005, 1.0, f)              # unadjusted stretches stay exact
        for k in ("open", "high", "low", "close") + (("last",) if "last" in g else ()):
            g[k] = g[k].to_numpy(float) * f
        g["volume"] = g["volume"].to_numpy(float) / f
        out.append(g)
    idx = idx[idx.date.isin(cal)] if cal else idx
    return pd.concat(out, ignore_index=True), idx


# ══════════════════════════════════════════════════════════════════════════
# loading
# ══════════════════════════════════════════════════════════════════════════
def manifest(cache_dir: str = OLD_CACHE) -> dict:
    with open(os.path.join(cache_dir, "_manifest.json")) as fh:
        return json.load(fh)


def _npz(cache_dir: str, name: str) -> dict | None:
    p = os.path.join(cache_dir, _data._cache_name(name))
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def load_symbol(cache_dir: str, symbol: str) -> _data.SymbolData | None:
    """One bar per session: SymbolData whose .daily() is the official daily bars. .member (bool per session) is
    attached for gate()."""
    z = _npz(cache_dir, symbol)
    if z is None or not len(z["date"]):
        return None
    n = len(z["date"])
    sd = _data.from_arrays(symbol, z["date"], np.full(n, BAR_T, np.int16), z["o"], z["h"], z["l"], z["c"], z["v"],
                           min_bars=1)
    sd.member = z["member"].astype(bool)
    return sd


def load_daily_aux(cache_dir: str, name: str) -> dict | None:
    """'_NIFTY50_day' / '_VIX_day' / '_NIFTYBANK_day' -> {"date","o","h","l","c","v"} (data.load_daily_aux format)."""
    z = _npz(cache_dir, name)
    if z is None:
        return None
    z.pop("member", None)
    return z


def symbols(cache_dir: str = OLD_CACHE, members_only: bool = True) -> list:
    """Stock symbols in the cache (only those that are a universe member at least once, by default)."""
    m = manifest(cache_dir)["symbols"]
    return sorted(s for s, x in m.items() if x["member_days"] > 0 or not members_only)


def calendar(cache_dir: str = OLD_CACHE) -> list:
    return list(manifest(cache_dir)["calendar"])


# ══════════════════════════════════════════════════════════════════════════
# point-in-time universe gate
# ══════════════════════════════════════════════════════════════════════════
class _Gated:
    """fn(d, p) wrapper: returns None (no new position) unless the symbol is a universe member on the decision day
    d.date[-1]. Membership comes from the cache's per-session flag (month list computed before that month)."""

    def __init__(self, fn, member_days: set):
        self.fn, self.member_days = fn, member_days
        self.__name__ = getattr(fn, "__name__", "gated")

    def __call__(self, d, p):
        if int(d.date[-1]) not in self.member_days:
            return None
        return self.fn(d, p)


def gate(spec, sd):
    """A copy of `spec` whose fn only opens positions on member days of `sd` (index specs: unchanged)."""
    import dataclasses
    if getattr(spec, "underlying", None) or getattr(sd, "member", None) is None or sd.member.all():
        return spec
    days = set(int(x) for x in sd.days[sd.member])
    return dataclasses.replace(spec, fn=_Gated(spec.fn, days))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.data_daily")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--old", default=OLD_DIR)
    ap.add_argument("--cache", default=OLD_CACHE)
    a = ap.parse_args(argv)
    if a.build:
        ensure_old_cache(a.old, a.cache)
    if a.info or a.build:
        m = manifest(a.cache)
        cal = m["calendar"]
        print(f"{len(m['symbols'])} symbols ({len(symbols(a.cache))} ever in the universe), {len(cal)} sessions "
              f"{cal[0]}..{cal[-1]}, aux {sorted(m['aux'])}")
        print(f"special sessions dropped: {m['special_dropped']}")
        print(f"renames applied: {len(m['renames_applied'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
