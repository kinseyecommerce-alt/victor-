"""
stratlab/nsedata.py — NSE end-of-day archive data for lab strategies, look-ahead-safe by construction, plus a
real-option-price simulator for multi-day option strategies judged by the unchanged protocol.

DATA (normalised files in NSE_DIR, default <scratchpad>/nse, override with env STRATLAB_NSE_DIR)
  delivery.csv.gz        sec_bhavdata_full, EQ series: delivery qty / %, volume, trades, official OHLC (UNADJUSTED)
  fo_parts/{fut,optidx,optstk,optstk_full}/YYYYMMDD.csv.gz  F&O bhavcopy per session (fetch_fo.py):
                         all futures; every NIFTY / BANKNIFTY option strike; per-(stock, expiry) option aggregates;
                         per-strike stock options of the two nearest expiries with OI > 0 or volume > 0
  participant_oi.csv.gz  participant-wise OI (Client / DII / FII / Pro / TOTAL), contracts
  participant_vol.csv.gz participant-wise traded contracts
  ban.csv.gz             F&O ban list, one row per (trade date, symbol)
  fii_stats.csv.gz       FII derivative statistics (buy / sell contracts and ₹ cr, OI) per instrument group.
                         PARTIAL: 2023-11-01..2025-11-18 only (the fetch stopped); FINAL-period decisions get no rows,
                         so a strategy needing it cannot be judged on FINAL - prefer participant().
All dates are int yyyymmdd. Symbols are the bars3y names (ZOMATO -> ETERNAL, MCDOWELL-N -> UNITDSPR mapped for
every date); indices are "NIFTY" and "BANKNIFTY".

AVAILABILITY (the one rule): NSE publishes the delivery, F&O bhavcopy and participant files in the EVENING of the
trade date D. A value dated D is usable for decisions from the OPEN of the next session after D. Every accessor
below therefore takes `before` and returns ONLY rows dated strictly before it:
    intraday @strategy (Session s)            before = s.date              -> EOD files up to the previous session
    Swing, default next-open fill             before = d.date[-1] + 1      -> EOD files up to AND INCLUDING the
                                                                              decision day d.date[-1] (published that
                                                                              evening, before the next open)
    Swing with decision_time="15:20" etc.     before = d.date[-1]          -> up to the previous session only (today's
                                                                              files do not exist yet at 15:20)
Use asof_session(s) / asof_swing(d) / asof_swing_intraday(d), which pick `before` for you.
The F&O BAN file is different: the file dated D is "Securities in Ban For Trade Date D", published the evening
BEFORE D (after D-1's close), so banned(sym, D) may be used for decisions at/after D's open (and by a next-open
Swing deciding on the evening of D-1). asof_*().banned() applies this.

Option chains: options_chain(sym, before, expiry) -> the EOD chain of the last session strictly before `before`.
option_eod(sym, expiry, strike, type, date) returns the row OF `date` itself: it is for MARKING / FILLING a held
position at that day's close inside the simulator, never for a decision on `date`. Inside the simulator every
strategy callback runs under cutoff(date), which makes any read dated >= date raise LookAheadError.

Field notes
  * delivery prices / volumes are NSE official and UNADJUSTED (bars3y are split/bonus adjusted): compare deliv_pct
    freely, but not close / volume / deliv_qty with the bars across corporate actions. ADANIPOWER has no rows
    2023-12-14..2024-03-15 (BE series).
  * F&O `oi` is in units (shares / index units), `contracts` is the number of contracts traded. Old-format
    bhavcopy (2023-11-01..2023-12-29; UDiFF from 2024-01-01) has no `underlying` and no `lot`; `spot` below comes from delivery.csv close (stocks)
    and the lab's NIFTY / BANKNIFTY daily closes (indices) instead.
  * option `close` is NSE's close (last trade; a theoretical price when the strike did not trade: contracts == 0).
    On the expiry day the old format's option `settle` is the underlying's final settlement price.
  * near = the nearest listed expiry on or after the row's date (on an expiry day: the expiring contract).

OPTION SIMULATOR (OptionSpec / run_option_specs, see their docstrings): entry and exits at a session's EOD close,
expiry settlement at intrinsic vs the expiring future's final settlement price, costs via
options_replay.leg_cost (= costs.round_trip_cost(buy_value, sell_value, "option") per leg + exercise STT on long ITM
settlement); lots from options_vrp.LOT_SCHEDULE (NIFTY 75, 65 from 2025-12-31; BANKNIFTY 35, 30) with a DISCOVERY-
only sensitivity on the exchange's historical lots (bhavcopy lot column)
plus a haircut of max(0.05, 1 % of price) per leg per fill; judged by protocol.evaluate exactly like
options_vrp.py (same calendar = NIFTY minute sessions, same DISCOVERY / FINAL split, per-data-directory FINAL
ledger, no --peek-final).

    cd algotrader_v4
    python -m stratlab.nsedata --check                                   # coverage report
    python -m stratlab.nsedata --data <bars3y> --out <scratch> --specs stratlab.my_opts:SPECS [--screen-only]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import glob
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
NSE_DIR = os.environ.get("STRATLAB_NSE_DIR", os.path.join(_SCRATCH, "nse"))
BARS_DIR = os.environ.get("STRATLAB_BARS_DIR", os.path.join(_SCRATCH, "bars3y"))
CACHE_VERSION = "nse3"
INDICES = ("NIFTY", "BANKNIFTY")
NAN = float("nan")


class LookAheadError(RuntimeError):
    """A read of data dated at or after the active cutoff (see cutoff())."""


# ══════════════════════════════════════════════════════════════════════════
# cutoff: the global look-ahead guard used by the simulator and the tests
# ══════════════════════════════════════════════════════════════════════════
_CUT: list = []          # stack of active cutoffs (int yyyymmdd): nothing dated >= min(cutoffs) may be read


@contextlib.contextmanager
def cutoff(day: int):
    """Inside this block every accessor refuses data dated >= day (accessors with `before` are clipped to it;
    option_eod / banned for a date >= day raise LookAheadError)."""
    _CUT.append(int(day))
    try:
        yield
    finally:
        _CUT.pop()


def _clip(before) -> int:
    b = int(before)
    return min(b, min(_CUT)) if _CUT else b


def _guard_exact(day: int, what: str):
    if _CUT and int(day) >= min(_CUT):
        raise LookAheadError(f"{what} dated {day} read while only data before {min(_CUT)} is allowed")


# ══════════════════════════════════════════════════════════════════════════
# loading and caching
# ══════════════════════════════════════════════════════════════════════════

def _iso_to_int(s) -> np.ndarray:
    return np.array([int(x.replace("-", "")) for x in s], dtype=np.int64)


def _cache_dir() -> str:
    d = os.path.join(NSE_DIR, "_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _parts(sub: str) -> list:
    return sorted(glob.glob(os.path.join(NSE_DIR, "fo_parts", sub, "*.csv.gz")))


def _ok_days() -> set:
    """F&O days whose bhavcopy parsed with a matching content date (status 'ok')."""
    p = os.path.join(NSE_DIR, "fo_parts", "status.csv")
    out = set()
    if os.path.exists(p):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r["status"] == "ok":
                    out.add(int(r["date"].replace("-", "")))
    return out


def _signature() -> str:
    h = hashlib.sha1(CACHE_VERSION.encode())
    for f in ["delivery.csv.gz", "participant_oi.csv.gz", "participant_vol.csv.gz", "ban.csv.gz",
              os.path.join("fo_parts", "status.csv")]:
        p = os.path.join(NSE_DIR, f)
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
    for sub in ("fut", "optidx", "optstk", "optstk_full"):
        fs = _parts(sub)
        h.update(f"{sub}:{len(fs)}:{fs[-1] if fs else ''}".encode())
    return h.hexdigest()[:16]


def aliases() -> dict:
    """Old NSE symbol -> bars3y file name, derived from delivery.csv's bars_symbol column."""
    _load()
    return dict(_S["alias"])


def _build_cache(cdir: str):
    import pandas as pd
    # ── delivery: bars symbols only
    dv = pd.read_csv(os.path.join(NSE_DIR, "delivery.csv.gz"), keep_default_na=True,
                     dtype={"symbol": str, "bars_symbol": str})
    alias = {a: b for a, b in dv.loc[dv.bars_symbol.notna() & (dv.symbol != dv.bars_symbol),
                                     ["symbol", "bars_symbol"]].drop_duplicates().itertuples(index=False)}
    dv = dv[dv.bars_symbol.notna()].copy()
    dv["sym"] = dv.bars_symbol
    dv = dv.sort_values(["sym", "date"])
    np.savez(os.path.join(cdir, "delivery.npz"), sym=dv.sym.to_numpy(str), date=dv.date.to_numpy(np.int64),
             **{k: dv[k].to_numpy(float) for k in ("open", "high", "low", "close", "prev_close", "avg_price", "volume",
                                                    "turnover_lacs", "trades", "deliv_qty", "deliv_pct")})
    ok = _ok_days()

    def cat(sub):
        fs = [f for f in _parts(sub) if int(os.path.basename(f)[:8]) in ok]
        if not fs:
            return None
        df = pd.concat([pd.read_csv(f, dtype={"symbol": str, "type": str}) for f in fs], ignore_index=True)
        df["date"] = _iso_to_int(df["date"])
        df["expiry"] = _iso_to_int(df["expiry"])
        df["symbol"] = df["symbol"].map(lambda s: alias.get(s, s))
        return df

    fut = cat("fut")
    if fut is not None:
        fut = fut.sort_values(["symbol", "date", "expiry"])
        np.savez(os.path.join(cdir, "fut.npz"), sym=fut.symbol.to_numpy(str), date=fut.date.to_numpy(np.int64),
                 expiry=fut.expiry.to_numpy(np.int64), idx=(fut.instrument == "FUTIDX").to_numpy(bool),
                 **{k: fut[k].to_numpy(float) for k in ("open", "high", "low", "close", "settle", "contracts", "oi",
                                                         "chg_oi", "underlying", "lot")})
    oi = cat("optidx")
    if oi is not None:
        oi = oi.sort_values(["symbol", "date", "expiry", "strike", "type"])
        for und, g in oi.groupby("symbol"):
            _save_chain(os.path.join(cdir, f"opt_{und}.npz"), g)
    full = cat("optstk_full")
    if full is not None:
        os.makedirs(os.path.join(cdir, "optstk"), exist_ok=True)
        full = full.sort_values(["symbol", "date", "expiry", "strike", "type"])
        for und, g in full.groupby("symbol"):
            _save_chain(os.path.join(cdir, "optstk", f"{und}.npz"), g)
    agg = cat("optstk")
    if agg is not None:
        agg = agg.sort_values(["symbol", "date", "expiry"])
        np.savez(os.path.join(cdir, "optstk_agg.npz"), sym=agg.symbol.to_numpy(str), date=agg.date.to_numpy(np.int64),
                 expiry=agg.expiry.to_numpy(np.int64),
                 **{k: agg[k].to_numpy(float) for k in ("call_oi", "put_oi", "call_vol", "put_vol", "chg_call_oi",
                                                         "chg_put_oi", "max_call_oi_strike", "max_put_oi_strike")})
    with open(os.path.join(cdir, "alias.json"), "w") as fh:
        json.dump(alias, fh)


def _save_chain(path, g):
    np.savez(path, date=g.date.to_numpy(np.int64), expiry=g.expiry.to_numpy(np.int64),
             strike=g.strike.to_numpy(float), ce=(g.type == "CE").to_numpy(bool),
             **{k: g[k].to_numpy(float) for k in ("open", "high", "low", "close", "settle", "contracts", "oi",
                                                   "chg_oi", "underlying", "lot")})


_S: dict = {}


def _npz(path) -> dict | None:
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _by_symbol(t: dict) -> dict:
    """{sym: slice} for a table sorted by sym."""
    out = {}
    s = t["sym"]
    if len(s) == 0:
        return out
    edges = np.flatnonzero(s[1:] != s[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(s)]
    for a, b in zip(starts, ends):
        out[str(s[a])] = slice(int(a), int(b))
    return out


def _load(force: bool = False):
    """Load (building the cache when the source files changed) once per process."""
    if _S and not force:
        return
    cdir = _cache_dir()
    sig = _signature()
    sp = os.path.join(cdir, "signature.txt")
    with open(os.path.join(cdir, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if force or not os.path.exists(sp) or open(sp).read().strip() != sig:
            for f in glob.glob(os.path.join(cdir, "*.npz")) + glob.glob(os.path.join(cdir, "optstk", "*.npz")):
                os.remove(f)
            _build_cache(cdir)
            with open(sp, "w") as fh:
                fh.write(sig)
    _S.clear()
    _S["alias"] = json.load(open(os.path.join(cdir, "alias.json")))
    dv = _npz(os.path.join(cdir, "delivery.npz"))
    _S["dv"], _S["dv_ix"] = dv, _by_symbol(dv)
    fut = _npz(os.path.join(cdir, "fut.npz"))
    _S["fut"] = fut
    _S["fut_ix"] = _by_symbol(fut) if fut is not None else {}
    agg = _npz(os.path.join(cdir, "optstk_agg.npz"))
    _S["agg"] = agg
    _S["agg_ix"] = _by_symbol(agg) if agg is not None else {}
    _S["chains"] = {}
    _S["fo_days"] = np.array(sorted(_ok_days()), dtype=np.int64)
    _S["daily_fut"], _S["daily_opt"] = {}, {}
    _load_participants()
    _load_ban()
    _load_fii()
    _S["spot_idx"] = _index_spot()


def _load_participants():
    import pandas as pd
    for kind in ("oi", "vol"):
        p = os.path.join(NSE_DIR, f"participant_{kind}.csv.gz")
        if not os.path.exists(p):
            _S[f"part_{kind}"] = None
            continue
        df = pd.read_csv(p)
        df["date"] = _iso_to_int(df["date"])
        cols = [c for c in df.columns if c not in ("date", "client_type")]
        wide = {}
        dates = np.array(sorted(df.date.unique()), dtype=np.int64)
        for ct, g in df.groupby("client_type"):
            g = g.set_index("date").reindex(dates)
            for c in cols:
                wide[f"{ct}_{c}"] = g[c].to_numpy(float)
        wide["date"] = dates
        _S[f"part_{kind}"] = wide


def _load_fii():
    import pandas as pd
    p = os.path.join(NSE_DIR, "fii_stats.csv.gz")
    out = {}
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["date"] = _iso_to_int(df["date"])
        for ins, g in df.sort_values("date").groupby("instrument"):
            out[str(ins)] = {"date": g.date.to_numpy(np.int64),
                             **{c: g[c].to_numpy(float) for c in g.columns if c not in ("date", "instrument")}}
    _S["fii"] = out


def _load_ban():
    import pandas as pd
    p = os.path.join(NSE_DIR, "ban.csv.gz")
    ban: dict = {}
    if os.path.exists(p):
        df = pd.read_csv(p)
        for d, s in zip(_iso_to_int(df["date"]), df["symbol"]):
            ban.setdefault(int(d), set()).add(_S["alias"].get(s, s))
    _S["ban"] = ban
    # trade dates for which a ban file exists (an empty list is a valid file): the raw file names
    files = glob.glob(os.path.join(NSE_DIR, "raw_part", "ban", "*.csv"))
    _S["ban_days"] = np.array(sorted(int(os.path.basename(f)[:8]) for f in files), dtype=np.int64)


def _index_spot() -> dict:
    """{index: (dates, closes)} from the lab bars: NIFTY day file, BANKNIFTY last minute close per day."""
    out = {}
    try:
        b = json.load(open(os.path.join(BARS_DIR, "_NIFTY50_day.json")))["bars"]
        out["NIFTY"] = (np.array([int(x[0][:10].replace("-", "")) for x in b], np.int64), np.array([x[4] for x in b], float))
    except Exception:
        pass
    try:
        b = json.load(open(os.path.join(BARS_DIR, "_NIFTYBANK_min.json")))["bars"]
        last: dict = {}
        for x in b:
            last[int(x[0][:10].replace("-", ""))] = x[4]
        ks = sorted(last)
        out["BANKNIFTY"] = (np.array(ks, np.int64), np.array([last[k] for k in ks], float))
    except Exception:
        pass
    return out


# ══════════════════════════════════════════════════════════════════════════
# result container
# ══════════════════════════════════════════════════════════════════════════

class Rows(dict):
    """dict of equal-length numpy arrays, oldest first; attribute access (r.deliv_pct) and len(r)."""
    __getattr__ = dict.__getitem__

    def __len__(self):
        return len(self["date"]) if "date" in self else 0

    def __bool__(self):
        return len(self) > 0


def _tail(t: dict, sl: slice, before: int, n: int, keys) -> Rows:
    d = t["date"][sl]
    k = int(np.searchsorted(d, before, side="left"))
    a = max(0, k - int(n))
    return Rows({c: t[c][sl][a:k].copy() for c in keys})


# ══════════════════════════════════════════════════════════════════════════
# public accessors (all: data dated strictly before `before`)
# ══════════════════════════════════════════════════════════════════════════
DELIVERY_FIELDS = ("date", "deliv_pct", "deliv_qty", "volume", "trades", "turnover_lacs", "open", "high", "low",
                   "close", "prev_close", "avg_price")


def delivery(sym: str, before: int, n: int = 1) -> Rows:
    """Last n sessions strictly before `before` of sec_bhavdata_full for a bars3y symbol: date, deliv_pct (0-100),
    deliv_qty, volume, trades, turnover_lacs, open/high/low/close/prev_close/avg_price (official, UNADJUSTED)."""
    _load()
    sl = _S["dv_ix"].get(sym)
    if sl is None:
        return Rows({c: np.array([]) for c in DELIVERY_FIELDS})
    return _tail(_S["dv"], sl, _clip(before), n, DELIVERY_FIELDS)


FUT_FIELDS = ("date", "oi", "chg_oi", "contracts", "near_expiry", "near_close", "near_settle", "near_oi",
              "next_close", "spot", "basis_pct", "dte", "lot")


def _spot(sym: str, day: int) -> float:
    if sym in INDICES:
        s = _S["spot_idx"].get(sym)
        if s is None:
            return NAN
        k = int(np.searchsorted(s[0], day))
        return float(s[1][k]) if k < len(s[0]) and s[0][k] == day else NAN
    sl = _S["dv_ix"].get(sym)
    if sl is None:
        return NAN
    d = _S["dv"]["date"][sl]
    k = int(np.searchsorted(d, day))
    return float(_S["dv"]["close"][sl][k]) if k < len(d) and d[k] == day else NAN


def _daily_fut(sym: str) -> Rows | None:
    """Per-day futures aggregate for one symbol (built lazily, cached)."""
    if sym in _S["daily_fut"]:
        return _S["daily_fut"][sym]
    t = _S["fut"]
    sl = _S["fut_ix"].get(sym) if t is not None else None
    if sl is None:
        _S["daily_fut"][sym] = None
        return None
    d, ex = t["date"][sl], t["expiry"][sl]
    days = np.unique(d)
    out = {c: np.full(len(days), NAN) for c in FUT_FIELDS}
    out["date"] = days
    starts = np.searchsorted(d, days, "left")
    ends = np.searchsorted(d, days, "right")
    for i, (a, b) in enumerate(zip(starts, ends)):
        e = ex[a:b]
        out["oi"][i] = np.nansum(t["oi"][sl][a:b])
        out["chg_oi"][i] = np.nansum(t["chg_oi"][sl][a:b])
        out["contracts"][i] = np.nansum(t["contracts"][sl][a:b])
        j = a + int(np.argmin(np.where(e >= days[i], e, 99999999)))     # nearest expiry on/after the day
        out["near_expiry"][i] = ex[j]
        out["near_close"][i] = t["close"][sl][j]
        out["near_settle"][i] = t["settle"][sl][j]
        out["near_oi"][i] = t["oi"][sl][j]
        later = [k for k in range(a, b) if ex[k] > ex[j]]
        if later:
            out["next_close"][i] = t["close"][sl][min(later, key=lambda k: ex[k])]
        lots = t["lot"][sl][a:b]
        out["lot"][i] = np.nanmax(lots) if np.isfinite(lots).any() else NAN
        sp = _spot(sym, int(days[i]))
        out["spot"][i] = sp
        out["basis_pct"][i] = 100.0 * (out["near_close"][i] / sp - 1.0) if sp == sp and sp > 0 else NAN
        out["dte"][i] = (_d(int(ex[j])) - _d(int(days[i]))).days
    r = Rows(out)
    _S["daily_fut"][sym] = r
    return r


def fut_oi(sym: str, before: int, n: int = 1) -> Rows:
    """Futures (stock or index) per session, last n sessions strictly before `before`: date, oi (all expiries,
    units), chg_oi, contracts (traded, all expiries), near_expiry, near_close, near_settle, near_oi, next_close
    (next expiry's close), spot (official cash close / index close), basis_pct = 100*(near_close/spot-1),
    dte (calendar days to near expiry), lot (NaN before 2024-01-01: old bhavcopy has no lot column)."""
    _load()
    r = _daily_fut(sym)
    if r is None:
        return Rows({c: np.array([]) for c in FUT_FIELDS})
    return _tail(r, slice(0, len(r)), _clip(before), n, FUT_FIELDS)


OPT_AGG_FIELDS = ("date", "call_oi", "put_oi", "pcr_oi", "call_vol", "put_vol", "pcr_vol", "chg_call_oi",
                  "chg_put_oi", "near_expiry", "near_call_oi", "near_put_oi", "max_call_oi_strike",
                  "max_put_oi_strike")


def _chain_table(sym: str) -> dict | None:
    if sym in _S["chains"]:
        return _S["chains"][sym]
    cdir = _cache_dir()
    p = os.path.join(cdir, f"opt_{sym}.npz") if sym in INDICES else os.path.join(cdir, "optstk", f"{sym}.npz")
    t = _npz(p)
    _S["chains"][sym] = t
    return t


def _daily_opt(sym: str) -> Rows | None:
    if sym in _S["daily_opt"]:
        return _S["daily_opt"][sym]
    out = None
    if sym in INDICES:
        t = _chain_table(sym)
        if t is not None:
            days = np.unique(t["date"])
            o = {c: np.full(len(days), NAN) for c in OPT_AGG_FIELDS}
            o["date"] = days
            st, en = np.searchsorted(t["date"], days, "left"), np.searchsorted(t["date"], days, "right")
            for i, (a, b) in enumerate(zip(st, en)):
                ce, oi, vol, ch, ex, k = (t["ce"][a:b], t["oi"][a:b], t["contracts"][a:b], t["chg_oi"][a:b],
                                          t["expiry"][a:b], t["strike"][a:b])
                o["call_oi"][i], o["put_oi"][i] = oi[ce].sum(), oi[~ce].sum()
                o["call_vol"][i], o["put_vol"][i] = vol[ce].sum(), vol[~ce].sum()
                o["chg_call_oi"][i], o["chg_put_oi"][i] = ch[ce].sum(), ch[~ce].sum()
                ne = ex[ex >= days[i]].min() if (ex >= days[i]).any() else ex.max()
                m = ex == ne
                o["near_expiry"][i] = ne
                o["near_call_oi"][i], o["near_put_oi"][i] = oi[m & ce].sum(), oi[m & ~ce].sum()
                if (m & ce).any() and oi[m & ce].max() > 0:
                    o["max_call_oi_strike"][i] = k[m & ce][np.argmax(oi[m & ce])]
                if (m & ~ce).any() and oi[m & ~ce].max() > 0:
                    o["max_put_oi_strike"][i] = k[m & ~ce][np.argmax(oi[m & ~ce])]
            out = o
    else:
        t, sl = _S["agg"], _S["agg_ix"].get(sym)
        if t is not None and sl is not None:
            d, ex = t["date"][sl], t["expiry"][sl]
            days = np.unique(d)
            o = {c: np.full(len(days), NAN) for c in OPT_AGG_FIELDS}
            o["date"] = days
            st, en = np.searchsorted(d, days, "left"), np.searchsorted(d, days, "right")
            for i, (a, b) in enumerate(zip(st, en)):
                g = {c: t[c][sl][a:b] for c in ("call_oi", "put_oi", "call_vol", "put_vol", "chg_call_oi", "chg_put_oi")}
                for c, v in g.items():
                    o[c][i] = np.nansum(v)
                e = ex[a:b]
                j = int(np.argmin(np.where(e >= days[i], e, 99999999)))
                o["near_expiry"][i] = e[j]
                o["near_call_oi"][i], o["near_put_oi"][i] = g["call_oi"][j], g["put_oi"][j]
                o["max_call_oi_strike"][i] = t["max_call_oi_strike"][sl][a + j]
                o["max_put_oi_strike"][i] = t["max_put_oi_strike"][sl][a + j]
            out = o
    if out is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["pcr_oi"] = np.where(out["call_oi"] > 0, out["put_oi"] / out["call_oi"], NAN)
            out["pcr_vol"] = np.where(out["call_vol"] > 0, out["put_vol"] / out["call_vol"], NAN)
        out = Rows(out)
    _S["daily_opt"][sym] = out
    return out


def opt_agg(sym: str, before: int, n: int = 1) -> Rows:
    """Option open interest / volume per session (all expiries), last n sessions strictly before `before`:
    call_oi, put_oi, pcr_oi (= put_oi / call_oi), call_vol, put_vol (contracts), pcr_vol, chg_call_oi, chg_put_oi,
    near_expiry, near_call_oi, near_put_oi, max_call_oi_strike / max_put_oi_strike (near expiry).
    Works for stocks and for NIFTY / BANKNIFTY."""
    _load()
    r = _daily_opt(sym)
    if r is None:
        return Rows({c: np.array([]) for c in OPT_AGG_FIELDS})
    return _tail(r, slice(0, len(r)), _clip(before), n, OPT_AGG_FIELDS)


def participant(before: int, n: int = 1, kind: str = "oi") -> Rows:
    """Participant-wise OI (kind="oi", contracts outstanding) or traded contracts (kind="vol"), last n sessions
    strictly before `before`. Keys: date and f"{who}_{col}" with who in Client, DII, FII, Pro, TOTAL and col in
    fut_idx_long, fut_idx_short, fut_stk_long, fut_stk_short, opt_idx_call_long, opt_idx_put_long,
    opt_idx_call_short, opt_idx_put_short, opt_stk_call_long, opt_stk_put_long, opt_stk_call_short,
    opt_stk_put_short, total_long, total_short.  E.g. FII index-futures net long ratio:
    p.FII_fut_idx_long / (p.FII_fut_idx_long + p.FII_fut_idx_short)."""
    _load()
    t = _S.get(f"part_{kind}")
    if t is None:
        return Rows({"date": np.array([])})
    return _tail(t, slice(0, len(t["date"])), _clip(before), n, list(t.keys()))


FII_STATS_GROUPS = ("INDEX FUTURES", "INDEX OPTIONS", "STOCK FUTURES", "STOCK OPTIONS", "NIFTY FUTURES",
                    "NIFTY OPTIONS", "BANKNIFTY FUTURES", "BANKNIFTY OPTIONS", "FINNIFTY FUTURES", "FINNIFTY OPTIONS",
                    "MIDCPNIFTY FUTURES", "MIDCPNIFTY OPTIONS", "NIFTYNXT50 FUTURES", "NIFTYNXT50 OPTIONS")


def fii_stats(before: int, n: int = 1, instrument: str = "INDEX FUTURES") -> Rows:
    """FII derivative statistics for one instrument group (FII_STATS_GROUPS), last n sessions strictly before
    `before`: date, buy_contracts, buy_amt_cr, sell_contracts, sell_amt_cr, oi_contracts, oi_amt_cr.
    PARTIAL COVERAGE 2023-11-01..2025-11-18 (empty / stale afterwards: check r.date[-1])."""
    _load()
    t = _S.get("fii", {}).get(instrument)
    if t is None:
        return Rows({"date": np.array([])})
    return _tail(t, slice(0, len(t["date"])), _clip(before), n, list(t.keys()))


def banned(sym: str, trade_date: int) -> bool:
    """True if `sym` is in the F&O ban period ON `trade_date` (the file "Securities in Ban For Trade Date D",
    published the evening before D, so usable at D's open). A banned security takes no fresh F&O positions
    (only reducing trades); its cash market trades normally. Returns False for dates with no ban file."""
    _load()
    _guard_exact(_prev_or_same_cut(trade_date), "ban list")
    return sym in _S["ban"].get(int(trade_date), ())


def _prev_or_same_cut(trade_date: int) -> int:
    """The ban file for trade date D is published after D-1's close: inside cutoff(c) (data < c allowed, i.e.
    a decision at c's open or later) the ban for D <= c is known. Map D to the date its information is 'dated'."""
    return int(trade_date) - 1 if _CUT else int(trade_date)


def ban_list(trade_date: int) -> set:
    """All symbols banned on `trade_date` (same semantics as banned())."""
    _load()
    _guard_exact(_prev_or_same_cut(trade_date), "ban list")
    return set(_S["ban"].get(int(trade_date), ()))


def fo_sessions(before: int | None = None) -> np.ndarray:
    """F&O bhavcopy session dates (status ok), optionally strictly before `before`."""
    _load()
    d = _S["fo_days"]
    return d if before is None else d[d < _clip(before)]


def next_fo_session(day: int) -> int | None:
    """First F&O / ban-calendar trade date after `day` (the exchange calendar is public in advance)."""
    _load()
    d = np.union1d(_S["fo_days"], _S["ban_days"])
    k = int(np.searchsorted(d, int(day), side="right"))
    return int(d[k]) if k < len(d) else None


# ── option chains ───────────────────────────────────────────────────────────

@dataclass
class Chain:
    """EOD option chain of one session (`date`) for one expiry, aligned on strikes (NaN = strike not listed / not
    kept). ce_*/pe_*: close, settle, oi, chg_oi, contracts (traded), plus spot (underlying close of `date`) and
    fut (near future close of `date`)."""
    symbol: str
    date: int
    expiry: int
    strike: np.ndarray
    ce_close: np.ndarray
    pe_close: np.ndarray
    ce_settle: np.ndarray
    pe_settle: np.ndarray
    ce_oi: np.ndarray
    pe_oi: np.ndarray
    ce_chg_oi: np.ndarray
    pe_chg_oi: np.ndarray
    ce_vol: np.ndarray
    pe_vol: np.ndarray
    spot: float
    fut: float

    def atm(self, ref: float | None = None) -> float:
        """Strike nearest to ref (default: spot) among strikes with both a CE and a PE close."""
        ok = np.isfinite(self.ce_close) & np.isfinite(self.pe_close)
        if not ok.any():
            return NAN
        r = self.spot if ref is None else ref
        k = self.strike[ok]
        return float(k[np.argmin(np.abs(k - r))])

    def price(self, strike: float, typ: str) -> float:
        i = np.flatnonzero(self.strike == strike)
        if not len(i):
            return NAN
        return float((self.ce_close if typ == "CE" else self.pe_close)[i[0]])


def expiries(sym: str, before: int) -> list:
    """Expiries listed in the last chain session strictly before `before` (sorted)."""
    _load()
    t = _chain_table(sym)
    if t is None:
        return []
    b = _clip(before)
    k = int(np.searchsorted(t["date"], b, "left"))
    if k == 0:
        return []
    day = t["date"][k - 1]
    a = int(np.searchsorted(t["date"], day, "left"))
    return sorted(int(x) for x in np.unique(t["expiry"][a:k]))


def options_chain(sym: str, before: int, expiry="near") -> Chain | None:
    """The EOD chain of the LAST session strictly before `before` (for an intraday decision on D: D-1's chain).
    expiry: "near" (nearest listed expiry >= `before`, i.e. still alive on the decision day), "next" (the one
    after), or an int yyyymmdd. NIFTY / BANKNIFTY: every strike. Stocks: two nearest expiries, strikes with
    OI > 0 or volume > 0 only."""
    _load()
    t = _chain_table(sym)
    if t is None:
        return None
    b = _clip(before)
    k = int(np.searchsorted(t["date"], b, "left"))
    if k == 0:
        return None
    day = int(t["date"][k - 1])
    a = int(np.searchsorted(t["date"], day, "left"))
    exs = sorted(int(x) for x in np.unique(t["expiry"][a:k]))
    alive = [e for e in exs if e >= int(before)]
    if expiry == "near":
        if not alive:
            return None
        ex = alive[0]
    elif expiry == "next":
        if len(alive) < 2:
            return None
        ex = alive[1]
    else:
        ex = int(expiry)
        if ex not in exs:
            return None
    m = np.flatnonzero(t["expiry"][a:k] == ex) + a
    strikes = np.unique(t["strike"][m])
    cols = {}
    for side, mask in (("ce", t["ce"][m]), ("pe", ~t["ce"][m])):
        mm = m[mask]
        pos = np.searchsorted(strikes, t["strike"][mm])
        for src, dst in (("close", "close"), ("settle", "settle"), ("oi", "oi"), ("chg_oi", "chg_oi"), ("contracts", "vol")):
            arr = np.full(len(strikes), NAN)
            arr[pos] = t[src][mm]
            cols[f"{side}_{dst}"] = arr
    if ex == day:
        # old-format rows carry the underlying's final settlement price in `settle` on the expiry day: never
        # expose it as an option settle
        cols["ce_settle"][:] = NAN
        cols["pe_settle"][:] = NAN
    sp = _spot(sym, day)
    und = t["underlying"][m]
    if not sp == sp and np.isfinite(und).any():
        sp = float(np.nanmedian(und))
    fr = _daily_fut(sym)
    fut = NAN
    if fr is not None:
        j = int(np.searchsorted(fr["date"], day))
        if j < len(fr["date"]) and fr["date"][j] == day:
            fut = float(fr["near_close"][j])
    return Chain(sym, day, ex, strikes, cols["ce_close"], cols["pe_close"], cols["ce_settle"], cols["pe_settle"],
                 cols["ce_oi"], cols["pe_oi"], cols["ce_chg_oi"], cols["pe_chg_oi"], cols["ce_vol"], cols["pe_vol"],
                 sp, fut)


index_options_chain = options_chain


def option_eod(sym: str, expiry: int, strike: float, typ: str, date: int) -> dict | None:
    """EOD row of ONE contract ON `date` (not before it): close, settle, open, high, low, contracts, oi, chg_oi,
    underlying, lot. For marking / filling a position the simulator holds at `date`'s close. Never use it for a
    decision taken on `date` (inside cutoff(date) it raises LookAheadError). None if the row does not exist."""
    _load()
    _guard_exact(int(date), "option_eod")
    t = _chain_table(sym)
    if t is None:
        return None
    a, b = np.searchsorted(t["date"], int(date), "left"), np.searchsorted(t["date"], int(date), "right")
    if a == b:
        return None
    m = np.flatnonzero((t["expiry"][a:b] == int(expiry)) & (t["strike"][a:b] == float(strike))
                       & (t["ce"][a:b] == (typ == "CE"))) + a
    if not len(m):
        return None
    i = int(m[0])
    r = {k: float(t[k][i]) for k in ("open", "high", "low", "close", "settle", "contracts", "oi", "chg_oi",
                                     "underlying", "lot")}
    if int(expiry) == int(date):
        r["settle"] = NAN               # old format: the underlying's final settlement price, not the option's
    return r


def final_settlement(sym: str, expiry: int) -> float:
    """Final settlement price of `sym` for `expiry`: the expiring future's settle price on the expiry day
    (NSE: the underlying's closing price). Falls back to the old-format option settle / underlying column, then
    to the spot close. Same-day data: simulator use only."""
    _load()
    _guard_exact(int(expiry), "final settlement")
    t, sl = _S["fut"], _S["fut_ix"].get(sym)
    if t is not None and sl is not None:
        d, ex = t["date"][sl], t["expiry"][sl]
        m = np.flatnonzero((d == int(expiry)) & (ex == int(expiry)))
        if len(m) and np.isfinite(t["settle"][sl][m[0]]):
            return float(t["settle"][sl][m[0]])
    c = _chain_table(sym)
    if c is not None:
        m = np.flatnonzero((c["date"] == int(expiry)) & (c["expiry"] == int(expiry)))
        if len(m):
            v = c["underlying"][m]
            if np.isfinite(v).any():
                return float(np.nanmedian(v))
            v = c["settle"][m]
            if np.isfinite(v).any():
                return float(np.nanmedian(v))
    return _spot(sym, int(expiry))


def lot_size(sym: str, day: int, historical: bool = False) -> int:
    """Lot size used by the simulator. NIFTY: options_vrp.LOT_SCHEDULE (75, 65 from 2025-12-31); BANKNIFTY:
    35, 30 from 2025-12-31 (same NSE circular FAOP70616, as sim.LOT_SIZES); stocks: the bhavcopy lot in force
    on `day` (the earliest known one before the bhavcopy carried lots, 2024-01-01).
    historical=True (sensitivity only): indices use the exchange's lot of `day` from the option bhavcopy (largest
    lot listed that day), before 2024-01-01 options_vrp.LOT_SCHEDULE_HISTORICAL (NIFTY) / 15 (BANKNIFTY)."""
    _load()
    if historical and sym in INDICES:
        t = _chain_table(sym)
        if t is not None:
            a, b = np.searchsorted(t["date"], int(day), "left"), np.searchsorted(t["date"], int(day), "right")
            l = t["lot"][a:b]
            if np.isfinite(l).any():
                return int(np.nanmax(l))
        if sym == "BANKNIFTY":
            return 15
        from .options_vrp import LOT_SCHEDULE_HISTORICAL
        lot = LOT_SCHEDULE_HISTORICAL[0][1]
        for f, n in LOT_SCHEDULE_HISTORICAL:
            if _d(int(day)) >= f:
                lot = n
        return int(lot)
    if sym == "NIFTY":
        from .options_vrp import LOT_SCHEDULE
        sch = LOT_SCHEDULE
    elif sym == "BANKNIFTY":
        sch = BANKNIFTY_LOT_SCHEDULE
    else:
        r = _daily_fut(sym)
        if r is None:
            raise KeyError(f"no futures rows for {sym}")
        ok = np.isfinite(r["lot"])
        if not ok.any():
            raise KeyError(f"no lot size for {sym}")
        d, l = r["date"][ok], r["lot"][ok]
        k = int(np.searchsorted(d, int(day), side="right")) - 1
        return int(l[max(k, 0)])
    lot = sch[0][1]
    for f, n in sch:
        if _d(int(day)) >= f:
            lot = n
    return int(lot)


BANKNIFTY_LOT_SCHEDULE = ((_date(2000, 1, 1), 35), (_date(2025, 12, 31), 30))


def _d(x: int) -> _date:
    x = int(x)
    return _date(x // 10000, x // 100 % 100, x % 100)


# ══════════════════════════════════════════════════════════════════════════
# strategy-side helpers
# ══════════════════════════════════════════════════════════════════════════

class Asof:
    """All accessors bound to one symbol and one `before`. Build it with asof_session / asof_swing /
    asof_swing_intraday rather than by hand."""

    def __init__(self, symbol: str, before: int, trade_date: int | None):
        self.symbol, self.before, self.trade_date = symbol, int(before), trade_date

    def delivery(self, n=1, sym=None):
        return delivery(sym or self.symbol, self.before, n)

    def fut_oi(self, n=1, sym=None):
        return fut_oi(sym or self.symbol, self.before, n)

    def opt_agg(self, n=1, sym=None):
        return opt_agg(sym or self.symbol, self.before, n)

    def participant(self, n=1, kind="oi"):
        return participant(self.before, n, kind)

    def fii_stats(self, n=1, instrument="INDEX FUTURES"):
        return fii_stats(self.before, n, instrument)

    def chain(self, expiry="near", sym=None):
        return options_chain(sym or self.symbol, self.before, expiry)

    def banned(self, sym=None) -> bool:
        """Ban status for the day the position would be opened (None trade date -> False)."""
        if self.trade_date is None:
            return False
        return banned(sym or self.symbol, self.trade_date)


def asof_session(s) -> Asof:
    """Inside an intraday @strategy: EOD data up to the previous session; ban list of today (s.date)."""
    return Asof(s.symbol, int(s.date), int(s.date))


def asof_swing(d) -> Asof:
    """Inside a Swing strategy with the default NEXT-OPEN fill: EOD data up to and including d.date[-1] (published
    that evening, before the fill); ban list of the next trade date (published that evening too)."""
    day = int(d.date[-1])
    return Asof(d.symbol, day + 1, next_fo_session(day))


def asof_swing_intraday(d) -> Asof:
    """Inside a Swing strategy with decision_time (fill during day d.date[-1]): EOD data up to the PREVIOUS
    session only; ban list of d.date[-1]."""
    day = int(d.date[-1])
    return Asof(d.symbol, day, day)


# ══════════════════════════════════════════════════════════════════════════
# real-option-price simulator
# ══════════════════════════════════════════════════════════════════════════
OPT_FILL_MODEL = "nse_eod1"     # bump when fill / settlement / haircut rules change (part of the FINAL-ledger key)
HAIRCUT_MIN = 0.05              # ₹ per unit, per leg per fill
HAIRCUT_PCT = 0.01              # of the EOD close
SENS_MULT = 2.5                 # haircut multiple for the DISCOVERY-only "net@5" sensitivity


def cost_model() -> str:
    import costs as _costs
    return f"{_costs.COST_MODEL_VERSION}/{OPT_FILL_MODEL}"


@dataclass
class Leg:
    typ: str            # "CE" | "PE"
    strike: float
    side: int           # +1 buy, -1 sell
    lots: int = 1
    expiry: int | None = None       # None = the plan's expiry


@dataclass
class Plan:
    """What decide() returns: legs to open at today's EOD close. exit: "expiry" (hold to settlement) or an int N
    (close at the EOD close of the N-th session after entry; settlement first if a leg expires earlier)."""
    expiry: int
    legs: list
    exit: object = "expiry"
    note: str = ""


@dataclass
class Ctx:
    """Passed to decide() / manage(). `date` is today's session; everything readable through `asof` (and the
    module accessors, which are clipped by the active cutoff) is dated strictly before `date`."""
    symbol: str
    date: int
    asof: Asof
    sessions_left_in_period: int

    def chain(self, expiry="near"):
        return self.asof.chain(expiry)


@dataclass
class Position:
    symbol: str
    entry_date: int
    plan: Plan
    qty_per_lot: int
    fills: list                 # per leg: entry fill price per unit
    marks: list = field(default_factory=list)   # [(date, value per unit of the position at that EOD close)]
    held: int = 0               # sessions held after entry


@dataclass
class OptionSpec:
    """A multi-day option strategy. decide(ctx, p) -> Plan | None is called once per session while flat (per
    underlying); manage(ctx, pos, p) -> bool (optional) once per later session BEFORE that session's close: True =
    close at today's EOD close. Both run under cutoff(today): they can only read EOD data dated before today.
    `underlyings`: NIFTY / BANKNIFTY / stock symbols (bars3y names)."""
    id: str
    desc: str
    params: dict
    decide: Callable
    underlyings: tuple = ("NIFTY",)
    manage: Optional[Callable] = None
    family: str = "nse_options"
    kind: str = "options"
    horizon: str = "positional"
    needs_index: bool = False

    @property
    def underlying(self):           # protocol: one-instrument strategies skip the 35 % symbol gate
        return self.underlyings[0] if len(self.underlyings) == 1 else None

    @property
    def p(self):
        return type("P", (), dict(self.params))

    @property
    def fingerprint(self) -> str:
        src = "".join(inspect.getsource(f) for f in (self.decide, self.manage, _simulate_one, _fill, _close_position)
                      if f is not None)
        blob = repr((self.id, sorted(self.params.items()), tuple(self.underlyings), src, OPT_FILL_MODEL))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _fill(close: float, side: int, mult: float) -> tuple[float, float]:
    """Fill at the EOD close with the haircut against us: (price, haircut per unit)."""
    h = max(HAIRCUT_MIN, HAIRCUT_PCT * close) * mult
    return (close + h, h) if side > 0 else (max(0.05, close - h), min(h, close - 0.05) if close > 0.05 else 0.0)


def _leg_expiry(pos_or_plan, leg) -> int:
    plan = pos_or_plan.plan if isinstance(pos_or_plan, Position) else pos_or_plan
    return int(leg.expiry or plan.expiry)


def _close_position(spec, pos: Position, day: int, reason: str, mult: float, settle: bool) -> tuple:
    from options_replay import leg_cost
    gross = fees = hair = 0.0
    exit_val = 0.0
    stale = False
    for leg, px_in in zip(pos.plan.legs, pos.fills):
        q = leg.lots * pos.qty_per_lot
        ex = _leg_expiry(pos, leg)
        if settle and ex <= day:
            s = final_settlement(pos.symbol, ex)
            px_out = max(0.0, s - leg.strike) if leg.typ == "CE" else max(0.0, leg.strike - s)
            h = 0.0
            settled = True
        else:
            r = option_eod(pos.symbol, ex, leg.strike, leg.typ, day)
            c = r["close"] if r is not None else NAN
            if not c == c:                                 # row missing: last known close
                c = _last_close(pos.symbol, ex, leg.strike, leg.typ, day)
                stale = True
            px_out, h = _fill(c, -leg.side, mult)
            settled = False
        gross += leg.side * (px_out - px_in) * q
        fees += leg_cost(px_in, px_out, leg.side, q, settled=settled)
        hair += h * q
        exit_val += -leg.side * px_out * leg.lots
    if stale:
        reason += "_stale"
    net = gross - fees
    entry_val = sum(leg.side * f * leg.lots for leg, f in zip(pos.plan.legs, pos.fills))
    side = 1 if entry_val > 0 else -1                     # +1 net debit, -1 net credit
    qty = pos.qty_per_lot
    tup = (spec.id, pos.symbol, pos.entry_date, side, "15:30", round(entry_val, 4), int(day), "15:30",
           round(exit_val, 4), qty, round(gross, 2), round(fees, 2), round(net, 2), reason)
    det = {"date": pos.entry_date, "symbol": pos.symbol, "exit_date": int(day), "expiry": pos.plan.expiry,
           "lot": qty, "legs": " | ".join(f"{'B' if l.side > 0 else 'S'}{l.lots} {l.strike:g}{l.typ} {f:.2f}"
                                          for l, f in zip(pos.plan.legs, pos.fills)),
           "entry_val": round(entry_val, 2), "exit_val": round(exit_val, 2), "gross": round(gross, 2),
           "fees": round(fees, 2), "haircut_exit": round(hair, 2), "net": round(net, 2), "reason": reason,
           "held": pos.held, "note": pos.plan.note}
    return tup, det


def _last_close(sym, ex, strike, typ, day) -> float:
    t = _chain_table(sym)
    m = np.flatnonzero((t["date"] < day) & (t["expiry"] == ex) & (t["strike"] == strike) & (t["ce"] == (typ == "CE")))
    return float(t["close"][m[-1]]) if len(m) else 0.0


def _mark(pos: Position, day: int) -> float:
    v = 0.0
    for leg in pos.plan.legs:
        r = option_eod(pos.symbol, _leg_expiry(pos, leg), leg.strike, leg.typ, day)
        c = r["close"] if r is not None else NAN
        v += leg.side * leg.lots * c
    return v


def _simulate_one(spec: OptionSpec, sym: str, calendar: list, lo: int, hi: int, mult: float = 1.0,
                  hist_lots: bool = False):
    """Trades of one spec on one underlying, entries in [lo, hi], every exit at the latest at the last session
    <= hi (reason period_end). Returns (trade tuples, details, skipped)."""
    fo = set(int(x) for x in fo_sessions())
    days = [d for d in calendar if d <= hi and d in fo]      # lab sessions that have an F&O bhavcopy
    trades, dets, skipped = [], [], 0
    pos: Position | None = None
    p = spec.p
    for i, day in enumerate(days):
        last = i == len(days) - 1
        if pos is not None:
            pos.held += 1
            first_exp = min(_leg_expiry(pos, l) for l in pos.plan.legs)
            reason = None
            if first_exp <= day:
                reason = "expiry"
            elif isinstance(pos.plan.exit, int) and pos.held >= pos.plan.exit:
                reason = "hold"
            elif spec.manage is not None:
                ctx = Ctx(sym, day, Asof(sym, day, day), len(days) - 1 - i)
                with cutoff(day):
                    if spec.manage(ctx, pos, p):
                        reason = "manage"
            if reason is None and last:
                reason = "period_end"
            if reason is not None:
                tup, det = _close_position(spec, pos, day, reason, mult, settle=(reason == "expiry"))
                trades.append(tup)
                dets.append(det)
                pos = None
                continue                                  # one action per session
            pos.marks.append((day, _mark(pos, day)))
            continue
        if day < lo or last:
            continue
        ctx = Ctx(sym, day, Asof(sym, day, day), len(days) - 1 - i)
        with cutoff(day):
            plan = spec.decide(ctx, p)
        if plan is None:
            continue
        if not isinstance(plan, Plan) or not plan.legs:
            raise TypeError(f"{spec.id}: decide() returns Plan(expiry, legs, exit) or None")
        if min(_leg_expiry(plan, l) for l in plan.legs) <= day:
            raise ValueError(f"{spec.id} {day}: entry at the close of an expiry day (or past it) is not allowed")
        fills, ok = [], True
        for leg in plan.legs:
            r = option_eod(sym, _leg_expiry(plan, leg), leg.strike, leg.typ, day)
            if r is None or not r["contracts"] > 0 or not r["close"] > 0:
                ok = False                                # the strike did not trade today: no fill
                break
            fills.append(_fill(r["close"], leg.side, mult)[0])
        if not ok:
            skipped += 1
            continue
        pos = Position(sym, day, plan, lot_size(sym, day, historical=hist_lots), fills)
        pos.marks.append((day, _mark(pos, day)))
    return trades, dets, skipped


def audit_lookahead(spec: OptionSpec, sym: str, calendar: list, trades: list, n: int = 4) -> str | None:
    """Re-run decide() for sampled entry days with every NSE table physically cut at the entry day (cache rebuilt
    from truncated arrays) is expensive; instead re-run it under a stricter cutoff of the same day and compare
    with a run where the module's tables are masked to dates < entry day. decide() must be identical."""
    if not trades:
        return None
    step = max(1, len(trades) // n)
    for tr in trades[::step][:n]:
        day = tr[2]
        ctx = Ctx(sym, day, Asof(sym, day, day), 0)
        with cutoff(day), _masked(day):
            a = spec.decide(ctx, spec.p)
        with cutoff(day):
            b = spec.decide(ctx, spec.p)
        fa = None if a is None else [(l.typ, l.strike, l.side, l.lots, l.expiry) for l in a.legs] + [a.expiry, a.exit]
        fb = None if b is None else [(l.typ, l.strike, l.side, l.lots, l.expiry) for l in b.legs] + [b.expiry, b.exit]
        if fa != fb:
            return f"{sym} {day}: decision changes when data dated >= {day} is removed ({fb} vs {fa})"
    return None


@contextlib.contextmanager
def _masked(day: int):
    """Temporarily replace every in-memory table with a copy holding only rows dated < day."""
    _load()
    saved = {k: _S[k] for k in ("dv", "dv_ix", "fut", "fut_ix", "agg", "agg_ix", "chains", "daily_fut", "daily_opt",
                                "part_oi", "part_vol", "ban", "spot_idx", "fii")}

    def cut(t):
        if t is None:
            return None
        m = t["date"] < day
        return {k: v[m] for k, v in t.items()}
    try:
        _S["dv"] = cut(saved["dv"]); _S["dv_ix"] = _by_symbol(_S["dv"])
        _S["fut"] = cut(saved["fut"]); _S["fut_ix"] = _by_symbol(_S["fut"]) if _S["fut"] is not None else {}
        _S["agg"] = cut(saved["agg"]); _S["agg_ix"] = _by_symbol(_S["agg"]) if _S["agg"] is not None else {}
        _S["chains"] = {}
        for k in list(saved["chains"]) + list(INDICES):
            if saved["chains"].get(k) is None:
                _chain_table(k)
        full = {**saved["chains"], **_S["chains"]}
        _S["chains"] = {k: cut(v) for k, v in full.items()}
        _S["daily_fut"], _S["daily_opt"] = {}, {}
        _S["part_oi"], _S["part_vol"] = cut(saved["part_oi"]), cut(saved["part_vol"])
        _S["ban"] = {k: v for k, v in saved["ban"].items() if k <= day}
        _S["fii"] = {k: cut(v) for k, v in saved["fii"].items()}
        _S["spot_idx"] = {k: (v[0][v[0] < day], v[1][v[0] < day]) for k, v in saved["spot_idx"].items()}
        yield
    finally:
        _S.update(saved)


def make_simulator(calendar: list, mult: float = 1.0, audit: bool = True, details: dict | None = None,
                   hist_lots: bool = False):
    def simulate(specs, lo, hi):
        out = {}
        for sp in specs:
            try:
                trs, sk, la = [], 0, None
                for sym in sp.underlyings:
                    t, det, s = _simulate_one(sp, sym, calendar, lo, hi, mult, hist_lots)
                    trs += t
                    sk += s
                    if details is not None:
                        for x in det:
                            details.setdefault(sp.id, {})[(x["date"], sym)] = x
                    if audit and la is None:
                        la = audit_lookahead(sp, sym, calendar, t)
                out[sp.id] = {"trades": sorted(trs, key=lambda x: (x[2], x[1])), "skipped": sk, "lookahead": la,
                              "error": None}
            except Exception as exc:
                out[sp.id] = {"trades": [], "skipped": 0, "lookahead": None, "error": repr(exc)[:300]}
        return out
    return simulate


def lab_calendar(data_dir: str) -> list:
    """The session calendar options_vrp.py uses (NIFTY minute sessions with >= MIN_SESSION_BARS bars): the same
    DISCOVERY / FINAL split for every option strategy."""
    from .options_vrp import load, calendar_of
    bars, _vix = load(data_dir)
    return calendar_of(bars)


def run_option_specs(specs: list, data_dir: str, out_dir: str, cache_dir: str = "", screen_only: bool = False) -> list:
    """Judge OptionSpecs with protocol.evaluate (each spec separately: n_symbols = its own underlyings, so its
    FINAL-ledger key does not depend on which other specs ran). Writes summary.txt, results.csv, trades/<id>.csv
    (DISCOVERY trades; FINAL ones only for screen passers). DISCOVERY-only sensitivity: haircut x2.5."""
    from . import data as D
    from . import protocol as P
    out = D.assert_outside_repo(out_dir)
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    cache = D.assert_outside_repo(cache_dir or D.default_cache_dir(data_dir))
    os.makedirs(cache, exist_ok=True)
    cal = lab_calendar(data_dir)
    split = P.split_sessions(cal)
    ledger = os.path.join(cache, "final_ledger.json")
    results, rows, details = [], [], {}
    L = [f"NSE EOD OPTIONS  data={os.path.realpath(data_dir)}  nse={NSE_DIR}",
         f"sessions={len(cal)} ({cal[0]}..{cal[-1]})  DISCOVERY {split.discovery[0]}..{split.discovery[1]} "
         f"(H1 to {split.h1[1]}, H2 from {split.h2[0]})  FINAL {split.final[0]}..{split.final[1]}",
         f"cost model {cost_model()}: fills at the EOD close, haircut max({HAIRCUT_MIN}, {HAIRCUT_PCT:.0%} of price) per "
         f"leg per fill, options_replay.leg_cost fees; expiry = intrinsic vs final settlement, no haircut.", ""]
    for sp in specs:
        with open(ledger + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            res, trades = P.evaluate([sp], cal, make_simulator(cal, details=details), n_symbols=len(sp.underlyings),
                                     ledger_path=ledger, dataset=os.path.realpath(data_dir), screen_only=screen_only,
                                     cost_model=cost_model())
        r = res[0]
        results.append(r)
        sens = make_simulator(cal, mult=SENS_MULT, audit=False)([sp], *split.discovery).get(sp.id, {})
        sc5 = P.screen(sens.get("trades", []), split, sp.horizon, single_underlying=bool(sp.underlying))
        hist = make_simulator(cal, audit=False, hist_lots=True)([sp], *split.discovery).get(sp.id, {})
        sc = r["screen"] or {}
        d = sc.get("disc", {})
        L.append(f"{sp.id}  ({sp.desc}; underlyings {','.join(sp.underlyings)})")
        L.append(f"  VERDICT {r['verdict']}: {r['why']}" + (f" | {r['final_note']}" if r.get("final_note") else "")
                 + f"   [entries skipped (untraded strike): {r['skipped']}]")
        if sc:
            L.append(f"    DISC n={d.get('trades')} net ₹{d.get('net', 0):,.0f} PF {d.get('pf')} win {d.get('win_pct')}%  "
                     f"H1 ₹{sc['h1']['net']:,.0f}  H2 ₹{sc['h2']['net']:,.0f}  top symbol {sc.get('top_symbol')} "
                     f"({sc.get('top_share', float('nan')):.0%})")
            s5 = P.stats(sens.get("trades", []))
            L.append(f"    net@5 (haircut x{SENS_MULT:g}) DISCOVERY ₹{s5['net']:,.0f} PF {s5['pf']} -> screen "
                     f"{'PASS' if sc5['passed'] else 'FAIL: ' + ', '.join(sc5['reasons'])}")
            sh = P.stats(hist.get("trades", []))
            L.append(f"    historical exchange lots DISCOVERY ₹{sh['net']:,.0f} PF {sh['pf']} (sensitivity only)")
        if r.get("final"):
            f = r["final"]
            L.append(f"    FINAL n={f['trades']} net ₹{f['net']:,.0f} PF {f['pf']} win {f['win_pct']}%")
        L.append("")
        fv = r.get("final")
        det = details.get(sp.id, {})
        with open(os.path.join(out, "trades", sp.id + ".csv"), "w", newline="") as fh:
            keys = list(next(iter(det.values())).keys()) if det else ["date"]
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for k in sorted(det):
                if k[0] <= split.discovery[1] or fv is not None:
                    w.writerow(det[k])
        rows.append({"id": sp.id, "verdict": r["verdict"], "why": r["why"], "final_note": r.get("final_note", ""),
                     "disc_trades": d.get("trades"), "disc_net": d.get("net"), "disc_pf": d.get("pf"),
                     "h1_net": sc.get("h1", {}).get("net"), "h2_net": sc.get("h2", {}).get("net"),
                     "sens_disc_net": P.stats(sens.get("trades", []))["net"], "sens_screen_passed": sc5["passed"],
                     "hist_lot_disc_net": P.stats(hist.get("trades", []))["net"],
                     "final_trades": (fv or {}).get("trades", ""), "final_net": (fv or {}).get("net", ""),
                     "final_pf": (fv or {}).get("pf", ""), "cost_model": cost_model()})
    L.append(P.luck_reminder(results))
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text)
    if rows:
        with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(text)
    return results


# ══════════════════════════════════════════════════════════════════════════
# coverage report / CLI
# ══════════════════════════════════════════════════════════════════════════

def coverage() -> str:
    _load()
    L = []
    dv = _S["dv"]
    L.append(f"delivery: {len(_S['dv_ix'])} bars symbols, {len(np.unique(dv['date']))} sessions "
             f"{dv['date'].min()}..{dv['date'].max()}")
    fo = _S["fo_days"]
    L.append(f"F&O bhavcopy: {len(fo)} sessions {fo.min() if len(fo) else '-'}..{fo.max() if len(fo) else '-'}; "
             f"futures symbols {len(_S['fut_ix'])}; stock option aggregates {len(_S['agg_ix'])}")
    for k in INDICES:
        t = _chain_table(k)
        if t is not None:
            L.append(f"  {k} options: {len(t['date'])} rows, {len(np.unique(t['date']))} sessions")
    nstk = len(glob.glob(os.path.join(_cache_dir(), "optstk", "*.npz")))
    L.append(f"  per-strike stock options: {nstk} symbols")
    for kind in ("oi", "vol"):
        t = _S.get(f"part_{kind}")
        if t is not None:
            L.append(f"participant {kind}: {len(t['date'])} sessions {t['date'].min()}..{t['date'].max()}")
    L.append(f"ban: {len(_S['ban_days'])} ban files, {sum(len(v) for v in _S['ban'].values())} (date, symbol) rows")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.nsedata")
    ap.add_argument("--check", action="store_true", help="print data coverage")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the npz cache")
    ap.add_argument("--data", default=BARS_DIR)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--specs", default="", help="module[:LIST] holding OptionSpec objects (default name SPECS)")
    ap.add_argument("--ids", default="")
    ap.add_argument("--screen-only", action="store_true")
    a = ap.parse_args(argv)
    if a.rebuild:
        _load(force=True)
    if a.check or not a.specs:
        print(coverage())
        return 0
    mod, _, name = a.specs.partition(":")
    specs = list(getattr(importlib.import_module(mod), name or "SPECS"))
    if a.ids:
        want = {x.strip() for x in a.ids.split(",") if x.strip()}
        specs = [s for s in specs if s.id in want]
    if not a.out:
        ap.error("--out is required to run specs")
    run_option_specs(specs, a.data, a.out, a.cache, a.screen_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
