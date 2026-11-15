"""
stratlab/optmin.py — 1-minute bars of NSE option and futures CONTRACTS (not continuous series), look-ahead-safe
by construction.  DATA ONLY: nothing here trades, simulates or judges a strategy.

DATA (OPTMIN_DIR, default <scratchpad>/databuild/optmin, override with env STRATLAB_OPTMIN_DIR; built by
<scratchpad>/databuild/scripts/{fetch_optmin.py, build_optmin.py}; data_check.json next to it)
  contracts.csv        one row per contract: tradingsymbol, file, kind (FUT|OPT), underlying, expiry, strike,
                       type (FUT|CE|PE), lot, tick, first_day, first_bar_day, last_bar_day, bar_days, bars
  bars/<file>.csv.gz   date, minute (minute of day of the bar START; 09:15 = 555), open, high, low, close, volume
                       (units, not contracts).  A minute without a trade has NO bar.  No open interest.
  daily.csv.gz         Kite daily candles per contract (used by the data check)
Coverage is what Kite still served on the fetch date: contracts LISTED on 2026-11-14 only (expired contracts are
gone), each from its first listed day.  NIFTY / BANKNIFTY options within +-8 % of the spot of the fetch date,
NIFTY / BANKNIFTY futures, current-month stock futures of the lab's symbols.  Kite's contract history is not
adjusted for anything; stock-future prices are as traded.

AVAILABILITY (the one rule): a bar that starts at minute m is complete at m+1.  A decision taken at time
(date, minute) may use every bar that STARTS strictly before `minute` on `date`, and all earlier sessions.
Every reader below takes that pair and returns nothing later:
    bars(sym, date, minute)     all bars strictly before (date, minute); minute=None -> sessions before `date` only
    session(sym, date, minute)  the bars of `date` that start before `minute`
    last(sym, date, minute)     the last such bar (the latest known price and how stale it is)
    chain(und, expiry, date, minute)   last known price of every strike of an expiry at that instant
fill(sym, date, minute) returns the bar that STARTS at (date, minute): it is for FILLING an order decided at the
close of the previous bar (the lab convention: decide at the close of bar i, fill at the open of bar i+1) and for
marking a held position inside a simulator, never for a decision at that minute.
Guards: inside optmin.cutoff(date, minute) any read of a bar starting at or after that instant raises
LookAheadError (readers with a later (date, minute) are clipped to it); nsedata.cutoff(day) is honoured too
(nothing dated >= day).

    cd algotrader_v4
    python -m stratlab.optmin --check
"""
from __future__ import annotations

import contextlib
import csv
import gzip
import os
import sys
from collections import OrderedDict

import numpy as np

from . import nsedata as _N
from .nsedata import LookAheadError, Rows

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
OPTMIN_DIR = os.environ.get("STRATLAB_OPTMIN_DIR", os.path.join(_SCRATCH, "databuild", "optmin"))
SESSION_OPEN = 9 * 60 + 15
DAY_END = 24 * 60
FIELDS = ("date", "t", "o", "h", "l", "c", "v")
_CACHE_MAX = 600
_S: dict = {}
_BARS: "OrderedDict[str, dict]" = OrderedDict()
_CUT: list = []           # stack of (date, minute) cutoffs


def use(optmin_dir: str):
    """Point the module at another data directory (tests) and drop the caches."""
    global OPTMIN_DIR
    OPTMIN_DIR = optmin_dir
    _S.clear()
    _BARS.clear()


def _key(date: int, minute: int) -> int:
    return int(date) * 10000 + int(minute)


@contextlib.contextmanager
def cutoff(date: int, minute: int = 0):
    """Inside this block no bar starting at or after (date, minute) can be read."""
    _CUT.append(_key(date, minute))
    try:
        yield
    finally:
        _CUT.pop()


def _limit() -> int | None:
    lims = list(_CUT)
    if _N._CUT:
        lims.append(_key(min(_N._CUT), 0))
    return min(lims) if lims else None


def _clip(date: int, minute: int | None) -> int:
    k = _key(date, 0 if minute is None else minute)
    lim = _limit()
    return k if lim is None else min(k, lim)


# ══════════════════════════════════════════════════════════════════════════
# contracts
# ══════════════════════════════════════════════════════════════════════════

def _load():
    if "contracts" in _S:
        return
    rows = []
    p = os.path.join(OPTMIN_DIR, "contracts.csv")
    if os.path.exists(p):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                r["expiry"] = int(r["expiry"])
                r["strike"] = float(r["strike"])
                r["lot"] = int(float(r["lot"] or 0))
                for k in ("first_day", "first_bar_day", "last_bar_day", "bar_days", "bars"):
                    r[k] = int(r[k]) if r.get(k) not in (None, "") else 0
                rows.append(r)
    _S["contracts"] = rows
    _S["by_sym"] = {r["tradingsymbol"]: r for r in rows}
    _S["by_key"] = {(r["underlying"], r["expiry"], round(r["strike"], 2), r["type"]): r for r in rows}


def contracts(underlying: str | None = None, kind: str | None = None, expiry: int | None = None,
              typ: str | None = None) -> list:
    """Contract rows (dicts) filtered by underlying / kind ("FUT" | "OPT") / expiry (yyyymmdd) / type.
    The list itself is exchange reference data (strikes and expiries are listed before they trade); a contract's
    `first_day` tells from when it existed."""
    _load()
    return [dict(r) for r in _S["contracts"]
            if (underlying is None or r["underlying"] == underlying) and (kind is None or r["kind"] == kind)
            and (expiry is None or r["expiry"] == int(expiry)) and (typ is None or r["type"] == typ)]


def expiries(underlying: str, kind: str = "OPT") -> list:
    _load()
    return sorted({r["expiry"] for r in _S["contracts"] if r["underlying"] == underlying and r["kind"] == kind})


def symbol(underlying: str, expiry: int, strike: float = 0.0, typ: str = "FUT") -> str | None:
    """Tradingsymbol of (underlying, expiry, strike, type) or None if that contract is not on disk."""
    _load()
    r = _S["by_key"].get((underlying, int(expiry), round(float(strike), 2), typ))
    return r["tradingsymbol"] if r else None


def info(sym: str) -> dict | None:
    _load()
    r = _S["by_sym"].get(sym)
    return dict(r) if r else None


# ══════════════════════════════════════════════════════════════════════════
# bars
# ══════════════════════════════════════════════════════════════════════════

def _table(sym: str) -> dict | None:
    _load()
    if sym in _BARS:
        _BARS.move_to_end(sym)
        return _BARS[sym]
    r = _S["by_sym"].get(sym)
    if r is None:
        return None
    p = os.path.join(OPTMIN_DIR, "bars", r["file"] + ".csv.gz")
    cols = [[] for _ in range(7)]
    if os.path.exists(p):
        with gzip.open(p, "rt", newline="") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for x in rd:
                for i in range(7):
                    cols[i].append(x[i])
    t = {"date": np.array(cols[0], np.int64), "t": np.array(cols[1], np.int64),
         "o": np.array(cols[2], float), "h": np.array(cols[3], float), "l": np.array(cols[4], float),
         "c": np.array(cols[5], float), "v": np.array(cols[6], float)}
    t["key"] = t["date"] * 10000 + t["t"]
    _BARS[sym] = t
    while len(_BARS) > _CACHE_MAX:
        _BARS.popitem(last=False)
    return t


def _empty() -> Rows:
    return Rows({c: np.array([]) for c in FIELDS})


def _cut_rows(t: dict, a: int, b: int) -> Rows:
    return Rows({c: t[c][a:b].copy() for c in FIELDS})


def bars(sym: str, date: int, minute: int | None = None, n: int | None = None) -> Rows:
    """Bars of contract `sym` that start strictly before (date, minute), oldest first; minute=None: only sessions
    before `date`; n: keep the last n bars.  Columns: date, t (minute of day of the bar start), o, h, l, c, v."""
    t = _table(sym)
    if t is None:
        return _empty()
    k = int(np.searchsorted(t["key"], _clip(date, minute), side="left"))
    a = 0 if n is None else max(0, k - int(n))
    return _cut_rows(t, a, k)


def session(sym: str, date: int, minute: int | None = None) -> Rows:
    """The bars of session `date` that start before `minute` (minute=None: the whole session, which under an
    active cutoff is clipped like everything else)."""
    t = _table(sym)
    if t is None:
        return _empty()
    hi = _clip(date, DAY_END if minute is None else minute)
    a = int(np.searchsorted(t["key"], _key(date, 0), side="left"))
    b = int(np.searchsorted(t["key"], hi, side="left"))
    return _cut_rows(t, a, max(a, b))


def last(sym: str, date: int, minute: int, same_day: bool = True) -> dict | None:
    """The last bar that starts before (date, minute): {date, t, o, h, l, c, v, age} with age = minutes between
    that bar's END and `minute` (0 = the bar just closed); None if there is none (same_day=True: on `date` only).
    Illiquid strikes trade rarely: check `age` before treating c as a current price."""
    t = _table(sym)
    if t is None:
        return None
    k = int(np.searchsorted(t["key"], _clip(date, minute), side="left")) - 1
    if k < 0 or (same_day and int(t["date"][k]) != int(date)):
        return None
    out = {c: (int(t[c][k]) if c in ("date", "t") else float(t[c][k])) for c in FIELDS}
    out["age"] = (int(minute) - out["t"] - 1) if out["date"] == int(date) else None
    return out


def fill(sym: str, date: int, minute: int) -> dict | None:
    """The bar that STARTS at (date, minute) — for filling / marking inside a simulator, never for a decision at
    that minute.  Raises LookAheadError under a cutoff at or before that instant; None if no trade that minute."""
    lim = _limit()
    if lim is not None and _key(date, minute) >= lim:
        raise LookAheadError(f"bar {sym} {date} {minute // 60:02d}:{minute % 60:02d} read while only bars before "
                             f"{lim // 10000} {lim % 10000 // 60:02d}:{lim % 10000 % 60:02d} are allowed")
    t = _table(sym)
    if t is None:
        return None
    k = int(np.searchsorted(t["key"], _key(date, minute), side="left"))
    if k >= len(t["key"]) or int(t["key"][k]) != _key(date, minute):
        return None
    return {c: (int(t[c][k]) if c in ("date", "t") else float(t[c][k])) for c in FIELDS}


def sessions(sym: str, before: int | None = None) -> list:
    """Dates on which the contract has at least one bar (optionally strictly before `before`)."""
    t = _table(sym)
    if t is None:
        return []
    k = int(np.searchsorted(t["key"], _clip(99991231 if before is None else before, None), side="left"))
    return [int(x) for x in np.unique(t["date"][:k])]


def chain(underlying: str, expiry: int, date: int, minute: int, max_age: int | None = None) -> dict:
    """Last known price of every strike of one expiry at the instant (date, minute):
    {(strike, "CE"|"PE"): {"c": last close, "t": bar start minute, "age": minutes since that bar closed, "v": its
    volume, "symbol": tradingsymbol}} from same-day bars that start before `minute`; strikes with no trade yet
    today (or older than max_age minutes, if given) are omitted."""
    _load()
    out = {}
    for r in _S["contracts"]:
        if r["underlying"] != underlying or r["expiry"] != int(expiry) or r["kind"] != "OPT":
            continue
        b = last(r["tradingsymbol"], date, minute)
        if b is None or (max_age is not None and b["age"] > max_age):
            continue
        out[(r["strike"], r["type"])] = {"c": b["c"], "t": b["t"], "age": b["age"], "v": b["v"],
                                         "symbol": r["tradingsymbol"]}
    return out


# ══════════════════════════════════════════════════════════════════════════
# coverage report
# ══════════════════════════════════════════════════════════════════════════

def coverage() -> str:
    _load()
    cs = _S["contracts"]
    lines = [f"optmin directory: {OPTMIN_DIR}", f"contracts: {len(cs)}"]
    if not cs:
        return "\n".join(lines + ["(no contracts.csv)"])
    groups: dict = {}
    for r in cs:
        k = (r["kind"], r["underlying"] if r["kind"] == "OPT" or r["underlying"] in ("NIFTY", "BANKNIFTY") else "stocks",
             r["expiry"])
        g = groups.setdefault(k, [0, 0, 99999999, 0])
        g[0] += 1
        g[1] += r["bars"]
        if r["first_bar_day"]:
            g[2] = min(g[2], r["first_bar_day"])
        g[3] = max(g[3], r["last_bar_day"])
    for (kind, und, exp), g in sorted(groups.items()):
        lines.append(f"  {kind} {und:<10} expiry {exp}: {g[0]:>4} contracts {g[1]:>9,} bars  {g[2]} .. {g[3]}")
    return "\n".join(lines)


def main(argv=None) -> int:
    print(coverage())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
