"""
stratlab/data.py — load 1-minute bars into numpy arrays grouped by session, and derive daily bars.

Two on-disk layouts are read (auto-detected per file):

  one symbol per file   <SYMBOL>.json = {"symbol": .., "interval": .., "bars": [[iso_ts, o, h, l, c, v], ...]}
                        files starting with "_" are indices / VIX (``_NIFTY50_min.json``, ``_VIX_day.json`` ...)
  many symbols per file <anything>.json = {SYMBOL: {"bars": [[ts, o, h, l, c, v], ...]}, ...}

Parsed arrays are cached as ``.npz`` (one per symbol) in a cache directory OUTSIDE the repository, keyed by the
source file's size and mtime, so a directory that is still downloading is re-parsed only where it changed.
Only bars from 09:15 to 15:29 IST are kept. Everything is processed one symbol at a time.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

SESSION_OPEN = 9 * 60 + 15      # 09:15 as minute of day
SESSION_LAST = 15 * 60 + 29     # 15:29, the last 1-minute bar of the session
MIN_SESSION_BARS = 200          # shorter sessions (special / broken days) are dropped

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hhmm(s: str) -> int:
    """'10:00' -> 600 (minute of day)."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def assert_outside_repo(path: str) -> str:
    """Caches and results never go inside the repository."""
    p = os.path.realpath(path)
    if p == _REPO_ROOT or p.startswith(_REPO_ROOT + os.sep):
        raise ValueError(f"{path} is inside the repository; use a scratch directory")
    return p


def default_cache_dir(data_dir: str) -> str:
    base = os.environ.get("STRATLAB_CACHE")
    d = os.path.realpath(data_dir)
    if not base:
        base = os.path.join(os.path.dirname(d), "stratlab_cache")
    tag = os.path.basename(d) + "_" + hashlib.md5(d.encode()).hexdigest()[:6]
    return assert_outside_repo(os.path.join(base, tag))


@dataclass
class SymbolData:
    """Flat minute arrays for one symbol plus the offsets of each session."""
    symbol: str
    date: np.ndarray        # int32 yyyymmdd per bar
    t: np.ndarray           # int16 minute of day per bar
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    starts: np.ndarray      # index of the first bar of each session
    ends: np.ndarray        # one past the last bar of each session
    days: np.ndarray        # int32 yyyymmdd per session
    dropped: list = None    # dates of sessions dropped for being too short

    @property
    def n_sessions(self) -> int:
        return len(self.days)

    def daily(self) -> dict:
        """Daily bars derived from the minute bars (one row per session)."""
        s = self.starts
        return {
            "date": self.days.copy(),
            "o": self.o[s].copy(),
            "h": np.maximum.reduceat(self.h, s),
            "l": np.minimum.reduceat(self.l, s),
            "c": self.c[self.ends - 1].copy(),
            "v": np.add.reduceat(self.v, s),
            "nbars": (self.ends - self.starts).astype(np.int32),
        }


def _parse_bars(bars) -> dict | None:
    """[[iso_ts, o, h, l, c, v], ...] -> raw arrays (unsorted, unfiltered)."""
    n = len(bars)
    if n == 0:
        return None
    date = np.empty(n, np.int32)
    t = np.empty(n, np.int16)
    px = np.empty((n, 5), np.float64)
    for i, b in enumerate(bars):
        ts = b[0]                                   # 'YYYY-MM-DDTHH:MM:SS[+05:30]' (IST wall clock)
        date[i] = int(ts[0:4]) * 10000 + int(ts[5:7]) * 100 + int(ts[8:10])
        t[i] = int(ts[11:13]) * 60 + int(ts[14:16]) if len(ts) >= 16 else 0
        px[i, 0] = b[1]; px[i, 1] = b[2]; px[i, 2] = b[3]; px[i, 3] = b[4]
        px[i, 4] = b[5] if len(b) > 5 and b[5] is not None else 0.0
    return {"date": date, "t": t, "px": px}


def _clean_minute(p: dict) -> dict | None:
    """Keep session hours and positive finite prices; sort; drop duplicate minutes."""
    keep = (p["t"] >= SESSION_OPEN) & (p["t"] <= SESSION_LAST)
    keep &= np.isfinite(p["px"][:, :4]).all(axis=1) & (p["px"][:, :4] > 0).all(axis=1)
    date, t, px = p["date"][keep], p["t"][keep], p["px"][keep]
    if len(date) == 0:
        return None
    key = date.astype(np.int64) * 10000 + t
    order = np.argsort(key, kind="stable")
    key, date, t, px = key[order], date[order], t[order], px[order]
    uniq = np.ones(len(key), bool)
    uniq[:-1] = key[1:] != key[:-1]                 # keep the last of any duplicate minute
    date, t, px = date[uniq], t[uniq], px[uniq]
    return {"date": date, "t": t, "o": px[:, 0], "h": px[:, 1], "l": px[:, 2], "c": px[:, 3], "v": px[:, 4]}


def _cache_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(" ", "_") + ".npz"


def prepare(data_dir: str, cache_dir: str | None = None, verbose: bool = False) -> dict:
    """Parse every changed JSON file in `data_dir` into the cache. Returns the manifest
    {"symbols": {SYM: file}, "aux": {NAME: file}, "problems": [...], "cache_dir": ...}. Safe to call repeatedly."""
    cache_dir = assert_outside_repo(cache_dir or default_cache_dir(data_dir))
    os.makedirs(cache_dir, exist_ok=True)
    mpath = os.path.join(cache_dir, "_manifest.json")
    try:
        with open(mpath) as fh:
            man = json.load(fh)
    except Exception:
        man = {}
    man.setdefault("files", {}); man.setdefault("symbols", {}); man.setdefault("aux", {})
    problems = []
    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    for gone in set(man["files"]) - set(names):
        man["files"].pop(gone, None)
    for fn in names:
        path = os.path.join(data_dir, fn)
        st = os.stat(path)
        sig = [st.st_size, int(st.st_mtime)]
        if man["files"].get(fn, {}).get("sig") == sig:
            continue
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception as exc:                    # half-written download: picked up on a later run
            problems.append(f"{fn}: unreadable ({str(exc)[:60]})")
            continue
        if isinstance(d, dict) and isinstance(d.get("bars"), list):
            items = {str(d.get("symbol") or fn[:-5]): d["bars"]}
            single = True
        elif isinstance(d, dict):
            items = {k: v.get("bars", []) for k, v in d.items() if isinstance(v, dict)}
            single = False
        else:
            problems.append(f"{fn}: unknown layout")
            continue
        wrote = []
        for sym, bars in items.items():
            p = _parse_bars(bars)
            if p is None:
                problems.append(f"{fn}: {sym} has no bars")
                continue
            if single and fn.startswith("_"):
                name = fn[:-5]                      # _NIFTY50_min / _VIX_day
                if name.endswith("_day"):
                    out = {"date": p["date"], "o": p["px"][:, 0], "h": p["px"][:, 1], "l": p["px"][:, 2],
                           "c": p["px"][:, 3], "v": p["px"][:, 4]}
                else:
                    out = _clean_minute(p)
                if out is None:
                    problems.append(f"{fn}: no usable bars")
                    continue
                np.savez(os.path.join(cache_dir, _cache_name(name)), **out)
                man["aux"][name] = fn
                wrote.append(name)
                continue
            out = _clean_minute(p)
            if out is None:
                problems.append(f"{fn}: {sym} has no bars in session hours")
                continue
            sym = fn[:-5] if single else sym        # one-symbol files are named after the trading symbol
            if man["symbols"].get(sym, fn) != fn:
                problems.append(f"{sym}: present in both {man['symbols'][sym]} and {fn}; using {fn}")
            np.savez(os.path.join(cache_dir, _cache_name(sym)), **out)
            man["symbols"][sym] = fn
            wrote.append(sym)
        man["files"][fn] = {"sig": sig, "wrote": wrote, "problems": [x for x in problems if x.startswith(fn + ":")]}
        if verbose:
            print(f"  parsed {fn}: {', '.join(wrote)[:80]}", flush=True)
    live = set(man["files"])
    man["symbols"] = {s: f for s, f in man["symbols"].items() if f in live}
    man["aux"] = {s: f for s, f in man["aux"].items() if f in live}
    tmp = mpath + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(man, fh)
    os.replace(tmp, mpath)
    man["problems"] = sorted(set(problems) | {x for f in man["files"].values() for x in f.get("problems", [])})
    man["cache_dir"] = cache_dir
    return man


def _bounds(date):
    cut = np.flatnonzero(date[1:] != date[:-1]) + 1
    return np.concatenate(([0], cut)).astype(np.int64), np.concatenate((cut, [len(date)])).astype(np.int64)


def _sessionize(symbol: str, z, min_bars: int = MIN_SESSION_BARS) -> SymbolData:
    """Group bars by day. Sessions with fewer than `min_bars` bars (Diwali muhurat hour, a half-downloaded
    day) are dropped completely - from the minute arrays and from the derived daily bars - and listed in
    `dropped`, so "yesterday" always means the last full session."""
    date = z["date"]
    starts, ends = _bounds(date)
    short = (ends - starts) < min_bars
    dropped = [int(x) for x in date[starts[short]]]
    if short.any():
        keep = np.repeat(~short, ends - starts)
        z = {k: v[keep] for k, v in z.items()}
        date = z["date"]
        starts, ends = _bounds(date) if len(date) else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    return SymbolData(symbol, date, z["t"], z["o"], z["h"], z["l"], z["c"], z["v"], starts, ends,
                      date[starts].astype(np.int32), dropped)


def load_symbol(cache_dir: str, symbol: str) -> SymbolData | None:
    path = os.path.join(cache_dir, _cache_name(symbol))
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        sd = _sessionize(symbol, {k: z[k] for k in z.files})
    return sd if sd.n_sessions else None


def load_daily_aux(cache_dir: str, name: str) -> dict | None:
    """A daily series such as '_VIX_day' or '_NIFTY50_day' -> {"date","o","h","l","c","v"}."""
    path = os.path.join(cache_dir, _cache_name(name))
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def from_arrays(symbol: str, date, t, o, h, l, c, v, min_bars: int = 1) -> SymbolData:
    """Build a SymbolData from in-memory arrays (tests, synthetic data)."""
    z = {"date": np.asarray(date, np.int32), "t": np.asarray(t, np.int16)}
    for k, a in (("o", o), ("h", h), ("l", l), ("c", c), ("v", v)):
        z[k] = np.asarray(a, np.float64)
    return _sessionize(symbol, z, min_bars)


def quality_report(sd: SymbolData) -> dict:
    """Simple per-symbol data checks, shown in the run summary."""
    n = sd.ends - sd.starts
    bad = int(((sd.h < sd.l) | (sd.h < np.maximum(sd.o, sd.c)) | (sd.l > np.minimum(sd.o, sd.c))).sum())
    last_t = sd.t[sd.ends - 1]
    d = sd.daily()
    jump = np.abs(d["o"][1:] / d["c"][:-1] - 1.0)
    return {
        "symbol": sd.symbol, "sessions": int(len(n)), "bars": int(n.sum()),
        "first": int(sd.days[0]), "last": int(sd.days[-1]),
        "short_sessions": list(sd.dropped or []),
        "ends_before_1515": int((last_t < 15 * 60 + 15).sum()),
        "missing_minutes": int(((last_t - sd.t[sd.starts] + 1) - n).sum()),
        "zero_volume_bars": int((sd.v <= 0).sum()),
        "bad_ohlc_bars": bad,
        "overnight_jumps_gt_15pct": [int(x) for x in d["date"][1:][jump > 0.15]],
    }
