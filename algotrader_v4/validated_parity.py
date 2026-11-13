"""
validated_parity.py — test harness: rebuild the lab's days from its minute-bar data set and run them through the live
validated agents' decision code (validated_data.summarize_sessions → the session store functions → plan_*).

Used only by test_pipeline.py's lab-parity acceptance tests (skipped when the data set is absent). The stratlab
loader is the I/O layer only: every session summary the live functions read is produced by
validated_data.summarize_sessions from Kite-shaped candle dicts, exactly as the live store is built.
"""
from __future__ import annotations

import bisect
import hashlib
import inspect
import json
import os
from datetime import date, datetime

import numpy as np

import validated_data as VD
from stratlab import data as D

PARITY_BARS = os.getenv("VALIDATED_PARITY_BARS",
                        "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad/bars3y")
PARITY_LAB = os.getenv("VALIDATED_PARITY_LAB",
                       "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad/dd")


def available(*rel) -> bool:
    return os.path.isdir(PARITY_BARS) and all(os.path.exists(os.path.join(PARITY_LAB, r)) for r in rel)


def to_date(k: int) -> date:
    return date(k // 10000, k // 100 % 100, k % 100)


def candles(z: dict, lo: int = 0, hi: int | None = None) -> list[dict]:
    """Minute arrays → Kite historical candle dicts ({"date": datetime, open, high, low, close, volume})."""
    hi = len(z["date"]) if hi is None else hi
    dd, tt = z["date"][lo:hi].tolist(), z["t"][lo:hi].tolist()
    o, h, l, c, v = (z[k][lo:hi].tolist() for k in ("o", "h", "l", "c", "v"))
    return [{"date": datetime(d // 10000, d // 100 % 100, d % 100, t // 60, t % 60), "open": o[i], "high": h[i],
             "low": l[i], "close": c[i], "volume": v[i]} for i, (d, t) in enumerate(zip(dd, tt))]


def raw_arrays(cache_dir: str, symbol: str) -> dict:
    """The cached cleaned minute arrays of a symbol INCLUDING short sessions (what Kite would return)."""
    with np.load(os.path.join(cache_dir, D._cache_name(symbol))) as z:
        return {k: z[k] for k in z.files}


def build_summaries(cache_dir: str, symbols, manifest: dict, verify: int = 3) -> dict:
    """{symbol: [SessionSummary]} through validated_data.summarize_sessions, cached outside the repo (keyed by the
    data files and the function's source). `verify` symbols are always recomputed and compared with the cache."""
    key = hashlib.sha1((inspect.getsource(VD.summarize_sessions) + json.dumps(manifest.get("files", {}),
                                                                             sort_keys=True)).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"_validated_summaries_{key}.json")
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
        out = {s: [VD.SessionSummary(*r) for r in rows] for s, rows in raw.items()}
        idx = np.linspace(0, len(symbols) - 1, verify).astype(int) if verify else []
        for i in idx:
            s = symbols[int(i)]
            fresh = VD.summarize_sessions(candles(raw_arrays(cache_dir, s)))
            assert fresh == out.get(s), f"cached session summaries of {s} differ from summarize_sessions"
        if set(out) >= set(symbols):
            return out
    for s in symbols:
        if s not in out:
            out[s] = VD.summarize_sessions(candles(raw_arrays(cache_dir, s)))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({s: [list(x.__dict__.values()) for x in v] for s, v in out.items()}, fh)
    os.replace(tmp, path)
    return out


class Store:
    """The live store as it stands on the morning of a day: the last KEEP_SESSIONS valid sessions before it."""

    def __init__(self, summaries: dict):
        self.valid = {s: VD.valid_sessions(v) for s, v in summaries.items()}
        self.days = {s: [x.day for x in v] for s, v in self.valid.items()}

    def before(self, symbol: str, day: int) -> list:
        i = bisect.bisect_left(self.days.get(symbol, []), day)
        return self.valid.get(symbol, [])[max(0, i - VD.KEEP_SESSIONS):i]


def load(lab_rel: str):
    """Data set → (manifest, cache_dir, symbols, summaries, calendar of all dates incl. dropped ones)."""
    man = D.prepare(PARITY_BARS)
    cdir = man["cache_dir"]
    syms = [s for s in VD.UNIVERSE if s in man["symbols"]]
    summ = build_summaries(cdir, syms, man)
    cal = sorted({x.day for v in summ.values() for x in v})
    return man, cdir, syms, summ, cal
