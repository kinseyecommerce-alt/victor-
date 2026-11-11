"""
stratlab/indicators.py — indicators computed once per symbol and handed to strategies as arrays.

Everything here is causal: the value at bar i uses bars 0..i only (EMA / RSI / ATR run continuously across
sessions; VWAP, its bands and the day's high/low restart at 09:15). `build_sessions()` returns one `Session`
per trading day; `View(session, n)` is the same object truncated to its first n bars, which is what a
strategy is allowed to know at the close of bar n-1.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd

from .data import SESSION_OPEN, SymbolData

Bars5 = namedtuple("Bars5", "o h l c v end")     # completed 5-minute bars; end = index of each bar's last minute


# ── plain array helpers (also for daily strategies) ─────────────────────────
def ema(x, n: int) -> np.ndarray:
    return pd.Series(np.asarray(x, float)).ewm(span=n, adjust=False).mean().to_numpy()


def sma(x, n: int) -> np.ndarray:
    return pd.Series(np.asarray(x, float)).rolling(n).mean().to_numpy()


def highest(x, n: int) -> np.ndarray:
    return pd.Series(np.asarray(x, float)).rolling(n).max().to_numpy()


def lowest(x, n: int) -> np.ndarray:
    return pd.Series(np.asarray(x, float)).rolling(n).min().to_numpy()


def rsi(c, n: int = 14) -> np.ndarray:
    """Wilder RSI. The first value is NaN."""
    d = pd.Series(np.asarray(c, float)).diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    out = 100.0 - 100.0 / (1.0 + up / dn.replace(0.0, np.nan))
    out[(dn == 0) & (up > 0)] = 100.0
    return out.to_numpy()


def true_range(h, l, c, first=None) -> np.ndarray:
    """True range; bars flagged in `first` (session opens) use high-low so an overnight gap is not counted."""
    h, l, c = np.asarray(h, float), np.asarray(l, float), np.asarray(c, float)
    pc = np.empty_like(c)
    pc[0] = c[0]; pc[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    if first is not None:
        tr[first] = (h - l)[first]
    return tr


def atr(h, l, c, n: int = 14, first=None) -> np.ndarray:
    return pd.Series(true_range(h, l, c, first)).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


# ── the per-session object strategies receive ───────────────────────────────
class _Api:
    """Methods shared by Session and View. They only use the arrays visible on `self`."""

    def idx(self, minute: int) -> int:
        """Index of the first bar at or after `minute` (minute of day); len(self) if there is none yet."""
        return int(np.searchsorted(self.t, minute))

    def orb(self, minutes: int):
        """Opening range of the first `minutes` minutes: (high, low, k). The range is complete, and may be
        traded, from bar index k onwards (signals at i >= k - 1 see the whole range)."""
        k = self.idx(SESSION_OPEN + minutes)
        if k == 0:
            return np.nan, np.nan, 0
        return float(self.h[:k].max()), float(self.l[:k].min()), k

    def bars5(self) -> Bars5:
        """Completed 5-minute bars (09:15-09:19, 09:20-09:24, ...)."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.t)


_ARRAYS = ("t", "o", "h", "l", "c", "v", "vwap", "vsd", "ema9", "ema21", "ema50", "rsi", "atr",
           "e9_5", "e21_5", "e50_5", "rsi5", "atr5", "vr", "day_hi", "day_lo", "nifty")


class Session(_Api):
    """One symbol, one trading day. Arrays have one element per 1-minute bar.

    arrays   t (minute of day), o h l c v, vwap, vsd (volume-weighted std of price around vwap),
             ema9 ema21 ema50 rsi atr           on 1-minute bars
             e9_5 e21_5 e50_5 rsi5 atr5         on completed 5-minute bars, held until the next one completes
             vr (volume / mean volume of the previous 20 bars), day_hi day_lo (running, including this bar),
             nifty (NIFTY 50 close at the same minute; None when the index is not in the data)
    scalars  symbol, date (yyyymmdd), pdo pdh pdl pdc (previous day), gap (open / pdc - 1),
             datr (14-day ATR to yesterday), dsma20 (mean of the last 20 daily closes to yesterday),
             dvol20 (mean daily volume, last 20 days), nifty_pdc
    prev     daily history up to YESTERDAY: prev.o/.h/.l/.c/.v arrays, prev.c[-1] is yesterday's close
    """
    truncated = False

    def bars5(self) -> Bars5:
        return self._b5

    def view(self, n: int) -> "View":
        return View(self, n)


class View(_Api):
    """The first n bars of a session: exactly what is known at the close of bar n-1."""
    truncated = True

    def __init__(self, s: Session, n: int):
        object.__setattr__(self, "_s", s)
        object.__setattr__(self, "_n", n)

    def __getattr__(self, name):
        v = getattr(self._s, name)
        if isinstance(v, np.ndarray):
            return v[:self._n]
        return v

    def __setattr__(self, k, v):
        raise AttributeError("views are read-only")

    def __len__(self) -> int:
        return self._n

    def bars5(self) -> Bars5:
        b = self._s._b5
        k = int(np.searchsorted(b.end, self._n - 1, side="right"))
        return Bars5(b.o[:k], b.h[:k], b.l[:k], b.c[:k], b.v[:k], b.end[:k])


class DailyHist:
    """Daily bars up to and including some day (arrays; index -1 is the most recent day)."""
    __slots__ = ("date", "o", "h", "l", "c", "v")

    def __init__(self, date, o, h, l, c, v):
        self.date, self.o, self.h, self.l, self.c, self.v = date, o, h, l, c, v

    def __len__(self):
        return len(self.c)


def _session_cum(x: np.ndarray, starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Cumulative sum that restarts at each session start."""
    cs = np.cumsum(x)
    base = np.concatenate(([0.0], cs))[starts]
    return cs - np.repeat(base, lens)


def build_sessions(sd: SymbolData, index: SymbolData | None = None) -> list[Session]:
    """All tradable sessions of one symbol with their indicator arrays (slices of shared flat arrays)."""
    N = len(sd.c)
    starts, ends = sd.starts, sd.ends
    lens = ends - starts
    o, h, l, c, v, t = sd.o, sd.h, sd.l, sd.c, sd.v, sd.t
    first = np.zeros(N, bool); first[starts] = True
    sess_of = np.repeat(np.arange(len(starts)), lens)

    # session VWAP and its volume-weighted standard deviation (equal weights if the bar has no volume data)
    w = np.where(v > 0, v, 0.0)
    if w.sum() == 0:
        w = np.ones(N)
    tp = (h + l + c) / 3.0
    cw = _session_cum(w, starts, lens)
    cw_safe = np.where(cw > 0, cw, 1.0)
    vwap = np.where(cw > 0, _session_cum(w * tp, starts, lens) / cw_safe, tp)
    var = np.where(cw > 0, _session_cum(w * tp * tp, starts, lens) / cw_safe, tp * tp) - vwap * vwap
    vsd = np.sqrt(np.maximum(var, 0.0))

    flat = {
        "t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "vwap": vwap, "vsd": vsd,
        "ema9": ema(c, 9), "ema21": ema(c, 21), "ema50": ema(c, 50),
        "rsi": rsi(c, 14), "atr": atr(h, l, c, 14, first),
    }

    # completed 5-minute bars, and indicators on them held forward on the 1-minute grid
    bid = sess_of.astype(np.int64) * 1000 + (t.astype(np.int64) - SESSION_OPEN) // 5
    bcut = np.flatnonzero(bid[1:] != bid[:-1]) + 1
    bst = np.concatenate(([0], bcut))
    bend = np.concatenate((bcut, [N])) - 1
    o5, c5 = o[bst], c[bend]
    h5, l5, v5 = np.maximum.reduceat(h, bst), np.minimum.reduceat(l, bst), np.add.reduceat(v, bst)
    first5 = first[bst]
    done = np.searchsorted(bend, np.arange(N), side="right") - 1     # last completed 5-min bar at each minute

    def hold(x5):
        out = np.full(N, np.nan)
        ok = done >= 0
        out[ok] = x5[done[ok]]
        return out

    flat["e9_5"], flat["e21_5"], flat["e50_5"] = hold(ema(c5, 9)), hold(ema(c5, 21)), hold(ema(c5, 50))
    flat["rsi5"] = hold(rsi(c5, 14))
    flat["atr5"] = hold(atr(h5, l5, c5, 14, first5))

    # volume relative to the previous 20 bars
    cv = np.concatenate(([0.0], np.cumsum(v)))
    i = np.arange(N)
    lo = np.maximum(i - 20, 0)
    prev_mean = (cv[i] - cv[lo]) / np.maximum(i - lo, 1)
    flat["vr"] = np.where(prev_mean > 0, v / np.where(prev_mean > 0, prev_mean, 1.0), np.nan)

    day_hi, day_lo = np.empty(N), np.empty(N)
    for a, b in zip(starts, ends):
        day_hi[a:b] = np.maximum.accumulate(h[a:b])
        day_lo[a:b] = np.minimum.accumulate(l[a:b])
    flat["day_hi"], flat["day_lo"] = day_hi, day_lo

    d = sd.daily()
    dtr = true_range(d["h"], d["l"], d["c"])
    idx_days = {int(x): k for k, x in enumerate(index.days)} if index is not None else {}
    b5_first = np.searchsorted(bst, starts)                 # first 5-min bar of each session
    b5_last = np.searchsorted(bst, ends)

    out: list[Session] = []
    for k in range(len(starts)):
        a, b = int(starts[k]), int(ends[k])
        s = Session()
        s.symbol, s.date, s.day_index = sd.symbol, int(sd.days[k]), k
        s._sd = sd                  # candles of EARLIER sessions (stratlab.candles); never read bars >= starts[k]
        for name, arr in flat.items():
            setattr(s, name, arr[a:b])
        s.prev = DailyHist(d["date"][:k], d["o"][:k], d["h"][:k], d["l"][:k], d["c"][:k], d["v"][:k])
        if k > 0:
            s.pdo, s.pdh, s.pdl, s.pdc = (float(d["o"][k - 1]), float(d["h"][k - 1]),
                                          float(d["l"][k - 1]), float(d["c"][k - 1]))
            s.gap = float(o[a] / s.pdc - 1.0)
        else:
            s.pdo = s.pdh = s.pdl = s.pdc = s.gap = np.nan
        s.datr = float(dtr[max(0, k - 14):k].mean()) if k >= 5 else np.nan
        s.dsma20 = float(d["c"][k - 20:k].mean()) if k >= 20 else np.nan
        s.dvol20 = float(d["v"][max(0, k - 20):k].mean()) if k >= 5 else np.nan
        f5, l5i = int(b5_first[k]), int(b5_last[k])
        s._b5 = Bars5(o5[f5:l5i], h5[f5:l5i], l5[f5:l5i], c5[f5:l5i], v5[f5:l5i], bend[f5:l5i] - a)
        s.nifty, s.nifty_pdc = None, np.nan
        ik = idx_days.get(s.date)
        if ik is not None:
            ia, ib = int(index.starts[ik]), int(index.ends[ik])
            pos = np.searchsorted(index.t[ia:ib], s.t, side="right") - 1   # index bar at or before each minute
            nf = np.where(pos >= 0, index.c[ia:ib][np.maximum(pos, 0)], np.nan)
            s.nifty = nf
            if ik > 0:
                s.nifty_pdc = float(index.c[index.ends[ik - 1] - 1])
        out.append(s)
    return out
