"""
stratlab/kite_tf.py — continuous multi-timeframe candles and a signal "machine" for the round-5 Kite-ecosystem
strategies (stratlab/catalog/r5_kite.py). Research only.

Candles are aligned at 09:15 like Kite / TradingView charts and KEEP the short last candle of a session (the
15:15-15:30 candle of 30- and 60-minute charts), because the platforms' indicators run on the broker's candle
series, which includes it. Indicators run over the symbol's whole candle series (earlier sessions + today's
COMPLETED candles); the value at candle j uses candles <= j only. A session only reads today's candles that
`candles.resample(..., complete_only=True)` reports complete on the visible bars, so a truncated View gives the
same earlier signals (the lab's look-ahead audit re-checks this on every run).

machine(): turns per-candle entry / exit conditions into the lab's `Entry` list: a decision at the close of candle
j is filled at the open of the next 1-minute bar; exits by condition are `Entry.exit_sig` masks (filled at the next
bar's open); an opposite entry decided on the same candle fills at that same open (stop-and-reverse). No decision
whose fill would be at or after 15:15 (the lab's square-off).
"""
from __future__ import annotations

import weakref

import numpy as np

from .api import LONG, SHORT, Entry
from .candles import resample
from .data import SESSION_OPEN, hhmm

SESSION_END = 15 * 60 + 30
SQ_OFF = hhmm("15:15")
NOSTOP = 0.5            # "no stop": a placeholder stop 50 % away (never reached intraday; NSE bands are <= 20 %)


class Series:
    """All tf-minute candles of one symbol (short last candles kept). Arrays per candle: o h l c v start end
    sess (session index) first (first candle of its session) day (yyyymmdd); sess_lo[k] = first candle of
    session k. `memo` caches per-series indicator arrays."""

    def __init__(self, sd, tf: int):
        lens = sd.ends - sd.starts
        sess_of = np.repeat(np.arange(len(sd.starts)), lens)
        t = sd.t.astype(np.int64)
        bid = sess_of.astype(np.int64) * 1000 + (t - SESSION_OPEN) // tf
        cut = np.flatnonzero(bid[1:] != bid[:-1]) + 1
        st = np.concatenate(([0], cut)).astype(np.int64)
        en = np.concatenate((cut, [len(bid)])).astype(np.int64) - 1
        self.tf = tf
        self.o = sd.o[st].astype(float)
        self.h = np.maximum.reduceat(sd.h.astype(float), st) if len(st) else np.zeros(0)
        self.l = np.minimum.reduceat(sd.l.astype(float), st) if len(st) else np.zeros(0)
        self.v = np.add.reduceat(sd.v.astype(float), st) if len(st) else np.zeros(0)
        self.c = sd.c[en].astype(float)
        self.sess = sess_of[st]
        self.start = SESSION_OPEN + ((t[st] - SESSION_OPEN) // tf) * tf
        self.end = np.minimum(self.start + tf, SESSION_END)
        self.first = np.ones(len(st), bool)
        self.first[1:] = self.sess[1:] != self.sess[:-1]
        self.day = sd.days[self.sess] if len(st) else np.zeros(0, np.int64)
        self.sess_lo = np.searchsorted(self.sess, np.arange(len(sd.starts)), side="left")
        self.memo: dict = {}

    def get(self, name, fn):
        x = self.memo.get(name)
        if x is None:
            x = fn(self)
            self.memo[name] = x
        return x

    def __len__(self):
        return len(self.c)


_CACHE: dict = {}


def series(sd, tf: int) -> Series:
    key = (id(sd), tf)
    ent = _CACHE.get(key)
    if ent is None or ent[0]() is not sd:
        if len(_CACHE) > 24:
            _CACHE.clear()
        ent = (weakref.ref(sd), Series(sd, tf))
        _CACHE[key] = ent
    return ent[1]


def today(s, tf: int):
    """(series, global indices of today's COMPLETE candles visible in s, their decision bars in s)."""
    base = getattr(s, "_s", s)
    sd = base._sd
    k = int(base.day_index)
    S = series(sd, tf)
    cd = resample(s.t, s.o, s.h, s.l, s.c, s.v, tf)
    m = len(cd.c)
    empty = (S, np.zeros(0, np.int64), np.zeros(0, np.int64))
    if m == 0:
        return empty
    g = int(S.sess_lo[k]) + np.arange(m)
    if g[-1] >= len(S) or (S.sess[g] != k).any() or (S.start[g] != cd.start[:m]).any():
        return empty                                    # grids disagree (missing minutes): no signals today
    return S, g, np.asarray(cd.sig[:m], np.int64)


def minute_series(sd, name, fn):
    """A per-symbol array over ALL minute bars (e.g. a continuous 1-minute EMA), cached on the 1-minute Series."""
    return series(sd, 1).get(name, fn)


def _stop_for(side, price, stop):
    if stop is None:
        return price * (1.0 - side * NOSTOP)
    return float(stop)


def machine(s, tf: int, long_in=None, short_in=None, long_out=None, short_out=None, *, bracket=None,
            stop_fn=None, target_fn=None, first_start=None, last_start=None, short_wins=False):
    """Entries for today's session from per-candle conditions over the GLOBAL candle series (bool arrays, or
    None = never). long_out / short_out = exit conditions of an open long / short (an opposite entry is NOT an
    exit unless it is included in them). first_start / last_start ('HH:MM'): only candles STARTING in that window
    may trigger an entry. stop_fn(j) / target_fn(j) -> absolute prices for the candle; bracket = Entry.bracket.
    short_wins: when long_in and short_in both fire on one candle, only the short is kept (Pine: the later
    strategy.entry call of the bar decides the position)."""
    S, g, sig = today(s, tf)
    n = len(s.t)
    out = []
    if not len(g):
        return out
    xl = np.zeros(n, bool)
    xs = np.zeros(n, bool)
    ok = (sig >= 0) & (sig < n)
    if long_out is not None:
        xl[sig[ok & np.asarray(long_out, bool)[g]]] = True
    if short_out is not None:
        xs[sig[ok & np.asarray(short_out, bool)[g]]] = True
    tt = np.asarray(s.t)
    can = ok & (tt[np.where(ok, sig, 0)] + 1 < SQ_OFF)     # fill (next minute) before 15:15; never reads bar sb+1
    lo_t = hhmm(first_start) if first_start else -1
    hi_t = hhmm(last_start) if last_start else 10 ** 6
    can &= (S.start[g] >= lo_t) & (S.start[g] <= hi_t)
    li = can & (np.asarray(long_in, bool)[g] if long_in is not None else False)
    si = can & (np.asarray(short_in, bool)[g] if short_in is not None else False)
    if short_wins:
        li = li & ~si
    for side, on, mask in ((LONG, li, xl), (SHORT, si, xs)):
        for jl in np.flatnonzero(on):
            j, sb = int(g[jl]), int(sig[jl])
            st = stop_fn(j, side) if stop_fn else None
            tg = target_fn(j, side) if target_fn else None
            if st is not None and not np.isfinite(st):
                continue
            out.append(Entry(sb, side, stop=_stop_for(side, S.c[j], st), target=tg, exit_sig=mask,
                             bracket=bracket))
    return out
