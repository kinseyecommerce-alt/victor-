"""
Round-3 TIMEFRAME 60-MINUTE / HOURLY indicator systems (pre-registered in scratchpad/r3/tf60m/PREREGISTERED.txt
before any run).

Hourly candles aligned at 09:15 (09:15-10:15 ... 14:15-15:15; the 15:15-15:30 stub is dropped from the indicator
history, as candles._symbol_candles does). Indicators run continuously over the symbol's whole hourly series
(earlier sessions + today's COMPLETED candles): the value at candle j uses candles <= j only. A session only reads
today's candles that `candles.resample(..., complete_only=True)` reports complete on the visible bars, so a
truncated View yields the same earlier signals (audited). Decisions at the close of candles ending 10:15-14:15.

Parameters are textbook / round values fixed before any result. Do not tune.
"""
from __future__ import annotations

import weakref

import numpy as np
import pandas as pd

from ..api import LONG, SHORT, Entry, strategy
from ..candles import resample
from ..data import SESSION_OPEN, hhmm

TF = 60
SESSION_END = 15 * 60 + 30
D_FIRST_END = hhmm("10:15")     # earliest decision candle END
D_LAST_END = hhmm("14:15")      # latest decision candle END
SQ_OFF = hhmm("15:15")


# ── per-symbol continuous hourly series (stub candles dropped), cached per SymbolData ─────────────────────────
class _Series:
    def __init__(self, sd):
        lens = sd.ends - sd.starts
        sess_of = np.repeat(np.arange(len(sd.starts)), lens)
        t = sd.t.astype(np.int64)
        bid = sess_of.astype(np.int64) * 1000 + (t - SESSION_OPEN) // TF
        cut = np.flatnonzero(bid[1:] != bid[:-1]) + 1
        st = np.concatenate(([0], cut)).astype(np.int64)
        en = np.concatenate((cut, [len(bid)])).astype(np.int64) - 1
        start = SESSION_OPEN + ((t[st] - SESSION_OPEN) // TF) * TF
        full = start + TF <= SESSION_END
        st, en, start = st[full], en[full], start[full]
        self.o = sd.o[st].astype(float)
        # h / l / v over [st, en] explicitly (a reduceat over the kept starts would swallow the dropped stubs)
        cs_v = np.concatenate(([0.0], np.cumsum(sd.v.astype(float))))
        self.v = cs_v[en + 1] - cs_v[st]
        if len(st):
            pairs = np.stack((st, en + 1), 1).ravel()
            ok = pairs < len(t)
            hi = np.maximum.reduceat(sd.h.astype(float), pairs[ok])
            lo = np.minimum.reduceat(sd.l.astype(float), pairs[ok])
            # even positions of the (start, end+1) pairs are the [st, en] blocks
            pos = np.flatnonzero(ok)
            keep = (pos % 2) == 0
            self.h, self.l = hi[keep], lo[keep]
        else:
            self.h = self.l = np.zeros(0)
        self.c = sd.c[en].astype(float)
        self.sess = sess_of[st]
        self.start = start
        self.end = start + TF
        self.last_bar = en
        self.first = np.ones(len(st), bool)
        self.first[1:] = self.sess[1:] != self.sess[:-1]
        self.sess_lo = np.searchsorted(self.sess, np.arange(len(sd.starts)), side="left")
        self.slot = (self.start - SESSION_OPEN) // TF
        self._m: dict = {}

    def get(self, name, fn):
        x = self._m.get(name)
        if x is None:
            x = fn(self)
            self._m[name] = x
        return x


_SER: dict = {}


def _series(sd) -> _Series:
    ent = _SER.get(id(sd))
    if ent is None or ent[0]() is not sd:
        if len(_SER) > 16:
            _SER.clear()
        ent = (weakref.ref(sd), _Series(sd))
        _SER[id(sd)] = ent
    return ent[1]


def _today(s):
    """(series, global indices of today's COMPLETE full hourly candles, their sig bars, today's candles)."""
    base = getattr(s, "_s", s)
    sd = base._sd
    k = int(base.day_index)
    ser = _series(sd)
    cd = resample(s.t, s.o, s.h, s.l, s.c, s.v, TF)
    full = (cd.end - cd.start) == TF
    m = int(full.sum())
    if m and not full[:m].all():
        return ser, np.zeros(0, np.int64), np.zeros(0, np.int64), cd
    g0 = int(ser.sess_lo[k])
    g = g0 + np.arange(m)
    if m and (g[-1] >= len(ser.c) or (ser.sess[g] != k).any() or (ser.start[g] != cd.start[:m]).any()):
        return ser, np.zeros(0, np.int64), np.zeros(0, np.int64), cd   # grids disagree: no signals
    return ser, g, np.asarray(cd.sig[:m], np.int64), cd


# ── causal indicator primitives ─────────────────────────────────────────────
def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def _wild(x, n):
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def _sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def _std(x, n):
    return pd.Series(x).rolling(n).std(ddof=0).to_numpy()


def _hh(x, n):
    return pd.Series(x).rolling(n).max().to_numpy()


def _ll(x, n):
    return pd.Series(x).rolling(n).min().to_numpy()


def _sh(x, k=1):
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def _tr(S):
    pc = _sh(S.c)
    tr = np.maximum(S.h - S.l, np.maximum(np.abs(S.h - pc), np.abs(S.l - pc)))
    if len(tr):
        tr[0] = S.h[0] - S.l[0]
    return tr


def _atr(n):
    def f(S):
        a = _wild(S.get("tr", _tr), n).copy()
        a[:n] = np.nan
        return a
    return f


def _rsi(n):
    def f(S):
        d = np.diff(S.c, prepend=S.c[0])
        up = _wild(np.clip(d, 0, None), n)
        dn = _wild(np.clip(-d, 0, None), n)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = 100.0 - 100.0 / (1.0 + up / dn)
        r[(dn == 0) & (up > 0)] = 100.0
        r[(dn == 0) & (up == 0)] = 50.0
        r[:n] = np.nan
        return r
    return f


def _dmi(S):
    up = S.h - _sh(S.h)
    dn = _sh(S.l) - S.l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    if len(pdm):
        pdm[0] = mdm[0] = 0.0
    atr = _wild(S.get("tr", _tr), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * _wild(pdm, 14) / atr
        mdi = 100 * _wild(mdm, 14) / atr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    dx = np.nan_to_num(dx, nan=0.0)
    adx = _wild(dx, 14).copy()
    adx[:28] = np.nan
    return pdi, mdi, adx


def _supertrend(S, n=10, mult=3.0):
    atr = _wild(S.get("tr", _tr), n)
    hl2 = (S.h + S.l) / 2
    bu, bl = hl2 + mult * atr, hl2 - mult * atr
    N = len(S.c)
    fu, fl = bu.copy(), bl.copy()
    d = np.ones(N, np.int64)
    line = np.full(N, np.nan)
    c = S.c
    for j in range(1, N):
        fu[j] = bu[j] if (bu[j] < fu[j - 1] or c[j - 1] > fu[j - 1]) else fu[j - 1]
        fl[j] = bl[j] if (bl[j] > fl[j - 1] or c[j - 1] < fl[j - 1]) else fl[j - 1]
        if d[j - 1] == 1:
            d[j] = -1 if c[j] < fl[j] else 1
        else:
            d[j] = 1 if c[j] > fu[j] else -1
        line[j] = fl[j] if d[j] == 1 else fu[j]
    d[:n] = 0
    return d, line


def _heikin(S):
    hc = (S.o + S.h + S.l + S.c) / 4.0
    N = len(hc)
    ho = np.empty(N)
    if N:
        ho[0] = (S.o[0] + S.c[0]) / 2.0
    for j in range(1, N):
        ho[j] = (ho[j - 1] + hc[j - 1]) / 2.0
    hh = np.maximum(S.h, np.maximum(ho, hc))
    hl = np.minimum(S.l, np.minimum(ho, hc))
    return ho, hc, hh, hl


def _slot_vmean(S, n=20):
    """Mean volume of the same hour-slot over the previous n sessions (NaN until n exist)."""
    out = np.full(len(S.v), np.nan)
    for k in np.unique(S.slot):
        idx = np.flatnonzero(S.slot == k)
        out[idx] = _sh(_sma(S.v[idx], n))
    return out


def A(S, name):
    f = {
        "atr14": _atr(14), "atr20": _atr(20),
        "ema20": lambda S: _ema(S.c, 20), "ema50": lambda S: _ema(S.c, 50),
        "sma5": lambda S: _sma(S.c, 5), "sma20": lambda S: _sma(S.c, 20), "sd20": lambda S: _std(S.c, 20),
        "rsi14": _rsi(14), "rsi2": _rsi(2), "dmi": _dmi, "st": _supertrend, "ha": _heikin,
        "macd": lambda S: _ema(S.c, 12) - _ema(S.c, 26),
        "macds": lambda S: _ema(A(S, "macd"), 9),
        "hh20p": lambda S: _sh(_hh(S.h, 20)), "ll20p": lambda S: _sh(_ll(S.l, 20)),
        "hh20": lambda S: _hh(S.h, 20), "ll20": lambda S: _ll(S.l, 20),
        "vslot20": _slot_vmean,
    }[name]
    return S.get(name, f)


# ── plumbing ────────────────────────────────────────────────────────────────
def _ok(*xs):
    return all(x is not None and np.isfinite(x) for x in xs)


def _run(s, sig_fn, first_end=D_FIRST_END):
    """Loop over today's decision candles; sig_fn(jl, j, sb, S) -> Entry | None."""
    S, g, sig, cd = _today(s)
    out = []
    for jl in range(len(g)):
        j, sb = int(g[jl]), int(sig[jl])
        if S.end[j] < first_end or S.end[j] > D_LAST_END or j < 1:
            continue
        if sb < 0 or sb >= len(s) or s.t[sb] >= SQ_OFF:
            continue
        e = sig_fn(jl, j, sb, S)
        if e is not None:
            out.append(e)
    return out


def _atr_entry(sb, side, c, atr, k, rr=None, target=None, trail=None):
    if not _ok(c, atr) or atr <= 0:
        return None
    return Entry(int(sb), side, stop=float(c - side * k * atr), rr=rr, target=target, trail=trail)


def _x_up(a, b, j):
    return _ok(a[j], a[j - 1], b[j], b[j - 1]) and a[j] > b[j] and a[j - 1] <= b[j - 1]


def _x_dn(a, b, j):
    return _ok(a[j], a[j - 1], b[j], b[j - 1]) and a[j] < b[j] and a[j - 1] >= b[j - 1]


def _dsma(s, n):
    c = s.prev.c
    if len(c) < n:
        return np.nan
    return float(np.mean(c[-n:]))


def _dtrend(s, n):
    """+1 if yesterday's close > n-day SMA of daily closes to yesterday, -1 if below, 0 unknown."""
    m = _dsma(s, n)
    if not _ok(m) or len(s.prev.c) == 0:
        return 0
    pc = float(s.prev.c[-1])
    return 1 if pc > m else (-1 if pc < m else 0)


# ── 1-3 EMA20 pullback ──────────────────────────────────────────────────────
def _ema_pb(s, p, dfilter):
    dt = _dtrend(s, p.dsma) if dfilter else 0
    if dfilter and dt == 0:
        return []
    S = _today(s)[0]
    e20, e50, atr = A(S, "ema20"), A(S, "ema50"), A(S, "atr14")

    def f(jl, j, sb, S):
        if not _ok(e20[j], e20[j - 1], e50[j]):
            return None
        if e20[j] > e50[j] and e20[j] > e20[j - 1] and S.l[j] <= e20[j] < S.c[j] and dt >= 0:
            return _atr_entry(sb, LONG, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        if e20[j] < e50[j] and e20[j] < e20[j - 1] and S.h[j] >= e20[j] > S.c[j] and dt <= 0:
            return _atr_entry(sb, SHORT, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        return None
    return _run(s, f)


@strategy("r3_tf60m_ema_pb_trend", "intraday", "ema_pullback",
          "Hourly EMA20 > EMA50 and rising: candle low touches EMA20 and closes above -> long (mirror); 1.5 ATR60 stop, 2R",
          {"fast": 20, "slow": 50, "atr_stop": 1.5, "rr": 2.0})
def r3_tf60m_ema_pb_trend(s, p):
    return _ema_pb(s, p, False)


@strategy("r3_tf60m_ema_pb_daily", "intraday", "ema_pullback",
          "Hourly EMA20/50 pullback (as ema_pb_trend) only in the direction of yesterday's close vs its 50-day SMA; 1.5 ATR60 stop, 2R",
          {"fast": 20, "slow": 50, "dsma": 50, "atr_stop": 1.5, "rr": 2.0})
def r3_tf60m_ema_pb_daily(s, p):
    return _ema_pb(s, p, True)


@strategy("r3_tf60m_adx_ema20_pb", "intraday", "ema_pullback",
          "Hourly ADX(14) > 25 and +DI > -DI: low touches EMA20, close above -> long (mirror); 1.5 ATR60 stop, 2R",
          {"ema": 20, "adx_min": 25, "atr_stop": 1.5, "rr": 2.0})
def r3_tf60m_adx_ema20_pb(s, p):
    S = _today(s)[0]
    e20, atr = A(S, "ema20"), A(S, "atr14")
    pdi, mdi, adx = A(S, "dmi")

    def f(jl, j, sb, S):
        if not _ok(e20[j], adx[j], pdi[j], mdi[j]) or adx[j] <= p.adx_min:
            return None
        if pdi[j] > mdi[j] and S.l[j] <= e20[j] < S.c[j]:
            return _atr_entry(sb, LONG, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        if mdi[j] > pdi[j] and S.h[j] >= e20[j] > S.c[j]:
            return _atr_entry(sb, SHORT, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        return None
    return _run(s, f)


# ── 4 EMA cross ─────────────────────────────────────────────────────────────
@strategy("r3_tf60m_ema20_50_cross", "intraday", "ema_cross",
          "Hourly EMA20 crosses EMA50: follow; 1.5 ATR60 stop, hold to 15:15",
          {"fast": 20, "slow": 50, "atr_stop": 1.5})
def r3_tf60m_ema20_50_cross(s, p):
    S = _today(s)[0]
    e20, e50, atr = A(S, "ema20"), A(S, "ema50"), A(S, "atr14")

    def f(jl, j, sb, S):
        if j < p.slow:
            return None
        if _x_up(e20, e50, j):
            return _atr_entry(sb, LONG, S.c[j], atr[j], p.atr_stop)
        if _x_dn(e20, e50, j):
            return _atr_entry(sb, SHORT, S.c[j], atr[j], p.atr_stop)
        return None
    return _run(s, f)


# ── 5-7 RSI ─────────────────────────────────────────────────────────────────
@strategy("r3_tf60m_rsi_regime_pb", "intraday", "rsi_regime",
          "Hourly RSI(14) bull range (prev 10 all >= 40, one >= 60) and close > EMA50: RSI re-crosses above 50 -> long (mirror); 1.5 ATR60, 2R",
          {"rsi": 14, "lo": 40, "hi": 60, "mid": 50, "look": 10, "ema": 50, "atr_stop": 1.5, "rr": 2.0})
def r3_tf60m_rsi_regime_pb(s, p):
    S = _today(s)[0]
    r, e50, atr = A(S, "rsi14"), A(S, "ema50"), A(S, "atr14")

    def f(jl, j, sb, S):
        if j < p.look + 1 or not _ok(r[j], r[j - 1], e50[j]):
            return None
        w = r[j - p.look:j]
        if not np.isfinite(w).all():
            return None
        if (w >= p.lo).all() and (w >= p.hi).any() and S.c[j] > e50[j] and r[j - 1] <= p.mid < r[j]:
            return _atr_entry(sb, LONG, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        if (w <= p.hi).all() and (w <= p.lo).any() and S.c[j] < e50[j] and r[j - 1] >= p.mid > r[j]:
            return _atr_entry(sb, SHORT, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        return None
    return _run(s, f)


@strategy("r3_tf60m_rsi30_70_fade", "intraday", "rsi_reversion",
          "Hourly RSI(14) crosses back above 30 -> long, back below 70 -> short; target hourly EMA20, 1.5 ATR60 stop, else 15:15",
          {"rsi": 14, "lo": 30, "hi": 70, "atr_stop": 1.5})
def r3_tf60m_rsi30_70_fade(s, p):
    S = _today(s)[0]
    r, e20, atr = A(S, "rsi14"), A(S, "ema20"), A(S, "atr14")

    def f(jl, j, sb, S):
        if not _ok(r[j], r[j - 1], e20[j]):
            return None
        c = S.c[j]
        if r[j - 1] < p.lo <= r[j] and e20[j] > c:
            return _atr_entry(sb, LONG, c, atr[j], p.atr_stop, target=float(e20[j]))
        if r[j - 1] > p.hi >= r[j] and e20[j] < c:
            return _atr_entry(sb, SHORT, c, atr[j], p.atr_stop, target=float(e20[j]))
        return None
    return _run(s, f)


@strategy("r3_tf60m_rsi2_daily_trend", "intraday", "rsi_reversion",
          "Yesterday above its 100-day SMA and hourly RSI(2) < 10 -> long (below and > 90 -> short); target hourly SMA5, 2 ATR60 stop",
          {"rsi": 2, "lo": 10, "hi": 90, "dsma": 100, "sma_exit": 5, "atr_stop": 2.0})
def r3_tf60m_rsi2_daily_trend(s, p):
    dt = _dtrend(s, p.dsma)
    if dt == 0:
        return []
    S = _today(s)[0]
    r, m5, atr = A(S, "rsi2"), A(S, "sma5"), A(S, "atr14")

    def f(jl, j, sb, S):
        if not _ok(r[j], m5[j]):
            return None
        c = S.c[j]
        if dt == 1 and r[j] < p.lo and m5[j] > c:
            return _atr_entry(sb, LONG, c, atr[j], p.atr_stop, target=float(m5[j]))
        if dt == -1 and r[j] > p.hi and m5[j] < c:
            return _atr_entry(sb, SHORT, c, atr[j], p.atr_stop, target=float(m5[j]))
        return None
    return _run(s, f)


# ── 8, 9, 21 Supertrend ─────────────────────────────────────────────────────
def _st_flip(s, p, dfilter, trail):
    dt = _dtrend_sma20(s) if dfilter else 0
    if dfilter and dt == 0:
        return []
    S = _today(s)[0]
    d, line = A(S, "st")
    atr = A(S, "atr14")

    def f(jl, j, sb, S):
        if d[j] == 0 or d[j - 1] == 0 or d[j] == d[j - 1] or not _ok(line[j], atr[j]) or atr[j] <= 0:
            return None
        side = int(d[j])
        if dfilter and side != dt:
            return None
        c = S.c[j]
        if (c - line[j]) * side <= 0:
            return None
        stop = line[j] if abs(c - line[j]) >= p.min_stop_atr * atr[j] else c - side * p.min_stop_atr * atr[j]
        return Entry(sb, side, stop=float(stop), trail=("atr", p.trail_atr) if trail else None)
    return _run(s, f)


def _dtrend_sma20(s):
    m = s.dsma20
    if not _ok(m) or not _ok(s.pdc):
        return 0
    return 1 if s.pdc > m else (-1 if s.pdc < m else 0)


@strategy("r3_tf60m_supertrend_flip", "intraday", "supertrend",
          "Hourly Supertrend(10,3) flips: follow, stop the line (>= 0.5 ATR60), trail 3 x 5m ATR, else 15:15",
          {"n": 10, "mult": 3.0, "min_stop_atr": 0.5, "trail_atr": 3.0})
def r3_tf60m_supertrend_flip(s, p):
    return _st_flip(s, p, False, True)


@strategy("r3_tf60m_supertrend_daily", "intraday", "supertrend",
          "Hourly Supertrend(10,3) flip in the direction of yesterday's close vs 20-day SMA; stop the line (>= 0.5 ATR60), trail 3 x 5m ATR",
          {"n": 10, "mult": 3.0, "min_stop_atr": 0.5, "trail_atr": 3.0})
def r3_tf60m_supertrend_daily(s, p):
    return _st_flip(s, p, True, True)


@strategy("r3_tf60m_supertrend_nifty", "futures", "supertrend_index",
          "NIFTY future: hourly Supertrend(10,3) flip -> follow; stop the line (>= 0.5 ATR60), hold to 15:15",
          {"n": 10, "mult": 3.0, "min_stop_atr": 0.5}, horizon="intraday", underlying="NIFTY")
def r3_tf60m_supertrend_nifty(s, p):
    return _st_flip(s, p, False, False)


# ── 10-11 MACD ──────────────────────────────────────────────────────────────
@strategy("r3_tf60m_macd_signal_trend", "intraday", "macd",
          "Hourly MACD(12,26,9) crosses its signal on the trend side of zero (above 0 -> long, below 0 -> short); 1.5 ATR60, 2R",
          {"fast": 12, "slow": 26, "sig": 9, "atr_stop": 1.5, "rr": 2.0})
def r3_tf60m_macd_signal_trend(s, p):
    S = _today(s)[0]
    m, sg, atr = A(S, "macd"), A(S, "macds"), A(S, "atr14")

    def f(jl, j, sb, S):
        if j < p.slow + p.sig:
            return None
        if _x_up(m, sg, j) and m[j] > 0:
            return _atr_entry(sb, LONG, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        if _x_dn(m, sg, j) and m[j] < 0:
            return _atr_entry(sb, SHORT, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        return None
    return _run(s, f)


@strategy("r3_tf60m_macd_zero_cross", "intraday", "macd",
          "Hourly MACD line (12,26) crosses zero: follow; 1.5 ATR60 stop, hold to 15:15",
          {"fast": 12, "slow": 26, "atr_stop": 1.5})
def r3_tf60m_macd_zero_cross(s, p):
    S = _today(s)[0]
    m, atr = A(S, "macd"), A(S, "atr14")

    def f(jl, j, sb, S):
        if j < p.slow:
            return None
        if _ok(m[j], m[j - 1]) and m[j] > 0 >= m[j - 1]:
            return _atr_entry(sb, LONG, S.c[j], atr[j], p.atr_stop)
        if _ok(m[j], m[j - 1]) and m[j] < 0 <= m[j - 1]:
            return _atr_entry(sb, SHORT, S.c[j], atr[j], p.atr_stop)
        return None
    return _run(s, f)


# ── 12-14 Bollinger ─────────────────────────────────────────────────────────
@strategy("r3_tf60m_bb_squeeze_break", "intraday", "bollinger_squeeze",
          "Previous hourly BB(20,2) bandwidth lowest of 120 candles, then a close outside the band: follow; stop middle band (>= 1 ATR60), 2R",
          {"n": 20, "k": 2.0, "look": 120, "min_stop_atr": 1.0, "rr": 2.0})
def r3_tf60m_bb_squeeze_break(s, p):
    S = _today(s)[0]
    m, sd, atr = A(S, "sma20"), A(S, "sd20"), A(S, "atr14")
    bw = S.get("bbw", lambda S: 2 * p.k * A(S, "sd20") / A(S, "sma20"))
    bwmin = S.get("bbw_min120", lambda S: _ll(bw, p.look))

    def f(jl, j, sb, S):
        c, a = S.c[j], atr[j]
        if not _ok(m[j], sd[j], a, bw[j - 1], bwmin[j - 1]) or a <= 0:
            return None
        if bw[j - 1] > bwmin[j - 1] + 1e-12:
            return None
        up, lo = m[j] + p.k * sd[j], m[j] - p.k * sd[j]
        if c > up:
            return Entry(sb, LONG, stop=float(min(m[j], c - p.min_stop_atr * a)), rr=p.rr)
        if c < lo:
            return Entry(sb, SHORT, stop=float(max(m[j], c + p.min_stop_atr * a)), rr=p.rr)
        return None
    return _run(s, f)


@strategy("r3_tf60m_ttm_squeeze", "intraday", "bollinger_squeeze",
          "Hourly TTM squeeze: BB(20,2) inside Keltner(SMA20 +- 1.5 ATR20) >= 6 candles then released; trade momentum sign; 1.5 ATR60, 2R",
          {"n": 20, "k": 2.0, "kc": 1.5, "min_on": 6, "atr_stop": 1.5, "rr": 2.0})
def r3_tf60m_ttm_squeeze(s, p):
    S = _today(s)[0]

    def mk(S):
        m, sd, a20 = A(S, "sma20"), A(S, "sd20"), A(S, "atr20")
        with np.errstate(invalid="ignore"):
            on = (p.k * sd < p.kc * a20)
        on = on & np.isfinite(sd) & np.isfinite(a20)
        mom = S.c - ((A(S, "hh20") + A(S, "ll20")) / 2 + m) / 2
        return on, mom
    on, mom = S.get("ttm", mk)
    atr = A(S, "atr14")

    def f(jl, j, sb, S):
        if j < p.min_on or on[j] or not on[j - p.min_on:j].all() or not _ok(mom[j]):
            return None
        side = LONG if mom[j] > 0 else (SHORT if mom[j] < 0 else 0)
        if not side:
            return None
        return _atr_entry(sb, side, S.c[j], atr[j], p.atr_stop, rr=p.rr)
    return _run(s, f)


@strategy("r3_tf60m_bb_fade_adx", "intraday", "bollinger_reversion",
          "Hourly ADX(14) < 20: previous close outside BB(20,2), this close back inside -> fade to middle band; stop 2-candle extreme +- 0.25 ATR60",
          {"n": 20, "k": 2.0, "adx_max": 20, "pad_atr": 0.25})
def r3_tf60m_bb_fade_adx(s, p):
    S = _today(s)[0]
    m, sd, atr = A(S, "sma20"), A(S, "sd20"), A(S, "atr14")
    _, _, adx = A(S, "dmi")

    def f(jl, j, sb, S):
        c, pc, a = S.c[j], S.c[j - 1], atr[j]
        if not _ok(m[j], m[j - 1], sd[j], sd[j - 1], a, adx[j]) or a <= 0 or adx[j] >= p.adx_max:
            return None
        if pc > m[j - 1] + p.k * sd[j - 1] and m[j] < c < m[j] + p.k * sd[j]:
            return Entry(sb, SHORT, stop=float(max(S.h[j], S.h[j - 1]) + p.pad_atr * a), target=float(m[j]))
        if pc < m[j - 1] - p.k * sd[j - 1] and m[j] - p.k * sd[j] < c < m[j]:
            return Entry(sb, LONG, stop=float(min(S.l[j], S.l[j - 1]) - p.pad_atr * a), target=float(m[j]))
        return None
    return _run(s, f)


# ── 15 Heikin-Ashi ──────────────────────────────────────────────────────────
@strategy("r3_tf60m_ha_flip_daily", "intraday", "heikin_ashi",
          "Hourly Heikin-Ashi turns green with no lower shadow while yesterday closed above its 20-day SMA -> long (mirror); 1.5 ATR60, 2R",
          {"atr_stop": 1.5, "rr": 2.0})
def r3_tf60m_ha_flip_daily(s, p):
    dt = _dtrend_sma20(s)
    if dt == 0:
        return []
    S = _today(s)[0]
    ho, hc, hh, hl = A(S, "ha")
    atr = A(S, "atr14")
    eps = 1e-9

    def f(jl, j, sb, S):
        if dt == 1 and hc[j] > ho[j] and hc[j - 1] < ho[j - 1] and hl[j] >= ho[j] - eps:
            return _atr_entry(sb, LONG, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        if dt == -1 and hc[j] < ho[j] and hc[j - 1] > ho[j - 1] and hh[j] <= ho[j] + eps:
            return _atr_entry(sb, SHORT, S.c[j], atr[j], p.atr_stop, rr=p.rr)
        return None
    return _run(s, f)


# ── 16 volume shock ─────────────────────────────────────────────────────────
@strategy("r3_tf60m_hour_volshock", "intraday", "volume_shock",
          "Hourly candle (ending 11:15-14:15) volume >= 3x its hour-slot 20-session mean, close in top 25 % and up -> long (mirror); stop candle extreme (>= 0.5 ATR60)",
          {"vol": 3.0, "slots": 20, "q": 0.25, "min_stop_atr": 0.5})
def r3_tf60m_hour_volshock(s, p):
    S = _today(s)[0]
    vm, atr = A(S, "vslot20"), A(S, "atr14")

    def f(jl, j, sb, S):
        h, l, c, o, a = S.h[j], S.l[j], S.c[j], S.o[j], atr[j]
        if not _ok(vm[j], a) or vm[j] <= 0 or a <= 0 or h <= l or S.v[j] < p.vol * vm[j]:
            return None
        pos = (c - l) / (h - l)
        if c > o and pos >= 1 - p.q:
            return Entry(sb, LONG, stop=float(min(l, c - p.min_stop_atr * a)))
        if c < o and pos <= p.q:
            return Entry(sb, SHORT, stop=float(max(h, c + p.min_stop_atr * a)))
        return None
    return _run(s, f, first_end=hhmm("11:15"))


# ── 17 idiosyncratic hourly shock fade ──────────────────────────────────────
@strategy("r3_tf60m_idio_hour_fade", "intraday", "idio_reversal",
          "Hourly candle (ending 11:15-14:15): stock return minus NIFTY's <= -1.5 % -> long (>= +1.5 % short); target 50 % retrace, 1.5 ATR60 stop",
          {"rel": 1.5, "atr_stop": 1.5, "retr": 0.5}, needs_index=True)
def r3_tf60m_idio_hour_fade(s, p):
    S, g, sig, cd = _today(s)
    nf = s.nifty
    if nf is None:
        return []
    atr = A(S, "atr14")
    last = np.asarray(cd.last, np.int64)

    def f(jl, j, sb, S):
        if jl < 1:
            return None
        a, b = int(last[jl - 1]), int(last[jl])
        if b >= len(nf) or not _ok(nf[a], nf[b]) or nf[a] <= 0:
            return None
        c0, c = S.c[j - 1], S.c[j]
        rs = c / c0 - 1.0
        rel = (rs - (nf[b] / nf[a] - 1.0)) * 100.0
        tgt = c0 + (1.0 - p.retr) * (c - c0)
        if rel <= -p.rel and rs < 0:
            return _atr_entry(sb, LONG, c, atr[j], p.atr_stop, target=float(tgt))
        if rel >= p.rel and rs > 0:
            return _atr_entry(sb, SHORT, c, atr[j], p.atr_stop, target=float(tgt))
        return None
    return _run(s, f, first_end=hhmm("11:15"))


# ── 18 gap against the hourly trend ─────────────────────────────────────────
@strategy("r3_tf60m_gap_vs_trend", "intraday", "gap_trend",
          "Hourly EMA20 > EMA50 at yesterday's close and gap <= -1 %: first hour closes above its open -> long at 10:15 (mirror); stop first-hour extreme (>= 0.5 ATR60)",
          {"fast": 20, "slow": 50, "gap": 1.0, "min_stop_atr": 0.5})
def r3_tf60m_gap_vs_trend(s, p):
    if not _ok(s.gap):
        return []
    S, g, sig, cd = _today(s)
    if len(g) < 1:
        return []
    j = int(g[0])
    if j < p.slow + 1 or S.end[j] != D_FIRST_END:
        return []
    e20, e50, atr = A(S, "ema20"), A(S, "ema50"), A(S, "atr14")
    y = j - 1                               # yesterday's last full hourly candle
    sb = int(sig[0])
    if not _ok(e20[y], e50[y], atr[j]) or atr[j] <= 0 or sb < 0 or sb >= len(s):
        return []
    c, a = S.c[j], atr[j]
    gp = s.gap * 100.0
    if e20[y] > e50[y] and gp <= -p.gap and c > S.o[j]:
        return [Entry(sb, LONG, stop=float(min(S.l[j], c - p.min_stop_atr * a)))]
    if e20[y] < e50[y] and gp >= p.gap and c < S.o[j]:
        return [Entry(sb, SHORT, stop=float(max(S.h[j], c + p.min_stop_atr * a)))]
    return []


# ── 19-20 Donchian on index futures ─────────────────────────────────────────
def _donchian(s, p):
    S = _today(s)[0]
    hh, ll, atr = A(S, "hh20p"), A(S, "ll20p"), A(S, "atr14")

    def f(jl, j, sb, S):
        c = S.c[j]
        if not _ok(hh[j], ll[j]):
            return None
        if c > hh[j]:
            return _atr_entry(sb, LONG, c, atr[j], p.atr_stop)
        if c < ll[j]:
            return _atr_entry(sb, SHORT, c, atr[j], p.atr_stop)
        return None
    return _run(s, f)


@strategy("r3_tf60m_donchian20_nifty", "futures", "donchian_index",
          "NIFTY future: hourly close beyond the previous 20 hourly candles' high/low -> follow; 2 ATR60 stop, hold to 15:15",
          {"n": 20, "atr_stop": 2.0}, horizon="intraday", underlying="NIFTY")
def r3_tf60m_donchian20_nifty(s, p):
    return _donchian(s, p)


@strategy("r3_tf60m_donchian20_banknifty", "futures", "donchian_index",
          "BANKNIFTY future: hourly close beyond the previous 20 hourly candles' high/low -> follow; 2 ATR60 stop, hold to 15:15",
          {"n": 20, "atr_stop": 2.0}, horizon="intraday", underlying="BANKNIFTY")
def r3_tf60m_donchian20_banknifty(s, p):
    return _donchian(s, p)
