"""
Round-3 TIMEFRAME 5-MINUTE indicator systems (pre-registered in scratchpad/r3/tf5m/PREREGISTERED.txt before any run).

Decisions only at the close of COMPLETED 5-minute candles (aligned at 09:15); the simulator fills at the next 1-minute
open. Indicators run continuously over the symbol's whole 5m (and 15m) candle series — earlier sessions plus today's
completed candles — exactly like the lab's own e9_5 / atr5: every value at candle j uses candles <= j only, and a
session only ever reads today's candles that `candles.resample(..., complete_only=True)` reports as complete on the
visible bars, so a truncated View yields the same earlier signals (audited).

Parameters are textbook / round values fixed before any result. Do not tune.
"""
from __future__ import annotations

import weakref

import numpy as np
import pandas as pd

from ..api import LONG, SHORT, Entry, strategy
from ..candles import resample
from ..data import SESSION_OPEN, hhmm

W_START = hhmm("09:45")      # first eligible candle START
W_END = hhmm("14:30")        # last eligible candle END
SQ_OFF = hhmm("15:15")


# ── per-symbol continuous candle series and indicators (cached per SymbolData) ──────────────────────────────────
class _Series:
    """All tf-minute candles of one symbol (every session), with lazily computed, cached indicator arrays."""

    def __init__(self, sd, tf: int):
        self.tf = tf
        lens = sd.ends - sd.starts
        sess_of = np.repeat(np.arange(len(sd.starts)), lens)
        t = sd.t.astype(np.int64)
        bid = sess_of.astype(np.int64) * 1000 + (t - SESSION_OPEN) // tf
        cut = np.flatnonzero(bid[1:] != bid[:-1]) + 1
        st = np.concatenate(([0], cut)).astype(np.int64)
        en = np.concatenate((cut, [len(bid)])).astype(np.int64) - 1
        self.o = sd.o[st].astype(float)
        self.h = np.maximum.reduceat(sd.h, st).astype(float)
        self.l = np.minimum.reduceat(sd.l, st).astype(float)
        self.c = sd.c[en].astype(float)
        self.v = np.add.reduceat(sd.v, st).astype(float)
        self.sess = sess_of[st]
        self.start = SESSION_OPEN + ((t[st] - SESSION_OPEN) // tf) * tf
        self.end = np.minimum(self.start + tf, 15 * 60 + 30)
        self.first = np.ones(len(st), bool)
        self.first[1:] = self.sess[1:] != self.sess[:-1]
        self.sess_lo = np.searchsorted(self.sess, np.arange(len(sd.starts)), side="left")
        self.key = self.sess.astype(np.int64) * 10000 + self.end
        self._m: dict = {}

    def get(self, name, fn):
        x = self._m.get(name)
        if x is None:
            x = fn(self)
            self._m[name] = x
        return x


_SER: dict = {}


def _series(sd, tf: int) -> _Series:
    ent = _SER.get(id(sd))
    if ent is None or ent[0]() is not sd:
        if len(_SER) > 16:
            _SER.clear()
        ent = (weakref.ref(sd), {})
        _SER[id(sd)] = ent
    got = ent[1].get(tf)
    if got is None:
        got = _Series(sd, tf)
        ent[1][tf] = got
    return got


def _today(s, tf: int = 5):
    """(series, global indices of today's COMPLETE candles, sig bar of each). Uses only the visible bars of s."""
    base = getattr(s, "_s", s)
    sd = base._sd
    k = int(base.day_index)
    ser = _series(sd, tf)
    cd = resample(s.t, s.o, s.h, s.l, s.c, s.v, tf)
    m = len(cd.o)
    g0 = int(ser.sess_lo[k])
    g = g0 + np.arange(m)
    if m and (g[-1] >= len(ser.c) or ser.sess[g[-1]] != k or ser.start[g[-1]] != cd.start[-1]):
        # candle grids disagree (should not happen); be safe: no signals
        return ser, np.zeros(0, np.int64), np.zeros(0, np.int64)
    return ser, g, np.asarray(cd.sig, np.int64)


# ── indicator primitives (causal) ───────────────────────────────────────────
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
    tr[S.first] = (S.h - S.l)[S.first]
    tr[0] = S.h[0] - S.l[0]
    return tr


def _atr_n(n):
    return lambda S: _wild(S.get("tr", _tr), n)


def _rsi14(S):
    d = np.diff(S.c, prepend=S.c[0])
    up = _wild(np.clip(d, 0, None), 14)
    dn = _wild(np.clip(-d, 0, None), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + up / dn)
    r[(dn == 0) & (up > 0)] = 100.0
    r[(dn == 0) & (up == 0)] = 50.0
    return r


def _dmi(S):
    up = S.h - _sh(S.h)
    dn = _sh(S.l) - S.l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[S.first] = 0.0
    mdm[S.first] = 0.0
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


def _psar(S, a0=0.02, step=0.02, amax=0.2):
    h, l = S.h, S.l
    N = len(h)
    sar = np.full(N, np.nan)
    d = np.zeros(N, np.int64)
    if N < 2:
        return d, sar
    up = h[1] >= h[0]
    ep = h[0] if up else l[0]
    s = l[0] if up else h[0]
    af = a0
    for j in range(1, N):
        s = s + af * (ep - s)
        if up:
            s = min(s, l[j - 1], l[j - 2] if j >= 2 else l[j - 1])
            if l[j] < s:
                up, s, ep, af = False, ep, l[j], a0
            elif h[j] > ep:
                ep, af = h[j], min(af + step, amax)
        else:
            s = max(s, h[j - 1], h[j - 2] if j >= 2 else h[j - 1])
            if h[j] > s:
                up, s, ep, af = True, ep, h[j], a0
            elif l[j] < ep:
                ep, af = l[j], min(af + step, amax)
        sar[j] = s
        d[j] = 1 if up else -1
    return d, sar


def _heikin(S):
    hc = (S.o + S.h + S.l + S.c) / 4.0
    N = len(hc)
    ho = np.empty(N)
    ho[0] = (S.o[0] + S.c[0]) / 2.0
    for j in range(1, N):
        ho[j] = (ho[j - 1] + hc[j - 1]) / 2.0
    hh = np.maximum(S.h, np.maximum(ho, hc))
    hl = np.minimum(S.l, np.minimum(ho, hc))
    return ho, hc, hh, hl


def _cci20(S):
    tp = (S.h + S.l + S.c) / 3.0
    n = 20
    out = np.full(len(tp), np.nan)
    if len(tp) >= n:
        w = np.lib.stride_tricks.sliding_window_view(tp, n)
        m = w.mean(axis=1)
        md = np.abs(w - m[:, None]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n - 1:] = (tp[n - 1:] - m) / (0.015 * md)
    return out


def _obv(S):
    d = np.sign(np.diff(S.c, prepend=S.c[0]))
    return np.cumsum(d * S.v)


# cached accessors
def A(S, name):
    f = {
        "atr14": _atr_n(14), "atr10": _atr_n(10), "atr20": _atr_n(20),
        "ema9": lambda S: _ema(S.c, 9), "ema20": lambda S: _ema(S.c, 20), "ema21": lambda S: _ema(S.c, 21),
        "sma20": lambda S: _sma(S.c, 20), "sd20": lambda S: _std(S.c, 20),
        "rsi14": _rsi14, "dmi": _dmi, "st": _supertrend, "psar": _psar, "ha": _heikin, "cci": _cci20,
        "obv": _obv,
        "macdh": lambda S: (lambda m: m - _ema(m, 9))(_ema(S.c, 12) - _ema(S.c, 26)),
        "vmean20": lambda S: _sh(_sma(S.v, 20)), "vmean50": lambda S: _sh(_sma(S.v, 50)),
        "hh20p": lambda S: _sh(_hh(S.h, 20)), "ll20p": lambda S: _sh(_ll(S.l, 20)),
        "hh10p": lambda S: _sh(_hh(S.h, 10)), "ll10p": lambda S: _sh(_ll(S.l, 10)),
        "hh20": lambda S: _hh(S.h, 20), "ll20": lambda S: _ll(S.l, 20),
    }[name]
    return S.get(name, f)


def _trend15(s, ser5, g):
    """+1 / -1 / 0: 15m trend of the last 15m candle ended at or before each of today's 5m candles."""
    base = getattr(s, "_s", s)
    S15 = _series(base._sd, 15)
    e = A(S15, "ema20")
    up = (S15.c > e) & (e > _sh(e))
    dn = (S15.c < e) & (e < _sh(e))
    q = np.searchsorted(S15.key, ser5.key[g], side="right") - 1
    out = np.zeros(len(g), np.int64)
    ok = q >= 1
    out[ok] = np.where(up[q[ok]], 1, np.where(dn[q[ok]], -1, 0))
    return out


def _st15(s, ser5, g):
    base = getattr(s, "_s", s)
    S15 = _series(base._sd, 15)
    d, _ = A(S15, "st")
    q = np.searchsorted(S15.key, ser5.key[g], side="right") - 1
    out = np.zeros(len(g), np.int64)
    ok = q >= 0
    out[ok] = d[q[ok]]
    return out


def _win(S, g):
    return (S.start[g] >= W_START) & (S.end[g] <= W_END)


def _ok(*xs):
    return all(x is not None and np.isfinite(x) for x in xs)


def _vw(s, sig):
    return float(s.vwap[sig])


def _nifty_side(s, sig):
    """+1 if NIFTY at this minute is above its 09:15 value, -1 below, 0 otherwise / missing."""
    nf = s.nifty
    if nf is None or not np.isfinite(nf[0]) or not np.isfinite(nf[sig]):
        return 0
    return 1 if nf[sig] > nf[0] else (-1 if nf[sig] < nf[0] else 0)


def _atr_entry(s, sig, side, c, atr, k, rr, hold=None, trail=None):
    if not _ok(c, atr) or atr <= 0:
        return None
    return Entry(int(sig), side, stop=float(c - side * k * atr), rr=rr, time_stop=hold, trail=trail)


def _run(s, sig_fn):
    """Loop over today's eligible 5m candles; sig_fn(j_local, gj, sig, S, g) -> Entry | None."""
    S, g, sig = _today(s, 5)
    if len(g) < 2:
        return []
    w = _win(S, g)
    out = []
    for jl in np.flatnonzero(w):
        sb = int(sig[jl])
        if sb < 0 or sb >= len(s) or s.t[sb] >= SQ_OFF:
            continue
        e = sig_fn(int(jl), int(g[jl]), sb, S, g)
        if e is not None:
            out.append(e)
    return out


def _x_up(a, b, j):
    return a[j] > b[j] and a[j - 1] <= b[j - 1]


def _x_dn(a, b, j):
    return a[j] < b[j] and a[j - 1] >= b[j - 1]


# ── ema_cross ───────────────────────────────────────────────────────────────
def _ema_cross(s, p, nifty):
    S, g, _ = _today(s, 5)
    if len(g) < 2:
        return []
    tr15 = _trend15(s, S, g)
    e9, e21, atr = A(S, "ema9"), A(S, "ema21"), A(S, "atr14")

    def f(jl, j, sb, S, g):
        side = 0
        if _x_up(e9, e21, j) and tr15[jl] == 1:
            side = LONG
        elif _x_dn(e9, e21, j) and tr15[jl] == -1:
            side = SHORT
        if not side:
            return None
        if nifty:
            if _nifty_side(s, sb) != side or np.sign(S.c[j] - _vw(s, sb)) != side:
                return None
        return _atr_entry(s, sb, side, S.c[j], atr[j], p.atr_stop, p.rr)
    return _run(s, f)


@strategy("r3_tf5m_ema_cross_15mtrend", "intraday", "ema_cross",
          "5m EMA9 crosses EMA21 in the direction of the 15m trend (close vs rising/falling EMA20); 1.5 ATR stop, 2R",
          {"fast": 9, "slow": 21, "ema15": 20, "atr_stop": 1.5, "rr": 2.0}, max_per_day=2)
def r3_tf5m_ema_cross_15mtrend(s, p):
    return _ema_cross(s, p, False)


@strategy("r3_tf5m_ema_cross_15m_nifty", "intraday", "ema_cross",
          "5m EMA9/21 cross with the 15m trend, close on the trade side of VWAP and NIFTY on the trade side of its 09:15 value; 1.5 ATR stop, 2R",
          {"fast": 9, "slow": 21, "ema15": 20, "atr_stop": 1.5, "rr": 2.0}, max_per_day=2, needs_index=True)
def r3_tf5m_ema_cross_15m_nifty(s, p):
    return _ema_cross(s, p, True)


# ── rsi ─────────────────────────────────────────────────────────────────────
@strategy("r3_tf5m_rsi_reentry_vwap", "intraday", "rsi",
          "5m RSI(14) back above 30 with close >= 1 ATR below VWAP: buy, target VWAP, 1.5 ATR stop, 60 min (mirror at 70)",
          {"lo": 30, "hi": 70, "vwap_gap_atr": 1.0, "atr_stop": 1.5, "hold": 60}, max_per_day=2)
def r3_tf5m_rsi_reentry_vwap(s, p):
    S, g, _ = _today(s, 5)
    r, atr = A(S, "rsi14"), A(S, "atr14")

    def f(jl, j, sb, S, g):
        a, c, vw = atr[j], S.c[j], _vw(s, sb)
        if not _ok(a, c, vw, r[j], r[j - 1]) or a <= 0:
            return None
        if r[j - 1] <= p.lo < r[j] and c <= vw - p.vwap_gap_atr * a:
            return Entry(sb, LONG, stop=float(c - p.atr_stop * a), target=float(vw), time_stop=p.hold)
        if r[j - 1] >= p.hi > r[j] and c >= vw + p.vwap_gap_atr * a:
            return Entry(sb, SHORT, stop=float(c + p.atr_stop * a), target=float(vw), time_stop=p.hold)
        return None
    return _run(s, f)


@strategy("r3_tf5m_rsi_pullback_trend", "intraday", "rsi",
          "15m trend up and close above VWAP: 5m RSI(14) crosses back above 40 -> buy (mirror 60 / down / below VWAP); 1.5 ATR stop, 2R",
          {"lo": 40, "hi": 60, "atr_stop": 1.5, "rr": 2.0}, max_per_day=2)
def r3_tf5m_rsi_pullback_trend(s, p):
    S, g, _ = _today(s, 5)
    if len(g) < 2:
        return []
    tr15 = _trend15(s, S, g)
    r, atr = A(S, "rsi14"), A(S, "atr14")

    def f(jl, j, sb, S, g):
        c, vw = S.c[j], _vw(s, sb)
        if not _ok(r[j], r[j - 1], vw):
            return None
        if tr15[jl] == 1 and c > vw and r[j - 1] <= p.lo < r[j]:
            return _atr_entry(s, sb, LONG, c, atr[j], p.atr_stop, p.rr)
        if tr15[jl] == -1 and c < vw and r[j - 1] >= p.hi > r[j]:
            return _atr_entry(s, sb, SHORT, c, atr[j], p.atr_stop, p.rr)
        return None
    return _run(s, f)


# ── bollinger ───────────────────────────────────────────────────────────────
@strategy("r3_tf5m_bb_squeeze_break", "intraday", "bollinger",
          "Previous 5m BB(20,2) bandwidth lowest of 120 candles, then a close outside the band: follow; stop middle band (>= 1 ATR), 2R",
          {"n": 20, "k": 2.0, "look": 120, "min_stop_atr": 1.0, "rr": 2.0}, max_per_day=2)
def r3_tf5m_bb_squeeze_break(s, p):
    S, g, _ = _today(s, 5)
    m, sd, atr = A(S, "sma20"), A(S, "sd20"), A(S, "atr14")
    bw = S.get("bbw", lambda S: 2 * p.k * A(S, "sd20") / A(S, "sma20"))
    bwmin = S.get("bbw_min120", lambda S: _ll(bw, p.look))

    def f(jl, j, sb, S, g):
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


@strategy("r3_tf5m_ttm_squeeze_fire", "intraday", "bollinger",
          "TTM squeeze: BB(20,2) inside Keltner(SMA20 +- 1.5 ATR20) for >= 6 candles then released; trade the momentum sign; 1.5 ATR stop, 2R",
          {"n": 20, "k": 2.0, "kc": 1.5, "min_on": 6, "atr_stop": 1.5, "rr": 2.0}, max_per_day=2)
def r3_tf5m_ttm_squeeze_fire(s, p):
    S, g, _ = _today(s, 5)

    def mk(S):
        m, sd, a20 = A(S, "sma20"), A(S, "sd20"), A(S, "atr20")
        on = (m + p.k * sd < m + p.kc * a20) & (m - p.k * sd > m - p.kc * a20)
        mom = S.c - ((A(S, "hh20") + A(S, "ll20")) / 2 + m) / 2
        return on, mom
    on, mom = S.get("ttm", mk)
    atr = A(S, "atr14")

    def f(jl, j, sb, S, g):
        if j < p.min_on or on[j] or not on[j - p.min_on:j].all() or not _ok(mom[j]):
            return None
        side = LONG if mom[j] > 0 else (SHORT if mom[j] < 0 else 0)
        if not side:
            return None
        return _atr_entry(s, sb, side, S.c[j], atr[j], p.atr_stop, p.rr)
    return _run(s, f)


@strategy("r3_tf5m_bb_fade_adx", "intraday", "bollinger",
          "ADX(14) < 20: previous 5m close outside BB(20,2), this close back inside -> fade to the middle band; stop 2-candle extreme +- 0.25 ATR, 60 min",
          {"n": 20, "k": 2.0, "adx_max": 20, "pad_atr": 0.25, "hold": 60}, max_per_day=2)
def r3_tf5m_bb_fade_adx(s, p):
    S, g, _ = _today(s, 5)
    m, sd, atr = A(S, "sma20"), A(S, "sd20"), A(S, "atr14")
    _, _, adx = A(S, "dmi")

    def f(jl, j, sb, S, g):
        c, pc, a = S.c[j], S.c[j - 1], atr[j]
        if not _ok(m[j], m[j - 1], sd[j], sd[j - 1], a, adx[j]) or a <= 0 or adx[j] >= p.adx_max:
            return None
        if pc > m[j - 1] + p.k * sd[j - 1] and c < m[j] + p.k * sd[j] and c > m[j]:
            return Entry(sb, SHORT, stop=float(max(S.h[j], S.h[j - 1]) + p.pad_atr * a), target=float(m[j]),
                         time_stop=p.hold)
        if pc < m[j - 1] - p.k * sd[j - 1] and c > m[j] - p.k * sd[j] and c < m[j]:
            return Entry(sb, LONG, stop=float(min(S.l[j], S.l[j - 1]) - p.pad_atr * a), target=float(m[j]),
                         time_stop=p.hold)
        return None
    return _run(s, f)


# ── supertrend ──────────────────────────────────────────────────────────────
def _st_flip(s, p, agree15):
    S, g, _ = _today(s, 5)
    if len(g) < 2:
        return []
    d, line = A(S, "st")
    d15 = _st15(s, S, g) if agree15 else None

    def f(jl, j, sb, S, g):
        if d[j] == 0 or d[j - 1] == 0 or d[j] == d[j - 1] or not _ok(line[j]):
            return None
        side = int(d[j])
        if agree15 and d15[jl] != side:
            return None
        if (S.c[j] - line[j]) * side <= 0:
            return None
        return Entry(sb, side, stop=float(line[j]), trail=("atr", p.trail_atr))
    return _run(s, f)


@strategy("r3_tf5m_supertrend_flip", "intraday", "supertrend",
          "5m Supertrend(10,3) flips: follow, stop at the Supertrend line, trail 3 x 5m ATR, hold to square-off",
          {"n": 10, "mult": 3.0, "trail_atr": 3.0}, max_per_day=2)
def r3_tf5m_supertrend_flip(s, p):
    return _st_flip(s, p, False)


@strategy("r3_tf5m_supertrend_15m_agree", "intraday", "supertrend",
          "5m Supertrend(10,3) flip only when the 15m Supertrend(10,3) already points the same way; stop at the line, trail 3 ATR",
          {"n": 10, "mult": 3.0, "trail_atr": 3.0}, max_per_day=2)
def r3_tf5m_supertrend_15m_agree(s, p):
    return _st_flip(s, p, True)


# ── macd ────────────────────────────────────────────────────────────────────
def _macd(s, p, dtrend):
    S, g, _ = _today(s, 5)
    hst, atr = A(S, "macdh"), A(S, "atr14")
    pdi, mdi, adx = A(S, "dmi")
    dsig = 0
    if dtrend:
        if not _ok(s.pdc, s.dsma20):
            return []
        dsig = 1 if s.pdc > s.dsma20 else (-1 if s.pdc < s.dsma20 else 0)

    def f(jl, j, sb, S, g):
        if not _ok(hst[j], hst[j - 1], adx[j], pdi[j], mdi[j]) or adx[j] <= p.adx_min:
            return None
        side = 0
        if hst[j] > 0 >= hst[j - 1] and pdi[j] > mdi[j]:
            side = LONG
        elif hst[j] < 0 <= hst[j - 1] and mdi[j] > pdi[j]:
            side = SHORT
        if not side:
            return None
        if dtrend and (dsig != side or np.sign(S.c[j] - _vw(s, sb)) != side):
            return None
        return _atr_entry(s, sb, side, S.c[j], atr[j], p.atr_stop, p.rr)
    return _run(s, f)


@strategy("r3_tf5m_macd_zero_adx", "intraday", "macd",
          "5m MACD(12,26,9) histogram crosses zero with ADX(14) > 20 and DI agreeing; 1.5 ATR stop, 2R",
          {"fast": 12, "slow": 26, "sig": 9, "adx_min": 20, "atr_stop": 1.5, "rr": 2.0}, max_per_day=2)
def r3_tf5m_macd_zero_adx(s, p):
    return _macd(s, p, False)


@strategy("r3_tf5m_macd_zero_dtrend", "intraday", "macd",
          "macd_zero_adx only with the daily trend (yesterday's close vs 20-day SMA) and close on the same side of VWAP",
          {"fast": 12, "slow": 26, "sig": 9, "adx_min": 20, "atr_stop": 1.5, "rr": 2.0}, max_per_day=2)
def r3_tf5m_macd_zero_dtrend(s, p):
    return _macd(s, p, True)


# ── keltner ─────────────────────────────────────────────────────────────────
def _keltner(s, p, vol):
    S, g, _ = _today(s, 5)
    e, a10, atr = A(S, "ema20"), A(S, "atr10"), A(S, "atr14")
    vm = A(S, "vmean20")

    def f(jl, j, sb, S, g):
        c, pc = S.c[j], S.c[j - 1]
        if not _ok(e[j], e[j - 1], a10[j], a10[j - 1], atr[j]) or atr[j] <= 0:
            return None
        up, pup = e[j] + p.mult * a10[j], e[j - 1] + p.mult * a10[j - 1]
        lo, plo = e[j] - p.mult * a10[j], e[j - 1] - p.mult * a10[j - 1]
        side = LONG if (c > up and pc <= pup) else (SHORT if (c < lo and pc >= plo) else 0)
        if not side:
            return None
        if vol and not (_ok(vm[j]) and vm[j] > 0 and S.v[j] >= p.vol_x * vm[j]):
            return None
        if side == LONG:
            stop = min(e[j], c - p.min_stop_atr * atr[j])
        else:
            stop = max(e[j], c + p.min_stop_atr * atr[j])
        return Entry(sb, side, stop=float(stop), rr=p.rr)
    return _run(s, f)


@strategy("r3_tf5m_keltner_break", "intraday", "keltner",
          "5m close crosses outside Keltner (EMA20 +- 2 x ATR10): follow; stop EMA20 (>= 1 ATR), 2R",
          {"ema": 20, "atr": 10, "mult": 2.0, "min_stop_atr": 1.0, "rr": 2.0}, max_per_day=2)
def r3_tf5m_keltner_break(s, p):
    return _keltner(s, p, False)


@strategy("r3_tf5m_keltner_break_vol", "intraday", "keltner",
          "keltner_break only when the breakout candle's volume >= 2 x the mean of the previous 20 5m candles",
          {"ema": 20, "atr": 10, "mult": 2.0, "min_stop_atr": 1.0, "rr": 2.0, "vol_x": 2.0}, max_per_day=2)
def r3_tf5m_keltner_break_vol(s, p):
    return _keltner(s, p, True)


# ── donchian ────────────────────────────────────────────────────────────────
def _donchian(s, p, nifty):
    S, g, _ = _today(s, 5)
    hh, ll, hx, lx, atr = A(S, "hh20p"), A(S, "ll20p"), A(S, "hh10p"), A(S, "ll10p"), A(S, "atr14")
    t0 = hhmm(p.start)

    def f(jl, j, sb, S, g):
        if S.start[j] < t0:
            return None
        c, pc = S.c[j], S.c[j - 1]
        if not _ok(hh[j], ll[j], hh[j - 1], ll[j - 1], hx[j], lx[j], atr[j]) or atr[j] <= 0:
            return None
        side = LONG if (c > hh[j] and pc <= hh[j - 1]) else (SHORT if (c < ll[j] and pc >= ll[j - 1]) else 0)
        if not side:
            return None
        if nifty and (_nifty_side(s, sb) != side or np.sign(c - _vw(s, sb)) != side):
            return None
        if side == LONG:
            stop = max(lx[j], c - p.max_stop_atr * atr[j])
        else:
            stop = min(hx[j], c + p.max_stop_atr * atr[j])
        if (c - stop) * side <= 0:
            return None
        return Entry(sb, side, stop=float(stop), trail=("atr", p.trail_atr))
    return _run(s, f)


@strategy("r3_tf5m_donchian20", "intraday", "donchian",
          "10:00-14:30 first 5m close beyond the previous 20-candle channel: follow; stop 10-candle extreme (<= 3 ATR), trail 3 ATR, square-off",
          {"n": 20, "exit_n": 10, "max_stop_atr": 3.0, "trail_atr": 3.0, "start": "10:00"}, max_per_day=2)
def r3_tf5m_donchian20(s, p):
    return _donchian(s, p, False)


@strategy("r3_tf5m_donchian20_nifty", "intraday", "donchian",
          "donchian20 only when NIFTY is on the trade side of its 09:15 value and the close on the trade side of VWAP",
          {"n": 20, "exit_n": 10, "max_stop_atr": 3.0, "trail_atr": 3.0, "start": "10:00"}, max_per_day=2,
          needs_index=True)
def r3_tf5m_donchian20_nifty(s, p):
    return _donchian(s, p, True)


# ── heikin-ashi ─────────────────────────────────────────────────────────────
@strategy("r3_tf5m_ha_trend_start", "intraday", "heikin_ashi",
          "Two 5m Heikin-Ashi candles bullish with no lower shadow after a bearish one, close above VWAP: buy (mirror); 1.5 ATR stop, 2R",
          {"atr_stop": 1.5, "rr": 2.0}, max_per_day=2)
def r3_tf5m_ha_trend_start(s, p):
    S, g, _ = _today(s, 5)
    ho, hc, hh, hl = A(S, "ha")
    atr = A(S, "atr14")
    eps = 1e-9

    def bull_strong(j):
        return hc[j] > ho[j] and hl[j] >= ho[j] - eps

    def bear_strong(j):
        return hc[j] < ho[j] and hh[j] <= ho[j] + eps

    def f(jl, j, sb, S, g):
        if j < 2:
            return None
        c, vw = S.c[j], _vw(s, sb)
        if bull_strong(j) and bull_strong(j - 1) and hc[j - 2] < ho[j - 2] and c > vw:
            return _atr_entry(s, sb, LONG, c, atr[j], p.atr_stop, p.rr)
        if bear_strong(j) and bear_strong(j - 1) and hc[j - 2] > ho[j - 2] and c < vw:
            return _atr_entry(s, sb, SHORT, c, atr[j], p.atr_stop, p.rr)
        return None
    return _run(s, f)


@strategy("r3_tf5m_ha_run_fade", "intraday", "heikin_ashi",
          ">= 8 bullish 5m Heikin-Ashi candles moving >= 3 ATR, then a bearish HA candle: short (mirror); stop run extreme + 0.25 ATR, 1.5R, 60 min",
          {"run": 8, "min_move_atr": 3.0, "pad_atr": 0.25, "rr": 1.5, "hold": 60}, max_per_day=2)
def r3_tf5m_ha_run_fade(s, p):
    S, g, _ = _today(s, 5)
    ho, hc, _, _ = A(S, "ha")
    atr = A(S, "atr14")
    bull = hc > ho
    bear = hc < ho

    def runback(mask, j):
        k = 0
        while j - k >= 0 and mask[j - k]:
            k += 1
        return k

    def f(jl, j, sb, S, g):
        a = atr[j]
        if not _ok(a) or a <= 0:
            return None
        if bear[j]:
            k = runback(bull, j - 1)
            if k >= p.run:
                j0 = j - k
                if S.c[j - 1] - S.o[j0] >= p.min_move_atr * a:
                    return Entry(sb, SHORT, stop=float(S.h[j0:j + 1].max() + p.pad_atr * a), rr=p.rr,
                                 time_stop=p.hold)
        if bull[j]:
            k = runback(bear, j - 1)
            if k >= p.run:
                j0 = j - k
                if S.o[j0] - S.c[j - 1] >= p.min_move_atr * a:
                    return Entry(sb, LONG, stop=float(S.l[j0:j + 1].min() - p.pad_atr * a), rr=p.rr,
                                 time_stop=p.hold)
        return None
    return _run(s, f)


# ── volume ──────────────────────────────────────────────────────────────────
@strategy("r3_tf5m_volshock_follow", "intraday", "volume",
          "5m candle with >= 4x the volume of the previous 50 candles, range >= 1.5 ATR, close in its top quarter above VWAP: buy (mirror); stop candle low, 2R",
          {"vol_x": 4.0, "vol_n": 50, "range_atr": 1.5, "close_frac": 0.25, "pad_atr": 0.1, "rr": 2.0}, max_per_day=2)
def r3_tf5m_volshock_follow(s, p):
    S, g, _ = _today(s, 5)
    vm, atr = A(S, "vmean50"), A(S, "atr14")

    def f(jl, j, sb, S, g):
        a = atr[j - 1]
        if not _ok(vm[j], a) or vm[j] <= 0 or a <= 0:
            return None
        h, l, c = S.h[j], S.l[j], S.c[j]
        rng = h - l
        if S.v[j] < p.vol_x * vm[j] or rng < p.range_atr * a:
            return None
        vw = _vw(s, sb)
        if c >= h - p.close_frac * rng and c > vw:
            return Entry(sb, LONG, stop=float(l - p.pad_atr * a), rr=p.rr)
        if c <= l + p.close_frac * rng and c < vw:
            return Entry(sb, SHORT, stop=float(h + p.pad_atr * a), rr=p.rr)
        return None
    return _run(s, f)


@strategy("r3_tf5m_obv_lead", "intraday", "volume",
          "5m OBV above its previous 20-candle high while price is not above its 20-candle closing high, close above VWAP: buy (mirror); 1.5 ATR stop, 2R",
          {"n": 20, "atr_stop": 1.5, "rr": 2.0}, max_per_day=2)
def r3_tf5m_obv_lead(s, p):
    S, g, _ = _today(s, 5)
    obv = A(S, "obv")
    oh = S.get("obv_hh20p", lambda S: _sh(_hh(obv, p.n)))
    ol = S.get("obv_ll20p", lambda S: _sh(_ll(obv, p.n)))
    ch = S.get("c_hh20p", lambda S: _sh(_hh(S.c, p.n)))
    cl = S.get("c_ll20p", lambda S: _sh(_ll(S.c, p.n)))
    atr = A(S, "atr14")

    def f(jl, j, sb, S, g):
        if not _ok(oh[j], ol[j], ch[j], cl[j]):
            return None
        c, vw = S.c[j], _vw(s, sb)
        if obv[j] > oh[j] and c <= ch[j] and c > vw:
            return _atr_entry(s, sb, LONG, c, atr[j], p.atr_stop, p.rr)
        if obv[j] < ol[j] and c >= cl[j] and c < vw:
            return _atr_entry(s, sb, SHORT, c, atr[j], p.atr_stop, p.rr)
        return None
    return _run(s, f)


# ── stochastic ──────────────────────────────────────────────────────────────
@strategy("r3_tf5m_stoch_cross_trend", "intraday", "stochastic",
          "5m Stochastic(14,3,3) %K crosses above %D with %D < 20 in a 15m uptrend: buy (mirror > 80, downtrend); 1.5 ATR stop, 2R",
          {"n": 14, "k": 3, "d": 3, "lo": 20, "hi": 80, "atr_stop": 1.5, "rr": 2.0}, max_per_day=2)
def r3_tf5m_stoch_cross_trend(s, p):
    S, g, _ = _today(s, 5)
    if len(g) < 2:
        return []

    def mk(S):
        hh, ll = _hh(S.h, 14), _ll(S.l, 14)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(hh > ll, 100 * (S.c - ll) / (hh - ll), 50.0)
        raw[~np.isfinite(hh)] = np.nan
        k = _sma(raw, 3)
        return k, _sma(k, 3)
    K, D = S.get("stoch", mk)
    tr15 = _trend15(s, S, g)
    atr = A(S, "atr14")

    def f(jl, j, sb, S, g):
        if not _ok(K[j], D[j], K[j - 1], D[j - 1]):
            return None
        if tr15[jl] == 1 and _x_up(K, D, j) and D[j] < p.lo:
            return _atr_entry(s, sb, LONG, S.c[j], atr[j], p.atr_stop, p.rr)
        if tr15[jl] == -1 and _x_dn(K, D, j) and D[j] > p.hi:
            return _atr_entry(s, sb, SHORT, S.c[j], atr[j], p.atr_stop, p.rr)
        return None
    return _run(s, f)


# ── psar ────────────────────────────────────────────────────────────────────
@strategy("r3_tf5m_psar_adx", "intraday", "psar",
          "5m Parabolic SAR(0.02,0.02,0.2) flips with ADX(14) > 25: follow, stop at the new SAR, 2R",
          {"af0": 0.02, "step": 0.02, "afmax": 0.2, "adx_min": 25, "rr": 2.0}, max_per_day=2)
def r3_tf5m_psar_adx(s, p):
    S, g, _ = _today(s, 5)
    d, sar = A(S, "psar")
    _, _, adx = A(S, "dmi")

    def f(jl, j, sb, S, g):
        if d[j] == 0 or d[j - 1] == 0 or d[j] == d[j - 1] or not _ok(sar[j], adx[j]) or adx[j] <= p.adx_min:
            return None
        side = int(d[j])
        if (S.c[j] - sar[j]) * side <= 0:
            return None
        return Entry(sb, side, stop=float(sar[j]), rr=p.rr)
    return _run(s, f)


# ── cci ─────────────────────────────────────────────────────────────────────
@strategy("r3_tf5m_cci_100", "intraday", "cci",
          "5m CCI(20) crosses above +100: buy; below -100: short (Lambert); 1.5 ATR stop, 2R, 60 min",
          {"n": 20, "level": 100, "atr_stop": 1.5, "rr": 2.0, "hold": 60}, max_per_day=2)
def r3_tf5m_cci_100(s, p):
    S, g, _ = _today(s, 5)
    cci, atr = A(S, "cci"), A(S, "atr14")

    def f(jl, j, sb, S, g):
        if not _ok(cci[j], cci[j - 1]):
            return None
        if cci[j - 1] <= p.level < cci[j]:
            return _atr_entry(s, sb, LONG, S.c[j], atr[j], p.atr_stop, p.rr, hold=p.hold)
        if cci[j - 1] >= -p.level > cci[j]:
            return _atr_entry(s, sb, SHORT, S.c[j], atr[j], p.atr_stop, p.rr, hold=p.hold)
        return None
    return _run(s, f)


# ── dmi ─────────────────────────────────────────────────────────────────────
@strategy("r3_tf5m_dmi_cross", "intraday", "dmi",
          "5m +DI(14) crosses above -DI with ADX >= 25 and rising: buy (mirror); 1.5 ATR stop, 2R",
          {"n": 14, "adx_min": 25, "atr_stop": 1.5, "rr": 2.0}, max_per_day=2)
def r3_tf5m_dmi_cross(s, p):
    S, g, _ = _today(s, 5)
    pdi, mdi, adx = A(S, "dmi")
    atr = A(S, "atr14")

    def f(jl, j, sb, S, g):
        if not _ok(pdi[j], mdi[j], pdi[j - 1], mdi[j - 1], adx[j], adx[j - 1]):
            return None
        if adx[j] < p.adx_min or adx[j] <= adx[j - 1]:
            return None
        if _x_up(pdi, mdi, j):
            return _atr_entry(s, sb, LONG, S.c[j], atr[j], p.atr_stop, p.rr)
        if _x_dn(pdi, mdi, j):
            return _atr_entry(s, sb, SHORT, S.c[j], atr[j], p.atr_stop, p.rr)
        return None
    return _run(s, f)


# ── ichimoku ────────────────────────────────────────────────────────────────
@strategy("r3_tf5m_ichimoku_tk", "intraday", "ichimoku",
          "5m Tenkan(9) crosses Kijun(26) with the close beyond the cloud (spans of 26 candles earlier, B=52): follow; stop Kijun (>= 1 ATR), 2R",
          {"tenkan": 9, "kijun": 26, "senkou_b": 52, "disp": 26, "min_stop_atr": 1.0, "rr": 2.0}, max_per_day=2)
def r3_tf5m_ichimoku_tk(s, p):
    S, g, _ = _today(s, 5)

    def mk(S):
        tk = (_hh(S.h, 9) + _ll(S.l, 9)) / 2
        kj = (_hh(S.h, 26) + _ll(S.l, 26)) / 2
        sa = _sh((tk + kj) / 2, 26)
        sb_ = _sh((_hh(S.h, 52) + _ll(S.l, 52)) / 2, 26)
        return tk, kj, np.fmax(sa, sb_), np.fmin(sa, sb_)
    tk, kj, ctop, cbot = S.get("ichi", mk)
    atr = A(S, "atr14")

    def f(jl, j, sb, S, g):
        if not _ok(tk[j], kj[j], tk[j - 1], kj[j - 1], ctop[j], cbot[j], atr[j]) or atr[j] <= 0:
            return None
        c = S.c[j]
        if _x_up(tk, kj, j) and c > ctop[j]:
            return Entry(sb, LONG, stop=float(min(kj[j], c - p.min_stop_atr * atr[j])), rr=p.rr)
        if _x_dn(tk, kj, j) and c < cbot[j]:
            return Entry(sb, SHORT, stop=float(max(kj[j], c + p.min_stop_atr * atr[j])), rr=p.rr)
        return None
    return _run(s, f)


# ── opening candle relative volume ──────────────────────────────────────────
@strategy("r3_tf5m_open5_rvol", "intraday", "volume_open",
          "First 5m candle with >= 3x the mean first-candle volume of 10 sessions, body >= 60 % of range, same sign as the gap: follow at 09:20, stop its other end",
          {"rvol": 3.0, "days": 10, "body_frac": 0.6}, max_per_day=1)
def r3_tf5m_open5_rvol(s, p):
    S, g, sig = _today(s, 5)
    if len(g) < 1 or S.start[g[0]] != SESSION_OPEN or not _ok(s.gap) or s.gap == 0:
        return []
    k = int(getattr(s, "_s", s).day_index)
    if k < p.days:
        return []
    fi = S.sess_lo[k - p.days:k]
    fi = fi[S.start[fi] == SESSION_OPEN]
    if len(fi) < p.days // 2:
        return []
    mv = float(S.v[fi].mean())
    j = int(g[0])
    sb = int(sig[0])
    if sb < 0 or mv <= 0 or S.v[j] < p.rvol * mv:
        return []
    o, h, l, c = S.o[j], S.h[j], S.l[j], S.c[j]
    rng = h - l
    if rng <= 0 or abs(c - o) < p.body_frac * rng:
        return []
    side = LONG if c > o else SHORT
    if np.sign(s.gap) != side:
        return []
    return [Entry(sb, side, stop=float(l if side == LONG else h))]
