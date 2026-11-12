"""
Round-3 TIMEFRAME 15-MINUTE indicator systems (stocks MIS + NIFTY / BANKNIFTY index futures).
Pre-registered BEFORE any result in scratchpad/r3/tf15m/PREREGISTERED.txt — textbook / round parameters, not tuned.

Conventions (see the pre-registration):
* 15m candles aligned at 09:15 (candles.resample). Indicators run continuously over up to HIST complete 15m candles of
  EARLIER sessions + today's COMPLETED candles (the forming candle is never seen), so every value is causal.
* A signal on candle j is decided at the close of its completion bar `sig[j]`; the simulator fills at the next open.
* Default window: candles ending 09:45 .. 14:30. Long and short (mirror). max_per_day = 1.
"""
from __future__ import annotations

import weakref
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .. import candles as _cd
from ..api import LONG, SHORT, Entry, strategy
from ..data import SESSION_OPEN
from ..indicators import atr as _atr, ema as _ema, rsi as _rsi, sma as _sma

TF = 15
HIST = 200                       # earlier-session 15m candles prepended (indicator warm-up)
SLOT_SESSIONS = 10               # slot volume ratio: same-slot mean of the previous 10 sessions
SLOT_MIN = 5                     # ... needing at least 5 of them


def _m(hm: str) -> int:
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


# ── history candles with volume (per SymbolData, cached) ────────────────────
_HISTC: dict = {}


def _hist_candles(sd):
    """All 15m candles of every session of the symbol: o h l c v, session index, start minute, first flag."""
    ent = _HISTC.get(id(sd))
    if ent is not None and ent[0]() is sd:
        return ent[1]
    if len(_HISTC) > 64:
        _HISTC.clear()
    lens = sd.ends - sd.starts
    sess_of = np.repeat(np.arange(len(sd.starts)), lens)
    t = sd.t.astype(np.int64)
    bid = sess_of.astype(np.int64) * 1000 + (t - SESSION_OPEN) // TF
    cut = np.flatnonzero(bid[1:] != bid[:-1]) + 1
    st = np.concatenate(([0], cut)).astype(np.int64)
    en = np.concatenate((cut, [len(bid)])).astype(np.int64) - 1
    sess = sess_of[st]
    first = np.ones(len(st), bool)
    first[1:] = sess[1:] != sess[:-1]
    start = SESSION_OPEN + ((t[st] - SESSION_OPEN) // TF) * TF
    got = SimpleNamespace(
        o=sd.o[st].astype(float), h=np.maximum.reduceat(sd.h, st).astype(float),
        l=np.minimum.reduceat(sd.l, st).astype(float), c=sd.c[en].astype(float),
        v=np.add.reduceat(sd.v, st).astype(float), sess=sess, start=start, first=first)
    # slot volume matrix [session, slot]
    nslot = (15 * 60 + 30 - SESSION_OPEN) // TF
    vm = np.full((len(sd.starts), nslot), np.nan)
    slot = (start - SESSION_OPEN) // TF
    ok = (slot >= 0) & (slot < nslot)
    vm[sess[ok], slot[ok]] = got.v[ok]
    got.vm = vm
    _HISTC[id(sd)] = (weakref.ref(sd), got)
    return got


# ── the continuous 15m series and its indicators ────────────────────────────
class _Ser:
    """Earlier-session candles (P of them) + today's complete candles; indicators computed lazily."""

    def __init__(self, s):
        base = getattr(s, "_s", s)
        today = _cd.resample(s.t, s.o, s.h, s.l, s.c, s.v, TF)
        self.m = m = len(today.c)
        self.today = today
        sd, k = getattr(base, "_sd", None), getattr(base, "day_index", 0)
        self.slot_mean = np.full(m, np.nan)
        if sd is not None and k > 0:
            H = _hist_candles(sd)
            j = int(np.searchsorted(H.sess, k))
            a = max(0, j - HIST)
            ph = SimpleNamespace(o=H.o[a:j], h=H.h[a:j], l=H.l[a:j], c=H.c[a:j], v=H.v[a:j], first=H.first[a:j])
            if m:
                vm = H.vm[max(0, k - SLOT_SESSIONS):k]
                cnt = np.sum(np.isfinite(vm) & (vm > 0), axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.nansum(np.where(vm > 0, vm, np.nan), axis=0) / np.maximum(cnt, 1)
                mean = np.where(cnt >= SLOT_MIN, mean, np.nan)
                slot = (today.start - SESSION_OPEN) // TF
                okk = (slot >= 0) & (slot < len(mean))
                self.slot_mean[okk] = mean[slot[okk]]
        else:
            z = np.zeros(0)
            ph = SimpleNamespace(o=z, h=z, l=z, c=z, v=z, first=np.zeros(0, bool))
        self.P = P = len(ph.c)
        self.o = np.concatenate((ph.o, today.o))
        self.h = np.concatenate((ph.h, today.h))
        self.l = np.concatenate((ph.l, today.l))
        self.c = np.concatenate((ph.c, today.c))
        self.v = np.concatenate((ph.v, today.v))
        self.first = np.concatenate((ph.first, np.arange(m) == 0))
        self._cache = {}

    def get(self, name):
        if name not in self._cache:
            self._cache[name] = getattr(self, "_" + name)()
        return self._cache[name]

    # indicators over the whole series
    def _atr14(self):
        a = np.array(_atr(self.h, self.l, self.c, 14, self.first), float)
        a[:14] = np.nan
        return a

    def _ema20(self):
        return _ema(self.c, 20)

    def _ema50(self):
        return _ema(self.c, 50)

    def _sma20(self):
        return _sma(self.c, 20)

    def _sd20(self):
        return pd.Series(self.c).rolling(20).std(ddof=0).to_numpy()

    def _rsi14(self):
        return _rsi(self.c, 14)

    def _macd(self):
        line = _ema(self.c, 12) - _ema(self.c, 26)
        sig = _ema(line, 9)
        line = line.copy()
        line[:35] = np.nan
        return line, sig

    def _adx(self):
        h, l, c = self.h, self.l, self.c
        up = np.zeros(len(h)); dn = np.zeros(len(h))
        up[1:] = h[1:] - h[:-1]
        dn[1:] = l[:-1] - l[1:]
        pdm = np.where((up > dn) & (up > 0), up, 0.0)
        mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr = np.maximum(h - l, np.abs(h - np.r_[c[0], c[:-1]]))
        tr = np.maximum(tr, np.abs(l - np.r_[c[0], c[:-1]]))
        al = 1.0 / 14
        str_ = pd.Series(tr).ewm(alpha=al, adjust=False).mean().to_numpy()
        spd = pd.Series(pdm).ewm(alpha=al, adjust=False).mean().to_numpy()
        smd = pd.Series(mdm).ewm(alpha=al, adjust=False).mean().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            pdi = 100 * spd / str_
            mdi = 100 * smd / str_
            dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
        dx = np.nan_to_num(dx, nan=0.0)
        adx = np.array(pd.Series(dx).ewm(alpha=al, adjust=False).mean().to_numpy(), float)
        adx[:28] = np.nan
        return adx, pdi, mdi

    def _st(self):
        """Supertrend(10, 3): (line, direction +1/-1); NaN / 0 during warm-up."""
        n = len(self.c)
        a = np.array(_atr(self.h, self.l, self.c, 10, self.first), float)
        hl2 = (self.h + self.l) / 2.0
        bu, bl = hl2 + 3.0 * a, hl2 - 3.0 * a
        fu, fl = bu.copy(), bl.copy()
        d = np.zeros(n, int)
        line = np.full(n, np.nan)
        c = self.c
        for i in range(n):
            if i < 10:
                continue
            if d[i - 1] == 0:
                fu[i], fl[i] = bu[i], bl[i]
                d[i] = 1 if c[i] > hl2[i] else -1
            else:
                fu[i] = bu[i] if (bu[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
                fl[i] = bl[i] if (bl[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
                if d[i - 1] == 1:
                    d[i] = -1 if c[i] < fl[i] else 1
                else:
                    d[i] = 1 if c[i] > fu[i] else -1
            line[i] = fl[i] if d[i] == 1 else fu[i]
        return line, d

    def _cci(self):
        tp = (self.h + self.l + self.c) / 3.0
        n = len(tp)
        out = np.full(n, np.nan)
        if n >= 20:
            w = np.lib.stride_tricks.sliding_window_view(tp, 20)
            mu = w.mean(axis=1)
            mad = np.abs(w - mu[:, None]).mean(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[19:] = (tp[19:] - mu) / (0.015 * mad)
        return out


_MEMO = weakref.WeakKeyDictionary()


def _ser(s):
    base = getattr(s, "_s", s)
    n = len(s)
    try:
        memo = _MEMO.setdefault(base, {})
    except TypeError:
        memo = {}
    got = memo.get(n)
    if got is None:
        if len(memo) > 64:
            memo.clear()
        got = _Ser(s)
        memo[n] = got
    return got if got.m else None


def _prev(x, k=1):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _win(S, lo="09:45", hi="14:30"):
    """Boolean over today's candles: candle end within [lo, hi]."""
    e = S.today.end
    return (e >= _m(lo)) & (e <= _m(hi))


def _emit(S, longs, shorts, stop_l, stop_s, win, rr=None, tgt_l=None, tgt_s=None):
    """Entries for every today-candle j where the long/short condition (full-series arrays) holds in the window."""
    P, sig = S.P, S.today.sig
    out = []
    with np.errstate(invalid="ignore"):
        for side, cond, stp, tgt in ((LONG, longs, stop_l, tgt_l), (SHORT, shorts, stop_s, tgt_s)):
            cond = np.asarray(cond, bool)[P:] & win
            for j in np.flatnonzero(cond):
                st = float(stp[P + j])
                tg = None if tgt is None else float(tgt[P + j])
                if not np.isfinite(st) or (tg is not None and not np.isfinite(tg)):
                    continue
                out.append(Entry(int(sig[j]), side, stop=st, target=tg, rr=None if tg is not None else rr))
    return out


def _dtrend(s):
    if not (np.isfinite(s.pdc) and np.isfinite(s.dsma20)):
        return 0
    return 1 if s.pdc > s.dsma20 else (-1 if s.pdc < s.dsma20 else 0)


# ── EMA20/50 trend pullback ─────────────────────────────────────────────────
def _ema_pb(s, p, dt_filter):
    S = _ser(s)
    if S is None:
        return []
    e20, e50, a = S.get("ema20"), S.get("ema50"), S.get("atr14")
    pc = _prev(S.c)
    pe = _prev(e20)
    with np.errstate(invalid="ignore"):
        lg = (e20 > e50) & (pc > pe) & (S.l <= e20) & (S.c > e20) & (S.c > S.o) & (a > 0)
        sh = (e20 < e50) & (pc < pe) & (S.h >= e20) & (S.c < e20) & (S.c < S.o) & (a > 0)
    if dt_filter:
        d = _dtrend(s)
        lg = lg & (d == 1)
        sh = sh & (d == -1)
    return _emit(S, lg, sh, S.l - p.pad * a, S.h + p.pad * a, _win(S), rr=p.rr)


@strategy("r3_tf15m_ema_pb", "intraday", "tf15m_ema_pullback",
          "15m EMA20 > EMA50, prior close above EMA20, candle dips to EMA20 and closes above it bullish: long, stop low-0.25ATR, 2R (mirror)",
          {"ema": (20, 50), "pad": 0.25, "rr": 2.0})
def ema_pb(s, p):
    return _ema_pb(s, p, False)


@strategy("r3_tf15m_ema_pb_dtrend", "intraday", "tf15m_ema_pullback",
          "15m EMA20/50 pullback (as r3_tf15m_ema_pb) only in the daily trend direction (pdc vs 20-day SMA), stop low-0.25ATR, 2R",
          {"ema": (20, 50), "pad": 0.25, "rr": 2.0})
def ema_pb_dtrend(s, p):
    return _ema_pb(s, p, True)


# ── RSI(14) mean reversion ──────────────────────────────────────────────────
def _rsi_mr(s, p, adx_max):
    S = _ser(s)
    if S is None:
        return []
    r, a, m20 = S.get("rsi14"), S.get("atr14"), S.get("sma20")
    rp = _prev(r)
    with np.errstate(invalid="ignore"):
        lg = (rp < p.lo) & (r >= p.lo) & (a > 0) & (m20 > S.c)
        sh = (rp > p.hi) & (r <= p.hi) & (a > 0) & (m20 < S.c)
        if adx_max is not None:
            adx = S.get("adx")[0]
            lg = lg & (adx < adx_max)
            sh = sh & (adx < adx_max)
    return _emit(S, lg, sh, S.c - p.stop_atr * a, S.c + p.stop_atr * a, _win(S), tgt_l=m20, tgt_s=m20)


@strategy("r3_tf15m_rsi_mr", "intraday", "tf15m_rsi_mr",
          "15m RSI(14) crosses back above 30: long to the 15m SMA20, stop 1.5 ATR15 (mirror back below 70)",
          {"rsi": 14, "lo": 30, "hi": 70, "stop_atr": 1.5})
def rsi_mr(s, p):
    return _rsi_mr(s, p, None)


@strategy("r3_tf15m_rsi_mr_adxlow", "intraday", "tf15m_rsi_mr",
          "15m RSI(14) crosses back above 30 with ADX(14) < 20: long to the 15m SMA20, stop 1.5 ATR15 (mirror at 70)",
          {"rsi": 14, "lo": 30, "hi": 70, "stop_atr": 1.5, "adx_max": 20})
def rsi_mr_adxlow(s, p):
    return _rsi_mr(s, p, p.adx_max)


# ── Bollinger ───────────────────────────────────────────────────────────────
def _bb_fade(s, p):
    S = _ser(s)
    if S is None:
        return []
    m, sd, a = S.get("sma20"), S.get("sd20"), S.get("atr14")
    up, dn = m + 2 * sd, m - 2 * sd
    with np.errstate(invalid="ignore"):
        pb = (S.c - dn) / (up - dn)
        pbp = _prev(pb)
        lg = (pbp < 0) & (pb >= 0) & (m > S.c) & (a > 0)
        sh = (pbp > 1) & (pb <= 1) & (m < S.c) & (a > 0)
    lo2 = np.minimum(S.l, _prev(S.l)) - p.pad * a
    hi2 = np.maximum(S.h, _prev(S.h)) + p.pad * a
    return _emit(S, lg, sh, lo2, hi2, _win(S), tgt_l=m, tgt_s=m)


@strategy("r3_tf15m_bb_fade", "intraday", "tf15m_bollinger",
          "15m %B: previous candle closed below BB(20,2), this one closes back inside: long to the middle band, stop 2-candle low-0.25ATR (mirror)",
          {"bb": (20, 2), "pad": 0.25})
def bb_fade(s, p):
    return _bb_fade(s, p)


@strategy("r3_tf15m_bb_fade_nifty", "futures", "tf15m_bollinger",
          "NIFTY future: 15m %B re-entry from below BB(20,2): long to the middle band, stop 2-candle low-0.25ATR (mirror)",
          {"bb": (20, 2), "pad": 0.25}, horizon="intraday", underlying="NIFTY")
def bb_fade_nifty(s, p):
    return _bb_fade(s, p)


@strategy("r3_tf15m_bb_squeeze_break", "intraday", "tf15m_bollinger",
          "Previous 15m BB(20,2) bandwidth lowest of 100 candles, close outside the band: follow, stop middle band (>= 1 ATR), 2R",
          {"bb": (20, 2), "lookback": 100, "min_stop": 1.0, "rr": 2.0})
def bb_squeeze_break(s, p):
    S = _ser(s)
    if S is None:
        return []
    m, sd, a = S.get("sma20"), S.get("sd20"), S.get("atr14")
    with np.errstate(invalid="ignore", divide="ignore"):
        bw = 4 * sd / m
    low = pd.Series(bw).rolling(p.lookback, min_periods=p.lookback).min().to_numpy()
    with np.errstate(invalid="ignore"):
        sq = _prev(bw) <= _prev(low)        # previous candle's bandwidth = min of the previous 100 (incl. itself)
        lg = sq & (S.c > m + 2 * sd) & (a > 0)
        sh = sq & (S.c < m - 2 * sd) & (a > 0)
    sl = np.minimum(m, S.c - p.min_stop * a)
    ss = np.maximum(m, S.c + p.min_stop * a)
    return _emit(S, lg, sh, sl, ss, _win(S), rr=p.rr)


# ── Supertrend(10,3) ────────────────────────────────────────────────────────
def _st_flip(s, p):
    S = _ser(s)
    if S is None:
        return []
    line, d = S.get("st")
    dp = np.r_[0, d[:-1]]
    lg = (dp == -1) & (d == 1)
    sh = (dp == 1) & (d == -1)
    return _emit(S, lg, sh, line, line, _win(S), rr=p.rr)


@strategy("r3_tf15m_st_flip", "intraday", "tf15m_supertrend",
          "15m Supertrend(10,3) flips up: long, stop the Supertrend line, 2R (mirror)",
          {"atr": 10, "mult": 3, "rr": 2.0})
def st_flip(s, p):
    return _st_flip(s, p)


@strategy("r3_tf15m_st_flip_nifty", "futures", "tf15m_supertrend",
          "NIFTY future: 15m Supertrend(10,3) flip, stop the Supertrend line, 2R (both sides)",
          {"atr": 10, "mult": 3, "rr": 2.0}, horizon="intraday", underlying="NIFTY")
def st_flip_nifty(s, p):
    return _st_flip(s, p)


@strategy("r3_tf15m_st_flip_bnf", "futures", "tf15m_supertrend",
          "BANKNIFTY future: 15m Supertrend(10,3) flip, stop the Supertrend line, 2R (both sides)",
          {"atr": 10, "mult": 3, "rr": 2.0}, horizon="intraday", underlying="BANKNIFTY")
def st_flip_bnf(s, p):
    return _st_flip(s, p)


# ── MACD(12,26,9) ───────────────────────────────────────────────────────────
def _macd_cross(s, p, filt):
    S = _ser(s)
    if S is None:
        return []
    line, sig = S.get("macd")
    a = S.get("atr14")
    lp, sp_ = _prev(line), _prev(sig)
    with np.errstate(invalid="ignore"):
        lg = (lp <= sp_) & (line > sig) & (line > 0) & (a > 0)
        sh = (lp >= sp_) & (line < sig) & (line < 0) & (a > 0)
    if filt:
        d = _dtrend(s)
        vw = np.full(len(S.c), np.nan)
        vw[S.P:] = np.asarray(s.vwap, float)[S.today.sig]
        with np.errstate(invalid="ignore"):
            lg = lg & (S.c > vw) & (d == 1)
            sh = sh & (S.c < vw) & (d == -1)
    return _emit(S, lg, sh, S.c - p.stop_atr * a, S.c + p.stop_atr * a, _win(S), rr=p.rr)


@strategy("r3_tf15m_macd_cross", "intraday", "tf15m_macd",
          "15m MACD(12,26,9) crosses above its signal while above zero: long, stop 1.5 ATR15, 2R (mirror below zero)",
          {"macd": (12, 26, 9), "stop_atr": 1.5, "rr": 2.0})
def macd_cross(s, p):
    return _macd_cross(s, p, False)


@strategy("r3_tf15m_macd_cross_vwap_dt", "intraday", "tf15m_macd",
          "15m MACD signal cross on the zero-line side, with close on that side of VWAP and the daily trend agreeing; stop 1.5 ATR15, 2R",
          {"macd": (12, 26, 9), "stop_atr": 1.5, "rr": 2.0})
def macd_cross_vwap_dt(s, p):
    return _macd_cross(s, p, True)


# ── ADX(14) > 25 ────────────────────────────────────────────────────────────
def _adx_cross(s, p):
    S = _ser(s)
    if S is None:
        return []
    adx, pdi, mdi = S.get("adx")
    a = S.get("atr14")
    ap = _prev(adx)
    with np.errstate(invalid="ignore"):
        x = (ap < p.level) & (adx >= p.level) & (a > 0)
        lg = x & (pdi > mdi)
        sh = x & (mdi > pdi)
    return _emit(S, lg, sh, S.c - p.stop_atr * a, S.c + p.stop_atr * a, _win(S), rr=p.rr)


@strategy("r3_tf15m_adx_cross", "intraday", "tf15m_adx",
          "15m ADX(14) crosses above 25 with +DI > -DI: long, stop 1.5 ATR15, 2R (mirror -DI > +DI)",
          {"adx": 14, "level": 25, "stop_atr": 1.5, "rr": 2.0})
def adx_cross(s, p):
    return _adx_cross(s, p)


@strategy("r3_tf15m_adx_cross_nifty", "futures", "tf15m_adx",
          "NIFTY future: 15m ADX(14) crosses above 25, trade the dominant DI, stop 1.5 ATR15, 2R",
          {"adx": 14, "level": 25, "stop_atr": 1.5, "rr": 2.0}, horizon="intraday", underlying="NIFTY")
def adx_cross_nifty(s, p):
    return _adx_cross(s, p)


# ── first-hour (four 15m candles) range break ───────────────────────────────
def _fh_break(s, p):
    S = _ser(s)
    if S is None:
        return []
    T = S.today
    fh = T.end <= _m("10:15")
    if fh.sum() < p.candles or T.start[0] != SESSION_OPEN:
        return []
    hi, lo = float(T.h[fh].max()), float(T.l[fh].min())
    if not hi > lo:
        return []
    mid = (hi + lo) / 2.0
    win = _win(S, "10:30", p.last)
    n = len(S.c)
    lg = np.zeros(n, bool); sh = np.zeros(n, bool)
    lg[S.P:] = T.c > hi
    sh[S.P:] = T.c < lo
    stop = np.full(n, mid)
    return _emit(S, lg, sh, stop, stop, win)


@strategy("r3_tf15m_fh_break", "intraday", "tf15m_first_hour",
          "10:30-13:00 first 15m close beyond the 09:15-10:15 (four-candle) range: follow, stop the range midpoint, out 15:15",
          {"candles": 4, "last": "13:00"})
def fh_break(s, p):
    return _fh_break(s, p)


@strategy("r3_tf15m_fh_break_nifty", "futures", "tf15m_first_hour",
          "NIFTY future: 10:30-13:00 first 15m close beyond the first-hour range, stop the midpoint, out 15:15",
          {"candles": 4, "last": "13:00"}, horizon="intraday", underlying="NIFTY")
def fh_break_nifty(s, p):
    return _fh_break(s, p)


@strategy("r3_tf15m_fh_break_bnf", "futures", "tf15m_first_hour",
          "BANKNIFTY future: 10:30-13:00 first 15m close beyond the first-hour range, stop the midpoint, out 15:15",
          {"candles": 4, "last": "13:00"}, horizon="intraday", underlying="BANKNIFTY")
def fh_break_bnf(s, p):
    return _fh_break(s, p)


# ── Donchian(20) incl. earlier sessions ─────────────────────────────────────
def _donch(s, p, rvol_min):
    S = _ser(s)
    if S is None:
        return []
    a = S.get("atr14")
    hh = _prev(pd.Series(S.h).rolling(p.n, min_periods=p.n).max().to_numpy())
    ll = _prev(pd.Series(S.l).rolling(p.n, min_periods=p.n).min().to_numpy())
    lx = _prev(pd.Series(S.l).rolling(p.exit_n, min_periods=p.exit_n).min().to_numpy())
    hx = _prev(pd.Series(S.h).rolling(p.exit_n, min_periods=p.exit_n).max().to_numpy())
    with np.errstate(invalid="ignore"):
        lg = (S.c > hh) & (a > 0)
        sh = (S.c < ll) & (a > 0)
        if rvol_min is not None:
            rv = np.full(len(S.c), np.nan)
            rv[S.P:] = S.today.v / S.slot_mean
            lg = lg & (rv >= rvol_min)
            sh = sh & (rv >= rvol_min)
    sl = np.maximum(lx, S.c - p.max_stop * a)
    ss = np.minimum(hx, S.c + p.max_stop * a)
    return _emit(S, lg, sh, sl, ss, _win(S, "10:00", "14:30"))


@strategy("r3_tf15m_donch20", "intraday", "tf15m_donchian",
          "10:00-14:30 15m close beyond the 20-candle Donchian channel (incl. earlier sessions): follow, stop the 10-candle extreme (<= 3 ATR), out 15:15",
          {"n": 20, "exit_n": 10, "max_stop": 3.0})
def donch20(s, p):
    return _donch(s, p, None)


@strategy("r3_tf15m_donch20_vol", "intraday", "tf15m_donchian",
          "15m Donchian(20) break (as r3_tf15m_donch20) on >= 1.5x the same-slot volume of the previous 10 sessions",
          {"n": 20, "exit_n": 10, "max_stop": 3.0, "rvol": 1.5})
def donch20_vol(s, p):
    return _donch(s, p, p.rvol)


# ── inside-candle breakout ──────────────────────────────────────────────────
@strategy("r3_tf15m_inside_break", "intraday", "tf15m_inside_candle",
          "15m inside candle (ending 09:45-14:00): first of the next 2 candles closing beyond the mother candle, stop its other end, 1.5R",
          {"wait": 2, "rr": 1.5, "last": "14:00"})
def inside_break(s, p):
    S = _ser(s)
    if S is None:
        return []
    T = S.today
    m = S.m
    out = []
    for j in range(1, m):                        # inside candle j, mother j-1 (both today)
        e = int(T.end[j])
        if e < _m("09:45") or e > _m(p.last):
            continue
        if not (T.h[j] < T.h[j - 1] and T.l[j] > T.l[j - 1]):
            continue
        mh, ml = float(T.h[j - 1]), float(T.l[j - 1])
        for q in range(j + 1, min(m, j + 1 + p.wait)):
            if T.c[q] > mh:
                out.append(Entry(int(T.sig[q]), LONG, stop=ml, rr=p.rr))
                break
            if T.c[q] < ml:
                out.append(Entry(int(T.sig[q]), SHORT, stop=mh, rr=p.rr))
                break
    return out


# ── CCI(20) ±100 ────────────────────────────────────────────────────────────
@strategy("r3_tf15m_cci_trend", "intraday", "tf15m_cci",
          "15m CCI(20) crosses above +100: long; below -100: short (Lambert); stop 1.5 ATR15, 2R",
          {"cci": 20, "level": 100, "stop_atr": 1.5, "rr": 2.0})
def cci_trend(s, p):
    S = _ser(s)
    if S is None:
        return []
    c, a = S.get("cci"), S.get("atr14")
    cp = _prev(c)
    with np.errstate(invalid="ignore"):
        lg = (cp <= p.level) & (c > p.level) & (a > 0)
        sh = (cp >= -p.level) & (c < -p.level) & (a > 0)
    return _emit(S, lg, sh, S.c - p.stop_atr * a, S.c + p.stop_atr * a, _win(S), rr=p.rr)


@strategy("r3_tf15m_cci_revert", "intraday", "tf15m_cci",
          "15m CCI(20) crosses back above -100: long to the 15m SMA20, stop 1.5 ATR15 (mirror back below +100)",
          {"cci": 20, "level": 100, "stop_atr": 1.5})
def cci_revert(s, p):
    S = _ser(s)
    if S is None:
        return []
    c, a, m20 = S.get("cci"), S.get("atr14"), S.get("sma20")
    cp = _prev(c)
    with np.errstate(invalid="ignore"):
        lg = (cp < -p.level) & (c >= -p.level) & (a > 0) & (m20 > S.c)
        sh = (cp > p.level) & (c <= p.level) & (a > 0) & (m20 < S.c)
    return _emit(S, lg, sh, S.c - p.stop_atr * a, S.c + p.stop_atr * a, _win(S), tgt_l=m20, tgt_s=m20)


# ── same-slot volume shock at a new day extreme ─────────────────────────────
@strategy("r3_tf15m_slotvol_shock", "intraday", "tf15m_volume_shock",
          "15m candle (09:45-14:00) on >= 3x its slot's 10-session volume, body >= 60 % of range, closing at a new day high: long, stop its low, out 15:15 (mirror)",
          {"rvol": 3.0, "body": 0.6, "last": "14:00"})
def slotvol_shock(s, p):
    S = _ser(s)
    if S is None:
        return []
    T = S.today
    m = S.m
    if m < 2:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = T.v / S.slot_mean
        rng = T.h - T.l
        body = np.abs(T.c - T.o)
        prior_hi = np.r_[np.nan, np.maximum.accumulate(T.h)[:-1]]
        prior_lo = np.r_[np.nan, np.minimum.accumulate(T.l)[:-1]]
        base = (rv >= p.rvol) & (rng > 0) & (body >= p.body * rng)
        lg = base & (T.c > T.o) & (T.c > prior_hi)
        sh = base & (T.c < T.o) & (T.c < prior_lo)
    n = len(S.c)
    L = np.zeros(n, bool); Sh = np.zeros(n, bool)
    L[S.P:] = lg
    Sh[S.P:] = sh
    sl = np.full(n, np.nan); ss = np.full(n, np.nan)
    sl[S.P:] = T.l
    ss[S.P:] = T.h
    return _emit(S, L, Sh, sl, ss, _win(S, "09:45", p.last))
