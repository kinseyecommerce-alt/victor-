"""
stratlab/candles.py — session-aligned candles and textbook candlestick-pattern detectors.

Everything here was written down BEFORE any backtest of the candlestick catalogue
(`catalog/candlestick.py`). The ratios below are fixed constants: do not tune them on results.

1. Candles
   `resample(t, o, h, l, c, v, tf)` turns one session of 1-minute bars into tf-minute candles aligned at the NSE
   open: 5m = 09:15-09:20, 09:20-09:25, ...; 15m = 09:15-09:30, ...; 60m = 09:15-10:15, 10:15-11:15, ...,
   14:15-15:15 and a short last candle 15:15-15:30. A candle [start, end) is COMPLETE once the bar of minute
   end-1 (e.g. 09:44 for the 09:30-09:45 candle) is known, or once any later bar is known when that minute is
   missing. Its signal bar `sig` is the 1-minute bar at whose close the candle is known to be complete: the
   end-1 bar, or the first later bar if the end-1 bar is missing. The simulator fills at the open of bar sig+1,
   so for a 15m candle ending 09:45 the earliest possible fill is the open of the 09:45 bar.
   Daily candles are the lab's daily bars (`SymbolData.daily()` / the DailyView a positional strategy gets).

2. Measurements of one candle (all prices of that candle)
   body B = |close - open|, range R = high - low, upper shadow U = high - max(open, close),
   lower shadow L = min(open, close) - low, white/bullish = close > open, black/bearish = close < open,
   avgB / avgR = mean body / range of the AVG_N candles BEFORE it (NaN until AVG_N candles exist).
   "Long" body:  B >  LONG_BODY_K  x avgB        (Nison: a real body clearly longer than recent ones;
   "Short" body: B <  SHORT_BODY_K x avgB         numerically the 10-candle averages codified by TA-Lib)
   "Meaningful" single candle: R > 0 and R >= MIN_RANGE_K x avgR (a dead, near-flat candle is no signal).

3. Prior trend (defined once, the same for every reversal pattern; measured on the candle p just BEFORE the
   pattern's first candle):  down-trend  close[p] < EMA20(close)[p]  and  close[p] < close[p-5]
                             up-trend    close[p] > EMA20(close)[p]  and  close[p] > close[p-5]
   Bullish reversals need a down-trend, bearish reversals an up-trend. EMA20 needs EMA_N candles of history.

4. Detectors are pure functions of OHLC arrays. They return a boolean array that is True at the LAST candle of
   the pattern and describe SHAPE only; `signals()` adds the trend context of `PATTERNS`. Every definition is in
   the detector's docstring (Nison, "Japanese Candlestick Charting Techniques"; Bulkowski, "Encyclopedia of
   Candlestick Charts").
"""
from __future__ import annotations

import weakref
from collections import namedtuple

import numpy as np

from .data import SESSION_OPEN
from .indicators import atr as _atr, ema as _ema

# ── constants (fixed before any backtest; do not tune) ──────────────────────
AVG_N = 10                     # candles in the average body / range
LONG_BODY_K = 1.0              # long body:  B > 1.0 x avgB
SHORT_BODY_K = 1.0             # short body: B < 1.0 x avgB
MIN_RANGE_K = 0.5              # single-candle shapes: R >= 0.5 x avgR
DOJI_BODY_FRAC = 0.10          # doji: B <= 10 % of R
HAMMER_SHADOW_MULT = 2.0       # hammer family: the long shadow >= 2 x body (Nison)
HAMMER_OTHER_FRAC = 0.10       # hammer family: the other shadow <= 10 % of R ("no or very little")
MARUBOZU_SHADOW_FRAC = 0.05    # marubozu: each shadow <= 5 % of R
DOJI_SIDE_FRAC = 0.10          # dragonfly / gravestone: the short-side shadow <= 10 % of R
LONGLEG_SHADOW_FRAC = 0.30     # long-legged doji: both shadows >= 30 % of R
SPIN_BODY_FRAC = 0.30          # spinning top: 10 % < B <= 30 % of R, both shadows longer than the body
TWEEZER_TOL = 0.05             # tweezers: |low1 - low2| (or highs) <= 0.05 x avgR
SOLDIER_SHADOW_FRAC = 0.25     # three soldiers / crows: close in the top (bottom) 25 % of each candle's range
EMA_N = 20                     # prior trend: EMA20 ...
TREND_N = 5                    # ... and a lower (higher) close than 5 candles earlier
ATR_N = 14                     # ATR for the minimum stop distance (same timeframe as the pattern)

TIMEFRAMES = (5, 15, 60)       # intraday candle sizes in minutes; daily candles come from the lab's daily bars
SESSION_END = 15 * 60 + 30     # 15:30 (end of the last candle)
HIST_CANDLES = 100             # earlier-session candles prepended for EMA / ATR / averages

Candles = namedtuple("Candles", "o h l c v start end first last sig")


# ── resampling ──────────────────────────────────────────────────────────────
def candle_end(start, tf: int):
    """End minute (exclusive) of the tf-minute candle that starts at `start`; the 60m day ends with 15:15-15:30."""
    return np.minimum(np.asarray(start) + tf, SESSION_END)


def resample(t, o, h, l, c, v, tf: int, complete_only: bool = True) -> Candles:
    """One session of 1-minute bars -> tf-minute candles aligned at 09:15 (see the module docstring).

    Returns arrays per candle: o h l c v, start / end (minute of day, end exclusive), first / last (index of the
    first / last 1-minute bar inside the candle) and sig (index of the bar at whose close the candle is known to
    be complete; -1 if it is not complete yet). With complete_only (default) an unfinished last candle is
    dropped, so a strategy never sees a candle that is still forming."""
    t = np.asarray(t, np.int64)
    n = len(t)
    if n == 0:
        z = np.zeros(0)
        zi = np.zeros(0, np.int64)
        return Candles(z, z, z, z, z, zi, zi, zi, zi, zi)
    b = (t - SESSION_OPEN) // tf
    cut = np.flatnonzero(b[1:] != b[:-1]) + 1
    first = np.concatenate(([0], cut)).astype(np.int64)
    last = np.concatenate((cut, [n])).astype(np.int64) - 1
    start = SESSION_OPEN + b[first] * tf
    end = candle_end(start, tf)
    sig = np.where(t[last] >= end - 1, last, last + 1)
    sig = np.where(sig < n, sig, -1)
    oo, cc = np.asarray(o, float)[first], np.asarray(c, float)[last]
    hh = np.maximum.reduceat(np.asarray(h, float), first)
    ll = np.minimum.reduceat(np.asarray(l, float), first)
    vv = np.add.reduceat(np.asarray(v, float), first)
    out = Candles(oo, hh, ll, cc, vv, start, end, first, last, sig)
    if complete_only and sig[-1] < 0:
        out = Candles(*(x[:-1] for x in out))
    return out


# ── measurements ────────────────────────────────────────────────────────────
def _prior_mean(x: np.ndarray, n: int = AVG_N) -> np.ndarray:
    """Mean of the n values BEFORE each index (NaN until n exist)."""
    out = np.full(len(x), np.nan)
    if len(x) > n:
        cs = np.concatenate(([0.0], np.cumsum(x)))
        out[n:] = (cs[n:-1] - cs[:-n - 1]) / n
    return out


def _sh(x: np.ndarray, k: int = 1):
    """x delayed by k candles: NaN (float) or False (bool) for the first k."""
    x = np.asarray(x)
    if x.dtype == bool:
        out = np.zeros(len(x), bool)
    else:
        out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


class _K:
    """Body / range / shadow arrays of a candle series."""
    __slots__ = ("o", "h", "l", "c", "B", "R", "U", "L", "bull", "bear", "avgB", "avgR", "long", "short", "mid")

    def __init__(self, o, h, l, c):
        o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
        self.o, self.h, self.l, self.c = o, h, l, c
        self.B = np.abs(c - o)
        self.R = h - l
        self.U = h - np.maximum(o, c)
        self.L = np.minimum(o, c) - l
        self.bull, self.bear = c > o, c < o
        self.avgB, self.avgR = _prior_mean(self.B), _prior_mean(self.R)
        with np.errstate(invalid="ignore"):
            self.long = self.B > LONG_BODY_K * self.avgB
            self.short = self.B < SHORT_BODY_K * self.avgB
        self.mid = (o + c) / 2.0


def _meaningful(k: _K) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (k.R > 0) & (k.R >= MIN_RANGE_K * k.avgR)


# ── single-candle detectors ─────────────────────────────────────────────────
def _hammer_shape(o, h, l, c):
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return (_meaningful(k) & (k.B > DOJI_BODY_FRAC * k.R) & (k.L >= HAMMER_SHADOW_MULT * k.B)
                & (k.U <= HAMMER_OTHER_FRAC * k.R))


def _inverted_shape(o, h, l, c):
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return (_meaningful(k) & (k.B > DOJI_BODY_FRAC * k.R) & (k.U >= HAMMER_SHADOW_MULT * k.B)
                & (k.L <= HAMMER_OTHER_FRAC * k.R))


def hammer(o, h, l, c):
    """Hammer (Nison) — bullish reversal, needs a prior DOWN-trend.
    A small real body (either colour, but not a doji: B > 10 % of R) at the top of the range; lower shadow at
    least 2 x the body; no or very little upper shadow (U <= 10 % of R); meaningful range."""
    return _hammer_shape(o, h, l, c)


def hanging_man(o, h, l, c):
    """Hanging man (Nison) — bearish reversal, needs a prior UP-trend. Same shape as the hammer."""
    return _hammer_shape(o, h, l, c)


def inverted_hammer(o, h, l, c):
    """Inverted hammer (Nison) — bullish reversal, needs a prior DOWN-trend.
    A small real body (B > 10 % of R) at the bottom of the range; upper shadow at least 2 x the body; no or very
    little lower shadow (L <= 10 % of R); meaningful range."""
    return _inverted_shape(o, h, l, c)


def shooting_star(o, h, l, c):
    """Shooting star (Nison) — bearish reversal, needs a prior UP-trend. Same shape as the inverted hammer
    (Nison prefers a gap above the prior body; the shape alone is used, so intraday candles qualify)."""
    return _inverted_shape(o, h, l, c)


def bullish_marubozu(o, h, l, c):
    """Bullish (white) marubozu (Nison / Bulkowski) — strength, no trend context.
    White candle with a long body (B > avgB) and (almost) no shadows: U and L each <= 5 % of R."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return (k.bull & k.long & (k.U <= MARUBOZU_SHADOW_FRAC * k.R) & (k.L <= MARUBOZU_SHADOW_FRAC * k.R))


def bearish_marubozu(o, h, l, c):
    """Bearish (black) marubozu — weakness, no trend context. Black, long body, U and L each <= 5 % of R."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return (k.bear & k.long & (k.U <= MARUBOZU_SHADOW_FRAC * k.R) & (k.L <= MARUBOZU_SHADOW_FRAC * k.R))


def doji(o, h, l, c):
    """Doji (Nison) — open and close equal or nearly so: B <= 10 % of R, meaningful range. Not traded on its
    own; the dragonfly / gravestone / long-legged variants are."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return _meaningful(k) & (k.B <= DOJI_BODY_FRAC * k.R)


def dragonfly_doji(o, h, l, c):
    """Dragonfly doji (Nison) — bullish reversal, needs a prior DOWN-trend.
    A doji whose open and close are at (or very near) the high: U <= 10 % of R, so the lower shadow is long."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return doji(o, h, l, c) & (k.U <= DOJI_SIDE_FRAC * k.R)


def gravestone_doji(o, h, l, c):
    """Gravestone doji (Nison) — bearish reversal, needs a prior UP-trend.
    A doji whose open and close are at (or very near) the low: L <= 10 % of R, so the upper shadow is long."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return doji(o, h, l, c) & (k.L <= DOJI_SIDE_FRAC * k.R)


def long_legged_doji(o, h, l, c):
    """Long-legged doji (Nison) — indecision after a trend: bullish after a DOWN-trend, bearish after an
    UP-trend. A doji with long upper AND lower shadows: U >= 30 % of R and L >= 30 % of R."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return doji(o, h, l, c) & (k.U >= LONGLEG_SHADOW_FRAC * k.R) & (k.L >= LONGLEG_SHADOW_FRAC * k.R)


def spinning_top(o, h, l, c):
    """Spinning top (Nison) — CONTEXT ONLY, never traded. A small real body (10 % < B <= 30 % of R) with an upper
    and a lower shadow each longer than the body; meaningful range."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return (_meaningful(k) & (k.B > DOJI_BODY_FRAC * k.R) & (k.B <= SPIN_BODY_FRAC * k.R)
                & (k.U > k.B) & (k.L > k.B))


# ── two-candle detectors (True at the second candle) ────────────────────────
def bullish_engulfing(o, h, l, c):
    """Bullish engulfing (Nison) — bullish reversal, needs a prior DOWN-trend.
    Candle 1 black, candle 2 white, and candle 2's real body engulfs candle 1's real body:
    open2 <= close1 and close2 >= open1, with at least one of the two strict (shadows need not be engulfed)."""
    k = _K(o, h, l, c)
    o1, c1, b1 = _sh(k.o), _sh(k.c), _sh(k.bear)
    with np.errstate(invalid="ignore"):
        return b1 & k.bull & (k.o <= c1) & (k.c >= o1) & ((k.o < c1) | (k.c > o1))


def bearish_engulfing(o, h, l, c):
    """Bearish engulfing (Nison) — bearish reversal, needs a prior UP-trend.
    Candle 1 white, candle 2 black, open2 >= close1 and close2 <= open1, at least one strict."""
    k = _K(o, h, l, c)
    o1, c1, w1 = _sh(k.o), _sh(k.c), _sh(k.bull)
    with np.errstate(invalid="ignore"):
        return w1 & k.bear & (k.o >= c1) & (k.c <= o1) & ((k.o > c1) | (k.c < o1))


def piercing_line(o, h, l, c):
    """Piercing line (Nison) — bullish reversal, needs a prior DOWN-trend.
    Candle 1 long black; candle 2 white, OPENS BELOW candle 1's LOW, and closes above the midpoint of candle 1's
    body but below candle 1's open (closing above the open would be an engulfing)."""
    k = _K(o, h, l, c)
    o1, l1, m1, b1, lg1 = _sh(k.o), _sh(k.l), _sh(k.mid), _sh(k.bear), _sh(k.long)
    with np.errstate(invalid="ignore"):
        return b1 & lg1 & k.bull & (k.o < l1) & (k.c > m1) & (k.c < o1)


def dark_cloud_cover(o, h, l, c):
    """Dark cloud cover (Nison) — bearish reversal, needs a prior UP-trend.
    Candle 1 long white; candle 2 black, OPENS ABOVE candle 1's HIGH, and closes below the midpoint of candle 1's
    body but above candle 1's open."""
    k = _K(o, h, l, c)
    o1, h1, m1, w1, lg1 = _sh(k.o), _sh(k.h), _sh(k.mid), _sh(k.bull), _sh(k.long)
    with np.errstate(invalid="ignore"):
        return w1 & lg1 & k.bear & (k.o > h1) & (k.c < m1) & (k.c > o1)


def bullish_harami(o, h, l, c):
    """Bullish harami (Nison) — bullish reversal, needs a prior DOWN-trend.
    Candle 1 long black; candle 2 has a short body (either colour) lying inside candle 1's body:
    max(open2, close2) <= open1 and min(open2, close2) >= close1, and body2 < body1."""
    k = _K(o, h, l, c)
    o1, c1, b1, lg1, B1 = _sh(k.o), _sh(k.c), _sh(k.bear), _sh(k.long), _sh(k.B)
    with np.errstate(invalid="ignore"):
        return (b1 & lg1 & k.short & (np.maximum(k.o, k.c) <= o1) & (np.minimum(k.o, k.c) >= c1) & (k.B < B1))


def bearish_harami(o, h, l, c):
    """Bearish harami (Nison) — bearish reversal, needs a prior UP-trend.
    Candle 1 long white; candle 2 short body inside candle 1's body: max(o2, c2) <= close1, min(o2, c2) >= open1,
    body2 < body1."""
    k = _K(o, h, l, c)
    o1, c1, w1, lg1, B1 = _sh(k.o), _sh(k.c), _sh(k.bull), _sh(k.long), _sh(k.B)
    with np.errstate(invalid="ignore"):
        return (w1 & lg1 & k.short & (np.maximum(k.o, k.c) <= c1) & (np.minimum(k.o, k.c) >= o1) & (k.B < B1))


def tweezer_bottom(o, h, l, c):
    """Tweezer bottom (Nison; Bulkowski's two-candle form) — bullish reversal, needs a prior DOWN-trend.
    Candle 1 black, candle 2 white, matching lows: |low1 - low2| <= 0.05 x avgR."""
    k = _K(o, h, l, c)
    l1, b1 = _sh(k.l), _sh(k.bear)
    with np.errstate(invalid="ignore"):
        return b1 & k.bull & (np.abs(k.l - l1) <= TWEEZER_TOL * k.avgR)


def tweezer_top(o, h, l, c):
    """Tweezer top — bearish reversal, needs a prior UP-trend. Candle 1 white, candle 2 black,
    |high1 - high2| <= 0.05 x avgR."""
    k = _K(o, h, l, c)
    h1, w1 = _sh(k.h), _sh(k.bull)
    with np.errstate(invalid="ignore"):
        return w1 & k.bear & (np.abs(k.h - h1) <= TWEEZER_TOL * k.avgR)


def inside_bar(o, h, l, c):
    """Inside bar (Bulkowski "inside day") — continuation: traded in the direction of the prior trend
    (long after an UP-trend, short after a DOWN-trend). Candle 2's range lies strictly inside candle 1's:
    high2 < high1 and low2 > low1."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    with np.errstate(invalid="ignore"):
        return (h < _sh(h)) & (l > _sh(l))


def outside_bar(o, h, l, c):
    """Outside bar (Bulkowski "outside day") — no trend context; traded in the direction of candle 2's colour
    (long if white, short if black). Candle 2's range engulfs candle 1's: high2 > high1 and low2 < low1."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    with np.errstate(invalid="ignore"):
        return (h > _sh(h)) & (l < _sh(l))


# ── three-candle detectors (True at the third candle) ───────────────────────
def morning_star(o, h, l, c):
    """Morning star (Nison) — bullish reversal, needs a prior DOWN-trend.
    Candle 1 long black; candle 2 (the star, either colour) has a short body that gaps below candle 1's body:
    max(open2, close2) < close1; candle 3 white and closes above the midpoint of candle 1's body."""
    k = _K(o, h, l, c)
    b1, lg1, c1, m1 = _sh(k.bear, 2), _sh(k.long, 2), _sh(k.c, 2), _sh(k.mid, 2)
    s2, top2 = _sh(k.short), _sh(np.maximum(k.o, k.c))
    with np.errstate(invalid="ignore"):
        return b1 & lg1 & s2 & (top2 < c1) & k.bull & (k.c > m1)


def evening_star(o, h, l, c):
    """Evening star (Nison) — bearish reversal, needs a prior UP-trend.
    Candle 1 long white; candle 2 short body gapping above candle 1's body: min(open2, close2) > close1;
    candle 3 black and closes below the midpoint of candle 1's body."""
    k = _K(o, h, l, c)
    w1, lg1, c1, m1 = _sh(k.bull, 2), _sh(k.long, 2), _sh(k.c, 2), _sh(k.mid, 2)
    s2, bot2 = _sh(k.short), _sh(np.minimum(k.o, k.c))
    with np.errstate(invalid="ignore"):
        return w1 & lg1 & s2 & (bot2 > c1) & k.bear & (k.c < m1)


def three_white_soldiers(o, h, l, c):
    """Three white soldiers (Nison) — bullish reversal, needs a prior DOWN-trend.
    Three long white candles with successively higher closes; candles 2 and 3 open within the prior candle's
    body (open1 <= open2 <= close1, open2 <= open3 <= close2); each closes at or near its high
    (U <= 25 % of R)."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        one = k.bull & k.long & (k.U <= SOLDIER_SHADOW_FRAC * k.R)
        opens_in = (k.o >= _sh(k.o)) & (k.o <= _sh(k.c)) & (k.c > _sh(k.c))
        return one & _sh(one) & _sh(one, 2) & opens_in & _sh(opens_in)


def three_black_crows(o, h, l, c):
    """Three black crows (Nison) — bearish reversal, needs a prior UP-trend.
    Three long black candles with successively lower closes; candles 2 and 3 open within the prior body
    (close1 <= open2 <= open1, ...); each closes at or near its low (L <= 25 % of R)."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        one = k.bear & k.long & (k.L <= SOLDIER_SHADOW_FRAC * k.R)
        opens_in = (k.o <= _sh(k.o)) & (k.o >= _sh(k.c)) & (k.c < _sh(k.c))
        return one & _sh(one) & _sh(one, 2) & opens_in & _sh(opens_in)


def three_inside_up(o, h, l, c):
    """Three inside up (Nison / Bulkowski) — bullish reversal, needs a prior DOWN-trend.
    A bullish harami (candles 1-2) followed by a white candle 3 that closes above candle 1's open."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return _sh(bullish_harami(o, h, l, c)) & k.bull & (k.c > _sh(k.o, 2))


def three_inside_down(o, h, l, c):
    """Three inside down — bearish reversal, needs a prior UP-trend.
    A bearish harami (candles 1-2) followed by a black candle 3 that closes below candle 1's open."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return _sh(bearish_harami(o, h, l, c)) & k.bear & (k.c < _sh(k.o, 2))


def three_outside_up(o, h, l, c):
    """Three outside up (Nison / Bulkowski) — bullish reversal, needs a prior DOWN-trend.
    A bullish engulfing (candles 1-2) followed by a white candle 3 with a higher close than candle 2."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return _sh(bullish_engulfing(o, h, l, c)) & k.bull & (k.c > _sh(k.c))


def three_outside_down(o, h, l, c):
    """Three outside down — bearish reversal, needs a prior UP-trend.
    A bearish engulfing (candles 1-2) followed by a black candle 3 with a lower close than candle 2."""
    k = _K(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return _sh(bearish_engulfing(o, h, l, c)) & k.bear & (k.c < _sh(k.c))


# ── trend context and the pattern table ─────────────────────────────────────
def trend_context(c) -> tuple[np.ndarray, np.ndarray]:
    """(down, up) at each candle p: down = close < EMA20 and close < close 5 candles earlier; up = the mirror.
    False until EMA_N candles exist. Causal: uses candles 0..p only."""
    c = np.asarray(c, float)
    n = len(c)
    down, up = np.zeros(n, bool), np.zeros(n, bool)
    if n > EMA_N:
        e = _ema(c, EMA_N)
        c5 = _sh(c, TREND_N)
        ok = np.arange(n) >= EMA_N
        with np.errstate(invalid="ignore"):
            down = ok & (c < e) & (c < c5)
            up = ok & (c > e) & (c > c5)
    return down, up


Pattern = namedtuple("Pattern", "fn n side context")
# side: +1 bullish, -1 bearish, 0 either way (decided by `context`)
# context: "down" (needs a prior down-trend), "up" (prior up-trend), "none",
#          "reversal" (long after a down-trend, short after an up-trend),
#          "trend" (continuation: long after an up-trend, short after a down-trend),
#          "colour" (long if the last candle is white, short if black)
PATTERNS = {
    "hammer":               Pattern(hammer, 1, 1, "down"),
    "inverted_hammer":      Pattern(inverted_hammer, 1, 1, "down"),
    "hanging_man":          Pattern(hanging_man, 1, -1, "up"),
    "shooting_star":        Pattern(shooting_star, 1, -1, "up"),
    "bull_marubozu":        Pattern(bullish_marubozu, 1, 1, "none"),
    "bear_marubozu":        Pattern(bearish_marubozu, 1, -1, "none"),
    "dragonfly_doji":       Pattern(dragonfly_doji, 1, 1, "down"),
    "gravestone_doji":      Pattern(gravestone_doji, 1, -1, "up"),
    "longlegged_doji":      Pattern(long_legged_doji, 1, 0, "reversal"),
    "bull_engulfing":       Pattern(bullish_engulfing, 2, 1, "down"),
    "bear_engulfing":       Pattern(bearish_engulfing, 2, -1, "up"),
    "piercing_line":        Pattern(piercing_line, 2, 1, "down"),
    "dark_cloud":           Pattern(dark_cloud_cover, 2, -1, "up"),
    "bull_harami":          Pattern(bullish_harami, 2, 1, "down"),
    "bear_harami":          Pattern(bearish_harami, 2, -1, "up"),
    "tweezer_bottom":       Pattern(tweezer_bottom, 2, 1, "down"),
    "tweezer_top":          Pattern(tweezer_top, 2, -1, "up"),
    "inside_bar":           Pattern(inside_bar, 2, 0, "trend"),
    "outside_bar":          Pattern(outside_bar, 2, 0, "colour"),
    "morning_star":         Pattern(morning_star, 3, 1, "down"),
    "evening_star":         Pattern(evening_star, 3, -1, "up"),
    "three_white_soldiers": Pattern(three_white_soldiers, 3, 1, "down"),
    "three_black_crows":    Pattern(three_black_crows, 3, -1, "up"),
    "three_inside_up":      Pattern(three_inside_up, 3, 1, "down"),
    "three_inside_down":    Pattern(three_inside_down, 3, -1, "up"),
    "three_outside_up":     Pattern(three_outside_up, 3, 1, "down"),
    "three_outside_down":   Pattern(three_outside_down, 3, -1, "up"),
}
CONTEXT_ONLY = {"doji": doji, "spinning_top": spinning_top}     # detected, never traded


def signals(o, h, l, c, names=None) -> dict:
    """{pattern: (long_mask, short_mask)} with the trend context applied; True at the pattern's last candle."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    down, up = trend_context(c)
    white, black = c > o, c < o
    out = {}
    for name in (names or PATTERNS):
        pat = PATTERNS[name]
        m = np.asarray(pat.fn(o, h, l, c), bool)
        dn, upc = _sh(down, pat.n), _sh(up, pat.n)          # trend on the candle before the first pattern candle
        z = np.zeros(len(c), bool)
        if pat.context == "down":
            lm, sm = m & dn, z
        elif pat.context == "up":
            lm, sm = z, m & upc
        elif pat.context == "none":
            lm, sm = (m, z) if pat.side > 0 else (z, m)
        elif pat.context == "reversal":
            lm, sm = m & dn, m & upc
        elif pat.context == "trend":
            lm, sm = m & upc, m & dn
        elif pat.context == "colour":
            lm, sm = m & white, m & black
        else:
            raise ValueError(pat.context)
        out[name] = (lm, sm)
    return out


def pattern_extremes(l, h, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(lowest low, highest high) of the n candles ending at each candle (NaN until n exist)."""
    lo = np.asarray(l, float).copy()
    hi = np.asarray(h, float).copy()
    for k in range(1, n):
        lo = np.fmin(lo, _sh(l, k)) if k < len(lo) else lo
        hi = np.fmax(hi, _sh(h, k)) if k < len(hi) else hi
    if n > 1:
        lo[:n - 1] = np.nan
        hi[:n - 1] = np.nan
    return lo, hi


# ── per-session candles for strategies (cached, causal) ─────────────────────
SessionCandles = namedtuple("SessionCandles", "o h l c start end sig atr sigs lo hi")

_HIST: dict = {}                                  # id(SymbolData) -> (weakref, {tf: (o h l c sess first)})
_SESS = weakref.WeakKeyDictionary()               # Session -> {(tf, n): SessionCandles | None}


def _symbol_candles(sd, tf: int):
    """All complete candles of every session of a symbol: o h l c, session index, first-of-session flag."""
    ent = _HIST.get(id(sd))
    if ent is None or ent[0]() is not sd:
        if len(_HIST) > 64:
            _HIST.clear()
        ent = (weakref.ref(sd), {})
        _HIST[id(sd)] = ent
    got = ent[1].get(tf)
    if got is None:
        lens = sd.ends - sd.starts
        sess_of = np.repeat(np.arange(len(sd.starts)), lens)
        bid = sess_of.astype(np.int64) * 1000 + (sd.t.astype(np.int64) - SESSION_OPEN) // tf
        cut = np.flatnonzero(bid[1:] != bid[:-1]) + 1
        st = np.concatenate(([0], cut)).astype(np.int64)
        en = np.concatenate((cut, [len(bid)])).astype(np.int64) - 1
        sess = sess_of[st]
        first = np.ones(len(st), bool)
        first[1:] = sess[1:] != sess[:-1]
        got = (sd.o[st].astype(float), np.maximum.reduceat(sd.h, st).astype(float),
               np.minimum.reduceat(sd.l, st).astype(float), sd.c[en].astype(float), sess, first)
        ent[1][tf] = got
    return got


def session_candles(s, tf: int):
    """The candles a strategy may use at the close of the last visible bar of `s` (a Session or a View):
    up to HIST_CANDLES complete candles of EARLIER sessions followed by today's COMPLETE candles, with the
    pattern signals and ATR computed on that series. Returns SessionCandles for today's candles only
    (sig = index of the 1-minute bar at whose close each candle is complete), or None."""
    base = getattr(s, "_s", s)                    # a View's underlying Session
    n = len(s)
    try:
        memo = _SESS.setdefault(base, {})
    except TypeError:
        memo = {}
    key = (tf, n)
    if key in memo:
        return memo[key]
    today = resample(s.t, s.o, s.h, s.l, s.c, s.v, tf)
    m = len(today.o)
    res = None
    if m:
        sd, k = getattr(base, "_sd", None), getattr(base, "day_index", 0)
        if sd is not None and k > 0:
            ho, hh, hl, hc, hs, hf = _symbol_candles(sd, tf)
            j = int(np.searchsorted(hs, k))       # candles of sessions before today only
            a = max(0, j - HIST_CANDLES)
            po, ph, pl, pc, pf = ho[a:j], hh[a:j], hl[a:j], hc[a:j], hf[a:j]
        else:
            po = ph = pl = pc = np.zeros(0)
            pf = np.zeros(0, bool)
        P = len(po)
        o = np.concatenate((po, today.o)); h = np.concatenate((ph, today.h))
        l = np.concatenate((pl, today.l)); c = np.concatenate((pc, today.c))
        first = np.concatenate((pf, np.arange(m) == 0))
        a14 = np.array(_atr(h, l, c, ATR_N, first), float)
        a14[:ATR_N] = np.nan
        sigs = {name: (lm[P:], sm[P:]) for name, (lm, sm) in signals(o, h, l, c).items()}
        lo, hi = {}, {}
        for npat in (1, 2, 3):
            x, y = pattern_extremes(l, h, npat)
            lo[npat], hi[npat] = x[P:], y[P:]
        res = SessionCandles(today.o, today.h, today.l, today.c, today.start, today.end, today.sig,
                             a14[P:], sigs, lo, hi)
    if len(memo) > 512:
        memo.clear()
    memo[key] = res
    return res


_DAILY: dict = {}


def daily_candles(d, lookback: int = 150):
    """Signals / ATR / extremes on the last `lookback` daily bars of a DailyView (index -1 = decision day).
    Cached by content, shared by every daily candlestick strategy."""
    o, h, l, c = (np.asarray(x[-lookback:], float) for x in (d.o, d.h, d.l, d.c))
    key = (getattr(d, "symbol", ""), len(d), o.tobytes() + h.tobytes() + l.tobytes() + c.tobytes())
    got = _DAILY.get(key)
    if got is None:
        a14 = np.array(_atr(h, l, c, ATR_N), float)
        a14[:ATR_N] = np.nan
        lo, hi = {}, {}
        for npat in (1, 2, 3):
            lo[npat], hi[npat] = pattern_extremes(l, h, npat)
        got = SessionCandles(o, h, l, c, None, None, None, a14, signals(o, h, l, c), lo, hi)
        if len(_DAILY) > 4096:
            _DAILY.clear()
        _DAILY[key] = got
    return got
