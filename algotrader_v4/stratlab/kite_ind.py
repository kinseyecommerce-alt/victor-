"""
stratlab/kite_ind.py — indicators exactly as the round-5 "Kite ecosystem" sources code them (research only).

Every function takes plain arrays (one value per candle, oldest first) and is CAUSAL: the value at index i uses
inputs 0..i only. Where a platform's formula differs from the textbook, the platform's version is the one here
(named after it), because the round tests strategies exactly as published:

  pine_*    TradingView Pine v5 built-ins as used by the AlgoTest templates (ta.rsi / ta.atr / ta.dmi use RMA =
            Wilder smoothing; ta.stdev is the population std; ta.stoch is the raw %K)
  talib_*   TA-Lib functions as called by AlgoBulls pyalgostrategypool (AROON, STOCH with SMA smoothing, ATR, RSI)
  ichimoku_algotest   the AlgoTest template's Ichimoku: lines are SMAs of (high+low)/2 (not mid-points of the
            highest high / lowest low) and both spans are the value `displacement` candles EARLIER
  supertrend          the standard ratcheting Supertrend (Wilder ATR on hl2), used by Streak and Tradetron
  supertrend_simplified_algotest   the AlgoTest template's non-ratcheting version (bands = hlc3 +- mult*ATR)

EMA seeding: every EMA here starts from the first value (pandas ewm adjust=False). TA-Lib seeds with an SMA of the
first n values and Pine with an SMA too; the difference decays as (1-2/(n+1))^k and is invisible after the
warm-up that every strategy in this round waits for. Tests: stratlab/tests_kite.py (hand-computed values).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _a(x) -> np.ndarray:
    return np.asarray(x, float)


def shift(x, k: int = 1) -> np.ndarray:
    """x delayed by k: shift(x)[i] == x[i-k] (NaN for the first k)."""
    x = _a(x)
    out = np.full(len(x), np.nan)
    if 0 < k < len(x):
        out[k:] = x[:len(x) - k]
    elif k == 0:
        out[:] = x
    return out


def ema(x, n: int) -> np.ndarray:
    x = _a(x)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if not ok.any():
        return out
    f = int(np.argmax(ok))                      # leading NaNs (e.g. a MACD signal of a warm-up) are skipped
    out[f:] = pd.Series(x[f:]).ewm(span=n, adjust=False).mean().to_numpy()
    return out


def ema_sma_seed(x, n: int) -> np.ndarray:
    """EMA seeded with the SMA of the first n values (TA-Lib EMA): NaN for the first n-1 values, the SMA at
    index n-1, then alpha = 2/(n+1). Used for the daily (EOD) Varsity rules, where history is short."""
    x = _a(x)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    y = x[n - 1:].copy()
    y[0] = float(np.mean(x[:n]))
    out[n - 1:] = pd.Series(y).ewm(span=n, adjust=False).mean().to_numpy()
    return out


def rma(x, n: int) -> np.ndarray:
    """Wilder's moving average (alpha = 1/n), started at the first finite value."""
    x = _a(x)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if not ok.any():
        return out
    f = int(np.argmax(ok))
    out[f:] = pd.Series(x[f:]).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
    return out


def sma(x, n: int) -> np.ndarray:
    return pd.Series(_a(x)).rolling(n).mean().to_numpy()


def stdev_pop(x, n: int) -> np.ndarray:
    return pd.Series(_a(x)).rolling(n).std(ddof=0).to_numpy()


def highest(x, n: int) -> np.ndarray:
    return pd.Series(_a(x)).rolling(n).max().to_numpy()


def lowest(x, n: int) -> np.ndarray:
    return pd.Series(_a(x)).rolling(n).min().to_numpy()


def true_range(h, l, c) -> np.ndarray:
    h, l, c = _a(h), _a(l), _a(c)
    pc = shift(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    if len(tr):
        tr[0] = h[0] - l[0]
    return tr


def atr(h, l, c, n: int) -> np.ndarray:
    """Wilder ATR (Pine ta.atr, TA-Lib ATR up to the seed); NaN for the first n-1 values."""
    a = rma(true_range(h, l, c), n)
    a[:n - 1] = np.nan
    return a


def rsi(x, n: int) -> np.ndarray:
    """Wilder RSI (Pine ta.rsi, TA-Lib RSI up to the seed). NaN until n changes exist; 100 if no losses."""
    x = _a(x)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if ok.sum() <= n:
        return out
    f = int(np.argmax(ok))
    y = x[f:]
    d = np.diff(y, prepend=y[0])
    up = rma(np.clip(d, 0, None), n)
    dn = rma(np.clip(-d, 0, None), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + up / dn)
    r[(dn == 0) & (up > 0)] = 100.0
    r[(dn == 0) & (up == 0)] = 50.0
    r[:n] = np.nan
    out[f:] = r
    return out


def macd(x, fast=12, slow=26, sig=9):
    """(macd line, signal line). Signal = EMA of the line; NaN until `slow` values exist."""
    line = ema(x, fast) - ema(x, slow)
    line[:slow - 1] = np.nan
    return line, ema(line, sig)


def bollinger(x, n=20, k=2.0):
    """(middle, upper, lower) with the population std (Pine ta.stdev, TA-Lib BBANDS)."""
    m = sma(x, n)
    s = stdev_pop(x, n)
    return m, m + k * s, m - k * s


def cross_up(a, b) -> np.ndarray:
    a = _a(a)
    b = np.broadcast_to(_a(b), a.shape)
    out = np.zeros(len(a), bool)
    with np.errstate(invalid="ignore"):
        out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def cross_dn(a, b) -> np.ndarray:
    a = _a(a)
    b = np.broadcast_to(_a(b), a.shape)
    out = np.zeros(len(a), bool)
    with np.errstate(invalid="ignore"):
        out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


def algobulls_cross(a, b, decimals: int = 2) -> np.ndarray:
    """AlgoBulls `utils.crossover` (platform code not published; CROSSOVER_ACCURACY_DECIMALS = 2 in every config):
    +1 where round(a-b, 2) goes from <= 0 to > 0 between the last two candles, -1 for the mirror, else 0."""
    d = np.round(_a(a) - np.broadcast_to(_a(b), np.shape(a)), decimals)
    out = np.zeros(len(d), np.int8)
    with np.errstate(invalid="ignore"):
        out[1:][(d[1:] > 0) & (d[:-1] <= 0)] = 1
        out[1:][(d[1:] < 0) & (d[:-1] >= 0)] = -1
    return out


# ── Pine (AlgoTest templates) ───────────────────────────────────────────────
def pine_dmi(h, l, c, di_len: int, adx_smooth: int):
    """Pine ta.dmi(diLength, adxSmoothing) -> (+DI, -DI, ADX)."""
    h, l = _a(h), _a(l)
    up = h - shift(h)
    dn = shift(l) - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    if len(pdm):
        pdm[0] = mdm[0] = np.nan
    tr = true_range(h, l, c)
    if len(tr):
        tr[0] = np.nan
    trur = rma(tr, di_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = 100.0 * rma(pdm, di_len) / trur
        minus = 100.0 * rma(mdm, di_len) / trur
        s = plus + minus
        adx = 100.0 * rma(np.abs(plus - minus) / np.where(s == 0, 1.0, s), adx_smooth)
    return plus, minus, adx


def pine_sar(h, l, c, start=0.02, inc=0.02, maximum=0.2) -> np.ndarray:
    """Pine ta.sar (Wilder's Parabolic SAR as in the Pine reference implementation)."""
    h, l, c = _a(h), _a(l), _a(c)
    n = len(c)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    up = c[1] > c[0]
    ep = h[1] if up else l[1]
    sar = l[0] if up else h[0]
    af = start
    out[1] = sar
    for i in range(2, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2])
            if l[i] < sar:                      # reversal to down
                up, sar, ep, af = False, ep, l[i], start
                sar = max(sar, h[i])
            elif h[i] > ep:
                ep, af = h[i], min(af + inc, maximum)
        else:
            sar = max(sar, h[i - 1], h[i - 2])
            if h[i] > sar:                      # reversal to up
                up, sar, ep, af = True, ep, h[i], start
                sar = min(sar, l[i])
            elif l[i] < ep:
                ep, af = l[i], min(af + inc, maximum)
        out[i] = sar
    return out


def pine_stoch(c, h, l, n: int) -> np.ndarray:
    """Pine ta.stoch(close, high, low, n): raw %K = 100 (c - lowest low n) / (highest high n - lowest low n)."""
    hh, ll = highest(h, n), lowest(l, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100.0 * (_a(c) - ll) / (hh - ll)
    return k


def williams_r(h, l, c, n: int = 14) -> np.ndarray:
    """AlgoTest template: (highest high n - close) / (highest high n - lowest low n) * -100."""
    hh, ll = highest(h, n), lowest(l, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (hh - _a(c)) / (hh - ll) * -100.0


def ichimoku_algotest(h, l, conv=9, base=26, span_b=52, disp=26):
    """AlgoTest template: tenkan = sma(hl2, 9), kijun = sma(hl2, 26), spanA = ((tenkan+kijun)/2)[26],
    spanB = sma(hl2, 52)[26] ([26] = the value 26 candles earlier). Returns (spanA, spanB)."""
    hl2 = (_a(h) + _a(l)) / 2.0
    tenkan, kijun = sma(hl2, conv), sma(hl2, base)
    return shift((tenkan + kijun) / 2.0, disp), shift(sma(hl2, span_b), disp)


def supertrend_simplified_algotest(h, l, c, n=10, mult=3.0):
    """AlgoTest template (no ratchet): upper = hlc3 + mult*atr, lower = hlc3 - mult*atr; is_up = close > lower,
    is_down = close < upper; trend_changed = (is_up[1] and is_down) or (is_down[1] and is_up).
    Returns (long_condition, short_condition) = (is_up and changed, is_down and changed)."""
    h, l, c = _a(h), _a(l), _a(c)
    a = atr(h, l, c, n)
    hlc3 = (h + l + c) / 3.0
    with np.errstate(invalid="ignore"):
        is_up = c > hlc3 - mult * a
        is_dn = c < hlc3 + mult * a
    pu, pd_ = np.zeros(len(c), bool), np.zeros(len(c), bool)
    pu[1:], pd_[1:] = is_up[:-1], is_dn[:-1]
    changed = (pu & is_dn) | (pd_ & is_up)
    return is_up & changed, is_dn & changed


# ── TA-Lib (AlgoBulls) ──────────────────────────────────────────────────────
def talib_aroon(h, l, n: int):
    """TA-Lib AROON(high, low, n) -> (aroon_down, aroon_up) over the last n+1 candles; ties -> the most recent
    extreme (TA-Lib scans with >= / <=). NaN for the first n values."""
    h, l = _a(h), _a(l)
    N = len(h)
    up, dn = np.full(N, np.nan), np.full(N, np.nan)
    for i in range(n, N):
        wh, wl = h[i - n:i + 1], l[i - n:i + 1]
        ih = n - int(np.argmax(wh[::-1]))       # index (0..n) of the most recent highest high
        il = n - int(np.argmin(wl[::-1]))
        up[i] = 100.0 * ih / n
        dn[i] = 100.0 * il / n
    return dn, up


def talib_stoch(h, l, c, fastk=5, slowk=3, slowd=3):
    """TA-Lib STOCH with SMA smoothing (matype 0): fastK raw over `fastk`, slowK = SMA(fastK, slowk),
    slowD = SMA(slowK, slowd). Returns (slowK, slowD)."""
    k = pine_stoch(c, h, l, fastk)
    sk = sma(k, slowk)
    return sk, sma(sk, slowd)


def session_vwap(tp, v, first) -> np.ndarray:
    """VWAP restarting at every True in `first` (pyalgotrading.indicator.vwap: typical price (h+l+c)/3 of the
    candles of the same calendar day, cumulative)."""
    tp, v = _a(tp), _a(v)
    first = np.asarray(first, bool)
    sid = np.cumsum(first)
    cv = pd.Series(v).groupby(sid).cumsum().to_numpy()
    cpv = pd.Series(tp * v).groupby(sid).cumsum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cv > 0, cpv / cv, np.nan)


# ── Supertrend (Streak, Tradetron) ──────────────────────────────────────────
def supertrend(h, l, c, n=10, mult=3.0):
    """Standard Supertrend on hl2 with Wilder ATR and ratcheting final bands. Returns (direction, line):
    direction +1 (green, close above the line) / -1 (red) / 0 during the ATR warm-up (first n candles)."""
    h, l, c = _a(h), _a(l), _a(c)
    a = rma(true_range(h, l, c), n)
    hl2 = (h + l) / 2.0
    bu, bl = hl2 + mult * a, hl2 - mult * a
    N = len(c)
    fu, fl = bu.copy(), bl.copy()
    d = np.ones(N, np.int64)
    line = np.full(N, np.nan)
    for j in range(1, N):
        fu[j] = bu[j] if (bu[j] < fu[j - 1] or c[j - 1] > fu[j - 1]) else fu[j - 1]
        fl[j] = bl[j] if (bl[j] > fl[j - 1] or c[j - 1] < fl[j - 1]) else fl[j - 1]
        if d[j - 1] == 1:
            d[j] = -1 if c[j] < fl[j] else 1
        else:
            d[j] = 1 if c[j] > fu[j] else -1
        line[j] = fl[j] if d[j] == 1 else fu[j]
    d[:n] = 0
    line[:n] = np.nan
    return d, line


def double_supertrend(h, l, c, a=(10, 3.0), b=(20, 2.0)) -> np.ndarray:
    """Tradetron double Supertrend state: +1 both green, -1 both red, 0 disagree / warm-up."""
    d1, _ = supertrend(h, l, c, *a)
    d2, _ = supertrend(h, l, c, *b)
    return np.where((d1 == 1) & (d2 == 1), 1, np.where((d1 == -1) & (d2 == -1), -1, 0))
