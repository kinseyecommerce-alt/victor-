"""
stratlab/r6_ind.py — formulas for the round-6 GitHub-sourced strategies (stratlab/catalog/r6_github.py,
stratlab/r6_portfolio.py), re-implemented from the source code named in prereg/github_v1.json with the source bugs
listed there fixed. Research only; nothing here is imported by a live agent.

Every function is causal: the value at index i uses inputs at indices <= i only (tests_r6.py checks this on
random data by truncation). Point functions (`*_at`) evaluate one index by walking back over the inputs, so a
per-day positional strategy does not recompute whole arrays.

  td_counts / td_at          TD Sequential setup counts and "perfection" (freqtrade TDSequentialStrategy.py; fix:
                             bar 8 or 9 low below BOTH bar-6 and bar-7 lows, as its own docstring states — the code
                             used "below either")
  wvf_signal                 Williams VIX Fix signal (openalgo example, Pine CM_Williams_Vix_Fix defaults)
  dbo2_lookback / dbo2_at    Dynamic Breakout II (QuantConnect tutorial 04 / Pruitt), adaptive lookback 20..60
  robust_vol, ewmac, mr_wings  pysystemtrade rules/mr_wings.py with its EWMAC and a reconstructed robust_vol_calc
  moon_phase_jde, quarter_events  Meeus, Astronomical Algorithms (2nd ed.) ch. 49, phases of the Moon
  ghost_replay               letianzj ghost_trader.py state machine (fix: Donchian stop window excludes the bar)
  rbreaker_levels            letianzj r_breaker.py pivots (floor-trader levels as coded)
  clenow_momentum            annualised exponential-regression slope x R^2 (NiftyOnTheMove indicators.Momentum)
  residual_score             QuantConnect tutorial 136 score on a market-only regression
  alpha101_*                 WorldQuant 101 Formulaic Alphas #9, #12, #33, #54, #101 (Kakushadze 2016), written from
                             the paper (not from the yli188 repository)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import kite_ind as K


def _a(x) -> np.ndarray:
    return np.asarray(x, float)


# ══════════════════════════════════════════════════════════════════════════
# TD Sequential (freqtrade berlinguyinca/TDSequentialStrategy.py)
# ══════════════════════════════════════════════════════════════════════════
def _run_len(cond: np.ndarray) -> np.ndarray:
    """Consecutive-True count ending at each index (0 where False)."""
    m = np.asarray(cond, bool)
    idx = np.arange(len(m))
    return idx - np.maximum.accumulate(np.where(~m, idx, -1))


def td_counts(c, h, l, as_coded: bool = False):
    """Full arrays (seq_buy, seq_sell, exceed_low, exceed_high), freqtrade's loop re-written.
    seq_buy[i] = run of closes below the close 4 bars earlier ending at i (seq_sell: above).
    exceed_low[i]: count 8 -> low8 < low6 and low7; count >= 9 -> this bar's low < lows of bars 6 and 7 of the count,
    and at count 9 also bar 8's flag. as_coded=True reproduces the source's 'below EITHER low' (for tests only)."""
    c, h, l = _a(c), _a(h), _a(l)
    n = len(c)
    c4 = K.shift(c, 4)
    with np.errstate(invalid="ignore"):
        sb = _run_len(c < c4)
        ss = _run_len(c > c4)
    xl = np.zeros(n, bool)
    xh = np.zeros(n, bool)
    pick = (lambda a, b: np.minimum(a, b)) if not as_coded else None
    for i in range(n):
        for seq, x, arr, below in ((sb[i], xl, l, True), (ss[i], xh, h, False)):
            if seq < 8:
                continue
            if seq == 8:
                r6, r7 = arr[i - 2], arr[i - 1]
            else:
                k = seq - 9
                r6, r7 = arr[i - 3 - k], arr[i - 2 - k]
            if as_coded:
                v = (arr[i] < r6 or arr[i] < r7) if below else (arr[i] > r6 or arr[i] > r7)
            else:
                v = arr[i] < pick(r6, r7) if below else arr[i] > max(r6, r7)
            if seq == 9:
                v = v or x[i - 1]
            x[i] = bool(v)
    return sb, ss, xl, xh


def _seq_at(c, i: int, sign: int) -> int:
    n = 0
    while i - n >= 4 and sign * (c[i - n] - c[i - n - 4]) < 0:
        n += 1
    return n


def _exceed_at(c, arr, i: int, sign: int) -> bool:
    """sign=+1: buy side (lows, 'below'); -1: sell side (highs, 'above')."""
    seq = _seq_at(c, i, sign)
    if seq < 8:
        return False
    if seq == 8:
        r6, r7 = arr[i - 2], arr[i - 1]
    else:
        k = seq - 9
        r6, r7 = arr[i - 3 - k], arr[i - 2 - k]
    v = arr[i] < min(r6, r7) if sign > 0 else arr[i] > max(r6, r7)
    if seq == 9 and not v:
        v = _exceed_at(c, arr, i - 1, sign)
    return bool(v)


def td_at(c, h, l, i: int = -1) -> tuple:
    """(seq_buy, seq_sell, exceed_low, exceed_high) at index i (default: the last bar), = td_counts(...)[*][i]."""
    c, h, l = _a(c), _a(h), _a(l)
    i = i % len(c)
    return (_seq_at(c, i, +1), _seq_at(c, i, -1), _exceed_at(c, l, i, +1), _exceed_at(c, h, i, -1))


# ══════════════════════════════════════════════════════════════════════════
# Williams VIX Fix (openalgo examples/python/william vix fix.py; Pine CM_Williams_Vix_Fix)
# ══════════════════════════════════════════════════════════════════════════
def wvf_signal(c, l, pd_: int = 22, bbl: int = 20, mult: float = 2.0, lb: int = 50, ph: float = 0.85):
    """(wvf, signal): wvf = (highest(close, 22) - low) / highest(close, 22) * 100; signal = wvf >= SMA20 + 2 x
    population stdev20 (Pine ta.stdev) OR wvf >= highest(wvf, 50) x 0.85. NaN comparisons are False."""
    c, l = _a(c), _a(l)
    hc = K.highest(c, pd_)
    wvf = (hc - l) / hc * 100.0
    upper = K.sma(wvf, bbl) + mult * K.stdev_pop(wvf, bbl)
    rh = K.highest(wvf, lb) * ph
    with np.errstate(invalid="ignore"):
        sig = (wvf >= upper) | (wvf >= rh)
    return wvf, sig & np.isfinite(wvf)


# ══════════════════════════════════════════════════════════════════════════
# Dynamic Breakout II (QuantConnect Strategy Library 04 / Pruitt & Hill 2012; letianzj dynamic_breakout_ii.py)
# ══════════════════════════════════════════════════════════════════════════
DBO_VOL_N, DBO_START, DBO_MIN, DBO_MAX = 30, 20, 20, 60


def _dbo_step(c, t: int, n_prev: int) -> int:
    v_t = float(np.std(c[t - DBO_VOL_N + 1: t + 1]))
    v_y = float(np.std(c[t - DBO_VOL_N: t]))
    if not (v_t > 0 and np.isfinite(v_t) and np.isfinite(v_y)):
        return n_prev
    dv = (v_t - v_y) / v_t
    return int(min(max(round(n_prev * (1.0 + dv)), DBO_MIN), DBO_MAX))


def dbo2_lookback(c, start: int = 0, n0: np.ndarray | None = None) -> np.ndarray:
    """Adaptive lookback N_t (int array, 0 before the first defined value). N = 20 at the first bar with 31 closes
    (t = 30); then N_t = clip(round(N_{t-1} (1 + dvol)), 20, 60), dvol = (vol_t - vol_{t-1}) / vol_t, vol = population
    std of the last 30 closes (the tutorial's text; its code's window indexing and letianzj's (v_t/v_y)/v_t are bugs).
    `n0` = an already computed prefix (resumed from its end)."""
    c = _a(c)
    n = len(c)
    out = np.zeros(n, np.int64)
    k0 = 0
    if n0 is not None and len(n0):
        k0 = min(len(n0), n)
        out[:k0] = n0[:k0]
    for t in range(max(k0, DBO_VOL_N), n):
        out[t] = DBO_START if t == DBO_VOL_N else _dbo_step(c, t, int(out[t - 1]))
    return out


def dbo2_at(c, h, l, N: np.ndarray, t: int = -1) -> dict | None:
    """Levels at index t with lookback N[t]: hh / ll of the N sessions BEFORE t, SMA and population-std Bollinger
    band (k = 2) of the N closes ending at t. None before the lookback exists."""
    c, h, l = _a(c), _a(h), _a(l)
    t = t % len(c)
    n = int(N[t])
    if n < DBO_MIN or t < n:
        return None
    w = c[t - n + 1: t + 1]
    ma, sd = float(w.mean()), float(w.std())
    return {"n": n, "hh": float(h[t - n: t].max()), "ll": float(l[t - n: t].min()), "ma": ma,
            "ub": ma + 2.0 * sd, "lb": ma - 2.0 * sd, "c": float(c[t])}


# ══════════════════════════════════════════════════════════════════════════
# pysystemtrade: EWMAC, robust volatility, mr_wings
# ══════════════════════════════════════════════════════════════════════════
def robust_vol(price, days: int = 35, min_periods: int = 10, floor_window: int = 500, floor_min_periods: int = 100,
               floor_q: float = 0.05) -> np.ndarray:
    """Daily PRICE-unit volatility as pysystemtrade's robust_vol_calc defaults (sysquant/estimators/vol.py is not in
    the downloaded copy; reconstructed from its documented defaults): EWMA std (span 35, min 10 obs, pandas
    adjust=True) of daily price changes, floored at the rolling 5 % quantile of itself over 500 days (>= 100 obs)."""
    s = pd.Series(_a(price)).diff()
    vol = s.ewm(span=days, min_periods=min_periods, adjust=True).std()
    vmin = vol.rolling(floor_window, min_periods=floor_min_periods).quantile(floor_q)
    out = np.where(np.isfinite(vmin.to_numpy()), np.maximum(vol.to_numpy(), vmin.to_numpy()), vol.to_numpy())
    return out


def ewmac(price, vol, fast: int, slow: int) -> np.ndarray:
    """rules/ewmac.py ewmac(): (EWMA_fast - EWMA_slow) / vol, spans, min_periods=1 (pandas adjust=True)."""
    p = pd.Series(_a(price))
    raw = p.ewm(span=fast, min_periods=1).mean() - p.ewm(span=slow, min_periods=1).mean()
    v = pd.Series(_a(vol)).ffill()
    return (raw / v).to_numpy()


def mr_wings(price, fast: int = 4) -> np.ndarray:
    """rules/mr_wings.py: ewmac(4, 16) / vol; zero where |ewmac| < 3 x its rolling(5000, min 3) std; sign flipped.
    Returns the continuous forecast (the strategy trades its SIGN: the binary-position version)."""
    vol = robust_vol(price)
    e = pd.Series(ewmac(price, vol, fast, fast * 4))
    sd = e.rolling(5000, min_periods=3).std()
    e = e.where(~(e.abs() < sd * 3), 0.0)
    return (-e).to_numpy()


# ══════════════════════════════════════════════════════════════════════════
# Moon phases — Meeus, Astronomical Algorithms (2nd ed.), chapter 49
# ══════════════════════════════════════════════════════════════════════════
_NEW = ((-0.40720, "M'"), (0.17241, "E M"), (0.01608, "2M'"), (0.01039, "2F"), (0.00739, "E M'-M"),
        (-0.00514, "E M'+M"), (0.00208, "E2 2M"), (-0.00111, "M'-2F"), (-0.00057, "M'+2F"), (0.00056, "E 2M'+M"),
        (-0.00042, "3M'"), (0.00042, "E M+2F"), (0.00038, "E M-2F"), (-0.00024, "E 2M'-M"), (-0.00017, "O"),
        (-0.00007, "M'+2M"), (0.00004, "2M'-2F"), (0.00004, "3M"), (0.00003, "M'+M-2F"), (0.00003, "2M'+2F"),
        (-0.00003, "M'+M+2F"), (0.00003, "M'-M+2F"), (-0.00002, "M'-M-2F"), (-0.00002, "3M'+M"), (0.00002, "4M'"))
_FULL = ((-0.40614, "M'"), (0.17302, "E M"), (0.01614, "2M'"), (0.01043, "2F"), (0.00734, "E M'-M"),
         (-0.00515, "E M'+M"), (0.00209, "E2 2M"), (-0.00111, "M'-2F"), (-0.00057, "M'+2F"), (0.00056, "E 2M'+M"),
         (-0.00042, "3M'"), (0.00042, "E M+2F"), (0.00038, "E M-2F"), (-0.00024, "E 2M'-M"), (-0.00017, "O"),
         (-0.00007, "M'+2M"), (0.00004, "2M'-2F"), (0.00004, "3M"), (0.00003, "M'+M-2F"), (0.00003, "2M'+2F"),
         (-0.00003, "M'+M+2F"), (0.00003, "M'-M+2F"), (-0.00002, "M'-M-2F"), (-0.00002, "3M'+M"), (0.00002, "4M'"))
_QUARTER = ((-0.62801, "M'"), (0.17172, "E M"), (-0.01183, "E M'+M"), (0.00862, "2M'"), (0.00804, "2F"),
            (0.00454, "E M'-M"), (0.00204, "E2 2M"), (-0.00180, "M'-2F"), (-0.00070, "M'+2F"), (-0.00040, "3M'"),
            (-0.00034, "E 2M'-M"), (0.00032, "E M+2F"), (0.00032, "E M-2F"), (-0.00028, "E2 M'+2M"),
            (0.00027, "E 2M'+M"), (-0.00017, "O"), (-0.00005, "M'-M-2F"), (0.00004, "2M'+2F"), (-0.00004, "M'+M+2F"),
            (0.00004, "M'-2M"), (0.00003, "M'+M-2F"), (0.00003, "3M"), (0.00002, "2M'-2F"), (0.00002, "M'-M+2F"),
            (-0.00002, "3M'+M"))
_A_COEF = (0.000325, 0.000165, 0.000164, 0.000126, 0.000110, 0.000062, 0.000060, 0.000056, 0.000047, 0.000042,
           0.000040, 0.000037, 0.000035, 0.000023)
DELTA_T_DAYS = 69.0 / 86400.0      # TD - UT around 2012-2026 (~67-69 s); a minute-level effect, stated in the prereg


_TERM = __import__("re").compile(r"([+-]?\d*)(M'|M|F|O)")


def _arg(expr: str, M, Mp, F, O):
    """Value of an argument such as "2M'-2F" or "M'+M+2F" (the E / E2 factor prefix is ignored here)."""
    terms = {"M": M, "M'": Mp, "F": F, "O": O}
    s = expr.replace("E2 ", "").replace("E ", "").replace(" ", "")
    tot = 0.0
    for coef, name in _TERM.findall(s):
        k = -1.0 if coef == "-" else (1.0 if coef in ("", "+") else float(coef))
        tot += k * terms[name]
    return tot


def _efac(expr: str, E):
    return E * E if expr.startswith("E2 ") else (E if expr.startswith("E ") else 1.0)


def moon_phase_jde(k: float) -> float:
    """JDE (Terrestrial Dynamical Time) of the phase k (integer = new moon, +0.25 first quarter, +0.5 full, +0.75
    last quarter), Meeus eq. 49.1 with the periodic and the 14 planetary-argument corrections."""
    T = k / 1236.85
    jde = (2451550.09766 + 29.530588861 * k + 0.00015437 * T ** 2 - 0.000000150 * T ** 3 + 0.00000000073 * T ** 4)
    E = 1 - 0.002516 * T - 0.0000074 * T ** 2
    d = math.radians
    M = d(2.5534 + 29.10535670 * k - 0.0000014 * T ** 2 - 0.00000011 * T ** 3)
    Mp = d(201.5643 + 385.81693528 * k + 0.0107582 * T ** 2 + 0.00001238 * T ** 3 - 0.000000058 * T ** 4)
    F = d(160.7108 + 390.67050284 * k - 0.0016118 * T ** 2 - 0.00000227 * T ** 3 + 0.000000011 * T ** 4)
    O = d(124.7746 - 1.56375588 * k + 0.0020672 * T ** 2 + 0.00000215 * T ** 3)
    frac = round((k - math.floor(k)) * 4) % 4
    table = _NEW if frac == 0 else (_FULL if frac == 2 else _QUARTER)
    corr = sum(cf * _efac(ex, E) * math.sin(_arg(ex, M, Mp, F, O)) for cf, ex in table)
    if frac in (1, 3):
        W = (0.00306 - 0.00038 * E * math.cos(M) + 0.00026 * math.cos(Mp) - 0.00002 * math.cos(Mp - M)
             + 0.00002 * math.cos(Mp + M) + 0.00002 * math.cos(2 * F))
        corr += W if frac == 1 else -W
    A = (299.77 + 0.107408 * k - 0.009173 * T ** 2, 251.88 + 0.016321 * k, 251.83 + 26.651886 * k,
         349.42 + 36.412478 * k, 84.66 + 18.206239 * k, 141.74 + 53.303771 * k, 207.14 + 2.453732 * k,
         154.84 + 7.306860 * k, 34.52 + 27.261239 * k, 207.19 + 0.121824 * k, 291.34 + 1.844379 * k,
         161.72 + 24.198154 * k, 239.56 + 25.513099 * k, 331.55 + 3.592518 * k)
    corr += sum(cf * math.sin(math.radians(a)) for cf, a in zip(_A_COEF, A))
    return jde + corr


def jd_to_datetime(jd: float):
    """Julian Day -> naive UTC datetime (proleptic Gregorian)."""
    import datetime as _dt
    return _dt.datetime(2000, 1, 1, 12) + _dt.timedelta(days=jd - 2451545.0)


def quarter_events(y0: int, y1: int) -> list:
    """[(utc_datetime, 'FQ' | 'LQ')] for first / last quarters between years y0 and y1 (inclusive), sorted.
    UT = TD - DELTA_T_DAYS."""
    out = []
    k0 = math.floor((y0 - 2000) * 12.3685) - 2
    k1 = math.ceil((y1 + 1 - 2000) * 12.3685) + 2
    for k in range(k0, k1):
        for f, tag in ((0.25, "FQ"), (0.75, "LQ")):
            dt = jd_to_datetime(moon_phase_jde(k + f) - DELTA_T_DAYS)
            if y0 <= dt.year <= y1:
                out.append((dt, tag))
    out.sort()
    return out


_LUNAR_CACHE: dict = {}


def lunar_state(date_int: int, close_utc_hour: float = 10.0) -> str:
    """'LQ' or 'FQ': the latest quarter event at or before the session close (15:30 IST = 10:00 UTC) of date_int.
    Long after a last quarter (7 days before the new moon), short after a first quarter (7 days before the full
    moon), as QuantConnect tutorial 61 codes it."""
    import bisect
    import datetime as _dt
    if "ev" not in _LUNAR_CACHE:
        ev = quarter_events(2010, 2028)
        _LUNAR_CACHE["ev"] = ev
        _LUNAR_CACHE["t"] = [e[0] for e in ev]
    y, m, d = date_int // 10000, date_int // 100 % 100, date_int % 100
    cl = _dt.datetime(y, m, d) + _dt.timedelta(hours=close_utc_hour)
    i = bisect.bisect_right(_LUNAR_CACHE["t"], cl) - 1
    return _LUNAR_CACHE["ev"][i][1]


# ══════════════════════════════════════════════════════════════════════════
# Ghost trader (letianzj backtest/ghost_trader.py)
# ══════════════════════════════════════════════════════════════════════════
GHOST = {"ma_short": 3, "ma_long": 21, "rsi_n": 9, "oversold": 30, "overbought": 70, "donchian_n": 21}


def ghost_replay(h, l, c, as_coded_window: bool = False) -> tuple:
    """Replays ghost_trader.on_tick over all bars. Returns (enter, exit_) int8 arrays: enter[i] = +1 / -1 = an entry
    decided at the close of bar i; exit_[i] = 1 = the open position is closed at the close of bar i. A decision at
    bar i changes the position from bar i+1 (fill at the next open), so `pos` below is the position at the START of
    each bar, as the source reads it. Indicators: TA-Lib-style EMA (SMA seed) 3 / 21 and Wilder RSI 9.
    Donchian stops use the 21 bars BEFORE i (fix; the source's window includes bar i, so 'close <= lowest low' could
    only fire on a close at the bar's own low)."""
    h, l, c = _a(h), _a(l), _a(c)
    n = len(c)
    g = GHOST
    e_s, e_l, r = K.ema_sma_seed(c, g["ma_short"]), K.ema_sma_seed(c, g["ma_long"]), K.rsi(c, g["rsi_n"])
    dn = g["donchian_n"]
    lookback = max(g["ma_long"], g["rsi_n"], dn)
    enter = np.zeros(n, np.int8)
    exit_ = np.zeros(n, np.int8)
    pos = 0
    lg = sg = False
    lp = sp = 0.0
    for i in range(n):
        if i + 1 < lookback or i < dn or i < 1:
            continue
        if as_coded_window:
            lstop, sstop = l[i - dn + 1: i + 1].min(), h[i - dn + 1: i + 1].max()
        else:
            lstop, sstop = l[i - dn: i].min(), h[i - dn: i].max()
        new = pos
        ok = np.isfinite(e_s[i]) and np.isfinite(e_l[i]) and np.isfinite(r[i])
        if pos == 0 and ok and e_s[i] > e_l[i] and r[i] < g["overbought"] and h[i] > h[i - 1]:
            if not lg:
                lp, lg = c[i], True
            if lg and lp > c[i]:
                lg = False
                enter[i], new = 1, 1
        elif pos > 0 and c[i] <= lstop:
            exit_[i], new = 1, 0
        if pos == 0 and ok and e_s[i] < e_l[i] and r[i] > g["oversold"] and l[i] < l[i - 1]:
            if not sg:
                sp, sg = c[i], True
            if sg and sp < c[i]:
                sg = False
                enter[i], new = -1, -1
        elif pos < 0 and h[i] >= sstop:
            exit_[i], new = 1, 0
        pos = new
    return enter, exit_


# ══════════════════════════════════════════════════════════════════════════
# R-Breaker (letianzj backtest/r_breaker.py)
# ══════════════════════════════════════════════════════════════════════════
def rbreaker_levels(yh: float, yl: float, yc: float) -> dict:
    """Seven levels from yesterday's high / low / close exactly as coded (classic floor pivots)."""
    pv = (yh + yc + yl) / 3.0
    return {"pivot": pv, "r1": 2 * pv - yl, "r2": pv + (yh - yl), "r3": yh + 2 * (pv - yl),
            "s1": 2 * pv - yh, "s2": pv - (yh - yl), "s3": yl - 2 * (yh - pv)}


def rbreaker_day(c, day_hi, day_lo, lv: dict) -> tuple:
    """State machine over one day's candles (closes c, running day high / low INCLUDING the candle). Returns
    (enter, flip) where enter[j] = +1 / -1 = position opened or reversed at candle j's close, flip[j] = the open
    position is closed (and reversed) at j. Flat -> long if close > r3, short if close < s3; long -> short if
    (day high > r2 and close < r1) or close < s3; short -> long if (day low < s2 and close > s1) or close > r3."""
    n = len(c)
    enter = np.zeros(n, np.int8)
    flip = np.zeros(n, bool)
    pos = 0
    for j in range(n):
        if pos == 0:
            if c[j] > lv["r3"]:
                enter[j], pos = 1, 1
            elif c[j] < lv["s3"]:
                enter[j], pos = -1, -1
        elif pos > 0:
            if (day_hi[j] > lv["r2"] and c[j] < lv["r1"]) or c[j] < lv["s3"]:
                enter[j], flip[j], pos = -1, True, -1
        else:
            if (day_lo[j] < lv["s2"] and c[j] > lv["s1"]) or c[j] > lv["r3"]:
                enter[j], flip[j], pos = 1, True, 1
    return enter, flip


# ══════════════════════════════════════════════════════════════════════════
# Cross-sectional formulas (portfolio strategies)
# ══════════════════════════════════════════════════════════════════════════
def clenow_momentum(closes: np.ndarray) -> np.ndarray:
    """Columns = stocks, rows = the last `window` closes (oldest first). Annualised exponential regression slope x
    R^2 x 100: ((exp(252 slope of ln close on 0..n-1) - 1) x 100) x r^2 (NiftyOnTheMove indicators.Momentum)."""
    y = np.log(np.asarray(closes, float))
    n = y.shape[0]
    x = np.arange(n, dtype=float)
    xd = x - x.mean()
    yd = y - y.mean(axis=0)
    ssx = (xd ** 2).sum()
    sxy = (xd[:, None] * yd).sum(axis=0)
    ssy = (yd ** 2).sum(axis=0)
    slope = sxy / ssx
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ssy > 0, sxy / np.sqrt(ssx * ssy), 0.0)
    return (np.exp(slope * 252) - 1) * 100 * r ** 2


def residual_score(stock_m: np.ndarray, mkt_m: np.ndarray) -> float:
    """QuantConnect tutorial 136 on a market-only regression: OLS of the 36 monthly returns on the market's
    (intercept + beta), residuals of the 12 most recent months excluding the latest one (11 values), score =
    sum / population std of those residuals. NaN if anything is missing or the std is 0."""
    y, x = np.asarray(stock_m, float), np.asarray(mkt_m, float)
    if len(y) != 36 or len(x) != 36 or not (np.isfinite(y).all() and np.isfinite(x).all()):
        return float("nan")
    X = np.column_stack([np.ones(36), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    w = e[-12:-1]
    sd = float(w.std())
    return float(w.sum() / sd) if sd > 0 else float("nan")


def alpha101(name: str, O, H, L, C, V, t: int) -> np.ndarray:
    """One WorldQuant alpha at session t for every stock (columns) from (N, M) arrays; uses rows <= t only. NaN where
    an input is missing. Ranks are applied by the caller (the long / short quintiles rank this value)."""
    o, h, l, c, v = O[t], H[t], L[t], C[t], V[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        if name == "a101":                        # (close - open) / ((high - low) + .001)
            return (c - o) / ((h - l) + 0.001)
        if name == "a33":                         # rank(-1 * ((1 - (open / close))^1))
            return -1.0 * (1.0 - o / c)
        if name == "a12":                         # sign(delta(volume, 1)) * (-1 * delta(close, 1))
            return np.sign(v - V[t - 1]) * (-1.0 * (c - C[t - 1]))
        if name == "a54":                         # (-1 * ((low - close) * (open^5))) / ((low - high) * (close^5))
            return (-1.0 * ((l - c) * o ** 5)) / ((l - h) * c ** 5)
        if name == "a9":                          # ts_min / ts_max of delta(close, 1) over 5 sessions
            dc = C[t - 4: t + 1] - C[t - 5: t]
            d0 = dc[-1]
            mn, mx = dc.min(axis=0), dc.max(axis=0)
            return np.where(0 < mn, d0, np.where(mx < 0, d0, -1.0 * d0))
    raise ValueError(name)


def pct_rank_desc(x: np.ndarray) -> np.ndarray:
    """pandas rank(ascending=False, pct=True) over the finite values (average ties); NaN elsewhere."""
    s = pd.Series(np.asarray(x, float))
    return s.rank(ascending=False, pct=True).to_numpy()
