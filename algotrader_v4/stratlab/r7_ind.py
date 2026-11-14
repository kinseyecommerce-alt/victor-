"""
stratlab/r7_ind.py — formulas for round 7 (second GitHub shortlist, OHLCV-only items). Pure functions on numpy arrays
whose LAST element is the decision bar; every function reads only its inputs (no future values exist in them).
Tested against hand-computed values in stratlab/tests_r7.py.

Sources (read only, never executed; sha256 in stratlab/prereg/github_v2.json):
  clenow_ct        Trading_Evolved_Clenow "Chapter 17 - Counter Trend Trading/Counter Trend Model.ipynb"
  turtle_soup*     Raschke & Connors, Street Smarts (1995) ch. "Turtle Soup" / "Turtle Soup Plus One" (book rules; the
                   GitHub ports use a centred scipy argrelextrema = look-ahead, or a 3-bar close-back-inside rule)
  smash            handiko TradingStrategy-Public "Smash Day/README.md" rule list + SmashDay.mq5 (Williams)
  holy_grail       Street Smarts ch. "Holy Grail" (ADX(14) > 30 and rising, pullback to EMA(20), buy stop)
  double7          pinkfish examples/strategies/double-7s/strategy.py (Connors & Alvarez)
  mrl / mrs        TuringTrader BooksAndPubs/Bensdorp_30MinStockTrader.cs (The 30-Minute Stock Trader)
  tps              TuringTrader BooksAndPubs/Connors_HighProbEtfTrading.cs (TPS)
  eighty_twenty    Street Smarts ch. "80-20's"
  oops             Larry Williams' Oops (Long-Term Secrets to Short-Term Trading)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kite_ind as K

TAIL = 400            # Wilder / EMA recursions are run on the last TAIL bars (decay (13/14)^400 ~ 1e-13)


def _a(x) -> np.ndarray:
    return np.asarray(x, float)


# ══════════════════════════════════════════════════════════════════════════
# generic
# ══════════════════════════════════════════════════════════════════════════
def ewm_adjust_mean(x, span: int) -> float:
    """pandas Series.ewm(span=span).mean() (adjust=True) evaluated at the last element of x."""
    x = _a(x)
    a = 2.0 / (span + 1.0)
    w = (1.0 - a) ** np.arange(len(x) - 1, -1, -1)
    return float(np.sum(w * x) / np.sum(w))


def atr_sma(h, l, c, n: int) -> np.ndarray:
    """MetaTrader 5 iATR: simple n-bar mean of the true range (first bar: high - low)."""
    return K.sma(K.true_range(h, l, c), n)


def wilder_dmi(h, l, c, n: int):
    """(+DI, -DI, ADX) with Wilder smoothing (alpha 1/n) for DI and ADX, on the last TAIL bars."""
    h, l, c = _a(h)[-TAIL:], _a(l)[-TAIL:], _a(c)[-TAIL:]
    return K.pine_dmi(h, l, c, n, n)


def wilder_rsi(c, n: int) -> np.ndarray:
    return K.rsi(_a(c)[-TAIL:], n)


# ══════════════════════════════════════════════════════════════════════════
# 4. Clenow counter-trend (ch. 17)
# ══════════════════════════════════════════════════════════════════════════
CLENOW = {"vola_window": 40, "slow_ma": 80, "fast_ma": 40, "high_window": 20, "days_to_hold": 20, "dip_buy": -3.0,
          "hist": 250}


def clenow_ct(c, prm: dict = CLENOW) -> tuple:
    """(trend, pullback) at the last bar, exactly as the notebook: hist = the last 250 closes; trend =
    ewm(span 40).mean() > ewm(span 80).mean() (pandas adjust=True on that window); std = sample std (ddof 1) of the
    last 40 close DIFFERENCES (points, not %); pullback = (close - max of the last 20 closes) / std."""
    c = _a(c)[-prm["hist"]:]
    if len(c) < prm["vola_window"] + 1:
        return False, float("nan")
    trend = ewm_adjust_mean(c, prm["fast_ma"]) > ewm_adjust_mean(c, prm["slow_ma"])
    sd = float(np.std(np.diff(c)[-prm["vola_window"]:], ddof=1))
    if not sd > 0:
        return bool(trend), float("nan")
    return bool(trend), float((c[-1] - np.max(c[-prm["high_window"]:])) / sd)


# ══════════════════════════════════════════════════════════════════════════
# 7. Turtle Soup / Turtle Soup Plus One (Street Smarts)
# ══════════════════════════════════════════════════════════════════════════
def _prior_extreme(x, n: int, side: int) -> tuple:
    """(value, age) of the lowest (side +1) / highest (side -1) of the last n bars of x, age = bars back from the
    last bar (0 = the last bar). Ties: the MOST RECENT occurrence (the conservative age)."""
    w = _a(x)[-n:]
    v = float(np.min(w) if side > 0 else np.max(w))
    idx = np.flatnonzero(w == v)
    return v, int(len(w) - 1 - idx[-1])


def turtle_soup(h, l, n: int = 20, min_age: int = 4) -> list:
    """Orders for the NEXT session (Turtle Soup, Street Smarts). The prior 20-day low P = the lowest low of the 20
    sessions up to and including the decision day (= the 20 sessions before the trade session); it must have been
    made at least `min_age` sessions before the trade session (age on the decision day >= min_age - 1). Long:
    armed when the trade session trades below P (a new 20-day low), buy stop at P + 5 ticks (ticks added by the
    caller). Short mirror on the highs. Returns [(side, P)] for each side that qualifies."""
    out = []
    if len(l) < n:
        return out
    p, age = _prior_extreme(l, n, +1)
    if age >= min_age - 1:
        out.append((+1, p))
    q, age = _prior_extreme(h, n, -1)
    if age >= min_age - 1:
        out.append((-1, q))
    return out


def turtle_soup_plus_one(c, h, l, n: int = 20, min_age: int = 3) -> list:
    """Day One = the decision day t: its low is below the prior 20-day low P (lowest low of t-20..t-1) and its close
    is at or below P; P was made at least `min_age` sessions before t. Day Two order: buy stop at P, stop one tick
    under the lower of Day One's and Day Two's lows. Short mirror. Returns [(side, P, day_one_extreme)]."""
    c, h, l = _a(c), _a(h), _a(l)
    out = []
    if len(l) < n + 1:
        return out
    p, age = _prior_extreme(l[:-1], n, +1)
    if age + 1 >= min_age and l[-1] < p and c[-1] <= p:
        out.append((+1, p, float(l[-1])))
    q, age = _prior_extreme(h[:-1], n, -1)
    if age + 1 >= min_age and h[-1] > q and c[-1] >= q:
        out.append((-1, q, float(h[-1])))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 8. Williams Smash Day (rule list of the handiko README; SmashDay.mq5 for the stop / target)
# ══════════════════════════════════════════════════════════════════════════
SMASH = {"lookback": 4, "atr": 21, "atr_mult": 2.1, "tp_factor": 0.75, "valid": 2}


def smash_day(o, h, l, c, prm: dict = SMASH):
    """At the decision day t ("yesterday" of the rule list): long when c[t] < l[t-1] and l[t] < min(l[t-lookback..
    t-1]): buy stop at h[t]; short when c[t] > h[t-1] and h[t] > max(h[t-lookback..t-1]): sell stop at l[t].
    Stop = level -/+ 2.1 x ATR(21) (MT5 SMA ATR at t), target = level +/- 0.75 x that distance (the EA's ATR mode).
    Returns (side, level, stop, target) or None."""
    h, l, c = _a(h), _a(l), _a(c)
    lb = prm["lookback"]
    if len(c) < max(lb + 1, prm["atr"] + 1):
        return None
    a = float(atr_sma(h[-(prm["atr"] + 1):], l[-(prm["atr"] + 1):], c[-(prm["atr"] + 1):], prm["atr"])[-1])
    if not np.isfinite(a) or a <= 0:
        return None
    dist = prm["atr_mult"] * a
    if c[-1] < l[-2] and l[-1] < np.min(l[-1 - lb:-1]):
        lv = float(h[-1])
        return (+1, lv, lv - dist, lv + prm["tp_factor"] * dist)
    if c[-1] > h[-2] and h[-1] > np.max(h[-1 - lb:-1]):
        lv = float(l[-1])
        return (-1, lv, lv + dist, lv - prm["tp_factor"] * dist)
    return None


# ══════════════════════════════════════════════════════════════════════════
# 9. Raschke Holy Grail (Street Smarts)
# ══════════════════════════════════════════════════════════════════════════
HOLY = {"adx_n": 14, "adx_min": 30.0, "ema": 20, "swing_n": 20}


def holy_grail(h, l, c, prm: dict = HOLY):
    """Touch bar = the decision day t. Long: ADX(14) at t-1 > 30 and above ADX at t-2 (strong and rising when the
    pullback reached the average), +DI > -DI at t-1, close t-1 above EMA(20) and low t at / below EMA(20) (the first
    touch). Order: buy stop one tick above h[t] (tick added by the caller), stop = the new swing low (min of l[t]
    and the fill session's low) - 1 tick, target = the recent swing high = highest high of the 20 sessions before
    t. Short mirror. Returns (side, touch_extreme_for_level, swing_stop_base, target) or None."""
    h, l, c = _a(h), _a(l), _a(c)
    n = prm["swing_n"]
    if len(c) < max(3 * prm["adx_n"], prm["ema"]) + n + 2:
        return None
    pdi, mdi, adx = wilder_dmi(h, l, c, prm["adx_n"])
    e = K.ema(c[-TAIL:], prm["ema"])
    if not (np.isfinite(adx[-2]) and np.isfinite(adx[-3]) and adx[-2] > prm["adx_min"] and adx[-2] > adx[-3]):
        return None
    if pdi[-2] > mdi[-2] and c[-2] > e[-2] and l[-1] <= e[-1]:
        return (+1, float(h[-1]), float(l[-1]), float(np.max(h[-1 - n:-1])))
    if mdi[-2] > pdi[-2] and c[-2] < e[-2] and h[-1] >= e[-1]:
        return (-1, float(l[-1]), float(h[-1]), float(np.min(l[-1 - n:-1])))
    return None


# ══════════════════════════════════════════════════════════════════════════
# 12. Double 7s (pinkfish double-7s/strategy.py)
# ══════════════════════════════════════════════════════════════════════════
def double7_entry(c, period: int = 7, sma: int = 200) -> bool:
    """close == the lowest close of the last 7 (incl. today) and close > SMA(200) of closes."""
    c = _a(c)
    if len(c) < max(period, sma):
        return False
    return bool(c[-1] == np.min(c[-period:]) and c[-1] > np.mean(c[-sma:]))


def double7_exit(c, period: int = 7) -> bool:
    c = _a(c)
    return len(c) >= period and bool(c[-1] == np.max(c[-period:]))


# ══════════════════════════════════════════════════════════════════════════
# 13. Bensdorp MRL / MRS (30-Minute Stock Trader; TuringTrader port)
# ══════════════════════════════════════════════════════════════════════════
MRL = {"sma": 150, "adx_n": 7, "min_adx": 45.0, "atr_n": 10, "min_atr": 0.04, "rsi_n": 3, "max_rsi": 30.0,
       "stretch": 0.04, "stop_atr": 2.5, "target": 0.03, "max_hold": 4}
MRS = {"adx_n": 7, "min_adx": 50.0, "atr_n": 10, "min_atr": 0.05, "rsi_n": 3, "min_rsi": 85.0,
       "stop_atr": 2.5, "target": 0.04, "max_hold": 2}


def atr_pct(h, l, c, n: int) -> float:
    """TuringTrader TrueRange().Divide(Close).SMA(n) at the last bar."""
    h, l, c = _a(h)[-(n + 1):], _a(l)[-(n + 1):], _a(c)[-(n + 1):]
    tr = K.true_range(h, l, c)[1:]
    return float(np.mean(tr / c[1:])) if len(tr) == n else float("nan")


def mrl_setup(h, l, c, prm: dict = MRL):
    """Long filter (Wilder ADX / RSI, the book's definitions): close > SMA(150), ADX(7) > 45, ATR%(10) > 4 %,
    RSI(3) < 30. Returns (limit, stop, target) for the next session or None. Limit = close x (1 - 4 %) (as coded:
    the stretch equals the 4 % ATR threshold); stop = limit x (1 - 2.5 x ATR%); target = limit x 1.03 (exit at the
    next open after a close at / above it); 4-session time exit."""
    h, l, c = _a(h), _a(l), _a(c)
    if len(c) < max(prm["sma"], 3 * prm["adx_n"] + 2):
        return None
    ap = atr_pct(h, l, c, prm["atr_n"])
    _p, _m, adx = wilder_dmi(h, l, c, prm["adx_n"])
    r = wilder_rsi(c, prm["rsi_n"])
    if not (c[-1] > np.mean(c[-prm["sma"]:]) and adx[-1] > prm["min_adx"] and ap > prm["min_atr"]
            and r[-1] < prm["max_rsi"]):
        return None
    lim = float(c[-1]) * (1.0 - prm["stretch"])
    return lim, lim * (1.0 - prm["stop_atr"] * ap), lim * (1.0 + prm["target"])


def mrs_setup(h, l, c, prm: dict = MRS):
    """Short filter: two up closes, ADX(7) > 50, ATR%(10) > 5 %, RSI(3) > 85. As coded the short limit is the
    decision close itself (no stretch); stop = limit x (1 + 2.5 x ATR%); target = limit x 0.96 (close at / below it ->
    exit next open); 2-session time exit. Returns (limit, stop, target) or None."""
    h, l, c = _a(h), _a(l), _a(c)
    if len(c) < 3 * prm["adx_n"] + 2:
        return None
    ap = atr_pct(h, l, c, prm["atr_n"])
    _p, _m, adx = wilder_dmi(h, l, c, prm["adx_n"])
    r = wilder_rsi(c, prm["rsi_n"])
    if not (c[-1] > c[-2] > c[-3] and adx[-1] > prm["min_adx"] and ap > prm["min_atr"] and r[-1] > prm["min_rsi"]):
        return None
    lim = float(c[-1])
    return lim, lim * (1.0 + prm["stop_atr"] * ap), lim * (1.0 - prm["target"])


# ══════════════════════════════════════════════════════════════════════════
# 15. Connors TPS (High Probability ETF Trading; TuringTrader port) - the position machine
# ══════════════════════════════════════════════════════════════════════════
TPS = {"sma": 200, "rsi_n": 2, "entry_rsi": 25.0, "exit_rsi": 70.0, "sizes": (0.1, 0.2, 0.3, 0.4)}


def tps_signals(c, prm: dict = TPS) -> list:
    """Replays the TPS state machine over closes c (Wilder RSI(2), SMA(200)) and returns one action per bar, decided
    at that bar's close: ('enter', side, frac) / ('add', side, frac) / ('exit', side, 0) / None. Long: flat, close >
    SMA200, RSI2 < 25 today and yesterday -> 10 %; each later close below the last entry price adds 20 / 30 / 40 %;
    exit when RSI2 > 70. Short mirror (close < SMA200, RSI2 > 75 two days; adds on closes above; exit RSI2 < 30).
    Fills are the caller's (the next session's open)."""
    c = _a(c)
    rsi = K.rsi(c, prm["rsi_n"])
    sma = K.sma(c, prm["sma"])
    lo_in, hi_out = prm["entry_rsi"], prm["exit_rsi"]
    out = [None] * len(c)
    pos, n_in, last = 0, 0, np.nan
    for i in range(1, len(c)):
        if not (np.isfinite(sma[i]) and np.isfinite(rsi[i]) and np.isfinite(rsi[i - 1])):
            continue
        if pos == 0:
            if c[i] > sma[i] and rsi[i] < lo_in and rsi[i - 1] < lo_in:
                pos, n_in, last = 1, 1, c[i]
                out[i] = ("enter", 1, prm["sizes"][0])
            elif c[i] < sma[i] and rsi[i] > 100 - lo_in and rsi[i - 1] > 100 - lo_in:
                pos, n_in, last = -1, 1, c[i]
                out[i] = ("enter", -1, prm["sizes"][0])
        elif (pos > 0 and rsi[i] > hi_out) or (pos < 0 and rsi[i] < 100 - hi_out):
            out[i] = ("exit", pos, 0.0)
            pos, n_in, last = 0, 0, np.nan
        elif n_in < len(prm["sizes"]) and ((pos > 0 and c[i] < last) or (pos < 0 and c[i] > last)):
            n_in += 1
            last = c[i]
            out[i] = ("add", pos, prm["sizes"][n_in - 1])
    return out


# ══════════════════════════════════════════════════════════════════════════
# 14. intraday: Raschke 80-20 and Williams Oops (1-minute bars)
# ══════════════════════════════════════════════════════════════════════════
def eighty_twenty_side(pdo: float, pdh: float, pdl: float, pdc: float) -> int:
    """+1 when yesterday opened in the top 20 % of its range and closed in the bottom 20 % (buy setup), -1 for the
    mirror (opened in the bottom 20 %, closed in the top 20 %), 0 otherwise or on a zero range."""
    rng = pdh - pdl
    if not rng > 0:
        return 0
    po, pc = (pdo - pdl) / rng, (pdc - pdl) / rng
    if po >= 0.8 and pc <= 0.2:
        return 1
    if po <= 0.2 and pc >= 0.8:
        return -1
    return 0


def stop_reentry_bar(hi, lo, cl, side: int, arm: float, level: float, start: int = 0) -> int:
    """First minute index m >= start at whose close a resting stop at `level` has been triggered after the price
    first traded beyond `arm` (side +1: low < arm, then high >= level; -1 mirror). The arming bar itself counts as
    a trigger only if it closes at / beyond the level (inside one bar the order of its high and low is unknown).
    Returns -1 if none."""
    hi, lo, cl = _a(hi), _a(lo), _a(cl)
    armed = (lo[start:] < arm) if side > 0 else (hi[start:] > arm)
    if not armed.any():
        return -1
    k = start + int(np.argmax(armed))
    if (cl[k] >= level) if side > 0 else (cl[k] <= level):
        return k
    hit = (hi[k + 1:] >= level) if side > 0 else (lo[k + 1:] <= level)
    return k + 1 + int(np.argmax(hit)) if hit.any() else -1
