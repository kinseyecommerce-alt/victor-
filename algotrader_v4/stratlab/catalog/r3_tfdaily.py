"""
Round-3 TIMEFRAME DAILY: classic daily indicator systems (stocks CNC long, stock futures short, NIFTY / BANKNIFTY
index futures). Pre-registered BEFORE any result in scratchpad/r3/tfdaily/PREREGISTERED.txt -- parameters are
textbook / round values; do not tune them.

Every function gets a DailyView `d` ending at the decision day (d.c[-1]); fill at the next session's open.
The positional simulator only exits by fixed stop / target / max_hold, so indicator exits are approximated by a
fixed holding period plus an initial indicator-defined stop (stated in the pre-registration).
"""
from __future__ import annotations

import numpy as np

from ..api import LONG, SHORT, Swing, strategy
from ..indicators import ema, sma, highest, lowest, true_range

HYG_N = 130          # data-hygiene window (sessions), same as r3_swing
HYG_JUMP = 0.25
PATH_N = 250         # window for path-dependent indicators (Supertrend, PSAR)
EMA_N = 400          # window for EMA-based indicators (converged)


# ── helpers ─────────────────────────────────────────────────────────────────
def _clean(d) -> bool:
    n = min(len(d), HYG_N + 1)
    if n < 2:
        return True
    c = d.c[-n:]
    o = d.o[-n:]
    pc = c[:-1]
    return not (np.any(np.abs(c[1:] / pc - 1.0) > HYG_JUMP) or np.any(np.abs(o[1:] / pc - 1.0) > HYG_JUMP))


def _ok(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def _wilder(x, n: int) -> np.ndarray:
    """Wilder smoothing (RMA) as an EWM with alpha 1/n."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    a = 1.0 / n
    acc = x[0]
    for i in range(len(x)):
        acc = x[i] if i == 0 else acc + a * (x[i] - acc)
        out[i] = acc
    return out


def _atr_arr(h, l, c, n: int) -> np.ndarray:
    return _wilder(true_range(h, l, c), n)


def _tail(d, k: int):
    k = min(len(d), k)
    return d.h[-k:], d.l[-k:], d.c[-k:]


def _atr_last(d, n: int = 14) -> float:
    if len(d) < 3 * n:
        return np.nan
    h, l, c = _tail(d, 150)
    return float(_atr_arr(h, l, c, n)[-1])


def _sma_last(c, n: int) -> float:
    return float(np.mean(c[-n:])) if len(c) >= n else np.nan


def _stop_atr(d, side: int, k: float, n: int = 14):
    a = _atr_last(d, n)
    if not _ok(a) or a <= 0:
        return None
    return float(d.c[-1] - side * k * a)


def _macd(c):
    c = c[-EMA_N:]
    m = ema(c, 12) - ema(c, 26)
    return m, ema(m, 9)


def _supertrend(h, l, c, n: int = 10, mult: float = 3.0):
    """Standard Supertrend: returns (line, direction) arrays; direction +1 up, -1 down."""
    a = _atr_arr(h, l, c, n)
    hl2 = (h + l) / 2.0
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fu, fl = ub.copy(), lb.copy()
    dirn = np.ones(len(c), int)
    line = np.empty(len(c))
    for i in range(len(c)):
        if i > 0:
            fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
            if dirn[i - 1] == 1:
                dirn[i] = -1 if c[i] < fl[i] else 1
            else:
                dirn[i] = 1 if c[i] > fu[i] else -1
        line[i] = fl[i] if dirn[i] == 1 else fu[i]
    return line, dirn


def _psar(h, l, af0: float = 0.02, step: float = 0.02, amax: float = 0.2):
    """Wilder Parabolic SAR: sar[i] = the SAR plotted on bar i (on a flip bar: the prior trend's extreme point,
    i.e. the new stop); dirn[i] = +1 long / -1 short after bar i."""
    n = len(h)
    sar = np.empty(n)
    dirn = np.ones(n, int)
    up, af, ep, s = True, af0, h[0], l[0]
    sar[0] = s
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < s:                       # flip to short
                up, s, ep, af = False, ep, l[i], af0
            elif h[i] > ep:
                ep, af = h[i], min(af + step, amax)
        else:
            s = max(s, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > s:                       # flip to long
                up, s, ep, af = True, ep, h[i], af0
            elif l[i] < ep:
                ep, af = l[i], min(af + step, amax)
        sar[i] = s
        dirn[i] = 1 if up else -1
    return sar, dirn


def _dmi(h, l, c, n: int = 14):
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[0] = mdm[0] = 0.0
    tr = _wilder(true_range(h, l, c), n)
    pdi = 100.0 * _wilder(pdm, n) / tr
    mdi = 100.0 * _wilder(mdm, n) / tr
    dx = 100.0 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12)
    return pdi, mdi, _wilder(dx, n)


# ── MA crosses ──────────────────────────────────────────────────────────────
@strategy("r3_tfdaily_ema20_50_cross", "swing", "ma_cross",
          "EMA20 crosses above EMA50: buy next open, stop close - 2.5 ATR14, hold 20",
          {"fast": 20, "slow": 50, "atr_k": 2.5, "max_hold": 20})
def r3_tfdaily_ema20_50_cross(d, p):
    if len(d) < 3 * p.slow or not _clean(d):
        return None
    c = d.c[-EMA_N:]
    f, s = ema(c, p.fast), ema(c, p.slow)
    if f[-1] > s[-1] and f[-2] <= s[-2]:
        st = _stop_atr(d, LONG, p.atr_k)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


@strategy("r3_tfdaily_golden_cross", "swing", "ma_cross",
          "SMA50 crosses above SMA200 (golden cross): buy next open, stop close - 3 ATR14, hold 40",
          {"fast": 50, "slow": 200, "atr_k": 3.0, "max_hold": 40})
def r3_tfdaily_golden_cross(d, p):
    if len(d) < p.slow + 1 or not _clean(d):
        return None
    c = d.c
    f1, s1 = _sma_last(c, p.fast), _sma_last(c, p.slow)
    f0, s0 = _sma_last(c[:-1], p.fast), _sma_last(c[:-1], p.slow)
    if _ok(f1, s1, f0, s0) and f1 > s1 and f0 <= s0:
        st = _stop_atr(d, LONG, p.atr_k)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


def _ema_regime(d, p):
    if len(d) < 3 * p.slow:
        return None
    c = d.c[-EMA_N:]
    f, s = ema(c, p.fast), ema(c, p.slow)
    side = LONG if f[-1] > s[-1] else SHORT
    return Swing(side, stop_pct=p.stop_pct, max_hold=p.max_hold)


for _u, _nm in (("NIFTY", "nifty"), ("BANKNIFTY", "bnf")):
    strategy(f"r3_tfdaily_{_nm}_ema20_50_regime", "futures", "ma_cross",
             f"While flat: {_u} EMA20 > EMA50 long, else short; next open, 3 % stop, hold 5 (always-in proxy)",
             {"fast": 20, "slow": 50, "stop_pct": 3.0, "max_hold": 5},
             horizon="positional", underlying=_u)(_ema_regime)


# ── MACD ────────────────────────────────────────────────────────────────────
@strategy("r3_tfdaily_macd_sma200", "swing", "macd",
          "MACD(12,26) crosses above signal(9) with MACD < 0 and close > SMA200: buy, stop 2.5 ATR14, hold 15",
          {"trend": 200, "atr_k": 2.5, "max_hold": 15})
def r3_tfdaily_macd_sma200(d, p):
    if len(d) < p.trend or not _clean(d):
        return None
    m, s = _macd(d.c)
    if m[-1] > s[-1] and m[-2] <= s[-2] and m[-1] < 0 and d.c[-1] > _sma_last(d.c, p.trend):
        st = _stop_atr(d, LONG, p.atr_k)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


@strategy("r3_tfdaily_macd_short_below200", "futures", "macd",
          "MACD crosses below signal with MACD > 0 and close < SMA200: short stock future, stop 2.5 ATR14, hold 15",
          {"trend": 200, "atr_k": 2.5, "max_hold": 15}, horizon="positional")
def r3_tfdaily_macd_short_below200(d, p):
    if len(d) < p.trend or not _clean(d):
        return None
    m, s = _macd(d.c)
    if m[-1] < s[-1] and m[-2] >= s[-2] and m[-1] > 0 and d.c[-1] < _sma_last(d.c, p.trend):
        st = _stop_atr(d, SHORT, p.atr_k)
        return None if st is None else Swing(SHORT, stop=st, max_hold=p.max_hold)
    return None


# ── Supertrend ──────────────────────────────────────────────────────────────
@strategy("r3_tfdaily_supertrend_flip", "swing", "supertrend",
          "Supertrend(10,3) flips to up: buy next open, stop at the Supertrend line, hold 20",
          {"n": 10, "mult": 3.0, "max_hold": 20})
def r3_tfdaily_supertrend_flip(d, p):
    if len(d) < 5 * p.n or not _clean(d):
        return None
    h, l, c = _tail(d, PATH_N)
    line, dr = _supertrend(h, l, c, p.n, p.mult)
    if dr[-1] == 1 and dr[-2] == -1:
        return Swing(LONG, stop=float(line[-1]), max_hold=p.max_hold)
    return None


def _st_regime(d, p):
    if len(d) < 5 * p.n:
        return None
    h, l, c = _tail(d, PATH_N)
    line, dr = _supertrend(h, l, c, p.n, p.mult)
    return Swing(int(dr[-1]), stop=float(line[-1]), max_hold=p.max_hold)


for _u, _nm in (("NIFTY", "nifty"), ("BANKNIFTY", "bnf")):
    strategy(f"r3_tfdaily_{_nm}_supertrend_regime", "futures", "supertrend",
             f"While flat: {_u} Supertrend(10,3) up long / down short, stop at the line, hold 5 (always-in proxy)",
             {"n": 10, "mult": 3.0, "max_hold": 5}, horizon="positional", underlying=_u)(_st_regime)


# ── Donchian / Turtle ───────────────────────────────────────────────────────
@strategy("r3_tfdaily_turtle55", "swing", "donchian",
          "Close above the highest high of the previous 55 sessions (Turtle S2): buy, stop 2 x ATR20, hold 20",
          {"n": 55, "atr_n": 20, "atr_k": 2.0, "max_hold": 20})
def r3_tfdaily_turtle55(d, p):
    if len(d) < p.n + 1 or not _clean(d):
        return None
    if d.c[-1] > d.h[-p.n - 1:-1].max():
        st = _stop_atr(d, LONG, p.atr_k, p.atr_n)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


# ── ADX / DMI ───────────────────────────────────────────────────────────────
def _adx_sig(d, p, side):
    if len(d) < 5 * p.n or not _clean(d):
        return None
    h, l, c = _tail(d, 200)
    pdi, mdi, adx = _dmi(h, l, c, p.n)
    a, b = (pdi, mdi) if side == LONG else (mdi, pdi)
    if a[-1] > b[-1] and a[-2] <= b[-2] and adx[-1] > p.adx_min:
        st = _stop_atr(d, side, p.atr_k)
        return None if st is None else Swing(side, stop=st, max_hold=p.max_hold)
    return None


@strategy("r3_tfdaily_adx_di_cross", "swing", "adx",
          "+DI(14) crosses above -DI(14) with ADX(14) > 25: buy next open, stop 2.5 ATR14, hold 15",
          {"n": 14, "adx_min": 25.0, "atr_k": 2.5, "max_hold": 15})
def r3_tfdaily_adx_di_cross(d, p):
    return _adx_sig(d, p, LONG)


@strategy("r3_tfdaily_adx_di_short", "futures", "adx",
          "-DI(14) crosses above +DI(14) with ADX(14) > 25: short stock future, stop 2.5 ATR14, hold 15",
          {"n": 14, "adx_min": 25.0, "atr_k": 2.5, "max_hold": 15}, horizon="positional")
def r3_tfdaily_adx_di_short(d, p):
    return _adx_sig(d, p, SHORT)


# ── Ichimoku ────────────────────────────────────────────────────────────────
def _ichimoku(d, p):
    """tenkan, kijun (arrays over the last k bars) and the cloud top/bottom at today and yesterday."""
    k = min(len(d), 200)
    h, l, c = d.h[-k:], d.l[-k:], d.c[-k:]
    ten = (highest(h, p.t) + lowest(l, p.t)) / 2.0
    kij = (highest(h, p.k) + lowest(l, p.k)) / 2.0
    sa = (ten + kij) / 2.0
    sb = (highest(h, p.s) + lowest(l, p.s)) / 2.0
    # cloud plotted at bar i = spans computed at bar i - 26
    top = np.full(k, np.nan)
    bot = np.full(k, np.nan)
    top[p.k:] = np.maximum(sa[:-p.k], sb[:-p.k])
    bot[p.k:] = np.minimum(sa[:-p.k], sb[:-p.k])
    return c, ten, kij, top, bot


@strategy("r3_tfdaily_ichimoku_kumo_break", "swing", "ichimoku",
          "Close crosses above the Ichimoku cloud with Tenkan > Kijun and close > close 26 ago: stop Kijun, hold 20",
          {"t": 9, "k": 26, "s": 52, "max_hold": 20})
def r3_tfdaily_ichimoku_kumo_break(d, p):
    if len(d) < p.s + p.k + 2 or not _clean(d):
        return None
    c, ten, kij, top, bot = _ichimoku(d, p)
    if not _ok(top[-1], top[-2], ten[-1], kij[-1]):
        return None
    if c[-1] > top[-1] and c[-2] <= top[-2] and ten[-1] > kij[-1] and c[-1] > c[-1 - p.k]:
        return Swing(LONG, stop=float(kij[-1]), max_hold=p.max_hold)
    return None


@strategy("r3_tfdaily_ichimoku_tk_cross", "swing", "ichimoku",
          "Tenkan(9) crosses above Kijun(26) while close is above the cloud: buy, stop 2.5 ATR14, hold 20",
          {"t": 9, "k": 26, "s": 52, "atr_k": 2.5, "max_hold": 20})
def r3_tfdaily_ichimoku_tk_cross(d, p):
    if len(d) < p.s + p.k + 2 or not _clean(d):
        return None
    c, ten, kij, top, bot = _ichimoku(d, p)
    if not _ok(top[-1], ten[-1], kij[-1], ten[-2], kij[-2]):
        return None
    if ten[-1] > kij[-1] and ten[-2] <= kij[-2] and c[-1] > top[-1]:
        st = _stop_atr(d, LONG, p.atr_k)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


# ── Parabolic SAR ───────────────────────────────────────────────────────────
@strategy("r3_tfdaily_psar_flip", "swing", "psar",
          "Parabolic SAR(0.02, 0.2) flips to long: buy next open, stop at the new SAR, hold 10",
          {"af": 0.02, "amax": 0.2, "max_hold": 10})
def r3_tfdaily_psar_flip(d, p):
    if len(d) < 30 or not _clean(d):
        return None
    k = min(len(d), PATH_N)
    sar, dr = _psar(d.h[-k:], d.l[-k:], p.af, p.af, p.amax)
    if dr[-1] == 1 and dr[-2] == -1:
        return Swing(LONG, stop=float(sar[-1]), max_hold=p.max_hold)
    return None


def _psar_idx(d, p):
    if len(d) < 30:
        return None
    k = min(len(d), PATH_N)
    sar, dr = _psar(d.h[-k:], d.l[-k:], p.af, p.af, p.amax)
    if dr[-1] != dr[-2]:
        return Swing(int(dr[-1]), stop=float(sar[-1]), max_hold=p.max_hold)
    return None


for _u, _nm in (("NIFTY", "nifty"), ("BANKNIFTY", "bnf")):
    strategy(f"r3_tfdaily_{_nm}_psar_flip", "futures", "psar",
             f"{_u} Parabolic SAR(0.02, 0.2) flips: enter the new direction next open, stop at the new SAR, hold 10",
             {"af": 0.02, "amax": 0.2, "max_hold": 10}, horizon="positional", underlying=_u)(_psar_idx)


# ── Stochastic / Williams %R ────────────────────────────────────────────────
@strategy("r3_tfdaily_stoch_oversold_up", "swing", "stochastic",
          "Slow %K(14,3) crosses above %D(3) with %D < 20 yesterday and close > SMA100: buy, stop 2.5 ATR, hold 10",
          {"n": 14, "smooth": 3, "lvl": 20.0, "trend": 100, "atr_k": 2.5, "max_hold": 10})
def r3_tfdaily_stoch_oversold_up(d, p):
    if len(d) < p.trend or not _clean(d):
        return None
    h, l, c = _tail(d, 60)
    hh, ll = highest(h, p.n), lowest(l, p.n)
    fk = 100.0 * (c - ll) / np.where(hh > ll, hh - ll, np.nan)
    sk = sma(fk, p.smooth)
    sd = sma(sk, p.smooth)
    if not _ok(sk[-1], sk[-2], sd[-1], sd[-2]):
        return None
    if sk[-1] > sd[-1] and sk[-2] <= sd[-2] and sd[-2] < p.lvl and d.c[-1] > _sma_last(d.c, p.trend):
        st = _stop_atr(d, LONG, p.atr_k)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


@strategy("r3_tfdaily_willr_overbought_short", "futures", "stochastic",
          "Williams %R(14) crosses below -20 from above with close < SMA100: short stock future, stop 2.5 ATR, hold 10",
          {"n": 14, "lvl": -20.0, "trend": 100, "atr_k": 2.5, "max_hold": 10}, horizon="positional")
def r3_tfdaily_willr_overbought_short(d, p):
    if len(d) < p.trend or not _clean(d):
        return None
    h, l, c = _tail(d, 30)
    hh, ll = highest(h, p.n), lowest(l, p.n)
    wr = -100.0 * (hh - c) / np.where(hh > ll, hh - ll, np.nan)
    if not _ok(wr[-1], wr[-2]):
        return None
    if wr[-1] < p.lvl and wr[-2] >= p.lvl and d.c[-1] < _sma_last(d.c, p.trend):
        st = _stop_atr(d, SHORT, p.atr_k)
        return None if st is None else Swing(SHORT, stop=st, max_hold=p.max_hold)
    return None


# ── CCI ─────────────────────────────────────────────────────────────────────
def _cci(d, n: int):
    h, l, c = _tail(d, 60)
    tp = (h + l + c) / 3.0
    out = np.full(len(tp), np.nan)
    for i in range(n - 1, len(tp)):
        w = tp[i - n + 1:i + 1]
        md = np.mean(np.abs(w - w.mean()))
        out[i] = (tp[i] - w.mean()) / (0.015 * md) if md > 0 else np.nan
    return out


@strategy("r3_tfdaily_cci_100_break", "swing", "cci",
          "CCI(20) crosses above +100 (Lambert buy): buy next open, stop 2.5 ATR14, hold 10",
          {"n": 20, "lvl": 100.0, "atr_k": 2.5, "max_hold": 10})
def r3_tfdaily_cci_100_break(d, p):
    if len(d) < 3 * p.n or not _clean(d):
        return None
    x = _cci(d, p.n)
    if _ok(x[-1], x[-2]) and x[-1] > p.lvl and x[-2] <= p.lvl:
        st = _stop_atr(d, LONG, p.atr_k)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


@strategy("r3_tfdaily_cci_m100_revert", "swing", "cci",
          "CCI(20) crosses back above -100 from below with close > SMA100: buy, stop 2.5 ATR14, hold 10",
          {"n": 20, "lvl": -100.0, "trend": 100, "atr_k": 2.5, "max_hold": 10})
def r3_tfdaily_cci_m100_revert(d, p):
    if len(d) < p.trend or not _clean(d):
        return None
    x = _cci(d, p.n)
    if _ok(x[-1], x[-2]) and x[-1] > p.lvl and x[-2] <= p.lvl and d.c[-1] > _sma_last(d.c, p.trend):
        st = _stop_atr(d, LONG, p.atr_k)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


# ── Chandelier ──────────────────────────────────────────────────────────────
@strategy("r3_tfdaily_chandelier_flip", "swing", "chandelier",
          "Close crosses above the Chandelier short line (LL22 + 3 ATR22): buy, stop HH22 - 3 ATR22, hold 20",
          {"n": 22, "mult": 3.0, "max_hold": 20})
def r3_tfdaily_chandelier_flip(d, p):
    if len(d) < 4 * p.n or not _clean(d):
        return None
    h, l, c = _tail(d, 150)
    a = _atr_arr(h, l, c, p.n)
    short_line = lowest(l, p.n) + p.mult * a
    long_line = highest(h, p.n) - p.mult * a
    if c[-1] > short_line[-1] and c[-2] <= short_line[-2]:
        return Swing(LONG, stop=float(long_line[-1]), max_hold=p.max_hold)
    return None


# ── Aroon ───────────────────────────────────────────────────────────────────
def _aroon(h, l, n: int):
    k = len(h)
    up = np.full(k, np.nan)
    dn = np.full(k, np.nan)
    for i in range(n, k):
        wh, wl = h[i - n:i + 1], l[i - n:i + 1]
        up[i] = 100.0 * (n - int(np.argmax(wh[::-1]))) / n     # argmax on reversed = sessions since the high
        dn[i] = 100.0 * (n - int(np.argmin(wl[::-1]))) / n
    return up, dn


@strategy("r3_tfdaily_aroon_cross", "swing", "aroon",
          "Aroon Up(25) crosses above Aroon Down(25) with Aroon Up >= 70: buy, stop 2.5 ATR14, hold 15",
          {"n": 25, "lvl": 70.0, "atr_k": 2.5, "max_hold": 15})
def r3_tfdaily_aroon_cross(d, p):
    if len(d) < p.n + 3 or not _clean(d):
        return None
    k = min(len(d), p.n + 5)
    up, dn = _aroon(d.h[-k:], d.l[-k:], p.n)
    if _ok(up[-1], up[-2], dn[-1], dn[-2]) and up[-1] > dn[-1] and up[-2] <= dn[-2] and up[-1] >= p.lvl:
        st = _stop_atr(d, LONG, p.atr_k)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


# ── Keltner ─────────────────────────────────────────────────────────────────
@strategy("r3_tfdaily_keltner_break", "swing", "keltner",
          "Close crosses above the upper Keltner band (EMA20 + 2 ATR10): buy next open, stop at EMA20, hold 15",
          {"n": 20, "mult": 2.0, "atr_n": 10, "max_hold": 15})
def r3_tfdaily_keltner_break(d, p):
    if len(d) < 4 * p.n or not _clean(d):
        return None
    h, l, c = _tail(d, 200)
    mid = ema(c, p.n)
    up = mid + p.mult * _atr_arr(h, l, c, p.atr_n)
    if c[-1] > up[-1] and c[-2] <= up[-2]:
        return Swing(LONG, stop=float(mid[-1]), max_hold=p.max_hold)
    return None


# ── MFI ─────────────────────────────────────────────────────────────────────
@strategy("r3_tfdaily_mfi_oversold_up", "swing", "mfi",
          "MFI(14) crosses back above 20 from below with close > SMA100: buy next open, stop 2.5 ATR14, hold 10",
          {"n": 14, "lvl": 20.0, "trend": 100, "atr_k": 2.5, "max_hold": 10})
def r3_tfdaily_mfi_oversold_up(d, p):
    if len(d) < p.trend or not _clean(d):
        return None
    k = p.n + 3
    h, l, c, v = d.h[-k:], d.l[-k:], d.c[-k:], d.v[-k:].astype(float)
    tp = (h + l + c) / 3.0
    mf = tp * v
    dtp = np.diff(tp)
    pos = np.where(dtp > 0, mf[1:], 0.0)
    neg = np.where(dtp < 0, mf[1:], 0.0)

    def mfi(end):                                  # MFI over the n money flows ending at index end (of pos/neg)
        ps, ng = pos[end - p.n + 1:end + 1].sum(), neg[end - p.n + 1:end + 1].sum()
        if ps + ng <= 0:
            return np.nan
        return 100.0 if ng == 0 else 100.0 - 100.0 / (1.0 + ps / ng)

    m1, m0 = mfi(len(pos) - 1), mfi(len(pos) - 2)
    if _ok(m1, m0) and m1 > p.lvl and m0 <= p.lvl and d.c[-1] > _sma_last(d.c, p.trend):
        st = _stop_atr(d, LONG, p.atr_k)
        return None if st is None else Swing(LONG, stop=st, max_hold=p.max_hold)
    return None


# ── Bollinger short in downtrend ────────────────────────────────────────────
@strategy("r3_tfdaily_bb_upper_short_downtrend", "futures", "bollinger",
          "Close above the upper Bollinger band (20, 2 sd) and close < SMA100: short stock future, stop 2.5 ATR, hold 5",
          {"n": 20, "k": 2.0, "trend": 100, "atr_k": 2.5, "max_hold": 5}, horizon="positional")
def r3_tfdaily_bb_upper_short_downtrend(d, p):
    if len(d) < p.trend or not _clean(d):
        return None
    w = d.c[-p.n:]
    upper = w.mean() + p.k * w.std()             # population sd (Bollinger's definition)
    if d.c[-1] > upper and d.c[-1] < _sma_last(d.c, p.trend):
        st = _stop_atr(d, SHORT, p.atr_k)
        return None if st is None else Swing(SHORT, stop=st, max_hold=p.max_hold)
    return None
