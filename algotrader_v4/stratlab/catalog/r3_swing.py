"""
Round-3 swing (CNC delivery, long only) candidates. Pre-registered BEFORE any result in
scratchpad/r3/swing/PREREGISTERED.txt -- parameters are textbook / round values; do not tune them.

Every function gets a DailyView `d` ending at the decision day (d.c[-1]); with decision_time="15:20" the last
bar is the day so far (15:19 close) and trend filters use closes up to yesterday only.
"""
from __future__ import annotations

import numpy as np

from ..api import LONG, Swing, strategy
from ..indicators import atr as _atr, rsi as _rsi

HYG_N = 130          # data-hygiene window (sessions)
HYG_JUMP = 0.25      # |close/prev close - 1| or |open/prev close - 1| above this -> unadjusted corporate action


# ── helpers ─────────────────────────────────────────────────────────────────
def _clean(d) -> bool:
    n = min(len(d), HYG_N + 1)
    if n < 2:
        return True
    c = d.c[-n:]
    o = d.o[-n:]
    pc = c[:-1]
    r1 = np.abs(c[1:] / pc - 1.0)
    r2 = np.abs(o[1:] / pc - 1.0)
    return not (np.any(r1 > HYG_JUMP) or np.any(r2 > HYG_JUMP))


def _sma(x, n: int, end: int | None = None) -> float:
    """Mean of the n values ending at index `end` (exclusive end; None = through the last)."""
    x = x if end is None else x[:end]
    if len(x) < n:
        return np.nan
    return float(np.mean(x[-n:]))


def _ret(c, n: int, end: int | None = None) -> float:
    c = c if end is None else c[:end]
    if len(c) < n + 1:
        return np.nan
    return float(c[-1] / c[-1 - n] - 1.0)


def _atr14(d) -> float:
    if len(d) < 30:
        return np.nan
    k = min(len(d), 120)                         # Wilder ATR on the last 120 sessions (converged)
    return float(_atr(d.h[-k:], d.l[-k:], d.c[-k:], 14)[-1])


def _rsi_last(c, n: int, k: int = 1):
    """Last k Wilder RSI(n) values computed on the last 100 closes (converged for n<=3)."""
    m = min(len(c), 100)
    r = _rsi(c[-m:], n)
    return r[-k:]


def _ok(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def _vix_ok(d, n: int) -> bool:
    return d.vix is not None and len(d.vix) >= n and np.all(np.isfinite(d.vix[-n:]))


def _nifty_ok(d, n: int) -> bool:
    return d.nifty is not None and len(d.nifty) >= n and np.all(np.isfinite(d.nifty[-n:]))


def _dom(date) -> int:
    return int(date) % 100


def _month(date) -> int:
    return (int(date) // 100) % 100


# ── short-term reversal (Connors) ───────────────────────────────────────────
@strategy("r3_swing_rsi2_sma200", "swing", "st_reversal",
          "RSI(2) < 10 and close > SMA200: buy next open, hold 5 sessions, 10 % disaster stop",
          {"rsi_n": 2, "lvl": 10.0, "sma": 200, "hold": 5, "stop_pct": 10.0})
def r3_swing_rsi2_sma200(d, p):
    if len(d) < p.sma + 1 or not _clean(d):
        return None
    sm = _sma(d.c, p.sma)
    r = _rsi_last(d.c, p.rsi_n)[-1]
    if _ok(sm, r) and r < p.lvl and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_rsi2_sma100_atr", "swing", "st_reversal",
          "RSI(2) < 10 and close > SMA100: buy next open, stop close - 2.5 ATR14, hold 5 sessions",
          {"rsi_n": 2, "lvl": 10.0, "sma": 100, "hold": 5, "atr_k": 2.5})
def r3_swing_rsi2_sma100_atr(d, p):
    if len(d) < p.sma + 1 or not _clean(d):
        return None
    sm = _sma(d.c, p.sma)
    r = _rsi_last(d.c, p.rsi_n)[-1]
    a = _atr14(d)
    if _ok(sm, r, a) and r < p.lvl and d.c[-1] > sm:
        return Swing(LONG, stop=float(d.c[-1] - p.atr_k * a), max_hold=p.hold)
    return None


@strategy("r3_swing_cumrsi2_sma100", "swing", "st_reversal",
          "RSI(2) today + yesterday < 35 and close > SMA100: buy next open, hold 5, 10 % stop",
          {"rsi_n": 2, "lvl": 35.0, "sma": 100, "hold": 5, "stop_pct": 10.0})
def r3_swing_cumrsi2_sma100(d, p):
    if len(d) < p.sma + 1 or not _clean(d):
        return None
    sm = _sma(d.c, p.sma)
    r = _rsi_last(d.c, p.rsi_n, 2)
    if len(r) == 2 and _ok(sm, r[0], r[1]) and r[0] + r[1] < p.lvl and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ── multi-day pullback in an uptrend ────────────────────────────────────────
def _three_down(d, sma_n):
    if len(d) < sma_n + 1 or not _clean(d):
        return False
    c = d.c
    sm = _sma(c, sma_n)
    return _ok(sm) and c[-1] < c[-2] < c[-3] < c[-4] and c[-1] > sm


@strategy("r3_swing_3down_sma100", "swing", "pullback",
          "3 consecutive lower closes and close > SMA100: buy next open, hold 5, 10 % stop",
          {"sma": 100, "hold": 5, "stop_pct": 10.0})
def r3_swing_3down_sma100(d, p):
    if _three_down(d, p.sma):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_3down_sma100_atr", "swing", "pullback",
          "3 consecutive lower closes and close > SMA100: stop close - 2 ATR14, target close + 2 ATR14, hold 10",
          {"sma": 100, "hold": 10, "atr_stop": 2.0, "atr_tgt": 2.0})
def r3_swing_3down_sma100_atr(d, p):
    if not _three_down(d, p.sma):
        return None
    a = _atr14(d)
    if not _ok(a):
        return None
    c = float(d.c[-1])
    return Swing(LONG, stop=c - p.atr_stop * a, target=c + p.atr_tgt * a, max_hold=p.hold)


@strategy("r3_swing_bb_lower_sma100", "swing", "pullback",
          "Close below the lower Bollinger band (20, 2 sd) and close > SMA100: buy next open, hold 5, 10 % stop",
          {"bb": 20, "k": 2.0, "sma": 100, "hold": 5, "stop_pct": 10.0})
def r3_swing_bb_lower_sma100(d, p):
    if len(d) < p.sma + 1 or not _clean(d):
        return None
    w = d.c[-p.bb:]
    lower = w.mean() - p.k * w.std()
    sm = _sma(d.c, p.sma)
    if _ok(sm, lower) and d.c[-1] < lower and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ── 126-session high anchoring (52-week proxy) ──────────────────────────────
@strategy("r3_swing_near_hi126", "swing", "high_anchor",
          "Close >= 0.98 x highest close of 126 sessions and close > SMA50: buy next open, hold 20, 8 % stop",
          {"n": 126, "near": 0.98, "sma": 50, "hold": 20, "stop_pct": 8.0})
def r3_swing_near_hi126(d, p):
    if len(d) < p.n or not _clean(d):
        return None
    hi = float(d.c[-p.n:].max())
    sm = _sma(d.c, p.sma)
    if _ok(sm) and d.c[-1] >= p.near * hi and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_fresh_hi126", "swing", "high_anchor",
          "First new 126-session closing high after 20 sessions without one: buy next open, hold 20, 8 % stop",
          {"n": 126, "quiet": 20, "hold": 20, "stop_pct": 8.0})
def r3_swing_fresh_hi126(d, p):
    if len(d) < p.n + p.quiet or not _clean(d):
        return None
    c = d.c
    if not c[-1] > c[-p.n:-1].max():
        return None
    for j in range(2, p.quiet + 2):              # days t-1 .. t-quiet: none was a new 126-session high
        if c[-j] > c[-j - p.n + 1:-j].max():
            return None
    return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)


@strategy("r3_swing_near_hi126_lowvol", "swing", "high_anchor",
          "Close >= 0.95 x 126-session high close and 20-day return sd < 1.5 %: buy next open, hold 20, 8 % stop",
          {"n": 126, "near": 0.95, "vol_n": 20, "max_sd": 0.015, "hold": 20, "stop_pct": 8.0})
def r3_swing_near_hi126_lowvol(d, p):
    if len(d) < p.n or not _clean(d):
        return None
    hi = float(d.c[-p.n:].max())
    r = np.diff(d.c[-p.vol_n - 1:]) / d.c[-p.vol_n - 1:-1]
    if d.c[-1] >= p.near * hi and r.std() < p.max_sd:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ── low volatility / low beta ───────────────────────────────────────────────
@strategy("r3_swing_lowvol_uptrend", "swing", "low_vol",
          "60-day log-return sd < 1.25 % and close > SMA50 > SMA100: buy next open, stop close - 3 ATR14, hold 20",
          {"vol_n": 60, "max_sd": 0.0125, "s1": 50, "s2": 100, "hold": 20, "atr_k": 3.0})
def r3_swing_lowvol_uptrend(d, p):
    if len(d) < p.s2 + 1 or not _clean(d):
        return None
    lr = np.diff(np.log(d.c[-p.vol_n - 1:]))
    s1, s2, a = _sma(d.c, p.s1), _sma(d.c, p.s2), _atr14(d)
    if _ok(s1, s2, a) and lr.std() < p.max_sd and d.c[-1] > s1 > s2:
        return Swing(LONG, stop=float(d.c[-1] - p.atr_k * a), max_hold=p.hold)
    return None


@strategy("r3_swing_lowbeta_rs", "swing", "low_vol",
          "60-day beta to NIFTY < 0.7 and 60-day return > NIFTY's: buy next open, hold 20, 8 % stop",
          {"n": 60, "max_beta": 0.7, "hold": 20, "stop_pct": 8.0}, needs_index=True)
def r3_swing_lowbeta_rs(d, p):
    if len(d) < p.n + 1 or not _nifty_ok(d, p.n + 1) or not _clean(d):
        return None
    sc, nc = d.c[-p.n - 1:], d.nifty[-p.n - 1:]
    rs, rn = np.diff(sc) / sc[:-1], np.diff(nc) / nc[:-1]
    var = rn.var()
    if var <= 0:
        return None
    beta = float(np.mean((rs - rs.mean()) * (rn - rn.mean())) / var)
    if beta < p.max_beta and sc[-1] / sc[0] > nc[-1] / nc[0]:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ── range contraction (NR7 / inside day) ────────────────────────────────────
def _nr7_at(h, l, j) -> bool:
    """Is the bar at negative index j (e.g. -2) the narrowest of the 7 bars ending at it?"""
    rg = (h - l)
    end = len(rg) + j + 1
    if end < 7:
        return False
    w = rg[end - 7:end]
    return bool(w[-1] < w[:-1].min())


@strategy("r3_swing_nr7_break", "swing", "range_contraction",
          "Yesterday NR7, today closes above yesterday's high and above SMA50: stop yesterday's low, 2R, hold 10",
          {"sma": 50, "rr": 2.0, "hold": 10})
def r3_swing_nr7_break(d, p):
    if len(d) < p.sma + 1 or not _clean(d):
        return None
    sm = _sma(d.c, p.sma)
    if _ok(sm) and _nr7_at(d.h, d.l, -2) and d.c[-1] > d.h[-2] and d.c[-1] > sm:
        return Swing(LONG, stop=float(d.l[-2]), rr=p.rr, max_hold=p.hold)
    return None


@strategy("r3_swing_inside_break", "swing", "range_contraction",
          "Yesterday an inside day, today closes above its high and above SMA50: stop inside-day low, 2R, hold 10",
          {"sma": 50, "rr": 2.0, "hold": 10})
def r3_swing_inside_break(d, p):
    if len(d) < p.sma + 1 or not _clean(d):
        return None
    sm = _sma(d.c, p.sma)
    inside = d.h[-2] < d.h[-3] and d.l[-2] > d.l[-3]
    if _ok(sm) and inside and d.c[-1] > d.h[-2] and d.c[-1] > sm:
        return Swing(LONG, stop=float(d.l[-2]), rr=p.rr, max_hold=p.hold)
    return None


@strategy("r3_swing_nr7_break_1520", "swing", "range_contraction",
          "15:20: yesterday NR7, price above yesterday's high, yesterday close > SMA50: buy 15:20, stop y-low, 2R, hold 10",
          {"sma": 50, "rr": 2.0, "hold": 10}, decision_time="15:20")
def r3_swing_nr7_break_1520(d, p):
    if len(d) < p.sma + 2 or not _clean(d):
        return None
    sm = _sma(d.c, p.sma, end=-1)                # closes up to yesterday
    if _ok(sm) and _nr7_at(d.h, d.l, -2) and d.c[-1] > d.h[-2] and d.c[-2] > sm:
        return Swing(LONG, stop=float(d.l[-2]), rr=p.rr, max_hold=p.hold)
    return None


# ── volume ──────────────────────────────────────────────────────────────────
@strategy("r3_swing_dryup_expand", "swing", "volume",
          "5-day volume < 0.6 x 50-day, then volume > 2x, +2 % close in top quarter of range: stop 1.5 ATR, hold 10",
          {"dry_n": 5, "dry": 0.6, "base_n": 50, "exp": 2.0, "up": 0.02, "top": 0.25, "hold": 10, "atr_k": 1.5})
def r3_swing_dryup_expand(d, p):
    if len(d) < p.base_n + 2 or not _clean(d):
        return None
    v = d.v
    base = float(v[-p.base_n - 1:-1].mean())
    dry = float(v[-p.dry_n - 1:-1].mean())
    rng = d.h[-1] - d.l[-1]
    a = _atr14(d)
    if not (_ok(a) and base > 0 and rng > 0):
        return None
    if dry < p.dry * base and v[-1] > p.exp * base and d.c[-1] >= (1 + p.up) * d.c[-2] \
            and (d.h[-1] - d.c[-1]) <= p.top * rng:
        return Swing(LONG, stop=float(d.c[-1] - p.atr_k * a), max_hold=p.hold)
    return None


@strategy("r3_swing_pocket_pivot", "swing", "volume",
          "Up day with volume > every down-day volume of the prior 10 sessions, close > SMA50 and >= SMA10: hold 10, 7 % stop",
          {"lb": 10, "s_long": 50, "s_short": 10, "hold": 10, "stop_pct": 7.0})
def r3_swing_pocket_pivot(d, p):
    if len(d) < p.s_long + 1 or not _clean(d):
        return None
    c, v = d.c, d.v
    if not c[-1] > c[-2]:
        return None
    pc = c[-p.lb - 2:-2]
    cc = c[-p.lb - 1:-1]
    vv = v[-p.lb - 1:-1]
    down = cc < pc
    mx = float(vv[down].max()) if down.any() else 0.0
    s1, s2 = _sma(c, p.s_long), _sma(c, p.s_short)
    if _ok(s1, s2) and v[-1] > mx and c[-1] > s1 and c[-1] >= s2:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ── event drift (earnings / news proxy) ─────────────────────────────────────
@strategy("r3_swing_gaphold_vol3", "swing", "event_drift",
          "Gap up >= 3 %, close >= open, volume >= 3 x 20-day mean: buy next open, stop today's low, hold 20",
          {"gap": 0.03, "vmult": 3.0, "vol_n": 20, "hold": 20})
def r3_swing_gaphold_vol3(d, p):
    if len(d) < p.vol_n + 2 or not _clean(d):
        return None
    v20 = float(d.v[-p.vol_n - 1:-1].mean())
    if v20 > 0 and d.o[-1] >= (1 + p.gap) * d.c[-2] and d.c[-1] >= d.o[-1] and d.v[-1] >= p.vmult * v20:
        return Swing(LONG, stop=float(d.l[-1]), max_hold=p.hold)
    return None


@strategy("r3_swing_gaphold_1520", "swing", "event_drift",
          "15:20: gap up >= 3 %, price >= open, volume so far >= 2.5 x 20-day mean: buy 15:20, stop day low, hold 20",
          {"gap": 0.03, "vmult": 2.5, "vol_n": 20, "hold": 20}, decision_time="15:20")
def r3_swing_gaphold_1520(d, p):
    if len(d) < p.vol_n + 2 or not _clean(d):
        return None
    v20 = float(d.v[-p.vol_n - 1:-1].mean())
    if v20 > 0 and d.o[-1] >= (1 + p.gap) * d.c[-2] and d.c[-1] >= d.o[-1] and d.v[-1] >= p.vmult * v20:
        return Swing(LONG, stop=float(d.l[-1]), max_hold=p.hold)
    return None


@strategy("r3_swing_hvol_up_drift", "swing", "event_drift",
          "Close >= +4 % on volume >= 3 x 20-day mean, close in top 30 % of range: buy next open, hold 20, 8 % stop",
          {"up": 0.04, "vmult": 3.0, "vol_n": 20, "top": 0.30, "hold": 20, "stop_pct": 8.0})
def r3_swing_hvol_up_drift(d, p):
    if len(d) < p.vol_n + 2 or not _clean(d):
        return None
    v20 = float(d.v[-p.vol_n - 1:-1].mean())
    rng = d.h[-1] - d.l[-1]
    if v20 > 0 and rng > 0 and d.c[-1] >= (1 + p.up) * d.c[-2] and d.v[-1] >= p.vmult * v20 \
            and (d.h[-1] - d.c[-1]) <= p.top * rng:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ── relative strength vs NIFTY ──────────────────────────────────────────────
@strategy("r3_swing_rs_newhigh", "swing", "rel_strength",
          "Close/NIFTY at its 63-session high, 63-day excess return >= 10 %, close > SMA50: hold 20, 8 % stop",
          {"n": 63, "excess": 0.10, "sma": 50, "hold": 20, "stop_pct": 8.0}, needs_index=True)
def r3_swing_rs_newhigh(d, p):
    if len(d) < p.n + 1 or not _nifty_ok(d, p.n + 1) or not _clean(d):
        return None
    rsl = d.c[-p.n:] / d.nifty[-p.n:]
    ex = _ret(d.c, p.n) - _ret(d.nifty, p.n)
    sm = _sma(d.c, p.sma)
    if _ok(ex, sm) and rsl[-1] >= rsl.max() and ex >= p.excess and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_rs_pullback", "swing", "rel_strength",
          "Excess return vs NIFTY t-68..t-5 >= 10 % and 5-day return <= -4 %: buy next open, hold 10, 10 % stop",
          {"n": 63, "skip": 5, "excess": 0.10, "pb": -0.04, "hold": 10, "stop_pct": 10.0}, needs_index=True)
def r3_swing_rs_pullback(d, p):
    if len(d) < p.n + p.skip + 1 or not _nifty_ok(d, p.n + p.skip + 1) or not _clean(d):
        return None
    ex = _ret(d.c, p.n, end=-p.skip) - _ret(d.nifty, p.n, end=-p.skip)
    r5 = _ret(d.c, p.skip)
    if _ok(ex, r5) and ex >= p.excess and r5 <= p.pb:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ── VIX regime ──────────────────────────────────────────────────────────────
@strategy("r3_swing_vix_spike", "swing", "vix",
          "India VIX >= 1.25 x its 20-day SMA and stock close > SMA100: buy next open, hold 5, 10 % stop",
          {"vix_n": 20, "mult": 1.25, "sma": 100, "hold": 5, "stop_pct": 10.0})
def r3_swing_vix_spike(d, p):
    if len(d) < p.sma + 1 or not _vix_ok(d, p.vix_n) or not _clean(d):
        return None
    vs = float(d.vix[-p.vix_n:].mean())
    sm = _sma(d.c, p.sma)
    if _ok(sm) and d.vix[-1] >= p.mult * vs and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_vix_capitulation", "swing", "vix",
          "India VIX at its 60-session high and stock 5-day return <= -5 %: buy next open, hold 10, 12 % stop",
          {"vix_n": 60, "r5": -0.05, "hold": 10, "stop_pct": 12.0})
def r3_swing_vix_capitulation(d, p):
    if len(d) < p.vix_n or not _vix_ok(d, p.vix_n) or not _clean(d):
        return None
    r5 = _ret(d.c, 5)
    if _ok(r5) and d.vix[-1] >= d.vix[-p.vix_n:].max() and r5 <= p.r5:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ── calendar: turn of the month ─────────────────────────────────────────────
def _tom_day(d, dom: int) -> bool:
    if len(d) < 2:
        return False
    t, y = d.date[-1], d.date[-2]
    return _dom(t) >= dom and (_dom(y) < dom or _month(y) != _month(t))


@strategy("r3_swing_tom", "swing", "calendar",
          "First session of the month with day-of-month >= 25: buy next open, hold 6 sessions, 10 % stop",
          {"dom": 25, "hold": 6, "stop_pct": 10.0})
def r3_swing_tom(d, p):
    if _tom_day(d, p.dom) and _clean(d):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_tom_uptrend", "swing", "calendar",
          "First session of the month with day-of-month >= 25 and close > SMA100: buy next open, hold 6, 10 % stop",
          {"dom": 25, "sma": 100, "hold": 6, "stop_pct": 10.0})
def r3_swing_tom_uptrend(d, p):
    if len(d) < p.sma or not _tom_day(d, p.dom) or not _clean(d):
        return None
    sm = _sma(d.c, p.sma)
    if _ok(sm) and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ── buy the close on a down day (15:20) ─────────────────────────────────────
@strategy("r3_swing_close_down2_sma100", "swing", "buy_close",
          "15:20: price <= 0.98 x previous close and previous close > SMA100: buy 15:20, hold 3, 10 % stop",
          {"dn": 0.02, "sma": 100, "hold": 3, "stop_pct": 10.0}, decision_time="15:20")
def r3_swing_close_down2_sma100(d, p):
    if len(d) < p.sma + 2 or not _clean(d):
        return None
    sm = _sma(d.c, p.sma, end=-1)
    if _ok(sm) and d.c[-1] <= (1 - p.dn) * d.c[-2] and d.c[-2] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_close_down4", "swing", "buy_close",
          "15:20: price <= 0.96 x previous close, no trend filter: buy 15:20, hold 5, 12 % stop",
          {"dn": 0.04, "hold": 5, "stop_pct": 12.0}, decision_time="15:20")
def r3_swing_close_down4(d, p):
    if len(d) < 2 or not _clean(d):
        return None
    if d.c[-1] <= (1 - p.dn) * d.c[-2]:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_gapdown_recover_1520", "swing", "gap",
          "15:20: gap down >= 3 % and price back at/above the open: buy 15:20, stop day low so far, hold 5",
          {"gap": 0.03, "hold": 5}, decision_time="15:20")
def r3_swing_gapdown_recover_1520(d, p):
    if len(d) < 2 or not _clean(d):
        return None
    if d.o[-1] <= (1 - p.gap) * d.c[-2] and d.c[-1] >= d.o[-1]:
        return Swing(LONG, stop=float(d.l[-1]), max_hold=p.hold)
    return None


# ── lead-lag, volume-conditioned reversal, squeeze, IBS, index reversal ─────
@strategy("r3_swing_leadlag_nifty", "swing", "lead_lag",
          "NIFTY 5-day return >= +3 %, stock 5-day return <= 0, close > SMA100: buy next open, hold 10, 8 % stop",
          {"n": 5, "idx_up": 0.03, "sma": 100, "hold": 10, "stop_pct": 8.0}, needs_index=True)
def r3_swing_leadlag_nifty(d, p):
    if len(d) < p.sma + 1 or not _nifty_ok(d, p.n + 1) or not _clean(d):
        return None
    rn, rs, sm = _ret(d.nifty, p.n), _ret(d.c, p.n), _sma(d.c, p.sma)
    if _ok(rn, rs, sm) and rn >= p.idx_up and rs <= 0 and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_quiet_loser_rev", "swing", "st_reversal_vol",
          "5-day return <= -8 % on volume <= the prior 50-day mean: buy next open, hold 5, 12 % stop",
          {"n": 5, "dn": -0.08, "base_n": 50, "vmax": 1.0, "hold": 5, "stop_pct": 12.0})
def r3_swing_quiet_loser_rev(d, p):
    if len(d) < p.base_n + p.n + 1 or not _clean(d):
        return None
    r = _ret(d.c, p.n)
    base = float(d.v[-p.base_n - p.n:-p.n].mean())
    recent = float(d.v[-p.n:].mean())
    if _ok(r) and base > 0 and r <= p.dn and recent <= p.vmax * base:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_squeeze_break", "swing", "vol_regime",
          "Bollinger width at its 120-session low within the last 5 sessions, close > upper band: stop 2 ATR, hold 10",
          {"bb": 20, "k": 2.0, "lb": 120, "recent": 5, "hold": 10, "atr_k": 2.0})
def r3_swing_squeeze_break(d, p):
    need = p.lb + p.bb
    if len(d) < need or not _clean(d):
        return None
    c = d.c[-need:]
    win = np.lib.stride_tricks.sliding_window_view(c, p.bb)       # rolling 20-close windows, causal
    m, sd = win.mean(axis=1), win.std(axis=1)
    width = (2 * p.k * sd) / m
    w = width[-p.lb:]
    a = _atr14(d)
    if not _ok(a):
        return None
    if w[-p.recent:].min() <= w.min() and c[-1] > m[-1] + p.k * sd[-1]:
        return Swing(LONG, stop=float(c[-1] - p.atr_k * a), max_hold=p.hold)
    return None


@strategy("r3_swing_ibs_low", "swing", "ibs",
          "IBS (c-l)/(h-l) < 0.15, down close, close > SMA100: buy next open, hold 3, 10 % stop",
          {"ibs": 0.15, "sma": 100, "hold": 3, "stop_pct": 10.0})
def r3_swing_ibs_low(d, p):
    if len(d) < p.sma + 1 or not _clean(d):
        return None
    rng = d.h[-1] - d.l[-1]
    sm = _sma(d.c, p.sma)
    if rng > 0 and _ok(sm) and (d.c[-1] - d.l[-1]) / rng < p.ibs and d.c[-1] < d.c[-2] and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r3_swing_nifty_rsi2", "swing", "index_reversal",
          "NIFTY RSI(2) < 10 with NIFTY > its SMA100, stock close > SMA100: buy next open, hold 5, 10 % stop",
          {"lvl": 10.0, "sma": 100, "hold": 5, "stop_pct": 10.0}, needs_index=True)
def r3_swing_nifty_rsi2(d, p):
    if len(d) < p.sma + 1 or not _nifty_ok(d, p.sma) or not _clean(d):
        return None
    nr = _rsi_last(d.nifty, 2)[-1]
    nsm, sm = _sma(d.nifty, p.sma), _sma(d.c, p.sma)
    if _ok(nr, nsm, sm) and nr < p.lvl and d.nifty[-1] > nsm and d.c[-1] > sm:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None
