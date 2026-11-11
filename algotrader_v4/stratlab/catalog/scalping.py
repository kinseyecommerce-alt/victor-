"""
Scalping strategies, batch 1: short holds (a time stop of 10-20 minutes), small targets, up to 3 trades per
symbol per day. A round trip on ₹1,00,000 costs roughly ₹80 in charges plus ~₹100 in slippage (about 0.18 %),
so a scalp has to clear that to show anything. Textbook parameter values; nothing was tuned on results.
"""
from __future__ import annotations

import numpy as np

from ..api import LONG, SHORT, Entry, between, cross_dn, cross_up, first_true, hhmm, prev_high, prev_low, strategy


def _each(mask, make):
    return [make(int(i)) for i in np.flatnonzero(mask)]


@strategy("scalp_orb5", "scalping", "orb",
          "Break of the first 5-minute range, stop at the range midpoint, 1R, out after 20 minutes",
          {"rr": 1.0, "hold": 20, "last": "10:30"}, max_per_day=1)
def scalp_orb5(s, p):
    hi, lo, k = s.orb(5)
    if k >= len(s) or not hi > lo:
        return []
    end, mid = s.idx(hhmm(p.last)), (hi + lo) / 2.0
    out = []
    i = first_true(s.c > hi, k, end)
    if i >= 0:
        out.append(Entry(i, LONG, stop=mid, rr=p.rr, time_stop=p.hold))
    i = first_true(s.c < lo, k, end)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=mid, rr=p.rr, time_stop=p.hold))
    return out


@strategy("scalp_orb15_retest", "scalping", "orb",
          "After a 15-minute range break, each pullback that touches the broken level and closes beyond it; stop at range midpoint, 1.5R, 20 minutes",
          {"rr": 1.5, "hold": 20, "last": "13:00"})
def scalp_orb15_retest(s, p):
    hi, lo, k = s.orb(15)
    if k >= len(s) or not hi > lo:
        return []
    end, mid = s.idx(hhmm(p.last)), (hi + lo) / 2.0
    idx = np.arange(len(s))
    out = []
    b = first_true(s.c > hi, k, end)
    if b >= 0:
        m = (idx > b) & (idx < end) & (s.l <= hi) & (s.c > hi)
        out += _each(m, lambda i: Entry(i, LONG, stop=mid, rr=p.rr, time_stop=p.hold))
    b = first_true(s.c < lo, k, end)
    if b >= 0:
        m = (idx > b) & (idx < end) & (s.h >= lo) & (s.c < lo)
        out += _each(m, lambda i: Entry(i, SHORT, stop=mid, rr=p.rr, time_stop=p.hold))
    return out


@strategy("scalp_vwap_bounce", "scalping", "vwap",
          "In a 5-minute uptrend (EMA9 > EMA21) a bar dips to session VWAP and closes back above it, bullish (mirror); stop past the bar, 1.5R, 15 minutes",
          {"rr": 1.5, "hold": 15})
def scalp_vwap_bounce(s, p):
    win = between(s, "09:45", "14:45")
    up = win & (s.e9_5 > s.e21_5) & (s.l <= s.vwap) & (s.c > s.vwap) & (s.c > s.o)
    dn = win & (s.e9_5 < s.e21_5) & (s.h >= s.vwap) & (s.c < s.vwap) & (s.c < s.o)
    return (_each(up, lambda i: Entry(i, LONG, stop=float(s.l[i]) * 0.9995, rr=p.rr, time_stop=p.hold))
            + _each(dn, lambda i: Entry(i, SHORT, stop=float(s.h[i]) * 1.0005, rr=p.rr, time_stop=p.hold)))


@strategy("scalp_ema9_21_cross", "scalping", "ema",
          "1-minute EMA9 crosses EMA21 on the same side of VWAP; stop at the 5-bar extreme, 1.5R, 15 minutes",
          {"rr": 1.5, "hold": 15})
def scalp_ema9_21_cross(s, p):
    win = between(s, "09:30", "14:45")
    up = win & cross_up(s.ema9, s.ema21) & (s.c > s.vwap)
    dn = win & cross_dn(s.ema9, s.ema21) & (s.c < s.vwap)
    return (_each(up, lambda i: Entry(i, LONG, stop=float(s.l[max(0, i - 4):i + 1].min()), rr=p.rr, time_stop=p.hold))
            + _each(dn, lambda i: Entry(i, SHORT, stop=float(s.h[max(0, i - 4):i + 1].max()), rr=p.rr,
                                        time_stop=p.hold)))


@strategy("scalp_rsi_pullback", "scalping", "mean_reversion",
          "1-minute RSI(14) drops under 25 while price is above the 5-minute EMA50 (mirror over 75); stop 1x 5-minute ATR, 1R, 10 minutes",
          {"lo": 25, "hi": 75, "rr": 1.0, "hold": 10})
def scalp_rsi_pullback(s, p):
    win = between(s, "09:45", "14:45") & (s.atr5 > 0)
    up = win & cross_dn(s.rsi, p.lo) & (s.c > s.e50_5)
    dn = win & cross_up(s.rsi, p.hi) & (s.c < s.e50_5)
    return (_each(up, lambda i: Entry(i, LONG, stop=float(s.c[i] - s.atr5[i]), rr=p.rr, time_stop=p.hold))
            + _each(dn, lambda i: Entry(i, SHORT, stop=float(s.c[i] + s.atr5[i]), rr=p.rr, time_stop=p.hold)))


@strategy("scalp_range20_break", "scalping", "breakout",
          "Close above the high of the previous 20 minutes on 2x volume (mirror); stop 1x 5-minute ATR, 1.5R, 15 minutes",
          {"n": 20, "vol": 2.0, "rr": 1.5, "hold": 15})
def scalp_range20_break(s, p):
    win = between(s, "09:45", "14:45") & (s.atr5 > 0)
    up = win & cross_up(s.c, prev_high(s.h, p.n)) & (s.vr >= p.vol)
    dn = win & cross_dn(s.c, prev_low(s.l, p.n)) & (s.vr >= p.vol)
    return (_each(up, lambda i: Entry(i, LONG, stop=float(s.c[i] - s.atr5[i]), rr=p.rr, time_stop=p.hold))
            + _each(dn, lambda i: Entry(i, SHORT, stop=float(s.c[i] + s.atr5[i]), rr=p.rr, time_stop=p.hold)))


@strategy("scalp_pivot_bounce", "scalping", "pivot",
          "Bounce off yesterday's floor pivot (H+L+C)/3: a bar touches it from above and closes bullish above it (mirror from below); stop 0.25 % past the pivot, 1.5R, 20 minutes",
          {"stop_pct": 0.25, "rr": 1.5, "hold": 20})
def scalp_pivot_bounce(s, p):
    if not np.isfinite(s.pdh):
        return []
    pv = (s.pdh + s.pdl + s.pdc) / 3.0
    win = between(s, "09:30", "14:45")
    prev_c = np.concatenate(([s.o[0]], s.c[:-1]))
    up = win & (prev_c > pv) & (s.l <= pv) & (s.c > pv) & (s.c > s.o)
    dn = win & (prev_c < pv) & (s.h >= pv) & (s.c < pv) & (s.c < s.o)
    return (_each(up, lambda i: Entry(i, LONG, stop=pv * (1 - p.stop_pct / 100.0), rr=p.rr, time_stop=p.hold))
            + _each(dn, lambda i: Entry(i, SHORT, stop=pv * (1 + p.stop_pct / 100.0), rr=p.rr, time_stop=p.hold)))


@strategy("scalp_climax_fade", "scalping", "mean_reversion",
          "Fade a volume climax: a 5x-volume, 3x-range bar that closes at a new day low is bought (new day high is sold); stop past the bar, 1R, 10 minutes",
          {"vol": 5.0, "range_atr": 3.0, "rr": 1.0, "hold": 10})
def scalp_climax_fade(s, p):
    win = between(s, "09:45", "14:45")
    rng = s.h - s.l
    big = win & (s.vr >= p.vol) & (rng >= p.range_atr * s.atr) & (rng > 0)
    up = big & (s.l <= s.day_lo) & (s.c < s.o)
    dn = big & (s.h >= s.day_hi) & (s.c > s.o)
    return (_each(up, lambda i: Entry(i, LONG, stop=float(s.l[i]) * 0.9995, rr=p.rr, time_stop=p.hold))
            + _each(dn, lambda i: Entry(i, SHORT, stop=float(s.h[i]) * 1.0005, rr=p.rr, time_stop=p.hold)))


@strategy("scalp_open_drive", "scalping", "time_of_day",
          "Open drive: the first three 1-minute bars all close the same way and beyond the open; join at 09:18, stop at the day's open, 1R, 15 minutes",
          {"rr": 1.0, "hold": 15}, max_per_day=1)
def scalp_open_drive(s, p):
    if len(s) < 3 or s.t[2] != hhmm("09:17"):
        return []
    c, o = s.c[:3], s.o[:3]
    if (c > o).all() and c[2] > c[1] > c[0]:
        return [Entry(2, LONG, stop=float(s.o[0]), rr=p.rr, time_stop=p.hold)]
    if (c < o).all() and c[2] < c[1] < c[0]:
        return [Entry(2, SHORT, stop=float(s.o[0]), rr=p.rr, time_stop=p.hold)]
    return []
