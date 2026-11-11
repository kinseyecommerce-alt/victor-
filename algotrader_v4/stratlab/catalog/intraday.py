"""
Intraday strategies, batch 1: well-known NSE intraday setups at textbook parameter values.
Nothing here was tuned on results - do not start now. Each function gets the whole Session `s` and its
parameters `p`, and returns candidate entries; the simulator takes the first one(s) it can.
"""
from __future__ import annotations

import numpy as np

from ..api import LONG, SHORT, Entry, between, cross_dn, cross_up, first_true, hhmm, runlen, strategy


# ── opening-range breakouts ─────────────────────────────────────────────────
def orb_entries(s, minutes: int, rr: float, last: str, allow_long=True, allow_short=True, stop_at="far"):
    """First 1-minute close outside the opening range, before `last`. Stop at the far side or the midpoint."""
    hi, lo, k = s.orb(minutes)
    if k >= len(s) or not hi > lo:
        return []
    end = s.idx(hhmm(last))
    mid = (hi + lo) / 2.0
    out = []
    if allow_long:
        i = first_true(s.c > hi, k, end)
        if i >= 0:
            out.append(Entry(i, LONG, stop=lo if stop_at == "far" else mid, rr=rr))
    if allow_short:
        i = first_true(s.c < lo, k, end)
        if i >= 0:
            out.append(Entry(i, SHORT, stop=hi if stop_at == "far" else mid, rr=rr))
    return out


@strategy("orb5_break", "intraday", "orb", "Break of the first 5-minute range, stop at the other side, 1.5R",
          {"minutes": 5, "rr": 1.5, "last": "11:00"})
def orb5_break(s, p):
    return orb_entries(s, p.minutes, p.rr, p.last)


@strategy("orb15_break", "intraday", "orb", "Break of the first 15-minute range, stop at the other side, 1.5R",
          {"minutes": 15, "rr": 1.5, "last": "12:00"})
def orb15_break(s, p):
    return orb_entries(s, p.minutes, p.rr, p.last)


@strategy("orb30_break", "intraday", "orb", "Break of the first 30-minute range, stop at the other side, 1.5R",
          {"minutes": 30, "rr": 1.5, "last": "13:00"})
def orb30_break(s, p):
    return orb_entries(s, p.minutes, p.rr, p.last)


@strategy("orb15_trend20", "intraday", "orb",
          "15-minute range break only in the direction of the 20-day trend, stop at range midpoint, 2.5R",
          {"minutes": 15, "rr": 2.5, "last": "12:00"})
def orb15_trend20(s, p):
    if not np.isfinite(s.dsma20):
        return []
    up = s.pdc > s.dsma20
    return orb_entries(s, p.minutes, p.rr, p.last, allow_long=up, allow_short=not up, stop_at="mid")


@strategy("orb15_retest", "intraday", "orb",
          "15-minute range break, then a pullback that touches the broken level and closes back beyond it, 1.5R",
          {"minutes": 15, "rr": 1.5, "last": "13:00"})
def orb15_retest(s, p):
    hi, lo, k = s.orb(p.minutes)
    if k >= len(s) or not hi > lo:
        return []
    end = s.idx(hhmm(p.last))
    mid = (hi + lo) / 2.0
    out = []
    b = first_true(s.c > hi, k, end)
    if b >= 0:
        i = first_true((s.l <= hi) & (s.c > hi), b + 1, end)
        if i >= 0 and s.l[b:i + 1].min() > mid:
            out.append(Entry(i, LONG, stop=mid, rr=p.rr))
    b = first_true(s.c < lo, k, end)
    if b >= 0:
        i = first_true((s.h >= lo) & (s.c < lo), b + 1, end)
        if i >= 0 and s.h[b:i + 1].max() < mid:
            out.append(Entry(i, SHORT, stop=mid, rr=p.rr))
    return out


# ── session VWAP ────────────────────────────────────────────────────────────
@strategy("vwap_reclaim", "intraday", "vwap",
          "After 30+ minutes on one side of session VWAP, a close across it on 1.5x volume; stop at the 10-bar extreme, 2R",
          {"minutes_away": 30, "vol": 1.5, "rr": 2.0})
def vwap_reclaim(s, p):
    win = between(s, "09:45", "14:30")
    below = runlen(s.c < s.vwap)
    above = runlen(s.c > s.vwap)
    out = []
    sig = win & (s.c > s.vwap) & (s.vr >= p.vol)
    sig[1:] &= below[:-1] >= p.minutes_away
    sig[0] = False
    i = first_true(sig)
    if i >= 0:
        out.append(Entry(i, LONG, stop=float(s.l[max(0, i - 9):i + 1].min()), rr=p.rr))
    sig = win & (s.c < s.vwap) & (s.vr >= p.vol)
    sig[1:] &= above[:-1] >= p.minutes_away
    sig[0] = False
    i = first_true(sig)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=float(s.h[max(0, i - 9):i + 1].max()), rr=p.rr))
    return out


@strategy("vwap_2sd_fade", "intraday", "vwap",
          "Fade a close beyond session VWAP +/- 2 sigma when 1-minute RSI is stretched; target VWAP, stop 1 sigma further",
          {"sd": 2.0, "rsi_lo": 30, "rsi_hi": 70})
def vwap_2sd_fade(s, p):
    win = between(s, "10:00", "14:30") & (s.vsd > 0)
    out = []
    i = first_true(win & (s.c < s.vwap - p.sd * s.vsd) & (s.rsi < p.rsi_lo))
    if i >= 0:
        out.append(Entry(i, LONG, stop=float(s.c[i] - s.vsd[i]), target=float(s.vwap[i])))
    i = first_true(win & (s.c > s.vwap + p.sd * s.vsd) & (s.rsi > p.rsi_hi))
    if i >= 0:
        out.append(Entry(i, SHORT, stop=float(s.c[i] + s.vsd[i]), target=float(s.vwap[i])))
    return out


# ── gaps ────────────────────────────────────────────────────────────────────
@strategy("gap_and_go", "intraday", "gap",
          "Gap of 0.5-3 % that holds: trade a break of the first 5-minute range in the gap direction, 2R",
          {"min_gap": 0.5, "max_gap": 3.0, "rr": 2.0, "last": "10:30"})
def gap_and_go(s, p):
    g = s.gap * 100.0
    if not np.isfinite(g) or not p.min_gap <= abs(g) <= p.max_gap:
        return []
    return orb_entries(s, 5, p.rr, p.last, allow_long=g > 0, allow_short=g < 0)


@strategy("gap_fade", "intraday", "gap",
          "Gap of 0.5-2 % that fails: price closes back through the first 15-minute range; target yesterday's close",
          {"min_gap": 0.5, "max_gap": 2.0, "last": "11:00"})
def gap_fade(s, p):
    g = s.gap * 100.0
    if not np.isfinite(g) or not p.min_gap <= abs(g) <= p.max_gap:
        return []
    hi, lo, k = s.orb(15)
    if k >= len(s) or not hi > lo:
        return []
    end = s.idx(hhmm(p.last))
    if g > 0:
        i = first_true(s.c < lo, k, end)
        return [Entry(i, SHORT, stop=hi, target=s.pdc)] if i >= 0 else []
    i = first_true(s.c > hi, k, end)
    return [Entry(i, LONG, stop=lo, target=s.pdc)] if i >= 0 else []


# ── previous day's levels ───────────────────────────────────────────────────
@strategy("pdh_pdl_break", "intraday", "prev_day",
          "First close through yesterday's high (or low) on 1.5x volume, stop 0.5 % back inside, 2R",
          {"vol": 1.5, "stop_pct": 0.5, "rr": 2.0})
def pdh_pdl_break(s, p):
    if not np.isfinite(s.pdh):
        return []
    win = between(s, "09:30", "14:00")
    out = []
    if s.o[0] < s.pdh:
        i = first_true(win & cross_up(s.c, s.pdh) & (s.vr >= p.vol))
        if i >= 0:
            out.append(Entry(i, LONG, stop=s.pdh * (1 - p.stop_pct / 100.0), rr=p.rr))
    if s.o[0] > s.pdl:
        i = first_true(win & cross_dn(s.c, s.pdl) & (s.vr >= p.vol))
        if i >= 0:
            out.append(Entry(i, SHORT, stop=s.pdl * (1 + p.stop_pct / 100.0), rr=p.rr))
    return out


@strategy("pdh_pdl_reject", "intraday", "prev_day",
          "Failed break of yesterday's high (or low): a bar pokes through and closes back inside; stop past the day's extreme, 1.5R",
          {"rr": 1.5})
def pdh_pdl_reject(s, p):
    if not np.isfinite(s.pdh):
        return []
    win = between(s, "09:30", "14:00")
    out = []
    i = first_true(win & (s.h > s.pdh) & (s.c < s.pdh) & (s.c < s.o))
    if i >= 0:
        out.append(Entry(i, SHORT, stop=float(s.day_hi[i]) * 1.0005, rr=p.rr))
    i = first_true(win & (s.l < s.pdl) & (s.c > s.pdl) & (s.c > s.o))
    if i >= 0:
        out.append(Entry(i, LONG, stop=float(s.day_lo[i]) * 0.9995, rr=p.rr))
    return out


@strategy("nr7_breakout", "intraday", "prev_day",
          "After an NR7 day (narrowest range of 7), trade the first close beyond that day's high or low, 2R",
          {"lookback": 7, "rr": 2.0, "last": "13:00"})
def nr7_breakout(s, p):
    d = s.prev
    if len(d) < p.lookback:
        return []
    rng = d.h[-p.lookback:] - d.l[-p.lookback:]
    if not rng[-1] <= rng.min():
        return []
    mid = (s.pdh + s.pdl) / 2.0
    end = s.idx(hhmm(p.last))
    out = []
    i = first_true(s.c > s.pdh, 0, end)
    if i >= 0:
        out.append(Entry(i, LONG, stop=mid, rr=p.rr))
    i = first_true(s.c < s.pdl, 0, end)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=mid, rr=p.rr))
    return out


# ── trend continuation ──────────────────────────────────────────────────────
@strategy("ema21_first_pullback", "intraday", "trend",
          "5-minute EMAs stacked 9>21>50 and price above VWAP: buy the first pullback that touches EMA21 and closes back above (mirror for shorts), 2R",
          {"rr": 2.0})
def ema21_first_pullback(s, p):
    win = between(s, "10:00", "14:00")
    out = []
    up = win & (s.e9_5 > s.e21_5) & (s.e21_5 > s.e50_5) & (s.c > s.vwap)
    i = first_true(up & (s.l <= s.e21_5) & (s.c > s.e21_5) & (s.c > s.o))
    if i >= 0:
        out.append(Entry(i, LONG, stop=float(s.l[max(0, i - 4):i + 1].min()) * 0.9995, rr=p.rr))
    dn = win & (s.e9_5 < s.e21_5) & (s.e21_5 < s.e50_5) & (s.c < s.vwap)
    i = first_true(dn & (s.h >= s.e21_5) & (s.c < s.e21_5) & (s.c < s.o))
    if i >= 0:
        out.append(Entry(i, SHORT, stop=float(s.h[max(0, i - 4):i + 1].max()) * 1.0005, rr=p.rr))
    return out


@strategy("inside_bar_5m", "intraday", "pattern",
          "First 5-minute inside bar after 09:45: trade the break of the mother bar, stop at its other side, 1.5R",
          {"rr": 1.5, "last": "14:00"})
def inside_bar_5m(s, p):
    b = s.bars5()
    if len(b.c) < 3:
        return []
    inside = np.zeros(len(b.c), bool)
    inside[1:] = (b.h[1:] < b.h[:-1]) & (b.l[1:] > b.l[:-1])
    inside &= s.t[b.end] >= hhmm("09:45")
    j = first_true(inside)
    if j < 0:
        return []
    mh, ml = float(b.h[j - 1]), float(b.l[j - 1])
    start, end = int(b.end[j]) + 1, s.idx(hhmm(p.last))
    out = []
    i = first_true(s.c > mh, start, end)
    if i >= 0:
        out.append(Entry(i, LONG, stop=ml, rr=p.rr))
    i = first_true(s.c < ml, start, end)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=mh, rr=p.rr))
    return out


@strategy("rev_1000", "intraday", "time_of_day",
          "10:00 reversal: a 0.75 %+ move from the open by 10:00 that then closes through the 1-minute EMA21; stop past the day's extreme, 1.5R",
          {"move": 0.75, "rr": 1.5, "until": "10:45"})
def rev_1000(s, p):
    k = s.idx(hhmm("10:00"))
    if k >= len(s) or k < 1:
        return []
    move = (s.c[k - 1] / s.o[0] - 1.0) * 100.0
    end = s.idx(hhmm(p.until))
    if move >= p.move:
        i = first_true(s.c < s.ema21, k, end)
        return [Entry(i, SHORT, stop=float(s.day_hi[i]) * 1.0005, rr=p.rr)] if i >= 0 else []
    if move <= -p.move:
        i = first_true(s.c > s.ema21, k, end)
        return [Entry(i, LONG, stop=float(s.day_lo[i]) * 0.9995, rr=p.rr)] if i >= 0 else []
    return []


@strategy("afternoon_vwap_trend", "intraday", "time_of_day",
          "13:00-14:30: price held above session VWAP since 12:00 and makes a new day high (mirror below VWAP); stop at VWAP, 2R",
          {"rr": 2.0})
def afternoon_vwap_trend(s, p):
    a, b, e = s.idx(hhmm("12:00")), s.idx(hhmm("13:00")), s.idx(hhmm("14:30"))
    if b >= len(s) or b <= a or a < 1:
        return []
    if (s.c[a:b] > s.vwap[a:b]).all():
        i = first_true(s.c > s.day_hi[b - 1], b, e)
        if i >= 0 and (s.c[a:i + 1] > s.vwap[a:i + 1]).all():
            return [Entry(i, LONG, stop=float(s.vwap[i]), rr=p.rr)]
    elif (s.c[a:b] < s.vwap[a:b]).all():
        i = first_true(s.c < s.day_lo[b - 1], b, e)
        if i >= 0 and (s.c[a:i + 1] < s.vwap[a:i + 1]).all():
            return [Entry(i, SHORT, stop=float(s.vwap[i]), rr=p.rr)]
    return []


@strategy("close_momentum", "intraday", "time_of_day",
          "Closing-hour momentum: at 14:30 a stock up 1 %+ from its open and above VWAP is bought until square-off (mirror), stop at VWAP",
          {"move": 1.0, "at": "14:29"})
def close_momentum(s, p):
    i = s.idx(hhmm(p.at))
    if i >= len(s):
        return []
    move = (s.c[i] / s.o[0] - 1.0) * 100.0
    if move >= p.move and s.c[i] > s.vwap[i]:
        return [Entry(i, LONG, stop=float(s.vwap[i]))]
    if move <= -p.move and s.c[i] < s.vwap[i]:
        return [Entry(i, SHORT, stop=float(s.vwap[i]))]
    return []


@strategy("rs_vs_nifty", "intraday", "relative_strength",
          "At 10:30, a stock 1 %+ stronger than NIFTY since the open and above VWAP is bought (mirror for weak); stop at VWAP, 2R",
          {"spread": 1.0, "at": "10:29", "rr": 2.0}, needs_index=True)
def rs_vs_nifty(s, p):
    i = s.idx(hhmm(p.at))
    if s.nifty is None or i >= len(s) or not np.isfinite(s.nifty[0]) or not np.isfinite(s.nifty[i]):
        return []
    rs = ((s.c[i] / s.o[0]) - (s.nifty[i] / s.nifty[0])) * 100.0
    if rs >= p.spread and s.c[i] > s.vwap[i]:
        return [Entry(i, LONG, stop=float(s.vwap[i]), rr=p.rr)]
    if rs <= -p.spread and s.c[i] < s.vwap[i]:
        return [Entry(i, SHORT, stop=float(s.vwap[i]), rr=p.rr)]
    return []


@strategy("vol_spike_continuation", "intraday", "volume",
          "A 1-minute bar on 5x volume with twice the usual range closing in its top quarter above VWAP (mirror); stop at the bar's other end, 2R",
          {"vol": 5.0, "range_atr": 2.0, "rr": 2.0})
def vol_spike_continuation(s, p):
    win = between(s, "09:30", "14:30")
    rng = s.h - s.l
    big = win & (s.vr >= p.vol) & (rng >= p.range_atr * s.atr) & (rng > 0)
    out = []
    i = first_true(big & (s.c >= s.h - 0.25 * rng) & (s.c > s.vwap))
    if i >= 0:
        out.append(Entry(i, LONG, stop=float(s.l[i]), rr=p.rr))
    i = first_true(big & (s.c <= s.l + 0.25 * rng) & (s.c < s.vwap))
    if i >= 0:
        out.append(Entry(i, SHORT, stop=float(s.h[i]), rr=p.rr))
    return out
