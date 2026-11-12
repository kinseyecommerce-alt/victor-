"""
Round-3 intraday (MIS stocks) candidates. Pre-registered BEFORE any result in
scratchpad/r3/intraday/PREREGISTERED.txt — parameters are textbook / round values; do not tune them.

Every function gets the whole Session `s` (vector style, audited for look-ahead) and returns candidate entries.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from .. import candles as _cd
from ..api import LONG, SHORT, Entry, first_true, hhmm, strategy
from ..indicators import ema as _ema, rsi as _rsi


# ── small helpers ───────────────────────────────────────────────────────────
def _at(s, hm: str) -> int:
    """Index of the decision bar for a time-triggered rule, or -1 if not visible yet."""
    i = s.idx(hhmm(hm))
    return i if i < len(s) else -1


def _ok(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def _rng(s, a: str, b: str):
    """[idx(a), idx(b)) clipped to the visible session."""
    return s.idx(hhmm(a)), s.idx(hhmm(b))


def _weekday(s) -> int:
    d = int(s.date)
    return _dt.date(d // 10000, (d // 100) % 100, d % 100).weekday()   # Monday 0


def _nret(s, i):
    """NIFTY return vs its previous close at bar i (None if unavailable)."""
    if s.nifty is None or not _ok(s.nifty_pdc) or not np.isfinite(s.nifty[i]):
        return None
    return float(s.nifty[i] / s.nifty_pdc - 1.0)


_DAYSETS: dict = {}


def _is_monthly_expiry(s) -> bool:
    """Stock F&O monthly expiry: last Thursday of the month (to Aug 2025), last Tuesday (from Sep 2025);
    if that date is not a session in the calendar, the last session before it."""
    d = int(s.date)
    y, m = d // 10000, (d // 100) % 100
    wd = 3 if d < 20250901 else 1
    nxt = _dt.date(y + (m == 12), m % 12 + 1, 1)
    e = nxt - _dt.timedelta(days=1)
    while e.weekday() != wd:
        e -= _dt.timedelta(days=1)
    sd = getattr(getattr(s, "_s", s), "_sd", None)
    if sd is None:
        return False
    key = id(sd)
    ent = _DAYSETS.get(key)
    if ent is None or ent[0] is not sd:
        if len(_DAYSETS) > 256:
            _DAYSETS.clear()
        ent = (sd, set(int(x) for x in sd.days))
        _DAYSETS[key] = ent
    days = ent[1]
    today = _dt.date(y, m, d % 100)
    if e < today:
        return False
    while e > today and int(e.strftime("%Y%m%d")) not in days:
        e -= _dt.timedelta(days=1)
    return e == today


def _tf_series(s, tf: int, hist: int = 100):
    """Completed tf-minute candles: closes of up to `hist` earlier-session candles + today's, and for today's
    candles their completion bar index `sig`. Returns (closes_all, P, sig) or None."""
    base = getattr(s, "_s", s)
    today = _cd.resample(s.t, s.o, s.h, s.l, s.c, s.v, tf)
    if len(today.c) == 0:
        return None
    sd, k = getattr(base, "_sd", None), getattr(base, "day_index", 0)
    if sd is not None and k > 0:
        ho, hh, hl, hc, hs, hf = _cd._symbol_candles(sd, tf)
        j = int(np.searchsorted(hs, k))
        pc = hc[max(0, j - hist):j]
    else:
        pc = np.zeros(0)
    return np.concatenate((pc, today.c)), len(pc), today.sig


def _hold_on_minutes(n: int, sig, vals) -> np.ndarray:
    """Value of the last completed candle at each 1-minute bar (NaN before the first)."""
    out = np.full(n, np.nan)
    pos = np.searchsorted(sig, np.arange(n), side="right") - 1
    ok = pos >= 0
    out[ok] = vals[pos[ok]]
    return out


# ── gap buckets ─────────────────────────────────────────────────────────────
@strategy("r3_intraday_gap_small_fade", "intraday", "gap",
          "0.3-1.0 % gap still unfilled at 09:29: fade it toward yesterday's close; stop day extreme + 0.1 ATR",
          {"lo": 0.3, "hi": 1.0, "at": "09:29", "buf": 0.1})
def gap_small_fade(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.gap, s.pdc, s.datr):
        return []
    g = abs(s.gap) * 100
    if not (p.lo <= g <= p.hi):
        return []
    if s.gap > 0 and s.day_lo[i] > s.pdc and s.c[i] > s.pdc:
        return [Entry(i, SHORT, stop=float(s.day_hi[i] + p.buf * s.datr), target=float(s.pdc))]
    if s.gap < 0 and s.day_hi[i] < s.pdc and s.c[i] < s.pdc:
        return [Entry(i, LONG, stop=float(s.day_lo[i] - p.buf * s.datr), target=float(s.pdc))]
    return []


@strategy("r3_intraday_gap_mid_fade_0920", "intraday", "gap",
          "1-3 % gap still beyond yesterday's close at 09:19: fade it all day, target yesterday's close, stop day extreme + 0.25 ATR",
          {"lo": 1.0, "hi": 3.0, "at": "09:19", "buf": 0.25})
def gap_mid_fade_0920(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.gap, s.pdc, s.datr):
        return []
    g = abs(s.gap) * 100
    if not (p.lo <= g <= p.hi):
        return []
    if s.gap > 0 and s.c[i] > s.pdc:
        return [Entry(i, SHORT, stop=float(s.day_hi[i] + p.buf * s.datr), target=float(s.pdc))]
    if s.gap < 0 and s.c[i] < s.pdc:
        return [Entry(i, LONG, stop=float(s.day_lo[i] - p.buf * s.datr), target=float(s.pdc))]
    return []


@strategy("r3_intraday_gap_large_follow", "intraday", "gap",
          "2-8 % gap that holds the first 30 minutes (close beyond the open and VWAP at 09:44): follow, stop the 30-minute extreme",
          {"lo": 2.0, "hi": 8.0, "at": "09:44"})
def gap_large_follow(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.gap):
        return []
    g = abs(s.gap) * 100
    if not (p.lo <= g <= p.hi):
        return []
    if s.gap > 0 and s.c[i] > s.o[0] and s.c[i] > s.vwap[i]:
        return [Entry(i, LONG, stop=float(s.day_lo[i]))]
    if s.gap < 0 and s.c[i] < s.o[0] and s.c[i] < s.vwap[i]:
        return [Entry(i, SHORT, stop=float(s.day_hi[i]))]
    return []


# ── VWAP band mean reversion with regime filters ────────────────────────────
def _band_fade(s, sd_in, sd_stop, a, b, extra_mask=None):
    lo_i, hi_i = _rng(s, a, b)
    ok = s.vsd > 0
    if extra_mask is not None:
        ok = ok & extra_mask
    up = s.vwap + sd_in * s.vsd
    dn = s.vwap - sd_in * s.vsd
    out = []
    i = first_true(ok & (s.c > up), lo_i, hi_i)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=float(s.vwap[i] + sd_stop * s.vsd[i]), target=float(s.vwap[i])))
    i = first_true(ok & (s.c < dn), lo_i, hi_i)
    if i >= 0:
        out.append(Entry(i, LONG, stop=float(s.vwap[i] - sd_stop * s.vsd[i]), target=float(s.vwap[i])))
    return out


@strategy("r3_intraday_vwap2sd_quietidx", "intraday", "vwap_mr",
          "10:30-14:30 fade a close beyond VWAP +/- 2 sigma while NIFTY is within 0.30 % of its 09:15 level; target VWAP, stop 3 sigma",
          {"sd": 2.0, "stop_sd": 3.0, "idx": 0.30}, needs_index=True)
def vwap2sd_quietidx(s, p):
    if s.nifty is None or not np.isfinite(s.nifty[0]):
        return []
    with np.errstate(invalid="ignore"):
        quiet = np.abs(s.nifty / s.nifty[0] - 1.0) * 100 < p.idx
    return _band_fade(s, p.sd, p.stop_sd, "10:30", "14:30", quiet)


@strategy("r3_intraday_vwap2sd_compress", "intraday", "vwap_mr",
          "Calm stock (yesterday's range < 0.75 ATR, |gap| < 0.5 %): 10:30-14:30 fade VWAP +/- 2 sigma, target VWAP, stop 3 sigma",
          {"sd": 2.0, "stop_sd": 3.0, "range_atr": 0.75, "gap": 0.5})
def vwap2sd_compress(s, p):
    if not _ok(s.pdh, s.pdl, s.datr, s.gap):
        return []
    if not ((s.pdh - s.pdl) < p.range_atr * s.datr and abs(s.gap) * 100 < p.gap):
        return []
    return _band_fade(s, p.sd, p.stop_sd, "10:30", "14:30")


@strategy("r3_intraday_vwap3sd_fade", "intraday", "vwap_mr",
          "10:00-14:30 fade a close beyond VWAP +/- 3 sigma; target VWAP, stop 4 sigma",
          {"sd": 3.0, "stop_sd": 4.0})
def vwap3sd_fade(s, p):
    return _band_fade(s, p.sd, p.stop_sd, "10:00", "14:30")


# ── relative strength vs NIFTY / dispersion ─────────────────────────────────
@strategy("r3_intraday_rs_rev_1100", "intraday", "relative_strength",
          "At 10:59 a stock 2.5 %+ weaker than NIFTY since yesterday's close is bought (stronger is sold); stop 0.75 daily ATR",
          {"rel": 2.5, "at": "10:59", "stop_atr": 0.75}, needs_index=True)
def rs_rev_1100(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.pdc, s.datr):
        return []
    n = _nret(s, i)
    if n is None:
        return []
    rel = (s.c[i] / s.pdc - 1.0 - n) * 100
    if rel <= -p.rel:
        return [Entry(i, LONG, stop=float(s.c[i] - p.stop_atr * s.datr))]
    if rel >= p.rel:
        return [Entry(i, SHORT, stop=float(s.c[i] + p.stop_atr * s.datr))]
    return []


@strategy("r3_intraday_rs_cont_1300", "intraday", "relative_strength",
          "At 12:59 a stock 1.5 %+ stronger than NIFTY since yesterday's close, above VWAP and in the top quarter of its day range: buy to 15:15 (mirror); stop VWAP",
          {"rel": 1.5, "at": "12:59", "q": 0.25}, needs_index=True)
def rs_cont_1300(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.pdc):
        return []
    n = _nret(s, i)
    if n is None:
        return []
    rel = (s.c[i] / s.pdc - 1.0 - n) * 100
    rg = s.day_hi[i] - s.day_lo[i]
    if rg <= 0:
        return []
    if rel >= p.rel and s.c[i] > s.vwap[i] and s.c[i] >= s.day_hi[i] - p.q * rg:
        return [Entry(i, LONG, stop=float(s.vwap[i]))]
    if rel <= -p.rel and s.c[i] < s.vwap[i] and s.c[i] <= s.day_lo[i] + p.q * rg:
        return [Entry(i, SHORT, stop=float(s.vwap[i]))]
    return []


@strategy("r3_intraday_idio_cont_1100", "intraday", "dispersion",
          "At 10:59 NIFTY within 0.25 % of its previous close but the stock 2 %+ up and above VWAP: buy to 15:15 (mirror); stop VWAP",
          {"idx": 0.25, "move": 2.0, "at": "10:59"}, needs_index=True)
def idio_cont_1100(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.pdc):
        return []
    n = _nret(s, i)
    if n is None or abs(n) * 100 > p.idx:
        return []
    r = (s.c[i] / s.pdc - 1.0) * 100
    if r >= p.move and s.c[i] > s.vwap[i]:
        return [Entry(i, LONG, stop=float(s.vwap[i]))]
    if r <= -p.move and s.c[i] < s.vwap[i]:
        return [Entry(i, SHORT, stop=float(s.vwap[i]))]
    return []


# ── afternoon ───────────────────────────────────────────────────────────────
@strategy("r3_intraday_pm_fade_1330", "intraday", "afternoon",
          "At 13:29 a stock 2 %+ away from its open is faded toward VWAP; stop day extreme + 0.1 ATR, out at 15:00",
          {"move": 2.0, "at": "13:29", "buf": 0.1, "exit": "15:00"})
def pm_fade_1330(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.datr):
        return []
    r = (s.c[i] / s.o[0] - 1.0) * 100
    fill_t = int(s.t[i]) + 1
    ts = max(1, hhmm(p.exit) - fill_t)
    if r >= p.move and s.vwap[i] < s.c[i]:
        return [Entry(i, SHORT, stop=float(s.day_hi[i] + p.buf * s.datr), target=float(s.vwap[i]), time_stop=ts)]
    if r <= -p.move and s.vwap[i] > s.c[i]:
        return [Entry(i, LONG, stop=float(s.day_lo[i] - p.buf * s.datr), target=float(s.vwap[i]), time_stop=ts)]
    return []


@strategy("r3_intraday_fh_lh_mom", "intraday", "afternoon",
          "First half-hour return (09:44 close vs previous close) of 1 %+ predicts the last half hour: enter at 14:44 in its sign, stop 1 %, out 15:15",
          {"r30": 1.0, "first": "09:44", "at": "14:44", "stop_pct": 1.0})
def fh_lh_mom(s, p):
    k = _at(s, p.first)
    i = _at(s, p.at)
    if k < 0 or i < 0 or not _ok(s.pdc):
        return []
    r = (s.c[k] / s.pdc - 1.0) * 100
    c = float(s.c[i])
    if r >= p.r30:
        return [Entry(i, LONG, stop=c * (1 - p.stop_pct / 100))]
    if r <= -p.r30:
        return [Entry(i, SHORT, stop=c * (1 + p.stop_pct / 100))]
    return []


@strategy("r3_intraday_mis_squareoff_fade", "intraday", "afternoon",
          "At 14:44 a stock 2.5 %+ away from its open is faded into the MIS square-off; stop day extreme + 0.1 ATR, out 15:15",
          {"move": 2.5, "at": "14:44", "buf": 0.1})
def mis_squareoff_fade(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.datr):
        return []
    r = (s.c[i] / s.o[0] - 1.0) * 100
    if r >= p.move:
        return [Entry(i, SHORT, stop=float(s.day_hi[i] + p.buf * s.datr))]
    if r <= -p.move:
        return [Entry(i, LONG, stop=float(s.day_lo[i] - p.buf * s.datr))]
    return []


# ── lunch-time range ────────────────────────────────────────────────────────
def _lunch(s):
    a, b = _rng(s, "12:00", "13:00")
    if b >= len(s) or b - a < 30:
        return None
    return float(s.h[a:b].max()), float(s.l[a:b].min()), b


@strategy("r3_intraday_lunch_break", "intraday", "lunch_range",
          "13:00-14:30 first close beyond the 12:00-13:00 range in the day's direction (vs the open); stop the other side, 2R",
          {"rr": 2.0, "last": "14:30"})
def lunch_break(s, p):
    L = _lunch(s)
    if L is None:
        return []
    hi, lo, b = L
    if not hi > lo:
        return []
    end = s.idx(hhmm(p.last))
    out = []
    i = first_true((s.c > hi) & (s.c > s.o[0]), b, end)
    if i >= 0:
        out.append(Entry(i, LONG, stop=lo, rr=p.rr))
    i = first_true((s.c < lo) & (s.c < s.o[0]), b, end)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=hi, rr=p.rr))
    return out


@strategy("r3_intraday_lunch_break_vol", "intraday", "lunch_range",
          "13:00-14:30 first close beyond the 12:00-13:00 range on 2x volume, either way; stop range midpoint, 2R",
          {"vr": 2.0, "rr": 2.0, "last": "14:30"})
def lunch_break_vol(s, p):
    L = _lunch(s)
    if L is None:
        return []
    hi, lo, b = L
    if not hi > lo:
        return []
    mid = (hi + lo) / 2
    end = s.idx(hhmm(p.last))
    out = []
    i = first_true((s.c > hi) & (s.vr >= p.vr), b, end)
    if i >= 0:
        out.append(Entry(i, LONG, stop=mid, rr=p.rr))
    i = first_true((s.c < lo) & (s.vr >= p.vr), b, end)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=mid, rr=p.rr))
    return out


# ── volume shocks ───────────────────────────────────────────────────────────
@strategy("r3_intraday_volshock_rev", "intraday", "volume_shock",
          "10:00-14:00 an 8x-volume, 3-ATR bar at a new day extreme, then within 5 bars a close back past its midpoint: fade to VWAP, stop beyond the bar + 1 ATR",
          {"vr": 8.0, "range_atr": 3.0, "confirm": 5})
def volshock_rev(s, p):
    a, b = _rng(s, "10:00", "14:00")
    rng = s.h - s.l
    with np.errstate(invalid="ignore"):
        shock = (s.vr >= p.vr) & (rng >= p.range_atr * s.atr) & (rng > 0)
    out = []
    n = len(s)
    for side in (SHORT, LONG):
        ext = (s.h >= s.day_hi) if side == SHORT else (s.l <= s.day_lo)
        cand = np.flatnonzero(shock & ext)
        cand = cand[(cand >= a) & (cand < b)]
        best = None
        for j in cand:
            mid = (s.h[j] + s.l[j]) / 2
            stop_ = (s.c > mid) if side == LONG else (s.c < mid)
            k = first_true(stop_, j + 1, min(n, j + 1 + p.confirm))
            if k < 0:
                continue
            tgt = float(s.vwap[k])
            if side == SHORT and tgt < s.c[k]:
                best = Entry(k, SHORT, stop=float(s.h[j] + s.atr[j]), target=tgt)
            elif side == LONG and tgt > s.c[k]:
                best = Entry(k, LONG, stop=float(s.l[j] - s.atr[j]), target=tgt)
            if best is not None:
                break
        if best is not None:
            out.append(best)
    return out


@strategy("r3_intraday_dvolshock_follow", "intraday", "volume_shock",
          "Yesterday 3x the 20-day volume and a 3 %+ move: at 09:29 if price is still beyond yesterday's close that way, follow; stop the 09:15-09:29 extreme",
          {"vol": 3.0, "ret": 3.0, "at": "09:29"})
def dvolshock_follow(s, p):
    i = _at(s, p.at)
    pv, pc = s.prev.v, s.prev.c
    if i < 0 or len(pc) < 22 or not _ok(s.pdc):
        return []
    base = float(np.mean(pv[-21:-1]))
    if not base > 0 or pv[-1] < p.vol * base:
        return []
    r = (pc[-1] / pc[-2] - 1.0) * 100
    if r >= p.ret and s.c[i] > s.pdc:
        return [Entry(i, LONG, stop=float(s.day_lo[i]))]
    if r <= -p.ret and s.c[i] < s.pdc:
        return [Entry(i, SHORT, stop=float(s.day_hi[i]))]
    return []


# ── previous-day levels ─────────────────────────────────────────────────────
@strategy("r3_intraday_gapthru_reclaim", "intraday", "prev_day",
          "Opened below yesterday's low, 09:30-12:00 first close back above it: buy, stop the day low, 2R (mirror at yesterday's high)",
          {"rr": 2.0})
def gapthru_reclaim(s, p):
    if not _ok(s.pdh, s.pdl):
        return []
    a, b = _rng(s, "09:30", "12:00")
    out = []
    if s.o[0] < s.pdl:
        i = first_true(s.c > s.pdl, a, b)
        if i >= 0:
            out.append(Entry(i, LONG, stop=float(s.day_lo[i]), rr=p.rr))
    if s.o[0] > s.pdh:
        i = first_true(s.c < s.pdh, a, b)
        if i >= 0:
            out.append(Entry(i, SHORT, stop=float(s.day_hi[i]), rr=p.rr))
    return out


@strategy("r3_intraday_gapthru_hold", "intraday", "prev_day",
          "Opened above yesterday's high, 09:30-12:00 first bar that dips to it and closes above: buy, stop 0.25 ATR below it, 2R (mirror)",
          {"rr": 2.0, "buf": 0.25})
def gapthru_hold(s, p):
    if not _ok(s.pdh, s.pdl, s.datr):
        return []
    a, b = _rng(s, "09:30", "12:00")
    out = []
    if s.o[0] > s.pdh:
        i = first_true((s.l <= s.pdh) & (s.c > s.pdh), a, b)
        if i >= 0:
            out.append(Entry(i, LONG, stop=float(s.pdh - p.buf * s.datr), rr=p.rr))
    if s.o[0] < s.pdl:
        i = first_true((s.h >= s.pdl) & (s.c < s.pdl), a, b)
        if i >= 0:
            out.append(Entry(i, SHORT, stop=float(s.pdl + p.buf * s.datr), rr=p.rr))
    return out


# ── range contraction (daily) ───────────────────────────────────────────────
def _yday_break(s, hi, lo, a, b, stop_mid):
    out = []
    i = first_true(s.c > hi, a, b)
    if i >= 0:
        out.append(Entry(i, LONG, stop=stop_mid))
    i = first_true(s.c < lo, a, b)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=stop_mid))
    return out


@strategy("r3_intraday_insideday_break", "intraday", "range_contraction",
          "Yesterday an inside day: 09:30-13:00 first close beyond the mother day's high/low, stop the middle of yesterday's range, out 15:15",
          {"a": "09:30", "b": "13:00"})
def insideday_break(s, p):
    H, L = s.prev.h, s.prev.l
    if len(H) < 2:
        return []
    if not (H[-1] < H[-2] and L[-1] > L[-2]):
        return []
    a, b = _rng(s, p.a, p.b)
    return _yday_break(s, float(H[-2]), float(L[-2]), a, b, float((H[-1] + L[-1]) / 2))


@strategy("r3_intraday_nr4id_break", "intraday", "range_contraction",
          "Yesterday NR4 and inside day (Crabel): 09:15-13:00 first close beyond yesterday's high/low, stop the middle of yesterday's range, out 15:15",
          {"nr": 4, "b": "13:00"})
def nr4id_break(s, p):
    H, L = s.prev.h, s.prev.l
    if len(H) < p.nr:
        return []
    r = H[-p.nr:] - L[-p.nr:]
    if not (r[-1] < r[:-1].min() and H[-1] < H[-2] and L[-1] > L[-2]):
        return []
    b = s.idx(hhmm(p.b))
    return _yday_break(s, float(H[-1]), float(L[-1]), 0, b, float((H[-1] + L[-1]) / 2))


# ── calendar: expiry and day of week ────────────────────────────────────────
@strategy("r3_intraday_expiry_fade", "intraday", "calendar_expiry",
          "Monthly stock-F&O expiry day: at 13:29 a stock 1.5 %+ from yesterday's close is faded toward VWAP; stop day extreme + 0.1 ATR",
          {"move": 1.5, "at": "13:29", "buf": 0.1})
def expiry_fade(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.pdc, s.datr) or not _is_monthly_expiry(s):
        return []
    r = (s.c[i] / s.pdc - 1.0) * 100
    if r >= p.move and s.vwap[i] < s.c[i]:
        return [Entry(i, SHORT, stop=float(s.day_hi[i] + p.buf * s.datr), target=float(s.vwap[i]))]
    if r <= -p.move and s.vwap[i] > s.c[i]:
        return [Entry(i, LONG, stop=float(s.day_lo[i] - p.buf * s.datr), target=float(s.vwap[i]))]
    return []


@strategy("r3_intraday_monday_gapdown_short", "intraday", "calendar_dow",
          "Mondays: gap down 0.5 %+ and at 09:44 below VWAP and the open: short to 15:15, stop the 09:15-09:44 high",
          {"gap": -0.5, "at": "09:44"})
def monday_gapdown_short(s, p):
    if _weekday(s) != 0:
        return []
    i = _at(s, p.at)
    if i < 0 or not _ok(s.gap) or s.gap * 100 > p.gap:
        return []
    if s.c[i] < s.vwap[i] and s.c[i] < s.o[0]:
        return [Entry(i, SHORT, stop=float(s.day_hi[i]))]
    return []


@strategy("r3_intraday_friday_pm_fade", "intraday", "calendar_dow",
          "Fridays: at 13:59 a stock 2 %+ up from yesterday's close is sold (2 %+ down is bought) into the weekend; stop day extreme + 0.1 ATR",
          {"move": 2.0, "at": "13:59", "buf": 0.1})
def friday_pm_fade(s, p):
    if _weekday(s) != 4:
        return []
    i = _at(s, p.at)
    if i < 0 or not _ok(s.pdc, s.datr):
        return []
    r = (s.c[i] / s.pdc - 1.0) * 100
    if r >= p.move:
        return [Entry(i, SHORT, stop=float(s.day_hi[i] + p.buf * s.datr))]
    if r <= -p.move:
        return [Entry(i, LONG, stop=float(s.day_lo[i] - p.buf * s.datr))]
    return []


# ── index lead-lag ──────────────────────────────────────────────────────────
def _leadlag(s, n, idx_move, lag_move, stop_atr, hold):
    if s.nifty is None or not _ok(s.datr):
        return []
    a, b = _rng(s, "10:00", "14:30")
    L = len(s)
    if L <= n:
        return []
    nr = np.full(L, np.nan)
    sr = np.full(L, np.nan)
    nr[n:] = (s.nifty[n:] / s.nifty[:-n] - 1.0) * 100
    sr[n:] = (s.c[n:] / s.c[:-n] - 1.0) * 100
    out = []
    with np.errstate(invalid="ignore"):
        i = first_true((nr >= idx_move) & (sr <= lag_move), a, b)
        if i >= 0:
            out.append(Entry(i, LONG, stop=float(s.c[i] - stop_atr * s.datr), time_stop=hold))
        i = first_true((nr <= -idx_move) & (sr >= -lag_move), a, b)
        if i >= 0:
            out.append(Entry(i, SHORT, stop=float(s.c[i] + stop_atr * s.datr), time_stop=hold))
    return out


@strategy("r3_intraday_leadlag_15m", "intraday", "lead_lag",
          "10:00-14:30 NIFTY up 0.30 %+ over 15 minutes while the stock is up <= 0.05 %: buy the laggard (mirror); stop 0.5 ATR, out after 60 minutes",
          {"n": 15, "idx": 0.30, "lag": 0.05, "stop_atr": 0.5, "hold": 60}, needs_index=True)
def leadlag_15m(s, p):
    return _leadlag(s, p.n, p.idx, p.lag, p.stop_atr, p.hold)


@strategy("r3_intraday_leadlag_5m", "intraday", "lead_lag",
          "10:00-14:30 NIFTY up 0.20 %+ over 5 minutes while the stock is flat or down: buy the laggard (mirror); stop 0.4 ATR, out after 30 minutes",
          {"n": 5, "idx": 0.20, "lag": 0.0, "stop_atr": 0.4, "hold": 30}, needs_index=True)
def leadlag_5m(s, p):
    return _leadlag(s, p.n, p.idx, p.lag, p.stop_atr, p.hold)


# ── index regime, short side ────────────────────────────────────────────────
@strategy("r3_intraday_downday_pdl_short", "intraday", "index_regime",
          "Short only, NIFTY 0.5 %+ below its previous close: 10:00-14:00 first close under yesterday's low on 1.5x volume; stop 0.3 ATR above it, 2R",
          {"idx": -0.5, "vr": 1.5, "buf": 0.3, "rr": 2.0}, needs_index=True)
def downday_pdl_short(s, p):
    if s.nifty is None or not _ok(s.nifty_pdc, s.pdl, s.datr):
        return []
    a, b = _rng(s, "10:00", "14:00")
    with np.errstate(invalid="ignore"):
        down = (s.nifty / s.nifty_pdc - 1.0) * 100 <= p.idx
        i = first_true(down & (s.c < s.pdl) & (s.vr >= p.vr), a, b)
    if i >= 0:
        return [Entry(i, SHORT, stop=float(s.pdl + p.buf * s.datr), rr=p.rr)]
    return []


@strategy("r3_intraday_downday_strong_fade", "intraday", "index_regime",
          "Short only: at 11:59 NIFTY 0.75 %+ below its previous close and the stock 1 %+ above its own: short to 15:15, stop day high + 0.1 ATR",
          {"idx": -0.75, "move": 1.0, "at": "11:59", "buf": 0.1}, needs_index=True)
def downday_strong_fade(s, p):
    i = _at(s, p.at)
    if i < 0 or not _ok(s.pdc, s.datr):
        return []
    n = _nret(s, i)
    if n is None or n * 100 > p.idx:
        return []
    if (s.c[i] / s.pdc - 1.0) * 100 >= p.move:
        return [Entry(i, SHORT, stop=float(s.day_hi[i] + p.buf * s.datr))]
    return []


# ── opening-drive exhaustion ────────────────────────────────────────────────
@strategy("r3_intraday_opendrive_exhaust", "intraday", "open_exhaustion",
          "09:15-09:44 drive of 0.75 daily ATR+, no new extreme 09:45-10:14 and 10:14 close back inside: fade to VWAP, stop day extreme + 0.1 ATR",
          {"atr_k": 0.75, "drive": "09:44", "at": "10:14", "buf": 0.1})
def opendrive_exhaust(s, p):
    k = _at(s, p.drive)
    i = _at(s, p.at)
    if k < 0 or i < 0 or not _ok(s.datr):
        return []
    d = s.c[k] - s.o[0]
    if d >= p.atr_k * s.datr and s.day_hi[i] <= s.day_hi[k] and s.c[i] < s.c[k] and s.vwap[i] < s.c[i]:
        return [Entry(i, SHORT, stop=float(s.day_hi[i] + p.buf * s.datr), target=float(s.vwap[i]))]
    if d <= -p.atr_k * s.datr and s.day_lo[i] >= s.day_lo[k] and s.c[i] > s.c[k] and s.vwap[i] > s.c[i]:
        return [Entry(i, LONG, stop=float(s.day_lo[i] - p.buf * s.datr), target=float(s.vwap[i]))]
    return []


@strategy("r3_intraday_opendrive_exhaust_rsi5", "intraday", "open_exhaustion",
          "Up 1 %+ by 09:44 with 5m RSI >= 80: 09:45-10:30 first bar with 5m RSI < 70 is sold, stop day high + 0.1 ATR, 1.5R (mirror 20/30)",
          {"move": 1.0, "hi": 80, "hi_x": 70, "lo": 20, "lo_x": 30, "rr": 1.5, "buf": 0.1})
def opendrive_exhaust_rsi5(s, p):
    k = _at(s, "09:44")
    if k < 0 or not _ok(s.datr) or not np.isfinite(s.rsi5[k]):
        return []
    a, b = k + 1, s.idx(hhmm("10:30"))
    r = (s.c[k] / s.o[0] - 1.0) * 100
    with np.errstate(invalid="ignore"):
        if r >= p.move and s.rsi5[k] >= p.hi:
            i = first_true(s.rsi5 < p.hi_x, a, b)
            if i >= 0:
                return [Entry(i, SHORT, stop=float(s.day_hi[i] + p.buf * s.datr), rr=p.rr)]
        if r <= -p.move and s.rsi5[k] <= p.lo:
            i = first_true(s.rsi5 > p.lo_x, a, b)
            if i >= 0:
                return [Entry(i, LONG, stop=float(s.day_lo[i] - p.buf * s.datr), rr=p.rr)]
    return []


# ── multi-timeframe trend pullbacks ─────────────────────────────────────────
@strategy("r3_intraday_mtf_15m5m_pb", "intraday", "mtf_trend",
          "15m close above a rising 15m EMA20 and 5m EMA9 > EMA21: 10:00-14:00 first 1m dip to the 5m EMA21 closing above it, stop 0.5 x 5m ATR below, 2R (mirror)",
          {"ema": 20, "slope": 4, "atr_k": 0.5, "rr": 2.0})
def mtf_15m5m_pb(s, p):
    ser = _tf_series(s, 15)
    if ser is None:
        return []
    c15, P, sig = ser
    e = _ema(c15, p.ema)
    e_prev = np.full(len(e), np.nan)
    if len(e) > p.slope:
        e_prev[p.slope:] = e[:-p.slope]
    n = len(s)
    up15 = _hold_on_minutes(n, sig, ((c15 > e) & (e > e_prev))[P:].astype(float)) == 1.0
    dn15 = _hold_on_minutes(n, sig, ((c15 < e) & (e < e_prev))[P:].astype(float)) == 1.0
    a, b = _rng(s, "10:00", "14:00")
    out = []
    with np.errstate(invalid="ignore"):
        i = first_true(up15 & (s.e9_5 > s.e21_5) & (s.l <= s.e21_5) & (s.c > s.e21_5) & (s.atr5 > 0), a, b)
        if i >= 0:
            out.append(Entry(i, LONG, stop=float(min(s.l[i], s.e21_5[i]) - p.atr_k * s.atr5[i]), rr=p.rr))
        i = first_true(dn15 & (s.e9_5 < s.e21_5) & (s.h >= s.e21_5) & (s.c < s.e21_5) & (s.atr5 > 0), a, b)
        if i >= 0:
            out.append(Entry(i, SHORT, stop=float(max(s.h[i], s.e21_5[i]) + p.atr_k * s.atr5[i]), rr=p.rr))
    return out


@strategy("r3_intraday_mtf_daily_15mrsi", "intraday", "mtf_trend",
          "Daily uptrend (above 20d SMA and 20-day momentum > 0): 10:00-14:00 completed 15m RSI(14) crosses back above 40, stop the 30-bar low, 2R (mirror 60)",
          {"rsi": 14, "lo": 40, "hi": 60, "n": 20, "lowbars": 30, "rr": 2.0})
def mtf_daily_15mrsi(s, p):
    pc = s.prev.c
    if len(pc) < p.n + 1 or not _ok(s.dsma20, s.pdc):
        return []
    up = s.pdc > s.dsma20 and pc[-1] > pc[-1 - p.n]
    dn = s.pdc < s.dsma20 and pc[-1] < pc[-1 - p.n]
    if not (up or dn):
        return []
    ser = _tf_series(s, 15)
    if ser is None:
        return []
    c15, P, sig = ser
    r = _rsi(c15, p.rsi)
    rp = np.full(len(r), np.nan)
    rp[1:] = r[:-1]
    with np.errstate(invalid="ignore"):
        x_up = ((rp < p.lo) & (r >= p.lo))[P:]
        x_dn = ((rp > p.hi) & (r <= p.hi))[P:]
    a, b = _rng(s, "10:00", "14:00")
    sel = (sig >= a) & (sig < b)
    if up:
        idxs = sig[sel & x_up]
        if len(idxs):
            i = int(idxs[0])
            return [Entry(i, LONG, stop=float(s.l[max(0, i - p.lowbars + 1):i + 1].min()), rr=p.rr)]
    if dn:
        idxs = sig[sel & x_dn]
        if len(idxs):
            i = int(idxs[0])
            return [Entry(i, SHORT, stop=float(s.h[max(0, i - p.lowbars + 1):i + 1].max()), rr=p.rr)]
    return []


# ── volatility (ATR-scaled) breakouts ───────────────────────────────────────
def _open_band_break(s, k, last):
    end = s.idx(hhmm(last))
    o0 = float(s.o[0])
    out = []
    i = first_true(s.c > o0 + k, 0, end)
    if i >= 0:
        out.append(Entry(i, LONG, stop=o0))
    i = first_true(s.c < o0 - k, 0, end)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=o0))
    return out


@strategy("r3_intraday_williams_vbo", "intraday", "vol_breakout",
          "Larry Williams volatility breakout: first close beyond the open +/- 0.5 x yesterday's range (until 14:00), stop at the open, out 15:15",
          {"k": 0.5, "last": "14:00"})
def williams_vbo(s, p):
    if not _ok(s.pdh, s.pdl) or not s.pdh > s.pdl:
        return []
    return _open_band_break(s, p.k * (s.pdh - s.pdl), p.last)


@strategy("r3_intraday_crabel_stretch", "intraday", "vol_breakout",
          "Crabel stretch: first close beyond the open +/- the 10-day mean of min(high-open, open-low) (until 14:00), stop at the open, out 15:15",
          {"n": 10, "last": "14:00"})
def crabel_stretch(s, p):
    d = s.prev
    if len(d.c) < p.n:
        return []
    st = float(np.mean(np.minimum(d.h[-p.n:] - d.o[-p.n:], d.o[-p.n:] - d.l[-p.n:])))
    if not st > 0:
        return []
    return _open_band_break(s, st, p.last)


# ── anchoring ───────────────────────────────────────────────────────────────
@strategy("r3_intraday_6mhigh_cross", "intraday", "anchoring",
          "09:30-14:00 first close above the 126-session high (yesterday's close below it): buy, stop 0.5 ATR under the level, out 15:15 (mirror at the low)",
          {"n": 126, "buf": 0.5})
def sixm_high_cross(s, p):
    d = s.prev
    if len(d.c) < p.n or not _ok(s.datr, s.pdc):
        return []
    H = float(d.h[-p.n:].max())
    L = float(d.l[-p.n:].min())
    a, b = _rng(s, "09:30", "14:00")
    out = []
    if s.pdc < H:
        i = first_true(s.c > H, a, b)
        if i >= 0:
            out.append(Entry(i, LONG, stop=H - p.buf * s.datr))
    if s.pdc > L:
        i = first_true(s.c < L, a, b)
        if i >= 0:
            out.append(Entry(i, SHORT, stop=L + p.buf * s.datr))
    return out


# ── short-term reversal ─────────────────────────────────────────────────────
@strategy("r3_intraday_3day_reversal", "intraday", "st_reversal",
          "Three lower closes totalling -5 %+: at 09:29 if price is above the open, buy, stop the day low (mirror for +5 % streaks)",
          {"days": 3, "move": 5.0, "at": "09:29"})
def three_day_reversal(s, p):
    pc = s.prev.c
    i = _at(s, p.at)
    if i < 0 or len(pc) < p.days + 1:
        return []
    w = pc[-(p.days + 1):]
    dif = np.diff(w)
    tot = (w[-1] / w[0] - 1.0) * 100
    if np.all(dif < 0) and tot <= -p.move and s.c[i] > s.o[0]:
        return [Entry(i, LONG, stop=float(s.day_lo[i]))]
    if np.all(dif > 0) and tot >= p.move and s.c[i] < s.o[0]:
        return [Entry(i, SHORT, stop=float(s.day_hi[i]))]
    return []


# ── intraday periodicity ────────────────────────────────────────────────────
@strategy("r3_intraday_same_slot", "intraday", "intraday_seasonality",
          "Yesterday's 14:44->15:14 return of 0.4 %+ repeats: at 14:44 enter in its sign, stop 1 %, out 15:15",
          {"r": 0.4, "at": "14:44", "end": "15:14", "stop_pct": 1.0})
def same_slot(s, p):
    i = _at(s, p.at)
    base = getattr(s, "_s", s)
    sd, k = getattr(base, "_sd", None), getattr(base, "day_index", 0)
    if i < 0 or sd is None or k < 1:
        return []
    a, b = int(sd.starts[k - 1]), int(sd.ends[k - 1])
    t = sd.t[a:b]
    ja = int(np.searchsorted(t, hhmm(p.at)))
    jb = int(np.searchsorted(t, hhmm(p.end)))
    if ja >= len(t) or jb >= len(t) or t[ja] != hhmm(p.at) or t[jb] != hhmm(p.end):
        return []
    r = (sd.c[a + jb] / sd.c[a + ja] - 1.0) * 100
    c = float(s.c[i])
    if r >= p.r:
        return [Entry(i, LONG, stop=c * (1 - p.stop_pct / 100))]
    if r <= -p.r:
        return [Entry(i, SHORT, stop=c * (1 + p.stop_pct / 100))]
    return []
