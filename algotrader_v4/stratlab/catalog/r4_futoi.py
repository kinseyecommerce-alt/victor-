"""
Round-4 NSE-SPECIFIC candidates, family STOCK FUTURES OPEN INTEREST (and stock option OI).
Pre-registered BEFORE any result in scratchpad/r4/oi/PREREGISTERED.txt - parameters are textbook / round values; do not
tune them. Every variant is one more candidate that can pass by luck.

All specs are kind="futures" without underlying: a stock future simulated on the stock's cash bars, futures costs.

LOOK-AHEAD. NSE end-of-day files (F&O bhavcopy, option OI) dated D are published the evening of D and are usable from
the next session's open. Data is read only through nsedata.asof_swing(d) (positional, next-open fill: rows <= D) or
nsedata.asof_session(s) (intraday on session T: rows <= T-1). On top of that, `_rows` re-checks EVERY result: a row dated
on/after the fill date (positional) or on/after the session date (intraday) raises LookAheadError (the spec is then
reported INVALID instead of silently using it), and a latest row that is not exactly the expected day is treated as
missing data (no signal).
"""
from __future__ import annotations

import numpy as np

from .. import nsedata as N
from ..api import LONG, SHORT, Entry, Swing, cross_dn, cross_up, first_true, hhmm, strategy

_CA_GAPS = {("SIEMENS", 20250407), ("VEDL", 20260430)}       # same list as r3 / ltr.CA_GAPS


# ── look-ahead guard and small helpers ───────────────────────────────────────
def _rows(rows, limit: int, expect: int | None):
    """Guard every nsedata result. limit = first date that must NOT appear (fill date / session date); expect = the
    date the latest row must have (else None = missing / stale data, no signal)."""
    if len(rows) and int(np.max(rows["date"])) >= int(limit):
        raise N.LookAheadError(f"nsedata returned a row dated {int(np.max(rows['date']))} >= {int(limit)}")
    if not len(rows):
        return None
    if expect is not None and int(rows["date"][-1]) != int(expect):
        return None
    return rows


def _swing_asof(d):
    """Asof for a next-open positional decision on day D = d.date[-1]: EOD rows <= D only (fill is the next session)."""
    a = N.asof_swing(d)
    D = int(d.date[-1])
    if a.before != D + 1:
        raise N.LookAheadError(f"asof_swing before={a.before} != {D + 1}")
    return a, D + 1, D                          # (asof, limit, expected latest row date)


def _session_asof(s):
    """Asof for an intraday decision on session T = s.date: EOD rows <= previous session only."""
    a = N.asof_session(s)
    T = int(s.date)
    if a.before != T:
        raise N.LookAheadError(f"asof_session before={a.before} != {T}")
    prev = int(s.prev.date[-1]) if len(s.prev) else None
    return a, T, prev


def _ca_clean(sym: str, dates) -> bool:
    if len(dates) == 0:
        return True
    recent = dates[-250:]
    return not any(s_ == sym and recent[0] <= day <= recent[-1] for s_, day in _CA_GAPS)


def _fin(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


# ════════════════════════════════════════════════════════════════════════════
# Q. Price-OI quadrants on day D
# ════════════════════════════════════════════════════════════════════════════
def _quadrant(f, closes, p) -> str | None:
    """f = fut_oi rows (>= 2, last = D); closes = bars closes ending at D. Returns LB / SB / SC / LU or None."""
    if f is None or len(f) < 2 or len(closes) < 2:
        return None
    ne0, ne1, d1 = f["near_expiry"][-2], f["near_expiry"][-1], f["date"][-1]
    if not _fin(ne0, ne1) or ne0 != ne1 or not ne1 > d1:          # F3 expiry contamination
        return None
    o0, o1 = float(f["oi"][-2]), float(f["oi"][-1])
    c0, c1 = float(closes[-2]), float(closes[-1])
    if not (_fin(o0, o1, c0, c1) and o0 > 0 and c0 > 0):
        return None
    doi = o1 / o0 - 1.0
    dp = c1 / c0 - 1.0
    if abs(doi) > p.oi_max / 100.0:                                # F2 lot / split adjustments
        return None
    up, dn = dp > p.dp / 100.0, dp < -p.dp / 100.0
    if doi > p.doi / 100.0:
        return "LB" if up else ("SB" if dn else None)
    if doi < -p.doi / 100.0:
        return "SC" if up else ("LU" if dn else None)
    return None


def _quad_swing(want: str, side: int):
    def fn(d, p):
        if len(d) < 3 or not _ca_clean(d.symbol, d.date):
            return None
        a, lim, exp = _swing_asof(d)
        f = _rows(a.fut_oi(2), lim, exp)
        if f is None or _quadrant(f, d.c, p) != want or a.banned():
            return None
        return Swing(side, stop_pct=p.stop_pct, max_hold=p.hold)
    return fn


def _quad_intraday(want: str, side: int):
    def fn(s, p):
        if len(s) < 1 or len(s.prev) < 3 or not _ca_clean(s.symbol, s.prev.date):
            return []
        a, lim, exp = _session_asof(s)
        f = _rows(a.fut_oi(2), lim, exp)
        if f is None or _quadrant(f, s.prev.c, p) != want or a.banned():
            return []
        c0 = float(s.c[0])
        return [Entry(0, side, stop=c0 * (1 - side * p.stop_pct / 100.0))]
    return fn


_QP = {"dp": 1.0, "doi": 5.0, "oi_max": 50.0}
for _q, _side, _name, _txt in (("LB", LONG, "lb_long", "long buildup (price > +1 %, OI > +5 %)"),
                               ("SB", SHORT, "sb_short", "short buildup (price < -1 %, OI > +5 %)"),
                               ("SC", LONG, "sc_long", "short covering (price > +1 %, OI < -5 %)"),
                               ("LU", SHORT, "lu_short", "long unwinding (price < -1 %, OI < -5 %)")):
    _dir = "LONG" if _side == LONG else "SHORT"
    strategy(f"r4_oi_{_name}_1d", "futures", "oi_quadrant",
             f"Stock future {_txt} on D (NSE EOD) -> {_dir} next open, exit that close, 3 % stop",
             {**_QP, "stop_pct": 3.0, "hold": 1}, horizon="positional")(_quad_swing(_q, _side))
    strategy(f"r4_oi_{_name}_5d", "futures", "oi_quadrant",
             f"Stock future {_txt} on D (NSE EOD) -> {_dir} next open, hold 5 sessions, 5 % stop",
             {**_QP, "stop_pct": 5.0, "hold": 5}, horizon="positional")(_quad_swing(_q, _side))
    if _q in ("LB", "SB"):
        strategy(f"r4_oi_{_name}_id", "futures", "oi_quadrant",
                 f"Stock future {_txt} on the previous session -> intraday {_dir} at 09:16, 1 % stop, exit 15:15",
                 {**_QP, "stop_pct": 1.0}, horizon="intraday")(_quad_intraday(_q, _side))


# ════════════════════════════════════════════════════════════════════════════
# R. Rollover week
# ════════════════════════════════════════════════════════════════════════════
def _roll(d, p):
    if len(d) < p.ret_n + 1 or not _ca_clean(d.symbol, d.date):
        return None
    a, lim, exp = _swing_asof(d)
    f = _rows(a.fut_oi(100), lim, exp)
    if f is None or len(f) < 40:
        return None
    D, ne = int(f["date"][-1]), f["near_expiry"][-1]
    if not _fin(ne) or not ne > D or N.next_fo_session(D) != int(ne):      # D must be the expiry eve
        return None
    oi, noi, dt, nx = f["oi"], f["near_oi"], f["date"], f["near_expiry"]

    def ro(i):
        return (oi[i] - noi[i]) / oi[i] if oi[i] > 0 and _fin(noi[i]) else np.nan

    prev = [i for i in range(len(f) - 1) if _fin(nx[i]) and int(dt[i + 1]) == int(nx[i])]   # earlier eves
    prev = [i for i in prev if np.isfinite(ro(i))][-p.ref:]
    now = ro(len(f) - 1)
    if len(prev) < p.ref or not np.isfinite(now):
        return None
    if not now > float(np.mean([ro(i) for i in prev])):
        return None
    ret = float(d.c[-1]) / float(d.c[-1 - p.ret_n]) - 1.0
    if not np.isfinite(ret) or ret == 0 or a.banned():
        return None
    return Swing(LONG if ret > 0 else SHORT, stop_pct=p.stop_pct, max_hold=p.hold)


strategy("r4_oi_roll_hi_5d", "futures", "oi_rollover",
         "Expiry eve: rollover share above its mean of the previous 3 eves -> trade the 20-session trend, hold 5, 5 % stop",
         {"ref": 3, "ret_n": 20, "hold": 5, "stop_pct": 5.0}, horizon="positional")(_roll)
strategy("r4_oi_roll_hi_15d", "futures", "oi_rollover",
         "Expiry eve: rollover share above its mean of the previous 3 eves -> trade the 20-session trend, hold 15, 8 % stop",
         {"ref": 3, "ret_n": 20, "hold": 15, "stop_pct": 8.0}, horizon="positional")(_roll)


# ════════════════════════════════════════════════════════════════════════════
# B. Basis extremes (annualised premium of the near future over the official cash close)
# ════════════════════════════════════════════════════════════════════════════
def _basis(side: int):
    def fn(d, p):
        if not _ca_clean(d.symbol, d.date):
            return None
        a, lim, exp = _swing_asof(d)
        f = _rows(a.fut_oi(p.win), lim, exp)
        if f is None or len(f) < p.min_valid:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            ab = 100.0 * (f["near_settle"] / f["spot"] - 1.0) * 365.0 / f["dte"]
        ok = np.isfinite(ab) & (f["dte"] >= p.min_dte) & (f["spot"] > 0)
        if not ok[-1] or ok.sum() < p.min_valid:
            return None
        x = ab[ok]
        sd = float(np.std(x, ddof=1))
        if not sd > 0:
            return None
        z = (float(ab[-1]) - float(np.mean(x))) / sd
        hit = z > p.z if side == SHORT else z < -p.z
        if not hit or a.banned():
            return None
        return Swing(side, stop_pct=p.stop_pct, max_hold=p.hold)
    return fn


_BP = {"z": 2.0, "win": 60, "min_valid": 40, "min_dte": 7, "hold": 5, "stop_pct": 5.0}
strategy("r4_oi_basis_hi_short_5d", "futures", "oi_basis",
         "Annualised near-future premium z > +2 vs its last 60 sessions (dte >= 7) -> SHORT next open, hold 5, 5 % stop",
         dict(_BP), horizon="positional")(_basis(SHORT))
strategy("r4_oi_basis_lo_long_5d", "futures", "oi_basis",
         "Annualised near-future premium z < -2 vs its last 60 sessions (dte >= 7) -> LONG next open, hold 5, 5 % stop",
         dict(_BP), horizon="positional")(_basis(LONG))


# ════════════════════════════════════════════════════════════════════════════
# H. OI at a 60-session high with price at a 60-session extreme
# ════════════════════════════════════════════════════════════════════════════
def _oihigh(side: int):
    def fn(d, p):
        if len(d) < p.win or not _ca_clean(d.symbol, d.date):
            return None
        a, lim, exp = _swing_asof(d)
        f = _rows(a.fut_oi(p.win), lim, exp)
        if f is None or len(f) < p.win:
            return None
        oi = f["oi"]
        if not (np.isfinite(oi).all() and _fin(f["dte"][-1]) and f["dte"][-1] >= p.min_dte):
            return None
        if oi[-1] < oi.max():
            return None
        c = d.c[-p.win:]
        hit = c[-1] >= c.max() if side == LONG else c[-1] <= c.min()
        if not hit or a.banned():
            return None
        return Swing(side, stop_pct=p.stop_pct, max_hold=p.hold)
    return fn


_HP = {"win": 60, "min_dte": 7, "hold": 10, "stop_pct": 5.0}
strategy("r4_oi_oihigh_px_high_long", "futures", "oi_high",
         "Futures OI at a 60-session high (outside expiry week) and close at a 60-session high -> LONG, hold 10, 5 % stop",
         dict(_HP), horizon="positional")(_oihigh(LONG))
strategy("r4_oi_oihigh_px_low_short", "futures", "oi_high",
         "Futures OI at a 60-session high (outside expiry week) and close at a 60-session low -> SHORT, hold 10, 5 % stop",
         dict(_HP), horizon="positional")(_oihigh(SHORT))


# ════════════════════════════════════════════════════════════════════════════
# P. Stock PCR (OI) extremes, contrarian
# ════════════════════════════════════════════════════════════════════════════
def _pcr(side: int):
    def fn(d, p):
        if not _ca_clean(d.symbol, d.date):
            return None
        a, lim, exp = _swing_asof(d)
        o = _rows(a.opt_agg(p.win), lim, exp)
        if o is None or len(o) < p.min_valid:
            return None
        D, ne = int(o["date"][-1]), o["near_expiry"][-1]
        if not _fin(ne) or (N._d(int(ne)) - N._d(D)).days < p.min_dte:
            return None
        if len(o) >= 2 and o["near_expiry"][-2] != ne:                   # first session of a new series
            return None
        x = o["pcr_oi"]
        ok = np.isfinite(x)
        if not ok[-1] or ok.sum() < p.min_valid:
            return None
        sd = float(np.std(x[ok], ddof=1))
        if not sd > 0:
            return None
        z = (float(x[-1]) - float(np.mean(x[ok]))) / sd
        hit = z > p.z if side == LONG else z < -p.z
        if not hit or a.banned():
            return None
        return Swing(side, stop_pct=p.stop_pct, max_hold=p.hold)
    return fn


_PP = {"z": 2.0, "win": 60, "min_valid": 40, "min_dte": 7, "hold": 5, "stop_pct": 5.0}
strategy("r4_oi_pcr_hi_long_5d", "futures", "oi_pcr",
         "Stock option PCR (OI) z > +2 vs its last 60 sessions (>= 7 days to expiry) -> contrarian LONG, hold 5, 5 % stop",
         dict(_PP), horizon="positional")(_pcr(LONG))
strategy("r4_oi_pcr_lo_short_5d", "futures", "oi_pcr",
         "Stock option PCR (OI) z < -2 vs its last 60 sessions (>= 7 days to expiry) -> contrarian SHORT, hold 5, 5 % stop",
         dict(_PP), horizon="positional")(_pcr(SHORT))


# ════════════════════════════════════════════════════════════════════════════
# M. Max-OI strikes as resistance / support
# ════════════════════════════════════════════════════════════════════════════
def _levels_intraday(s, p):
    """(R, S) from the previous session's near-expiry option OI, or None."""
    if len(s) < 1 or len(s.prev) < 2 or not _ca_clean(s.symbol, s.prev.date) or not _fin(s.pdc):
        return None
    a, lim, exp = _session_asof(s)
    o = _rows(a.opt_agg(1), lim, exp)
    f = _rows(a.fut_oi(1), lim, exp)
    if o is None or f is None:
        return None
    D, ne = int(o["date"][-1]), o["near_expiry"][-1]
    R, S, spot = o["max_call_oi_strike"][-1], o["max_put_oi_strike"][-1], f["spot"][-1]
    if not _fin(ne, R, S, spot) or not ne > D:                           # expiry still alive today
        return None
    if abs(float(spot) / float(s.pdc) - 1.0) > p.px_tol / 100.0:         # unadjusted strikes vs adjusted bars
        return None
    if not (S < s.pdc < R) or a.banned():
        return None
    return float(R), float(S)


def _maxoi_fade(s, p):
    lv = _levels_intraday(s, p)
    if lv is None:
        return []
    R, S = lv
    win = (s.t >= hhmm(p.start)) & (s.t < hhmm(p.end))
    b = p.buf / 100.0
    out = []
    i = first_true(win & (s.h >= R) & (s.c < R))
    if i >= 0:
        out.append(Entry(i, SHORT, stop=R * (1 + b), rr=p.rr))
    j = first_true(win & (s.l <= S) & (s.c > S))
    if j >= 0:
        out.append(Entry(j, LONG, stop=S * (1 - b), rr=p.rr))
    return out


def _maxoi_break(s, p):
    lv = _levels_intraday(s, p)
    if lv is None:
        return []
    R, S = lv
    win = (s.t >= hhmm(p.start)) & (s.t < hhmm(p.end))
    b = p.buf / 100.0
    out = []
    i = first_true(win & cross_up(s.c, R))
    if i >= 0:
        out.append(Entry(i, LONG, stop=R * (1 - b), rr=p.rr))
    j = first_true(win & cross_dn(s.c, S))
    if j >= 0:
        out.append(Entry(j, SHORT, stop=S * (1 + b), rr=p.rr))
    return out


strategy("r4_oi_maxoi_fade_id", "futures", "oi_maxoi",
         "Intraday: reject at yesterday's max-call-OI strike -> SHORT, at max-put-OI strike -> LONG; 0.5 % stop, 1.5R",
         {"buf": 0.5, "rr": 1.5, "start": "09:30", "end": "14:30", "px_tol": 2.0}, horizon="intraday")(_maxoi_fade)
strategy("r4_oi_maxoi_break_id", "futures", "oi_maxoi",
         "Intraday: close crosses above yesterday's max-call-OI strike -> LONG, below max-put-OI -> SHORT; 0.5 % stop, 2R",
         {"buf": 0.5, "rr": 2.0, "start": "09:30", "end": "14:30", "px_tol": 2.0}, horizon="intraday")(_maxoi_break)


@strategy("r4_oi_maxoi_break_5d", "futures", "oi_maxoi",
          "NSE close of D breaks above D's max-call-OI strike -> LONG, below max-put-OI -> SHORT; hold 5, 5 % stop",
          {"hold": 5, "stop_pct": 5.0}, horizon="positional")
def r4_oi_maxoi_break_5d(d, p):
    if not _ca_clean(d.symbol, d.date):
        return None
    a, lim, exp = _swing_asof(d)
    o = _rows(a.opt_agg(2), lim, exp)
    f = _rows(a.fut_oi(2), lim, exp)
    if o is None or f is None or len(o) < 2 or len(f) < 2:
        return None
    if not np.array_equal(o["date"], f["date"]):
        return None
    D = int(o["date"][-1])
    ne = o["near_expiry"]
    if not _fin(ne[-1]) or not ne[-1] > D:              # not on the expiry day (strikes of an expiring series)
        return None
    R0, R1 = o["max_call_oi_strike"]
    S0, S1 = o["max_put_oi_strike"]
    c0, c1 = f["spot"]
    if not _fin(R0, R1, S0, S1, c0, c1) or a.banned():
        return None
    if c1 > R1 and c0 <= R0:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    if c1 < S1 and c0 >= S0:
        return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.hold)
    return None
