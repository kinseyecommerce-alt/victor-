"""
Round-4 NSE-specific candidates, family DELIVERY % AND VOLUME QUALITY (22 strategies, ids r4_delivery_*).
Pre-registered BEFORE any result in scratchpad/r4/delivery/PREREGISTERED.txt - parameters are round / textbook
values fixed there; do not tune them. Every variant is one more candidate that can pass by luck.

NSE data comes ONLY through stratlab.nsedata's as-of helpers:
  * swing / futures-positional (default next-open fill): N.asof_swing(d) -> delivery rows dated <= d.date[-1]
    (published that evening; the fill is the next session's open). Ban status = the fill date.
  * intraday: N.asof_session(s) -> delivery rows dated <= the previous session; nothing dated s.date is read.
_guard_rows() re-checks every Rows object read and raises LookAheadError if a row is dated at/after the date it
must precede, so a mistake can never be used silently. Stale rows (last row not dated the expected day) skip
the signal.
"""
from __future__ import annotations

import numpy as np

from .. import nsedata as N
from ..api import LONG, SHORT, Entry, Swing, strategy

HYG_N = 130          # data-hygiene window (sessions)
HYG_JUMP = 0.25      # |close/prev close - 1| or |open/prev close - 1| above this -> unadjusted corporate action
_CA_GAPS = {("SIEMENS", 20250407), ("VEDL", 20260430)}     # same list as stratlab.ltr.CA_GAPS
HIST = 60            # own-history window (delivery rows) for percentiles
MIN_FINITE = 50      # finite values required in that window
VOL_N = 20           # bars volume average window


# ── guards ──────────────────────────────────────────────────────────────────
def _guard_rows(rows, must_precede: int, what: str):
    """Explicit look-ahead guard: every row must be dated strictly before `must_precede`."""
    if len(rows) and int(np.max(rows["date"])) >= int(must_precede):
        raise N.LookAheadError(f"r4_delivery: {what} returned a row dated {int(np.max(rows['date']))} "
                               f">= {int(must_precede)} (decision/fill date)")
    return rows


def _deliv_swing(d, n: int, sym: str | None = None, a=None):
    """Delivery rows for a next-open Swing deciding on the evening of D = d.date[-1]: rows dated <= D only
    (D's file is published that evening; the fill is the next session's open). None if D's row is missing."""
    day = int(d.date[-1])
    a = a or N.asof_swing(d)
    r = _guard_rows(a.delivery(n=n, sym=sym), day + 1, "asof_swing.delivery")   # never a row dated > D
    if len(r) == 0 or int(r.date[-1]) != day:
        return None
    return r


def _deliv_intraday(s, n: int):
    """Delivery rows for an intraday decision on session T = s.date: rows dated <= T-1 only (T's own file does
    not exist until the evening of T). None unless the last row is the previous bars session."""
    a = N.asof_session(s)
    r = _guard_rows(a.delivery(n=n), int(s.date), "asof_session.delivery")      # never a row dated >= T
    if len(r) == 0 or len(s.prev.date) == 0 or int(r.date[-1]) != int(s.prev.date[-1]):
        return None
    return r


# ── hygiene / helpers ───────────────────────────────────────────────────────
def _clean_arrays(sym, dates, o, c) -> bool:
    n = min(len(c), HYG_N + 1)
    if n >= 2:
        cc, oo = c[-n:], o[-n:]
        pc = cc[:-1]
        if np.any(np.abs(cc[1:] / pc - 1.0) > HYG_JUMP) or np.any(np.abs(oo[1:] / pc - 1.0) > HYG_JUMP):
            return False
    if sym and len(dates):
        recent = dates[-250:]
        for s_, day in _CA_GAPS:
            if s_ == sym and recent[0] <= day <= recent[-1]:
                return False
    return True


def _clean(d) -> bool:
    return _clean_arrays(getattr(d, "symbol", None), d.date, d.o, d.c)


def _ok(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def _prior_today(r, hist: int = HIST):
    """(prior finite deliv_pct values of the `hist` rows before the last row, last row's deliv_pct)."""
    x = np.asarray(r.deliv_pct, dtype=float)
    if len(x) < hist + 1:
        return None, np.nan
    prior = x[-hist - 1:-1]
    prior = prior[np.isfinite(prior)]
    if len(prior) < MIN_FINITE:
        return None, np.nan
    return prior, float(x[-1])


def _vol_ratio(d) -> float:
    if len(d) < VOL_N + 1:
        return np.nan
    m = float(np.mean(d.v[-VOL_N - 1:-1]))
    return float(d.v[-1]) / m if m > 0 else np.nan


def _sma(x, n: int) -> float:
    return float(np.mean(x[-n:])) if len(x) >= n else np.nan


def _need(d, n: int) -> bool:
    return len(d) >= n and _clean(d)


# ════════════════════════════════════════════════════════════════════════════
# A / B. Accumulation (long) and distribution (short) on high-delivery volume days
# ════════════════════════════════════════════════════════════════════════════
def _hi_deliv_volume_day(d, direction: int, trend: bool):
    if not _need(d, 60):
        return False
    up = d.c[-1] > d.c[-2] if direction > 0 else d.c[-1] < d.c[-2]
    vr = _vol_ratio(d)
    if not (up and _ok(vr) and vr > 1.5):
        return False
    if trend:
        sm = _sma(d.c, 50)
        if not _ok(sm) or not (d.c[-1] > sm if direction > 0 else d.c[-1] < sm):
            return False
    r = _deliv_swing(d, HIST + 1)
    if r is None:
        return False
    prior, today = _prior_today(r)
    return prior is not None and _ok(today) and today > np.percentile(prior, 90)


def _distrib(d, p):
    if not _hi_deliv_volume_day(d, -1, p.trend):
        return None
    if N.asof_swing(d).banned():
        return None
    return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.hold)


def _accum_fn(d, p):
    if _hi_deliv_volume_day(d, +1, p.trend):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


_AB = "deliv_pct > 90th pct of own prior 60 on an {} with volume > 1.5x 20d avg"
strategy("r4_delivery_accum_h5", "swing", "deliv_accum",
         _AB.format("up-day") + ": buy next open, 8 % stop, hold 5",
         {"stop_pct": 8.0, "hold": 5, "trend": False})(_accum_fn)
strategy("r4_delivery_accum_h10", "swing", "deliv_accum",
         _AB.format("up-day") + ": buy next open, 8 % stop, hold 10",
         {"stop_pct": 8.0, "hold": 10, "trend": False})(_accum_fn)
strategy("r4_delivery_accum_trend", "swing", "deliv_accum",
         _AB.format("up-day") + " and close > SMA50: buy next open, 8 % stop, hold 10",
         {"stop_pct": 8.0, "hold": 10, "trend": True})(_accum_fn)
strategy("r4_delivery_distrib_h5", "futures", "deliv_distrib",
         _AB.format("down-day") + ", not F&O-banned: short future next open, 8 % stop, hold 5",
         {"stop_pct": 8.0, "hold": 5, "trend": False}, horizon="positional")(_distrib)
strategy("r4_delivery_distrib_h10", "futures", "deliv_distrib",
         _AB.format("down-day") + ", not F&O-banned: short future next open, 8 % stop, hold 10",
         {"stop_pct": 8.0, "hold": 10, "trend": False}, horizon="positional")(_distrib)
strategy("r4_delivery_distrib_trend", "futures", "deliv_distrib",
         _AB.format("down-day") + ", close < SMA50, not banned: short future next open, 8 % stop, hold 10",
         {"stop_pct": 8.0, "hold": 10, "trend": True}, horizon="positional")(_distrib)


# ════════════════════════════════════════════════════════════════════════════
# C. Quiet accumulation: rising delivery share with flat price
# ════════════════════════════════════════════════════════════════════════════
def _deliv_rising(d) -> bool:
    """mean deliv_pct of the last 5 rows (D-4..D) >= 1.2 x mean of the 60 rows before them (D-64..D-5)."""
    r = _deliv_swing(d, 65)
    if r is None or len(r) < 65:
        return False
    x = np.asarray(r.deliv_pct, dtype=float)
    last5, base = x[-5:], x[-65:-5]
    base = base[np.isfinite(base)]
    if not np.all(np.isfinite(last5)) or len(base) < MIN_FINITE:
        return False
    b = float(base.mean())
    return b > 0 and float(last5.mean()) >= 1.2 * b


@strategy("r4_delivery_quiet_brk", "swing", "deliv_quiet",
          "5d mean deliv_pct >= 1.2x prior-60 mean, |20d return| < 5 %, close > prior 20d high close: buy, 8 % stop, hold 10",
          {"ratio": 1.2, "flat_n": 20, "flat_pct": 5.0, "brk_n": 20, "stop_pct": 8.0, "hold": 10})
def r4_delivery_quiet_brk(d, p):
    if not _need(d, 70):
        return None
    c = d.c
    if not (abs(c[-1] / c[-1 - p.flat_n] - 1.0) < p.flat_pct / 100.0 and c[-1] > c[-1 - p.brk_n:-1].max()):
        return None
    if _deliv_rising(d):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r4_delivery_quiet_flat", "swing", "deliv_quiet",
          "5d mean deliv_pct >= 1.2x prior-60 mean and |10d return| < 3 %: buy next open, 8 % stop, hold 10",
          {"ratio": 1.2, "flat_n": 10, "flat_pct": 3.0, "stop_pct": 8.0, "hold": 10})
def r4_delivery_quiet_flat(d, p):
    if not _need(d, 70):
        return None
    if not abs(d.c[-1] / d.c[-1 - p.flat_n] - 1.0) < p.flat_pct / 100.0:
        return None
    if _deliv_rising(d):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r4_delivery_quiet_top10", "swing", "deliv_quiet",
          "5d mean deliv_pct > 90th pct of its own prior 60 values and 10d close range < 5 %: buy, 8 % stop, hold 10",
          {"range_n": 10, "range_pct": 5.0, "stop_pct": 8.0, "hold": 10})
def r4_delivery_quiet_top10(d, p):
    if not _need(d, 70):
        return None
    w = d.c[-p.range_n:]
    if not (w.max() - w.min()) / w.min() < p.range_pct / 100.0:
        return None
    r = _deliv_swing(d, 65)                       # 5-day means ending D-60..D need rows D-64..D
    if r is None or len(r) < 65:
        return None
    x = np.asarray(r.deliv_pct, dtype=float)
    m5 = np.convolve(x, np.ones(5) / 5.0, mode="valid")          # m5[-1] ends D, m5[-61:-1] end D-60..D-1
    if len(m5) < 61 or not np.isfinite(m5[-1]):
        return None
    prior = m5[-61:-1]
    prior = prior[np.isfinite(prior)]
    if len(prior) < MIN_FINITE:
        return None
    if m5[-1] > np.percentile(prior, 90):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# D. Low-delivery speculative spike: next-day fade
# ════════════════════════════════════════════════════════════════════════════
def _spike_fade(s, p):
    """Session T: the previous session moved >= 3 % (side-dependent) on bottom-decile delivery -> fade at the
    09:16 open. Uses deliv_pct dated T-1 (published the evening of T-1) only."""
    if len(s) < 1 or len(s.prev.c) < 3:
        return []
    if not _clean_arrays(s.symbol, s.prev.date, s.prev.o, s.prev.c):
        return []
    ret = float(s.prev.c[-1] / s.prev.c[-2] - 1.0)
    if not (ret >= p.move_pct / 100.0 if p.side < 0 else ret <= -p.move_pct / 100.0):
        return []
    r = _deliv_intraday(s, HIST + 1)
    if r is None:
        return []
    prior, today = _prior_today(r)
    if prior is None or not _ok(today) or not today < np.percentile(prior, 10):
        return []
    c0 = float(s.c[0])
    stop = c0 * (1.0 + p.stop_pct / 100.0) if p.side < 0 else c0 * (1.0 - p.stop_pct / 100.0)
    return [Entry(0, SHORT if p.side < 0 else LONG, stop=stop)]


strategy("r4_delivery_spike_fade_short", "intraday", "deliv_spike",
         "Prev session >= +3 % on bottom-decile deliv_pct (own prior 60): short 09:16 open, 2 % stop, exit 15:15",
         {"move_pct": 3.0, "stop_pct": 2.0, "side": -1})(_spike_fade)
strategy("r4_delivery_spike_fade_long", "intraday", "deliv_spike",
         "Prev session <= -3 % on bottom-decile deliv_pct (own prior 60): buy 09:16 open, 2 % stop, exit 15:15",
         {"move_pct": 3.0, "stop_pct": 2.0, "side": +1})(_spike_fade)


@strategy("r4_delivery_spike_fade_3d", "futures", "deliv_spike",
          "Day >= +3 % on bottom-decile deliv_pct, not F&O-banned: short future next open, 5 % stop, hold 3",
          {"move_pct": 3.0, "stop_pct": 5.0, "hold": 3}, horizon="positional")
def r4_delivery_spike_fade_3d(d, p):
    if not _need(d, 3) or not d.c[-1] / d.c[-2] - 1.0 >= p.move_pct / 100.0:
        return None
    r = _deliv_swing(d, HIST + 1)
    if r is None:
        return None
    prior, today = _prior_today(r)
    if prior is None or not _ok(today) or not today < np.percentile(prior, 10):
        return None
    if N.asof_swing(d).banned():
        return None
    return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.hold)


# ════════════════════════════════════════════════════════════════════════════
# E. Delivery-weighted relative strength vs NIFTY
# ════════════════════════════════════════════════════════════════════════════
def _dw_parts(d, n: int = 20):
    """(R_hi, R_lo, DW, PLAIN) over the last n sessions, or None. Delivery rows must match the n bar dates."""
    if not _need(d, n + 2) or d.nifty is None or len(d.nifty) < n + 1:
        return None
    nf = np.asarray(d.nifty[-n - 1:], dtype=float)
    if not np.all(np.isfinite(nf)) or np.any(nf <= 0):
        return None
    r = _deliv_swing(d, HIST + n)
    if r is None or len(r) < HIST + n or not np.array_equal(r.date[-n:], d.date[-n:]):
        return None
    x = np.asarray(r.deliv_pct, dtype=float)
    win, base = x[-n:], x[-n - HIST:-n]
    base = base[np.isfinite(base)]
    if not np.all(np.isfinite(win)) or len(base) < MIN_FINITE:
        return None
    ex = np.diff(np.log(d.c[-n - 1:])) - np.diff(np.log(nf))
    hi = win >= np.median(base)
    w = win / win.mean() if win.mean() > 0 else None
    if w is None:
        return None
    return float(ex[hi].sum()), float(ex[~hi].sum()), float((w * ex).sum()), float(ex.sum())


@strategy("r4_delivery_dwrs_long", "swing", "deliv_rs",
          "20d excess return vs NIFTY on high-delivery days > +5 % and on low-delivery days <= 0: buy, 8 % stop, hold 10",
          {"n": 20, "thr": 0.05, "stop_pct": 8.0, "hold": 10})
def r4_delivery_dwrs_long(d, p):
    q = _dw_parts(d, p.n)
    if q is not None and q[0] > p.thr and q[1] <= 0:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r4_delivery_dwrs_short", "futures", "deliv_rs",
          "20d excess return on high-delivery days < -5 % and on low-delivery days >= 0, not banned: short, 8 % stop, hold 10",
          {"n": 20, "thr": 0.05, "stop_pct": 8.0, "hold": 10}, horizon="positional")
def r4_delivery_dwrs_short(d, p):
    q = _dw_parts(d, p.n)
    if q is not None and q[0] < -p.thr and q[1] >= 0 and not N.asof_swing(d).banned():
        return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r4_delivery_dwrs_weighted", "swing", "deliv_rs",
          "Delivery-weighted 20d excess return vs NIFTY > +5 % and > the unweighted one: buy, 8 % stop, hold 10",
          {"n": 20, "thr": 0.05, "stop_pct": 8.0, "hold": 10})
def r4_delivery_dwrs_weighted(d, p):
    q = _dw_parts(d, p.n)
    if q is not None and q[2] > p.thr and q[2] > q[3]:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# F. Delivery quantity surge
# ════════════════════════════════════════════════════════════════════════════
@strategy("r4_delivery_dqty_surge", "swing", "deliv_qty",
          "deliv_qty > 3x its prior 20-session mean on a >= +1 % day: buy next open, 8 % stop, hold 10",
          {"n": 20, "mult": 3.0, "min_ret_pct": 1.0, "stop_pct": 8.0, "hold": 10})
def r4_delivery_dqty_surge(d, p):
    if not _need(d, p.n + 2) or not d.c[-1] / d.c[-2] - 1.0 >= p.min_ret_pct / 100.0:
        return None
    r = _deliv_swing(d, p.n + 1)
    if r is None or len(r) < p.n + 1 or not np.array_equal(r.date, d.date[-p.n - 1:]):
        return None
    # unadjusted quantities: the official close ratio must agree with the adjusted bars over the window
    off = float(r.close[-1] / r.close[0]) if r.close[0] > 0 else np.nan
    bar = float(d.c[-1] / d.c[-p.n - 1])
    if not _ok(off) or abs(off / bar - 1.0) > 0.20:
        return None
    q = np.asarray(r.deliv_qty, dtype=float)
    base = q[:-1]
    if not np.all(np.isfinite(q)) or base.mean() <= 0:
        return None
    if q[-1] > p.mult * base.mean():
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# G. Low-delivery multi-day moves
# ════════════════════════════════════════════════════════════════════════════
def _low_deliv_5d(d) -> bool:
    """mean deliv_pct of D-4..D < 20th percentile of the 60 daily rows D-64..D-5."""
    r = _deliv_swing(d, 65)
    if r is None or len(r) < 65:
        return False
    x = np.asarray(r.deliv_pct, dtype=float)
    last5, base = x[-5:], x[-65:-5]
    base = base[np.isfinite(base)]
    return bool(np.all(np.isfinite(last5)) and len(base) >= MIN_FINITE
                and last5.mean() < np.percentile(base, 20))


@strategy("r4_delivery_lowdel_rally_short", "futures", "deliv_lowmove",
          "5d return >= +8 % with 5d mean deliv_pct < 20th pct of prior 60, not banned: short future, 8 % stop, hold 5",
          {"n": 5, "move_pct": 8.0, "stop_pct": 8.0, "hold": 5}, horizon="positional")
def r4_delivery_lowdel_rally_short(d, p):
    if not _need(d, 70) or not d.c[-1] / d.c[-1 - p.n] - 1.0 >= p.move_pct / 100.0:
        return None
    if _low_deliv_5d(d) and not N.asof_swing(d).banned():
        return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r4_delivery_lowdel_selloff_long", "swing", "deliv_lowmove",
          "5d return <= -8 % with 5d mean deliv_pct < 20th pct of prior 60: buy next open, 10 % stop, hold 5",
          {"n": 5, "move_pct": 8.0, "stop_pct": 10.0, "hold": 5})
def r4_delivery_lowdel_selloff_long(d, p):
    if not _need(d, 70) or not d.c[-1] / d.c[-1 - p.n] - 1.0 <= -p.move_pct / 100.0:
        return None
    if _low_deliv_5d(d):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# H. Delivery-confirmed vs unconfirmed breakouts
# ════════════════════════════════════════════════════════════════════════════
def _brk_deliv(d, n: int):
    """(today's deliv_pct, prior 60 values) if close(D) > highest close of D-n..D-1, else None."""
    if not _need(d, max(n + 1, 70)) or not d.c[-1] > d.c[-n - 1:-1].max():
        return None
    r = _deliv_swing(d, HIST + 1)
    if r is None:
        return None
    prior, today = _prior_today(r)
    if prior is None or not _ok(today):
        return None
    return today, prior


@strategy("r4_delivery_brk_confirmed", "swing", "deliv_breakout",
          "Close > prior 50-session high close with deliv_pct >= 75th pct of own prior 60: buy, 8 % stop, hold 10",
          {"n": 50, "pct": 75, "stop_pct": 8.0, "hold": 10})
def r4_delivery_brk_confirmed(d, p):
    q = _brk_deliv(d, p.n)
    if q is not None and q[0] >= np.percentile(q[1], p.pct):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


@strategy("r4_delivery_brk_fake_short", "futures", "deliv_breakout",
          "Close > prior 50-session high close with deliv_pct <= 25th pct of own prior 60, not banned: short, 5 % stop, hold 5",
          {"n": 50, "pct": 25, "stop_pct": 5.0, "hold": 5}, horizon="positional")
def r4_delivery_brk_fake_short(d, p):
    q = _brk_deliv(d, p.n)
    if q is not None and q[0] <= np.percentile(q[1], p.pct) and not N.asof_swing(d).banned():
        return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# I. Market-wide delivery breadth -> NIFTY index future
# ════════════════════════════════════════════════════════════════════════════
_BREADTH: dict = {}


def _universe() -> list:
    N._load()
    return sorted(N._S["dv_ix"].keys())          # the 97 bars3y symbols (delivery universe)


def _breadth(d) -> float:
    """Median over the universe of z = (deliv_pct(D) - mean prior 60) / sd prior 60; D = d.date[-1]. Only rows
    dated <= D (asof_swing). NaN when fewer than 60 symbols qualify."""
    day = int(d.date[-1])
    if day in _BREADTH:
        return _BREADTH[day]
    a = N.asof_swing(d)
    zs = []
    for sym in _universe():
        r = _deliv_swing(d, HIST + 1, sym=sym, a=a)
        if r is None:
            continue
        prior, today = _prior_today(r)
        if prior is None or not _ok(today):
            continue
        sd = float(prior.std())
        if sd > 0:
            zs.append((today - float(prior.mean())) / sd)
    v = float(np.median(zs)) if len(zs) >= 60 else np.nan
    _BREADTH[day] = v
    return v


def _mkt(d, p):
    if len(d) < 2 or not (d.c[-1] > d.c[-2] if p.side > 0 else d.c[-1] < d.c[-2]):
        return None
    b = _breadth(d)
    if _ok(b) and b > p.z:
        return Swing(LONG if p.side > 0 else SHORT, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


strategy("r4_delivery_mkt_accum_nifty", "futures", "deliv_breadth",
         "Median NIFTY-100 deliv_pct z-score (own prior 60) > +0.5 on a NIFTY up-day: long NIFTY fut, 3 % stop, hold 5",
         {"z": 0.5, "side": +1, "stop_pct": 3.0, "hold": 5}, horizon="positional", underlying="NIFTY")(_mkt)
strategy("r4_delivery_mkt_distrib_nifty", "futures", "deliv_breadth",
         "Median NIFTY-100 deliv_pct z-score (own prior 60) > +0.5 on a NIFTY down-day: short NIFTY fut, 3 % stop, hold 5",
         {"z": 0.5, "side": -1, "stop_pct": 3.0, "hold": 5}, horizon="positional", underlying="NIFTY")(_mkt)
