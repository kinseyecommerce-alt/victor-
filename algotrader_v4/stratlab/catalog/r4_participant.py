"""
Round-4 NSE-SPECIFIC candidates, family PARTICIPANT POSITIONING (FII / Pro / Client participant-wise OI).
Pre-registered BEFORE any result in scratchpad/r4/participant/PREREGISTERED.txt - parameters are round / textbook
values; do not tune them. Every variant is one more candidate that can pass by luck.

All specs: kind="futures", underlying NIFTY / BANKNIFTY (index future simulated on the index bars), positional,
default NEXT-OPEN fill: decided on the evening of D = d.date[-1], filled at the next session's open.

LOOK-AHEAD. The participant-wise OI file and the F&O bhavcopy dated D are published the evening of D and are usable
from the next session's open. Data is read only through nsedata.asof_swing(d) (before = D+1, rows <= D). On top of
that every result is re-checked by `_rows`: a row dated on/after the fill date (D+1 or later) raises LookAheadError
(the spec is then reported INVALID instead of silently using it), and the latest row must be dated exactly D
(otherwise missing data -> no signal, never forward-filled).
"""
from __future__ import annotations

import numpy as np

from .. import nsedata as N
from ..api import LONG, SHORT, Swing, strategy


# ── look-ahead guard and small helpers ───────────────────────────────────────
def _rows(rows, limit: int, expect: int | None):
    """Guard every nsedata result. limit = first date that must NOT appear (the fill date D+1); expect = the date the
    latest row must have (D), else None = missing / stale data -> no signal."""
    if len(rows) and int(np.max(rows["date"])) >= int(limit):
        raise N.LookAheadError(f"nsedata returned a row dated {int(np.max(rows['date']))} >= {int(limit)}")
    if not len(rows):
        return None
    if expect is not None and int(rows["date"][-1]) != int(expect):
        return None
    return rows


def _asof(d):
    """Asof for a next-open positional decision on day D = d.date[-1]: EOD rows <= D only (fill is the next session).
    Explicit guard that nsedata is never queried for the fill date (the decision is made before that session)."""
    a = N.asof_swing(d)
    D = int(d.date[-1])
    if a.before != D + 1:
        raise N.LookAheadError(f"asof_swing before={a.before} != {D + 1}")
    return a, D + 1, D


def _part(d, n: int):
    """Last n participant-OI rows ending EXACTLY at D (consecutive NSE sessions), or None."""
    a, lim, exp = _asof(d)
    r = _rows(a.participant(n), lim, exp)
    if r is None or len(r) < n:
        return None, a, lim, exp
    return r, a, lim, exp


def _fin(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def _ratio(r, who: str, ins: str = "fut_idx"):
    lo, sh = np.asarray(r[f"{who}_{ins}_long"], float), np.asarray(r[f"{who}_{ins}_short"], float)
    g = lo + sh
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(g > 0, 100.0 * lo / g, np.nan), g


def _opt(r):
    cl, cs = np.asarray(r["FII_opt_idx_call_long"], float), np.asarray(r["FII_opt_idx_call_short"], float)
    pl, ps = np.asarray(r["FII_opt_idx_put_long"], float), np.asarray(r["FII_opt_idx_put_short"], float)
    g = cl + cs + pl + ps
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(g > 0, 100.0 * ((cl - cs) - (pl - ps)) / g, np.nan), g


def _gross_ok(g, i: int, j: int, maxchg: float = 25.0) -> bool:
    """F1: gross OI changed by <= 25 % between rows i and j (roll / lot-size regime guard)."""
    a, b = float(g[i]), float(g[j])
    return _fin(a, b) and a > 0 and abs(b / a - 1.0) <= maxchg / 100.0


def _expiry_days(a, lim: int, exp: int, n: int, kind: str) -> np.ndarray | None:
    """Booleans for the last n NSE sessions ending at D: True where the session is a NIFTY monthly futures expiry
    (kind "M") or a NIFTY options (weekly/monthly) expiry (kind "W"). Rows <= D only (guarded)."""
    if kind == "M":
        r = _rows(a.fut_oi(n, sym="NIFTY"), lim, exp)
    else:
        r = _rows(a.opt_agg(n, sym="NIFTY"), lim, exp)
    if r is None or len(r) < n:
        return None
    ne = np.asarray(r["near_expiry"], float)
    if not np.isfinite(ne).all():
        return None
    return ne.astype(np.int64) == np.asarray(r["date"], np.int64)


def _swing(side: int, p):
    return Swing(side, stop_pct=p.stop_pct, max_hold=p.hold)


# ════════════════════════════════════════════════════════════════════════════
# A. FII index-futures long ratio extremes (contrarian / trend)
# ════════════════════════════════════════════════════════════════════════════
def _fii_lr(contra: bool):
    def fn(d, p):
        r, *_ = _part(d, 1)
        if r is None:
            return None
        lr = float(_ratio(r, "FII")[0][-1])
        if not _fin(lr):
            return None
        if lr < p.lo:
            return _swing(LONG if contra else SHORT, p)
        if lr > p.hi:
            return _swing(SHORT if contra else LONG, p)
        return None
    return fn


_AP = {"lo": 20.0, "hi": 70.0, "hold": 5, "stop_pct": 3.0}
strategy("r4_participant_fii_lr_contra_nifty_5d", "futures", "participant_fii_lr",
         "FII index-futures long ratio (D, NSE EOD) < 20 % -> LONG NIFTY, > 70 % -> SHORT; next open, hold 5, 3 % stop",
         _AP, horizon="positional", underlying="NIFTY")(_fii_lr(True))
strategy("r4_participant_fii_lr_trend_nifty_5d", "futures", "participant_fii_lr",
         "FII index-futures long ratio (D) < 20 % -> SHORT NIFTY, > 70 % -> LONG (follow FIIs); hold 5, 3 % stop",
         _AP, horizon="positional", underlying="NIFTY")(_fii_lr(False))
strategy("r4_participant_fii_lr_contra_bnf_5d", "futures", "participant_fii_lr",
         "FII index-futures long ratio (D) < 20 % -> LONG BANKNIFTY, > 70 % -> SHORT; next open, hold 5, 3 % stop",
         _AP, horizon="positional", underlying="BANKNIFTY")(_fii_lr(True))


# ════════════════════════════════════════════════════════════════════════════
# B. Turn from an FII extreme
# ════════════════════════════════════════════════════════════════════════════
@strategy("r4_participant_fii_lr_turn_nifty_5d", "futures", "participant_fii_turn",
          "FII long ratio was < 20 % within 5 sessions and rose > 3 pp over 3 sessions -> LONG NIFTY (mirror > 70 % / "
          "-3 pp -> SHORT); hold 5, 3 % stop",
          {"lo": 20.0, "hi": 70.0, "look": 5, "lag": 3, "chg": 3.0, "hold": 5, "stop_pct": 3.0},
          horizon="positional", underlying="NIFTY")
def r4_participant_fii_lr_turn_nifty_5d(d, p):
    r, *_ = _part(d, max(p.look, p.lag + 1))
    if r is None:
        return None
    lr = _ratio(r, "FII")[0]
    w = lr[-p.look:]
    if not np.isfinite(w).all() or not _fin(lr[-1 - p.lag]):
        return None
    ch = float(lr[-1] - lr[-1 - p.lag])
    if w.min() < p.lo and ch > p.chg:
        return _swing(LONG, p)
    if w.max() > p.hi and ch < -p.chg:
        return _swing(SHORT, p)
    return None


# ════════════════════════════════════════════════════════════════════════════
# C. / D3 / F3. Daily change in a participant's index-futures long ratio
# ════════════════════════════════════════════════════════════════════════════
def _dlr(who: str, follow: bool):
    def fn(d, p):
        r, a, lim, exp = _part(d, 2)
        if r is None:
            return None
        lr, g = _ratio(r, who)
        if not _fin(lr[-1], lr[-2]) or not _gross_ok(g, -2, -1):
            return None
        m = _expiry_days(a, lim, exp, 1, "M")
        if m is None or m[-1]:
            return None
        ch = float(lr[-1] - lr[-2])
        if ch > p.thr:
            return _swing(LONG if follow else SHORT, p)
        if ch < -p.thr:
            return _swing(SHORT if follow else LONG, p)
        return None
    return fn


strategy("r4_participant_fii_dlr_nifty_1d", "futures", "participant_fii_flow",
         "FII index-futures long ratio change on D > +2 pp -> LONG NIFTY next open (< -2 pp SHORT), exit that close, "
         "1.5 % stop", {"thr": 2.0, "hold": 1, "stop_pct": 1.5},
         horizon="positional", underlying="NIFTY")(_dlr("FII", True))
strategy("r4_participant_fii_dlr_nifty_3d", "futures", "participant_fii_flow",
         "FII index-futures long ratio change on D > +2 pp -> LONG NIFTY next open (< -2 pp SHORT), hold 3, 2.5 % stop",
         {"thr": 2.0, "hold": 3, "stop_pct": 2.5},
         horizon="positional", underlying="NIFTY")(_dlr("FII", True))
strategy("r4_participant_fii_dlr_bnf_1d", "futures", "participant_fii_flow",
         "FII index-futures long ratio change on D > +2 pp -> LONG BANKNIFTY next open (< -2 pp SHORT), exit that "
         "close, 1.5 % stop", {"thr": 2.0, "hold": 1, "stop_pct": 1.5},
         horizon="positional", underlying="BANKNIFTY")(_dlr("FII", True))
strategy("r4_participant_client_dlr_contra_nifty_1d", "futures", "participant_client",
         "Client index-futures long ratio change on D > +2 pp -> SHORT NIFTY next open (< -2 pp LONG), exit that "
         "close, 1.5 % stop", {"thr": 2.0, "hold": 1, "stop_pct": 1.5},
         horizon="positional", underlying="NIFTY")(_dlr("Client", False))


# ════════════════════════════════════════════════════════════════════════════
# D. Client positioning percentile (contrarian)
# ════════════════════════════════════════════════════════════════════════════
def _client_pct(d, p):
    r, *_ = _part(d, p.window)
    if r is None:
        return None
    lr = _ratio(r, "Client")[0]
    if not np.isfinite(lr).all():
        return None
    pct = 100.0 * float(np.mean(lr <= lr[-1]))          # percentile rank of D among D-59..D (incl. D)
    if pct >= p.hi:
        return _swing(SHORT, p)
    if pct <= p.lo:
        return _swing(LONG, p)
    return None


_DP = {"window": 60, "lo": 10.0, "hi": 90.0, "hold": 5, "stop_pct": 3.0}
strategy("r4_participant_client_pct_contra_nifty_5d", "futures", "participant_client",
         "Client index-futures long ratio in the top 10 % of its last 60 sessions -> SHORT NIFTY (bottom 10 % LONG); "
         "hold 5, 3 % stop", _DP, horizon="positional", underlying="NIFTY")(_client_pct)
strategy("r4_participant_client_pct_contra_bnf_5d", "futures", "participant_client",
         "Client index-futures long ratio in the top 10 % of its last 60 sessions -> SHORT BANKNIFTY (bottom 10 % "
         "LONG); hold 5, 3 % stop", _DP, horizon="positional", underlying="BANKNIFTY")(_client_pct)


# ════════════════════════════════════════════════════════════════════════════
# E. FII index options: net put writing vs call writing
# ════════════════════════════════════════════════════════════════════════════
def _opt_state(d, p):
    r, *_ = _part(d, 1)
    if r is None:
        return None
    x = float(_opt(r)[0][-1])
    if not _fin(x):
        return None
    if x > p.thr:
        return _swing(LONG, p)
    if x < -p.thr:
        return _swing(SHORT, p)
    return None


_EP = {"thr": 10.0, "hold": 5, "stop_pct": 3.0}
strategy("r4_participant_fii_opt_state_nifty_5d", "futures", "participant_fii_options",
         "FII bullish index-option exposure (net long calls + net short puts, % of FII gross) > +10 -> LONG NIFTY, "
         "< -10 -> SHORT; hold 5, 3 % stop", _EP, horizon="positional", underlying="NIFTY")(_opt_state)
strategy("r4_participant_fii_opt_state_bnf_5d", "futures", "participant_fii_options",
         "FII bullish index-option exposure > +10 -> LONG BANKNIFTY, < -10 -> SHORT; hold 5, 3 % stop",
         _EP, horizon="positional", underlying="BANKNIFTY")(_opt_state)


@strategy("r4_participant_fii_opt_chg_nifty_1d", "futures", "participant_fii_options",
          "FII bullish index-option exposure change on D > +5 pp -> LONG NIFTY next open (< -5 pp SHORT), exit that "
          "close, 1.5 % stop; skip NIFTY expiry days", {"thr": 5.0, "hold": 1, "stop_pct": 1.5},
          horizon="positional", underlying="NIFTY")
def r4_participant_fii_opt_chg_nifty_1d(d, p):
    r, a, lim, exp = _part(d, 2)
    if r is None:
        return None
    x, g = _opt(r)
    if not _fin(x[-1], x[-2]) or not _gross_ok(g, -2, -1):
        return None
    w, m = _expiry_days(a, lim, exp, 1, "W"), _expiry_days(a, lim, exp, 1, "M")
    if w is None or m is None or w[-1] or m[-1]:
        return None
    ch = float(x[-1] - x[-2])
    if ch > p.thr:
        return _swing(LONG, p)
    if ch < -p.thr:
        return _swing(SHORT, p)
    return None


# ════════════════════════════════════════════════════════════════════════════
# F. Pro vs Client divergence
# ════════════════════════════════════════════════════════════════════════════
def _div(d, p):
    r, *_ = _part(d, 1)
    if r is None:
        return None
    x = float(_ratio(r, "Pro")[0][-1] - _ratio(r, "Client")[0][-1])
    if not _fin(x):
        return None
    if x > p.thr:
        return _swing(LONG, p)
    if x < -p.thr:
        return _swing(SHORT, p)
    return None


_FP = {"thr": 20.0, "hold": 5, "stop_pct": 3.0}
strategy("r4_participant_pro_client_div_nifty_5d", "futures", "participant_pro_client",
         "Pro minus Client index-futures long ratio (D) > +20 pp -> LONG NIFTY, < -20 pp -> SHORT; hold 5, 3 % stop",
         _FP, horizon="positional", underlying="NIFTY")(_div)
strategy("r4_participant_pro_client_div_bnf_5d", "futures", "participant_pro_client",
         "Pro minus Client index-futures long ratio (D) > +20 pp -> LONG BANKNIFTY, < -20 pp -> SHORT; hold 5, 3 % stop",
         _FP, horizon="positional", underlying="BANKNIFTY")(_div)


@strategy("r4_participant_pro_client_dchg_nifty_1d", "futures", "participant_pro_client",
          "On D Pro long ratio +2 pp and Client -2 pp -> LONG NIFTY next open (mirror -> SHORT), exit that close, "
          "1.5 % stop", {"thr": 2.0, "hold": 1, "stop_pct": 1.5}, horizon="positional", underlying="NIFTY")
def r4_participant_pro_client_dchg_nifty_1d(d, p):
    r, a, lim, exp = _part(d, 2)
    if r is None:
        return None
    lp, gp = _ratio(r, "Pro")
    lc, gc = _ratio(r, "Client")
    if not _fin(lp[-1], lp[-2], lc[-1], lc[-2]) or not (_gross_ok(gp, -2, -1) and _gross_ok(gc, -2, -1)):
        return None
    m = _expiry_days(a, lim, exp, 1, "M")
    if m is None or m[-1]:
        return None
    dp, dc = float(lp[-1] - lp[-2]), float(lc[-1] - lc[-2])
    if dp > p.thr and dc < -p.thr:
        return _swing(LONG, p)
    if dp < -p.thr and dc > p.thr:
        return _swing(SHORT, p)
    return None


# ════════════════════════════════════════════════════════════════════════════
# G. FII stock-futures positioning -> broad market (NIFTY)
# ════════════════════════════════════════════════════════════════════════════
def _stk(d, p):
    r, a, lim, exp = _part(d, p.window)
    if r is None:
        return None
    s, g = _ratio(r, "FII", "fut_stk")
    if not _fin(s[-1], s[0]) or not _gross_ok(g, 0, -1):
        return None
    m = _expiry_days(a, lim, exp, p.window, "M")
    if m is None or m.any():
        return None
    ch = float(s[-1] - s[0])
    if ch > p.thr:
        return _swing(LONG, p)
    if ch < -p.thr:
        return _swing(SHORT, p)
    return None


strategy("r4_participant_fii_stk_dlr_nifty_1d", "futures", "participant_fii_stock_fut",
         "FII stock-futures long ratio change on D > +1 pp -> LONG NIFTY next open (< -1 pp SHORT), exit that close, "
         "1.5 % stop", {"window": 2, "thr": 1.0, "hold": 1, "stop_pct": 1.5},
         horizon="positional", underlying="NIFTY")(_stk)
strategy("r4_participant_fii_stk_dlr5_nifty_5d", "futures", "participant_fii_stock_fut",
         "FII stock-futures long ratio change D-4..D > +3 pp -> LONG NIFTY next open (< -3 pp SHORT), hold 5, 3 % stop",
         {"window": 5, "thr": 3.0, "hold": 5, "stop_pct": 3.0},
         horizon="positional", underlying="NIFTY")(_stk)
