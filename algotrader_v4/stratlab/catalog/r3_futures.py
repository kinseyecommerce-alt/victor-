"""
Round-3 FUTURES / SHORT-SIDE candidates (index futures on NIFTY / BANKNIFTY minute bars, and stock futures).
Pre-registered BEFORE any result in scratchpad/r3/futures/PREREGISTERED.txt - parameters are textbook / round values;
do not tune them. Every variant is one more candidate that can pass by luck.

Index intraday specs are vector style (audited for look-ahead); positional specs receive a DailyView cut at the
decision day (or at 15:20 of the decision day with decision_time="15:20").
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from ..api import LONG, SHORT, Entry, Swing, first_true, hhmm, strategy
from ..indicators import atr as _atr, ema as _ema, rsi as _rsi, sma as _sma

# Known unadjusted corporate-action gaps (same list as stratlab.ltr.CA_GAPS; copied so the catalogue does not
# import the basket module). A stock signal is skipped when one lies inside the last 250 sessions of its history.
_CA_GAPS = {("SIEMENS", 20250407), ("VEDL", 20260430)}


# ── helpers ─────────────────────────────────────────────────────────────────
def _at(s, hm: str) -> int:
    """Index of the decision bar for a time-triggered rule, or -1 if it is not visible yet."""
    i = s.idx(hhmm(hm))
    return i if i < len(s) else -1


def _ok(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def _date(d: int) -> _dt.date:
    d = int(d)
    return _dt.date(d // 10000, (d // 100) % 100, d % 100)


def _ca_clean(d, n: int = 250) -> bool:
    sym = getattr(d, "symbol", None)
    if not sym:
        return True
    recent = d.date[-n:]
    for s_, day in _CA_GAPS:
        if s_ == sym and recent[0] <= day <= recent[-1]:
            return False
    return True


def _atr14(d) -> float:
    return float(_atr(d.h, d.l, d.c, 14)[-1])


# ════════════════════════════════════════════════════════════════════════════
# A. Overnight vs intraday split: the intraday half, short (index intraday)
# ════════════════════════════════════════════════════════════════════════════
def _id_short(s, p):
    if len(s) < 1:
        return []
    c0 = float(s.c[0])
    return [Entry(0, SHORT, stop=c0 * (1 + p.stop_pct / 100.0))]


strategy("r3_futures_nifty_id_short", "futures", "ovn_split",
         "Every session short NIFTY at the 09:16 open (intraday half of the overnight/intraday split), 1 % stop, exit 15:15",
         {"stop_pct": 1.0}, horizon="intraday", underlying="NIFTY")(_id_short)
strategy("r3_futures_bnf_id_short", "futures", "ovn_split",
         "Every session short BANKNIFTY at the 09:16 open (intraday half of the overnight split), 1 % stop, exit 15:15",
         {"stop_pct": 1.0}, horizon="intraday", underlying="BANKNIFTY")(_id_short)


# ════════════════════════════════════════════════════════════════════════════
# B. Index gap fade by gap size (index intraday)
# ════════════════════════════════════════════════════════════════════════════
def _gapfade(s, p):
    if len(s) < 1 or not _ok(s.pdc, s.gap):
        return []
    g = s.gap
    if not (p.gap_min / 100.0 <= abs(g) <= p.gap_max / 100.0):
        return []
    o0 = float(s.o[0])
    dist = abs(o0 - s.pdc)
    if g > 0:
        return [Entry(0, SHORT, stop=o0 + dist, target=float(s.pdc))]
    return [Entry(0, LONG, stop=o0 - dist, target=float(s.pdc))]


strategy("r3_futures_nifty_gapfade", "futures", "gap_fade",
         "NIFTY 0.25-1.0 % opening gap: fade at the 09:16 open, target previous close, stop one gap beyond the open",
         {"gap_min": 0.25, "gap_max": 1.0}, horizon="intraday", underlying="NIFTY")(_gapfade)
strategy("r3_futures_bnf_gapfade", "futures", "gap_fade",
         "BANKNIFTY 0.25-1.0 % opening gap: fade at the 09:16 open, target previous close, stop one gap beyond the open",
         {"gap_min": 0.25, "gap_max": 1.0}, horizon="intraday", underlying="BANKNIFTY")(_gapfade)


# ════════════════════════════════════════════════════════════════════════════
# C. Index opening range in the direction of the daily trend (index intraday)
# ════════════════════════════════════════════════════════════════════════════
def _orb_trend(s, p):
    hi, lo, k = s.orb(p.minutes)
    if k >= len(s) or not hi > lo or not _ok(s.pdc, s.dsma20):
        return []
    end = s.idx(hhmm(p.last))
    if s.pdc > s.dsma20:
        i = first_true(s.c > hi, k, end)
        return [Entry(i, LONG, stop=lo, rr=p.rr)] if i >= 0 else []
    i = first_true(s.c < lo, k, end)
    return [Entry(i, SHORT, stop=hi, rr=p.rr)] if i >= 0 else []


strategy("r3_futures_nifty_orb30_trend", "futures", "orb_index",
         "NIFTY 30-min range break only in the daily-trend direction (pdc vs 20-day SMA), stop other side, 2R, until 13:00",
         {"minutes": 30, "rr": 2.0, "last": "13:00", "sma": 20}, horizon="intraday", underlying="NIFTY")(_orb_trend)
strategy("r3_futures_bnf_orb30_trend", "futures", "orb_index",
         "BANKNIFTY 30-min range break only in the daily-trend direction (pdc vs 20-day SMA), stop other side, 2R",
         {"minutes": 30, "rr": 2.0, "last": "13:00", "sma": 20}, horizon="intraday", underlying="BANKNIFTY")(_orb_trend)


# ════════════════════════════════════════════════════════════════════════════
# D. Index VWAP +/- 2 sd fade (index intraday)
# ════════════════════════════════════════════════════════════════════════════
def _vwap_fade(s, p):
    a, b = s.idx(hhmm("10:00")), s.idx(hhmm("14:30"))
    if a >= len(s):
        return []
    up = s.vwap + p.k * s.vsd
    dn = s.vwap - p.k * s.vsd
    out = []
    i = first_true((s.c > up) & (s.vsd > 0), a, b)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=float(s.c[i]) * (1 + p.stop_pct / 100.0), target=float(s.vwap[i])))
    j = first_true((s.c < dn) & (s.vsd > 0), a, b)
    if j >= 0:
        out.append(Entry(j, LONG, stop=float(s.c[j]) * (1 - p.stop_pct / 100.0), target=float(s.vwap[j])))
    return out


strategy("r3_futures_nifty_vwap2sd_fade", "futures", "vwap_fade_index",
         "NIFTY 10:00-14:30 first close beyond VWAP +/- 2 sd: fade to VWAP, stop 0.4 % beyond the signal close",
         {"k": 2.0, "stop_pct": 0.4}, horizon="intraday", underlying="NIFTY")(_vwap_fade)
strategy("r3_futures_bnf_vwap2sd_fade", "futures", "vwap_fade_index",
         "BANKNIFTY 10:00-14:30 first close beyond VWAP +/- 2 sd: fade to VWAP, stop 0.4 % beyond the signal close",
         {"k": 2.0, "stop_pct": 0.4}, horizon="intraday", underlying="BANKNIFTY")(_vwap_fade)


# ════════════════════════════════════════════════════════════════════════════
# E. Expiry eve (index positional)
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_futures_nifty_expiry_eve_long", "futures", "expiry",
          "Session before the NIFTY weekly expiry weekday (Wed->Thu, from Sep 2025 Mon->Tue): long at 15:20, exit expiry close",
          {"stop_pct": 2.0}, horizon="positional", decision_time="15:20", underlying="NIFTY")
def r3_futures_nifty_expiry_eve_long(d, p):
    today = _date(d.date[-1])
    exp_wd = 1 if today >= _dt.date(2025, 9, 1) else 3          # Tuesday / Thursday (options_replay rule)
    if today.weekday() != exp_wd - 1:
        return None
    return Swing(LONG, stop_pct=p.stop_pct, max_hold=1)


# ════════════════════════════════════════════════════════════════════════════
# F. Weekend effect (index positional)
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_futures_nifty_weekend_short", "futures", "day_of_week",
          "Every Friday short NIFTY at 15:20, exit at the next session's (Monday's) close, 2 % disaster stop",
          {"stop_pct": 2.0}, horizon="positional", decision_time="15:20", underlying="NIFTY")
def r3_futures_nifty_weekend_short(d, p):
    if _date(d.date[-1]).weekday() != 4:
        return None
    return Swing(SHORT, stop_pct=p.stop_pct, max_hold=1)


# ════════════════════════════════════════════════════════════════════════════
# G. Index short-horizon mean reversion (positional)
# ════════════════════════════════════════════════════════════════════════════
def _ibs(d, p):
    h, l, c = float(d.h[-1]), float(d.l[-1]), float(d.c[-1])
    if not h > l:
        return None
    x = (c - l) / (h - l)
    if x < p.lo:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=1)
    if x > p.hi:
        return Swing(SHORT, stop_pct=p.stop_pct, max_hold=1)
    return None


strategy("r3_futures_nifty_ibs_ls", "futures", "ibs",
         "NIFTY at 15:20: IBS of the day so far < 0.2 long, > 0.8 short, exit next session's close, 2 % disaster stop",
         {"lo": 0.2, "hi": 0.8, "stop_pct": 2.0}, horizon="positional", decision_time="15:20", underlying="NIFTY")(_ibs)
strategy("r3_futures_bnf_ibs_ls", "futures", "ibs",
         "BANKNIFTY at 15:20: IBS of the day so far < 0.2 long, > 0.8 short, exit next session's close, 2 % stop",
         {"lo": 0.2, "hi": 0.8, "stop_pct": 2.0}, horizon="positional", decision_time="15:20",
         underlying="BANKNIFTY")(_ibs)


@strategy("r3_futures_nifty_bigday_fade", "futures", "big_day_reversal",
          "After a NIFTY day moving more than 1 x ATR14 close-to-close, fade it next open, stop 2 ATR, hold 3 sessions",
          {"k": 1.0, "stop_atr": 2.0, "hold": 3}, horizon="positional", underlying="NIFTY")
def r3_futures_nifty_bigday_fade(d, p):
    if len(d) < 30:
        return None
    a = _atr14(d)
    c, pc = float(d.c[-1]), float(d.c[-2])
    if not _ok(a) or abs(c - pc) <= p.k * a:
        return None
    if c > pc:
        return Swing(SHORT, stop=c + p.stop_atr * a, max_hold=p.hold)
    return Swing(LONG, stop=c - p.stop_atr * a, max_hold=p.hold)


@strategy("r3_futures_nifty_rsi2_ls", "futures", "rsi2_index",
          "NIFTY RSI(2) < 10 long, > 90 short at the next open, 3 % stop, hold 5 sessions",
          {"lo": 10, "hi": 90, "stop_pct": 3.0, "hold": 5}, horizon="positional", underlying="NIFTY")
def r3_futures_nifty_rsi2_ls(d, p):
    if len(d) < 20:
        return None
    r = float(_rsi(d.c, 2)[-1])
    if not _ok(r):
        return None
    if r < p.lo:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    if r > p.hi:
        return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# H. Index trend following (positional)
# ════════════════════════════════════════════════════════════════════════════
def _ema_state(d, p):
    if len(d) < p.slow + 10:
        return None
    f, s_ = float(_ema(d.c, p.fast)[-1]), float(_ema(d.c, p.slow)[-1])
    a, c = _atr14(d), float(d.c[-1])
    if not _ok(f, s_, a):
        return None
    if f > s_:
        return Swing(LONG, stop=c - p.stop_atr * a, max_hold=p.hold)
    return Swing(SHORT, stop=c + p.stop_atr * a, max_hold=p.hold)


strategy("r3_futures_nifty_ema_state", "futures", "trend_index",
         "NIFTY whenever flat: EMA20 > EMA50 long else short at the next open, stop 2 x ATR14, hold 10 sessions",
         {"fast": 20, "slow": 50, "stop_atr": 2.0, "hold": 10}, horizon="positional", underlying="NIFTY")(_ema_state)
strategy("r3_futures_bnf_ema_state", "futures", "trend_index",
         "BANKNIFTY whenever flat: EMA20 > EMA50 long else short at the next open, stop 2 x ATR14, hold 10 sessions",
         {"fast": 20, "slow": 50, "stop_atr": 2.0, "hold": 10}, horizon="positional",
         underlying="BANKNIFTY")(_ema_state)


# ════════════════════════════════════════════════════════════════════════════
# I. VIX regime (index positional)
# ════════════════════════════════════════════════════════════════════════════
def _vix_ok(d, n: int):
    if d.vix is None or len(d) < n + 2:
        return None
    v = np.asarray(d.vix[-n:], float)
    if not np.isfinite(v).all():
        return None
    return float(v[-1]), float(v.mean())


def _vixspike(d, p):
    x = _vix_ok(d, p.sma)
    if x is None or len(d) <= p.lookback:
        return None
    if x[0] > x[1] and d.c[-1] < d.c[-1 - p.lookback]:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


strategy("r3_futures_nifty_vixspike_long", "futures", "vix_regime",
         "India VIX above its 20-day SMA and NIFTY below its close 5 sessions ago: long next open, 3 % stop, hold 5",
         {"sma": 20, "lookback": 5, "stop_pct": 3.0, "hold": 5}, horizon="positional", underlying="NIFTY")(_vixspike)
strategy("r3_futures_bnf_vixspike_long", "futures", "vix_regime",
         "India VIX above its 20-day SMA and BANKNIFTY below its close 5 sessions ago: long next open, 3 % stop, hold 5",
         {"sma": 20, "lookback": 5, "stop_pct": 3.0, "hold": 5}, horizon="positional",
         underlying="BANKNIFTY")(_vixspike)


@strategy("r3_futures_nifty_vixlow_mom", "futures", "vix_regime",
          "India VIX below its 20-day SMA: trade the sign of NIFTY's 20-session return, stop 2 x ATR14, hold 5",
          {"sma": 20, "mom": 20, "stop_atr": 2.0, "hold": 5}, horizon="positional", underlying="NIFTY")
def r3_futures_nifty_vixlow_mom(d, p):
    x = _vix_ok(d, p.sma)
    if x is None or len(d) < max(p.mom + 1, 30) or not x[0] < x[1]:
        return None
    a, c = _atr14(d), float(d.c[-1])
    if not _ok(a):
        return None
    if c > d.c[-1 - p.mom]:
        return Swing(LONG, stop=c - p.stop_atr * a, max_hold=p.hold)
    if c < d.c[-1 - p.mom]:
        return Swing(SHORT, stop=c + p.stop_atr * a, max_hold=p.hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# J. BANKNIFTY vs NIFTY spread, single leg on BANKNIFTY
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_futures_bnf_laggard_1100", "futures", "spread_bnf_nifty",
          "At 11:00 BANKNIFTY return since pdc minus NIFTY's <= -0.4 % long BANKNIFTY, >= +0.4 % short; 0.5 % stop, exit 15:15",
          {"thr": 0.4, "stop_pct": 0.5}, horizon="intraday", underlying="BANKNIFTY", needs_index=True)
def r3_futures_bnf_laggard_1100(s, p):
    i = _at(s, "11:00")
    if i < 0 or s.nifty is None or not _ok(s.pdc, s.nifty_pdc) or not np.isfinite(s.nifty[i]):
        return []
    rel = (s.c[i] / s.pdc - 1.0) - (s.nifty[i] / s.nifty_pdc - 1.0)
    c = float(s.c[i])
    if rel <= -p.thr / 100.0:
        return [Entry(i, LONG, stop=c * (1 - p.stop_pct / 100.0))]
    if rel >= p.thr / 100.0:
        return [Entry(i, SHORT, stop=c * (1 + p.stop_pct / 100.0))]
    return []


@strategy("r3_futures_bnf_spread5d", "futures", "spread_bnf_nifty",
          "BANKNIFTY 5-session return minus NIFTY's <= -1.5 % long BANKNIFTY, >= +1.5 % short, next open, 3 % stop, hold 5",
          {"thr": 1.5, "n": 5, "stop_pct": 3.0, "hold": 5}, horizon="positional", underlying="BANKNIFTY")
def r3_futures_bnf_spread5d(d, p):
    if d.nifty is None or len(d) <= p.n:
        return None
    n0, n1 = float(d.nifty[-1 - p.n]), float(d.nifty[-1])
    if not _ok(n0, n1) or n0 <= 0:
        return None
    rel = (d.c[-1] / d.c[-1 - p.n] - 1.0) - (n1 / n0 - 1.0)
    if rel <= -p.thr / 100.0:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
    if rel >= p.thr / 100.0:
        return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# K. Stock-futures breakdown shorts (positional)
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_futures_stk_50dlow_short_niftydown", "futures", "breakdown_short",
          "Stock closes below its 50-session low while NIFTY is below its 50-day SMA: short next open, stop 2 ATR, hold 20",
          {"n": 50, "nifty_sma": 50, "stop_atr": 2.0, "hold": 20}, horizon="positional")
def r3_futures_stk_50dlow_short_niftydown(d, p):
    if len(d) < p.n + 1 or d.nifty is None or not _ca_clean(d):
        return None
    c = float(d.c[-1])
    if not c < float(d.l[-p.n - 1:-1].min()):
        return None
    nf = np.asarray(d.nifty[-p.nifty_sma:], float)
    if len(nf) < p.nifty_sma or not np.isfinite(nf).all() or not nf[-1] < nf.mean():
        return None
    a = _atr14(d)
    return Swing(SHORT, stop=c + p.stop_atr * a, max_hold=p.hold) if _ok(a) else None


@strategy("r3_futures_stk_52wlow_short", "futures", "breakdown_short",
          "Stock close within 2 % of its 250-session low and below its 50-day SMA: short next open, stop 2 ATR, hold 20",
          {"n": 250, "near": 2.0, "sma": 50, "stop_atr": 2.0, "hold": 20}, horizon="positional")
def r3_futures_stk_52wlow_short(d, p):
    if len(d) < p.n or not _ca_clean(d):
        return None
    c = float(d.c[-1])
    if not c <= (1 + p.near / 100.0) * float(d.l[-p.n:].min()):
        return None
    if not c < float(d.c[-p.sma:].mean()):
        return None
    a = _atr14(d)
    return Swing(SHORT, stop=c + p.stop_atr * a, max_hold=p.hold) if _ok(a) else None


# ════════════════════════════════════════════════════════════════════════════
# L. Stock-futures overbought shorts (positional)
# ════════════════════════════════════════════════════════════════════════════
def _rsi2_ob(d, p, nifty_filter: bool):
    if len(d) < p.sma or not _ca_clean(d):
        return None
    c = float(d.c[-1])
    if not c < float(d.c[-p.sma:].mean()):
        return None
    r = float(_rsi(d.c, 2)[-1])
    if not (_ok(r) and r > p.thr):
        return None
    if nifty_filter:
        if d.nifty is None:
            return None
        nf = np.asarray(d.nifty[-p.nifty_sma:], float)
        if len(nf) < p.nifty_sma or not np.isfinite(nf).all() or not nf[-1] < nf.mean():
            return None
    a = _atr14(d)
    return Swing(SHORT, stop=c + p.stop_atr * a, max_hold=p.hold) if _ok(a) else None


@strategy("r3_futures_stk_rsi2_ob_short", "futures", "overbought_short",
          "RSI(2) > 95 with close below the 200-day SMA: short the stock future next open, 3 ATR stop, hold 5",
          {"thr": 95, "sma": 200, "stop_atr": 3.0, "hold": 5}, horizon="positional")
def r3_futures_stk_rsi2_ob_short(d, p):
    return _rsi2_ob(d, p, False)


@strategy("r3_futures_stk_rsi2_ob_short_nifty", "futures", "overbought_short",
          "RSI(2) > 95, close below 200-day SMA and NIFTY below its 50-day SMA: short next open, 3 ATR stop, hold 5",
          {"thr": 95, "sma": 200, "nifty_sma": 50, "stop_atr": 3.0, "hold": 5}, horizon="positional")
def r3_futures_stk_rsi2_ob_short_nifty(d, p):
    return _rsi2_ob(d, p, True)


@strategy("r3_futures_stk_keltner_ext_short", "futures", "overbought_short",
          "Close above 20-day SMA + 3 x ATR14 with RSI14 > 80: short the stock future next open, 2 ATR stop, hold 5",
          {"sma": 20, "k": 3.0, "rsi_thr": 80, "stop_atr": 2.0, "hold": 5}, horizon="positional")   # RSI period fixed at 14 (catalogue: <= 5 params)
def r3_futures_stk_keltner_ext_short(d, p):
    if len(d) < 40 or not _ca_clean(d):
        return None
    c, a = float(d.c[-1]), _atr14(d)
    if not _ok(a) or not c > float(d.c[-p.sma:].mean()) + p.k * a:
        return None
    r = float(_rsi(d.c, 14)[-1])
    if not (_ok(r) and r > p.rsi_thr):
        return None
    return Swing(SHORT, stop=c + p.stop_atr * a, max_hold=p.hold)


# ════════════════════════════════════════════════════════════════════════════
# M. Volume + gap drift (post-earnings-drift proxy, positional)
# ════════════════════════════════════════════════════════════════════════════
def _gapvol(d, p, side):
    if len(d) < 22 or not _ca_clean(d):
        return None
    o, c, pc = float(d.o[-1]), float(d.c[-1]), float(d.c[-2])
    g = o / pc - 1.0
    avg = float(np.mean(d.v[-21:-1]))
    if not avg > 0 or not d.v[-1] >= p.vol * avg:
        return None
    a = _atr14(d)
    if not _ok(a):
        return None
    if side == SHORT and g <= -p.gap / 100.0 and c <= o:
        return Swing(SHORT, stop=c + p.stop_atr * a, max_hold=p.hold)
    if side == LONG and g >= p.gap / 100.0 and c >= o:
        return Swing(LONG, stop=c - p.stop_atr * a, max_hold=p.hold)
    return None


@strategy("r3_futures_stk_gapdown_vol_drift", "futures", "gap_volume_drift",
          "Gap down >= 3 % on >= 3x 20-day volume with close <= open: short the stock future next open, 3 ATR stop, hold 10",
          {"gap": 3.0, "vol": 3.0, "stop_atr": 3.0, "hold": 10}, horizon="positional")
def r3_futures_stk_gapdown_vol_drift(d, p):
    return _gapvol(d, p, SHORT)


@strategy("r3_futures_stk_gapup_vol_drift", "futures", "gap_volume_drift",
          "Gap up >= 3 % on >= 3x 20-day volume with close >= open: buy the stock future next open, 3 ATR stop, hold 10",
          {"gap": 3.0, "vol": 3.0, "stop_atr": 3.0, "hold": 10}, horizon="positional")
def r3_futures_stk_gapup_vol_drift(d, p):
    return _gapvol(d, p, LONG)


# ════════════════════════════════════════════════════════════════════════════
# N. Lottery (MAX) short (positional)
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_futures_stk_max_lottery_short", "futures", "lottery_short",
          "Max one-day return in the last 20 sessions >= 6 % and close below the 20-day SMA: short next open, 2 ATR, hold 10",
          {"max_pct": 6.0, "n": 20, "stop_atr": 2.0, "hold": 10}, horizon="positional")
def r3_futures_stk_max_lottery_short(d, p):
    if len(d) < 40 or not _ca_clean(d):
        return None
    c = np.asarray(d.c[-p.n - 1:], float)
    mx = float(np.max(c[1:] / c[:-1] - 1.0))
    last = float(c[-1])
    if not mx >= p.max_pct / 100.0 or not last < float(c[1:].mean()):
        return None
    a = _atr14(d)
    return Swing(SHORT, stop=last + p.stop_atr * a, max_hold=p.hold) if _ok(a) else None


# ════════════════════════════════════════════════════════════════════════════
# O. Turn of the month (stock futures, positional)
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_futures_stk_tom_long", "futures", "calendar_tom",
          "First session with day-of-month >= 25: buy the stock future at 15:20, 5 % stop, hold 5 sessions",
          {"dom": 25, "stop_pct": 5.0, "hold": 5}, horizon="positional", decision_time="15:20")
def r3_futures_stk_tom_long(d, p):
    if len(d) < 2 or not _ca_clean(d):
        return None
    t, y = _date(d.date[-1]), _date(d.date[-2])
    if t.day < p.dom:
        return None
    if y.year == t.year and y.month == t.month and y.day >= p.dom:
        return None
    return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.hold)
