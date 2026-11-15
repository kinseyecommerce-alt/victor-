"""
Round 6 — GitHub-sourced strategies, re-implemented from the source code with the known source bugs fixed, run
once with no tuning. Pre-registration: stratlab/prereg/github_v1.json (byte-identical to
<scratchpad>/PREREGISTERED_GITHUB.json), written before any run: every id below with its source file, the rule as
coded, the bug fixes, the ONE plain default for each unspecified field and the eras. Changing anything here makes it
a new, unregistered strategy.

Per-symbol strategies only (intraday index futures and daily stocks / index futures). The monthly / weekly / daily
cross-sectional portfolios of this round live in stratlab/r6_portfolio.py; formulas in stratlab/r6_ind.py.

Conventions (stated in the pre-registration):
  * Index futures = the lab's index futures (spot index bars stand in for the future, 1 lot, 2026 lot sizes,
    1 bps slippage). Stocks: the lab universe (2023-26: ~97 NIFTY-100 names; 2012-23: point-in-time top-100).
  * Daily rules decide at the close and fill at the next open; rule exits fill at the next open (Swing.exit_fn);
    an opposite entry decided at the exit close fills at the same open (stop-and-reverse).
  * Intraday: decisions at candle close (Kite-aligned candles, stratlab.kite_tf), fill at the next 1-minute open,
    15:15 square-off.
"""
from __future__ import annotations

import numpy as np

from ..api import LONG, SHORT, Entry, Swing, strategy
from .. import kite_ind as K
from .. import r6_ind as R
from ..kite_tf import NOSTOP, SQ_OFF, today

MANY = 75
FOREVER = 100_000                   # positional "no max hold": closed by its exit rule or at the period end


# ══════════════════════════════════════════════════════════════════════════
# 6. R-Breaker (letianzj QuantResearch backtest/r_breaker.py) — NIFTY / BANKNIFTY futures, 5-minute candles
# ══════════════════════════════════════════════════════════════════════════
def _rbreaker(s, p):
    S, g, sig = today(s, p.tf)
    if not len(g) or not (np.isfinite(s.pdh) and np.isfinite(s.pdl) and np.isfinite(s.pdc)):
        return []
    lv = R.rbreaker_levels(float(s.pdh), float(s.pdl), float(s.pdc))
    c = S.c[g]
    dh = np.maximum.accumulate(S.h[g])
    dl = np.minimum.accumulate(S.l[g])
    enter, flip = R.rbreaker_day(c, dh, dl, lv)
    n = len(s.t)
    xl = np.zeros(n, bool)                        # long closed (flipped to short) at these decision bars
    xs = np.zeros(n, bool)
    for j in np.flatnonzero(flip):
        if 0 <= sig[j] < n:
            (xl if enter[j] < 0 else xs)[sig[j]] = True
    out = []
    for j in np.flatnonzero(enter):
        sb = int(sig[j])
        if not (0 <= sb < n) or s.t[sb] + 1 >= SQ_OFF:
            continue
        side = int(enter[j])
        out.append(Entry(sb, side, stop=float(c[j]) * (1.0 - side * NOSTOP), exit_sig=xl if side > 0 else xs))
    return out


@strategy("r6_gh_rbreaker_nifty", "futures", "gh_letianzj",
          "letianzj r_breaker.py on NIFTY 5m: prev-day H/L/C pivots; close>r3 long, <s3 short; flips (hi>r2 & c<r1 | c<s3 ...); 15:15",
          {"tf": 5}, horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r6_gh_rbreaker_nifty(s, p):
    return _rbreaker(s, p)


@strategy("r6_gh_rbreaker_banknifty", "futures", "gh_letianzj",
          "letianzj r_breaker.py on BANKNIFTY 5m: prev-day H/L/C pivots; close>r3 long, <s3 short; flips (hi>r2 & c<r1 | c<s3 ...); 15:15",
          {"tf": 5}, horizon="intraday", underlying="BANKNIFTY", max_per_day=MANY)
def r6_gh_rbreaker_banknifty(s, p):
    return _rbreaker(s, p)


# ══════════════════════════════════════════════════════════════════════════
# 7. 5-EMA alert candle (Screeni-py Screener.monitorFiveEma + the original method): 5m shorts, 15m longs
# ══════════════════════════════════════════════════════════════════════════
def _ema5(S):
    return K.ema_sma_seed(S.c, 5)


def _five_ema_leg(s, tf: int, side: int, gap: float, rr: float) -> list:
    S, g, sig = today(s, tf)
    if not len(g):
        return []
    e = S.get("r6_ema5", _ema5)
    n = len(s.t)
    t = np.asarray(s.t)
    out = []
    for jl, j in enumerate(g):
        ej = round(float(e[j]), 2) if np.isfinite(e[j]) else np.nan
        lo_, hi_ = round(float(S.l[j]), 2), round(float(S.h[j]), 2)
        if not np.isfinite(ej):
            continue
        if side == SHORT:
            alert, level, stop = (lo_ > ej and lo_ - ej > gap), float(S.l[j]), float(S.h[j])
        else:
            alert, level, stop = (hi_ < ej and ej - hi_ > gap), float(S.h[j]), float(S.l[j])
        if not alert:
            continue
        a = int(sig[jl]) + 1                      # the next candle's minutes: t in [end_j, end_j + tf)
        if a >= n:
            continue
        w_end = int(S.end[j]) + tf
        b = a + int(np.searchsorted(t[a:], w_end))
        if b <= a:
            continue
        hit = (s.l[a:b] < level) if side == SHORT else (s.h[a:b] > level)
        if not hit.any():
            continue
        m = a + int(np.argmax(hit))
        if t[m] + 1 >= SQ_OFF:
            continue
        out.append(Entry(m, side, stop=stop, rr=rr))
    return out


def _five_ema(s, p):
    return sorted(_five_ema_leg(s, p.tf_short, SHORT, p.gap, p.rr) + _five_ema_leg(s, p.tf_long, LONG, p.gap, p.rr),
                  key=lambda e: e.i)


_FIVE = {"ema": 5, "tf_short": 5, "tf_long": 15, "gap": 0.5, "rr": 3.0}


@strategy("r6_gh_ema5_nifty", "futures", "gh_screenipy",
          "5-EMA alert candle on NIFTY: 5m candle wholly above EMA5 -> short on break of its low, SL its high; 15m below -> long; 1:3",
          dict(_FIVE), horizon="intraday", underlying="NIFTY", max_per_day=MANY)
def r6_gh_ema5_nifty(s, p):
    return _five_ema(s, p)


@strategy("r6_gh_ema5_banknifty", "futures", "gh_screenipy",
          "5-EMA alert candle on BANKNIFTY: 5m candle wholly above EMA5 -> short on break of its low, SL its high; 15m below -> long; 1:3",
          dict(_FIVE), horizon="intraday", underlying="BANKNIFTY", max_per_day=MANY)
def r6_gh_ema5_banknifty(s, p):
    return _five_ema(s, p)


# ══════════════════════════════════════════════════════════════════════════
# 8. TD Sequential 9 with perfection (freqtrade-strategies berlinguyinca/TDSequentialStrategy.py), long only
# ══════════════════════════════════════════════════════════════════════════
def _td_exit(v):
    _sb, ss, _xl, xh = R.td_at(v.c, v.h, v.l)
    return bool(xh or ss >= 9)


def _td(d, p):
    if len(d.c) < 13:
        return None
    sb, _ss, xl, _xh = R.td_at(d.c, d.h, d.l)
    if sb >= 9 and xl:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=FOREVER, exit_fn=_td_exit)
    return None


_TD = {"setup": 9, "lag": 4, "stop_pct": 5.0}


@strategy("r6_gh_td9_stock", "swing", "gh_freqtrade",
          "freqtrade TDSequential (daily): buy setup >= 9 with perfection; exit perfected sell count 8+ or sell count 9; -5 % stop; CNC",
          dict(_TD))
def r6_gh_td9_stock(d, p):
    return _td(d, p)


@strategy("r6_gh_td9_nifty", "futures", "gh_freqtrade",
          "freqtrade TDSequential on NIFTY daily futures: buy setup >= 9 with perfection; exit perfected sell 8+ / sell 9; -5 % stop",
          dict(_TD), horizon="positional", underlying="NIFTY")
def r6_gh_td9_nifty(d, p):
    return _td(d, p)


@strategy("r6_gh_td9_banknifty", "futures", "gh_freqtrade",
          "freqtrade TDSequential on BANKNIFTY daily futures: buy setup >= 9 with perfection; exit perfected sell 8+ / sell 9; -5 % stop",
          dict(_TD), horizon="positional", underlying="BANKNIFTY")
def r6_gh_td9_banknifty(d, p):
    return _td(d, p)


# ══════════════════════════════════════════════════════════════════════════
# 10. mr_wings (pysystemtrade systems/provided/rules/mr_wings.py), binary position, daily index futures
# ══════════════════════════════════════════════════════════════════════════
def _mrw_side(v) -> int:
    f = R.mr_wings(v.c)[-1]
    return 0 if not np.isfinite(f) or f == 0 else (1 if f > 0 else -1)


def _mrw(d, p):
    if len(d.c) < 20:
        return None
    side = _mrw_side(d)
    if side == 0:
        return None
    return Swing(side, max_hold=FOREVER, exit_fn=(lambda v, sd=side: _mrw_side(v) != sd))


_MRW = {"fast": 4, "slow": 16, "k_std": 3.0, "vol_span": 35, "std_window": 5000}


@strategy("r6_gh_mrwings_nifty", "futures", "gh_pysystemtrade",
          "pysystemtrade mr_wings on NIFTY daily: fade EWMAC(4,16)/vol beyond 3 rolling std; binary +-1 lot while the forecast is non-zero",
          dict(_MRW), horizon="positional", underlying="NIFTY")
def r6_gh_mrwings_nifty(d, p):
    return _mrw(d, p)


@strategy("r6_gh_mrwings_banknifty", "futures", "gh_pysystemtrade",
          "pysystemtrade mr_wings on BANKNIFTY daily: fade EWMAC(4,16)/vol beyond 3 rolling std; binary +-1 lot while non-zero",
          dict(_MRW), horizon="positional", underlying="BANKNIFTY")
def r6_gh_mrwings_banknifty(d, p):
    return _mrw(d, p)


# ══════════════════════════════════════════════════════════════════════════
# 11. Williams VIX Fix (openalgo examples/python/"william vix fix.py"), stocks, 5-session hold
# ══════════════════════════════════════════════════════════════════════════
@strategy("r6_gh_wvf_stock", "swing", "gh_openalgo",
          "Williams VIX Fix (22; BB 20,2; 0.85 x highest 50) signal at the close -> buy next open, hold 5 sessions, no stop; CNC",
          {"pd": 22, "bbl": 20, "mult": 2.0, "lb": 50, "ph": 0.85, "hold": 5})
def r6_gh_wvf_stock(d, p):
    tail = 100                                   # every window is finite (<= 71 closes): the tail gives exact values
    _w, sig = R.wvf_signal(d.c[-tail:], d.l[-tail:], p.pd, p.bbl, p.mult, p.lb, p.ph)
    if len(sig) and sig[-1]:
        return Swing(LONG, max_hold=p.hold)
    return None


# ══════════════════════════════════════════════════════════════════════════
# 13. Lunar cycle (QuantConnect Strategy Library 61), NIFTY futures long / short
# ══════════════════════════════════════════════════════════════════════════
def _lunar_side(v) -> int:
    return LONG if R.lunar_state(int(v.date[-1])) == "LQ" else SHORT


@strategy("r6_gh_lunar_nifty", "futures", "gh_quantconnect",
          "QC 61 lunar cycle on NIFTY futures: long after the last quarter, short after the first quarter (Meeus phases), always in",
          {"long_after": "LQ", "short_after": "FQ"}, horizon="positional", underlying="NIFTY")
def r6_gh_lunar_nifty(d, p):
    side = _lunar_side(d)
    return Swing(side, max_hold=FOREVER, exit_fn=(lambda v, sd=side: _lunar_side(v) != sd))


# ══════════════════════════════════════════════════════════════════════════
# 14. Ghost trader (letianzj backtest/ghost_trader.py), NIFTY / BANKNIFTY daily futures
# ══════════════════════════════════════════════════════════════════════════
def _ghost_exit_long(v):
    n = R.GHOST["donchian_n"]
    return len(v.c) > n and bool(v.c[-1] <= v.l[-n - 1:-1].min())


def _ghost_exit_short(v):
    n = R.GHOST["donchian_n"]
    return len(v.c) > n and bool(v.h[-1] >= v.h[-n - 1:-1].max())


def _ghost(d, p):
    if len(d.c) < 23:
        return None
    enter, _x = R.ghost_replay(d.h, d.l, d.c)
    side = int(enter[-1])
    if side == 0:
        return None
    return Swing(side, max_hold=FOREVER, exit_fn=_ghost_exit_long if side > 0 else _ghost_exit_short)


@strategy("r6_gh_ghost_nifty", "futures", "gh_letianzj",
          "letianzj ghost trader on NIFTY daily: EMA3/21 + RSI9 + new high/low, trade only after the ghost trade is at a loss; Donchian-21 exit",
          dict(R.GHOST), horizon="positional", underlying="NIFTY")
def r6_gh_ghost_nifty(d, p):
    return _ghost(d, p)


@strategy("r6_gh_ghost_banknifty", "futures", "gh_letianzj",
          "letianzj ghost trader on BANKNIFTY daily: EMA3/21 + RSI9 + new high/low, trade only after the ghost trade is at a loss; Donchian-21 exit",
          dict(R.GHOST), horizon="positional", underlying="BANKNIFTY")
def r6_gh_ghost_banknifty(d, p):
    return _ghost(d, p)


# ══════════════════════════════════════════════════════════════════════════
# 18. Dynamic Breakout II (QuantConnect Strategy Library 04 / letianzj dynamic_breakout_ii.py), stock futures
# ══════════════════════════════════════════════════════════════════════════
_DBO_CACHE: dict = {}


def _dbo_lookback(v) -> np.ndarray:
    """N_t over v (prefix-cached per symbol: the recursion runs from the first bar, so each day's call extends the
    previous one; a cache entry is reused only if the first date, the last cached date and close agree)."""
    key = getattr(v, "symbol", None)
    n = len(v.c)
    pre = None
    ent = _DBO_CACHE.get(key)
    if ent is not None:
        dates, closes, N = ent
        L = min(len(N), n)
        if L and dates[0] == v.date[0] and dates[L - 1] == v.date[L - 1] and closes[L - 1] == v.c[L - 1]:
            pre = N[:L]
    N = R.dbo2_lookback(v.c, n0=pre)
    if ent is None or len(N) >= len(ent[2]):
        _DBO_CACHE[key] = (np.array(v.date), np.array(v.c, float), N)
    return N


def _dbo_exit(side):
    def f(v):
        lv = R.dbo2_at(v.c, v.h, v.l, _dbo_lookback(v))
        return lv is not None and (lv["c"] < lv["ma"] if side > 0 else lv["c"] > lv["ma"])
    return f


@strategy("r6_gh_dbo2_stock", "futures", "gh_quantconnect",
          "Dynamic Breakout II (daily stock futures): lookback 20..60 adapts to 30-day vol; close > BB(N,2) top & > N-day high long (mirror short); exit SMA(N)",
          {"vol_n": R.DBO_VOL_N, "n0": R.DBO_START, "n_min": R.DBO_MIN, "n_max": R.DBO_MAX, "k": 2.0},
          horizon="positional")
def r6_gh_dbo2_stock(d, p):
    if len(d.c) < 32:
        return None
    lv = R.dbo2_at(d.c, d.h, d.l, _dbo_lookback(d))
    if lv is None:
        return None
    if lv["c"] > lv["ub"] and lv["c"] > lv["hh"]:
        return Swing(LONG, max_hold=FOREVER, exit_fn=_dbo_exit(LONG))
    if lv["c"] < lv["lb"] and lv["c"] < lv["ll"]:
        return Swing(SHORT, max_hold=FOREVER, exit_fn=_dbo_exit(SHORT))
    return None
