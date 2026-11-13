"""
Round-3 TIMEFRAME WEEKLY candidates (swing CNC long-only, stock futures positional shorts, index-future TSMOM).
Pre-registered BEFORE any result in scratchpad/r3/tfweekly/PREREGISTERED.txt -- textbook / round parameters; do not
tune them. Every variant is one more candidate that can pass by luck.

Mechanics shared by every rule:
* decision_time="09:16" on the FIRST session of a new (Monday-based) calendar week -- or of a new calendar month for
  the Faber rules. The DailyView's last bar is then today-so-far (09:15 minute only) and is NEVER used: weekly /
  monthly bars are built from d.*[:-1], so only completed weeks (months) enter a decision. Fill = 09:16 bar open.
* The first weekly / monthly bar of the history is dropped (the data may start mid-week / mid-month).
* An indicator is used only once its full lookback of completed bars exists.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from ..api import LONG, SHORT, Swing, strategy
from ..indicators import atr as _atr, ema as _ema, rsi as _rsi, sma as _sma

HYG_JUMP = 0.25
_CA_GAPS = {("SIEMENS", 20250407), ("VEDL", 20260430)}     # same list as stratlab.ltr.CA_GAPS
_ORD: dict[int, int] = {}


# ── helpers ─────────────────────────────────────────────────────────────────
def _ordinal(x: int) -> int:
    o = _ORD.get(x)
    if o is None:
        o = _dt.date(x // 10000, (x // 100) % 100, x % 100).toordinal()
        _ORD[x] = o
    return o


def _wk(x) -> int:
    """Monday-based week id (date.toordinal(): 0001-01-01 is a Monday, ordinal 1)."""
    return (_ordinal(int(x)) - 1) // 7


def _mo(x) -> int:
    return int(x) // 100


def _new_week(d) -> bool:
    return len(d) >= 3 and _wk(d.date[-1]) != _wk(d.date[-2])


def _new_month(d) -> bool:
    return len(d) >= 3 and _mo(d.date[-1]) != _mo(d.date[-2])


def _group(keys: np.ndarray, o, h, l, c, v):
    """Aggregate consecutive daily rows with equal key; drop the first group (may be partial)."""
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    if len(starts) < 2:
        return None
    ends = np.r_[starts[1:], len(keys)]
    W = {"o": o[starts], "h": np.maximum.reduceat(h, starts), "l": np.minimum.reduceat(l, starts),
         "c": c[ends - 1], "v": np.add.reduceat(v, starts)}
    return {k: x[1:] for k, x in W.items()}


def _weekly(d):
    """Completed weekly bars from the daily history BEFORE today (d.*[:-1])."""
    dates = d.date[:-1]
    keys = np.fromiter((_wk(x) for x in dates), np.int64, len(dates))
    return _group(keys, d.o[:-1], d.h[:-1], d.l[:-1], d.c[:-1], d.v[:-1])


def _monthly(d):
    dates = d.date[:-1]
    keys = np.asarray(dates, np.int64) // 100
    return _group(keys, d.o[:-1], d.h[:-1], d.l[:-1], d.c[:-1], d.v[:-1])


def _weekly_close(dates, c):
    """Weekly closes of an aux daily series (e.g. NIFTY) aligned to the same dates, before today."""
    keys = np.fromiter((_wk(x) for x in dates), np.int64, len(dates))
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    if len(starts) < 2:
        return None
    ends = np.r_[starts[1:], len(keys)]
    return np.asarray(c, float)[ends - 1][1:]


def _clean(d, n: int) -> bool:
    """No unadjusted corporate action in the last n sessions before today (and no known CA gap)."""
    k = min(len(d) - 1, n + 1)
    if k < 2:
        return True
    c = d.c[-1 - k:-1]
    o = d.o[-1 - k:-1]
    pc = c[:-1]
    if np.any(np.abs(c[1:] / pc - 1.0) > HYG_JUMP) or np.any(np.abs(o[1:] / pc - 1.0) > HYG_JUMP):
        return False
    sym = getattr(d, "symbol", None)
    if sym:
        rec = d.date[-251:]
        for s_, day in _CA_GAPS:
            if s_ == sym and rec[0] <= day <= rec[-1]:
                return False
    return True


def _ok(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def _nifty_weekly(d):
    if d.nifty is None:
        return None
    nc = np.asarray(d.nifty[:-1], float)
    if len(nc) < 3 or not np.all(np.isfinite(nc[-min(len(nc), 200):])):
        return None
    return _weekly_close(d.date[:-1], nc)


def _supertrend(h, l, c, n: int, m: float):
    """Textbook Supertrend on bars h/l/c: returns (line, trend) arrays; trend +1 up / -1 down (0 before ready)."""
    N = len(c)
    a = _atr(h, l, c, n)
    hl2 = (h + l) / 2.0
    bu, bl = hl2 + m * a, hl2 - m * a
    fu, fl = bu.copy(), bl.copy()
    tr = np.zeros(N, int)
    line = np.full(N, np.nan)
    if N <= n:
        return line, tr
    tr[n] = 1 if c[n] >= hl2[n] else -1
    line[n] = fl[n] if tr[n] > 0 else fu[n]
    for i in range(n + 1, N):
        fu[i] = bu[i] if (bu[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = bl[i] if (bl[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if tr[i - 1] > 0:
            tr[i] = -1 if c[i] < fl[i] else 1
        else:
            tr[i] = 1 if c[i] > fu[i] else -1
        line[i] = fl[i] if tr[i] > 0 else fu[i]
    return line, tr


def _prep(d, min_weeks: int):
    """Weekly bars if today starts a new week, the history is clean and min_weeks completed weeks exist."""
    if not _new_week(d):
        return None
    if not _clean(d, 5 * min_weeks + 10):
        return None
    W = _weekly(d)
    if W is None or len(W["c"]) < min_weeks:
        return None
    return W


# ════════════════════════════════════════════════════════════════════════════
# 1-2  weekly EMA10 / EMA30
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_tfweekly_ema10_30_cross", "swing", "weekly_ma_cross",
          "Weekly EMA10 crosses above EMA30 on the last completed week: buy Monday 09:16, 10% stop, hold <= 60 sessions",
          {"fast": 10, "slow": 30, "stop_pct": 10.0, "max_hold": 60}, decision_time="09:16")
def r3_tfweekly_ema10_30_cross(d, p):
    W = _prep(d, p.slow + 1)
    if W is None:
        return None
    f, s = _ema(W["c"], p.fast), _ema(W["c"], p.slow)
    if f[-1] > s[-1] and f[-2] <= s[-2]:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


@strategy("r3_tfweekly_ema10_30_nifty", "swing", "weekly_ma_cross",
          "Stock weekly EMA10>EMA30 and close>EMA30, NIFTY weekly close>its EMA30: buy Monday 09:16, 10% stop, 20 sessions",
          {"fast": 10, "slow": 30, "stop_pct": 10.0, "max_hold": 20}, decision_time="09:16", needs_index=True)
def r3_tfweekly_ema10_30_nifty(d, p):
    W = _prep(d, p.slow)
    if W is None:
        return None
    nw = _nifty_weekly(d)
    if nw is None or len(nw) < p.slow or len(nw) != len(W["c"]):
        return None
    f, s = _ema(W["c"], p.fast), _ema(W["c"], p.slow)
    ns = _ema(nw, p.slow)
    if f[-1] > s[-1] and W["c"][-1] > s[-1] and nw[-1] > ns[-1]:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 3  weekly RSI(14) range shift above 60
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_tfweekly_rsi14_60", "swing", "weekly_rsi_momentum",
          "Weekly RSI(14) crosses above 60 (<=60 the week before): buy Monday 09:16, 10% stop, hold <= 40 sessions",
          {"n": 14, "level": 60.0, "stop_pct": 10.0, "max_hold": 40}, decision_time="09:16")
def r3_tfweekly_rsi14_60(d, p):
    W = _prep(d, 2 * p.n)
    if W is None:
        return None
    r = _rsi(W["c"], p.n)
    if _ok(r[-1], r[-2]) and r[-1] > p.level and r[-2] <= p.level:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 4-5  weekly Donchian(20)
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_tfweekly_donch20_long", "swing", "weekly_donchian",
          "Weekly close above the highest weekly high of the previous 20 weeks: buy Monday 09:16, 10% stop, <= 60 sessions",
          {"n": 20, "stop_pct": 10.0, "max_hold": 60}, decision_time="09:16")
def r3_tfweekly_donch20_long(d, p):
    W = _prep(d, p.n + 1)
    if W is None:
        return None
    if W["c"][-1] > W["h"][-p.n - 1:-1].max():
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


@strategy("r3_tfweekly_donch20_short", "futures", "weekly_donchian",
          "Stock future short: weekly close below the lowest weekly low of the previous 20 weeks, 10% stop, <= 40 sessions",
          {"n": 20, "stop_pct": 10.0, "max_hold": 40}, horizon="positional", decision_time="09:16")
def r3_tfweekly_donch20_short(d, p):
    W = _prep(d, p.n + 1)
    if W is None:
        return None
    if W["c"][-1] < W["l"][-p.n - 1:-1].min():
        return Swing(SHORT, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 6  weekly MACD(12,26,9) signal cross
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_tfweekly_macd_cross", "swing", "weekly_macd",
          "Weekly MACD(12,26) crosses above its 9-week signal: buy Monday 09:16, 10% stop, hold <= 40 sessions",
          {"fast": 12, "slow": 26, "sig": 9, "stop_pct": 10.0, "max_hold": 40}, decision_time="09:16")
def r3_tfweekly_macd_cross(d, p):
    W = _prep(d, p.slow + p.sig)
    if W is None:
        return None
    macd = _ema(W["c"], p.fast) - _ema(W["c"], p.slow)
    sig = _ema(macd, p.sig)
    if macd[-1] > sig[-1] and macd[-2] <= sig[-2]:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 7  13-week relative strength vs NIFTY
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_tfweekly_rs13_nifty", "swing", "weekly_rel_strength",
          "13-week return beats NIFTY's by >= 10 pp and weekly close > 30-week SMA: buy Monday 09:16, 10% stop, 20 sessions",
          {"n": 13, "edge_pp": 10.0, "sma": 30, "stop_pct": 10.0, "max_hold": 20}, decision_time="09:16",
          needs_index=True)
def r3_tfweekly_rs13_nifty(d, p):
    W = _prep(d, p.sma)
    if W is None:
        return None
    nw = _nifty_weekly(d)
    if nw is None or len(nw) != len(W["c"]) or len(nw) < p.n + 1:
        return None
    c = W["c"]
    rs = (c[-1] / c[-1 - p.n] - 1.0) - (nw[-1] / nw[-1 - p.n] - 1.0)
    if _ok(rs) and rs * 100.0 >= p.edge_pp and c[-1] > float(np.mean(c[-p.sma:])):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 8-9  weekly Supertrend(10, 3)
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_tfweekly_supertrend_long", "swing", "weekly_supertrend",
          "Weekly Supertrend(10,3) flips up on the last completed week: buy Monday 09:16, stop = Supertrend line, <= 60",
          {"n": 10, "mult": 3.0, "max_hold": 60}, decision_time="09:16")
def r3_tfweekly_supertrend_long(d, p):
    W = _prep(d, 2 * p.n)
    if W is None:
        return None
    line, tr = _supertrend(W["h"], W["l"], W["c"], p.n, p.mult)
    if tr[-1] > 0 and tr[-2] < 0 and _ok(line[-1]):
        return Swing(LONG, stop=float(line[-1]), max_hold=p.max_hold)
    return None


@strategy("r3_tfweekly_supertrend_short", "futures", "weekly_supertrend",
          "Stock future short: weekly Supertrend(10,3) flips down; stop = Supertrend line, hold <= 40 sessions",
          {"n": 10, "mult": 3.0, "max_hold": 40}, horizon="positional", decision_time="09:16")
def r3_tfweekly_supertrend_short(d, p):
    W = _prep(d, 2 * p.n)
    if W is None:
        return None
    line, tr = _supertrend(W["h"], W["l"], W["c"], p.n, p.mult)
    if tr[-1] < 0 and tr[-2] > 0 and _ok(line[-1]):
        return Swing(SHORT, stop=float(line[-1]), max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 10-11  Faber: monthly close above 10-month SMA
# ════════════════════════════════════════════════════════════════════════════
def _faber(d, p, with_nifty: bool):
    if not _new_month(d) or not _clean(d, 22 * p.months + 10):
        return None
    M = _monthly(d)
    if M is None or len(M["c"]) < p.months:
        return None
    if not M["c"][-1] > float(np.mean(M["c"][-p.months:])):
        return None
    if with_nifty:
        if d.nifty is None:
            return None
        nc = np.asarray(d.nifty[:-1], float)
        keys = np.asarray(d.date[:-1], np.int64) // 100
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        if len(starts) < 2:
            return None
        ends = np.r_[starts[1:], len(keys)]
        nm = nc[ends - 1][1:]
        if len(nm) < p.months or not np.all(np.isfinite(nm[-p.months:])):
            return None
        if not nm[-1] > float(np.mean(nm[-p.months:])):
            return None
    return Swing(LONG, max_hold=p.max_hold)


@strategy("r3_tfweekly_faber10m", "swing", "monthly_faber",
          "First session of a month: last monthly close > 10-month SMA -> buy 09:16, no stop, hold 20 sessions",
          {"months": 10, "max_hold": 20}, decision_time="09:16")
def r3_tfweekly_faber10m(d, p):
    return _faber(d, p, False)


@strategy("r3_tfweekly_faber10m_nifty", "swing", "monthly_faber",
          "As faber10m and NIFTY's last monthly close > its 10-month SMA: buy 09:16, no stop, hold 20 sessions",
          {"months": 10, "max_hold": 20}, decision_time="09:16", needs_index=True)
def r3_tfweekly_faber10m_nifty(d, p):
    return _faber(d, p, True)


# ════════════════════════════════════════════════════════════════════════════
# 12-13  one-week loser reversal
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_tfweekly_wk_loser_rev", "swing", "weekly_reversal",
          "Last completed week's return <= -10%: buy Monday 09:16, 10% stop, hold 10 sessions",
          {"drop_pct": 10.0, "stop_pct": 10.0, "max_hold": 10}, decision_time="09:16")
def r3_tfweekly_wk_loser_rev(d, p):
    W = _prep(d, 3)
    if W is None:
        return None
    r = W["c"][-1] / W["c"][-2] - 1.0
    if r * 100.0 <= -p.drop_pct:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


@strategy("r3_tfweekly_wk_excess_loser_rev", "swing", "weekly_reversal",
          "Last week's stock return minus NIFTY's <= -8 pp: buy Monday 09:16, 10% stop, hold 10 sessions",
          {"excess_pp": 8.0, "stop_pct": 10.0, "max_hold": 10}, decision_time="09:16", needs_index=True)
def r3_tfweekly_wk_excess_loser_rev(d, p):
    W = _prep(d, 3)
    if W is None:
        return None
    nw = _nifty_weekly(d)
    if nw is None or len(nw) != len(W["c"]) or len(nw) < 2:
        return None
    ex = (W["c"][-1] / W["c"][-2] - 1.0) - (nw[-1] / nw[-2] - 1.0)
    if _ok(ex) and ex * 100.0 <= -p.excess_pp:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 14-15  weekly volume shock (Gervais-Kaniel-Mingelgrin high-volume premium)
# ════════════════════════════════════════════════════════════════════════════
def _volshock(d, p):
    W = _prep(d, p.n + 2)
    if W is None:
        return None
    v = W["v"]
    base = float(np.mean(v[-p.n - 1:-1]))
    if not base > 0 or not v[-1] >= p.mult * base:
        return None
    return W["c"][-1] / W["c"][-2] - 1.0


@strategy("r3_tfweekly_volshock_up", "swing", "weekly_volume",
          "Last week's volume >= 2x the 10 prior weeks' mean and week return >= +5%: buy Monday 09:16, 10% stop, 20 sessions",
          {"n": 10, "mult": 2.0, "ret_pct": 5.0, "stop_pct": 10.0, "max_hold": 20}, decision_time="09:16")
def r3_tfweekly_volshock_up(d, p):
    r = _volshock(d, p)
    if r is not None and r * 100.0 >= p.ret_pct:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


@strategy("r3_tfweekly_volshock_flat", "swing", "weekly_volume",
          "Last week's volume >= 2x the 10 prior weeks' mean and |week return| <= 2%: buy Monday 09:16, 10% stop, 20 sessions",
          {"n": 10, "mult": 2.0, "ret_pct": 2.0, "stop_pct": 10.0, "max_hold": 20}, decision_time="09:16")
def r3_tfweekly_volshock_flat(d, p):
    r = _volshock(d, p)
    if r is not None and abs(r) * 100.0 <= p.ret_pct:
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 16  three down weeks inside an uptrend
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_tfweekly_3down_uptrend", "swing", "weekly_pullback",
          "3 consecutive lower weekly closes and last weekly close > 30-week SMA: buy Monday 09:16, 10% stop, 20 sessions",
          {"k": 3, "sma": 30, "stop_pct": 10.0, "max_hold": 20}, decision_time="09:16")
def r3_tfweekly_3down_uptrend(d, p):
    W = _prep(d, p.sma)
    if W is None:
        return None
    c = W["c"]
    if np.all(np.diff(c[-p.k - 1:]) < 0) and c[-1] > float(np.mean(c[-p.sma:])):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 17  market sell-off week: buy stocks that held up and are in an uptrend
# ════════════════════════════════════════════════════════════════════════════
@strategy("r3_tfweekly_mkt_selloff_rs", "swing", "market_dispersion",
          "NIFTY week <= -3%, stock's week > NIFTY's and stock weekly close > 30-week SMA: buy Monday 09:16, 10% stop, 20",
          {"mkt_drop_pct": 3.0, "sma": 30, "stop_pct": 10.0, "max_hold": 20}, decision_time="09:16", needs_index=True)
def r3_tfweekly_mkt_selloff_rs(d, p):
    W = _prep(d, p.sma)
    if W is None:
        return None
    nw = _nifty_weekly(d)
    if nw is None or len(nw) != len(W["c"]) or len(nw) < 2:
        return None
    c = W["c"]
    nr = nw[-1] / nw[-2] - 1.0
    sr = c[-1] / c[-2] - 1.0
    if _ok(nr, sr) and nr * 100.0 <= -p.mkt_drop_pct and sr > nr and c[-1] > float(np.mean(c[-p.sma:])):
        return Swing(LONG, stop_pct=p.stop_pct, max_hold=p.max_hold)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 18-19  index-future time-series momentum, re-decided every week
# ════════════════════════════════════════════════════════════════════════════
def _tsmom(d, p):
    if not _new_week(d):
        return None
    W = _weekly(d)
    if W is None or len(W["c"]) < p.n + 1:
        return None
    r = W["c"][-1] / W["c"][-1 - p.n] - 1.0
    if not _ok(r) or r == 0:
        return None
    return Swing(LONG if r > 0 else SHORT, max_hold=p.max_hold)


@strategy("r3_tfweekly_tsmom13_nifty", "futures", "index_tsmom",
          "NIFTY future, every new week: long if the 13-week return > 0 else short, enter 09:16, hold 5 sessions, no stop",
          {"n": 13, "max_hold": 5}, horizon="positional", decision_time="09:16", underlying="NIFTY")
def r3_tfweekly_tsmom13_nifty(d, p):
    return _tsmom(d, p)


@strategy("r3_tfweekly_tsmom13_banknifty", "futures", "index_tsmom",
          "BANKNIFTY future, every new week: long if the 13-week return > 0 else short, enter 09:16, hold 5, no stop",
          {"n": 13, "max_hold": 5}, horizon="positional", decision_time="09:16", underlying="BANKNIFTY")
def r3_tfweekly_tsmom13_banknifty(d, p):
    return _tsmom(d, p)
