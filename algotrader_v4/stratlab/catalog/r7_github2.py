"""
Round 7 — second GitHub-sourced shortlist (OHLCV-only items), re-implemented from the source code / the book rules
with the known bugs fixed, run once with no tuning. Pre-registration: stratlab/prereg/github_v2.json (byte-identical
to <scratchpad>/PREREGISTERED_GITHUB2.json), written before any run. Changing anything here makes it a new,
unregistered strategy.

Per-symbol strategies only. The portfolio items (volatility-managed NIFTY, weekly stock-futures carry, sector
rotation) live in stratlab/r7_portfolio.py, Connors TPS (scale-in) in stratlab/r7_tps.py; formulas in
stratlab/r7_ind.py.

Conventions (stated in the pre-registration):
  * Index futures = the lab's index futures (spot index bars stand in for the future, 1 lot, 2026 lot sizes, 1 bps
    slippage). Stocks = stock futures on the cash price (long / short) or CNC delivery (long only), Rs 1 lakh.
  * Daily rules decide at the close; market entries fill at the next open; stop / limit entries use the round-7
    resting-order fill (api.Swing.entry: next session(s) only, fill-session stop taken pessimistically, targets
    from the next session). Rule exits at the next open.
  * Intraday (80-20, Oops): 1-minute bars; a resting stop is approximated by the lab convention (decision at the
    close of the first minute that trades through the level, fill at the next minute's open); 15:15 square-off.
"""
from __future__ import annotations

import numpy as np

from ..api import LONG, SHORT, Entry, Swing, strategy
from .. import r7_ind as R7
from ..sim import SQUAREOFF, equity_tick

FOREVER = 100_000
INDEX_TICK = 0.05


def _tick(sym: str, px: float, date: int) -> float:
    return INDEX_TICK if str(sym).startswith("_") else equity_tick(float(px), int(date))


# ══════════════════════════════════════════════════════════════════════════
# 4. Clenow counter-trend (Trading Evolved ch. 17), long only as coded
# ══════════════════════════════════════════════════════════════════════════
def _clenow_exit(v):
    trend, _pb = R7.clenow_ct(v.c)
    return not trend


def _clenow(d, p):
    if len(d.c) < p.slow_ma + 1:
        return None
    trend, pb = R7.clenow_ct(d.c)
    if trend and np.isfinite(pb) and pb < p.dip_buy:
        return Swing(LONG, max_hold=p.days_to_hold, exit_fn=_clenow_exit)
    return None


_CL = dict(R7.CLENOW)


@strategy("r7_gh2_clenow_ct_nifty", "futures", "gh2_clenow",
          "Clenow ch.17 counter-trend on NIFTY: EMA40>EMA80, close 3 std(price diffs, 40d) below the 20d high -> long; out on trend loss or after 20 sessions",
          dict(_CL), horizon="positional", underlying="NIFTY")
def r7_gh2_clenow_ct_nifty(d, p):
    return _clenow(d, p)


@strategy("r7_gh2_clenow_ct_banknifty", "futures", "gh2_clenow",
          "Clenow ch.17 counter-trend on BANKNIFTY: EMA40>EMA80, close 3 std(price diffs, 40d) below the 20d high -> long; out on trend loss / 20 sessions",
          dict(_CL), horizon="positional", underlying="BANKNIFTY")
def r7_gh2_clenow_ct_banknifty(d, p):
    return _clenow(d, p)


@strategy("r7_gh2_clenow_ct_stock", "futures", "gh2_clenow",
          "Clenow ch.17 counter-trend on stock futures: EMA40>EMA80, close 3 std(price diffs, 40d) below the 20d high -> long; out on trend loss / 20 sessions",
          dict(_CL), horizon="positional")
def r7_gh2_clenow_ct_stock(d, p):
    return _clenow(d, p)


# ══════════════════════════════════════════════════════════════════════════
# 7. Turtle Soup and Turtle Soup Plus One (Street Smarts), long / short
# ══════════════════════════════════════════════════════════════════════════
def _ts(d, p):
    if len(d.c) < p.n + 1:
        return None
    cands = R7.turtle_soup(d.h, d.l, p.n, p.min_age)
    if not cands:
        return None
    if len(cands) == 2:                       # both sides set up: the side whose arm level is nearer the close
        c = float(d.c[-1])
        cands = sorted(cands, key=lambda x: (abs(c - x[1]), -x[0]))
    side, x = cands[0]
    tk = _tick(d.symbol, x, d.date[-1])
    return Swing(side, entry=("stop", x + side * p.ticks * tk, x), entry_stop_ticks=1, max_hold=p.max_hold)


def _tsp1(d, p):
    if len(d.c) < p.n + 2:
        return None
    cands = R7.turtle_soup_plus_one(d.c, d.h, d.l, p.n, p.min_age)
    if not cands:
        return None
    side, x, ext = cands[0]
    return Swing(side, entry=("stop", x), stop=ext, entry_stop_ticks=1, max_hold=p.max_hold)


_TS = {"n": 20, "min_age": 4, "ticks": 5, "max_hold": 5}
_TSP1 = {"n": 20, "min_age": 3, "max_hold": 5}


@strategy("r7_gh2_ts_stock", "futures", "gh2_turtle_soup",
          "Turtle Soup (stock futures): prior 20d low >= 4 sessions old; next session trades below it -> buy stop 5 ticks above; stop 1 tick under that day's low; 5 sessions (mirror short)",
          dict(_TS), horizon="positional")
def r7_gh2_ts_stock(d, p):
    return _ts(d, p)


@strategy("r7_gh2_ts_nifty", "futures", "gh2_turtle_soup",
          "Turtle Soup on NIFTY futures: prior 20d low >= 4 sessions old; next session trades below it -> buy stop 5 ticks above; stop 1 tick under that low; 5 sessions (mirror)",
          dict(_TS), horizon="positional", underlying="NIFTY")
def r7_gh2_ts_nifty(d, p):
    return _ts(d, p)


@strategy("r7_gh2_ts_banknifty", "futures", "gh2_turtle_soup",
          "Turtle Soup on BANKNIFTY futures: prior 20d low >= 4 sessions old; next session trades below it -> buy stop 5 ticks above; stop 1 tick under; 5 sessions (mirror)",
          dict(_TS), horizon="positional", underlying="BANKNIFTY")
def r7_gh2_ts_banknifty(d, p):
    return _ts(d, p)


@strategy("r7_gh2_tsp1_stock", "futures", "gh2_turtle_soup",
          "Turtle Soup +1 (stock futures): day 1 new 20d low closing at/below the prior low (>= 3 sessions old) -> day-2 buy stop at it; stop under the lower low; 5 sessions (mirror)",
          dict(_TSP1), horizon="positional")
def r7_gh2_tsp1_stock(d, p):
    return _tsp1(d, p)


@strategy("r7_gh2_tsp1_nifty", "futures", "gh2_turtle_soup",
          "Turtle Soup +1 on NIFTY futures: day 1 new 20d low closing at/below the prior low (>= 3 sessions old) -> day-2 buy stop at it; 5 sessions (mirror)",
          dict(_TSP1), horizon="positional", underlying="NIFTY")
def r7_gh2_tsp1_nifty(d, p):
    return _tsp1(d, p)


@strategy("r7_gh2_tsp1_banknifty", "futures", "gh2_turtle_soup",
          "Turtle Soup +1 on BANKNIFTY futures: day 1 new 20d low closing at/below the prior low (>= 3 sessions old) -> day-2 buy stop at it; 5 sessions (mirror)",
          dict(_TSP1), horizon="positional", underlying="BANKNIFTY")
def r7_gh2_tsp1_banknifty(d, p):
    return _tsp1(d, p)


# ══════════════════════════════════════════════════════════════════════════
# 8. Williams Smash Day (rule list), long / short
# ══════════════════════════════════════════════════════════════════════════
def _smash(d, p):
    r = R7.smash_day(d.o, d.h, d.l, d.c, p.__dict__)
    if r is None:
        return None
    side, lv, sl, tp = r
    return Swing(side, entry=("stop", lv), stop=sl, target=tp, valid=p.valid, max_hold=FOREVER)


_SM = dict(R7.SMASH)


@strategy("r7_gh2_smash_stock", "futures", "gh2_smash",
          "Williams Smash Day (stock futures): close < prior low and a 4-day low -> buy stop at its high for 2 sessions; SL 2.1 ATR21, TP 0.75 x SL (mirror short)",
          dict(_SM), horizon="positional")
def r7_gh2_smash_stock(d, p):
    return _smash(d, p)


@strategy("r7_gh2_smash_nifty", "futures", "gh2_smash",
          "Williams Smash Day on NIFTY futures: close < prior low and a 4-day low -> buy stop at its high for 2 sessions; SL 2.1 ATR21, TP 0.75 x SL (mirror)",
          dict(_SM), horizon="positional", underlying="NIFTY")
def r7_gh2_smash_nifty(d, p):
    return _smash(d, p)


@strategy("r7_gh2_smash_banknifty", "futures", "gh2_smash",
          "Williams Smash Day on BANKNIFTY futures: close < prior low and a 4-day low -> buy stop at its high for 2 sessions; SL 2.1 ATR21, TP 0.75 x SL (mirror)",
          dict(_SM), horizon="positional", underlying="BANKNIFTY")
def r7_gh2_smash_banknifty(d, p):
    return _smash(d, p)


# ══════════════════════════════════════════════════════════════════════════
# 9. Raschke Holy Grail (daily), long / short
# ══════════════════════════════════════════════════════════════════════════
def _hg(d, p):
    r = R7.holy_grail(d.h, d.l, d.c, p.__dict__)
    if r is None:
        return None
    side, ext, base, tgt = r
    tk = _tick(d.symbol, ext, d.date[-1])
    return Swing(side, entry=("stop", ext + side * tk), stop=base, entry_stop_ticks=1, target=tgt, max_hold=FOREVER)


_HG = dict(R7.HOLY)


@strategy("r7_gh2_hg_stock", "futures", "gh2_holy_grail",
          "Raschke Holy Grail (stock futures, daily): ADX14>30 rising, first pullback to EMA20 -> buy stop above the touch bar; stop new swing low, target 20d swing high (mirror)",
          dict(_HG), horizon="positional")
def r7_gh2_hg_stock(d, p):
    return _hg(d, p)


@strategy("r7_gh2_hg_nifty", "futures", "gh2_holy_grail",
          "Raschke Holy Grail on NIFTY futures (daily): ADX14>30 rising, first pullback to EMA20 -> buy stop above the touch bar; swing-low stop, swing-high target (mirror)",
          dict(_HG), horizon="positional", underlying="NIFTY")
def r7_gh2_hg_nifty(d, p):
    return _hg(d, p)


@strategy("r7_gh2_hg_banknifty", "futures", "gh2_holy_grail",
          "Raschke Holy Grail on BANKNIFTY futures (daily): ADX14>30 rising, first pullback to EMA20 -> buy stop above the touch bar; swing-low stop, swing-high target (mirror)",
          dict(_HG), horizon="positional", underlying="BANKNIFTY")
def r7_gh2_hg_banknifty(d, p):
    return _hg(d, p)


# ══════════════════════════════════════════════════════════════════════════
# 12. Double 7s (pinkfish), NIFTY / BANKNIFTY, long only, next-open fills
# ══════════════════════════════════════════════════════════════════════════
def _d7_exit(v):
    return R7.double7_exit(v.c, 7)


def _d7(d, p):
    if R7.double7_entry(d.c, p.period, p.sma):
        return Swing(LONG, max_hold=FOREVER, exit_fn=_d7_exit)
    return None


_D7 = {"period": 7, "sma": 200, "fill": "next open"}


@strategy("r7_gh2_d7_nifty", "futures", "gh2_double7",
          "Double 7s on NIFTY futures: close = 7-day closing low and > SMA200 -> long next open; exit next open after a 7-day closing high",
          dict(_D7), horizon="positional", underlying="NIFTY")
def r7_gh2_d7_nifty(d, p):
    return _d7(d, p)


@strategy("r7_gh2_d7_banknifty", "futures", "gh2_double7",
          "Double 7s on BANKNIFTY futures: close = 7-day closing low and > SMA200 -> long next open; exit next open after a 7-day closing high",
          dict(_D7), horizon="positional", underlying="BANKNIFTY")
def r7_gh2_d7_banknifty(d, p):
    return _d7(d, p)


# ══════════════════════════════════════════════════════════════════════════
# 13. Bensdorp MRL (CNC long) / MRS (stock futures short), limit-entry stretch
# ══════════════════════════════════════════════════════════════════════════
def _mrl(d, p):
    r = R7.mrl_setup(d.h, d.l, d.c, p.__dict__)
    if r is None:
        return None
    lim, sl, tg = r
    return Swing(LONG, entry=("limit", lim), stop=sl, max_hold=p.max_hold,
                 exit_fn=(lambda v, t=tg: bool(v.c[-1] >= t)))


def _mrs(d, p):
    r = R7.mrs_setup(d.h, d.l, d.c, p.__dict__)
    if r is None:
        return None
    lim, sl, tg = r
    return Swing(SHORT, entry=("limit", lim), stop=sl, max_hold=p.max_hold,
                 exit_fn=(lambda v, t=tg: bool(v.c[-1] <= t)))


@strategy("r7_gh2_mrl_stock", "swing", "gh2_bensdorp",
          "Bensdorp MRL: close>SMA150, ADX7>45, ATR%10>4%, RSI3<30 -> limit buy 4% under the close; stop 2.5 ATR%, exit next open after a close >= +3%, 4 sessions; CNC",
          dict(R7.MRL))
def r7_gh2_mrl_stock(d, p):
    return _mrl(d, p)


@strategy("r7_gh2_mrs_stock", "futures", "gh2_bensdorp",
          "Bensdorp MRS (stock futures): 2 up closes, ADX7>50, ATR%10>5%, RSI3>85 -> sell limit at the close; stop 2.5 ATR%, exit after a close <= -4%, 2 sessions",
          dict(R7.MRS), horizon="positional")
def r7_gh2_mrs_stock(d, p):
    return _mrs(d, p)


# ══════════════════════════════════════════════════════════════════════════
# 14. Raschke 80-20 and Williams Oops, intraday on 1-minute bars
# ══════════════════════════════════════════════════════════════════════════
def _itick(s, px):
    return INDEX_TICK if str(s.symbol).startswith("_") else equity_tick(float(px), int(s.date))


def _intraday_entry(s, side, arm, level):
    m = R7.stop_reentry_bar(s.h, s.l, s.c, side, arm, level)
    if m < 0 or s.t[m] + 1 >= SQUAREOFF:
        return []
    tk = _itick(s, level)
    stop = float(np.min(s.l[:m + 1])) - tk if side > 0 else float(np.max(s.h[:m + 1])) + tk
    return [Entry(m, side, stop=stop)]


def _e2080(s, p):
    if not len(s.t) or not all(np.isfinite(x) for x in (s.pdo, s.pdh, s.pdl, s.pdc)):
        return []
    side = R7.eighty_twenty_side(float(s.pdo), float(s.pdh), float(s.pdl), float(s.pdc))
    if side == 0:
        return []
    level = float(s.pdl) if side > 0 else float(s.pdh)
    tk = _itick(s, level)
    return _intraday_entry(s, side, level - side * p.ticks * tk, level)


def _oops(s, p):
    if not len(s.t) or not (np.isfinite(s.pdl) and np.isfinite(s.pdh)):
        return []
    o = float(s.o[0])
    if o < s.pdl:
        return _intraday_entry(s, LONG, float(s.pdl), float(s.pdl))
    if o > s.pdh:
        return _intraday_entry(s, SHORT, float(s.pdh), float(s.pdh))
    return []


_E28 = {"top": 0.8, "bottom": 0.2, "ticks": 5, "exit": "15:15"}
_OOPS = {"gap": "open beyond yesterday's low / high", "exit": "15:15"}


@strategy("r7_gh2_8020_stock", "intraday", "gh2_raschke_8020",
          "Raschke 80-20 (stocks, 1m): yesterday opened top 20% / closed bottom 20% of range; today >= 5 ticks under its low -> buy stop at it; stop under the day's low; 15:15 (mirror)",
          dict(_E28))
def r7_gh2_8020_stock(s, p):
    return _e2080(s, p)


@strategy("r7_gh2_8020_nifty", "futures", "gh2_raschke_8020",
          "Raschke 80-20 on NIFTY futures (1m): yesterday open top 20% / close bottom 20%; today >= 5 ticks under its low -> buy stop at it; stop under the day's low; 15:15 (mirror)",
          dict(_E28), horizon="intraday", underlying="NIFTY")
def r7_gh2_8020_nifty(s, p):
    return _e2080(s, p)


@strategy("r7_gh2_8020_banknifty", "futures", "gh2_raschke_8020",
          "Raschke 80-20 on BANKNIFTY futures (1m): yesterday open top 20% / close bottom 20%; today >= 5 ticks under its low -> buy stop at it; stop under the low; 15:15 (mirror)",
          dict(_E28), horizon="intraday", underlying="BANKNIFTY")
def r7_gh2_8020_banknifty(s, p):
    return _e2080(s, p)


@strategy("r7_gh2_oops_stock", "intraday", "gh2_williams_oops",
          "Williams Oops (stocks, 1m): open below yesterday's low -> buy stop at that low; stop under the day's low; exit 15:15 (mirror on a gap above the high)",
          dict(_OOPS))
def r7_gh2_oops_stock(s, p):
    return _oops(s, p)


@strategy("r7_gh2_oops_nifty", "futures", "gh2_williams_oops",
          "Williams Oops on NIFTY futures (1m): open below yesterday's low -> buy stop at that low; stop under the day's low; exit 15:15 (mirror)",
          dict(_OOPS), horizon="intraday", underlying="NIFTY")
def r7_gh2_oops_nifty(s, p):
    return _oops(s, p)


@strategy("r7_gh2_oops_banknifty", "futures", "gh2_williams_oops",
          "Williams Oops on BANKNIFTY futures (1m): open below yesterday's low -> buy stop at that low; stop under the day's low; exit 15:15 (mirror)",
          dict(_OOPS), horizon="intraday", underlying="BANKNIFTY")
def r7_gh2_oops_banknifty(s, p):
    return _oops(s, p)
