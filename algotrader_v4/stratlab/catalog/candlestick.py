"""
Candlestick-pattern strategies: every textbook pattern of `stratlab.candles` on 5m, 15m, 60m and daily candles,
with ONE fixed trade plan decided before any backtest (not tuned):

  * signal when the pattern candle CLOSES (for a 15m candle ending 09:45 that is the close of the 09:44 bar),
    entry at the next candle's open (the simulator fills at the next 1-minute bar's open + 5 bps);
  * stop just beyond the pattern's extreme (lowest low of the pattern candles - 1 tick for longs, highest
    high + 1 tick for shorts), but at least 0.3 ATR(14, same timeframe) from the decision candle's close;
  * target 2R from the fill; intraday: time stop = the 15:15 square-off; daily: at most 10 sessions;
  * intraday 5m / 15m: the decision candle must end between 09:30 and 14:30 (fill 09:30-14:30), MIS;
    60m: the decision candle must end by 14:15 (fills until 14:15); one trade per symbol per day;
    all candles of an intraday pattern must belong to today's session (prior sessions give the trend context);
  * daily: bullish patterns are kind "swing" (delivery, long only); bearish patterns are kind "futures",
    horizon "positional" (short); the three two-sided patterns (long-legged doji, inside bar, outside bar)
    are "futures" positional in both directions;
  * `_conf` variants wait for one confirmation candle: the candle after the pattern must close beyond the pattern
    candle's close in the trade direction; the entry is then at the open after the confirmation candle (stop
    still beyond the pattern's extreme, minimum distance measured from the confirmation close). Confirmation
    variants exist only for the patterns Nison says need confirmation (hammer family, doji family, harami,
    tweezers).

Ids: cs_<pattern>_<tf>[_conf], tf in 5m 15m 60m d. MULTIPLE TESTING: `N_REGISTERED` strategies are registered
here (27 patterns x 4 timeframes + 11 confirmation patterns x 4 timeframes = 152). With that many candidates a
handful of screen passes are expected from luck alone.
"""
from __future__ import annotations

import numpy as np

from ..api import LONG, SHORT, Entry, Swing, hhmm, strategy
from ..candles import PATTERNS, daily_candles, session_candles

RR = 2.0                     # target = 2R
MIN_STOP_ATR = 0.3           # stop at least 0.3 ATR from the decision close
TICK = 0.05                  # NSE tick: "just beyond" the pattern extreme
MAX_HOLD = 10                # daily: sessions
WINDOW = {5: (hhmm("09:30"), hhmm("14:30")), 15: (hhmm("09:30"), hhmm("14:30")), 60: (0, hhmm("14:15"))}
TF_LABEL = {5: "5m", 15: "15m", 60: "60m", "d": "d"}
CONFIRM = ("hammer", "inverted_hammer", "hanging_man", "shooting_star", "dragonfly_doji", "gravestone_doji",
           "longlegged_doji", "bull_harami", "bear_harami", "tweezer_bottom", "tweezer_top")

_NICE = {
    "hammer": "Hammer after a down-trend", "inverted_hammer": "Inverted hammer after a down-trend",
    "hanging_man": "Hanging man after an up-trend", "shooting_star": "Shooting star after an up-trend",
    "bull_marubozu": "Bullish marubozu", "bear_marubozu": "Bearish marubozu",
    "dragonfly_doji": "Dragonfly doji after a down-trend", "gravestone_doji": "Gravestone doji after an up-trend",
    "longlegged_doji": "Long-legged doji against the prior trend",
    "bull_engulfing": "Bullish engulfing after a down-trend", "bear_engulfing": "Bearish engulfing after an up-trend",
    "piercing_line": "Piercing line after a down-trend", "dark_cloud": "Dark cloud cover after an up-trend",
    "bull_harami": "Bullish harami after a down-trend", "bear_harami": "Bearish harami after an up-trend",
    "tweezer_bottom": "Tweezer bottom after a down-trend", "tweezer_top": "Tweezer top after an up-trend",
    "inside_bar": "Inside bar with the prior trend", "outside_bar": "Outside bar in its close direction",
    "morning_star": "Morning star after a down-trend", "evening_star": "Evening star after an up-trend",
    "three_white_soldiers": "Three white soldiers after a down-trend",
    "three_black_crows": "Three black crows after an up-trend",
    "three_inside_up": "Three inside up after a down-trend", "three_inside_down": "Three inside down after an up-trend",
    "three_outside_up": "Three outside up after a down-trend",
    "three_outside_down": "Three outside down after an up-trend",
}


def _stop(cs, side: int, npat: int, q: int, d: int) -> float | None:
    """Stop beyond the extreme of the pattern ending at candle q, >= MIN_STOP_ATR x ATR from candle d's close."""
    a = float(cs.atr[d])
    if not np.isfinite(a) or a <= 0:
        return None
    if side > 0:
        ext = float(cs.lo[npat][q])
        return min(ext - TICK, float(cs.c[d]) - MIN_STOP_ATR * a) if np.isfinite(ext) else None
    ext = float(cs.hi[npat][q])
    return max(ext + TICK, float(cs.c[d]) + MIN_STOP_ATR * a) if np.isfinite(ext) else None


def candle_entries(s, pattern: str, tf: int, confirm: bool, rr: float = RR) -> list:
    """Entries of one pattern on tf-minute candles of session s (a Session or a truncated View)."""
    cs = session_candles(s, tf)
    if cs is None:
        return []
    npat = PATTERNS[pattern].n
    w0, w1 = WINDOW[tf]
    m = len(cs.c)
    out = []
    for side, mask in ((LONG, cs.sigs[pattern][0]), (SHORT, cs.sigs[pattern][1])):
        for q in np.flatnonzero(mask):
            q = int(q)
            if q < npat - 1:                      # a pattern candle belongs to an earlier session
                continue
            d = q + 1 if confirm else q
            if d >= m or not (w0 <= int(cs.end[d]) <= w1):
                continue
            if confirm and not side * (cs.c[d] - cs.c[q]) > 0:
                continue
            stop = _stop(cs, side, npat, q, d)
            if stop is not None:
                out.append(Entry(int(cs.sig[d]), side, stop=stop, rr=rr))
    return out


def daily_signal(d, pattern: str, confirm: bool, sides: tuple, rr: float = RR):
    """Swing for one pattern on daily candles ending at the decision day d[-1], or None."""
    cs = daily_candles(d)
    m = len(cs.c)
    q, k = (m - 2, m - 1) if confirm else (m - 1, m - 1)
    if q < 0:
        return None
    npat = PATTERNS[pattern].n
    lm, sm = cs.sigs[pattern]
    for side, mask in ((LONG, lm), (SHORT, sm)):
        if side not in sides or not mask[q]:
            continue
        if confirm and not side * (cs.c[k] - cs.c[q]) > 0:
            continue
        stop = _stop(cs, side, npat, q, k)
        if stop is not None:
            return Swing(side, stop=stop, rr=rr, max_hold=MAX_HOLD)
    return None


def _make_intraday(pattern: str, tf: int, confirm: bool):
    def fn(s, p):
        return candle_entries(s, p.pattern, p.tf, p.confirm, p.rr)
    fn.__name__ = f"cs_{pattern}_{TF_LABEL[tf]}" + ("_conf" if confirm else "")
    return fn


def _make_daily(pattern: str, confirm: bool, sides: tuple):
    def fn(d, p):
        return daily_signal(d, p.pattern, p.confirm, sides, p.rr)
    fn.__name__ = f"cs_{pattern}_d" + ("_conf" if confirm else "")
    return fn


def _desc(pattern: str, tf, confirm: bool) -> str:
    conf = ", after a confirmation candle" if confirm else ""
    if tf == "d":
        return (f"{_NICE[pattern]} on daily candles{conf}: enter next open, stop beyond the pattern extreme "
                f"(>= {MIN_STOP_ATR} ATR), {RR:g}R, max {MAX_HOLD} sessions")
    last = "14:15" if tf == 60 else "14:30"
    return (f"{_NICE[pattern]} on {tf}m candles{conf}: enter next candle open (until {last}), stop beyond the "
            f"pattern extreme (>= {MIN_STOP_ATR} ATR), {RR:g}R, square-off 15:15")


IDS: list = []
for _pat, _spec in PATTERNS.items():
    for _conf in ((False, True) if _pat in CONFIRM else (False,)):
        _params = {"pattern": _pat, "confirm": _conf, "rr": RR, "min_stop_atr": MIN_STOP_ATR}
        for _tf in (5, 15, 60):
            _id = f"cs_{_pat}_{TF_LABEL[_tf]}" + ("_conf" if _conf else "")
            strategy(_id, "intraday", "candlestick", _desc(_pat, _tf, _conf), dict(_params, tf=_tf))(
                _make_intraday(_pat, _tf, _conf))
            IDS.append(_id)
        _id = f"cs_{_pat}_d" + ("_conf" if _conf else "")
        if _spec.side > 0:
            strategy(_id, "swing", "candlestick", _desc(_pat, "d", _conf), _params)(_make_daily(_pat, _conf, (LONG,)))
        else:
            _sides = (SHORT,) if _spec.side < 0 else (LONG, SHORT)
            strategy(_id, "futures", "candlestick", _desc(_pat, "d", _conf), _params, horizon="positional")(
                _make_daily(_pat, _conf, _sides))
        IDS.append(_id)

N_REGISTERED = len(IDS)
