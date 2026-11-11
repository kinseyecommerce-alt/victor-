"""
Reference implementations for the other kinds, so the swing / futures paths of the simulator are exercised
by real strategies. One of each; the full swing and futures batches belong in their own files.
"""
from __future__ import annotations

import numpy as np

from ..api import LONG, SHORT, Swing, strategy
from ..indicators import highest, lowest
from .intraday import orb_entries


@strategy("swing_20d_breakout", "swing", "breakout",
          "Close above the highest high of the previous 20 sessions: buy next open, 5 % stop, 10 % target, 10 sessions",
          {"n": 20, "stop_pct": 5.0, "target_pct": 10.0, "max_hold": 10})
def swing_20d_breakout(d, p):
    if len(d) < p.n + 1:
        return None
    if d.c[-1] > d.h[-p.n - 1:-1].max():
        return Swing(LONG, stop_pct=p.stop_pct, target_pct=p.target_pct, max_hold=p.max_hold)
    return None


@strategy("fut_donchian20", "futures", "breakout",
          "20-session Donchian on closes, both directions: enter next open, 4 % stop, 8 % target, 10 sessions",
          {"n": 20, "stop_pct": 4.0, "target_pct": 8.0, "max_hold": 10}, horizon="positional")
def fut_donchian20(d, p):
    if len(d) < p.n + 1:
        return None
    hi, lo = highest(d.h, p.n)[-2], lowest(d.l, p.n)[-2]
    if np.isfinite(hi) and d.c[-1] > hi:
        return Swing(LONG, stop_pct=p.stop_pct, target_pct=p.target_pct, max_hold=p.max_hold)
    if np.isfinite(lo) and d.c[-1] < lo:
        return Swing(SHORT, stop_pct=p.stop_pct, target_pct=p.target_pct, max_hold=p.max_hold)
    return None


@strategy("fut_orb15_break", "futures", "orb",
          "Stock-futures version of the 15-minute range break (futures costs), stop at the other side, 1.5R",
          {"minutes": 15, "rr": 1.5, "last": "12:00"}, horizon="intraday")
def fut_orb15_break(s, p):
    return orb_entries(s, p.minutes, p.rr, p.last)
