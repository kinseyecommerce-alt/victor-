"""
strategies/base.py
Shared types for the daily-bar positional strategies.

All three strategies are pure signal generators: they hold per-instrument
state (open position, filters) but perform no I/O. The caller feeds them
one completed daily bar history at EOD and acts on the returned Signal
next-day at open (per the spec: "signal EOD, next-day open execute").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from strategies.indicators import DailyBar


class Action(str, Enum):
    ENTER_LONG  = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    PYRAMID     = "PYRAMID"      # add one unit to an existing position
    EXIT        = "EXIT"         # close the whole position
    HOLD        = "HOLD"


@dataclass
class Signal:
    action:  Action
    symbol:  str
    reason:  str = ""
    price:   float = 0.0          # reference (signal-day close)
    stop:    Optional[float] = None   # protective stop for the position
    n_atr:   Optional[float] = None   # ATR at signal time (sizing input)
    system:  str = ""             # e.g. "S1", "S2", "MA_XOVER", "TSMOM"
    weight:  Optional[float] = None   # TSMOM vol-target weight (∝ 1/vol)


@dataclass
class OpenPosition:
    symbol:       str
    side:         str                 # "LONG" | "SHORT"
    system:       str = ""
    entries:      list[float] = field(default_factory=list)   # entry price per unit
    stop:         float = 0.0
    entry_n:      float = 0.0         # N (ATR) at initial entry — pyramid spacing
    last_add_ref: float = 0.0         # price of the most recent unit entry

    @property
    def units(self) -> int:
        return len(self.entries)


class PositionalStrategy:
    """Interface all three strategies implement."""
    name: str = "base"

    def evaluate(self, symbol: str, bars: list[DailyBar]) -> Signal:
        raise NotImplementedError

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        raise NotImplementedError
