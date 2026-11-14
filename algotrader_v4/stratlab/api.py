"""
stratlab/api.py — what a strategy file imports: the `@strategy` decorator, `Entry` / `Swing`, small helpers.

    from stratlab.api import strategy, Entry, hhmm, first_true, cross_up

See README.md for the worked example.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .data import hhmm  # noqa: F401  (re-exported)

KINDS = ("intraday", "scalping", "swing", "futures")
LONG, SHORT = 1, -1
UNDERLYINGS = ("NIFTY", "BANKNIFTY")   # index futures the lab can simulate (on index minute bars)


@dataclass
class Entry:
    """An intraday entry decided at the close of bar `i`; it is filled at the open of bar i+1.

    side      +1 long, -1 short
    stop      stop price (must be on the losing side of the fill, otherwise the trade is skipped)
    target    target price, or leave None and give `rr`: target = fill ± rr × (fill − stop)
    time_stop exit at the open of the first bar this many minutes after the fill (None = hold to square-off)
    trail     None, ("pct", x): trail the stop x % behind the best price; ("atr", m): m × the 5-minute ATR at
              entry behind the best price; ("be", r): move the stop to the entry price once r × risk in profit
    """
    i: int
    side: int
    stop: float
    target: float | None = None
    rr: float | None = None
    time_stop: int | None = None
    trail: tuple | None = None
    # Optional (round 5, default None = unchanged behaviour):
    # exit_sig  bool array over the session's bars: True at bar k = exit decided at the close of bar k, filled at
    #           the open of bar k+1 (market order, pays slippage). Only bars k >= the fill bar count. Must be causal
    #           (audited like the entry). A new entry decided at that same bar k may fill at the same open (reversal).
    # bracket   ("pts", stop, target) or ("pct", stop, target): stop / target placed at that distance from the FILL
    #           price (points, or % of the fill); replaces `stop` / `target`. A stop distance of None keeps the
    #           absolute `stop` price (fill-relative target, fixed stop).
    exit_sig: object = None
    bracket: tuple | None = None


@dataclass
class Swing:
    """A multi-day entry decided on the latest daily bar of the view.

    Give stops/targets as a percentage of the fill price (stop_pct=3 means 3 % against you) or as prices.
    rr (with no target): target = fill ± rr × |fill − stop|, like Entry.rr.
    max_hold = number of sessions the position may be held; it is closed at that session's close."""
    side: int = LONG
    stop_pct: float | None = None
    target_pct: float | None = None
    stop: float | None = None
    target: float | None = None
    max_hold: int = 10
    rr: float | None = None
    # Optional (round 5, default None = unchanged behaviour):
    # exit_fn  callable(DailyView) -> bool, called at the close of every held session (the view ends at that
    #          session); True = exit at the next session's open (market order, pays slippage). After such an exit the
    #          strategy is asked again on the signal session, so an opposite entry fills at the same open (reversal).
    # lots     index futures only: lots for this trade instead of the spec's `lots`.
    exit_fn: object = None
    lots: int | None = None
    # Optional (round 7, default None = unchanged behaviour: market entry at the next open):
    # entry    a resting order for the next `valid` sessions instead of the market entry (next-open specs only):
    #          ("stop", level) buy stop (sell stop for a short): fills when the session trades at / through `level`,
    #             at max(open, level) for a long (min for a short), paying slippage;
    #          ("stop", level, arm) the stop becomes active only once the session trades beyond `arm` (below it for a
    #             long, above for a short), e.g. Turtle Soup; fills at `level` (+ slippage). Daily bars cannot tell
    #             whether the high came before the low, so a session that reaches both `arm` and `level` counts as
    #             filled (the pessimistic reading: the unknown cases close below the level);
    #          ("limit", level) buy limit (sell limit for a short): fills only if the session trades one tick through
    #             `level`, at min(open, level) for a long (max for a short), no slippage (a resting limit).
    #          Not filled after `valid` sessions: no trade, the strategy is asked again on the next session.
    #          On the fill session only the stop is checked (pessimistically: a low at / below the stop is taken to
    #          come after the fill); targets and exit_fn start at that session's close / the next session.
    # entry_stop_ticks  stop = min(stop, fill-session low) - ticks x tick for a long (max(stop, high) + ticks x tick
    #          for a short), fixed at the fill session's close ("one tick under today's low"); stop may be None.
    entry: tuple | None = None
    valid: int = 1
    entry_stop_ticks: int | None = None


@dataclass
class Spec:
    id: str
    kind: str
    family: str
    desc: str
    fn: object
    params: dict = field(default_factory=dict)
    style: str = "vector"             # intraday: "vector" (whole session, audited) or "bar" (called per bar)
    max_per_day: int = 1              # intraday: max trades per symbol per day
    horizon: str = "intraday"         # "intraday" or "positional" (derived from kind; futures must say)
    decision_time: int | None = None  # positional: decide and enter at this minute of the day (None = next open)
    needs_index: bool = False
    module: str = ""
    underlying: str | None = None     # futures only: "NIFTY" / "BANKNIFTY" = index future on the index's bars
    lots: int = 1                     # index futures: lots per trade (sim.LOT_SIZES)

    @property
    def p(self) -> SimpleNamespace:
        return SimpleNamespace(**self.params)

    @property
    def cost_kind(self) -> str:
        return {"swing": "equity_cnc", "futures": "future"}.get(self.kind, "equity_mis")

    @property
    def fingerprint(self) -> str:
        try:
            src = inspect.getsource(self.fn)
        except Exception:
            src = repr(self.fn)
        parts = (self.id, self.kind, self.horizon, self.style, self.max_per_day, self.decision_time,
                 sorted(self.params.items()), src)
        if self.underlying:                 # stock strategies keep their earlier fingerprints
            parts = parts + (self.underlying, self.lots)
        blob = repr(parts)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


REGISTRY: dict[str, Spec] = {}


def strategy(id: str, kind: str, family: str, desc: str, params: dict | None = None, *, style: str = "vector",
             max_per_day: int | None = None, horizon: str | None = None, decision_time: str | None = None,
             needs_index: bool = False, underlying: str | None = None, lots: int = 1):
    """Register a strategy function `fn(s, p)`.

    kind="futures" with underlying="NIFTY" or "BANKNIFTY": an index future, simulated on that index's minute bars
    (not on the stocks), `lots` lots per trade, futures costs, index slippage. Without `underlying` a futures
    strategy is a stock future simulated on each stock's cash bars.

    intraday / scalping / futures(horizon="intraday"):
        style="vector": fn gets the whole Session, returns a list of Entry (any order). Audited for look-ahead.
        style="bar":    fn gets a View ending at the current bar, returns one Entry (its `i` is ignored) or None.
    swing / futures(horizon="positional"):
        fn gets a DailyView ending at the decision day, returns a Swing or None.
    """
    if kind not in KINDS:
        raise ValueError(f"{id}: kind must be one of {KINDS} (options need premium data the lab does not have)")
    if kind == "futures" and horizon not in ("intraday", "positional"):
        raise ValueError(f"{id}: futures strategies must say horizon='intraday' or 'positional'")
    hz = "positional" if kind == "swing" else ("intraday" if kind in ("intraday", "scalping") else horizon)
    if style not in ("vector", "bar"):
        raise ValueError(f"{id}: style must be 'vector' or 'bar'")
    if not desc or "\n" in desc:
        raise ValueError(f"{id}: give a one-line description")
    if underlying is not None:
        if kind != "futures" or underlying not in UNDERLYINGS:
            raise ValueError(f"{id}: underlying= is for kind='futures' and one of {UNDERLYINGS}")
        if int(lots) < 1:
            raise ValueError(f"{id}: lots must be >= 1")

    def deco(fn):
        if id in REGISTRY and REGISTRY[id].fn.__code__ is not fn.__code__:
            raise ValueError(f"duplicate strategy id {id!r}")
        REGISTRY[id] = Spec(
            id=id, kind=kind, family=family, desc=desc, fn=fn, params=dict(params or {}), style=style,
            max_per_day=max_per_day if max_per_day is not None else (3 if kind == "scalping" else 1),
            horizon=hz, decision_time=hhmm(decision_time) if decision_time else None,
            needs_index=needs_index, module=getattr(fn, "__module__", ""), underlying=underlying, lots=int(lots))
        return fn
    return deco


# ── helpers for vectorised strategies ───────────────────────────────────────
def first_true(mask, start: int = 0, stop: int | None = None) -> int:
    """Index of the first True in mask[start:stop], or -1."""
    m = np.asarray(mask[start:stop])
    if m.size == 0:
        return -1
    k = int(m.argmax())
    return start + k if m[k] else -1


def cross_up(a, b) -> np.ndarray:
    """True at i where a crosses above b (a[i] > b[i] and a[i-1] <= b[i-1]). b may be a scalar."""
    a = np.asarray(a, float)
    b = np.broadcast_to(np.asarray(b, float), a.shape)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def cross_dn(a, b) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.broadcast_to(np.asarray(b, float), a.shape)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


def between(s, start: str, end: str) -> np.ndarray:
    """Mask of bars whose time is in [start, end) given as 'HH:MM'."""
    return (s.t >= hhmm(start)) & (s.t < hhmm(end))


def runlen(mask) -> np.ndarray:
    """Number of consecutive True values ending at each index (0 where False)."""
    m = np.asarray(mask, bool)
    idx = np.arange(len(m))
    return idx - np.maximum.accumulate(np.where(~m, idx, -1))


def shift(x, k: int = 1) -> np.ndarray:
    """x delayed by k bars (the first k values are NaN): shift(x)[i] == x[i-1]."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def prev_high(h, n: int) -> np.ndarray:
    """Highest value of the n bars BEFORE each bar (NaN until n bars exist)."""
    from .indicators import highest
    return shift(highest(h, n))


def prev_low(l, n: int) -> np.ndarray:
    from .indicators import lowest
    return shift(lowest(l, n))
