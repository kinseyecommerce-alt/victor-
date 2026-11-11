"""
options_replay.py — research-only replay of index-option strategies on real index minute bars.

Kite serves no history for expired option contracts, so option prices here are MODELLED:
Black-Scholes on the underlying's real minute bars, with India VIX (close of the previous session)
as a flat implied vol for every strike. Nothing in this module is imported by the live agents;
it places no orders and calls no API. Its numbers are evidence for or against an idea, not a fill log.

Assumptions (all overridable from the CLI, none fitted to data):
  pricing     Black-Scholes (greeks_engine.bs_price), r = 6.5 %, calendar time to 15:30 IST on
              expiry day (--time-basis trading counts open-market minutes only, 375 x 252 a year),
              one vol for all strikes (no smile/skew) and constant within a session.
  vol         India VIX close of the previous session / 100. Without a VIX series: 20-session
              close-to-close realised vol (min 10 returns) x 1.1, annualised with sqrt(252).
  expiry      NIFTY: weekly, Tuesday from 2025-09-01 (Thursday before). BANKNIFTY: monthly only
              (NSE dropped its weekly series in Nov 2024), last Tuesday from 2025-09-01 (last
              Thursday before). An expiry on a holiday moves to the previous trading day. As in
              options_resolver, a contract expiring today is skipped unless --allow-0dte.
  lots        options_resolver._SIM_LOT (NIFTY 75, BANKNIFTY 30) else 75 / 35; --lot-size overrides.
              Strikes on 50 (NIFTY) / 100 (BANKNIFTY). One position open at a time per run.
  schedule    each decision time is used once per session (no re-entry after a mid-day exit);
              no entries on weekend special sessions (Muhurat), whose bars still mark positions.
  fills       a decision at minute t sees only bars stamped before t and fills at bar t's open.
              Every leg pays a haircut per fill: max(min_pts, pct x model premium), buys above the
              model and sells below it (floored at one tick).
  exits       targets are checked on the bar close; stops on the worst model value inside the bar
              (its high, low, close and any strike it crossed) and fill at that worst value.
              Expiry settles at intrinsic against the mean close of 15:00-15:29 (NSE settles on a
              last-30-minute average), with no haircut.
  costs       costs.round_trip_cost(..., "option") per leg (brokerage, STT, exchange, SEBI, stamp,
              GST). A long leg settled in the money pays exercise STT of 0.15 % of intrinsic (2026),
              the same rate round_trip_cost already charges on the settlement value, so nothing extra.
  lot sizes   one lot size for every date (--lot-size), or ReplayConfig.lot_schedule = ((from_date, lot), ...)
              for the size in force on each entry date.

Usage:
  python options_replay.py --strategy IRON_CONDOR --data <index.json | bars12m dir>
  python options_replay.py --strategy CREDIT_SPREAD --direction first_hour --data ...
  python options_replay.py --strategy LONG_STRADDLE --event-dates 2026-10-08,2026-12-04 --data ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Optional

from costs import round_trip_cost
from greeks_engine import _N, _d1d2, bs_price

RISK_FREE = 0.065
TICK = 0.05
MINUTES_PER_YEAR = 365 * 24 * 60
EXPIRY_CLOSE = time(15, 30)
SETTLE_FROM = time(15, 0)
# Exercise STT (costs.STT_OPTION_EXERCISE, 0.15 % of intrinsic) minus the option sell-side STT that
# round_trip_cost already charges on the settlement value (0.15 % since Budget 2026) = 0.
EXERCISE_STT_EXTRA = 0.0015 - 0.0015
WEEKLY_SWITCH = date(2025, 9, 1)              # NSE moved index expiries from Thursday to Tuesday

STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100}
_FALLBACK_LOT = {"NIFTY": 75, "BANKNIFTY": 35}
try:                                           # the repo's own paper-sim lot sizes, when importable
    from options_resolver import _SIM_LOT as _REPO_LOT
except Exception:                              # pragma: no cover - research module stays standalone
    _REPO_LOT = {}
LOT_SIZE = {u: int(_REPO_LOT.get(u, _FALLBACK_LOT[u])) for u in _FALLBACK_LOT}
LOT_SOURCE = "options_resolver._SIM_LOT" if _REPO_LOT else "fallback"

SLIP_MIN_PTS = {"NIFTY": 1.0, "BANKNIFTY": 2.0}
SLIP_PCT = 0.02

# data-file keys / filename fragments → underlying
_INDEX_KEYS = {"NIFTY 50": "NIFTY", "NIFTY50": "NIFTY", "NIFTY": "NIFTY",
               "NIFTY BANK": "BANKNIFTY", "NIFTYBANK": "BANKNIFTY", "BANKNIFTY": "BANKNIFTY"}
_VIX_KEYS = ("INDIA VIX", "INDIAVIX", "VIX")


# ══════════════════════════════════════════════════════════════════════════
# Pricing
# ══════════════════════════════════════════════════════════════════════════

def bs(S: float, K: float, T: float, sigma: float, opt: str, r: float = RISK_FREE) -> float:
    return max(0.0, bs_price(S, K, T, r, sigma, opt))


def bs_delta(S: float, K: float, T: float, sigma: float, opt: str, r: float = RISK_FREE) -> float:
    if T <= 0 or sigma <= 0:
        itm = S > K if opt == "CE" else S < K
        return (1.0 if opt == "CE" else -1.0) if itm else 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return _N(d1) if opt == "CE" else _N(d1) - 1.0


def years_between(now: datetime, expiry: date) -> float:
    end = datetime.combine(expiry, EXPIRY_CLOSE)
    return max((end - now).total_seconds() / 60.0, 1.0) / MINUTES_PER_YEAR


SESSION_MIN = 375                                  # 09:15-15:30


def trading_years_between(now: datetime, expiry: date, is_td: Callable[[date], bool]) -> float:
    """Trading-time clock: open-market minutes left (375 per session, 252 sessions a year).
    Puts all decay inside market hours; calendar time spreads it over nights and weekends."""
    close_today = datetime.combine(now.date(), EXPIRY_CLOSE)
    mins = max(0.0, (close_today - now).total_seconds() / 60.0)
    d = now.date() + timedelta(days=1)
    while d <= expiry:
        if is_td(d):
            mins += SESSION_MIN
        d += timedelta(days=1)
    return max(mins, 1.0) / (252 * SESSION_MIN)


def round_strike(x: float, step: int) -> float:
    return float(round(x / step) * step)


def haircut(model: float, und: str, slip_min: Optional[float] = None, slip_pct: float = SLIP_PCT) -> float:
    m = SLIP_MIN_PTS.get(und, 1.0) if slip_min is None else slip_min
    return max(m, slip_pct * model)


def fill_price(model: float, side: int, und: str, slip_min: Optional[float] = None,
               slip_pct: float = SLIP_PCT) -> float:
    """side +1 = we buy (pay above model), -1 = we sell (receive below model)."""
    h = haircut(model, und, slip_min, slip_pct)
    return model + h if side > 0 else max(TICK, model - h)


def leg_cost(entry_px: float, exit_px: float, side: int, qty: int, settled: bool = False) -> float:
    """₹ costs of one leg's round trip (side = opening side), via costs.round_trip_cost."""
    ev, xv = entry_px * qty, exit_px * qty
    buy_v, sell_v = (ev, xv) if side > 0 else (xv, ev)
    c = round_trip_cost(buy_v, sell_v, "option")
    if settled and side > 0 and exit_px > 0:
        c += round(xv * EXERCISE_STT_EXTRA, 2)
    return c


# ══════════════════════════════════════════════════════════════════════════
# Calendar / expiry
# ══════════════════════════════════════════════════════════════════════════

def _expiry_weekday(d: date) -> int:
    return 1 if d >= WEEKLY_SWITCH else 3          # Tuesday / Thursday


def _calendar_trading_day(d: date) -> bool:
    try:
        from ist_clock import is_trading_day
        return is_trading_day(d)
    except Exception:                               # pragma: no cover
        return d.weekday() < 5


def make_trading_day_fn(sessions: list[date]) -> Callable[[date], bool]:
    """Inside the data range a weekday with no bars is treated as a holiday; outside it, ist_clock."""
    known = set(sessions)
    lo, hi = (min(sessions), max(sessions)) if sessions else (None, None)

    def fn(d: date) -> bool:
        if lo and lo <= d <= hi:
            return d in known
        return _calendar_trading_day(d)
    return fn


def _adjust_back(d: date, is_td: Callable[[date], bool]) -> date:
    for _ in range(10):
        if is_td(d):
            return d
        d -= timedelta(days=1)
    return d


def _last_weekday_of_month(y: int, m: int, wd: int) -> date:
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    while d.weekday() != wd:
        d -= timedelta(days=1)
    return d


def next_expiry(und: str, today: date, is_td: Callable[[date], bool], skip_today: bool = True) -> date:
    """NIFTY → next weekly expiry; BANKNIFTY → next monthly expiry; holiday-adjusted."""
    for k in range(0, 120 if und == "BANKNIFTY" else 400):
        if und == "BANKNIFTY":
            y, m = today.year + (today.month - 1 + k) // 12, (today.month - 1 + k) % 12 + 1
            nominal = _last_weekday_of_month(y, m, _expiry_weekday(date(y, m, 28)))
        else:
            # day by day, so the weekday rule is read on the expiry date itself: from Fri 2025-08-29 the
            # next expiry is Tue 2025-09-02 (the first Tuesday expiry), not Thu 2025-09-04
            nominal = today + timedelta(days=k)
            if nominal.weekday() != _expiry_weekday(nominal):
                continue
        exp = _adjust_back(nominal, is_td)
        if exp > today or (exp == today and not skip_today):
            return exp
    raise ValueError(f"no expiry found for {und} after {today}")


# ══════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════

Bar = tuple  # (datetime, o, h, l, c)


def _parse_bars(rows) -> list[Bar]:
    out = []
    for r in rows:
        ts = datetime.fromisoformat(str(r[0]).replace("Z", "+00:00"))
        if ts.tzinfo is not None:                   # keep IST wall-clock time, drop tz
            ts = ts.replace(tzinfo=None)
        out.append((ts, float(r[1]), float(r[2]), float(r[3]), float(r[4])))
    out.sort(key=lambda b: b[0])
    return out


def _daily_closes(bars: list[Bar]) -> dict[date, float]:
    d: dict[date, float] = {}
    for b in bars:
        d[b[0].date()] = b[4]
    return d


def load_data(path: str) -> tuple[dict[str, list[Bar]], Optional[dict[date, float]]]:
    """→ ({underlying: minute bars}, {date: VIX close} or None).

    Accepts the index.json layout {"NIFTY 50": {"bars": [...]}, ...} or a directory of per-symbol
    files {"symbol", "interval", "bars"} named like _NIFTY50_min.json / _VIX_day.json."""
    und: dict[str, list[Bar]] = {}
    vix: Optional[dict[date, float]] = None
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.json")))
    for f in files:
        with open(f) as fh:
            js = json.load(fh)
        if isinstance(js, dict) and "bars" in js:   # one symbol per file
            name = os.path.basename(f).upper()
            sym = str(js.get("symbol", "")).upper()
            if "VIX" in name or "VIX" in sym:
                vix = _daily_closes(_parse_bars(js["bars"]))
                continue
            if "_DAY" in name or str(js.get("interval", "")).lower() in ("day", "1d", "daily"):
                continue
            key = sym.replace(" ", "") or name
            for frag, u in (("NIFTYBANK", "BANKNIFTY"), ("BANKNIFTY", "BANKNIFTY"), ("NIFTY50", "NIFTY")):
                if frag in key.replace(" ", "") or frag in name:
                    und[u] = _parse_bars(js["bars"])
                    break
        elif isinstance(js, dict):                  # index.json: {name: {"bars": [...]}}
            for k, v in js.items():
                rows = v.get("bars") if isinstance(v, dict) else v
                if not rows:
                    continue
                if k.upper() in _VIX_KEYS:
                    vix = _daily_closes(_parse_bars(rows))
                elif k.upper() in _INDEX_KEYS:
                    und[_INDEX_KEYS[k.upper()]] = _parse_bars(rows)
    return und, vix


def realised_vol(closes: list[float], lookback: int = 20, min_returns: int = 10) -> Optional[float]:
    c = closes[-(lookback + 1):]
    rets = [math.log(c[i] / c[i - 1]) for i in range(1, len(c)) if c[i - 1] > 0]
    if len(rets) < min_returns:
        return None
    mu = sum(rets) / len(rets)
    var = sum((x - mu) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var * 252)


# ══════════════════════════════════════════════════════════════════════════
# What a strategy may see
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MarketView:
    """Everything a strategy may use at `now`. `bars` holds only bars stamped before `now`
    (a copy, so later bars are unreachable); `spot` is the open of the bar at `now`."""
    underlying: str
    now: datetime
    spot: float
    bars: list
    prior_closes: list               # [(date, close)] of completed sessions before today
    sigma: float
    expiry: date
    step: int
    T: float                         # years to expiry on the replay's clock
    r: float = RISK_FREE
    is_td: Optional[Callable[[date], bool]] = None    # the replay's trading-day calendar

    def today_bars(self) -> list:
        return [b for b in self.bars if b[0].date() == self.now.date()]

    def atm(self) -> float:
        return round_strike(self.spot, self.step)

    def delta(self, opt: str, K: float) -> float:
        return bs_delta(self.spot, K, self.T, self.sigma, opt, self.r)

    def strike_for_delta(self, opt: str, target: float, max_steps: int = 60) -> float:
        """OTM strike whose |delta| is closest to `target`."""
        atm, sgn = self.atm(), (1 if opt == "CE" else -1)
        best, best_d = atm, 9.0
        for i in range(0, max_steps + 1):
            k = atm + sgn * i * self.step
            d = abs(abs(self.delta(opt, k)) - target)
            if d < best_d:
                best, best_d = k, d
        return best


@dataclass
class LegSpec:
    opt: str            # "CE" / "PE"
    strike: float
    side: int           # +1 buy, -1 sell


# ══════════════════════════════════════════════════════════════════════════
# Strategies
# ══════════════════════════════════════════════════════════════════════════

class OptionStrategy:
    """Decide at `decision_times` from a MarketView; the engine fills, marks and exits.

    hold = "intraday": square off at `time_exit` the same session.
    hold = "expiry":   carry until a stop/target or expiry settlement.
    Credit positions: target when P&L >= target_frac x credit; stop when the cost to buy back
    reaches stop_mult x credit (premium doubled for stop_mult=2). Debit positions: target when
    P&L >= target_frac x debit, stop when P&L <= -stop_frac x debit."""
    name = "BASE"
    decision_times: list = [time(10, 20)]
    hold = "intraday"
    time_exit = time(15, 15)
    target_frac = 0.5
    stop_mult = 2.0
    stop_frac = 0.5
    exits_early = True               # False: hold to expiry, never checked for stop/target (faster)

    def decide(self, view: MarketView) -> Optional[list[LegSpec]]:
        raise NotImplementedError

    def exit_reason(self, entry_net: float, pnl_close: float, close_cost_worst: float,
                    pnl_worst: float) -> Optional[str]:
        """entry_net > 0 = credit received per unit; costs are per unit of quantity."""
        if entry_net > 0:
            if close_cost_worst >= self.stop_mult * entry_net:
                return "stop"
            if pnl_close >= self.target_frac * entry_net:
                return "target"
        else:
            debit = -entry_net
            if pnl_worst <= -self.stop_frac * debit:
                return "stop"
            if pnl_close >= self.target_frac * debit:
                return "target"
        return None


class IronCondor(OptionStrategy):
    name = "IRON_CONDOR"

    def __init__(self, entry: time = time(10, 20), hold: str = "intraday", short_delta: float = 0.175,
                 wing_pts: Optional[float] = None, target_frac: float = 0.5, stop_mult: float = 2.0,
                 time_exit: time = time(15, 15)):
        self.decision_times = [entry]
        self.hold, self.short_delta, self.wing_pts = hold, short_delta, wing_pts
        self.target_frac, self.stop_mult, self.time_exit = target_frac, stop_mult, time_exit

    def decide(self, v: MarketView):
        wing = self.wing_pts or 3 * v.step                        # 150 NIFTY / 300 BANKNIFTY
        kc = v.strike_for_delta("CE", self.short_delta)
        kp = v.strike_for_delta("PE", self.short_delta)
        return [LegSpec("CE", kc, -1), LegSpec("CE", kc + wing, +1),
                LegSpec("PE", kp, -1), LegSpec("PE", kp - wing, +1)]


class CreditSpread(OptionStrategy):
    """BULL_PUT when the signal is up, BEAR_CALL when it is down.
    direction="first_hour": sign of the 09:15-10:14 move (decides at 10:15).
    direction="trend20":    previous close vs its 20-session mean."""
    name = "CREDIT_SPREAD"

    def __init__(self, direction: str = "first_hour", entry: Optional[time] = None, hold: str = "intraday",
                 short_delta: float = 0.25, width_pts: Optional[float] = None, target_frac: float = 0.5,
                 stop_mult: float = 2.0, time_exit: time = time(15, 15)):
        self.direction = direction
        self.decision_times = [entry or time(10, 15)]
        self.hold, self.short_delta, self.width_pts = hold, short_delta, width_pts
        self.target_frac, self.stop_mult, self.time_exit = target_frac, stop_mult, time_exit
        self.last_kind = ""

    def signal(self, v: MarketView) -> int:
        if self.direction == "trend20":
            closes = [c for _, c in v.prior_closes][-20:]
            if len(closes) < 20:
                return 0
            return 1 if closes[-1] > sum(closes) / len(closes) else -1
        tb = [b for b in v.today_bars() if b[0].time() < time(10, 15)]
        if len(tb) < 30:
            return 0
        mv = tb[-1][4] - tb[0][1]
        return 1 if mv > 0 else -1 if mv < 0 else 0

    def decide(self, v: MarketView):
        s = self.signal(v)
        if s == 0:
            return None
        w = self.width_pts or 2 * v.step                         # 100 NIFTY / 200 BANKNIFTY
        if s > 0:
            self.last_kind = "BULL_PUT"
            k = v.strike_for_delta("PE", self.short_delta)
            return [LegSpec("PE", k, -1), LegSpec("PE", k - w, +1)]
        self.last_kind = "BEAR_CALL"
        k = v.strike_for_delta("CE", self.short_delta)
        return [LegSpec("CE", k, -1), LegSpec("CE", k + w, +1)]


class LongStraddle(OptionStrategy):
    """Buy the ATM call and put on listed event days (or every session with event_dates=None,
    as an unconditional baseline). Intraday by default."""
    name = "LONG_STRADDLE"

    def __init__(self, event_dates: Optional[set] = None, entry: time = time(9, 20), hold: str = "intraday",
                 target_frac: float = 0.5, stop_frac: float = 0.4, time_exit: time = time(15, 15)):
        self.event_dates = event_dates
        self.decision_times = [entry]
        self.hold, self.target_frac, self.stop_frac, self.time_exit = hold, target_frac, stop_frac, time_exit

    def decide(self, v: MarketView):
        if self.event_dates is not None and v.now.date() not in self.event_dates:
            return None
        k = v.atm()
        return [LegSpec("CE", k, +1), LegSpec("PE", k, +1)]


class ShortStraddle(OptionStrategy):
    """Sell the ATM call and put (undefined risk). stop_mult / target_frac as for any credit position;
    pass a large stop and target (e.g. 99) for a pure hold to expiry / time exit."""
    name = "SHORT_STRADDLE"

    def __init__(self, entry: time = time(10, 20), hold: str = "expiry", target_frac: float = 0.5,
                 stop_mult: float = 2.0, time_exit: time = time(15, 15)):
        self.decision_times = [entry]
        self.hold, self.target_frac, self.stop_mult, self.time_exit = hold, target_frac, stop_mult, time_exit

    def decide(self, v: MarketView):
        k = v.atm()
        return [LegSpec("CE", k, -1), LegSpec("PE", k, -1)]


def previous_trading_day(d: date, is_td: Callable[[date], bool]) -> date:
    p = d - timedelta(days=1)
    for _ in range(15):
        if is_td(p):
            return p
        p -= timedelta(days=1)
    return p


def trading_days_left(d: date, expiry: date, is_td: Callable[[date], bool]) -> int:
    """Trading sessions after `d` up to and including `expiry`."""
    n, x = 0, d + timedelta(days=1)
    while x <= expiry:
        n += bool(is_td(x))
        x += timedelta(days=1)
    return n


class ShortStrangle(OptionStrategy):
    """Sell the OTM call and put whose |delta| is closest to short_delta (undefined risk).

    wing_pts:      also buy the CE at (short CE + wing_pts) and the PE at (short PE - wing_pts): a defined-risk
                   iron condor with far wings.
    weekly_cycle:  enter only on the first trading day after an expiry (the previous trading day was an expiry
                   of this underlying), and skip that week when fewer than `min_days_left` trading days remain
                   after the entry day up to and including the expiry. A skipped week is not entered later.
    target_frac / stop_mult = None: no target / no stop (hold to expiry settlement)."""
    name = "SHORT_STRANGLE"

    def __init__(self, entry: time = time(10, 20), hold: str = "expiry", short_delta: float = 0.16,
                 target_frac: Optional[float] = 0.5, stop_mult: Optional[float] = 2.0,
                 time_exit: time = time(15, 15), wing_pts: Optional[float] = None,
                 weekly_cycle: bool = False, min_days_left: int = 0):
        self.decision_times = [entry]
        self.hold, self.short_delta = hold, short_delta
        self.target_frac, self.stop_mult, self.time_exit = target_frac, stop_mult, time_exit
        self.wing_pts, self.weekly_cycle, self.min_days_left = wing_pts, weekly_cycle, min_days_left
        self.exits_early = not (hold == "expiry" and target_frac is None and stop_mult is None)
        self.skipped_short_weeks: list = []

    def exit_reason(self, entry_net, pnl_close, close_cost_worst, pnl_worst):
        if self.stop_mult is not None and close_cost_worst >= self.stop_mult * entry_net:
            return "stop"
        if self.target_frac is not None and pnl_close >= self.target_frac * entry_net:
            return "target"
        return None

    def in_cycle(self, v: MarketView) -> bool:
        is_td = v.is_td or _calendar_trading_day
        d = v.now.date()
        prev = previous_trading_day(d, is_td)
        if next_expiry(v.underlying, prev, is_td, skip_today=False) != prev:
            return False                                   # not the first session after an expiry
        if trading_days_left(d, v.expiry, is_td) < self.min_days_left:
            self.skipped_short_weeks.append(d)
            return False
        return True

    def decide(self, v: MarketView):
        if self.weekly_cycle and not self.in_cycle(v):
            return None
        kc = v.strike_for_delta("CE", self.short_delta)
        kp = v.strike_for_delta("PE", self.short_delta)
        legs = [LegSpec("CE", kc, -1), LegSpec("PE", kp, -1)]
        if self.wing_pts:
            legs += [LegSpec("CE", kc + self.wing_pts, +1), LegSpec("PE", kp - self.wing_pts, +1)]
        return legs


# ══════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Leg:
    opt: str
    strike: float
    side: int
    entry_px: float
    exit_px: float = 0.0
    cost: float = 0.0


@dataclass
class Trade:
    strategy: str
    kind: str
    underlying: str
    entry_ts: datetime
    expiry: date
    spot_in: float
    sigma: float
    vol_source: str
    qty: int
    legs: list
    entry_net: float                  # per unit: + credit received / - debit paid
    max_risk: float = 0.0             # ₹ for defined-risk structures, 0 if undefined
    exit_ts: Optional[datetime] = None
    slippage: float = 0.0             # ₹ paid to the per-leg haircut, entry + exit (inside gross)
    spot_out: float = 0.0
    reason: str = ""
    gross: float = 0.0
    costs: float = 0.0

    @property
    def net(self) -> float:
        return self.gross - self.costs


@dataclass
class ReplayConfig:
    underlying: str = "NIFTY"
    lots: int = 1
    lot_size: Optional[int] = None
    r: float = RISK_FREE
    slip_min: Optional[float] = None
    slip_pct: float = SLIP_PCT
    allow_0dte: bool = False
    rv_lookback: int = 20
    rv_min_returns: int = 10
    rv_mult: float = 1.1
    time_basis: str = "calendar"     # "calendar" | "trading"
    lot_schedule: Optional[tuple] = None   # ((from_date, lot size), ...): the size in force on the entry date


def lot_size_on(schedule: tuple, d: date) -> int:
    """Lot size in force on `d` from ((from_date, lot), ...); the earliest entry also applies before its date."""
    sch = sorted(schedule)
    lot = sch[0][1]
    for frm, n in sch:
        if d >= frm:
            lot = n
    return int(lot)


@dataclass
class ReplayResult:
    trades: list
    sessions: list
    vol_source_days: dict = field(default_factory=dict)
    skipped_no_vol: int = 0
    lot_size: int = 0


def _max_risk(legs: list[Leg], entry_net: float, qty: int) -> float:
    """Worst expiry loss of a defined-risk structure (0 if a short leg is uncovered)."""
    for opt in ("CE", "PE"):
        shorts = sum(1 for l in legs if l.opt == opt and l.side < 0)
        longs = sum(1 for l in legs if l.opt == opt and l.side > 0)
        if shorts > longs:
            return 0.0
    if entry_net <= 0:
        return -entry_net * qty
    widths = [abs(a.strike - b.strike) for a in legs for b in legs
              if a.opt == b.opt and a.side < 0 < b.side]
    return max(0.0, (max(widths) if widths else 0.0) - entry_net) * qty


class Replay:
    def __init__(self, bars: list[Bar], strategy: OptionStrategy, cfg: ReplayConfig,
                 vix: Optional[dict[date, float]] = None):
        self.bars, self.strat, self.cfg, self.vix = bars, strategy, cfg, vix
        self.und = cfg.underlying
        self.step = STRIKE_STEP[self.und]
        self.lot = cfg.lot_size or LOT_SIZE[self.und]
        self.qty = cfg.lots * self.lot
        self.is_td: Callable[[date], bool] = _calendar_trading_day

    def _T(self, now: datetime, expiry: date) -> float:
        if self.cfg.time_basis == "trading":
            return trading_years_between(now, expiry, self.is_td)
        return years_between(now, expiry)

    # ── vol ───────────────────────────────────────────────────────────────
    def _sigma(self, d: date, prior_closes: list) -> tuple[Optional[float], str]:
        if self.vix:
            prev = [k for k in self.vix if k < d]
            if prev:
                return self.vix[max(prev)] / 100.0, "vix"
            return None, "none"
        rv = realised_vol([c for _, c in prior_closes], self.cfg.rv_lookback, self.cfg.rv_min_returns)
        return (rv * self.cfg.rv_mult, "rv20x1.1") if rv else (None, "none")

    # ── valuation ─────────────────────────────────────────────────────────
    def _model(self, leg_opt: str, K: float, S: float, T: float, sigma: float) -> float:
        return bs(S, K, T, sigma, leg_opt, self.cfg.r)

    def _close_value(self, tr: Trade, S: float, now: datetime, sigma: float) -> tuple[float, list]:
        """Per-unit cost to close now at executable prices (+ = we pay) and the exit fills."""
        T = self._T(now, tr.expiry)
        pay, fills = 0.0, []
        for l in tr.legs:
            m = self._model(l.opt, l.strike, S, T, sigma)
            px = fill_price(m, -l.side, self.und, self.cfg.slip_min, self.cfg.slip_pct)
            fills.append((px, abs(px - m)))
            pay += px if l.side < 0 else -px
        return pay, fills

    def _close(self, tr: Trade, ts: datetime, S: float, fills: list, reason: str, settled: bool = False):
        tr.exit_ts, tr.spot_out, tr.reason = ts, S, reason
        gross = costs = 0.0
        for l, (px, slip) in zip(tr.legs, fills):
            tr.slippage += slip * tr.qty
            l.exit_px = px
            gross += (px - l.entry_px) * l.side * tr.qty
            l.cost = leg_cost(l.entry_px, px, l.side, tr.qty, settled=settled)
            costs += l.cost
        tr.gross, tr.costs, tr.slippage = round(gross, 2), round(costs, 2), round(tr.slippage, 2)

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self) -> ReplayResult:
        sessions = sorted({b[0].date() for b in self.bars})
        is_td = self.is_td = make_trading_day_fn(sessions)
        closes = _daily_closes(self.bars)
        by_day: dict[date, list[Bar]] = {}
        for b in self.bars:
            by_day.setdefault(b[0].date(), []).append(b)

        trades: list[Trade] = []
        vol_days: dict[str, int] = {}
        skipped = 0
        pos: Optional[Trade] = None
        history: list[Bar] = []                       # bars strictly before the current one
        for d in sessions:
            day = by_day[d]
            prior = [(k, closes[k]) for k in sessions if k < d]
            sigma, vsrc = self._sigma(d, prior)
            vol_days[vsrc] = vol_days.get(vsrc, 0) + 1
            if sigma is None:
                skipped += 1
            decided = set()
            for i, b in enumerate(day):
                ts, o, h, lo, c = b
                # 1) decisions at bar open, from bars before ts only. Each decision time is
                #    consumed by the first bar at/after it whether or not a position is open,
                #    so a mid-day exit never triggers an off-schedule re-entry. No entries on
                #    weekend special sessions (Muhurat); those bars are still used for marking.
                for t_dec in self.strat.decision_times:
                    if t_dec in decided or ts.time() < t_dec:
                        continue
                    decided.add(t_dec)
                    if pos is not None or sigma is None or d.weekday() >= 5:
                        continue
                    if self.strat.hold == "intraday" and ts.time() >= self.strat.time_exit:
                        continue
                    exp = next_expiry(self.und, d, is_td, skip_today=not self.cfg.allow_0dte)
                    view = MarketView(self.und, ts, o, list(history), list(prior), sigma, exp, self.step,
                                      self._T(ts, exp), self.cfg.r, is_td)
                    specs = self.strat.decide(view)
                    if specs:
                        pos = self._open(specs, view, vsrc)
                        trades.append(pos)
                # 2) manage the open position on this bar
                if pos is not None:
                    self._manage(pos, b, sigma, d, day, i)
                    if pos.exit_ts is not None:
                        pos = None
                history.append(b)
            # end of session: intraday positions must be flat already; expiry handled in _manage
        if pos is not None and pos.exit_ts is None:      # data ended with a position open
            last = self.bars[-1]
            sig = pos.sigma
            _, fills = self._close_value(pos, last[4], last[0], sig)
            self._close(pos, last[0], last[4], fills, "data_end")
        return ReplayResult(trades, sessions, vol_days, skipped, self.lot)

    def _open(self, specs: list[LegSpec], v: MarketView, vsrc: str) -> Trade:
        legs, net, slip = [], 0.0, 0.0
        qty = (self.cfg.lots * lot_size_on(self.cfg.lot_schedule, v.now.date())
               if self.cfg.lot_schedule else self.qty)
        for s in specs:
            m = self._model(s.opt, s.strike, v.spot, v.T, v.sigma)
            px = fill_price(m, s.side, self.und, self.cfg.slip_min, self.cfg.slip_pct)
            legs.append(Leg(s.opt, s.strike, s.side, px))
            net += -s.side * px
            slip += abs(px - m) * qty
        kind = getattr(self.strat, "last_kind", "") or self.strat.name
        tr = Trade(self.strat.name, kind, self.und, v.now, v.expiry, v.spot, v.sigma, vsrc,
                   qty, legs, round(net, 4))
        tr.max_risk = round(_max_risk(legs, net, qty), 2)
        tr.slippage = slip
        return tr

    def _manage(self, tr: Trade, b: Bar, sigma: Optional[float], d: date, day: list, i: int) -> None:
        ts, o, h, lo, c = b
        sig = sigma if sigma is not None else tr.sigma
        is_last_bar = i == len(day) - 1
        if d == tr.expiry and is_last_bar:                # expiry settlement
            settle_bars = [x[4] for x in day if x[0].time() >= SETTLE_FROM] or [c]
            S = sum(settle_bars) / len(settle_bars)
            fills = [(max(0.0, (S - l.strike) if l.opt == "CE" else (l.strike - S)), 0.0) for l in tr.legs]
            self._close(tr, ts, S, fills, "expiry", settled=True)
            return
        if self.strat.hold == "intraday" and ts.time() >= self.strat.time_exit:
            _, fills = self._close_value(tr, o, ts, sig)
            self._close(tr, ts, o, fills, "time")
            return
        if not getattr(self.strat, "exits_early", True):          # pure hold to expiry: nothing to check
            return
        # stop: worst executable value inside the bar
        candidates = {h, lo, c} | {l.strike for l in tr.legs if lo <= l.strike <= h}
        worst_pay, worst_fills, worst_S = -1e18, None, c
        for S in candidates:
            pay, fills = self._close_value(tr, S, ts, sig)
            if pay > worst_pay:
                worst_pay, worst_fills, worst_S = pay, fills, S
        pay_c, fills_c = self._close_value(tr, c, ts, sig)
        pnl_close = tr.entry_net - pay_c
        pnl_worst = tr.entry_net - worst_pay
        reason = self.strat.exit_reason(tr.entry_net, pnl_close, worst_pay, pnl_worst)
        if reason == "stop":
            self._close(tr, ts, worst_S, worst_fills, "stop")
        elif reason:
            self._close(tr, ts, c, fills_c, reason)
        elif self.strat.hold == "intraday" and is_last_bar:       # no bar at/after time_exit
            self._close(tr, ts, c, fills_c, "time")


# ══════════════════════════════════════════════════════════════════════════
# Metrics / output
# ══════════════════════════════════════════════════════════════════════════

def _stats(trades: list[Trade]) -> dict:
    nets = [t.net for t in trades]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    gw, gl = sum(wins), -sum(losses)
    eq = peak = dd = 0.0
    for t in sorted(trades, key=lambda t: t.exit_ts or t.entry_ts):
        eq += t.net
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return {
        "trades": len(trades),
        "gross": round(sum(t.gross for t in trades), 2),
        "costs": round(sum(t.costs for t in trades), 2),
        "slippage": round(sum(t.slippage for t in trades), 2),
        "net": round(sum(nets), 2),
        "win_rate": round(100.0 * len(wins) / len(nets), 1) if nets else 0.0,
        "profit_factor": round(gw / gl, 2) if gl > 0 else (float("inf") if gw > 0 else 0.0),
        "avg_win": round(gw / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gl / len(losses), 2) if losses else 0.0,
        "max_drawdown": round(dd, 2),
    }


def metrics(res: ReplayResult) -> dict:
    half = len(res.sessions) // 2
    first = set(res.sessions[:half])
    a = [t for t in res.trades if t.entry_ts.date() in first]
    b = [t for t in res.trades if t.entry_ts.date() not in first]
    reasons: dict[str, int] = {}
    for t in res.trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    return {
        "all": _stats(res.trades),
        "first_half": {"sessions": f"{res.sessions[0]}..{res.sessions[half - 1]}" if half else "", **_stats(a)},
        "second_half": {"sessions": f"{res.sessions[half]}..{res.sessions[-1]}" if res.sessions else "",
                        **_stats(b)},
        "exit_reasons": reasons,
        "sessions": len(res.sessions),
        "vol_source_days": res.vol_source_days,
        "sessions_skipped_no_vol": res.skipped_no_vol,
        "lot_size": res.lot_size,
    }


def write_csv(trades: list[Trade], path: str):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "kind", "underlying", "entry_ts", "exit_ts", "expiry", "reason", "spot_in",
                    "spot_out", "sigma", "vol_source", "qty", "entry_net_per_unit", "max_risk", "gross",
                    "costs", "net", "slippage_in_gross", "legs"])
        for t in trades:
            legs = " | ".join(f"{'B' if l.side > 0 else 'S'} {int(l.strike)}{l.opt} {l.entry_px:.2f}->{l.exit_px:.2f}"
                              for l in t.legs)
            w.writerow([t.strategy, t.kind, t.underlying, t.entry_ts.isoformat(),
                        t.exit_ts.isoformat() if t.exit_ts else "", t.expiry.isoformat(), t.reason,
                        round(t.spot_in, 2), round(t.spot_out, 2), round(t.sigma, 4), t.vol_source, t.qty,
                        round(t.entry_net, 2), t.max_risk, t.gross, t.costs, round(t.net, 2), t.slippage, legs])


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _t(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def build_strategy(a) -> OptionStrategy:
    s = a.strategy.upper()
    if s == "IRON_CONDOR":
        return IronCondor(entry=_t(a.entry or "10:20"), hold=a.hold, short_delta=a.short_delta or 0.175,
                          wing_pts=a.width, target_frac=a.target or 0.5, stop_mult=a.stop or 2.0,
                          time_exit=_t(a.time_exit))
    if s in ("CREDIT_SPREAD", "BULL_PUT", "BEAR_CALL"):
        return CreditSpread(direction=a.direction, entry=_t(a.entry) if a.entry else None, hold=a.hold,
                            short_delta=a.short_delta or 0.25, width_pts=a.width, target_frac=a.target or 0.5,
                            stop_mult=a.stop or 2.0, time_exit=_t(a.time_exit))
    if s == "LONG_STRADDLE":
        ev = None
        if a.event_dates and a.event_dates.upper() != "ALL":
            ev = {date.fromisoformat(x.strip()) for x in a.event_dates.split(",") if x.strip()}
        elif not a.event_dates:
            ev = set()
        return LongStraddle(event_dates=ev, entry=_t(a.entry or "09:20"), hold=a.hold,
                            target_frac=a.target or 0.5, stop_frac=a.stop or 0.4, time_exit=_t(a.time_exit))
    if s == "SHORT_STRADDLE":
        return ShortStraddle(entry=_t(a.entry or "10:20"), hold=a.hold, target_frac=a.target or 0.5,
                             stop_mult=a.stop or 2.0, time_exit=_t(a.time_exit))
    if s == "SHORT_STRANGLE":
        return ShortStrangle(entry=_t(a.entry or "10:20"), hold=a.hold, short_delta=a.short_delta or 0.16,
                             target_frac=a.target or 0.5, stop_mult=a.stop or 2.0, time_exit=_t(a.time_exit),
                             wing_pts=a.width)
    raise SystemExit(f"unknown strategy {a.strategy}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Index-option strategy replay (Black-Scholes on real index bars)")
    p.add_argument("--strategy", required=True,
                   help="IRON_CONDOR | CREDIT_SPREAD (BULL_PUT/BEAR_CALL by signal) | LONG_STRADDLE | "
                        "SHORT_STRADDLE | SHORT_STRANGLE (--width adds long wings that far out)")
    p.add_argument("--data", required=True, help="index.json file or a bars12m-style directory")
    p.add_argument("--underlying", default="NIFTY", choices=["NIFTY", "BANKNIFTY"])
    p.add_argument("--hold", default="intraday", choices=["intraday", "expiry"])
    p.add_argument("--entry", help="HH:MM decision time (IC 10:20, spreads 10:15, straddle 09:20)")
    p.add_argument("--time-exit", default="15:15")
    p.add_argument("--direction", default="first_hour", choices=["first_hour", "trend20"])
    p.add_argument("--event-dates", default="", help="comma-separated YYYY-MM-DD, or ALL for every session")
    p.add_argument("--short-delta", type=float)
    p.add_argument("--width", type=float, help="wing / spread width in index points")
    p.add_argument("--target", type=float, help="fraction of credit (or debit) to take profit")
    p.add_argument("--stop", type=float, help="credit: buy-back multiple of credit; debit: loss fraction")
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--lot-size", type=int)
    p.add_argument("--slip-min", type=float, help="min haircut per leg per fill, points")
    p.add_argument("--slip-pct", type=float, default=SLIP_PCT)
    p.add_argument("--allow-0dte", action="store_true")
    p.add_argument("--time-basis", default="calendar", choices=["calendar", "trading"],
                   help="BS clock: calendar minutes (default) or open-market minutes only")
    p.add_argument("--csv", help="per-trade CSV path (default: system temp dir)")
    p.add_argument("--json", action="store_true", help="print metrics as JSON")
    a = p.parse_args(argv)

    bars, vix = load_data(a.data)
    if a.underlying not in bars:
        raise SystemExit(f"{a.underlying} bars not found in {a.data} (have: {sorted(bars)})")
    strat = build_strategy(a)
    cfg = ReplayConfig(underlying=a.underlying, lots=a.lots, lot_size=a.lot_size, slip_min=a.slip_min,
                       slip_pct=a.slip_pct, allow_0dte=a.allow_0dte, time_basis=a.time_basis)
    res = Replay(bars[a.underlying], strat, cfg, vix).run()
    m = metrics(res)
    out = a.csv or os.path.join(tempfile.gettempdir(),
                                f"options_replay_{strat.name}_{a.underlying}_{a.hold}.csv")
    write_csv(res.trades, out)
    if a.json:
        print(json.dumps(m, indent=2, default=str))
    else:
        vol_note = ("India VIX (previous close)" if vix else
                    "NO VIX FILE — 20-session realised vol x 1.1 (min 10 returns); first sessions skipped")
        print(f"{strat.name} {a.underlying} hold={a.hold} clock={a.time_basis} lot={res.lot_size} ({LOT_SOURCE}) x{a.lots}  "
              f"sessions={m['sessions']} ({res.sessions[0]}..{res.sessions[-1]})")
        print(f"vol: {vol_note}; days by source {m['vol_source_days']}")
        for k in ("all", "first_half", "second_half"):
            s = m[k]
            print(f"  {k:<11} trades={s['trades']:<3} net=₹{s['net']:>10,.0f}  gross=₹{s['gross']:>10,.0f}  "
                  f"costs=₹{s['costs']:>8,.0f}  haircut(in gross)=₹{s['slippage']:>7,.0f}  win={s['win_rate']:>5}%  PF={s['profit_factor']}  "
                  f"maxDD=₹{s['max_drawdown']:,.0f}  {s.get('sessions', '')}")
        print(f"  exits: {m['exit_reasons']}")
        print(f"  csv: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
