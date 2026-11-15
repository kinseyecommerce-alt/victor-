"""
stratlab/events_strategies.py — the pre-registered EVENT strategies (promoter / insider buying, earnings moves).

Nine variants, textbook definitions, round parameters, fixed in prereg/events_v1.json BEFORE any result. They are
ordinary lab specs (api.Spec: a Swing function for sim.sim_positional, an Entry function for sim.sim_session) but are
NOT added to the catalogue REGISTRY: they need the event tables (events_data.py) and the point-in-time daily cache
(events_daily.py), and are run only by events_run.py.

  A  promoter / promoter-group OPEN-MARKET trades disclosed under SEBI PIT Reg 7(2) (timed on the exchange broadcast)
     ev_promoter_buy_h20        cluster of market purchases >= Rs 1 crore in 30 days -> buy next open, 20 sessions
     ev_promoter_buy_h60        same, 60 sessions
     ev_promoter_buy_dip_h60    same, only if the close is >= 20 % below its 252-session high, 60 sessions
     ev_promoter_sell_short_h20 cluster of market SALES >= Rs 1 crore in 30 days -> short the stock future, 20 sessions
  B  quarterly results (timed on the first public broadcast of the numbers)
     ev_pead_long_h20           reaction-day abnormal return vs NIFTY >= +4 % on above-average volume -> buy next
                                open, 20 sessions
     ev_pead_long_h60           same, 60 sessions
     ev_pead_short_h20          abnormal return <= -4 % on above-average volume -> short the stock future, 20 sessions
     ev_pre_earnings_h5         buy 5 sessions before the notified results board meeting, sell at the close of the
                                last session before it
     ev_egap_intraday           results out after the previous close and before 09:00: if the opening gap is at
                                least 0.5 x the 90-session std of overnight returns, trade WITH the gap from 09:16 to
                                the 15:15 square-off (long and short; Chan's earnings-gap momentum)
Stops: daily variants carry a 20 % catastrophe stop, the intraday one 5 %; no targets: the exit is the clock.

REPORT-ONLY twins (suffix `_rr3`, not trials for proof): the same entry with a 1:3 bracket (daily 5 % stop / 15 %
target, intraday 1 % / 3 %) and the same time limit.
"""
from __future__ import annotations

import numpy as np

from . import events_data as E
from .api import Entry, Spec, Swing, LONG, SHORT

CRORE = 1e7
CAL_DAYS_30 = 30


def _minus_days(day: int, n: int) -> int:
    import datetime as _dt
    d = _dt.date(day // 10000, day // 100 % 100, day % 100) - _dt.timedelta(days=n)
    return d.year * 10000 + d.month * 100 + d.day


def _cluster(d, p, side: str):
    """(triggered, cluster value): a NEW promoter market-`side` disclosure for this decision, and the Rs value of all
    such disclosures broadcast in the 30 calendar days up to the order cut-off."""
    a = E.asof_swing(d)
    if a.entry is None:
        return False, 0.0
    rows = a.insider(since=E.stamp(_minus_days(a.entry, p.window_days), E.ORDER_CUTOFF))
    if not len(rows):
        return False, 0.0
    m = rows["promoter"] & rows[side] & np.isfinite(rows["value"]) & (rows["value"] > 0)
    if not m.any():
        return False, 0.0
    fresh = m & (rows["ts"] >= a.fresh_since())
    total = float(rows["value"][m].sum())
    return bool(fresh.any()) and total >= p.min_value, total


def promoter_buy(d, p):
    ok, _ = _cluster(d, p, "buy")
    if not ok:
        return None
    return Swing(LONG, stop_pct=p.stop_pct, target_pct=getattr(p, "target_pct", None), max_hold=p.hold)


def promoter_buy_dip(d, p):
    if len(d) < p.high_n:
        return None
    ok, _ = _cluster(d, p, "buy")
    if not ok:
        return None
    if E.ca_excluded(d.symbol, int(d.date[-p.high_n]), int(d.date[-1])):
        return None                                   # the 252-session high would straddle an unadjusted action
    if not d.c[-1] <= (1.0 - p.dip_pct / 100.0) * float(np.max(d.h[-p.high_n:])):
        return None
    return Swing(LONG, stop_pct=p.stop_pct, target_pct=getattr(p, "target_pct", None), max_hold=p.hold)


def promoter_sell_short(d, p):
    ok, _ = _cluster(d, p, "sell")
    if not ok or not E.fo_listed(d.symbol, int(d.date[-1])):
        return None
    return Swing(SHORT, stop_pct=p.stop_pct, target_pct=getattr(p, "target_pct", None), max_hold=p.hold)


def _reaction(d, p):
    """Abnormal return of the results reaction session (today), or None. Needs yesterday = the previous session."""
    if len(d) < p.vol_n + 2 or d.nifty is None:
        return None
    a = E.asof_swing(d)
    if a.prev is None or int(d.date[-2]) != a.prev:
        return None                                   # the stock did not trade on the previous session
    if not len(a.results_reacted_today()):
        return None
    n1, n0 = float(d.nifty[-1]), float(d.nifty[-2])
    if not (n1 > 0 and n0 > 0 and d.c[-2] > 0):
        return None
    if E.ca_excluded(d.symbol, int(d.date[-2]), int(d.date[-1])):
        return None
    if not d.v[-1] > float(np.mean(d.v[-p.vol_n - 1:-1])):
        return None
    return (float(d.c[-1]) / float(d.c[-2]) - 1.0) - (n1 / n0 - 1.0)


def pead_long(d, p):
    ar = _reaction(d, p)
    if ar is None or ar < p.min_ar_pct / 100.0:
        return None
    return Swing(LONG, stop_pct=p.stop_pct, target_pct=getattr(p, "target_pct", None), max_hold=p.hold)


def pead_short(d, p):
    ar = _reaction(d, p)
    if ar is None or ar > -p.min_ar_pct / 100.0 or not E.fo_listed(d.symbol, int(d.date[-1])):
        return None
    return Swing(SHORT, stop_pct=p.stop_pct, target_pct=getattr(p, "target_pct", None), max_hold=p.hold)


def pre_earnings(d, p):
    """Entry session E and exit session X (the last session strictly before the meeting date M) with X = E + hold - 1.
    Uses only notices broadcast before E's order cut-off and the public session calendar."""
    a = E.asof_swing(d)
    if a.entry is None:
        return None
    rows = a.board_meetings()
    if not len(rows):
        return None
    cal = E.calendar()
    e_idx = E.session_index(a.entry)
    for m in np.unique(rows["meeting"][rows["results"] & (rows["meeting"] > a.entry)]):
        if int(m) > cal[-1]:
            continue                                  # beyond the known calendar: the session count is unknown
        x_idx = E.session_index(int(m)) - 1           # last session strictly before the meeting date
        if x_idx - e_idx == p.hold - 1:
            return Swing(LONG, stop_pct=p.stop_pct, target_pct=getattr(p, "target_pct", None), max_hold=p.hold)
    return None


def egap_intraday(s, p):
    if len(s) < 2 or len(s.prev) < p.lookback + 1 or not np.isfinite(s.gap):
        return []
    if not len(E.asof_session(s).results_before_open()):
        return []
    c2o = s.prev.o[-p.lookback:] / s.prev.c[-p.lookback - 1:-1] - 1.0
    sd = float(np.std(c2o))
    if not sd > 0:
        return []
    px = float(s.c[0])
    br = getattr(p, "bracket", None)
    if s.gap >= p.k * sd:
        return [Entry(0, LONG, stop=px * (1 - p.stop_pct / 100.0), bracket=br)]
    if s.gap <= -p.k * sd:
        return [Entry(0, SHORT, stop=px * (1 + p.stop_pct / 100.0), bracket=br)]
    return []


def _spec(id, kind, family, desc, fn, params, horizon=None) -> Spec:
    hz = "positional" if kind == "swing" else ("intraday" if kind == "intraday" else horizon)
    return Spec(id=id, kind=kind, family=family, desc=desc, fn=fn, params=dict(params), style="vector",
                max_per_day=1, horizon=hz, decision_time=None, needs_index=False, module=__name__)


_INS = {"min_value": CRORE, "window_days": CAL_DAYS_30, "stop_pct": 20.0}
_PEAD = {"min_ar_pct": 4.0, "vol_n": 20, "stop_pct": 20.0}

SPECS = [
    _spec("ev_promoter_buy_h20", "swing", "insider",
          "Promoter market purchases >= Rs 1 cr disclosed within 30 days: buy next open, hold 20 sessions, 20 % stop",
          promoter_buy, {**_INS, "hold": 20}),
    _spec("ev_promoter_buy_h60", "swing", "insider",
          "Promoter market purchases >= Rs 1 cr disclosed within 30 days: buy next open, hold 60 sessions, 20 % stop",
          promoter_buy, {**_INS, "hold": 60}),
    _spec("ev_promoter_buy_dip_h60", "swing", "insider",
          "Promoter market purchases >= Rs 1 cr within 30 days while >= 20 % below the 252-session high: hold 60",
          promoter_buy_dip, {**_INS, "hold": 60, "dip_pct": 20.0, "high_n": 252}),
    _spec("ev_promoter_sell_short_h20", "futures", "insider",
          "Promoter market sales >= Rs 1 cr disclosed within 30 days: short the stock future next open, 20 sessions",
          promoter_sell_short, {**_INS, "hold": 20}, horizon="positional"),
    _spec("ev_pead_long_h20", "swing", "earnings",
          "Results reaction day abnormal return >= +4 % vs NIFTY on above-average volume: buy next open, 20 sessions",
          pead_long, {**_PEAD, "hold": 20}),
    _spec("ev_pead_long_h60", "swing", "earnings",
          "Results reaction day abnormal return >= +4 % vs NIFTY on above-average volume: buy next open, 60 sessions",
          pead_long, {**_PEAD, "hold": 60}),
    _spec("ev_pead_short_h20", "futures", "earnings",
          "Results reaction day abnormal return <= -4 % vs NIFTY on above-average volume: short the future, 20 sessions",
          pead_short, {**_PEAD, "hold": 20}, horizon="positional"),
    _spec("ev_pre_earnings_h5", "swing", "earnings",
          "Buy 5 sessions before the notified results board meeting, sell at the close of the last session before it",
          pre_earnings, {"hold": 5, "stop_pct": 20.0}),
    _spec("ev_egap_intraday", "intraday", "earnings",
          "Results out overnight and the opening gap >= 0.5 x 90-session overnight std: trade with the gap to 15:15",
          egap_intraday, {"lookback": 90, "k": 0.5, "stop_pct": 5.0}),
]
BY_ID = {sp.id: sp for sp in SPECS}
DAILY = [sp for sp in SPECS if sp.horizon == "positional"]
INTRADAY = [sp for sp in SPECS if sp.horizon == "intraday"]

# report-only 1:3 bracket twins: same entry, stop X / target 3X from the fill, same time limit
RR3_DAILY = {"stop_pct": 5.0, "target_pct": 15.0}
RR3_INTRADAY = ("pct", 1.0, 3.0)


def rr3_twin(sp: Spec) -> Spec:
    q = dict(sp.params)
    if sp.horizon == "positional":
        q.update(RR3_DAILY)
    else:
        q["bracket"] = RR3_INTRADAY
    return Spec(id=sp.id + "_rr3", kind=sp.kind, family=sp.family, desc=sp.desc + " [report-only 1:3 bracket]",
                fn=sp.fn, params=q, style=sp.style, max_per_day=sp.max_per_day, horizon=sp.horizon,
                decision_time=None, needs_index=False, module=__name__)


RR3 = [rr3_twin(sp) for sp in SPECS]
