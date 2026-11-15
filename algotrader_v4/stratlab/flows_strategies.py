"""
stratlab/flows_strategies.py — the pre-registered FLOW strategies: big-investor bulk / block deals and NIFTY index
inclusions / deletions (index-rebalancing flows).

Eight variants, published / textbook definitions, round parameters, frozen in prereg/flows_v1.json BEFORE any
result on real data. Ordinary lab specs (api.Spec with a Swing function for sim.sim_positional), NOT added to the
catalogue REGISTRY: they need the flow tables (flowdata.py) and the point-in-time daily cache (events_daily.py) and
are run only by flows_run.py.

  A  bulk + block deals (flowdata.deals; a deal dated D is published after D's close -> usable from D+1's open)
     fl_inst_buy_h20       an INSTITUTION-class client (classify(): mutual fund / insurance / fund-type names, fixed
                           keyword list below) is a net BUYER of >= Rs 10 crore in the stock on the session (its bulk
                           and block deals of the day netted per client name, intra-group transfers removed) -> buy
                           next open, hold 20 sessions
     fl_inst_buy_h60       same, hold 60 sessions
     fl_inst_sell_short_h20  an institution-class client is a net SELLER of >= Rs 10 crore -> short the stock future
                           next open, 20 sessions (stock future must be listed on the decision day)
     fl_inst_cluster_h20   >= 2 DISTINCT institution-class client names, each a net buyer of >= Rs 10 crore on some
                           session of the stock's last 10 sessions (today included, today must add one) -> buy next
                           open, hold 20 sessions
  B  NIFTY index changes (flowdata.index_changes), POOLED: additions = IN to NIFTY 50 or IN to NIFTY 100 (= a new
     NIFTY Next 50 member from outside the top 100); deletions = OUT of NIFTY 50 or OUT of NIFTY 100. A move
     between NIFTY 50 and NIFTY Next 50 is an addition to / deletion from NIFTY 50. Rows whose effective date is
     known only to the month are never used. An announcement is assumed to be made after the close of its date:
     the first fill is the open of the session after the first symbol session on / after the announcement date.
     fl_idx_add_runup     addition with a REAL press-release date (announce_source == "press"): buy at the first open
                           after the announcement, sell at the close of the last session before the effective date
     fl_idx_add_reversal  addition (any announce_source): decide at the close of the effective date (first session of
                           the new composition), short the stock future at the next open, 20 sessions
     fl_idx_del_rebound   deletion (any announce_source): decide at the close of the effective date, buy at the next
                           open, 20 sessions
     fl_idx_del_rundown   deletion with a real press-release date: short the stock future at the first open after
                           the announcement, cover at the close of the last session before the effective date
Stops: every variant carries a 20 % catastrophe stop; no targets: the exit is the clock (as the event study).

REPORT-ONLY twins (suffix `_rr3`): same entries with a 1:3 bracket (5 % stop / 15 % target) and the same time limit.
"""
from __future__ import annotations

import datetime as _dt
import functools
import re

import numpy as np

from . import events_data as E
from . import flowdata as F
from .api import Spec, Swing, LONG, SHORT

# ── client classes: a FIXED, ordered keyword list (first matching class wins); written before any return was seen ──
CLIENT_CLASSES = (
    ("MF", (r"MUTUAL\s*FUND", r"\bMF\b", r"TRUSTEE\s+CO", r"\bAMC\b")),
    ("INSURANCE", (r"INSURANCE", r"ASSURANCE")),
    ("PROP_BROKER", (r"SECURITIES", r"BROKING", r"\bBROKERS?\b", r"\bSHARES?\s*(?:AND|&)\s*STOCK", r"STOCK\s*BROK",
                     r"ARBITRAGE", r"FINANCIAL\s+MARKETS", r"GLOBAL\s+MARKETS", r"CAPITAL\s+MARKETS", r"\bTRADING\b",
                     r"\bTRADERS?\b", r"RESEARCH", r"COMMODITIES", r"DERIVATIVES", r"FINSTOCK", r"FINSOL", r"\bODI\b",
                     r"MARKETS\s+LLP")),
    ("FUND", (r"\bFUNDS?\b", r"\bETFS?\b", r"PENSION", r"INVESTMENT\s+AUTHORITY", r"SOVEREIGN",
              r"GOVERNMENT\s+OF\s+SINGAPORE", r"MONETARY\s+AUTHORITY", r"NORGES\s+BANK", r"VANGUARD", r"ISHARES",
              r"BLACKROCK", r"FIDELITY", r"\bAIF\b", r"ALTERNATIVE\s+INVESTMENT", r"ENDOWMENT", r"RETIREMENT",
              r"PROVIDENT")),
)
OTHER = "OTHER"                                   # individuals, companies, promoters, trusts, banks, ...
INSTITUTION = ("MF", "INSURANCE", "FUND")
_RX = tuple((c, tuple(re.compile(p) for p in pats)) for c, pats in CLIENT_CLASSES)


@functools.lru_cache(maxsize=None)
def classify(client: str) -> str:
    """Client name -> class (MF | INSURANCE | PROP_BROKER | FUND | OTHER). Case- and whitespace-insensitive,
    deterministic, depends on nothing but the name."""
    s = " ".join(str(client or "").upper().replace(".", " ").split())
    for c, rxs in _RX:
        if any(rx.search(s) for rx in rxs):
            return c
    return OTHER


_STOP = {"THE", "M/S", "M/S."}


def _norm(client) -> str:
    return " ".join(str(client or "").upper().split())


def _group_word(client) -> str:
    w = [x for x in _norm(client).replace(".", " ").split() if x not in _STOP]
    return w[0] if w else ""


def group_transfer_mask(rows) -> np.ndarray:
    """True for rows that are one leg of an INTRA-GROUP TRANSFER: a BUY and a SELL in the same stock on the same
    session with the same quantity and price whose client names start with the same word (e.g. 2022's iShares
    Mauritius -> iShares ETF restructuring). Both legs are dropped: such a transfer is not a flow."""
    n = len(rows["date"])
    out = np.zeros(n, bool)
    sells: dict = {}
    for i in range(n):
        if rows["side"][i] < 0:
            sells.setdefault((int(rows["date"][i]), float(rows["qty"][i]), float(rows["price"][i]),
                              _group_word(rows["client"][i])), []).append(i)
    for i in range(n):
        if rows["side"][i] > 0:
            k = (int(rows["date"][i]), float(rows["qty"][i]), float(rows["price"][i]), _group_word(rows["client"][i]))
            lst = sells.get(k)
            if lst and k[3]:
                j = lst.pop()
                out[i] = out[j] = True
    return out


def net_flows(rows) -> list:
    """[(date, client, net Rs crore)] from flowdata.deals rows: every client's bulk + block deals of one session
    netted (BUY - SELL), intra-group transfers (group_transfer_mask) removed first. A same-day round trip nets to
    about zero."""
    acc: dict = {}
    drop = group_transfer_mask(rows)
    for dt, cl, sd, v, x in zip(rows["date"], rows["client"], rows["side"], rows["value_cr"], drop):
        if v == v and not x:
            k = (int(dt), _norm(cl))
            acc[k] = acc.get(k, 0.0) + float(sd) * float(v)
    return [(d, c, n) for (d, c), n in sorted(acc.items())]


def inst_events(sym: str, before: int, since: int, min_cr: float, side: int) -> list:
    """[(date, client)] of institution-class net buyers (side=+1) / sellers (side=-1) of >= min_cr crore in `sym`,
    deal dates in [since, before)."""
    rows = F.deals(sym, before, since=since)
    if not len(rows):
        return []
    return [(d, c) for d, c, n in net_flows(rows) if side * n >= min_cr and classify(c) in INSTITUTION]


def _swing(side, p):
    return Swing(side, stop_pct=p.stop_pct, target_pct=getattr(p, "target_pct", None), max_hold=p.hold)


def _prev(d) -> int:
    return int(d.date[-2]) if len(d) > 1 else 0


def inst_buy(d, p):
    a = F.asof_swing(d)                           # deals dated <= d.date[-1] (published by that evening)
    if inst_events(d.symbol, a.before, _prev(d) + 1, p.min_cr, +1):
        return _swing(LONG, p)
    return None


def inst_sell_short(d, p):
    a = F.asof_swing(d)
    if not inst_events(d.symbol, a.before, _prev(d) + 1, p.min_cr, -1):
        return None
    if not E.fo_listed(d.symbol, int(d.date[-1])):
        return None
    return _swing(SHORT, p)


def inst_cluster(d, p):
    if len(d) < p.window:
        return None
    a = F.asof_swing(d)
    ev = inst_events(d.symbol, a.before, int(d.date[-p.window]), p.min_cr, +1)
    if not ev or not any(dt > _prev(d) for dt, _c in ev):
        return None                               # today must add a qualifying buyer
    if len({c for _d, c in ev}) < p.min_buyers:
        return None
    return _swing(LONG, p)


# ── index changes ─────────────────────────────────────────────────────────────────────────────────────────────
POOL = {"add": (("NIFTY50", "IN"), ("NIFTY100", "IN")), "del": (("NIFTY50", "OUT"), ("NIFTY100", "OUT"))}
LOOKBACK_DAYS = 120          # an announcement precedes its effective date by 9-51 calendar days in the data


def _minus_days(day: int, n: int) -> int:
    x = _dt.date(day // 10000, day // 100 % 100, day % 100) - _dt.timedelta(days=n)
    return x.year * 10000 + x.month * 100 + x.day


def index_events(sym: str, before: int, kind: str, since: int, need_press: bool) -> list:
    """Pooled index-change events of `sym` announced in [since, before): sorted unique
    [(announce_date, effective_date)] for kind "add" / "del". Month-precision rows are dropped; need_press keeps
    only rows with a real press-release announcement date."""
    out = set()
    for ix, act in POOL[kind]:
        ch = F.index_changes(ix, before, since=since)
        if not len(ch):
            continue
        for a, e, ac, lab, src, prec in zip(ch["announce_date"], ch["effective_date"], ch["action"], ch["lab"],
                                            ch["announce_source"], ch["effective_precision"]):
            if lab != sym or ac != act or prec == "month":
                continue
            if need_press and src != "press":
                continue
            out.add((int(a), int(e)))
    return sorted(out)


def _runup(d, p, kind: str, side: int):
    """Entry at the open after the first symbol session on / after the announcement date, exit at the close of the
    last calendar session before the effective date."""
    day = int(d.date[-1])
    ev = index_events(d.symbol, day + 1, kind, _prev(d) + 1, need_press=True)
    if not ev:
        return None
    if side < 0 and not E.fo_listed(d.symbol, day):
        return None
    _a, eff = ev[0]
    x = E.next_session(day)
    y = E.prev_session(eff)
    if x is None or y is None:
        return None
    n = E.session_index(y) - E.session_index(x) + 1
    if n < 1:
        return None
    return Swing(side, stop_pct=p.stop_pct, target_pct=getattr(p, "target_pct", None), max_hold=int(n))


def _after_effective(d, p, kind: str, side: int):
    """Decide at the close of the first symbol session on / after the effective date; enter at the next open."""
    day = int(d.date[-1])
    ev = index_events(d.symbol, day + 1, kind, _minus_days(day, LOOKBACK_DAYS), need_press=False)
    if not any(_prev(d) < eff <= day for _a, eff in ev):
        return None
    if side < 0 and not E.fo_listed(d.symbol, day):
        return None
    return _swing(side, p)


def idx_add_runup(d, p):
    return _runup(d, p, "add", LONG)


def idx_del_rundown(d, p):
    return _runup(d, p, "del", SHORT)


def idx_add_reversal(d, p):
    return _after_effective(d, p, "add", SHORT)


def idx_del_rebound(d, p):
    return _after_effective(d, p, "del", LONG)


def _spec(id, kind, family, desc, fn, params) -> Spec:
    return Spec(id=id, kind=kind, family=family, desc=desc, fn=fn, params=dict(params), style="vector",
                max_per_day=1, horizon="positional", decision_time=None, needs_index=False, module=__name__)


_DEAL = {"min_cr": 10.0, "stop_pct": 20.0}
_IDX = {"stop_pct": 20.0}

SPECS = [
    _spec("fl_inst_buy_h20", "swing", "deals",
          "Institution-class client net buyer of >= Rs 10 cr (bulk/block deals of the day): buy next open, 20 sessions",
          inst_buy, {**_DEAL, "hold": 20}),
    _spec("fl_inst_buy_h60", "swing", "deals",
          "Institution-class client net buyer of >= Rs 10 cr (bulk/block deals of the day): buy next open, 60 sessions",
          inst_buy, {**_DEAL, "hold": 60}),
    _spec("fl_inst_sell_short_h20", "futures", "deals",
          "Institution-class client net seller of >= Rs 10 cr: short the stock future next open, 20 sessions",
          inst_sell_short, {**_DEAL, "hold": 20}),
    _spec("fl_inst_cluster_h20", "swing", "deals",
          ">= 2 distinct institution-class net buyers of >= Rs 10 cr within 10 sessions: buy next open, 20 sessions",
          inst_cluster, {**_DEAL, "hold": 20, "window": 10, "min_buyers": 2}),
    _spec("fl_idx_add_runup", "swing", "index",
          "NIFTY 50 / NIFTY 100 addition (real announcement date): buy first open after it, sell close before effective",
          idx_add_runup, {**_IDX}),
    _spec("fl_idx_add_reversal", "futures", "index",
          "NIFTY 50 / NIFTY 100 addition: short the future at the open after the effective date, 20 sessions",
          idx_add_reversal, {**_IDX, "hold": 20}),
    _spec("fl_idx_del_rebound", "swing", "index",
          "NIFTY 50 / NIFTY 100 deletion: buy at the open after the effective date, 20 sessions",
          idx_del_rebound, {**_IDX, "hold": 20}),
    _spec("fl_idx_del_rundown", "futures", "index",
          "NIFTY 50 / NIFTY 100 deletion (real announcement date): short first open after it, cover close before effective",
          idx_del_rundown, {**_IDX}),
]
BY_ID = {sp.id: sp for sp in SPECS}
DAILY = list(SPECS)

RR3_DAILY = {"stop_pct": 5.0, "target_pct": 15.0}


def rr3_twin(sp: Spec) -> Spec:
    q = dict(sp.params)
    q.update(RR3_DAILY)
    return Spec(id=sp.id + "_rr3", kind=sp.kind, family=sp.family, desc=sp.desc + " [report-only 1:3 bracket]",
                fn=sp.fn, params=q, style=sp.style, max_per_day=sp.max_per_day, horizon=sp.horizon,
                decision_time=None, needs_index=False, module=__name__)


RR3 = [rr3_twin(sp) for sp in SPECS]
