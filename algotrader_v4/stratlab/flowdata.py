"""
stratlab/flowdata.py — event / flow data sets for lab strategies, look-ahead-safe by construction (the same
`before` discipline and the same global cutoff() as nsedata.py).  DATA ONLY: nothing here trades or simulates.

DATA (normalised files in FLOW_DIR, default <scratchpad>/databuild, override with env STRATLAB_FLOW_DIR; built by the
scripts in <scratchpad>/databuild/scripts, each data set with a data_check.json next to it)

  deals/deals.csv.gz                  NSE bulk and block deals, 2012-01-02 .. 2026-11-13
      date, kind (bulk|block), symbol (as traded that day), lab_symbol, in_lab, security, client, side (BUY|SELL),
      qty, price, value_cr, remarks
  corpactions/corpactions.csv.gz      NSE corporate actions by ex-date, 2012 .. 2026 (dividends, bonus, split,
      consolidation, rights, demerger / scheme, capital reduction, buyback)
      ex_date, symbol (NSE's CURRENT symbol), symbol_then (symbol on the ex-date), lab_symbol, in_lab, series, isin,
      face_value, kind, ratio_a, ratio_b, fv_from, fv_to, dividend, rights_premium, factor, record_date,
      first_listed, known_date, subject
  corpactions/bar_adjustment_issues.csv   every ex-date where the lab's bars disagree with the official action
  index_membership/index_changes.csv  NIFTY 50 / NIFTY 100 / NIFTY NEXT 50 constituent changes from NSE Indices
      press releases: index, announce_date, effective_date, action (IN|OUT), symbol, lab_symbol, in_lab, company, source
  index_membership/index_members.csv.gz   point-in-time membership intervals rebuilt from the changes and anchored
      on NSE's own constituent files: index, symbol (latest NSE symbol), lab_symbol, in_lab, from_date, to_date
  fii_stats/fii_stats.csv.gz          FII derivative statistics, complete: date (ISO), instrument, buy_contracts,
      buy_amt_cr, sell_contracts, sell_amt_cr, oi_contracts, oi_amt_cr

All dates are int yyyymmdd (fii_stats is stored ISO like nsedata's file and converted on load).  `lab_symbol` is the
lab's bar-file name for the company (rename chains resolved: ZOMATO -> ETERNAL, MCDOWELL-N -> UNITDSPR, SESAGOA ->
SSLT -> VEDL ...); stocks outside the lab's universe keep their latest NSE symbol and have in_lab = 0.

AVAILABILITY — every accessor takes `before` (int yyyymmdd) and returns only rows KNOWN strictly before it; inside
nsedata.cutoff(day) `before` is clipped to `day`.  Pick `before` exactly as for nsedata:
    intraday @strategy (Session s)        before = s.date            (what was published up to the previous session)
    Swing, next-open fill                 before = d.date[-1] + 1    (published by that evening)
    Swing with decision_time              before = d.date[-1]
  deals          published after the close of the trade date D            -> a row dated D is known from D+1's open
  corp actions   known_date = first session whose NSE "Bc" file lists the action (published that evening), at the
                 latest the session before the ex-date                    -> visible when known_date < before.
                 So an ex-date is visible on the ex-date itself (and earlier, once listed): upcoming ex-dates are
                 returned too; filter on r.ex_date for past / future.
  index changes  announce_date = date of the NSE Indices press release (published after market hours)
                                                                           -> visible when announce_date < before
                 index_members(index, before) = the constituents in force on the last day before `before`, built
                 only from changes whose effective_date < before
  fii stats      published in the evening of the trade date               -> rows dated < before

    cd algotrader_v4
    python -m stratlab.flowdata --check            # coverage report of whatever is on disk
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys

import numpy as np

from .nsedata import Rows, _clip           # the one global look-ahead guard (nsedata.cutoff) applies here too

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
FLOW_DIR = os.environ.get("STRATLAB_FLOW_DIR", os.path.join(_SCRATCH, "databuild"))
INDICES = ("NIFTY50", "NIFTY100", "NIFTYNEXT50")
CA_KINDS = ("dividend", "bonus", "split", "consolidation", "rights", "demerger", "capital_reduction", "buyback")
FAR = 99991231
_S: dict = {}


def use(flow_dir: str):
    """Point the module at another data directory (tests) and drop every cached table."""
    global FLOW_DIR
    FLOW_DIR = flow_dir
    _S.clear()


def _open(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def _read(rel: str) -> list:
    p = os.path.join(FLOW_DIR, rel)
    if not os.path.exists(p):
        return []
    with _open(p) as fh:
        return list(csv.DictReader(fh))


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _i(x, default=0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _by(keys: np.ndarray) -> dict:
    """{key: slice} for an array sorted by key."""
    out = {}
    if len(keys) == 0:
        return out
    cut = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    st = np.concatenate([[0], cut])
    en = np.concatenate([cut, [len(keys)]])
    for a, b in zip(st, en):
        out[str(keys[a])] = slice(int(a), int(b))
    return out


def _empty(fields) -> Rows:
    return Rows({c: np.array([]) for c in fields})


# ══════════════════════════════════════════════════════════════════════════
# bulk and block deals
# ══════════════════════════════════════════════════════════════════════════
DEAL_FIELDS = ("date", "kind", "symbol", "client", "side", "qty", "price", "value_cr")


def _load_deals():
    if "deals" in _S:
        return
    rows = _read(os.path.join("deals", "deals.csv.gz"))
    rows.sort(key=lambda r: (r["lab_symbol"], int(r["date"])))
    t = {"lab": np.array([r["lab_symbol"] for r in rows], dtype=object),
         "date": np.array([int(r["date"]) for r in rows], np.int64),
         "kind": np.array([r["kind"] for r in rows], dtype=object),
         "symbol": np.array([r["symbol"] for r in rows], dtype=object),
         "client": np.array([r["client"] for r in rows], dtype=object),
         "side": np.array([1 if r["side"] == "BUY" else -1 for r in rows], np.int8),
         "qty": np.array([_f(r["qty"]) for r in rows], float),
         "price": np.array([_f(r["price"]) for r in rows], float),
         "value_cr": np.array([_f(r["value_cr"]) for r in rows], float)}
    _S["deals"] = t
    _S["deals_ix"] = _by(t["lab"])
    order = np.argsort(t["date"], kind="stable")
    _S["deals_all"] = {k: v[order] for k, v in t.items()}


def deals(sym: str | None, before: int, since: int | None = None, kind: str | None = None) -> Rows:
    """Bulk / block deals of one lab symbol (sym=None: every symbol, plus a `lab` column) dated strictly before
    `before` and, if given, on or after `since`; kind = "bulk" | "block" | None (both).  Oldest first.
    Columns: date, kind, symbol (as traded), client, side (+1 BUY / -1 SELL), qty (shares), price (Rs, unadjusted:
    compare with nsedata.delivery prices, not with split-adjusted bars), value_cr.
    A deal dated D was published after D's close: with before = s.date an intraday strategy sees deals up to the
    previous session."""
    _load_deals()
    b = _clip(before)
    if sym is None:
        t = _S["deals_all"]
        sl = slice(0, len(t["date"]))
        fields = DEAL_FIELDS + ("lab",)
    else:
        t = _S["deals"]
        sl = _S["deals_ix"].get(sym)
        fields = DEAL_FIELDS
        if sl is None:
            return _empty(fields)
    d = t["date"][sl]
    k = int(np.searchsorted(d, b, side="left"))
    a = int(np.searchsorted(d, int(since), side="left")) if since is not None else 0
    out = {c: t[c][sl][a:k].copy() for c in fields}
    if kind is not None:
        m = out["kind"] == kind
        out = {c: v[m] for c, v in out.items()}
    return Rows(out)


# ══════════════════════════════════════════════════════════════════════════
# corporate actions
# ══════════════════════════════════════════════════════════════════════════
CA_FIELDS = ("ex_date", "kind", "factor", "dividend", "ratio_a", "ratio_b", "fv_from", "fv_to", "rights_premium",
             "record_date", "known_date", "symbol", "subject")


class CARows(Rows):
    def __len__(self):
        return len(self["ex_date"]) if "ex_date" in self else 0


def _load_ca():
    if "ca" in _S:
        return
    rows = _read(os.path.join("corpactions", "corpactions.csv.gz"))
    rows.sort(key=lambda r: (r["lab_symbol"], int(r["known_date"]), int(r["ex_date"])))
    t = {"lab": np.array([r["lab_symbol"] for r in rows], dtype=object),
         "ex_date": np.array([int(r["ex_date"]) for r in rows], np.int64),
         "known_date": np.array([int(r["known_date"]) for r in rows], np.int64),
         "record_date": np.array([_i(r["record_date"]) for r in rows], np.int64),
         "kind": np.array([r["kind"] for r in rows], dtype=object),
         "symbol": np.array([r["symbol"] for r in rows], dtype=object),
         "subject": np.array([r["subject"] for r in rows], dtype=object)}
    for c in ("factor", "dividend", "ratio_a", "ratio_b", "fv_from", "fv_to", "rights_premium"):
        t[c] = np.array([_f(r[c]) for r in rows], float)
    _S["ca"] = t
    _S["ca_ix"] = _by(t["lab"])


def corp_actions(sym: str, before: int, kinds=None, since: int | None = None, upcoming: bool = True) -> CARows:
    """Corporate actions of one lab symbol that were KNOWN strictly before `before` (known_date < before), ordered
    by ex-date.  kinds: iterable of CA_KINDS or None (all); since: only ex_date >= since; upcoming=False drops
    actions whose ex-date is on or after `before` (keeps only those already ex).
    Columns: ex_date, kind, factor (prices before the ex-date x factor are comparable with later prices; NaN for
    dividends / demergers / buybacks), dividend (Rs per share, NaN if none), ratio_a, ratio_b, fv_from, fv_to,
    rights_premium, record_date (0 = none), known_date, symbol (NSE's current symbol), subject."""
    _load_ca()
    b = _clip(before)
    sl = _S["ca_ix"].get(sym)
    if sl is None:
        return CARows({c: np.array([]) for c in CA_FIELDS})
    t = _S["ca"]
    m = t["known_date"][sl] < b
    if kinds is not None:
        ks = set(kinds)
        m &= np.array([k in ks for k in t["kind"][sl]], bool)
    if since is not None:
        m &= t["ex_date"][sl] >= int(since)
    if not upcoming:
        m &= t["ex_date"][sl] < b
    out = {c: t[c][sl][m] for c in CA_FIELDS}
    order = np.argsort(out["ex_date"], kind="stable")
    return CARows({c: v[order].copy() for c, v in out.items()})


def adj_factor(sym: str, day: int, before: int) -> float:
    """Product of the split / bonus / consolidation / rights factors with day < ex_date < before that were known
    before `before`: multiply an UNADJUSTED price dated `day` by it to compare with unadjusted prices just before
    `before`.  Demergers and dividends are not included (no official factor)."""
    r = corp_actions(sym, before, kinds=("bonus", "split", "consolidation", "rights"), upcoming=False)
    f = 1.0
    for ex, x in zip(r["ex_date"], r["factor"]):
        if int(day) < int(ex) and x == x:
            f *= float(x)
    return f


def bar_adjustment_issues(sym: str | None = None) -> list:
    """Rows of corpactions/bar_adjustment_issues.csv (dicts), optionally for one lab symbol.  A data-quality list,
    not a signal: it describes how the bar files as they are on disk today treat each ex-date."""
    rows = _read(os.path.join("corpactions", "bar_adjustment_issues.csv"))
    return [r for r in rows if sym is None or r["symbol"] == sym]


# ══════════════════════════════════════════════════════════════════════════
# index membership
# ══════════════════════════════════════════════════════════════════════════
CHANGE_FIELDS = ("announce_date", "effective_date", "action", "symbol", "lab", "company")


class ChangeRows(Rows):
    def __len__(self):
        return len(self["announce_date"]) if "announce_date" in self else 0


def _load_index():
    if "idx_changes" in _S:
        return
    ch = _read(os.path.join("index_membership", "index_changes.csv"))
    ch.sort(key=lambda r: (r["index"], int(r["announce_date"]), int(r["effective_date"]), r["action"], r["symbol"]))
    t = {"index": np.array([r["index"] for r in ch], dtype=object),
         "announce_date": np.array([int(r["announce_date"]) for r in ch], np.int64),
         "effective_date": np.array([int(r["effective_date"]) for r in ch], np.int64),
         "action": np.array([r["action"] for r in ch], dtype=object),
         "symbol": np.array([r["symbol"] for r in ch], dtype=object),
         "lab": np.array([r["lab_symbol"] for r in ch], dtype=object),
         "company": np.array([r.get("company", "") for r in ch], dtype=object)}
    _S["idx_changes"] = t
    _S["idx_changes_ix"] = _by(t["index"])
    mem: dict = {}
    for r in _read(os.path.join("index_membership", "index_members.csv.gz")):
        mem.setdefault(r["index"], []).append((r["lab_symbol"], int(r["from_date"]), _i(r["to_date"], FAR) or FAR))
    _S["idx_members"] = mem


def index_changes(index: str, before: int, since: int | None = None) -> ChangeRows:
    """Constituent changes of `index` (INDICES) ANNOUNCED strictly before `before` (press release dated < before),
    oldest announcement first; since: only announce_date >= since.  Pending changes (effective_date >= before) are
    included: that is the rebalancing-event calendar.  Columns: announce_date, effective_date (first session with
    the new composition), action ("IN" | "OUT"), symbol (as in the release), lab (lab symbol), company."""
    _load_index()
    sl = _S["idx_changes_ix"].get(index)
    if sl is None:
        return ChangeRows({c: np.array([]) for c in CHANGE_FIELDS})
    t = _S["idx_changes"]
    d = t["announce_date"][sl]
    k = int(np.searchsorted(d, _clip(before), side="left"))
    a = int(np.searchsorted(d, int(since), side="left")) if since is not None else 0
    return ChangeRows({c: t[c][sl][a:k].copy() for c in CHANGE_FIELDS})


def index_members(index: str, before: int) -> set:
    """Lab symbols (latest NSE symbols for stocks outside the lab) that were constituents of `index` on the last
    calendar day strictly before `before`.  Built from membership intervals [from_date, to_date]; an interval
    counts when from_date < before and to_date >= before - 1 day, i.e. only changes already in force are used.
    Empty set when `before` is outside the covered period (see index_coverage())."""
    _load_index()
    b = _clip(before)
    lo, hi = index_coverage(index)
    if lo is None or b <= lo:
        return set()
    return {s for s, f, t in _S["idx_members"].get(index, ()) if f < b <= _next_day(t)}


def _next_day(d: int) -> int:
    if d >= FAR:
        return FAR
    from datetime import date, timedelta
    x = date(d // 10000, d // 100 % 100, d % 100) + timedelta(days=1)
    return x.year * 10000 + x.month * 100 + x.day


def index_coverage(index: str) -> tuple:
    """(first date, last date) for which index_members() is defined: from the earliest interval start to the build
    date recorded in index_membership/data_check.json (None, None when the index is not on disk)."""
    _load_index()
    iv = _S["idx_members"].get(index)
    if not iv:
        return None, None
    lo = min(f for _, f, _ in iv)
    hi = None
    p = os.path.join(FLOW_DIR, "index_membership", "data_check.json")
    if os.path.exists(p):
        try:
            hi = json.load(open(p)).get("built_through")
        except Exception:
            hi = None
    return lo, hi


def was_member(index: str, sym: str, before: int) -> bool:
    return sym in index_members(index, before)


# ══════════════════════════════════════════════════════════════════════════
# FII derivative statistics (complete file)
# ══════════════════════════════════════════════════════════════════════════

def _load_fii():
    if "fii" in _S:
        return
    out: dict = {}
    rows = _read(os.path.join("fii_stats", "fii_stats.csv.gz"))
    rows.sort(key=lambda r: (r["instrument"], r["date"]))
    cols = [c for c in (rows[0].keys() if rows else ()) if c not in ("date", "instrument")]
    cur: dict = {}
    for r in rows:
        cur.setdefault(r["instrument"], []).append(r)
    for ins, rs in cur.items():
        out[ins] = {"date": np.array([int(r["date"].replace("-", "")) for r in rs], np.int64),
                    **{c: np.array([_f(r[c]) for r in rs], float) for c in cols}}
    _S["fii"] = out


def fii_stats(before: int, n: int = 1, instrument: str = "INDEX FUTURES") -> Rows:
    """FII derivative statistics for one instrument group, last n sessions strictly before `before`: date,
    buy_contracts, buy_amt_cr, sell_contracts, sell_amt_cr, oi_contracts, oi_amt_cr.  Same contract as
    nsedata.fii_stats but on the completed file (see fii_stats/data_check.json for the covered dates and the
    groups NSE published in each period: check r.date[-1] before trusting a row as "yesterday")."""
    _load_fii()
    t = _S["fii"].get(instrument)
    if t is None:
        return Rows({"date": np.array([])})
    d = t["date"]
    k = int(np.searchsorted(d, _clip(before), side="left"))
    a = max(0, k - int(n))
    return Rows({c: v[a:k].copy() for c, v in t.items()})


def fii_instruments() -> list:
    _load_fii()
    return sorted(_S["fii"])


# ══════════════════════════════════════════════════════════════════════════
# Asof: accessors bound to a symbol and a `before`
# ══════════════════════════════════════════════════════════════════════════

class Asof:
    """Build with asof_session / asof_swing / asof_swing_intraday (same choices of `before` as nsedata's)."""

    def __init__(self, symbol: str, before: int):
        self.symbol, self.before = symbol, int(before)

    def deals(self, since=None, kind=None, sym=None):
        return deals(sym or self.symbol, self.before, since, kind)

    def corp_actions(self, kinds=None, since=None, upcoming=True, sym=None):
        return corp_actions(sym or self.symbol, self.before, kinds, since, upcoming)

    def index_changes(self, index="NIFTY50", since=None):
        return index_changes(index, self.before, since)

    def index_members(self, index="NIFTY50"):
        return index_members(index, self.before)

    def fii_stats(self, n=1, instrument="INDEX FUTURES"):
        return fii_stats(self.before, n, instrument)


def asof_session(s) -> Asof:
    """Intraday @strategy: what was published up to the previous session (plus ex-dates / index changes known by
    then, including today's)."""
    return Asof(s.symbol, int(s.date))


def asof_swing(d) -> Asof:
    """Swing with the default next-open fill: everything published by the evening of d.date[-1]."""
    return Asof(d.symbol, int(d.date[-1]) + 1)


def asof_swing_intraday(d) -> Asof:
    """Swing with decision_time: only what was published before d.date[-1]'s open."""
    return Asof(d.symbol, int(d.date[-1]))


# ══════════════════════════════════════════════════════════════════════════
# coverage report
# ══════════════════════════════════════════════════════════════════════════

def coverage() -> str:
    lines = [f"flow data directory: {FLOW_DIR}"]
    _load_deals()
    d = _S["deals_all"]
    if len(d["date"]):
        lines.append(f"deals          {len(d['date']):>8,} rows  {d['date'][0]} .. {d['date'][-1]}  "
                     f"(bulk {int((d['kind'] == 'bulk').sum()):,}, block {int((d['kind'] == 'block').sum()):,})")
    else:
        lines.append("deals          missing")
    _load_ca()
    c = _S["ca"]
    if len(c["ex_date"]):
        ks = {k: int((c["kind"] == k).sum()) for k in CA_KINDS}
        lines.append(f"corp actions   {len(c['ex_date']):>8,} rows  ex-dates {c['ex_date'].min()} .. {c['ex_date'].max()}  {ks}")
    else:
        lines.append("corp actions   missing")
    _load_index()
    for ix in INDICES:
        ch = index_changes(ix, FAR)
        lo, hi = index_coverage(ix)
        lines.append(f"{ix:<14} {len(ch):>8,} change rows, announcements "
                     f"{ch['announce_date'][0] if len(ch) else '-'} .. {ch['announce_date'][-1] if len(ch) else '-'}; "
                     f"membership {lo} .. {hi}")
    _load_fii()
    for ins in ("INDEX FUTURES", "INDEX OPTIONS", "STOCK FUTURES"):
        t = _S["fii"].get(ins)
        lines.append(f"fii {ins:<14} " + (f"{len(t['date']):>5,} days {t['date'][0]} .. {t['date'][-1]}" if t else "missing"))
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv or not argv:
        print(coverage())
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
