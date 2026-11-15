"""
stratlab/events_data.py — point-in-time corporate EVENT data for lab strategies, look-ahead-safe by construction.

Tables (built by events_fetch.py + events_parse.py from NSE's public filing feeds; directory EVENTS_DIR, default
<scratchpad>/events, env STRATLAB_EVENTS_DIR):
  insider.csv.gz    SEBI PIT Reg 7(2) disclosures (promoter / insider trades) with the exchange BROADCAST time
  results.csv.gz    quarterly results: the first moment the numbers were public on NSE (`first_ts`)
  boardmeet.csv.gz  board-meeting notices (meeting date, purpose, time of the notice)

AVAILABILITY (the one rule): a row becomes usable at the minute after its exchange broadcast time, never earlier.
Transaction dates (insider `from_date` / `to_date`) and meeting dates are CONTENT, not availability. Every accessor
takes `before`, a STAMP = yyyymmdd * 10000 + hhmm (IST), and returns only rows broadcast STRICTLY BEFORE it:

    next-open Swing deciding on session D, filled at the open of the next session X   before = stamp(X, ORDER_CUTOFF)
        (ORDER_CUTOFF = 09:00: the order must be in before the pre-open; a filing at 09:05 waits a day)
    "the market reacted to it on session D" (close-to-close reaction)                  before = stamp(D, CLOSE)
        and broadcast at / after stamp(previous session, CLOSE): a filing at 16:10 on D belongs to D+1's reaction
    intraday @strategy on session s                                                   before = stamp(s.date, ORDER_CUTOFF)

Use asof_swing(d) / asof_session(s), which pick `before` for you from the session calendar (set_calendar()). The
calendar (exchange holidays) is public in advance, so knowing the NEXT session's date is not look-ahead.
`cutoff(stamp)` is the global guard used by the simulator wrapper and the tests: inside it any accessor asked for
rows at / after the cutoff raises LookAheadError.

Also here: fo_listed(symbol, day) (point-in-time stock-futures list from the F&O bhavcopy, EOD of `day`), and
ca_excluded(symbol, d0, d1) (demerger / scheme ex-dates and unexplained overnight gaps: see events_daily.py).
"""
from __future__ import annotations

import bisect
import contextlib
import csv
import gzip
import json
import os

import numpy as np

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
EVENTS_DIR = os.environ.get("STRATLAB_EVENTS_DIR", os.path.join(_SCRATCH, "events"))
ORDER_CUTOFF = 900          # 09:00 IST
CLOSE = 1530                # 15:30 IST
MODE_BUY = "market purchase"
MODE_SELL = "market sale"


class LookAheadError(RuntimeError):
    """A read of event rows broadcast at or after the active cutoff (see cutoff())."""


def stamp(day: int, hhmm: int = 0) -> int:
    return int(day) * 10000 + int(hhmm)


def ts_stamp(ts: str) -> int:
    """'2024-01-10 21:21:00' -> 202401102121."""
    return int(ts[0:4] + ts[5:7] + ts[8:10] + ts[11:13] + ts[14:16])


# ── cutoff guard ────────────────────────────────────────────────────────────
_CUT: list = []


@contextlib.contextmanager
def cutoff(st: int):
    """Inside this block no accessor may be asked for rows broadcast at / after stamp `st`."""
    _CUT.append(int(st))
    try:
        yield
    finally:
        _CUT.pop()


def _guard(before: int, what: str) -> int:
    b = int(before)
    if _CUT and b > min(_CUT):
        raise LookAheadError(f"{what}: rows before {b} requested while only rows before {min(_CUT)} are allowed")
    return b


# ── calendar ────────────────────────────────────────────────────────────────
_CAL: list = []


def set_calendar(days) -> None:
    """The session calendar (sorted yyyymmdd) used by next_session / prev_session / asof_*."""
    global _CAL
    _CAL = sorted(set(int(x) for x in days))


def calendar() -> list:
    return _CAL


def next_session(day: int) -> int | None:
    i = bisect.bisect_right(_CAL, int(day))
    return _CAL[i] if i < len(_CAL) else None


def prev_session(day: int) -> int | None:
    i = bisect.bisect_left(_CAL, int(day))
    return _CAL[i - 1] if i > 0 else None


def session_index(day: int) -> int:
    """Number of sessions strictly before `day` (= the index of `day` if it is a session)."""
    return bisect.bisect_left(_CAL, int(day))


# ── tables ──────────────────────────────────────────────────────────────────
_T: dict = {}


def use(events_dir: str | None = None) -> None:
    """Point the module at another directory (tests) and drop the loaded tables."""
    global EVENTS_DIR
    if events_dir is not None:
        EVENTS_DIR = events_dir
    _T.clear()


def _read(name: str) -> list:
    p = os.path.join(EVENTS_DIR, name)
    if not os.path.exists(p):
        return []
    with gzip.open(p, "rt", newline="") as f:
        return list(csv.DictReader(f))


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _by_symbol(rows: list, key) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["symbol"], []).append(r)
    for v in out.values():
        v.sort(key=key)
    return out


def _load(name: str) -> dict:
    """{symbol: {"ts": int64 stamps (sorted), ...columns}}"""
    if name in _T:
        return _T[name]
    if name == "insider":
        rows = [r for r in _read("insider.csv.gz") if r.get("broadcast")]
        tab = {}
        for sym, rr in _by_symbol(rows, lambda r: r["broadcast"]).items():
            mode = [r["mode"].strip().lower() for r in rr]
            eq = np.array([r["sec_type"].strip().lower().startswith("equity") for r in rr])
            tab[sym] = {"ts": np.array([ts_stamp(r["broadcast"]) for r in rr], np.int64),
                        "value": np.array([_f(r["value"]) for r in rr]), "qty": np.array([_f(r["qty"]) for r in rr]),
                        "promoter": np.array([r["is_promoter"] == "1" for r in rr]),
                        "buy": eq & np.array([m == MODE_BUY for m in mode]),
                        "sell": eq & np.array([m == MODE_SELL for m in mode]),
                        "person": np.array([r["person"].strip().lower() for r in rr], dtype=object)}
    elif name == "results":
        rows = [r for r in _read("results.csv.gz") if r.get("first_ts")]
        tab = {}
        for sym, rr in _by_symbol(rows, lambda r: r["first_ts"]).items():
            tab[sym] = {"ts": np.array([ts_stamp(r["first_ts"]) for r in rr], np.int64),
                        "period_end": np.array([int(r["period_end"].replace("-", "")) for r in rr], np.int64)}
    elif name == "boardmeet":
        rows = [r for r in _read("boardmeet.csv.gz") if r.get("notice_ts") and r.get("meeting_date")]
        tab = {}
        for sym, rr in _by_symbol(rows, lambda r: r["notice_ts"]).items():
            tab[sym] = {"ts": np.array([ts_stamp(r["notice_ts"]) for r in rr], np.int64),
                        "meeting": np.array([int(r["meeting_date"].replace("-", "")) for r in rr], np.int64),
                        "results": np.array([r["is_results"] == "1" for r in rr])}
    else:
        raise KeyError(name)
    _T[name] = tab
    return tab


class Rows(dict):
    """Column arrays of the selected rows (oldest first). len() = number of rows."""

    def __len__(self):
        return len(self["ts"]) if "ts" in self else 0


def _select(name: str, sym: str, before: int, since: int) -> Rows:
    b = _guard(before, name)
    t = _load(name).get(sym)
    if t is None:
        return Rows()
    lo = int(np.searchsorted(t["ts"], int(since), side="left"))
    hi = int(np.searchsorted(t["ts"], b, side="left"))          # strictly before
    return Rows({k: v[lo:hi] for k, v in t.items()})


def insider(sym: str, before: int, since: int = 0) -> Rows:
    """Insider-trade disclosures of `sym` broadcast in [since, before). Columns: ts, value (Rs), qty, promoter,
    buy (market purchase of equity shares), sell (market sale of equity shares), person."""
    return _select("insider", sym, before, since)


def results(sym: str, before: int, since: int = 0) -> Rows:
    """Quarterly results of `sym` first public in [since, before). Columns: ts, period_end."""
    return _select("results", sym, before, since)


def board_meetings(sym: str, before: int, since: int = 0) -> Rows:
    """Board-meeting NOTICES of `sym` broadcast in [since, before). Columns: ts (notice), meeting (date), results."""
    return _select("boardmeet", sym, before, since)


def symbols(name: str) -> list:
    return sorted(_load(name))


# ── as-of helpers ───────────────────────────────────────────────────────────
class Asof:
    """Accessors bound to one symbol and one decision. Build with asof_swing(d) / asof_session(s)."""

    def __init__(self, symbol: str, day: int, entry: int | None, prev: int | None):
        self.symbol, self.day, self.entry, self.prev = symbol, int(day), entry, prev
        # rows usable for an order at the open of the entry session (None: no next session known -> nothing new)
        self.order_before = stamp(entry, ORDER_CUTOFF) if entry is not None else stamp(day, CLOSE)

    def insider(self, since: int = 0) -> Rows:
        return insider(self.symbol, self.order_before, since)

    def fresh_since(self) -> int:
        """Stamp from which a row is NEW for this decision: the order cut-off of the previous decision."""
        return stamp(self.day, ORDER_CUTOFF)

    def results_reacted_today(self) -> Rows:
        """Results first public after the previous session's close and before this session's close: this
        session's close-to-close return is the market's reaction to them."""
        since = stamp(self.prev, CLOSE) if self.prev is not None else 0
        return results(self.symbol, stamp(self.day, CLOSE), since)

    def results_before_open(self) -> Rows:
        """Results first public after the previous session's close and before today's order cut-off (intraday)."""
        since = stamp(self.prev, CLOSE) if self.prev is not None else 0
        return results(self.symbol, stamp(self.day, ORDER_CUTOFF), since)

    def board_meetings(self) -> Rows:
        return board_meetings(self.symbol, self.order_before)


def asof_swing(d) -> Asof:
    """Inside a next-open Swing strategy: decision on session d.date[-1], order for the next session's open."""
    day = int(d.date[-1])
    return Asof(d.symbol, day, next_session(day), prev_session(day))


def asof_session(s) -> Asof:
    """Inside an intraday @strategy: today's session, rows known before today's order cut-off."""
    day = int(s.date)
    return Asof(s.symbol, day, day, prev_session(day))


# ── point-in-time F&O list and corporate-action exclusions ──────────────────
_FO: dict = {}
_EXC: dict = {}


def set_fo(by_day: dict) -> None:
    """{yyyymmdd: frozenset(symbols with a stock future that day)} (built by events_run from the F&O bhavcopy)."""
    _FO.clear()
    _FO.update(by_day)
    _FO["_days"] = sorted(k for k in by_day)


def fo_listed(sym: str, day: int) -> bool:
    """Was a stock future of `sym` traded on the last F&O session at or before `day` (known that evening)?"""
    days = _FO.get("_days") or []
    i = bisect.bisect_right(days, int(day))
    return i > 0 and sym in _FO[days[i - 1]]


def set_exclusions(pairs) -> None:
    """[(symbol, yyyymmdd)]: demerger / scheme / large-rights ex-dates and unexplained overnight gaps."""
    _EXC.clear()
    for s, d in pairs:
        _EXC.setdefault(str(s), []).append(int(d))
    for v in _EXC.values():
        v.sort()


def ca_excluded(sym: str, d0: int, d1: int) -> bool:
    """Is there an excluded corporate-action day of `sym` with d0 < day <= d1?"""
    v = _EXC.get(sym)
    if not v:
        return False
    i = bisect.bisect_right(v, int(d0))
    return i < len(v) and v[i] <= int(d1)


def coverage() -> dict:
    out = {}
    for name in ("insider", "results", "boardmeet"):
        t = _load(name)
        by: dict = {}
        for v in t.values():
            for y in (v["ts"] // 100000000).tolist():
                by[int(y)] = by.get(int(y), 0) + 1
        out[name] = {"symbols": len(t), "rows": int(sum(by.values())), "by_year": dict(sorted(by.items()))}
    return out


if __name__ == "__main__":
    print(json.dumps(coverage(), indent=1))
