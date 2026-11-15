"""
pead_data.py — the live inputs of the EV-PEAD-LONG-H60 PAPER candidate (agents/validated_pead.py), built the way
the lab built them. Read-only towards trading: nothing here places an order. Every network helper is polite
(sequential, a pause between requests, a generic browser User-Agent, no credentials, no personal or project
information) and never raises: it returns a status, and the agent takes NO entry when an input is missing.

Lab ↔ live (the lab's own functions are imported and called, never copied):

  daily bars      lab: stratlab/events_daily.py read_new() on NSE `sec_bhavdata_full` (2023-11 .. 2026-11): EQ row
                  preferred over BE, official open / high / low / volume, close = LAST traded price
                  (data_daily._rows_to_arrays); corporate actions from NSE's corporate-actions feed, factor checked
                  against the ex-day gap (events_daily.new_factors → decide_factor) and multiplied into every
                  EARLIER row (prices × f, volume ÷ f); unexplained gaps (open / previous close < 0.6 or > 1.67)
                  and demerger / scheme / large-rights ex-days are the exclusion list (events_data.ca_excluded).
                  live: the same files, downloaded after the close from nsearchives.nseindia.com into
                  STORE/raw_cm/ under the lab's file name, read with the SAME read_new(); the same new_factors() on
                  the corporate-actions feed (fetched with events_fetch's client); the same gap rule.
  universe        lab: events_daily.monthly_universe() — each month, top 500 EQ equities (ETFs excluded) by
                  average daily turnover over the 60 sessions strictly before the month's first session,
                  traded on >= 75 % of them and on the last one. live: the same function on the same 61 files;
                  equity list = NSE EQUITY_L.csv minus eq_etfseclist.csv (lab: the same two files plus the
                  old-era INE list, which only matters for stocks delisted since). Renames: NSE symbolchange.csv
                  (nsearchives, refreshed every 3 days) applied by the lab's apply_renames_fast in read_bhav().
  NIFTY 50        lab: official daily closes (Kite day candles in the new era). live: NSE ind_close_all_DDMMYYYY.csv
                  ("Nifty 50" open / close); Kite day candles as the fallback.
  results events  lab: events_parse.build_results() on the NSE filing feeds fetched by events_fetch.py (structured
                  results: corporates-financial-results / integrated-filing-results; announcements: "Financial
                  Result Updates" and "Outcome of Board Meeting" on a board-meeting date notified for results):
                  first_ts = the first moment the numbers were public. live: the same feeds, the same client, the
                  same parser, over a rolling ~200-day window (STORE/raw/<feed>/*.json.gz, the lab's layout), plus
                  PROVISIONAL rows: a qualifying announcement whose structured filing is not out yet (the lab, with
                  hindsight, dates the event at that announcement — ~10 % of 2024-26 events had their filing only
                  after the evening decision).
                  If a feed fails, or the day's window holds no announcement at all on a trading day, the agent
                  takes NO entries that day (skip `no_event_feed`) — it never guesses.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import tempfile
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

ARCHIVE = "https://nsearchives.nseindia.com"
BHAV_URL = ARCHIVE + "/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
INDEX_URL = ARCHIVE + "/content/indices/ind_close_all_{ddmmyyyy}.csv"
EQUITY_L_URL = ARCHIVE + "/content/equities/EQUITY_L.csv"
ETF_URL = ARCHIVE + "/content/equities/eq_etfseclist.csv"
PAUSE_S = 1.5                       # between two requests to the same site
TIMEOUT_S = 30
KEEP_BHAV_SESSIONS = 100            # 61 for the universe, 22 for the reaction rule, up to ~65 for a hold's clock
KEEP_RAW_DAYS = 210                 # filing feeds: the lab accepts a structured filing up to 200 days after period end
BACKFILL_DAYS = 120                 # filing / corporate-action history fetched once when the store is empty
FEEDS = ("ifr", "fr", "ann_fr", "ann_obm", "bm", "ca")
NIFTY_50_TOKEN = 256265


def store_dir() -> Path:
    p = os.getenv("VALIDATED_PEAD_DIR")
    if p:
        return Path(p)
    perf = Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
    return perf.parent / "pead"


def ymd(d) -> int:
    if isinstance(d, int):
        return d
    return d.year * 10000 + d.month * 100 + d.day


def to_date(k: int) -> date:
    return date(k // 10000, k // 100 % 100, k % 100)


# ── session calendar (live) ──────────────────────────────────────────────────
def sessions_between(lo: date, hi: date, include_special: bool = False) -> list[int]:
    """Trading sessions in [lo, hi] (ist_clock.is_trading_day); special / short sessions (Muhurat, DR drills) are
    dropped as the lab's cache dropped them, unless include_special (the lab's universe history kept them)."""
    from ist_clock import is_trading_day
    import validated_data as VD
    special = VD.special_sessions()
    out, d = [], lo
    while d <= hi:
        if d in special:
            if include_special:
                out.append(ymd(d))
        elif is_trading_day(d):
            out.append(ymd(d))
        d += timedelta(days=1)
    return out


# ── small persistent state ───────────────────────────────────────────────────
def _state_file() -> Path:
    return store_dir() / "state.json"


def load_state() -> dict:
    try:
        return json.loads(_state_file().read_text())
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.error("[pead_data] state unreadable ({}) — starting empty", exc)
        return {}


def save_state(st: dict) -> None:
    f = _state_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=f.parent, prefix=".state.")
    with os.fdopen(fd, "w") as fh:
        json.dump(st, fh, separators=(",", ":"))
    os.replace(tmp, f)


# ── HTTP (archives) ──────────────────────────────────────────────────────────
_last_req = [0.0]


def _http_get(url: str) -> tuple[str, bytes]:
    """('ok', body) | ('missing', b'') for 404 | ('error', reason). Never raises."""
    import requests
    from stratlab.events_fetch import UA
    wait = PAUSE_S - (_time.time() - _last_req[0])
    if wait > 0:
        _time.sleep(wait)
    _last_req[0] = _time.time()
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept": "text/csv,text/plain,*/*",
                                       "Accept-Language": "en-US,en;q=0.9"}, timeout=TIMEOUT_S)
    except Exception as exc:
        return "error", str(exc)[:160].encode()
    if r.status_code == 200 and r.content:
        return "ok", r.content
    if r.status_code == 404:
        return "missing", b""
    return "error", f"HTTP {r.status_code}".encode()


# ── bhavcopy store ───────────────────────────────────────────────────────────
def raw_cm_dir() -> Path:
    return store_dir() / "raw_cm"


def bhav_path(day: int) -> Path:
    return raw_cm_dir() / f"sec_bhavdata_full_{day}.csv.gz"


def bhav_date_inside(body: bytes) -> Optional[int]:
    """The DATE1 of the file's first data row (NSE serves the previous session's file on holidays)."""
    from stratlab.events_parse import parse_date
    try:
        lines = body.decode("latin-1").splitlines()
        hdr = [c.strip() for c in lines[0].split(",")]
        i = hdr.index("DATE1")
        d = parse_date(lines[1].split(",")[i].strip())
        return None if d is None else ymd(d)
    except Exception:
        return None


def fetch_bhav(day: int) -> str:
    """Download one session's sec_bhavdata_full into the store (the lab's file name). 'ok' | 'have' | 'missing' |
    'wrong_date' | 'error'."""
    p = bhav_path(day)
    if p.exists():
        return "have"
    d = to_date(day)
    status, body = _http_get(BHAV_URL.format(ddmmyyyy=d.strftime("%d%m%Y")))
    if status != "ok":
        return status
    if bhav_date_inside(body) != day:
        return "wrong_date"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        f.write(body)
    os.replace(tmp, p)
    return "ok"


def ensure_bhav(days: Iterable[int], max_fetch: int = 80) -> dict:
    """Make sure the files of `days` are in the store (fetching at most max_fetch). {day: status}."""
    out, n = {}, 0
    for day in sorted(set(int(x) for x in days)):
        if bhav_path(day).exists():
            out[day] = "have"
            continue
        if n >= max_fetch:
            out[day] = "not_fetched"
            continue
        out[day] = fetch_bhav(day)
        n += 1
    return out


def prune_bhav(keep_from: int) -> int:
    n = 0
    for f in raw_cm_dir().glob("sec_bhavdata_full_*.csv.gz"):
        try:
            if int(f.name[-15:-7]) < keep_from:
                f.unlink()
                n += 1
        except ValueError:
            continue
    return n


def read_bhav(lo: int, hi: int, renames: bool = True):
    """The lab's events_daily.read_new() over the store: rows dated (lo, hi], series EQ / BE, every NSE rename in
    the range followed to the newer symbol (events_daily.apply_renames_fast, the lab's rename step; NSE's filing
    feeds serve current symbols). None when empty."""
    from stratlab import events_daily as ED
    if not any(raw_cm_dir().glob("sec_bhavdata_full_*.csv.gz")):
        return None
    try:
        eq = ED.read_new(str(raw_cm_dir()), lo, hi)
    except ValueError:                      # pandas.concat of nothing
        return None
    ch = [c for c in symbol_changes() if lo < c[0] <= hi] if renames else []
    if ch:
        eq, _ = ED.apply_renames_fast(eq, ch)
    return eq


# ── NSE symbol changes ───────────────────────────────────────────────────────
SYMBOLCHANGE_URL = ARCHIVE + "/content/equities/symbolchange.csv"


def symbol_changes() -> list[tuple]:
    """[(effective yyyymmdd, old, new)] from the stored NSE symbolchange.csv (events_parse.load_changes)."""
    from stratlab.events_parse import load_changes
    return load_changes(str(store_dir() / "ref" / "symbolchange.csv"))


def ensure_symbol_changes(st: dict, today: date, refresh_days: int = 3) -> list[tuple]:
    """Refresh NSE's symbolchange.csv every few days (the last good copy is kept)."""
    if st.get("symbolchange_fetched", "") < (today - timedelta(days=refresh_days)).isoformat():
        status, body = _http_get(SYMBOLCHANGE_URL)
        if status == "ok" and body.count(b"\n") > 100:
            p = store_dir() / "ref" / "symbolchange.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(body)
            os.replace(tmp, p)
            st["symbolchange_fetched"] = today.isoformat()
    return symbol_changes()


# ── equity list and the monthly universe ─────────────────────────────────────
def parse_equity_list(equity_l: bytes, etf_list: bytes) -> set[str]:
    """EQUITY_L.csv SYMBOL column minus the ETF list's first column (events_daily.build's equity rule, live)."""
    import csv
    import io
    eq = {r["SYMBOL"].strip().upper() for r in csv.DictReader(io.StringIO(equity_l.decode("latin-1")))
          if r.get("SYMBOL")}
    etf = {r[0].strip().upper() for r in csv.reader(io.StringIO(etf_list.decode("latin-1"))) if r}
    return eq - etf


def equity_symbols(st: dict, today: date, refresh_days: int = 7) -> Optional[set[str]]:
    """The equity list, refreshed weekly; the last good copy is kept when NSE does not answer."""
    eqs = st.get("equity") or {}
    fresh = eqs.get("fetched", "") >= (today - timedelta(days=refresh_days)).isoformat()
    if not fresh:
        s1, b1 = _http_get(EQUITY_L_URL)
        s2, b2 = _http_get(ETF_URL) if s1 == "ok" else ("skip", b"")
        if s1 == "ok" and s2 == "ok":
            try:
                syms = parse_equity_list(b1, b2)
                if len(syms) > 1000:
                    st["equity"] = eqs = {"fetched": today.isoformat(), "symbols": sorted(syms)}
            except Exception as exc:
                logger.warning("[pead_data] equity list unparsable: {}", exc)
    return set(eqs["symbols"]) if eqs.get("symbols") else None


def month_first_session(day: int, sessions: list[int]) -> Optional[int]:
    m = day // 100
    firsts = [s for s in sessions if s // 100 == m]
    return firsts[0] if firsts else None


def universe_for(day: int, st: dict, equity: set[str], top: int = 500) -> tuple[Optional[set[str]], str]:
    """(members of `day`'s month | None, status). Built once per month with events_daily.monthly_universe() on the
    61 stored files (60 sessions before the month's first session + that session), cached in the state."""
    from stratlab import events_daily as ED
    month = day // 100
    cached = (st.get("universe") or {}).get(str(month))
    if cached and cached.get("symbols"):
        return set(cached["symbols"]), "cached"
    d = to_date(day)
    cal = sessions_between(d - timedelta(days=130), d, include_special=True)
    first = month_first_session(day, [s for s in cal if not _is_special(s)])
    if first is None:
        return None, "no_first_session"
    hist = [s for s in cal if s < first][-60:]
    if len(hist) < 60:
        return None, "short_calendar"
    need = hist + [first]
    have = [s for s in need if bhav_path(s).exists()]
    if len(have) < len(need):
        return None, f"missing_files:{len(need) - len(have)}"
    eq = read_bhav(hist[0] - 1, first)
    if eq is None or not len(eq):
        return None, "no_rows"
    try:
        uni = ED.monthly_universe(eq, equity, top)
    except Exception as exc:
        return None, f"error:{str(exc)[:80]}"
    rows = uni[uni["month"] == month]
    if len(rows) < top * 0.9:
        return None, f"short_universe:{len(rows)}"
    syms = sorted(rows["symbol"].astype(str))
    st.setdefault("universe", {})[str(month)] = {"effective_from": int(first), "symbols": syms,
                                                 "built": datetime.now().isoformat(timespec="seconds")}
    for k in sorted(st["universe"])[:-3]:              # keep the last three months
        st["universe"].pop(k, None)
    return set(syms), "built"


def _is_special(k: int) -> bool:
    import validated_data as VD
    return to_date(k) in VD.special_sessions()


# ── NIFTY 50 ─────────────────────────────────────────────────────────────────
def parse_index_file(body: bytes, name: str = "Nifty 50") -> Optional[tuple[float, float]]:
    """(open, close) of `name` from ind_close_all_DDMMYYYY.csv."""
    import csv
    import io
    for r in csv.DictReader(io.StringIO(body.decode("latin-1"))):
        if (r.get("Index Name") or "").strip().lower() == name.lower():
            try:
                o, c = float(r["Open Index Value"]), float(r["Closing Index Value"])
            except (KeyError, TypeError, ValueError):
                return None
            return (o, c) if o > 0 and c > 0 else None
    return None


def ensure_nifty(st: dict, days: Iterable[int]) -> dict[int, tuple[float, float]]:
    """NIFTY 50 (open, close) of each session in `days`: NSE's index file, else Kite day candles."""
    nf = st.setdefault("nifty", {})
    missing = [d for d in sorted(set(days)) if str(d) not in nf]
    kite_rows = None
    for day in missing:
        status, body = _http_get(INDEX_URL.format(ddmmyyyy=to_date(day).strftime("%d%m%Y")))
        oc = parse_index_file(body) if status == "ok" else None
        if oc is None:
            if kite_rows is None:
                kite_rows = _kite_nifty_days(min(missing), max(missing))
            oc = kite_rows.get(day)
        if oc is not None:
            nf[str(day)] = [round(oc[0], 2), round(oc[1], 2)]
    for k in sorted(nf)[:-KEEP_BHAV_SESSIONS - 20]:
        nf.pop(k, None)
    return {int(k): (float(v[0]), float(v[1])) for k, v in nf.items()}


def _kite_nifty_days(lo: int, hi: int) -> dict[int, tuple[float, float]]:
    try:
        from kite_client import kite_client
        rows = kite_client.historical_data(NIFTY_50_TOKEN, datetime.combine(to_date(lo), datetime.min.time()),
                                           datetime.combine(to_date(hi), datetime.max.time().replace(microsecond=0)),
                                           "day") or []
    except Exception as exc:
        logger.warning("[pead_data] NIFTY day candles unavailable: {}", exc)
        return {}
    out = {}
    for r in rows:
        ts = r["date"] if not isinstance(r["date"], str) else datetime.fromisoformat(r["date"])
        if float(r.get("open") or 0) > 0 and float(r.get("close") or 0) > 0:
            out[ymd(ts.date())] = (float(r["open"]), float(r["close"]))
    return out


# ── NSE filing feeds (events_fetch's client, events_parse's parser) ─────────
def raw_dir() -> Path:
    return store_dir() / "raw"


def _windows(feed: str, lo: date, hi: date) -> list[tuple[date, date]]:
    """Calendar-month windows covering [lo, hi] (the lab's month windows), cut at lo / hi."""
    out, d = [], date(lo.year, lo.month, 1)
    while d <= hi:
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        out.append((max(d, lo), min(nxt - timedelta(days=1), hi)))
        d = nxt
    return out


class _Client:
    """stratlab.events_fetch.Client with a 30 s request timeout (the lab's research downloads allowed 180 s)."""

    def __init__(self, pause: float = PAUSE_S):
        from stratlab import events_fetch as EF
        self.EF = EF
        self.cl = EF.Client(pause)
        get = self.cl.s.get
        self.cl.s.get = lambda url, **kw: get(url, **{**kw, "timeout": min(float(kw.get("timeout") or TIMEOUT_S),
                                                                               TIMEOUT_S)})

    def fetch(self, feed: str, d0: date, d1: date) -> tuple[str, list]:
        """(status, [payload bytes per page])."""
        EF = self.EF
        if feed != "ifr":
            status, body = self.cl.get(EF.api_path(feed, d0, d1), EF.PAGES[feed])
            return status, ([body] if status == "ok" else [])
        bodies = []
        for page in range(1, 31):
            status, body = self.cl.get(EF.api_path(feed, d0, d1) + f"&page={page}", EF.PAGES[feed])
            if status != "ok":
                return status, bodies
            try:
                n = len(EF.rows_of(json.loads(body)))
            except ValueError:
                return "error", bodies
            if n:
                bodies.append(body)
            if n < EF.IFR_PAGE:
                break
        return "ok", bodies


def _save_payloads(feed: str, d0: date, d1: date, bodies: list) -> None:
    dd = raw_dir() / feed
    dd.mkdir(parents=True, exist_ok=True)
    for i, body in enumerate(bodies):
        p = dd / f"{d0:%Y%m%d}_{d1:%Y%m%d}_p{i + 1:02d}.json.gz"
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            f.write(body)
        os.replace(tmp, p)


def prune_raw(today: date) -> int:
    """Drop feed files whose window ENDED more than KEEP_RAW_DAYS ago."""
    cut = (today - timedelta(days=KEEP_RAW_DAYS)).strftime("%Y%m%d")
    n = 0
    for f in raw_dir().glob("*/*.json.gz"):
        parts = f.name.split("_")
        if len(parts) >= 2 and parts[1] < cut:
            f.unlink()
            n += 1
    return n


def feed_windows(feed: str, today: date, backfill: bool) -> list[tuple[date, date]]:
    """Windows to ask for: the last ~6 days (the board-meeting feed by MEETING date, the corporate-action feed by
    EX-date, both a little into the future); on the first run the BACKFILL_DAYS before that, in month windows."""
    lo = today - timedelta(days=6)
    hi = today + timedelta(days={"ca": 10, "bm": 3}.get(feed, 0))
    out = []
    if backfill:
        out += _windows(feed, today - timedelta(days=BACKFILL_DAYS), lo - timedelta(days=1))
    out.append((lo, hi))
    return out


def fetch_feeds(today: date, feeds: Iterable[str] = FEEDS, client=None) -> dict:
    """Fetch the filing / corporate-action feeds for the recent window (and a one-off backfill). Stops at the first
    failure: {"ok": bool, "feeds": {feed: status}, "rows": {feed: n}}."""
    st = load_state()
    done = set((st.get("feeds_backfilled") or {}).keys())
    cl = client or _Client()
    res = {"ok": True, "feeds": {}, "rows": {}}
    for feed in feeds:
        backfill = feed not in done
        n = 0
        for d0, d1 in feed_windows(feed, today, backfill):
            try:
                status, bodies = cl.fetch(feed, d0, d1)
            except Exception as exc:
                status, bodies = "error", []
                logger.warning("[pead_data] {} fetch raised: {}", feed, exc)
            if status != "ok":
                res["ok"] = False
                res["feeds"][feed] = status
                logger.warning("[pead_data] {} {}..{}: {}", feed, d0, d1, status)
                break
            _save_payloads(feed, d0, d1, bodies)
            try:
                n += sum(len(cl.EF.rows_of(json.loads(b))) for b in bodies) if hasattr(cl, "EF") else len(bodies)
            except Exception:
                pass
        else:
            res["feeds"][feed] = "ok"
            if backfill:
                st.setdefault("feeds_backfilled", {})[feed] = today.isoformat()
        res["rows"][feed] = n
        if not res["ok"]:
            break
    prune_raw(today)
    save_state(st)
    return res


# ── results events (lab parser + provisional announcements) ─────────────────
def quarter_end_before(d: date) -> date:
    """The last calendar quarter end strictly before `d`."""
    for m in (12, 9, 6, 3):
        qe = date(d.year, m, 31 if m in (3, 12) else 30)
        if qe < d:
            return qe
    return date(d.year - 1, 12, 31)


def results_events(raw: Optional[str] = None) -> list[dict]:
    """[{symbol, first_ts, period_end, src}] — events_parse.build_results() on the stored feeds (exactly the lab's
    first_ts rule), plus provisional rows for qualifying announcements whose structured filing is not out yet."""
    from stratlab import events_parse as EP
    raw = raw or str(raw_dir())
    bm, _ = EP.build_boardmeet(EP.load_feed(raw, "bm"), [])
    meetings: dict = {}
    for r in bm:
        if r["is_results"]:
            meetings.setdefault(r["symbol"], set()).add(date.fromisoformat(r["meeting_date"]))
    ann_fr, ann_obm = list(EP.load_feed(raw, "ann_fr")), list(EP.load_feed(raw, "ann_obm"))
    res, _ = EP.build_results(EP.load_feed(raw, "fr"), ann_fr, ann_obm, [], EP.load_feed(raw, "ifr"), meetings)
    out = [{"symbol": r["symbol"], "first_ts": r["first_ts"], "period_end": r["period_end"], "src": r["src_first"]}
           for r in res]
    filed = {(r["symbol"], r["period_end"]) for r in res}
    prov: dict = {}
    for kind, rows in (("fr", ann_fr), ("obm", ann_obm)):
        for r in rows:
            ts = EP.parse_ts(r.get("an_dt"))
            if ts is None:
                continue
            sym = (r.get("symbol") or "").strip().upper()
            if kind == "obm" and ts.date() not in meetings.get(sym, ()):
                continue
            qe = quarter_end_before(ts.date()).isoformat()
            if (sym, qe) in filed:
                continue                   # the structured filing is out: build_results already dated the event
            k = (sym, qe)
            if k not in prov or ts < prov[k]:
                prov[k] = ts
    out += [{"symbol": s, "first_ts": EP.fmt(t), "period_end": qe, "src": "provisional"}
            for (s, qe), t in sorted(prov.items())]
    return out


def window_has_announcements(raw: Optional[str], lo_ts: datetime, hi_ts: datetime) -> int:
    """Number of feed rows (announcements and structured filings) broadcast in [lo_ts, hi_ts): 0 on a trading day
    means the feed is not answering (NSE files dozens of board-meeting outcomes every session)."""
    from stratlab import events_parse as EP
    raw = raw or str(raw_dir())
    n = 0
    for feed, key in (("ann_obm", "an_dt"), ("ann_fr", "an_dt"), ("ifr", "broadcast_Date"), ("fr", "broadCastDate")):
        for r in EP.load_feed(raw, feed):
            ts = EP.parse_ts(r.get(key))
            if ts is not None and lo_ts <= ts < hi_ts:
                n += 1
    return n


def corporate_actions(raw: Optional[str] = None) -> list[tuple]:
    """[(symbol, ex yyyymmdd, subject)] — events_daily.load_actions() on the stored corporate-action feed."""
    from stratlab import events_daily as ED
    return ED.load_actions(raw or str(raw_dir()))


# ── adjusted daily bars (events_daily.build's new-era rules on the live store) ─────
@dataclass(frozen=True)
class Bars:
    """One symbol's adjusted daily bars, oldest first (index -1 = the latest session)."""
    dates: tuple
    o: tuple
    h: tuple
    l: tuple
    c: tuple            # LAST traded price (the lab's close), CLOSE_PRICE when LAST is missing
    v: tuple


def build_bars(eq, actions: list, sessions: list[int]) -> tuple[dict, list, list]:
    """eq: read_new() rows; actions: corporate_actions(); sessions: the calendar to keep (special sessions out).
    Returns ({symbol: Bars}, exclusions [(symbol, day)], applied factors [(symbol, first_session, f, status)]) with
    events_daily.build's rules: EQ preferred over BE, new_factors() on the actions (factor × every earlier row,
    volume ÷), exclusions = ambiguous ex-days (scheme / rights adjusted by the gap, scheme not applied) and
    unexplained gaps (open / previous close < 0.6 or > 1.67 after adjustment)."""
    import numpy as np
    from stratlab import events_daily as ED
    from stratlab import data_daily as DD
    if eq is None or not len(eq):
        return {}, [], []
    keep = set(int(x) for x in sessions)
    eq = eq[eq["date"].isin(keep)].copy()
    eq["_pri"] = (eq["series"] != "EQ").astype(int)
    eq = eq.sort_values(["symbol", "date", "_pri"]).drop_duplicates(["symbol", "date"], keep="first")
    eq = eq.drop(columns="_pri").reset_index(drop=True)
    lo = int(eq["date"].min()) - 1
    applied, calog = ED.new_factors(eq, actions, lo=lo)
    date_arr = eq["date"].to_numpy()
    pos = eq.groupby("symbol", sort=False).indices
    cols = {c: eq[c].to_numpy(float).copy() for c in ("open", "high", "low", "close", "last", "volume")}
    for sym, fs, f, _st, _sb in applied:
        ii = pos[sym]
        ii = ii[date_arr[ii] < fs]
        for c in ("open", "high", "low", "close", "last"):
            cols[c][ii] *= f
        cols["volume"][ii] /= f
    for c, v in cols.items():
        eq[c] = v
    excl = {(r["symbol"], int(r["first_session"])) for r in calog
            if r["status"] in ("applied_gap_scheme", "applied_gap_rights", "not_applied_scheme")}
    same = eq["symbol"].eq(eq["symbol"].shift())
    g = (eq["open"] / eq["close"].shift()).where(same)
    big = eq[(g < 0.6) | (g > 1.67)]
    excl |= {(str(s), int(d)) for s, d in zip(big["symbol"], big["date"])}
    ok = (eq[["open", "high", "low", "close"]] > 0).all(axis=1) & \
        np.isfinite(eq[["open", "high", "low", "close"]]).all(axis=1)
    eq = eq[ok]
    out = {}
    for sym, gr in eq.groupby("symbol", sort=True):
        a = DD._rows_to_arrays(gr)
        out[str(sym)] = Bars(tuple(int(x) for x in a["date"]), tuple(a["o"].tolist()), tuple(a["h"].tolist()),
                             tuple(a["l"].tolist()), tuple(a["c"].tolist()), tuple(a["v"].tolist()))
    return out, sorted(excl), [(s, int(fs), float(f), st) for s, fs, f, st, _ in applied]


def session_rows(day: int, symbols: Iterable[str]) -> dict[str, dict]:
    """Unadjusted rows of `day` (EQ preferred over BE) for `symbols`: open, high, low, close (LAST, else CLOSE),
    prev_close. {} when the file is not in the store."""
    eq = read_bhav(day - 1, day)
    if eq is None:
        return {}
    want = set(symbols)
    eq = eq[eq["symbol"].isin(want)].copy()
    eq["_pri"] = (eq["series"] != "EQ").astype(int)
    eq = eq.sort_values(["symbol", "_pri"]).drop_duplicates(["symbol"], keep="first")
    out = {}
    for r in eq.itertuples(index=False):
        last = float(r.last) if math.isfinite(float(r.last)) and float(r.last) > 0 else float(r.close)
        out[str(r.symbol)] = {"open": float(r.open), "high": float(r.high), "low": float(r.low), "close": last,
                              "close_price": float(r.close), "prev_close": float(r.prev_close),
                              "volume": float(r.volume)}
    return out


def traded_days(symbol: str, lo: int, hi: int) -> Optional[list[int]]:
    """Sessions in [lo, hi] on which `symbol` has a row in the store (the lab counts the stock's own bars), or None
    when a session file of the range is missing."""
    d0, d1 = to_date(lo), to_date(hi)
    cal = sessions_between(d0, d1)
    if any(not bhav_path(s).exists() for s in cal):
        return None
    eq = read_bhav(lo - 1, hi)
    if eq is None:
        return None
    days = sorted(set(int(x) for x in eq.loc[eq["symbol"] == symbol, "date"]) & set(cal))
    return days
