"""
builder_data.py — the DATA LAYER of the automatic strategy builder: the data the lab's strategies read, kept on the
server's volume, so strategies can be built and evaluated over ~10 years without the research sandbox.

Everything is flag-gated: BUILDER_DATA_ENABLED (default false) -> no job is scheduled, no network call is made, the
jobs return at once. PAPER only. Read-only towards trading: no order entry point, agent, risk or trading state is
imported, read or changed.

LAYOUT (<root> = BUILDER_DATA_DIR, else <dir of PERFORMANCE_FILE>/builder_data — the Railway volume)
  minute/       deep 1-minute bars 2016 -> (builder_minute.py; the forward lab's store format and reader) of the
                forward lab's 99 instruments + (BUILDER_MINUTE_UNIVERSE=nifty200, default) the added NIFTY 200 names
                (builder_universe.py; universe.json lists them)
  nse/          NSE end-of-day archives 2012 -> in EXACTLY the layout stratlab/nsedata.py reads
                (STRATLAB_NSE_DIR=<root>/nse STRATLAB_BARS_DIR=<root>/nse/bars and lab code runs unchanged):
      delivery.csv.gz            sec_bhavdata_full EQ rows (2020 ->), CM bhavcopy + MTO delivery file before 2020
                                 (the old-era fetch's sources), columns of the lab's build_delivery; bars_symbol = the
                                 lab's 97 bar names (+ its three aliases ZOMATO / MCDOWELL-N / MACROTECH), as in the
                                 lab's 3-year file. One gzip member per session appended (delivery.idx.json: offset,
                                 length, rows per session; a crashed append is truncated back to the committed length)
      fo_parts/{fut,optidx,optstk}/YYYYMMDD.csv.gz + status.csv   F&O bhavcopy (UDiFF from 2024, the old format
                                 before), parsed with the lab's fetch_fo normalisation: all futures, every NIFTY /
                                 BANKNIFTY option strike, per-(stock, expiry) option aggregates
      fo_parts/optstk_full/      per-strike STOCK option chains are NOT kept (lab: 217 MB for 2023-11..2026-11 and
                                 311 MB for 2012..2023, ~0.53 GB skipped) EXCEPT the nearest expiry on the sessions
                                 1-4 calendar days before a stock expiry (the chain the expiry-pin rules read, ~3 %)
      participant_oi.csv.gz, participant_vol.csv.gz, ban.csv.gz (+ raw_part/{oi,vol,ban}/YYYYMMDD.csv), parsed by
                                 stratlab.oldera_fo's own parsers; fii_stats.csv.gz (+ raw_fii/YYYYMMDD.csv, parsed
                                 like databuild build_fii: rows dated another day inside the sheet are dropped)
      bars/_NIFTY50_day.json, _NIFTYBANK_min.json, _VIX_day.json   index closes from NSE's ind_close_all files
                                 (nsedata's index spot, the old-era view's convention)
  nse_view/     the last VIEW_SESSIONS (400) sessions of nse/ (symlinks + a 400-session delivery file): what the
                forward lab's engine reads (STRATLAB_NSE_DIR, see engine_env()); nsedata builds its .npz cache there
                (~0.2 GB; the cache of the whole 2012 -> archive would be ~2 GB, too big for the volume)
  flow/         stratlab/flowdata.py's layout (STRATLAB_FLOW_DIR): deals/deals.csv.gz, corpactions/corpactions.csv.gz,
                index_membership/{index_changes.csv, index_members.csv.gz, index_weights.csv.gz, data_check.json},
                fii_stats/fii_stats.csv.gz (link to nse/)
  events/       stratlab/events_data.py's layout (STRATLAB_EVENTS_DIR): results.csv.gz, boardmeet.csv.gz built by
                stratlab.events_parse from the NSE filing feeds (raw/<feed>/YYYYMM.json.gz, pead_data's client)

AVAILABILITY: unchanged — the files carry the trade date / broadcast time and the lab's accessors apply their own
`before` rule (an EOD value dated D is usable from the next session's open).

NETWORK ETIQUETTE: one request at a time, >= 1.5 s apart, a generic browser User-Agent (stratlab.events_fetch.UA), no
credentials, exponential backoff on errors. Every request counts against the run's budget and time window — including
each page warm-up, retry and IFR page the filing-feed client makes (guard_client wraps that client's session for the
builder only; a guard that trips mid-pagination saves nothing and the window stays pending). A 401 / 403 is a REFUSAL: three in a row stop the run, the
refusal is recorded (status "refused") and nothing is retried for 24 h; bot protection is never evaded. With NSE
refusing, the builder simply has no new data.

SCHEDULE (IST, only with the flag on): NSE archives 23:05 and 05:30 daily (+ 10:05 / 15:05 on Sat-Sun), never
inside 19:20-23:00 or 08:10-08:30 (the PEAD agent's 19:35-22:35 / 08:20 fetches from the same archive); Kite minute
backfill 18:40 and 06:10 daily (+ 11:40 / 14:40 Sat-Sun), never 09:00-15:45 on a trading day.

DISK: BUILDER_DATA_MAX_MB (default 2,600) for the whole <root>; measured as allocated blocks (symlinks not followed,
hard links counted once). At the cap nothing more is written (status "disk_cap"); nothing is ever deleted except the
builder's own nse_view, which is rebuilt in place.
Measured from the lab's scratch copies (2026-11, allocated blocks), full backfill 2012-01 .. 2026-11:
  minute bars 2016 -> (99 instruments, ~49 KB / instrument-month before 2023, ~55 KB after)   ~655 MB  (+65 MB/yr)
  delivery (77 KB / session from 2020, 61 KB before; ~3,690 sessions)                         ~247 MB  (+19 MB/yr)
  fo_parts fut + optidx + optstk (lab files: 190 MB for 2012-2023, 65 MB for 2023-26)         ~256 MB  (+22 MB/yr)
  fo_parts optstk_full, expiry eves only (full chains skipped: 526 MB)                         ~49 MB   (+5 MB/yr)
  raw participant OI / volume / ban, FII rows, index closes (small files, 4 KB blocks)          ~70 MB   (+5 MB/yr)
  filing feeds + results / board meetings, deals, corporate actions, index files               ~50 MB   (+3 MB/yr)
  nse_view (400-session delivery file + nsedata's .npz cache: 332 MB per 753 sessions)        ~210 MB  (flat)
  total ~1.54 GB at the end of the backfill, ~115 MB a year after -> the 1.8 GB cap in ~2 years (then it stops).
NIFTY 200 minute universe (builder_universe.py, 2026-11): ~105 current NIFTY 200 names + ~32 historical NIFTY 100
members since 2016 (some not on Kite) at the same ~6.7 MB per instrument for 2016-01 .. 2026-11            <= ~920 MB
  (worst case: every added name with full history; recent listings have less)                              (+90 MB/yr)
  total worst case ~2.46 GB at the end of the backfill, ~205 MB a year after -> cap 2,600 MB. Volume (5,000 MB) worst
  case = forward lab 500 + option-bar archiver 1,024 + builder 400 + builder data 2,600 = 4,524 MB (<= 4,600).
  GET /api/v2/builder/data -> datasets.minute.universe.projection re-measures it from the store itself.
"""
from __future__ import annotations

import ast
import csv
import gzip
import io
import json
import os
import re
import shutil
import threading
import time as _time
import zipfile
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
NSE_START = date(2012, 1, 1)
SEC_FULL_FROM = 20200101            # sec_bhavdata_full exists on nsearchives from 2020 (404 before)
UDIFF_FROM = 20240101               # F&O UDiFF bhavcopy first from here, the old format first before
FII_FROM = 20141201                 # the earliest FII statistics sheet NSE serves
VIEW_SESSIONS = 400
FINAL_AFTER_DAYS = 3                # a 404 / no-session answer for a day older than this is final
PAUSE_S = 1.5
TIMEOUT_S = 30
REFUSAL_STRIKES = 3
REFUSAL_BACKOFF_H = 24
MAX_FAILURES = 12                   # failed requests in a row (after retries) that end a run
DEFAULT_MAX_MB = 2600.0              # 1,800 before the NIFTY 200 minute universe (builder_universe.py)
ARCHIVE = "https://nsearchives.nseindia.com"
SITE = "https://www.nseindia.com"
# the lab's build_delivery aliases (historic NSE symbol -> bars3y name)
ALIASES = {"ZOMATO": "ETERNAL", "MCDOWELL-N": "UNITDSPR", "MACROTECH": "LODHA"}
DELIVERY_COLS = ["date", "symbol", "open", "high", "low", "close", "prev_close", "avg_price", "volume",
                 "turnover_lacs", "trades", "deliv_qty", "deliv_pct", "bars_symbol"]
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
INDEX_NAMES = {"NIFTY": ("nifty 50", "cnx nifty", "s&p cnx nifty"),
               "BANKNIFTY": ("nifty bank", "bank nifty", "cnx bank", "cnx bank index"),
               "INDIAVIX": ("india vix",),
               "NIFTYNEXT50": ("nifty next 50", "cnx nifty junior", "nifty junior"),
               "NIFTY100": ("nifty 100", "cnx 100")}
DAY_SETS = ("cash", "idx", "fo", "poi", "pvol", "ban", "fii")
FEEDS = ("bm", "fr", "ann_fr", "ann_obm", "ifr")
EVENTS_FROM = date(2011, 10, 1)
_LOCK = threading.RLock()


class DiskCapReached(RuntimeError):
    pass


class BudgetReached(RuntimeError):
    pass


class NSERefused(RuntimeError):
    pass


class StopRun(RuntimeError):
    pass


# ── small helpers ─────────────────────────────────────────────────────────────────────────────────────────────────
def ymd(d) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def to_date(k) -> date:
    k = int(k)
    return date(k // 10000, k // 100 % 100, k % 100)


def _settings():
    from config import settings
    return settings


def enabled() -> bool:
    try:
        s = _settings()
        return bool(getattr(s, "builder_data_enabled", False)) and s.trading_mode == "PAPER"
    except Exception:
        return False


def default_root() -> Path:
    p = os.getenv("BUILDER_DATA_DIR")
    if p:
        return Path(p)
    perf = os.getenv("PERFORMANCE_FILE")
    base = Path(perf).parent if perf else HERE / "data"
    return base / "builder_data"


def max_bytes() -> int:
    try:
        mb = float(getattr(_settings(), "builder_data_max_mb", DEFAULT_MAX_MB))
    except Exception:
        mb = float(os.getenv("BUILDER_DATA_MAX_MB", DEFAULT_MAX_MB))
    return int(mb * 1024 * 1024)


def disk_bytes(root) -> int:
    """Allocated bytes of the regular files under root (symlinks not followed, hard links counted once)."""
    root = Path(root)
    if not root.exists():
        return 0
    seen, tot = set(), 0
    for dp, dns, fns in os.walk(root):
        for fn in fns:
            try:
                st = os.lstat(os.path.join(dp, fn))
            except OSError:
                continue
            if not (st.st_mode & 0o170000 == 0o100000):        # regular files only
                continue
            if (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            tot += max(st.st_size, getattr(st, "st_blocks", 0) * 512)
    return tot


def in_market_hours(t: datetime) -> bool:
    from ist_clock import is_trading_day
    return is_trading_day(t.date()) and dtime(9, 0) <= t.time() < dtime(15, 45)


def in_pead_window(t: datetime) -> bool:
    """The PEAD agent fetches from the same NSE archive at 19:35-22:35 and 08:20: the builder stays out of the way."""
    return dtime(19, 20) <= t.time() < dtime(23, 0) or dtime(8, 10) <= t.time() < dtime(8, 30)


def _atomic(path: Path, data: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return len(data)


def _jload(path: Path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _jsave(path: Path, obj) -> None:
    _atomic(Path(path), json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode())


# ── network ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _requests_get(url: str) -> tuple[str, bytes]:
    """('ok', body) | ('missing', b'') 404 | ('refused', b'') 401/403 | ('blocked', reason) | ('error', reason)."""
    import requests
    from stratlab.events_fetch import UA
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"},
                         timeout=TIMEOUT_S)
    except requests.exceptions.ProxyError as exc:
        return "blocked", str(exc)[:160].encode()
    except Exception as exc:
        return "error", str(exc)[:160].encode()
    if r.status_code == 200 and r.content:
        return "ok", r.content
    if r.status_code == 404:
        return "missing", b""
    if r.status_code in (401, 403):
        return "refused", f"HTTP {r.status_code}".encode()
    return "error", f"HTTP {r.status_code}".encode()


class Http:
    """One request at a time, >= pause s apart, counted against the run's budget; backoff on errors; three
    refusals in a row raise NSERefused (never evaded)."""

    def __init__(self, budget: int, get: Optional[Callable] = None, pause: float = PAUSE_S,
                 sleep: Callable = _time.sleep, stop: Optional[Callable[[], Optional[str]]] = None) -> None:
        self.budget, self.n = int(budget), 0
        self._get = get or _requests_get
        self.pause, self.sleep = pause, sleep
        self.stop = stop or (lambda: None)
        self.last = 0.0
        self.refusals = 0
        self.failures = 0
        self.statuses: dict = {}

    def _one(self, url: str) -> tuple[str, bytes]:
        if self.n >= self.budget:
            raise BudgetReached(f"request budget {self.budget} reached")
        why = self.stop()
        if why:
            raise StopRun(why)
        w = self.pause - (_time.time() - self.last)
        if w > 0:
            self.sleep(w)
        self.last = _time.time()
        self.n += 1
        return self._get(url)

    def before_request(self) -> None:
        """The budget and time-window check of ONE request (also used for requests made by other clients)."""
        if self.n >= self.budget:
            raise BudgetReached(f"request budget {self.budget} reached")
        why = self.stop()
        if why:
            raise StopRun(why)
        self.n += 1

    def after_status(self, code: int, resets: bool = True) -> None:
        """Refusal accounting of ONE answer: three 401 / 403 in a row raise NSERefused (an ordinary page load that
        answers 200 — a client's page warm-up — does not count as a success: resets=False)."""
        self.statuses[f"http_{code}"] = self.statuses.get(f"http_{code}", 0) + 1
        if code in (401, 403):
            self.refusals += 1
            if self.refusals >= REFUSAL_STRIKES:
                raise NSERefused(f"NSE refused {self.refusals} requests in a row (HTTP {code})")
        elif code == 200 and resets:
            self.refusals = 0

    def get(self, url: str, retries: int = 2) -> tuple[str, bytes]:
        delay = 5.0
        for attempt in range(retries + 1):
            status, body = self._one(url)
            self.statuses[status] = self.statuses.get(status, 0) + 1
            if status == "refused":
                self.refusals += 1
                if self.refusals >= REFUSAL_STRIKES:
                    raise NSERefused(f"NSE refused {self.refusals} requests in a row ({body.decode(errors='replace')})")
                return status, body
            self.refusals = 0
            if status in ("ok", "missing"):
                self.failures = 0
                return status, body
            if status == "blocked" or attempt == retries:
                self.failures += 1
                if self.failures >= MAX_FAILURES:
                    raise StopRun(f"{self.failures} failed requests in a row (network / NSE unavailable): {status}")
                return status, body
            self.sleep(delay)
            delay *= 2
        return "error", b""


def guard_client(client, http: Http) -> bool:
    """Route EVERY request a filing-feed client makes (events_fetch.Client's page warm-ups, retries and
    pead_data._Client's IFR pages) through the builder run's budget, time-window and refusal checks, by wrapping the
    `get` of that client's own requests session. Only the client object the builder created / was given is changed,
    so pead_data / events_fetch behave exactly as before for the live PEAD agent. False when no session is found."""
    inner_client = getattr(client, "cl", client)
    sess = getattr(inner_client, "s", None)
    if sess is None or not hasattr(sess, "get"):
        return False
    if getattr(sess, "_builder_http", None) is http:
        return True
    base = getattr(sess, "_builder_base_get", None) or sess.get

    def get(url, **kw):
        http.before_request()
        r = base(url, **kw)
        http.after_status(int(getattr(r, "status_code", 0) or 0), resets="/api/" in str(url))
        return r
    sess._builder_base_get = base
    sess.get = get
    sess._builder_http = http
    sess._builder_guarded = True
    return True


# ── symbols (NSE renames; the lab's names) ────────────────────────────────────────────────────────────────────────
def lab_names() -> list[str]:
    import forward_test
    return list(forward_test.UNIVERSE)


def bars_symbol(sym: str, universe: Optional[set] = None) -> str:
    u = universe if universe is not None else set(lab_names())
    return sym if sym in u else ALIASES.get(sym, "")


class SymMap:
    """databuild symmap.py's rename logic on NSE's symbolchange.csv with the forward lab's 97 names as the lab."""

    def __init__(self, changes: list, lab: list) -> None:
        self.ren: dict = {}
        for d, old, new in changes:
            self.ren.setdefault(old, []).append((int(d), new))
        for v in self.ren.values():
            v.sort()
        self.lab = set(lab)
        self.back = {}
        for name in lab:
            latest = self.latest_symbol(name, 19000101)
            if latest != name:
                self.back[latest] = name

    def latest_symbol(self, sym: str, day: int) -> str:
        cur, d = sym, int(day)
        for _ in range(12):
            nxt = [(e, n) for e, n in self.ren.get(cur, ()) if e > d]
            if not nxt:
                break
            d, cur = nxt[0]
        return cur

    def symbol_on(self, sym: str, day: int) -> str:
        cur, d = sym.strip().upper(), int(day)
        for _ in range(12):
            prev = sorted((e, old) for old, lst in self.ren.items() for e, new in lst if new == cur and e > d)
            if not prev:
                break
            e, old = prev[0]
            if [x for x, _ in self.ren.get(cur, ()) if d < x <= e]:
                break
            cur = old
        return cur

    def lab_symbol(self, sym: str, day: int) -> str:
        s = self.latest_symbol(sym.strip().upper(), day)
        return self.back.get(s, s)

    def in_lab(self, lab_sym: str) -> int:
        return int(lab_sym in self.lab)


# ── parsers (the lab's fetch / build scripts' normalisation) ──────────────────────────────────────────────────────
def _num(s):
    import pandas as pd
    return pd.to_numeric(s.astype(str).str.strip().replace({"": None, "-": None}), errors="coerce")


def parse_sec_full(body: bytes, day: int, universe: Optional[set] = None):
    """sec_bhavdata_full -> delivery rows (build_delivery.py: SERIES EQ, DATE1 must be `day`, else None = NSE served
    another session's file)."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(io.BytesIO(body), skipinitialspace=True, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda s: s.str.strip())
    df = df[df["SERIES"] == "EQ"]
    if not len(df):
        return None
    d = pd.to_datetime(df["DATE1"], format="%d-%b-%Y").dt.strftime("%Y%m%d").astype(int)
    if not (d == int(day)).all():
        return None
    out = pd.DataFrame({"date": d.to_numpy(), "symbol": df["SYMBOL"].to_numpy()})
    for src, dst in [("OPEN_PRICE", "open"), ("HIGH_PRICE", "high"), ("LOW_PRICE", "low"), ("CLOSE_PRICE", "close"),
                     ("PREV_CLOSE", "prev_close"), ("AVG_PRICE", "avg_price"), ("TTL_TRD_QNTY", "volume"),
                     ("TURNOVER_LACS", "turnover_lacs"), ("NO_OF_TRADES", "trades"), ("DELIV_QTY", "deliv_qty"),
                     ("DELIV_PER", "deliv_pct")]:
        out[dst] = pd.to_numeric(df[src].replace("-", np.nan), errors="coerce").to_numpy()
    u = universe if universe is not None else set(lab_names())
    out["bars_symbol"] = out["symbol"].map(lambda s: bars_symbol(s, u))
    return out.sort_values("symbol").reset_index(drop=True)


def parse_cm_mto(cm_zip: bytes, mto: Optional[bytes], day: int, universe: Optional[set] = None):
    """Old era (before sec_bhavdata_full): CM bhavcopy + MTO delivery file (nse_old scripts/fetch_old.py) -> the same
    delivery columns (oldera_fo._build_delivery: avg_price = turnover / volume, turnover_lacs = turnover / 1e5)."""
    import numpy as np
    import pandas as pd
    z = zipfile.ZipFile(io.BytesIO(cm_zip))
    txt = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for r in csv.DictReader(io.StringIO(txt)):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if r.get("SERIES") != "EQ":
            continue
        ts = r.get("TIMESTAMP", "")
        if ts:
            try:
                if ymd(datetime.strptime(ts, "%d-%b-%Y").date()) != int(day):
                    return None
            except ValueError:
                pass
        rows.append(r)
    if not rows:
        return None
    deliv = {}
    if mto:
        for line in mto.decode("latin-1").splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 7 and f[0] == "20" and f[3] == "EQ":
                deliv[f[2]] = (f[5], f[6])
    f = lambda x: pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    sym = [r["SYMBOL"] for r in rows]
    vol = f([r["TOTTRDQTY"] for r in rows])
    turn = f([r["TOTTRDVAL"] for r in rows])
    out = pd.DataFrame({"date": int(day), "symbol": sym, "open": f([r["OPEN"] for r in rows]),
                        "high": f([r["HIGH"] for r in rows]), "low": f([r["LOW"] for r in rows]),
                        "close": f([r["CLOSE"] for r in rows]), "prev_close": f([r["PREVCLOSE"] for r in rows]),
                        "avg_price": np.where(vol > 0, turn / np.where(vol > 0, vol, 1), np.nan), "volume": vol,
                        "turnover_lacs": turn / 1e5, "trades": f([r.get("TOTALTRADES", "") for r in rows]),
                        "deliv_qty": f([deliv.get(s, ("", ""))[0] for s in sym]),
                        "deliv_pct": f([deliv.get(s, ("", ""))[1] for s in sym])})
    u = universe if universe is not None else set(lab_names())
    out["bars_symbol"] = out["symbol"].map(lambda s: bars_symbol(s, u))
    return out.sort_values("symbol").reset_index(drop=True)


def _read_zip_csv(content: bytes):
    import pandas as pd
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        with z.open(names[0]) as fh:
            return pd.read_csv(fh, dtype=str, keep_default_na=False)


def norm_udiff(df):
    """fetch_fo.py norm_udiff."""
    import pandas as pd
    m = {"IDF": "FUTIDX", "STF": "FUTSTK", "IDO": "OPTIDX", "STO": "OPTSTK"}
    return pd.DataFrame({
        "date": df["TradDt"].str.strip(), "instrument": df["FinInstrmTp"].str.strip().map(m),
        "symbol": df["TckrSymb"].str.strip(),
        "expiry": pd.to_datetime(df["XpryDt"].str.strip(), format="%Y-%m-%d").dt.strftime("%Y-%m-%d"),
        "strike": _num(df["StrkPric"]), "type": df["OptnTp"].str.strip(),
        "open": _num(df["OpnPric"]), "high": _num(df["HghPric"]), "low": _num(df["LwPric"]),
        "close": _num(df["ClsPric"]), "settle": _num(df["SttlmPric"]),
        "contracts": _num(df["TtlTradgVol"]), "oi": _num(df["OpnIntrst"]),
        "chg_oi": _num(df["ChngInOpnIntrst"]), "underlying": _num(df["UndrlygPric"]),
        "lot": _num(df["NewBrdLotQty"])})


def norm_old(df, d: date):
    """fetch_fo.py norm_old (+ the content date of the TIMESTAMP column)."""
    import pandas as pd
    df = df.rename(columns=lambda c: c.strip())
    out = pd.DataFrame({
        "date": d.strftime("%Y-%m-%d"), "instrument": df["INSTRUMENT"].str.strip(), "symbol": df["SYMBOL"].str.strip(),
        "expiry": pd.to_datetime(df["EXPIRY_DT"].str.strip(), format="%d-%b-%Y").dt.strftime("%Y-%m-%d"),
        "strike": _num(df["STRIKE_PR"]), "type": df["OPTION_TYP"].str.strip(),
        "open": _num(df["OPEN"]), "high": _num(df["HIGH"]), "low": _num(df["LOW"]),
        "close": _num(df["CLOSE"]), "settle": _num(df["SETTLE_PR"]),
        "contracts": _num(df["CONTRACTS"]), "oi": _num(df["OPEN_INT"]),
        "chg_oi": _num(df["CHG_IN_OI"]), "underlying": float("nan"), "lot": float("nan")})
    out["content_date"] = pd.to_datetime(df["TIMESTAMP"].str.strip(), format="%d-%b-%Y").dt.strftime("%Y-%m-%d")
    return out


FUT_COLS = ["date", "symbol", "expiry", "instrument", "open", "high", "low", "close", "settle", "contracts", "oi",
            "chg_oi", "underlying", "lot"]
OPT_COLS = ["date", "symbol", "expiry", "strike", "type", "open", "high", "low", "close", "settle", "contracts", "oi",
            "chg_oi", "underlying", "lot"]


def fo_parts(n, d: date) -> dict:
    """fetch_fo.py write_parts -> {sub: DataFrame}; optstk_full only on the sessions 1-4 calendar days before the
    nearest stock-option expiry, nearest expiry only (the lab kept the two nearest expiries every session)."""
    import pandas as pd
    fut = n[n.instrument.isin(["FUTIDX", "FUTSTK"])][FUT_COLS]
    oi = n[(n.instrument == "OPTIDX") & n.symbol.isin(["NIFTY", "BANKNIFTY"])][OPT_COLS]
    st = n[n.instrument == "OPTSTK"].copy()
    rows = []
    for (sym, exp), g in st.groupby(["symbol", "expiry"], sort=True):
        c = g[g.type == "CE"]
        p = g[g.type == "PE"]
        rows.append({
            "date": d.strftime("%Y-%m-%d"), "symbol": sym, "expiry": exp,
            "call_oi": c.oi.sum(), "put_oi": p.oi.sum(), "call_vol": c.contracts.sum(), "put_vol": p.contracts.sum(),
            "chg_call_oi": c.chg_oi.sum(), "chg_put_oi": p.chg_oi.sum(),
            "max_call_oi_strike": c.loc[c.oi.idxmax(), "strike"] if len(c) and c.oi.max() > 0 else float("nan"),
            "max_put_oi_strike": p.loc[p.oi.idxmax(), "strike"] if len(p) and p.oi.max() > 0 else float("nan"),
            "n_strikes": g.strike.nunique(), "underlying": g.underlying.median()})
    agg = pd.DataFrame(rows, columns=["date", "symbol", "expiry", "call_oi", "put_oi", "call_vol", "put_vol",
                                      "chg_call_oi", "chg_put_oi", "max_call_oi_strike", "max_put_oi_strike",
                                      "n_strikes", "underlying"])
    out = {"fut": fut, "optidx": oi, "optstk": agg}
    if len(st):
        ex1 = sorted(st.expiry.unique())[0]
        gap = (date.fromisoformat(ex1) - d).days
        if 0 < gap <= 4:
            out["optstk_full"] = st[(st.expiry == ex1) & ((st.oi > 0) | (st.contracts > 0))][OPT_COLS]
    return out


def parse_fo(content: bytes, d: date, fmt: str):
    """-> (normalised frame, None) or (None, 'stale') when the file inside is dated another session."""
    raw = _read_zip_csv(content)
    n = norm_udiff(raw) if fmt == "udiff" else norm_old(raw, d)
    cd = n["content_date"] if "content_date" in n else n["date"]
    if cd.ne(d.isoformat()).sum():
        return None, "stale"
    return n.drop(columns=["content_date"], errors="ignore"), None


def parse_index_file(body: bytes, day: int) -> Optional[list]:
    """ind_close_all_DDMMYYYY.csv -> [(key, open, high, low, close)] for INDEX_NAMES; None if dated another day."""
    rows = list(csv.reader(io.StringIO(body.decode("latin-1"))))
    if not rows:
        return None
    hdr = [h.strip().lower() for h in rows[0]]
    try:
        i_name, i_date = hdr.index("index name"), hdr.index("index date")
        i_o, i_h, i_l, i_c = (hdr.index(x) for x in ("open index value", "high index value", "low index value",
                                                      "closing index value"))
    except ValueError:
        return None
    out = []
    for r in rows[1:]:
        if len(r) <= i_c:
            continue
        name = r[i_name].strip().lower()
        key = next((k for k, v in INDEX_NAMES.items() if name in v), None)
        if key is None:
            continue
        try:
            dd = datetime.strptime(r[i_date].strip(), "%d-%m-%Y").date()
        except ValueError:
            continue
        if ymd(dd) != int(day):
            return None

        def f(x):
            try:
                return float(x.replace(",", ""))
            except ValueError:
                return float("nan")
        out.append((key, f(r[i_o]), f(r[i_h]), f(r[i_l]), f(r[i_c])))
    return out


def parse_fii_xls(raw: bytes, day: int) -> tuple[Optional[list], str]:
    """databuild build_fii.py's parse of one fii_stats sheet -> ([row dicts], status)."""
    import pandas as pd
    try:
        x = pd.read_excel(io.BytesIO(raw), header=None)
    except ImportError:
        return None, "unparsed:xlrd missing"
    except Exception as exc:
        return None, f"error:parse {str(exc)[:80]}"
    date_iso = f"{str(day)[:4]}-{str(day)[4:6]}-{str(day)[6:]}"
    head = " ".join(str(v) for v in x.iloc[:3].values.ravel() if isinstance(v, str))
    m = re.search(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", head)
    mon = {mm: i + 1 for i, mm in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
    if m:
        cd = f"{m.group(3)}-{mon[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
        if cd != date_iso:
            return None, "stale"
    recs = []
    for _, r in x.iterrows():
        lab = r[0]
        if not isinstance(lab, str) or not re.search(r"(FUTURES|OPTIONS)\s*$", lab.strip().upper()):
            continue
        try:
            vals = [float(r[i]) for i in range(1, 7)]
        except Exception:
            continue
        recs.append({"date": date_iso, "instrument": re.sub(r"\s+", " ", lab.strip().upper()), "buy_contracts": vals[0],
                     "buy_amt_cr": vals[1], "sell_contracts": vals[2], "sell_amt_cr": vals[3],
                     "oi_contracts": vals[4], "oi_amt_cr": vals[5]})
    return (recs, "ok") if recs else (None, "error:no instrument rows")


_CA_PARSE: list = []


def lab_ca_parser() -> Callable:
    """The `parse` function of stratlab/databuild_scripts/build_corpactions.py, taken from the lab's source (the
    script runs its build at import, so its parser is compiled on its own; nothing is copied or changed)."""
    if _CA_PARSE:
        return _CA_PARSE[0]
    path = HERE / "stratlab" / "databuild_scripts" / "build_corpactions.py"
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in ("NUM", "RS") for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "parse")]
    ns: dict = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    _CA_PARSE.append(ns["parse"])
    return ns["parse"]


# ── the delivery file (append-only gzip members + index) ──────────────────────────────────────────────────────────
class DeliveryFile:
    def __init__(self, nse_dir: Path) -> None:
        self.path = Path(nse_dir) / "delivery.csv.gz"
        self.idx_path = Path(nse_dir) / "delivery.idx.json"

    def index(self) -> dict:
        return _jload(self.idx_path, {"committed": 0, "days": {}})

    def _ensure(self) -> dict:
        ix = self.index()
        if not self.path.exists() or ix.get("committed", 0) == 0:
            hdr = gzip.compress((",".join(DELIVERY_COLS) + "\n").encode(), mtime=0)
            _atomic(self.path, hdr)
            ix = {"committed": len(hdr), "header": len(hdr), "days": {}}
            _jsave(self.idx_path, ix)
        elif self.path.stat().st_size != ix["committed"]:
            # an append that crashed before its index was written: cut our own uncommitted tail
            with open(self.path, "r+b") as fh:
                fh.truncate(ix["committed"])
        return ix

    def has(self, day: int) -> bool:
        return str(int(day)) in self.index().get("days", {})

    def append(self, day: int, df) -> int:
        with _LOCK:
            ix = self._ensure()
            if str(int(day)) in ix["days"]:
                return 0
            body = df[DELIVERY_COLS].to_csv(index=False, header=False, float_format="%.10g", lineterminator="\n")
            blob = gzip.compress(body.encode(), mtime=0)
            off = ix["committed"]
            with open(self.path, "ab") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            ix["days"][str(int(day))] = [off, len(blob), int(len(df))]
            ix["committed"] = off + len(blob)
            _jsave(self.idx_path, ix)
            return len(blob)

    def member(self, day: int) -> Optional[bytes]:
        e = self.index().get("days", {}).get(str(int(day)))
        if not e:
            return None
        with open(self.path, "rb") as fh:
            fh.seek(e[0])
            return fh.read(e[1])

    def header(self) -> bytes:
        ix = self.index()
        with open(self.path, "rb") as fh:
            return fh.read(int(ix.get("header") or 0))

    def read_day(self, day: int):
        import pandas as pd
        b = self.member(day)
        if b is None:
            return None
        return pd.read_csv(io.BytesIO(gzip.decompress(b)), names=DELIVERY_COLS, dtype={"symbol": str,
                                                                                        "bars_symbol": str})

    def days(self) -> list[int]:
        return sorted(int(k) for k in self.index().get("days", {}))


# ── the archive ───────────────────────────────────────────────────────────────────────────────────────────────────
class Builder:
    def __init__(self, root=None) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.nse = self.root / "nse"
        self.view = self.root / "nse_view"
        self.flow = self.root / "flow"
        self.events = self.root / "events"
        self.minute = self.root / "minute"
        self.state_path = self.root / "state.json"
        self.status_path = self.root / "status.json"
        self.delivery = DeliveryFile(self.nse)
        self._used: Optional[int] = None
        self._writes = 0

    # ── state ──
    def state(self) -> dict:
        st = _jload(self.state_path, {})
        st.setdefault("days", {})
        st.setdefault("windows", {})
        st.setdefault("fo_status", {})
        st.setdefault("ca_first_seen", {})
        st.setdefault("datasets", {})
        return st

    def save(self, st: dict) -> None:
        _jsave(self.state_path, st)

    # ── disk cap ──
    def used(self, fresh: bool = False) -> int:
        if fresh or self._used is None or self._writes >= 200:
            self._used, self._writes = disk_bytes(self.root), 0
        return self._used

    def check_cap(self, extra: int = 0) -> None:
        u = self.used()
        if u + extra >= max_bytes():
            raise DiskCapReached(f"disk cap reached ({u / 1048576:.1f} MB >= {max_bytes() / 1048576:.0f} MB)")

    def wrote(self, n: int) -> int:
        if self._used is not None:
            self._used += int(n)
        self._writes += 1
        return n

    def write(self, path: Path, data: bytes) -> int:
        self.check_cap(len(data))
        return self.wrote(_atomic(path, data))

    def write_csv_gz(self, path: Path, df) -> int:
        return self.write(path, gzip.compress(df.to_csv(index=False).encode(), mtime=0))

    # ── finality ──
    @staticmethod
    def final(status: Optional[str], day: int, today: date) -> bool:
        if not status:
            return False
        if status in ("ok", "n/a"):
            return True
        if status in ("missing", "no_session", "stale"):
            return (today - to_date(day)).days > FINAL_AFTER_DAYS
        return False

    # ── per-session datasets ──
    def fetch_cash(self, http: Http, d: date, universe: set) -> str:
        k = ymd(d)
        if self.delivery.has(k):
            return "ok"
        if k >= SEC_FULL_FROM:
            status, body = http.get(f"{ARCHIVE}/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv")
            if status == "missing":
                return "no_session"
            if status != "ok":
                return f"error:{status}"
            if not body.lstrip().startswith(b"SYMBOL"):
                return "error:unexpected content"
            df = parse_sec_full(body, k, universe)
        else:
            status, cm = http.get(f"{ARCHIVE}/content/historical/EQUITIES/{d.year}/{MON[d.month - 1]}/"
                                  f"cm{d:%d}{MON[d.month - 1]}{d.year}bhav.csv.zip")
            if status == "missing":
                return "no_session"
            if status != "ok":
                return f"error:{status}"
            s2, mto = http.get(f"{ARCHIVE}/archives/equities/mto/MTO_{d:%d%m%Y}.DAT")
            if s2 not in ("ok", "missing"):
                return f"error:mto {s2}"
            try:
                df = parse_cm_mto(cm, mto if s2 == "ok" else None, k, universe)
            except zipfile.BadZipFile:
                return "no_session"
        if df is None:
            return "stale"
        self.check_cap(len(df) * 40)
        self.wrote(self.delivery.append(k, df))
        return "ok"

    def fetch_idx(self, http: Http, d: date) -> str:
        k = ymd(d)
        status, body = http.get(f"{ARCHIVE}/content/indices/ind_close_all_{d:%d%m%Y}.csv")
        if status != "ok":
            return status if status == "missing" else f"error:{status}"
        rows = parse_index_file(body, k)
        if rows is None:
            return "stale"
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["index", "open", "high", "low", "close"])
        w.writerows(rows)
        self.write(self.nse / "raw_idx" / f"{k}.csv", buf.getvalue().encode())
        return "ok"

    def fetch_fo(self, http: Http, d: date, st: dict) -> str:
        k = ymd(d)
        udiff = f"{ARCHIVE}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
        old = (f"{ARCHIVE}/content/historical/DERIVATIVES/{d.year}/{MON[d.month - 1]}/"
               f"fo{d.day:02d}{MON[d.month - 1]}{d.year}bhav.csv.zip")
        order = [("udiff", udiff), ("old", old)] if k >= UDIFF_FROM else [("old", old), ("udiff", udiff)]
        got = None
        for fmt, url in order:
            status, body = http.get(url)
            if status == "ok" and body[:2] == b"PK":
                got = (fmt, body)
                break
            if status not in ("missing",) and not (status == "ok"):
                return f"error:{status}"
        if got is None:
            st["fo_status"][str(k)] = f"{d.isoformat()},missing,,0,0,0,0"
            return "missing"
        fmt, body = got
        n, bad = parse_fo(body, d, fmt)
        if n is None:
            st["fo_status"][str(k)] = f"{d.isoformat()},stale,{fmt},0,0,0,0"
            return "stale"
        parts = fo_parts(n, d)
        for sub, df in parts.items():
            self.write_csv_gz(self.nse / "fo_parts" / sub / f"{k}.csv.gz", df)
        st["fo_status"][str(k)] = (f"{d.isoformat()},ok,{fmt},{len(n)},{len(parts['fut'])},{len(parts['optidx'])},"
                                   f"{len(parts['optstk'])}")
        return "ok"

    def fetch_raw_part(self, http: Http, d: date, kind: str) -> str:
        k = ymd(d)
        url = {"poi": f"{ARCHIVE}/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv",
               "pvol": f"{ARCHIVE}/content/nsccl/fao_participant_vol_{d:%d%m%Y}.csv",
               "ban": f"{ARCHIVE}/archives/fo/sec_ban/fo_secban_{d:%d%m%Y}.csv"}[kind]
        check = b"Securities" if kind == "ban" else b"Participant"
        status, body = http.get(url)
        if status != "ok":
            return status if status == "missing" else f"error:{status}"
        if check not in body[:300]:
            return "error:unexpected content"
        sub = {"poi": "oi", "pvol": "vol", "ban": "ban"}[kind]
        self.write(self.nse / "raw_part" / sub / f"{k}.csv", body)
        return "ok"

    def fetch_fii(self, http: Http, d: date) -> str:
        k = ymd(d)
        if k < FII_FROM:
            return "n/a"
        status, body = http.get(f"{ARCHIVE}/content/fo/fii_stats_{d:%d}-{d:%b}-{d:%Y}.xls")
        if status != "ok":
            return status if status == "missing" else f"error:{status}"
        if not body.startswith(b"\xd0\xcf\x11\xe0") and not body.startswith(b"PK"):
            return "error:not xls"
        recs, why = parse_fii_xls(body, k)
        if recs is None:
            if why.startswith("unparsed"):
                self.write(self.nse / "raw_fii" / f"{k}.xls.gz", gzip.compress(body, mtime=0))
            return why
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(recs[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(recs)
        self.write(self.nse / "raw_fii" / f"{k}.csv", buf.getvalue().encode())
        return "ok"

    def process_day(self, http: Http, d: date, st: dict, today: date, universe: set) -> bool:
        """All per-session archives of calendar day d; True when something was fetched."""
        k = ymd(d)
        ds = st["days"].setdefault(str(k), {})
        if all(self.final(ds.get(x), k, today) for x in DAY_SETS) or \
                (self.final(ds.get("cash"), k, today) and ds.get("cash") != "ok"):
            return False
        did = False
        if not self.final(ds.get("cash"), k, today):
            ds["cash"] = self.fetch_cash(http, d, universe)
            did = True
        if ds["cash"] != "ok":
            return did
        for name, fn in (("idx", lambda: self.fetch_idx(http, d)), ("fo", lambda: self.fetch_fo(http, d, st)),
                         ("poi", lambda: self.fetch_raw_part(http, d, "poi")),
                         ("pvol", lambda: self.fetch_raw_part(http, d, "pvol")),
                         ("ban", lambda: self.fetch_raw_part(http, d, "ban")), ("fii", lambda: self.fetch_fii(http, d))):
            if not self.final(ds.get(name), k, today):
                ds[name] = fn()
                did = True
        return did

    # ── windowed feeds (deals, corporate actions, index membership, filing feeds) ──
    @staticmethod
    def quarters(lo: date, hi: date) -> list[tuple[date, date, str]]:
        out = []
        y, q = lo.year, (lo.month - 1) // 3
        while date(y, 3 * q + 1, 1) <= hi:
            s = date(y, 3 * q + 1, 1)
            e = date(y + (q == 3), (3 * q + 3) % 12 + 1, 1) - timedelta(days=1)
            out.append((s, e, f"{y}Q{q + 1}"))
            y, q = (y + 1, 0) if q == 3 else (y, q + 1)
        return out

    @staticmethod
    def months(lo: date, hi: date) -> list[tuple[date, date, str]]:
        out = []
        y, m = lo.year, lo.month
        while date(y, m, 1) <= hi:
            s = date(y, m, 1)
            e = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
            out.append((s, e, f"{y}{m:02d}"))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out

    def _wdone(self, st: dict, key: str, end: date, today: date) -> bool:
        rec = st["windows"].get(key)
        return bool(rec) and rec.get("status") in ("ok", "missing") and int(rec.get("at", 0)) > ymd(end) + 3

    def fetch_refs(self, http: Http, st: dict, today: date) -> None:
        rec = st["windows"].get("ref:symbolchange") or {}
        if int(rec.get("at", 0)) >= ymd(today - timedelta(days=7)):
            return
        status, body = http.get(f"{ARCHIVE}/content/equities/symbolchange.csv")
        if status == "ok" and body.count(b"\n") > 100:
            self.write(self.flow / "raw" / "ref" / "symbolchange.csv", body)
            self.write(self.events / "raw" / "ref" / "symbolchange.csv", body)
            st["windows"]["ref:symbolchange"] = {"status": "ok", "at": ymd(today)}
        else:
            st["windows"]["ref:symbolchange"] = {"status": status, "at": 0}

    def fetch_deals(self, http: Http, st: dict, today: date, newest_only: bool) -> bool:
        did = False
        qs = self.quarters(NSE_START, today)
        for s, e, tag in (reversed(qs[-1:]) if newest_only else reversed(qs)):
            for kind, opt in (("bulk", "bulk_deals"), ("block", "block_deals")):
                key = f"deals:{kind}:{tag}"
                if self._wdone(st, key, e, today):
                    continue
                status, body = http.get(f"{SITE}/api/historicalOR/bulk-block-short-deals?optionType={opt}"
                                        f"&from={s:%d-%m-%Y}&to={min(e, today):%d-%m-%Y}&csv=true")
                did = True
                if status == "ok" and b"Date" in body[:40]:
                    self.write(self.flow / "raw" / "deals" / kind / f"{tag}.csv.gz", gzip.compress(body, mtime=0))
                    st["windows"][key] = {"status": "ok", "at": ymd(today)}
                else:
                    st["windows"][key] = {"status": status if status != "ok" else "error:not csv", "at": ymd(today)}
        return did

    def fetch_ca(self, http: Http, st: dict, today: date, newest_only: bool) -> bool:
        did = False
        qs = self.quarters(NSE_START, today + timedelta(days=92))
        sel = qs[-2:] if newest_only else qs
        for s, e, tag in reversed(sel):
            key = f"ca:{tag}"
            if self._wdone(st, key, e, today):
                continue
            status, body = http.get(f"{SITE}/api/corporates-corporateActions?index=equities"
                                    f"&from_date={s:%d-%m-%Y}&to_date={e:%d-%m-%Y}")
            did = True
            try:
                rows = json.loads(body) if status == "ok" else None
                assert isinstance(rows, list)
            except Exception:
                st["windows"][key] = {"status": status if status != "ok" else "error:not json", "at": ymd(today)}
                continue
            for r in rows:
                ex = r.get("exDate") or ""
                try:
                    exk = ymd(datetime.strptime(ex.strip(), "%d-%b-%Y").date())
                except ValueError:
                    continue
                fk = f"{(r.get('symbol') or '').strip().upper()}|{exk}"
                st["ca_first_seen"].setdefault(fk, ymd(today))
            self.write(self.flow / "raw" / "ca" / f"{tag}.json.gz", gzip.compress(body, mtime=0))
            st["windows"][key] = {"status": "ok", "at": ymd(today)}
        return did

    def fetch_index(self, http: Http, st: dict, today: date, newest_only: bool) -> bool:
        did = False
        # today's constituent lists (exact day of every future change)
        rec = st["windows"].get("index:daily") or {}
        if int(rec.get("at", 0)) < ymd(today):
            ok = True
            for ix, fn in (("NIFTY50", "ind_nifty50list.csv"), ("NIFTYNEXT50", "ind_niftynext50list.csv"),
                           ("NIFTY100", "ind_nifty100list.csv")):
                status, body = http.get(f"{ARCHIVE}/content/indices/{fn}")
                did = True
                if status == "ok" and b"Symbol" in body[:400]:
                    dd = self.flow / "raw" / "index" / "daily" / ix
                    prev = sorted(dd.glob("*.csv")) if dd.exists() else []
                    if not prev or prev[-1].read_bytes() != body:
                        self.write(dd / f"{ymd(today)}.csv", body)
                else:
                    ok = False
            st["windows"]["index:daily"] = {"status": "ok" if ok else "error", "at": ymd(today) if ok else 0}
        # the NIFTY 200 list (builder_universe.py: the minute store's added names); kept raw only (the membership
        # files stay NIFTY 50 / NEXT 50 / 100), its own window so a failure never touches the three lists above
        rec = st["windows"].get("index:daily200") or {}
        if int(rec.get("at", 0)) < ymd(today):
            status, body = http.get(f"{ARCHIVE}/content/indices/ind_nifty200list.csv")
            did = True
            ok = status == "ok" and b"Symbol" in body[:400]
            if ok:
                dd = self.flow / "raw" / "index" / "daily" / "NIFTY200"
                prev = sorted(dd.glob("*.csv")) if dd.exists() else []
                if not prev or prev[-1].read_bytes() != body:
                    self.write(dd / f"{ymd(today)}.csv", body)
            st["windows"]["index:daily200"] = {"status": "ok" if ok else str(status), "at": ymd(today) if ok else 0}
        # NSE Indices' month-end constituent + weight files (mcwb), niftyindices.com then nsearchives
        ms = self.months(NSE_START, today.replace(day=1) - timedelta(days=1))
        for s, e, tag in (reversed(ms[-1:]) if newest_only else reversed(ms)):
            key = f"mcwb:{tag}"
            if self._wdone(st, key, e, today) or (st["windows"].get(key) or {}).get("status") == "ok":
                continue
            name = f"mcwb_{MON[s.month - 1].lower()}{s.year % 100:02d}.zip"
            status = "missing"
            for base in ("https://www.niftyindices.com/Market_Capitalisation_Weightage_Beta_for_NIFTY_50_And_NIFTY_"
                         "Next_50/", f"{ARCHIVE}/content/indices/"):
                status, body = http.get(base + name, retries=1)
                did = True
                if status == "ok" and body[:2] == b"PK":
                    self.write(self.flow / "raw" / "index" / "mcwb" / f"{tag}.zip", body)
                    break
                status = status if status != "ok" else "error:not zip"
            st["windows"][key] = {"status": status, "at": ymd(today)}
        return did

    def fetch_events(self, http: Http, st: dict, today: date, newest_only: bool, client=None) -> bool:
        """Results / board-meeting feeds through pead_data's client (stratlab.events_fetch.Client: an ordinary page
        load first, as a browser does; a refusal is recorded, never worked around)."""
        did = False
        ms = self.months(EVENTS_FROM, today)
        for feed in FEEDS:
            sel = [m for m in ms if feed != "ifr" or m[1] >= date(2025, 4, 1)]
            for s, e, tag in (reversed(sel[-1:]) if newest_only else reversed(sel)):
                key = f"ev:{feed}:{tag}"
                if self._wdone(st, key, e, today):
                    continue
                if client is None:
                    import pead_data
                    client = pead_data._Client(PAUSE_S)
                guarded = guard_client(client, http)
                if not guarded:                 # a client without a requests session (tests): one request per fetch
                    http.before_request()
                try:
                    status, bodies = client.fetch(feed, s, min(e, today))
                except (BudgetReached, StopRun, NSERefused):
                    raise                       # a guard tripped mid-pagination: nothing saved, the window stays pending
                except Exception as exc:
                    status, bodies = f"error:{str(exc)[:60]}", []
                did = True
                if not guarded:
                    http.after_status(403 if status == "refused" else 200)
                if status != "ok":
                    st["windows"][key] = {"status": status, "at": ymd(today)}
                    continue
                if feed == "ifr":
                    for i, b in enumerate(bodies):
                        self.write(self.events / "raw" / feed / f"{tag}_p{i + 1:02d}.json.gz", gzip.compress(b, mtime=0))
                else:
                    self.write(self.events / "raw" / feed / f"{tag}.json.gz",
                               gzip.compress(bodies[0] if bodies else b"[]", mtime=0))
                st["windows"][key] = {"status": "ok", "at": ymd(today)}
        return did

    # ── derived files ──
    def rebuild_nse_small(self, st: dict) -> dict:
        """participant / ban (stratlab.oldera_fo's parsers), fii_stats, the index bars, fo_parts/status.csv."""
        import pandas as pd
        from stratlab import oldera_fo as OF
        rep = {}
        lines = ["date,status,fmt,n_raw,n_fut,n_optidx,n_optstk_agg"] + [st["fo_status"][k]
                                                                          for k in sorted(st["fo_status"])]
        self.write(self.nse / "fo_parts" / "status.csv", ("\n".join(lines) + "\n").encode())
        for sub in ("fut", "optidx", "optstk", "optstk_full"):
            (self.nse / "fo_parts" / sub).mkdir(parents=True, exist_ok=True)
        if (self.nse / "raw_part" / "oi").exists() and (self.nse / "raw_part" / "ban").exists():
            self.check_cap(2 << 20)
            r = OF.build_participants(str(self.nse))
            rep["participant_problems"] = len(r.get("problems", []))
        vd = self.nse / "raw_part" / "vol"
        if vd.exists():
            recs, problems = [], []
            for f in sorted(vd.glob("*.csv")):
                kk = f.name[:8]
                recs += OF._parse_participant(str(f), f"{kk[:4]}-{kk[4:6]}-{kk[6:]}", problems)
            df = pd.DataFrame(recs, columns=["date", "client_type"] + list(OF.COLMAP.values()))
            self.write(self.nse / "participant_vol.csv.gz", gzip.compress(df.to_csv(index=False).encode(), mtime=0))
        fd = self.nse / "raw_fii"
        if fd.exists() and any(fd.glob("*.csv")):
            df = pd.concat([pd.read_csv(f) for f in sorted(fd.glob("*.csv"))], ignore_index=True)
            df = df.sort_values(["date", "instrument"]).drop_duplicates(["date", "instrument"])
            self.write_csv_gz(self.nse / "fii_stats.csv.gz", df)
            link = self.flow / "fii_stats" / "fii_stats.csv.gz"
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.is_symlink():
                os.symlink(self.nse / "fii_stats.csv.gz", link)
        idd = self.nse / "raw_idx"
        if idd.exists():
            ser: dict = {}
            for f in sorted(idd.glob("*.csv")):
                kk = int(f.name[:8])
                for r in csv.DictReader(io.StringIO(f.read_text())):
                    ser.setdefault(r["index"], []).append((kk, r["open"], r["high"], r["low"], r["close"]))
            for key, fn, t in (("NIFTY", "_NIFTY50_day.json", "00:00:00"), ("INDIAVIX", "_VIX_day.json", "00:00:00"),
                               ("BANKNIFTY", "_NIFTYBANK_min.json", "15:29:00")):
                rows = sorted(ser.get(key, []))
                bars = [[f"{k // 10000:04d}-{k // 100 % 100:02d}-{k % 100:02d}T{t}", float(o), float(h), float(l),
                         float(c), 0] for k, o, h, l, c in rows]
                self.write(self.nse / "bars" / fn, json.dumps({"symbol": fn[:-5], "interval": "day" if "day" in fn
                                                               else "minute", "bars": bars}).encode())
        return rep

    def symmap(self) -> SymMap:
        from stratlab.events_parse import load_changes
        return SymMap(load_changes(str(self.flow / "raw" / "ref" / "symbolchange.csv")), lab_names())

    def rebuild_deals(self, sm: SymMap) -> int:
        """databuild build_deals.py's normalisation -> flow/deals/deals.csv.gz."""
        rows = []
        for kind in ("bulk", "block"):
            for fn in sorted((self.flow / "raw" / "deals" / kind).glob("*.csv.gz")):
                txt = gzip.decompress(fn.read_bytes()).decode("utf-8-sig")
                rd = csv.reader(io.StringIO(txt))
                hdr = [h.strip() for h in next(rd, [])]
                if hdr[:7] != ["Date", "Symbol", "Security Name", "Client Name", "Buy / Sell", "Quantity Traded",
                               "Trade Price / Wght. Avg. Price"]:
                    continue
                for r in rd:
                    if not r or not r[0].strip():
                        continue
                    try:
                        d = int(datetime.strptime(r[0].strip(), "%d-%b-%Y").strftime("%Y%m%d"))
                        sym, side = r[1].strip().upper(), r[4].strip().upper()
                        qty, px = int(r[5].replace(",", "").strip()), float(r[6].replace(",", "").strip())
                        assert side in ("BUY", "SELL") and qty > 0 and px > 0 and sym
                    except Exception:
                        continue
                    rem = r[7].strip() if len(r) > 7 else ""
                    lab = sm.lab_symbol(sym, d)
                    rows.append((d, kind, sym, lab, sm.in_lab(lab), re.sub(r"\s+", " ", r[2].strip()),
                                 re.sub(r"\s+", " ", r[3].strip().upper()), side, qty, px, round(qty * px / 1e7, 4),
                                 "" if rem in ("-", "") else rem))
        rows = sorted(set(rows), key=lambda x: (x[0], x[1], x[2], x[6], x[7], x[8], x[9]))
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["date", "kind", "symbol", "lab_symbol", "in_lab", "security", "client", "side", "qty", "price",
                    "value_cr", "remarks"])
        w.writerows(rows)
        self.write(self.flow / "deals" / "deals.csv.gz", gzip.compress(buf.getvalue().encode(), mtime=0))
        return len(rows)

    def rebuild_ca(self, sm: SymMap, st: dict) -> int:
        """databuild build_corpactions.py's rows (its own `parse`), known_date = the first day the server saw the
        action when that is before the ex-date, else the session before the ex-date (the lab's fallback; the lab's
        daily Bc files are not fetched)."""
        import bisect
        parse = lab_ca_parser()
        ses = self.delivery.days()

        def prev_session(day):
            i = bisect.bisect_left(ses, day)
            if i > 0 and (to_date(day) - to_date(ses[i - 1])).days < 10:
                return ses[i - 1]
            dd = to_date(day) - timedelta(days=1)
            while dd.weekday() >= 5:
                dd -= timedelta(days=1)
            return ymd(dd)

        raw = []
        for fn in sorted((self.flow / "raw" / "ca").glob("*.json.gz")):
            try:
                raw += json.loads(gzip.decompress(fn.read_bytes()))
            except Exception:
                continue
        seen, agg = set(), {}
        for r in raw:
            if r.get("series") not in ("EQ", "BE", "BZ"):
                continue
            try:
                ex = int(datetime.strptime((r.get("exDate") or "").strip(), "%d-%b-%Y").strftime("%Y%m%d"))
            except ValueError:
                continue
            sym = (r.get("symbol") or "").strip().upper()
            subj = re.sub(r"\s+", " ", r.get("subject") or "").strip()
            if (sym, ex, subj) in seen:
                continue
            seen.add((sym, ex, subj))
            comps = parse(subj)
            try:
                fv = float(r.get("faceVal") or "")
            except ValueError:
                fv = None
            try:
                rec = int(datetime.strptime((r.get("recDate") or "").strip(), "%d-%b-%Y").strftime("%Y%m%d"))
            except ValueError:
                rec = None
            for c in comps:
                k = (sym, ex, c["kind"])
                if k in agg:
                    a = agg[k]
                    if c["kind"] == "dividend" and c.get("dividend") is not None:
                        a["dividend"] = round((a.get("dividend") or 0) + c["dividend"], 4)
                    a["subject"] += " || " + subj
                    continue
                agg[k] = {**c, "ex_date": ex, "symbol": sym, "series": r["series"], "isin": r.get("isin") or "",
                          "face_value": fv, "record_date": rec, "subject": subj}
        closes: dict = {}
        rows = []
        for (sym, ex, kind), a in agg.items():
            f = None
            then = sm.symbol_on(sym, ex)
            if kind == "bonus" and a.get("ratio_a") and a.get("ratio_b"):
                f = a["ratio_b"] / (a["ratio_a"] + a["ratio_b"])
            elif kind in ("split", "consolidation") and a.get("fv_from") and a.get("fv_to"):
                f = a["fv_to"] / a["fv_from"]
            elif kind == "rights" and a.get("ratio_a") and a.get("ratio_b") and a.get("rights_premium") is not None:
                ps = prev_session(ex)
                if ps not in closes:
                    df = self.delivery.read_day(ps)
                    closes[ps] = dict(zip(df["symbol"], df["close"])) if df is not None else {}
                p = closes[ps].get(then)
                if p and p > 0:
                    issue = a["rights_premium"] + (a["face_value"] or 0)
                    f = min(1.0, (a["ratio_b"] * p + a["ratio_a"] * issue) / ((a["ratio_a"] + a["ratio_b"]) * p))
            fs = st["ca_first_seen"].get(f"{sym}|{ex}")
            fl = fs if (fs is not None and fs < ex) else None
            known = fl if fl is not None else prev_session(ex)
            lab = sm.lab_symbol(sym, ex)
            rows.append([ex, sym, then, lab, sm.in_lab(lab), a["series"], a["isin"],
                         "" if a["face_value"] is None else a["face_value"], kind,
                         a.get("ratio_a", ""), a.get("ratio_b", ""), a.get("fv_from", ""), a.get("fv_to", ""),
                         a.get("dividend", "") if a.get("dividend") is not None else "",
                         a.get("rights_premium", "") if a.get("rights_premium") is not None else "",
                         "" if f is None else round(f, 6), a["record_date"] or "", fl or "", known, a["subject"]])
        rows.sort(key=lambda x: (x[0], x[1], x[8]))
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["ex_date", "symbol", "symbol_then", "lab_symbol", "in_lab", "series", "isin", "face_value", "kind",
                    "ratio_a", "ratio_b", "fv_from", "fv_to", "dividend", "rights_premium", "factor", "record_date",
                    "first_listed", "known_date", "subject"])
        w.writerows(rows)
        self.write(self.flow / "corpactions" / "corpactions.csv.gz", gzip.compress(buf.getvalue().encode(), mtime=0))
        return len(rows)

    def rebuild_index(self, sm: SymMap, today: date) -> dict:
        """Point-in-time NIFTY 50 / NEXT 50 / 100 membership from the month-end mcwb files (build_index.py's parse;
        precision "month": a change is dated at the first file that shows it, never earlier than known) and the
        daily constituent lists the server saves from its first run on (precision "day"). Press releases (exact
        announcement days) are not fetched: announce_source = "none" (announce_date = the session before)."""
        import bisect
        ses = self.delivery.days()
        snaps: list = []           # (date, index, {latest: (symbol, name)}, precision, source)
        weights = []
        for fn in sorted((self.flow / "raw" / "index" / "mcwb").glob("*.zip")):
            ym = int(fn.name[:6])
            y, mth = divmod(ym, 100)
            last_day = ymd(date(y + (mth == 12), mth % 12 + 1, 1) - timedelta(days=1))
            i = bisect.bisect_right(ses, last_day)
            sdate = ses[i - 1] if i > 0 and ses[i - 1] // 100 == ym else last_day
            try:
                z = zipfile.ZipFile(fn)
            except zipfile.BadZipFile:
                continue
            got = {}
            for n in z.namelist():
                if not n.lower().endswith(".csv"):
                    continue
                t = z.read(n).decode("latin-1").splitlines()
                title = (t[0] if t else "").lower()
                ix = "NIFTYNEXT50" if re.search(r"jr|junior|next\s*50", title + " " + n.lower()) else "NIFTY50"
                hi = next((k for k, l in enumerate(t[:8]) if "symbol" in l.lower()), None)
                if hi is None:
                    continue
                rd = csv.reader(io.StringIO("\n".join(t[hi:])))
                hdr = [x.strip().lower() for x in next(rd)]
                ks = next(k for k, x in enumerate(hdr) if "symbol" in x)
                kw = next((k for k, x in enumerate(hdr) if "weight" in x), None)
                km = next((k for k, x in enumerate(hdr) if "free float" in x or "market cap" in x), None)
                kn = next((k for k, x in enumerate(hdr) if "name" in x), None)
                d = {}
                for r in rd:
                    if len(r) <= ks or not r[0].strip().isdigit() or not r[ks].strip():
                        continue
                    sym = r[ks].strip().upper()

                    def num(k):
                        try:
                            return float(r[k].replace(",", "")) if k is not None and len(r) > k else ""
                        except ValueError:
                            return ""
                    d[sm.latest_symbol(sym, sdate)] = (sym, r[kn].strip() if kn is not None and len(r) > kn else "")
                    lab = sm.lab_symbol(sym, sdate)
                    weights.append([ym, sdate, ix, sym, lab, sm.in_lab(lab), num(kw), num(km)])
                got[ix] = d
            if "NIFTY50" in got and "NIFTYNEXT50" in got:
                got["NIFTY100"] = {**got["NIFTY50"], **got["NIFTYNEXT50"]}
                for ix, d in got.items():
                    snaps.append((sdate, ix, d, "month", "mcwb"))
        for ix in ("NIFTY50", "NIFTYNEXT50", "NIFTY100"):
            for fn in sorted((self.flow / "raw" / "index" / "daily" / ix).glob("*.csv")):
                dk = int(fn.name[:8])
                d = {}
                for r in csv.DictReader(io.StringIO(fn.read_bytes().decode("utf-8-sig", errors="replace"))):
                    r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
                    if r.get("symbol"):
                        d[sm.latest_symbol(r["symbol"].upper(), dk)] = (r["symbol"].upper(), r.get("company name", ""))
                if d:
                    snaps.append((dk, ix, d, "day", "list"))
        members, changes, check = [], [], {"built_through": ymd(today), "sources": {
            "monthly_constituent_files": "niftyindices.com mcwb_<mon><yy>.zip (nsearchives fallback)",
            "daily_lists": "nsearchives ind_nifty50list / ind_niftynext50list / ind_nifty100list (from the first run)"}}
        for ix in ("NIFTY50", "NIFTYNEXT50", "NIFTY100"):
            ss = sorted((s for s in snaps if s[1] == ix), key=lambda s: s[0])
            if not ss:
                continue
            check[ix] = {"reliable_from": ss[0][0], "snapshots": len(ss)}
            check.setdefault("origin", ss[0][0])
            open_: dict = {}
            prev = None
            for sdate, _, d, prec, src in ss:
                if prev is not None:
                    i = bisect.bisect_left(ses, sdate)
                    ann = ses[i - 1] if i > 0 else ymd(to_date(sdate) - timedelta(days=1))
                    for lat in sorted(set(d) - set(prev[2])):
                        sym, name = d[lat]
                        lab = sm.lab_symbol(sym, sdate)
                        changes.append([ix, ann, sdate, "IN", sym, lab, sm.in_lab(lab), name, src, "none", prec, ""])
                    for lat in sorted(set(prev[2]) - set(d)):
                        sym, name = prev[2][lat]
                        lab = sm.lab_symbol(sym, sdate)
                        changes.append([ix, ann, sdate, "OUT", sym, lab, sm.in_lab(lab), name, src, "none", prec, ""])
                for lat in set(d) - set(open_):
                    open_[lat] = (sdate, prec)
                for lat in [x for x in open_ if x not in d]:
                    f, fp = open_.pop(lat)
                    lab = sm.lab_symbol(lat, sdate)
                    members.append([ix, lat, lab, sm.in_lab(lab), f, ymd(to_date(sdate) - timedelta(days=1)), fp, prec])
                prev = (sdate, ix, d)
            for lat, (f, fp) in open_.items():
                lab = sm.lab_symbol(lat, ymd(today))
                members.append([ix, lat, lab, sm.in_lab(lab), f, "", fp, ""])
        out = self.flow / "index_membership"

        def gz(rows, hdr):
            b = io.StringIO()
            w = csv.writer(b, lineterminator="\n")
            w.writerow(hdr)
            w.writerows(rows)
            return b.getvalue().encode()
        self.write(out / "index_changes.csv", gz(sorted(changes), ["index", "announce_date", "effective_date",
                                                                   "action", "symbol", "lab_symbol", "in_lab",
                                                                   "company", "source", "announce_source",
                                                                   "effective_precision", "note"]))
        self.write(out / "index_members.csv.gz", gzip.compress(gz(sorted(members, key=lambda r: (r[0], r[1], r[4])), [
            "index", "symbol", "lab_symbol", "in_lab", "from_date", "to_date", "from_precision", "to_precision"]),
            mtime=0))
        self.write(out / "index_weights.csv.gz", gzip.compress(gz(weights, [
            "month", "snapshot_date", "index", "symbol", "lab_symbol", "in_lab", "weight_pct", "ff_mcap_cr"]), mtime=0))
        self.write(out / "data_check.json", json.dumps(check, indent=1).encode())
        return {"members": len(members), "changes": len(changes), "snapshots": len(snaps)}

    def rebuild_events(self) -> dict:
        """stratlab.events_parse (unchanged) -> events/boardmeet.csv.gz, events/results.csv.gz."""
        from stratlab import events_parse as EP
        raw = str(self.events / "raw")
        changes = EP.load_changes(os.path.join(raw, "ref", "symbolchange.csv"))
        bm, _ = EP.build_boardmeet(EP.load_feed(raw, "bm"), changes)
        self.check_cap(4 << 20)
        self.events.mkdir(parents=True, exist_ok=True)
        EP.write_csv(str(self.events / "boardmeet.csv.gz"), EP.BM_COLS, bm)
        meetings: dict = {}
        for r in bm:
            if r["is_results"]:
                meetings.setdefault(r["symbol"], set()).add(date.fromisoformat(r["meeting_date"]))
        res, _ = EP.build_results(EP.load_feed(raw, "fr"), EP.load_feed(raw, "ann_fr"), EP.load_feed(raw, "ann_obm"),
                                  changes, EP.load_feed(raw, "ifr"), meetings)
        EP.write_csv(str(self.events / "results.csv.gz"), EP.RES_COLS, res)
        return {"boardmeet": len(bm), "results": len(res)}

    def build_view(self, sessions: int = VIEW_SESSIONS) -> dict:
        """nse_view = the last `sessions` sessions of nse/ in nsedata's layout (symlinks; a short delivery file)."""
        days = self.delivery.days()
        if not days:
            return {"sessions": 0}
        keep = days[-int(sessions):]
        lo = keep[0]
        size = sum(self.delivery.index()["days"][str(k)][1] for k in keep)
        self.check_cap(size)
        tmp = self.view.with_name(self.view.name + ".new")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        with open(tmp / "delivery.csv.gz", "wb") as out:
            out.write(self.delivery.header())
            for k in keep:
                out.write(self.delivery.member(k))
        self.wrote(size)
        fo = self.nse / "fo_parts"
        (tmp / "fo_parts").mkdir()
        for sub in ("fut", "optidx", "optstk", "optstk_full"):
            (tmp / "fo_parts" / sub).mkdir()
            for f in sorted((fo / sub).glob("*.csv.gz")) if (fo / sub).exists() else []:
                if int(f.name[:8]) >= lo:
                    os.symlink(f.resolve(), tmp / "fo_parts" / sub / f.name)
        if (fo / "status.csv").exists():
            lines = (fo / "status.csv").read_text().splitlines()
            keepl = [lines[0]] + [l for l in lines[1:] if int(l[:10].replace("-", "")) >= lo]
            (tmp / "fo_parts" / "status.csv").write_text("\n".join(keepl) + "\n")
        for name in ("participant_oi.csv.gz", "participant_vol.csv.gz", "ban.csv.gz", "fii_stats.csv.gz", "raw_part",
                     "bars"):
            if (self.nse / name).exists():
                os.symlink((self.nse / name).resolve(), tmp / name)
        old = self.view.with_name(self.view.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        if self.view.exists():
            os.replace(self.view, old)
        os.replace(tmp, self.view)
        if old.exists():
            shutil.rmtree(old)
        return {"sessions": len(keep), "from": lo, "to": keep[-1]}

    # ── one run ──
    def run(self, now: datetime, http: Optional[Http] = None, feed_client=None, budget: Optional[int] = None,
            clock: Optional[Callable[[], datetime]] = None, backfill: bool = True, log: Callable = print) -> dict:
        clock = clock or (lambda: now)
        today = now.date()
        latest = today if now.time() >= dtime(19, 0) else today - timedelta(days=1)
        budget = int(budget if budget is not None else getattr(_settings(), "builder_nse_max_requests_per_run", 2500))
        out = {"started": now.isoformat(timespec="seconds"), "requests": 0, "days": 0, "stopped": None,
               "errors": []}
        with _LOCK:
            self.root.mkdir(parents=True, exist_ok=True)
            st = self.state()
            ru = st.get("refused_until")
            if ru and now.isoformat() < ru:
                out["stopped"] = f"refused: waiting until {ru}"
                st["last_run"] = out
                self.save(st)
                return out
            if http is None:
                http = Http(budget, stop=lambda: "PEAD window" if in_pead_window(clock()) else None)
            universe = set(lab_names())
            self.used(fresh=True)
            changed = {"nse": False, "deals": False, "ca": False, "index": False, "events": False}
            try:
                self.check_cap()
                self.fetch_refs(http, st, today)
                # 1. the recent sessions first (the forward lab's data), newest first
                d = latest
                for _ in range(10):
                    if self.process_day(http, d, st, today, universe):
                        changed["nse"] = True
                        out["days"] += 1
                    d -= timedelta(days=1)
                self.save(st)
                # 2. the windowed feeds: newest windows, then their backfill
                changed["deals"] |= self.fetch_deals(http, st, today, newest_only=not backfill)
                changed["ca"] |= self.fetch_ca(http, st, today, newest_only=not backfill)
                changed["index"] |= self.fetch_index(http, st, today, newest_only=not backfill)
                changed["events"] |= self.fetch_events(http, st, today, newest_only=not backfill,
                                                       client=feed_client)
                self.save(st)
                # 3. the archive backfill, newest to oldest
                if backfill:
                    d = latest - timedelta(days=10)
                    n = 0
                    while d >= NSE_START:
                        if self.process_day(http, d, st, today, universe):
                            changed["nse"] = True
                            out["days"] += 1
                            n += 1
                            if n % 25 == 0:
                                self.save(st)
                        d -= timedelta(days=1)
            except DiskCapReached as exc:
                out["stopped"] = f"disk_cap: {exc}"
            except BudgetReached as exc:
                out["stopped"] = f"budget: {exc}"
            except StopRun as exc:
                out["stopped"] = f"stopped: {exc}"
            except NSERefused as exc:
                out["stopped"] = f"refused: {exc}"
                st["refused_until"] = (now + timedelta(hours=REFUSAL_BACKOFF_H)).isoformat(timespec="seconds")
                st["refused_at"] = now.isoformat(timespec="seconds")
            except Exception as exc:                        # never raise into the scheduler
                import traceback
                out["stopped"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
                out["errors"].append(traceback.format_exc()[-800:])
            self.save(st)
            out["requests"] = http.n
            out["http"] = dict(http.statuses)
            # 4. derived files (skipped at the disk cap: nothing more is written)
            if not str(out["stopped"] or "").startswith("disk_cap"):
                try:
                    if changed["nse"] or not (self.nse / "fo_parts" / "status.csv").exists():
                        if any(self.nse.glob("*")):
                            out["nse_small"] = self.rebuild_nse_small(st)
                            out["view"] = self.build_view()
                    if any(changed[k] for k in ("deals", "ca", "index")):
                        sm = self.symmap()
                        if changed["deals"]:
                            out["deals_rows"] = self.rebuild_deals(sm)
                        if changed["ca"]:
                            out["ca_rows"] = self.rebuild_ca(sm, st)
                        if changed["index"]:
                            out["index"] = self.rebuild_index(sm, today)
                    if changed["events"]:
                        out["events"] = self.rebuild_events()
                except DiskCapReached as exc:
                    out["stopped"] = f"disk_cap: {exc}"
                except Exception as exc:
                    out["errors"].append(f"derived files: {type(exc).__name__}: {str(exc)[:200]}")
            out["finished"] = clock().isoformat(timespec="seconds")
            st["last_run"] = out
            st.setdefault("runs", [])
            st["runs"] = st["runs"][-30:] + [{k: v for k, v in out.items() if k != "errors"}]
            if out["errors"] or (out["stopped"] or "").startswith(("refused", "error", "disk_cap")):
                st["last_error"] = {"at": out["finished"], "error": out["stopped"] if not out["errors"] else
                                    out["errors"][-1][-300:]}
            self.save(st)
        log(f"[builder-nse] {out['requests']} requests, {out['days']} days, stopped: {out['stopped']}")
        return out

    # ── status ──
    def status(self, now: Optional[datetime] = None) -> dict:
        import builder_minute as BM
        now = now or datetime.now()
        today = now.date()
        latest = today if now.time() >= dtime(19, 0) else today - timedelta(days=1)
        st = self.state()
        days = st["days"]
        cal = (latest - NSE_START).days + 1
        sessions = sorted(int(k) for k, v in days.items() if v.get("cash") == "ok")
        cash_final = sum(1 for k, v in days.items() if self.final(v.get("cash"), int(k), today))
        cash_prog = min(1.0, cash_final / cal) if cal > 0 else 0.0

        def mb(*paths):
            return round(sum(disk_bytes(p) if Path(p).is_dir() else
                             (max(os.lstat(p).st_size, os.lstat(p).st_blocks * 512) if os.path.lexists(p) and
                              not os.path.islink(p) else 0) for p in paths) / 1048576, 2)

        def per_day(name, paths, lo=None):
            ok = sorted(int(k) for k, v in days.items() if v.get(name) == "ok")
            fin = sum(1 for k in sessions if self.final(days[str(k)].get(name), k, today))
            base = [k for k in sessions if lo is None or k >= lo]
            return {"from": ok[0] if ok else None, "to": ok[-1] if ok else None, "rows": None, "files": len(ok),
                    "mb": mb(*paths), "progress_pct": round(100.0 * cash_prog * (fin / len(sessions) if sessions
                                                                                     else 0.0), 2),
                    "sessions_expected": len(base)}

        ds = self.delivery.index().get("days", {})
        dd = sorted(int(k) for k in ds)
        out = {
            "delivery": {"from": dd[0] if dd else None, "to": dd[-1] if dd else None,
                         "rows": int(sum(v[2] for v in ds.values())), "files": len(dd),
                         "mb": mb(self.nse / "delivery.csv.gz", self.nse / "delivery.idx.json"),
                         "progress_pct": round(100.0 * cash_prog, 2),
                         "source": "nsearchives sec_bhavdata_full (2020 ->), CM bhavcopy + MTO (2012-2019)"},
            "fo_parts": {**per_day("fo", [self.nse / "fo_parts"]),
                         "source": "nsearchives F&O bhavcopy (UDiFF / old format): fut, optidx, optstk aggregates; "
                                   "optstk_full only near-expiry eves"},
            "participant_oi": {**per_day("poi", [self.nse / "raw_part" / "oi", self.nse / "participant_oi.csv.gz"]),
                               "source": "nsearchives fao_participant_oi"},
            "participant_vol": {**per_day("pvol", [self.nse / "raw_part" / "vol",
                                                   self.nse / "participant_vol.csv.gz"]),
                                "source": "nsearchives fao_participant_vol"},
            "ban": {**per_day("ban", [self.nse / "raw_part" / "ban", self.nse / "ban.csv.gz"]),
                    "source": "nsearchives fo_secban"},
            "fii_stats": {**per_day("fii", [self.nse / "raw_fii", self.nse / "fii_stats.csv.gz"], FII_FROM),
                          "source": "nsearchives fii_stats_DD-Mon-YYYY.xls (from 2014-12)"},
            "index_spot": {**per_day("idx", [self.nse / "raw_idx", self.nse / "bars"]),
                           "source": "nsearchives ind_close_all (NIFTY 50, NIFTY BANK, INDIA VIX)"},
        }
        for name, prefix, path, src in (
                ("deals", "deals:", self.flow / "deals", "www.nseindia.com bulk / block deals (quarterly CSV)"),
                ("corpactions", "ca:", self.flow / "corpactions", "www.nseindia.com corporate actions (quarterly)"),
                ("index_membership", "mcwb:", self.flow / "index_membership",
                 "niftyindices.com month-end files + daily NSE constituent lists"),
                ("events", "ev:", self.events, "www.nseindia.com filing feeds (results, board meetings)")):
            ws = {k: v for k, v in st["windows"].items() if k.startswith(prefix)}
            okw = sorted(k for k, v in ws.items() if v.get("status") == "ok")
            total = {"deals:": 2 * len(self.quarters(NSE_START, latest)),
                     "ca:": len(self.quarters(NSE_START, latest + timedelta(days=92))),
                     "mcwb:": len(self.months(NSE_START, today.replace(day=1) - timedelta(days=1))),
                     "ev:": sum(len([m for m in self.months(EVENTS_FROM, latest)
                                     if f != "ifr" or m[1] >= date(2025, 4, 1)]) for f in FEEDS)}[prefix]
            fin = sum(1 for v in ws.values() if v.get("status") in ("ok", "missing"))
            errs = [f"{k}: {v.get('status')}" for k, v in ws.items() if v.get("status") not in ("ok", "missing")]
            tags = [k.split(":")[-1] for k in okw]
            out[name] = {"from": min(tags) if tags else None, "to": max(tags) if tags else None, "rows": None,
                         "files": len(okw), "mb": mb(path, self.flow / "raw" / prefix.split(":")[0]),
                         "progress_pct": round(100.0 * min(1.0, fin / total), 2) if total else 0.0, "source": src,
                         "last_error": errs[-1] if errs else None}
        out["events"]["mb"] = mb(self.events)
        out["index_membership"]["mb"] = mb(self.flow / "index_membership", self.flow / "raw" / "index")
        # minute bars
        import builder_universe as BU
        mbf = BM.MinuteBackfill(self.minute)
        mst = mbf.state()
        msyms = BU.minute_symbols(BU.mode(), self.root)
        prog = mbf.progress(mst, today, symbols=msyms, now=now)
        mfiles = sum(1 for _ in (self.minute / "bars").rglob("*.npz")) if (self.minute / "bars").exists() else 0
        out["minute"] = {"from": prog["from"], "to": prog["to"], "rows": None, "files": mfiles,
                         "mb": mb(self.minute), "progress_pct": prog["progress_pct"],
                         "windows_final": prog["windows_final"], "windows_total": prog["windows_total"],
                         "source": f"Kite historical_data 1-minute ({len(msyms) - 2} stocks, BUILDER_MINUTE_UNIVERSE="
                                   f"{BU.mode()}: the 97 + the added NIFTY 200 names, + NIFTY 50 + NIFTY BANK), daily "
                                   "NIFTY 50 / INDIA VIX", "missing_symbols": mst.get("missing", []),
                         "last_update": (mst.get("last_run") or {}).get("finished"),
                         "last_error": (mst.get("last_error") or {}).get("error")}
        try:
            uni = BU.minute_status(self.root, now)
            uni["projection"] = BU.projection(self.minute, uni["added_names"], today,
                                              cap_mb=round(max_bytes() / 1048576, 1),
                                              used_mb=disk_bytes(self.root) / 1048576)
            uni["kite_calls_to_backfill"] = sum(len(mbf.pending(mst, n, today, now)) for n in uni["added_names"]
                                                if n not in set(uni["missing_on_kite"]))
            out["minute"]["universe"] = uni
        except Exception as exc:
            out["minute"]["universe"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        out["nse_view"] = {"from": None, "to": None, "rows": None, "files": len(list(self.view.rglob("*")))
                           if self.view.exists() else 0, "mb": mb(self.view), "progress_pct": None,
                           "source": f"last {VIEW_SESSIONS} sessions of nse/ (forward-lab engine + nsedata cache)"}
        if (st.get("last_run") or {}).get("view"):
            out["nse_view"].update({k: st["last_run"]["view"].get(k) for k in ("from", "to")})
        lr = st.get("last_run") or {}
        for k, v in out.items():
            if k == "minute":
                continue
            v.setdefault("last_update", lr.get("finished"))
            v.setdefault("last_error", (st.get("last_error") or {}).get("error"))
        used = disk_bytes(self.root)
        return {"datasets": out, "disk": {"used_mb": round(used / 1048576, 2),
                                          "cap_mb": round(max_bytes() / 1048576, 2),
                                          "cap_reached": used >= max_bytes()},
                "refused_until": st.get("refused_until"), "refused_at": st.get("refused_at"),
                "last_run": {k: v for k, v in lr.items() if k != "errors"} or None,
                "minute_last_run": mst.get("last_run")}


# ── jobs / API ────────────────────────────────────────────────────────────────────────────────────────────────────
def engine_env(root=None) -> dict:
    """Environment for the forward lab's engine process: the server NSE view when it exists (the lab code reads
    STRATLAB_NSE_DIR / STRATLAB_BARS_DIR at import). An operator-set STRATLAB_NSE_DIR wins."""
    if os.getenv("STRATLAB_NSE_DIR"):
        return {}
    view = Builder(root).view
    if (view / "delivery.csv.gz").exists():
        return {"STRATLAB_NSE_DIR": str(view), "STRATLAB_BARS_DIR": str(view / "bars")}
    return {}


def nse_job(now: Optional[datetime] = None, **kw) -> dict:
    """Scheduler entry point (NSE archives). Off (flag false or LIVE) -> returns at once: nothing read or written."""
    if not enabled():
        return {"skipped": "BUILDER_DATA_ENABLED is off or not PAPER"}
    import ist_clock
    now = now or ist_clock.now_ist().replace(tzinfo=None)
    if in_pead_window(now):
        return {"skipped": "PEAD fetch window (19:20-23:00, 08:10-08:30 IST)"}
    clock = kw.pop("clock", None) or (lambda: ist_clock.now_ist().replace(tzinfo=None))
    return Builder(kw.pop("root", None)).run(now, clock=clock, **kw)


def minute_job(now: Optional[datetime] = None, root=None, **kw) -> dict:
    """Scheduler entry point (Kite minute backfill). Off -> returns at once: no Kite call."""
    if not enabled():
        return {"skipped": "BUILDER_DATA_ENABLED is off or not PAPER"}
    import ist_clock
    import builder_minute as BM
    now = now or ist_clock.now_ist().replace(tzinfo=None)
    if in_market_hours(now):
        return {"skipped": "market hours (09:00-15:45 IST on a trading day)"}
    b = Builder(root)
    try:
        import option_bar_archiver as OBA
        lock = OBA._LOCK
    except Exception:
        lock = None

    def cap():
        if disk_bytes(b.root) >= max_bytes():
            raise DiskCapReached(f"disk cap reached ({max_bytes() / 1048576:.0f} MB)")
    kw.setdefault("max_calls", int(getattr(_settings(), "builder_kite_max_calls_per_run", 1500)))
    kw.setdefault("clock", lambda: ist_clock.now_ist().replace(tzinfo=None))
    kw.setdefault("market_hours", in_market_hours)
    kw.setdefault("lock", lock)
    prepare_minute_universe(b, kw)
    return BM.MinuteBackfill(b.minute).run(now, cap_check=cap, **kw)


def prepare_minute_universe(b: "Builder", kw: dict, token_fn: Optional[Callable] = None) -> dict:
    """BUILDER_MINUTE_UNIVERSE: the instrument list of this run (the forward lab's 99 first, unchanged, then the added
    NIFTY 200 names, builder_universe.py), their Kite tokens (names Kite does not list are recorded as missing) and
    <minute>/universe.json. Sets kw["symbols"] / kw["tokens"] unless the caller gave them."""
    import builder_minute as BM
    import builder_universe as BU
    m = BU.mode()
    syms = kw.get("symbols") or BU.minute_symbols(m, b.root)
    if kw.get("tokens") is None:
        tok = dict(BM.kite_tokens())
        extra = [s for s in syms if s not in tok and not s.startswith("_")]
        if extra:
            tok.update((token_fn or BU.kite_tokens)(extra))
        kw["tokens"] = tok
    kw["symbols"] = list(syms)
    missing = [s for s in syms if not kw["tokens"].get(s)]
    return BU.write_universe_file(b.minute, missing=missing, root=b.root, m=m, symbols=kw["symbols"])


_API_CACHE: dict = {}


def api_status(root=None, max_age_s: float = 60.0) -> dict:
    """The /api/v2/builder/data payload (walks the builder directory: cached for a minute)."""
    key = str(Builder(root).root)
    hit = _API_CACHE.get(key)
    if hit and _time.time() - hit[0] < max_age_s:
        return hit[1]
    out = _api_status(root)
    _API_CACHE[key] = (_time.time(), out)
    return out


def _api_status(root=None) -> dict:
    try:
        import ist_clock
        now = ist_clock.now_ist().replace(tzinfo=None)
    except Exception:
        now = datetime.now()
    out = {"enabled": enabled(), "root": str(Builder(root).root),
           "schedule": {"nse": "23:05 and 05:30 IST daily, 10:05 / 15:05 Sat-Sun",
                        "minute": "18:40 and 06:10 IST daily, 11:40 / 14:40 Sat-Sun"},
           "budgets": {"kite_calls_per_run": int(getattr(_settings(), "builder_kite_max_calls_per_run", 1500)),
                       "nse_requests_per_run": int(getattr(_settings(), "builder_nse_max_requests_per_run", 2500))}}
    try:
        out.update(Builder(root).status(now))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out
