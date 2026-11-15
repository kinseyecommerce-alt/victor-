"""
stratlab/datagen_features.py — point-in-time feature panels for the DATA-DRIVEN DISCOVERY study (datagen_run.py,
prereg/datagen_v1.json). Research only; caches go to <scratchpad>/datagen/cache (outside the repository).

Two grids:
  DAILY   one row per stock of the events_daily cache (point-in-time top-500 universe, 2012-01-02 .. 2026-11-09, official
          open / high / low / volume, close = last traded price, corporate actions applied, delisted names kept) x one
          column per session of its calendar. A cell (s, t) is a DECISION taken in the evening of session t: it may
          read every end-of-day file dated <= t (NSE publishes delivery, F&O bhavcopy, participant OI and deals in the
          evening of their date) and the ban file dated t+1 (published the evening before its trade date); the trade
          is entered at the open of session t+1 at the earliest.
  MINUTE  one row per name of the lab's minute data (today's NIFTY-100 names + NIFTY 50 / NIFTY BANK; survivorship-
          biased) x one column per session of the joint minute calendar (2016-01-01 .. 2023-11-09 Kite cache +
          2023-11-10 .. 2026-11-09 bars3y), both through the corrected-data overlay (corrected.py). A cell (s, t) is
          the session t itself: a rule reads only bars of earlier sessions and the bars of t up to its decision minute
          (each summary field below names the last minute it reads), and end-of-day files dated < t.

Session summary fields (MINUTE grid), per symbol and session; a field is NaN when its bar is missing:
  pc            close of the last 1-minute bar of the symbol's previous session in the data
  o0915         open of the 09:15 bar (the call-auction print); NaN if the session's first bar is not 09:15
  oHHMM         open of the first bar at / after HH:MM and before HH:MM + 10 minutes (09:20 09:30 09:45 14:30 15:15
                15:21); a decision taken at the close of the bar before HH:MM fills there
  cHHMM         close of the last bar at / before HH:MM (09:19 09:29 09:44 14:29 15:20)
  v_a_b         volume of the bars from a to b inclusive (09:15-09:29, 09:15-09:44, 14:30-15:20, 09:15-15:20, whole day)
  nb            number of bars in the session

Nothing here computes a trade or a return over a holding period; datagen_run.py does.
"""
from __future__ import annotations

import bisect
import csv
import glob
import gzip
import json
import os

import numpy as np

from . import data as D
from . import data_daily as DD
from . import events_daily as ED

SCRATCH = DD._SCRATCH
DG_DIR = os.environ.get("STRATLAB_DATAGEN_DIR", os.path.join(SCRATCH, "datagen"))
CACHE = os.path.join(DG_DIR, "cache")
NSE_NEW = os.environ.get("STRATLAB_NSE_DIR", os.path.join(SCRATCH, "nse"))
NSE_OLD = os.path.join(SCRATCH, "nse_old")
OLD_VIEW = os.path.join(SCRATCH, "nse_old_view")
EVENTS = os.path.join(SCRATCH, "events")
FLOWS = os.path.join(SCRATCH, "databuild")
MIN_CACHES = {"old": os.path.join(SCRATCH, "oldera_min", "cache"),
              "3y": os.path.join(SCRATCH, "stratlab_cache", "bars3y_fb3d4d")}
OLD_END = 20231109                 # last session of the old era (both minute and daily)
INDEX_MIN = ("_NIFTY50_min", "_NIFTYBANK_min")
SPECIAL_MEDIAN_BARS = 300          # attention.py's rule: median bar count across symbols < 300 = special session

OPEN_FIELDS = {"o0920": 560, "o0930": 570, "o0945": 585, "o1430": 870, "o1515": 915, "o1521": 921}
CLOSE_FIELDS = {"c0919": 559, "c0929": 569, "c0944": 584, "c1429": 869, "c1520": 920}
VOL_FIELDS = {"v_0915_0929": (555, 569), "v_0915_0944": (555, 584), "v_1430_1520": (870, 920),
              "v_0915_1520": (555, 920), "v_day": (555, 929)}
MIN_FIELDS = ("pc", "o0915") + tuple(OPEN_FIELDS) + tuple(CLOSE_FIELDS) + tuple(VOL_FIELDS) + ("nb",)


def _cache(name: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name)


# ══════════════════════════════════════════════════════════════════════════
# symbol renames (NSE symbolchange.csv): every name -> the current symbol
# ══════════════════════════════════════════════════════════════════════════
_CH: dict = {}


def _changes() -> dict:
    if not _CH:
        from .events_parse import load_changes
        ch = load_changes(os.path.join(EVENTS, "raw", "ref", "symbolchange.csv"))
        by: dict = {}
        for d, old, new in ch:
            by.setdefault(old, []).append((d, new))
        _CH["list"], _CH["by"] = ch, by
    return _CH["by"]


def canon(sym: str, day: int) -> str:
    """events_parse.canon (every later rename followed), fast for the common case of no rename."""
    by = _changes()
    s = (sym or "").strip().upper()
    if s not in by:
        return s
    from .events_parse import canon as _c
    return _c(s, int(day), _CH["list"])


def canon_frame(df, sym_col: str = "symbol", day_col: str = "day"):
    """Vectorised canon over a pandas frame (only rows whose symbol was ever renamed are looped)."""
    by = _changes()
    out = df[sym_col].str.strip().str.upper()
    m = out.isin(set(by))
    if m.any():
        sub = df.loc[m, [sym_col, day_col]]
        pairs = {(s, int(d)) for s, d in zip(out[m], sub[day_col])}
        mp = {k: canon(k[0], k[1]) for k in pairs}
        out.loc[m] = [mp[(s, int(d))] for s, d in zip(out[m], sub[day_col])]
    return out


# ══════════════════════════════════════════════════════════════════════════
# DAILY grid
# ══════════════════════════════════════════════════════════════════════════
class DailyPanel:
    """Aligned [symbols x sessions] arrays of the events_daily cache (NaN where a symbol did not trade)."""

    def __init__(self, syms, cal, o, h, l, c, v, member, fo, nifty, vix, excl):
        self.syms, self.cal = list(syms), np.asarray(cal, np.int64)
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.member, self.fo = member, fo
        self.nifty, self.vix = nifty, vix               # {"o": [T], "c": [T]} official index; vix {"c": [T]}
        self.excl = excl                                 # {sym index: sorted int64 dates}
        self.six = {s: i for i, s in enumerate(self.syms)}
        self.tix = {int(d): i for i, d in enumerate(self.cal)}

    @property
    def shape(self):
        return self.c.shape

    def excluded(self, s: int, d0: int, d1: int) -> bool:
        """events_data.ca_excluded: an excluded day of symbol s with d0 < day <= d1."""
        v = self.excl.get(s)
        if v is None or not len(v):
            return False
        i = int(np.searchsorted(v, d0, side="right"))
        return i < len(v) and v[i] <= d1


def load_fo_by_day() -> dict:
    with gzip.open(os.path.join(EVENTS, "fo_by_day.json.gz"), "rt") as f:
        return {int(k): frozenset(v) for k, v in json.load(f).items()}


def build_daily_panel(cache: str = ED.CACHE, symbols=None, fo_by_day: dict | None = None) -> DailyPanel:
    m = DD.manifest(cache)
    cal = np.asarray(m["calendar"], np.int64)
    syms = sorted(s for s, x in m["symbols"].items() if x["member_days"] > 0 and not s.startswith("_"))
    if symbols is not None:
        syms = [s for s in syms if s in set(symbols)]
    S, T = len(syms), len(cal)
    arr = {k: np.full((S, T), np.nan) for k in "ohlcv"}
    member = np.zeros((S, T), bool)
    for i, s in enumerate(syms):
        sd = DD.load_symbol(cache, s)
        if sd is None:
            continue
        j = np.searchsorted(cal, sd.days.astype(np.int64))
        ok = (j < T) & (cal[np.minimum(j, T - 1)] == sd.days)
        j = j[ok]
        for k in "ohlcv":
            arr[k][i, j] = getattr(sd, k)[ok]
        member[i, j] = sd.member[ok]
    fo = np.zeros((S, T), bool)
    fbd = load_fo_by_day() if fo_by_day is None else fo_by_day
    fdays = sorted(fbd)
    six = {s: i for i, s in enumerate(syms)}
    for t, d in enumerate(cal):
        k = bisect.bisect_right(fdays, int(d))
        if k:
            for s in fbd[fdays[k - 1]]:
                i = six.get(s)
                if i is not None:
                    fo[i, t] = True
    nf = DD.load_daily_aux(cache, "_NIFTY50_day")
    vx = DD.load_daily_aux(cache, "_VIX_day")

    def _align(a, key):
        out = np.full(T, np.nan)
        if a is None:
            return out
        j = np.searchsorted(cal, a["date"].astype(np.int64))
        ok = (j < T) & (cal[np.minimum(j, T - 1)] == a["date"])
        out[j[ok]] = a[key][ok]
        return out

    nifty = {"o": _align(nf, "o"), "c": _align(nf, "c")}
    vix = {"c": _align(vx, "c")}
    excl: dict = {}
    for s, d in [tuple(x) for x in m.get("bad_gaps", [])] + [tuple(x) for x in m.get("ca_ambiguous", [])]:
        i = six.get(s)
        if i is not None:
            excl.setdefault(i, []).append(int(d))
    excl = {k: np.asarray(sorted(v), np.int64) for k, v in excl.items()}
    return DailyPanel(syms, cal, arr["o"], arr["h"], arr["l"], arr["c"], arr["v"], member, fo, nifty, vix, excl)


def _to_panel(panel: DailyPanel, df, value_col: str, agg: str = "last", dtype=float, fill=np.nan):
    """df with columns `cur` (current symbol) and `day` -> [S x T] array (rows outside the panel dropped)."""
    out = np.full(panel.shape, fill, dtype=dtype)
    si = df["cur"].map(panel.six)
    ti = df["day"].map(panel.tix)
    ok = si.notna() & ti.notna()
    if not ok.any():
        return out
    sub = df.loc[ok, [value_col]].copy()
    sub["si"], sub["ti"] = si[ok].astype(int).values, ti[ok].astype(int).values
    if agg == "sum":
        g = sub.groupby(["si", "ti"])[value_col].sum()
    else:
        g = sub.groupby(["si", "ti"])[value_col].last()
    idx = np.array(g.index.tolist(), dtype=int).reshape(-1, 2)
    out[idx[:, 0], idx[:, 1]] = g.values.astype(dtype)
    return out


def build_delivery(panel: DailyPanel):
    """deliv_pct [S x T] (NSE delivery %, EQ series): old view up to 2023-11-09, nse/ afterwards."""
    import pandas as pd
    p = _cache("delivery_panel.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if list(z["syms"]) == panel.syms and len(z["cal"]) == len(panel.cal):
            return z["dp"]
    parts = []
    for path, lo, hi in ((os.path.join(OLD_VIEW, "delivery.csv.gz"), 0, OLD_END),
                         (os.path.join(NSE_NEW, "delivery.csv.gz"), OLD_END + 1, 99999999)):
        df = pd.read_csv(path, usecols=["date", "symbol", "deliv_pct"], dtype={"symbol": str})
        df["day"] = df["date"].astype(int)
        df = df[(df["day"] >= lo) & (df["day"] <= hi)]
        df["cur"] = canon_frame(df)
        parts.append(df[["cur", "day", "deliv_pct"]])
    df = pd.concat(parts)
    df["deliv_pct"] = pd.to_numeric(df["deliv_pct"], errors="coerce")
    dp = _to_panel(panel, df, "deliv_pct")
    np.savez_compressed(p, dp=dp.astype(np.float32), syms=np.array(panel.syms, dtype=object), cal=panel.cal)
    return dp.astype(np.float32)


def build_oi(panel: DailyPanel):
    """doi [S x T]: change of stock-futures open interest from the previous F&O session to session t, over the SAME
    contracts (those expiring after t): sum OI_t / sum OI_{t-1} - 1 (contracts listed on t count with OI_{t-1} = 0;
    the contract expiring on t is left out on both days, so expiry and rollover do not show as OI changes)."""
    import pandas as pd
    p = _cache("oi_panel.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if list(z["syms"]) == panel.syms and len(z["cal"]) == len(panel.cal):
            return z["doi"]
    parts = []
    for path, lo, hi in ((os.path.join(NSE_OLD, "fut.csv.gz"), 0, OLD_END),
                         (os.path.join(NSE_NEW, "fut.csv.gz"), OLD_END + 1, 99999999)):
        df = pd.read_csv(path, usecols=["date", "symbol", "expiry", "instrument", "oi"], dtype=str)
        df = df[df["instrument"].isin(["FUTSTK", "STF"])]
        df["day"] = df["date"].str.replace("-", "").astype(int)
        df = df[(df["day"] >= lo) & (df["day"] <= hi)]
        df["exp"] = df["expiry"].str.replace("-", "").astype(int)
        df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0.0)
        parts.append(df[["day", "symbol", "exp", "oi"]])
    g = oi_change(pd.concat(parts))
    g["cur"] = canon_frame(g)
    doi = _to_panel(panel, g, "doi")
    np.savez_compressed(p, doi=doi.astype(np.float32), syms=np.array(panel.syms, dtype=object), cal=panel.cal)
    return doi.astype(np.float32)


def oi_change(df):
    """df [day, symbol, exp, oi] (one row per contract and session) -> [day, symbol, oi, oi_prev, doi]: on each
    session, the contracts expiring AFTER it; oi_prev = the same contracts' OI on the previous F&O session (0 for a
    contract listed that day)."""
    df = df.drop_duplicates(["day", "symbol", "exp"], keep="last")
    days = np.array(sorted(df["day"].unique()))
    prev_of = dict(zip(days[1:], days[:-1]))
    df = df.assign(pday=df["day"].map(prev_of))
    prev = df[["day", "symbol", "exp", "oi"]].rename(columns={"day": "pday", "oi": "oi_prev"})
    df = df.merge(prev, on=["pday", "symbol", "exp"], how="left")
    df["oi_prev"] = df["oi_prev"].fillna(0.0)
    df = df[df["exp"] > df["day"]]
    g = df.groupby(["day", "symbol"]).agg(oi=("oi", "sum"), oi_prev=("oi_prev", "sum")).reset_index()
    g = g[g["oi_prev"] > 0].copy()
    g["doi"] = g["oi"] / g["oi_prev"] - 1.0
    return g


def build_ban(panel: DailyPanel):
    """ban [S x T] bool: in the F&O ban list FOR trade date t (file dated t, published the evening before)."""
    import pandas as pd
    parts = []
    for path, lo, hi in ((os.path.join(OLD_VIEW, "ban.csv.gz"), 0, OLD_END),
                         (os.path.join(NSE_NEW, "ban.csv.gz"), OLD_END + 1, 99999999)):
        df = pd.read_csv(path, dtype=str)
        df["day"] = df["date"].str.replace("-", "").astype(int)
        df = df[(df["day"] >= lo) & (df["day"] <= hi)]
        df["cur"] = canon_frame(df)
        df["one"] = 1.0
        parts.append(df[["cur", "day", "one"]])
    df = pd.concat(parts)
    b = _to_panel(panel, df, "one", fill=0.0)
    return b > 0


def build_participant(panel: DailyPanel) -> dict:
    """{field: [T]} participant-wise OI per session (NaN where missing): FII and Client index-futures long / short."""
    import pandas as pd
    out = {}
    parts = []
    for path, lo, hi in ((os.path.join(OLD_VIEW, "participant_oi.csv.gz"), 0, OLD_END),
                         (os.path.join(NSE_NEW, "participant_oi.csv.gz"), OLD_END + 1, 99999999)):
        df = pd.read_csv(path)
        df["day"] = df["date"].astype(str).str.replace("-", "").astype(int)
        df = df[(df["day"] >= lo) & (df["day"] <= hi)]
        parts.append(df)
    df = pd.concat(parts)
    for who in ("FII", "Client"):
        sub = df[df["client_type"].astype(str).str.strip().str.upper() == who.upper()].drop_duplicates("day", keep="last")
        for col in ("fut_idx_long", "fut_idx_short"):
            a = np.full(len(panel.cal), np.nan)
            for d, v in zip(sub["day"], pd.to_numeric(sub[col], errors="coerce")):
                t = panel.tix.get(int(d))
                if t is not None:
                    a[t] = v
            out[f"{who.lower()}_{col}"] = a
    return out


def build_deals(panel: DailyPanel) -> dict:
    """{"inst": [S x T], "other": [S x T]} net Rs crore of the session's bulk + block deals by client class
    (flows_strategies: intra-group transfers removed, each client netted per session, classes by the fixed keyword
    list; PROP_BROKER excluded), summed over the clients of the class. Dated D: usable from D's evening."""
    import pandas as pd
    from . import flows_strategies as FS
    p = os.path.join(FLOWS, "deals", "deals.csv.gz")
    df = pd.read_csv(p, dtype=str)
    rows = {"date": df["date"].astype(int).values, "client": df["client"].fillna("").values,
            "side": np.where(df["side"] == "BUY", 1, -1),
            "value_cr": pd.to_numeric(df["value_cr"], errors="coerce").values,
            "price": pd.to_numeric(df["price"], errors="coerce").values,
            "qty": pd.to_numeric(df["qty"], errors="coerce").values,
            "lab": df["lab_symbol"].fillna("").values}
    out = {"inst": np.zeros(panel.shape), "other": np.zeros(panel.shape)}
    order = np.lexsort((rows["date"], rows["lab"]))
    labs = rows["lab"][order]
    bounds = np.flatnonzero(np.r_[True, labs[1:] != labs[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        sym = labs[a]
        i = panel.six.get(sym)
        if i is None:
            continue
        idx = order[a:b]
        sub = {k: v[idx] for k, v in rows.items()}
        for d, cl, net in FS.net_flows(sub):
            t = panel.tix.get(int(d))
            if t is None:
                continue
            cls = FS.classify(cl)
            if cls in FS.INSTITUTION:
                out["inst"][i, t] += net
            elif cls == FS.OTHER:
                out["other"][i, t] += net
    return out


def _results_table() -> dict:
    """{current symbol: sorted int64 stamps yyyymmddhhmm of the first public moment of each quarterly result}."""
    tab: dict = {}
    with gzip.open(os.path.join(EVENTS, "results.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            ts = r.get("first_ts") or ""
            if len(ts) < 16:
                continue
            st = int(ts[0:4] + ts[5:7] + ts[8:10] + ts[11:13] + ts[14:16])
            tab.setdefault(r["symbol"].strip().upper(), []).append(st)
    return {k: np.asarray(sorted(v), np.int64) for k, v in tab.items()}


def results_reaction(syms, cal, tab: dict | None = None) -> np.ndarray:
    """react [S x T] bool: session t is the first session whose CLOSE (15:30) comes after a result's first public
    moment, i.e. stamp(t-1, 15:30) <= ts < stamp(t, 15:30). Known by the close of t."""
    tab = _results_table() if tab is None else tab
    cal = np.asarray(cal, np.int64)
    closes = cal * 10000 + 1530
    out = np.zeros((len(syms), len(cal)), bool)
    for i, s in enumerate(syms):
        ts = tab.get(s)
        if ts is None:
            continue
        j = np.searchsorted(closes, ts, side="right")          # first close strictly after ts
        j = j[(j > 0) & (j < len(cal))]
        out[i, j] = True
    return out


def results_premarket(syms, cal, tab: dict | None = None) -> np.ndarray:
    """pre [S x T] bool: a result became public between the previous session's close (15:30) and 09:15 of session t
    (out of market hours, before t's open)."""
    tab = _results_table() if tab is None else tab
    cal = np.asarray(cal, np.int64)
    out = np.zeros((len(syms), len(cal)), bool)
    for i, s in enumerate(syms):
        ts = tab.get(s)
        if ts is None:
            continue
        for t in range(1, len(cal)):
            lo, hi = cal[t - 1] * 10000 + 1530, cal[t] * 10000 + 915
            k = int(np.searchsorted(ts, lo))
            if k < len(ts) and ts[k] < hi:
                out[i, t] = True
    return out


def corp_action_events(panel: DailyPanel) -> dict:
    """{"splitbonus": [(s, known_day, ex_day)], "dividend": [(s, known_day, ex_day, rupees)]} from the corporate-action
    table (databuild/corpactions; `known_date` = the first session whose NSE Bc file lists the action)."""
    out = {"splitbonus": [], "dividend": []}
    with gzip.open(os.path.join(FLOWS, "corpactions", "corpactions.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            i = panel.six.get(r.get("lab_symbol") or "")
            if i is None:
                continue
            try:
                ex, kn = int(r["ex_date"]), int(r["known_date"])
            except Exception:
                continue
            if r["kind"] in ("bonus", "split"):
                out["splitbonus"].append((i, kn, ex))
            elif r["kind"] == "dividend":
                try:
                    dv = float(r["dividend"])
                except Exception:
                    continue
                if dv > 0:
                    out["dividend"].append((i, kn, ex, dv))
    return out


def index_expiries() -> dict:
    """{"NIFTY": sorted expiry days, "BANKNIFTY": ...}: days on which an option contract of the index traded on its
    own expiry date (both bhavcopy files), i.e. the index's weekly / monthly expiry sessions."""
    import pandas as pd
    p = _cache("expiries.json")
    if os.path.exists(p):
        with open(p) as f:
            return {k: list(v) for k, v in json.load(f).items()}
    acc = {"NIFTY": set(), "BANKNIFTY": set()}
    for path in (os.path.join(NSE_OLD, "opt_index.csv.gz"), os.path.join(NSE_NEW, "opt_index.csv.gz")):
        for ch in pd.read_csv(path, usecols=["date", "symbol", "expiry"], dtype=str, chunksize=2_000_000):
            ch = ch[(ch["date"] == ch["expiry"]) & ch["symbol"].isin(list(acc))]
            for s, d in zip(ch["symbol"], ch["date"]):
                acc[s].add(int(d.replace("-", "")))
    out = {k: sorted(v) for k, v in acc.items()}
    with open(p, "w") as f:
        json.dump(out, f)
    return out


# ══════════════════════════════════════════════════════════════════════════
# MINUTE grid
# ══════════════════════════════════════════════════════════════════════════
def session_summary(sd) -> dict:
    """Per session of one minute SymbolData: the MIN_FIELDS (see the module docstring) + `days`."""
    n = sd.n_sessions
    out = {k: np.full(n, np.nan) for k in MIN_FIELDS}
    out["days"] = sd.days.astype(np.int64)
    for k in range(n):
        a, b = int(sd.starts[k]), int(sd.ends[k])
        tt = sd.t[a:b].astype(np.int64)
        if k > 0:
            out["pc"][k] = float(sd.c[int(sd.ends[k - 1]) - 1])
        if len(tt) and tt[0] == 555:
            out["o0915"][k] = float(sd.o[a])
        for f, m in OPEN_FIELDS.items():
            j = int(np.searchsorted(tt, m))
            if j < len(tt) and tt[j] < m + 10:
                out[f][k] = float(sd.o[a + j])
        for f, m in CLOSE_FIELDS.items():
            j = int(np.searchsorted(tt, m, side="right")) - 1
            if j >= 0:
                out[f][k] = float(sd.c[a + j])
        cv = np.concatenate([[0.0], np.cumsum(sd.v[a:b].astype(float))])
        for f, (lo, hi) in VOL_FIELDS.items():
            j0, j1 = int(np.searchsorted(tt, lo)), int(np.searchsorted(tt, hi, side="right"))
            out[f][k] = cv[j1] - cv[j0]
        out["nb"][k] = b - a
    return out


def _sum_one(args):
    cache, sym = args
    sd = D.load_symbol(cache, sym)
    if sd is None:
        return sym, None
    return sym, session_summary(sd)


class MinutePanel:
    """[names x sessions] summary arrays on the joint minute calendar; names = stocks then the two index files."""

    def __init__(self, syms, cal, fields: dict, special: set, dirty: dict):
        self.syms, self.cal = list(syms), np.asarray(cal, np.int64)
        self.f = fields
        self.special = special
        self.dirty = dirty                  # {name index: sorted int64 dirty dates}
        self.six = {s: i for i, s in enumerate(self.syms)}
        self.tix = {int(d): i for i, d in enumerate(self.cal)}
        self.stocks = [i for i, s in enumerate(self.syms) if not s.startswith("_")]

    @property
    def shape(self):
        return self.f["nb"].shape

    def has(self) -> np.ndarray:
        return np.isfinite(self.f["nb"])


def build_minute_panel(jobs: int = 4, symbols=None) -> MinutePanel:
    """Both eras through corrected.py's overlay, cached in <datagen>/cache/minsum.npz."""
    import multiprocessing as mp
    from . import corrected
    p = _cache("minsum.npz")
    if os.path.exists(p) and symbols is None:
        z = np.load(p, allow_pickle=True)
        syms, cal = list(z["syms"]), z["cal"]
        fields = {k: z[k] for k in MIN_FIELDS}
        special = set(int(x) for x in z["special"])
        dirty = json.loads(str(z["dirty"]))
        return MinutePanel(syms, cal, fields, special, {int(k): np.asarray(v, np.int64) for k, v in dirty.items()})
    corrected.install(minute=True, daily=True)
    per_era = {}
    for era, cache in MIN_CACHES.items():
        with open(os.path.join(cache, "_manifest.json")) as fh:
            names = sorted(json.load(fh)["symbols"])
        names = [s for s in names if not s.startswith("_")]
        if symbols is not None:
            names = [s for s in names if s in set(symbols)]
        names += list(INDEX_MIN)
        with mp.get_context("fork").Pool(jobs) as pool:
            per_era[era] = dict(pool.map(_sum_one, [(cache, s) for s in names], chunksize=2))
    cal = set()
    for era in per_era:
        z = per_era[era].get("_NIFTY50_min")
        cal.update(int(x) for x in z["days"])
    cal = np.asarray(sorted(cal), np.int64)
    syms = sorted({s for e in per_era.values() for s, v in e.items() if v is not None and not s.startswith("_")})
    syms += list(INDEX_MIN)
    S, T = len(syms), len(cal)
    fields = {k: np.full((S, T), np.nan) for k in MIN_FIELDS}
    for i, s in enumerate(syms):
        for era in ("old", "3y"):
            z = per_era[era].get(s)
            if z is None:
                continue
            j = np.searchsorted(cal, z["days"])
            ok = (j < T) & (cal[np.minimum(j, T - 1)] == z["days"])
            for k in MIN_FIELDS:
                fields[k][i, j[ok]] = z[k][ok]
    # previous close across the seam (the first 3y session): the last bar close of the last old-era session, when
    # that session is the previous calendar session (the corrected view puts both eras on one price basis)
    for i, s in enumerate(syms):
        sd = D.load_symbol(MIN_CACHES["old"], s)
        sd3 = D.load_symbol(MIN_CACHES["3y"], s)
        if sd is None or sd3 is None or not sd.n_sessions or not sd3.n_sessions:
            continue
        j = int(np.searchsorted(cal, int(sd3.days[0])))
        if j < T and int(cal[j]) == int(sd3.days[0]) and int(sd.days[-1]) == int(cal[j - 1]):
            fields["pc"][i, j] = float(sd.c[-1])
    nbs = fields["nb"][: len(syms) - len(INDEX_MIN)]
    med = np.nanmedian(np.where(np.isfinite(nbs), nbs, 0.0), axis=0)
    special = {int(d) for d, m in zip(cal, med) if m < SPECIAL_MEDIAN_BARS}
    with open(os.path.join(MIN_CACHES["old"], "dirty_links.json")) as fh:
        dl = json.load(fh)
    from .ltr import CA_GAPS
    dirty: dict = {}
    for i, s in enumerate(syms):
        v = [int(x) for x in dl.get(s, [])] + [int(d) for (ss, d) in CA_GAPS if ss == s]
        if v:
            dirty[i] = np.asarray(sorted(v), np.int64)
    if symbols is None:
        np.savez_compressed(p, syms=np.array(syms, dtype=object), cal=cal, special=np.array(sorted(special)),
                            dirty=json.dumps({str(k): v.tolist() for k, v in dirty.items()}),
                            **{k: v.astype(np.float64) for k, v in fields.items()})
    return MinutePanel(syms, cal, fields, special, dirty)


# ══════════════════════════════════════════════════════════════════════════
# helpers used by the rules (all causal: value at t uses columns <= t only)
# ══════════════════════════════════════════════════════════════════════════
def lag(x: np.ndarray, k: int = 1) -> np.ndarray:
    """x shifted right by k columns (value at t = x at t-k), NaN-filled."""
    out = np.full_like(x, np.nan, dtype=float)
    if k < x.shape[-1]:
        out[..., k:] = x[..., :-k]
    return out


def trailing_mean(x: np.ndarray, n: int, min_count: int, include_current: bool = False) -> np.ndarray:
    """Mean of the previous n columns (t-n .. t-1; with include_current t-n+1 .. t), NaN-aware; NaN if fewer than
    min_count finite values."""
    x = np.asarray(x, float)
    if not include_current:
        x = lag(x, 1)
    v = np.where(np.isfinite(x), x, 0.0)
    c = np.isfinite(x).astype(float)
    cv = np.cumsum(v, axis=-1)
    cc = np.cumsum(c, axis=-1)
    sv = cv.copy()
    sc = cc.copy()
    sv[..., n:] = cv[..., n:] - cv[..., :-n]
    sc[..., n:] = cc[..., n:] - cc[..., :-n]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = sv / sc
    m[sc < min_count] = np.nan
    return m


def trailing_std(x: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """Sample std of the previous n columns (t-n .. t-1), NaN-aware."""
    x1 = lag(np.asarray(x, float), 1)
    v = np.where(np.isfinite(x1), x1, 0.0)
    c = np.isfinite(x1).astype(float)

    def _roll(a):
        ca = np.cumsum(a, axis=-1)
        r = ca.copy()
        r[..., n:] = ca[..., n:] - ca[..., :-n]
        return r
    s1, s2, cnt = _roll(v), _roll(v * v), _roll(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / cnt) / (cnt - 1)
    var[cnt < min_count] = np.nan
    return np.sqrt(np.maximum(var, 0.0))


def trailing_rank_pct(x: np.ndarray, n: int, min_count: int) -> np.ndarray:
    """Percentile (0..1) of x[t] among x[t-n .. t-1] (1-D series), NaN if too few values."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        if not np.isfinite(x[t]):
            continue
        w = x[max(0, t - n):t]
        w = w[np.isfinite(w)]
        if len(w) >= min_count:
            out[t] = float((w < x[t]).mean())
    return out



# ══════════════════════════════════════════════════════════════════════════
# context: every panel and derived feature, built once (before forking workers)
# ══════════════════════════════════════════════════════════════════════════
class Ctx:
    """Lazily built panels. `daily` / `minute` are DailyPanel / MinutePanel; derived features are cached by name."""

    def __init__(self, daily: DailyPanel | None = None, minute: MinutePanel | None = None):
        self.daily, self.minute = daily, minute
        self._c: dict = {}

    def get(self, name: str, fn):
        if name not in self._c:
            self._c[name] = fn()
        return self._c[name]

    # ── daily derived (value at t uses columns <= t: the decision is taken in the evening of t) ──
    def d(self, name: str):
        P = self.daily
        f = {
            "ret1": lambda: P.c / lag(P.c, 1) - 1.0,
            "ret5": lambda: P.c / lag(P.c, 5) - 1.0,
            "ret20": lambda: P.c / lag(P.c, 20) - 1.0,
            "gap": lambda: P.o / lag(P.c, 1) - 1.0,
            "range": lambda: (P.h - P.l) / P.c,
            "vr": lambda: P.v / trailing_mean(P.v, 20, 15),
            "nifty_ret1": lambda: P.nifty["c"] / lag(P.nifty["c"], 1) - 1.0,
            "abn1": lambda: self.d("ret1") - self.d("nifty_ret1")[None, :],
            "vix_chg": lambda: P.vix["c"] / lag(P.vix["c"], 1) - 1.0,
            "vix_pct": lambda: trailing_rank_pct(P.vix["c"], 252, 200),
            "dp": lambda: build_delivery(P).astype(float),
            "dp_shock": lambda: self.d("dp") - trailing_mean(self.d("dp"), 20, 15),
            "dp_trend": lambda: trailing_mean(self.d("dp"), 5, 4, include_current=True)
            - trailing_mean(self.d("dp"), 60, 45),
            "doi": lambda: build_oi(P).astype(float),
            "ban": lambda: build_ban(P),
            "deals": lambda: build_deals(P),
            "part": lambda: build_participant(P),
            "react": lambda: results_reaction(P.syms, P.cal),
            "ca": lambda: corp_action_events(P),
        }[name]
        return self.get("d:" + name, f)

    # ── minute derived (value at t uses earlier sessions + bars of t up to the field's minute) ──
    def m(self, name: str):
        M = self.minute
        F = M.f if M is not None else {}
        f = {
            "gap": lambda: F["o0915"] / F["pc"] - 1.0,
            "r5": lambda: F["c0919"] / F["o0915"] - 1.0,
            "r_early": lambda: F["c0944"] / F["o0915"] - 1.0,
            "r_last": lambda: F["c1520"] / F["o1430"] - 1.0,
            "vE": lambda: F["v_0915_0944"] / trailing_mean(F["v_0915_0944"], 20, 15),
            "v15": lambda: F["v_0915_0929"] / trailing_mean(F["v_0915_0929"], 20, 15),
            "vday": lambda: F["v_0915_1520"] / trailing_mean(F["v_day"], 20, 15),
            "vlast": lambda: F["v_1430_1520"] / trailing_mean(F["v_1430_1520"], 20, 15),
            "pre": lambda: results_premarket(M.syms, M.cal),
            "prev_ok": lambda: lag(np.isfinite(F["nb"]).astype(float), 1) > 0,
            "exp": lambda: index_expiries(),
        }[name]
        return self.get("m:" + name, f)


def _sgn(x):
    return np.sign(np.nan_to_num(np.asarray(x, float), nan=0.0)).astype(np.int8)


def _out(trig, side, rank, hold=None):
    side = np.broadcast_to(np.asarray(side, np.int8), np.shape(trig))
    rank = np.broadcast_to(np.asarray(rank, float), np.shape(trig))
    trig = np.asarray(trig, bool) & np.isfinite(rank) & (side != 0)
    return trig, np.where(trig, side, 0).astype(np.int8), np.where(trig, rank, np.nan), hold


def _stock_rows(M: MinutePanel, x: np.ndarray) -> np.ndarray:
    m = np.zeros(M.shape, bool)
    m[M.stocks] = True
    return np.asarray(x, bool) & m


# ══════════════════════════════════════════════════════════════════════════
# RULES. Each returns (trig, side, rank, hold) on its grid; side +1 long / -1 short; rank = strength (the 10 largest
# per day are taken); hold = per-cell holding period in sessions or None (the candidate's fixed horizon).
# ══════════════════════════════════════════════════════════════════════════
MAX_GAP = 0.20                      # |gap| above this = corporate action / circuit (auction.py)


def r_gap_path(ctx: Ctx, g: float, mode: str):
    """A: opening-auction gap and the 09:15-09:20 path. go = the first five minutes confirm the gap (same sign as the
    gap): trade with the gap; revfollow = they moved against it: trade against it; fade_day = fade from the auction."""
    M = ctx.minute
    gap, r5 = ctx.m("gap"), ctx.m("r5")
    base = _stock_rows(M, (np.abs(gap) >= g - 1e-12) & (np.abs(gap) <= MAX_GAP) & ctx.m("prev_ok"))
    if mode == "go":
        trig, side = base & (_sgn(r5) == _sgn(gap)), _sgn(gap)
    elif mode == "revfollow":
        trig, side = base & (_sgn(r5) == -_sgn(gap)), -_sgn(gap)
    else:
        trig, side = base, -_sgn(gap)
    return _out(trig, side, np.abs(gap))


def r_open5_shock(ctx: Ctx, r: float, mode: str):
    """A: no real gap (|gap| < 0.5 %) but the first five minutes move >= r: continue / fade from 09:20."""
    M = ctx.minute
    gap, r5 = ctx.m("gap"), ctx.m("r5")
    trig = _stock_rows(M, (np.abs(gap) < 0.005) & (np.abs(r5) >= r - 1e-12) & ctx.m("prev_ok"))
    return _out(trig, _sgn(r5) if mode == "cont" else -_sgn(r5), np.abs(r5))


def r_idio_gap(ctx: Ctx, g: float):
    """A: the stock gaps >= g against NIFTY 50's own opening gap (idiosyncratic auction imbalance): fade from 09:20."""
    M = ctx.minute
    gap = ctx.m("gap")
    ng = gap[M.six["_NIFTY50_min"]][None, :]
    trig = _stock_rows(M, (np.abs(gap) >= g - 1e-12) & (np.abs(gap) <= MAX_GAP) & (_sgn(gap) == -_sgn(ng))
                       & ctx.m("prev_ok"))
    return _out(trig, -_sgn(gap), np.abs(gap))


def r_early_vol(ctx: Ctx, k: float, mode: str):
    """B: volume 09:15-09:44 >= k x its 20-session mean and |move 09:15 open -> 09:44 close| >= 0.5 %: with (cont) /
    against (fade) the move from 09:45."""
    M = ctx.minute
    vE, r = ctx.m("vE"), ctx.m("r_early")
    trig = _stock_rows(M, (vE >= k) & (np.abs(r) >= 0.005))
    return _out(trig, _sgn(r) if mode == "cont" else -_sgn(r), vE)


def r_down_attention(ctx: Ctx, mode: str):
    """B: volume 09:15-15:20 >= 3 x the mean full-day volume of the previous 20 sessions on a DOWN day (15:20 close <
    previous close): overnight long (rev) / short (cont). The proven SWING-ATTN-ON-3X is the UP-day long."""
    M = ctx.minute
    vd = ctx.m("vday")
    trig = _stock_rows(M, (vd >= 3.0) & (M.f["c1520"] < M.f["pc"]))
    return _out(trig, 1 if mode == "rev" else -1, vd)


def r_lasthour_vol(ctx: Ctx, k: float, mode: str):
    """B: last-hour volume (14:30-15:20) >= k x its 20-session mean with |14:30 open -> 15:20 close| >= 0.5 %:
    overnight with (cont) / against (fade) that move."""
    M = ctx.minute
    vl, r = ctx.m("vlast"), ctx.m("r_last")
    trig = _stock_rows(M, (vl >= k) & (np.abs(r) >= 0.005))
    return _out(trig, _sgn(r) if mode == "cont" else -_sgn(r), vl)


def r_attention_swing(ctx: Ctx, day: str):
    """B: volume 09:15-15:20 >= 3 x the 20-session full-day mean on an up / down day: long for 5 sessions."""
    M = ctx.minute
    vd = ctx.m("vday")
    cond = (M.f["c1520"] > M.f["pc"]) if day == "up" else (M.f["c1520"] < M.f["pc"])
    return _out(_stock_rows(M, (vd >= 3.0) & cond), 1, vd)


def r_gap_volume(ctx: Ctx, mode: str):
    """B: |gap| >= 2 % with first-15-minute volume <= 0.5 x normal (no news: fade) or >= 3 x normal (news: follow),
    from 09:30."""
    M = ctx.minute
    gap, v15 = ctx.m("gap"), ctx.m("v15")
    base = (np.abs(gap) >= 0.02) & (np.abs(gap) <= MAX_GAP) & ctx.m("prev_ok")
    if mode == "lowvol_fade":
        trig, side = base & (v15 <= 0.5), -_sgn(gap)
    else:
        trig, side = base & (v15 >= 3.0), _sgn(gap)
    return _out(_stock_rows(M, trig), side, np.abs(gap))


def r_expiry(ctx: Ctx, when: str, side: int):
    """C: index weekly / monthly expiry. on = the index's own expiry session, overnight to the next open; pre = the
    session before the index's expiry session, overnight into it."""
    M = ctx.minute
    exp = ctx.m("exp")
    trig = np.zeros(M.shape, bool)
    for name, key in (("_NIFTY50_min", "NIFTY"), ("_NIFTYBANK_min", "BANKNIFTY")):
        i = M.six[name]
        days = set(int(x) for x in exp[key])
        for t, d in enumerate(M.cal):
            if when == "on" and int(d) in days:
                trig[i, t] = True
            elif when == "pre" and t + 1 < len(M.cal) and int(M.cal[t + 1]) in days:
                trig[i, t] = True
    return _out(trig, side, 1.0)


def r_vix_spike_losers(ctx: Ctx, x: float):
    """D: India VIX up >= x close to close: buy the stocks that fell most that day (forced de-risking)."""
    vc, r1 = ctx.d("vix_chg"), ctx.d("ret1")
    return _out((vc[None, :] >= x - 1e-12) & (r1 < 0), 1, -r1)


def r_vix_high_losers(ctx: Ctx):
    """D: India VIX in the top decile of its previous 252 sessions: buy the stocks whose 5-session fall is >= 5 %."""
    vp, r5 = ctx.d("vix_pct"), ctx.d("ret5")
    return _out((vp[None, :] >= 0.9) & (r5 <= -0.05), 1, -r5)


def r_vix_high_index(ctx: Ctx):
    """D: India VIX in the top decile of its previous 252 sessions: long NIFTY (fear premium)."""
    vp = ctx.d("vix_pct")
    return _out((vp >= 0.9)[None, :], 1, 1.0)


def r_delivery_shock(ctx: Ctx, pp: float, day: str):
    """E: delivery % >= its 20-session mean + pp points on >= 2 x volume with a >= 2 % move: up -> long (genuine
    accumulation), down -> short (genuine distribution)."""
    ds, vr, r1 = ctx.d("dp_shock"), ctx.d("vr"), ctx.d("ret1")
    base = (ds >= pp) & (vr >= 2.0)
    if day == "up":
        return _out(base & (r1 >= 0.02), 1, ds)
    return _out(base & (r1 <= -0.02), -1, ds)


def r_low_delivery_spike(ctx: Ctx):
    """E: delivery % <= its 20-session mean - 15 points on >= 3 x volume with |move| >= 3 % (speculative churn):
    fade the move."""
    ds, vr, r1 = ctx.d("dp_shock"), ctx.d("vr"), ctx.d("ret1")
    return _out((ds <= -15.0) & (vr >= 3.0) & (np.abs(r1) >= 0.03), -_sgn(r1), vr)


def r_delivery_trend(ctx: Ctx):
    """E: 5-session mean delivery % >= the previous 60-session mean + 10 points with a positive 20-session return
    (steady accumulation): long."""
    dt, r20 = ctx.d("dp_trend"), ctx.d("ret20")
    return _out((dt >= 10.0) & (r20 > 0), 1, dt)


def r_fii_flow(ctx: Ctx, z: float):
    """F: FII net index-futures position (long - short contracts) changes by >= z standard deviations (previous 250
    daily changes) in a day: follow the FIIs on NIFTY."""
    pa = ctx.d("part")
    net = pa["fii_fut_idx_long"] - pa["fii_fut_idx_short"]
    dn = net - lag(net, 1)
    zz = dn / trailing_std(dn, 250, 200)
    return _out((np.abs(zz) >= z - 1e-12)[None, :], _sgn(zz)[None, :], np.abs(zz)[None, :])


def r_client_contrarian(ctx: Ctx):
    """F: Client (retail) index-futures long share in the top / bottom decile of its previous 250 sessions: trade
    NIFTY against the crowd."""
    pa = ctx.d("part")
    L, S = pa["client_fut_idx_long"], pa["client_fut_idx_short"]
    pct = trailing_rank_pct(L / (L + S), 250, 200)
    side = np.where(pct >= 0.9, -1, np.where(pct <= 0.1, 1, 0)).astype(np.int8)
    return _out((side != 0)[None, :], side[None, :], 1.0)


def r_ban(ctx: Ctx, event: str, side: int):
    """G: F&O ban list. entry = the first trade date in the ban (the file of t+1 lists it, t's did not); exit = the
    first trade date out of it. Decided in the evening of t (t+1's ban file is published after t's close), entered at
    t+1's open."""
    b = ctx.d("ban")
    nb = np.zeros_like(b)
    nb[:, :-1] = b[:, 1:]
    trig = (nb & ~b) if event == "entry" else (b & ~nb)
    trig[:, -1] = False
    return _out(trig, side, 1.0)


def r_oi_quadrant(ctx: Ctx, quad: str, thr: float):
    """H: stock-futures OI change vs price change on the day (|move| >= 2 %, |OI change| >= thr): long build-up
    (price up, OI up) -> long; short build-up (down, up) -> short; short covering (up, OI down) -> short (a forced move
    fades); long unwinding (down, down) -> long."""
    doi, r1 = ctx.d("doi"), ctx.d("ret1")
    up, dn = r1 >= 0.02, r1 <= -0.02
    oiu, oid = doi >= thr - 1e-12, doi <= -thr + 1e-12
    trig, s = {"long_buildup": (up & oiu, 1), "short_buildup": (dn & oiu, -1),
               "short_covering": (up & oid, -1), "long_unwinding": (dn & oid, 1)}[quad]
    return _out(trig, s, np.abs(doi))


def r_deals(ctx: Ctx, cls: str, flow: str, side: int):
    """I: net bulk / block deals of the client class on the day >= Rs 10 crore (buy) or <= -Rs 10 crore (sell)."""
    x = ctx.d("deals")[cls]
    return _out((x >= 10.0) if flow == "buy" else (x <= -10.0), side, np.abs(x))


def r_results_gap(ctx: Ctx, g: float, mode: str):
    """J: results published out of market hours (after the previous close, before 09:15) and the stock gaps >= g:
    with (cont) / against (fade) the gap from 09:20."""
    M = ctx.minute
    gap = ctx.m("gap")
    trig = _stock_rows(M, ctx.m("pre") & (np.abs(gap) >= g - 1e-12) & (np.abs(gap) <= MAX_GAP) & ctx.m("prev_ok"))
    return _out(trig, _sgn(gap) if mode == "cont" else -_sgn(gap), np.abs(gap))


def r_results_reaction(ctx: Ctx, a: float, mode: str):
    """J: on the results reaction day the stock beats (up) / lags (down) NIFTY by >= a close to close: up -> long,
    down -> short; fade = against the move either way."""
    ab, re = ctx.d("abn1"), ctx.d("react")
    if mode == "up":
        return _out(re & (ab >= a - 1e-12), 1, np.abs(ab))
    if mode == "down":
        return _out(re & (ab <= -a + 1e-12), -1, np.abs(ab))
    return _out(re & (np.abs(ab) >= a - 1e-12), -_sgn(ab), np.abs(ab))


def r_split_bonus(ctx: Ctx, mode: str):
    """K: split / bonus. runup = buy at the open after the session the action became known, sell at the close before
    the ex-date (1..40 sessions); post = buy at the ex-date open (known before), hold 20 sessions."""
    P = ctx.daily
    trig = np.zeros(P.shape, bool)
    hold = np.zeros(P.shape, np.int32)
    for i, kn, ex in ctx.d("ca")["splitbonus"]:
        te = int(np.searchsorted(P.cal, ex))            # the ex-date session (or the next session)
        if te >= len(P.cal) or te < 1:
            continue
        if mode == "runup":
            tk = int(np.searchsorted(P.cal, kn))        # decision in the evening of the known session
            h = te - 1 - tk                              # entry tk+1 .. exit te-1
            if tk < len(P.cal) and 1 <= h <= 40:
                trig[i, tk], hold[i, tk] = True, h
        else:
            t = te - 1
            if kn <= int(P.cal[t]):
                trig[i, t], hold[i, t] = True, 20
    return _out(trig, 1, 1.0, hold)


def r_dividend_runup(ctx: Ctx, y: float):
    """K: dividend >= y of the price (known by the decision): buy 5 sessions before the ex-date, sell at the close of
    the last cum-dividend session (dividend-capture demand)."""
    P = ctx.daily
    trig = np.zeros(P.shape, bool)
    rank = np.full(P.shape, np.nan)
    for i, kn, ex, dv in ctx.d("ca")["dividend"]:
        te = int(np.searchsorted(P.cal, ex))
        t = te - 6
        if te >= len(P.cal) or t < 0 or kn > int(P.cal[t]):
            continue
        c = P.c[i, t]
        if np.isfinite(c) and c > 0 and dv / c >= y - 1e-12:
            trig[i, t], rank[i, t] = True, dv / c
    hold = np.where(trig, 5, 0).astype(np.int32)
    return _out(trig, 1, rank, hold)


# ── M: one pre-declared machine-learning model (L2 logistic, fixed C = 1.0, walk-forward inside DISCOVERY) ──
ML_FEATURES = ("ret1", "ret5", "ret20", "logvr", "dp_shock", "doi", "gap", "range")
ML_C = 1.0                          # sklearn's default inverse regularisation (0.5 |w|^2 + C sum logloss)
ML_HORIZON = 5
ML_FIRST_TRAIN_YEARS = 2            # the first predictions are for 2014 (fitted on 2012-2013)


def ml_matrix(ctx: Ctx):
    """Cross-sectional z-scores (per session over F&O universe members, clipped +-3, missing -> 0) of ML_FEATURES on
    the DAILY grid; all known at the decision (close of t / EOD files dated t)."""
    P = ctx.daily
    raw = {"ret1": ctx.d("ret1"), "ret5": ctx.d("ret5"), "ret20": ctx.d("ret20"),
           "logvr": np.log(np.maximum(ctx.d("vr"), 1e-6)), "dp_shock": ctx.d("dp_shock"), "doi": ctx.d("doi"),
           "gap": ctx.d("gap"), "range": ctx.d("range")}
    elig = P.member & P.fo
    X = np.zeros((len(ML_FEATURES),) + P.shape, np.float32)
    for k, name in enumerate(ML_FEATURES):
        x = np.where(elig & np.isfinite(raw[name]), raw[name], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu = np.nanmean(x, axis=0)
                sd = np.nanstd(x, axis=0)
            z = (x - mu[None, :]) / sd[None, :]
        X[k] = np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -3, 3)
    return X, elig


def ml_label(ctx: Ctx) -> np.ndarray:
    """1 if the stock beats NIFTY from the next open to the close ML_HORIZON sessions after the decision (entry t+1 ..
    exit t+5), 0 if not, NaN if unknown. Used ONLY to fit the model on decision days whose window ended before the
    fit date."""
    P = ctx.daily
    h = ML_HORIZON
    o1 = np.full(P.shape, np.nan)
    o1[:, :-1] = P.o[:, 1:]
    ch = np.full(P.shape, np.nan)
    ch[:, :-h] = P.c[:, h:]
    no1 = np.full(len(P.cal), np.nan)
    no1[:-1] = P.nifty["o"][1:]
    nch = np.full(len(P.cal), np.nan)
    nch[:-h] = P.nifty["c"][h:]
    with np.errstate(invalid="ignore", divide="ignore"):
        ex = (ch / o1 - 1.0) - (nch / no1 - 1.0)[None, :]
    return np.where(np.isfinite(ex), (ex > 0).astype(float), np.nan)


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = ML_C, iters: int = 50) -> np.ndarray:
    """Newton-IRLS for 0.5 |w|^2 + C sum logloss (intercept unpenalised). X [n x k] -> w [k + 1] (last = intercept)."""
    n, k = X.shape
    A = np.hstack([X.astype(float), np.ones((n, 1))])
    w = np.zeros(k + 1)
    reg = np.ones(k + 1) / C
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y) + reg * w
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def ml_scores(ctx: Ctx, split_day: int = OLD_END) -> np.ndarray:
    """Predicted P(beat NIFTY) per cell. DISCOVERY (<= split_day): walk-forward by calendar year Y >= 2014, each year
    predicted by a model fitted on decision days whose label window closed before Y's first session. After split_day:
    ONE model fitted on every DISCOVERY decision day whose label window closed by split_day, frozen (never refitted)."""
    def build():
        P = ctx.daily
        X, elig = ml_matrix(ctx)
        y = ml_label(ctx)
        cal = P.cal
        years = cal // 10000
        score = np.full(P.shape, np.nan)
        h = ML_HORIZON
        tt = np.arange(len(cal))

        def fit_until(last_label_t):
            m = elig & np.isfinite(y) & (tt + h <= last_label_t)[None, :]
            si, ti = np.nonzero(m)
            return fit_logistic(X[:, si, ti].T, y[si, ti])

        def predict(w, tsel):
            si, ti = np.nonzero(elig & tsel[None, :])
            z = X[:, si, ti].T.astype(float) @ w[:-1] + w[-1]
            score[si, ti] = 1.0 / (1.0 + np.exp(-z))

        first = int(years[0]) + ML_FIRST_TRAIN_YEARS
        disc_end = int(np.searchsorted(cal, split_day, side="right")) - 1
        fits = {}
        for Y in range(first, int(split_day // 10000) + 1):
            t0 = int(np.searchsorted(cal, Y * 10000 + 101))
            w = fit_until(t0 - 1)
            fits[Y] = w.tolist()
            predict(w, (years == Y) & (tt <= disc_end))
        w = fit_until(disc_end)
        fits["frozen"] = w.tolist()
        predict(w, tt > disc_end)
        ctx._c["ml_fits"] = fits
        return score
    return ctx.get("ml_scores", build)


def r_ml(ctx: Ctx, mode: str):
    """M: rank F&O universe members by the model's probability; long = P > 0.5 (highest first), short = P < 0.5
    (lowest first), ls = both (strongest |P - 0.5| first)."""
    sc = ml_scores(ctx)
    if mode == "long":
        return _out(sc > 0.5, 1, sc - 0.5)
    if mode == "short":
        return _out(sc < 0.5, -1, 0.5 - sc)
    return _out(np.isfinite(sc) & (sc != 0.5), np.where(sc > 0.5, 1, -1), np.abs(sc - 0.5))
