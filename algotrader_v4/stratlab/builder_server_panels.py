"""
stratlab/builder_server_panels.py — the builder's SERVER data source: datagen_features panels built from the builder
DATA LAYER's files on the volume (REBUILD.md section 6; produced by builder_data.py / builder_minute.py, which this
module only reads):

  <root>/minute   forward_lab_store.Store: 1-minute bars 2016 -> of the lab universe + _NIFTY50_min / _NIFTYBANK_min
  <root>/nse      nsedata layout: delivery.csv.gz (sec_bhavdata_full EQ rows, official OHLC, prev_close, deliv_pct),
                  fo_parts/{fut,optidx}/YYYYMMDD.csv.gz + status.csv, ban.csv.gz, participant_oi.csv.gz,
                  bars/_NIFTY50_day.json, _NIFTYBANK_min.json, _VIX_day.json (NSE index closes)
  <root>/flow     deals/deals.csv.gz, corpactions/corpactions.csv.gz
  <root>/events   results.csv.gz

DAILY PANEL (the server's counterpart of events_daily's cache): every EQ symbol of delivery.csv.gz; prices adjusted
backwards for capital changes with NSE's own prev_close (factor prev_close_t / close_{t-1} when it differs by > 2 %);
point-in-time universe = the 500 largest by mean traded value (close x volume) over the previous 60 sessions (>= 40
observations; computed from data <= t only); an unexplained |open / previous close - 1| > 40 % is an exclusion day
(events_data.ca_excluded's rule, DailyPanel.excluded). F&O membership and the stock-futures OI change come from the
fo_parts/fut files (datagen_features.oi_change unchanged). Delivery %, ban, participant OI, deals, results timing and
corporate actions go through datagen_features' own functions with their file locations pointed at the data layer.
MINUTE PANEL: datagen_features.session_summary over Store.load_symbol for every stored symbol (no corrected overlay:
the store holds Kite's split-adjusted history), special sessions by the same median-bar rule, no dirty-link list.
Everything derived is cached on EPHEMERAL disk (BUILDER_CACHE_DIR), keyed by the input signature; nothing is written
to the volume. `upto` truncates every input at a session (forward runs).

make_synthetic_tree() writes a small synthetic data set in exactly this layout (tests).
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os

import numpy as np

from . import datagen_features as F
from .builder_source import file_id

TOP_N = 500
TV_WINDOW, TV_MIN = 60, 40
CA_FACTOR_TOL = 0.02
BAD_GAP = 0.40


DELIVERY_CHUNK = 500_000         # rows of the cash table read at a time (memory, not results)


def _cache_root() -> str:
    return os.environ.get("BUILDER_CACHE_DIR") or "/tmp/builder_cache"


def _sig(parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _bars_json(path: str, upto: int | None) -> dict | None:
    try:
        b = json.load(open(path))["bars"]
    except Exception:
        return None
    d = np.array([int(x[0][:10].replace("-", "")) for x in b], np.int64)
    a = np.array([[x[1], x[2], x[3], x[4], x[5] if len(x) > 5 else 0] for x in b], float).reshape(-1, 5)
    if upto is not None:
        k = d <= int(upto)
        d, a = d[k], a[k]
    last: dict = {}
    for i, dd in enumerate(d.tolist()):
        last[dd] = i                                  # minute files: keep the day's last bar for the close
    first: dict = {}
    for i, dd in enumerate(d.tolist()):
        first.setdefault(dd, i)
    days = np.array(sorted(last), np.int64)
    o = np.array([a[first[x], 0] for x in days])
    h = np.array([a[first[x]:last[x] + 1, 1].max() for x in days])
    l_ = np.array([a[first[x]:last[x] + 1, 2].min() for x in days])
    c = np.array([a[last[x], 3] for x in days])
    return {"date": days, "o": o, "h": h, "l": l_, "c": c, "v": np.zeros(len(days))}


UNIVERSE_FILE = "universe.json"


def read_universe_file(minute_root: str) -> dict | None:
    try:
        with open(os.path.join(minute_root, UNIVERSE_FILE)) as fh:
            return json.load(fh)
    except Exception:
        return None


def default_base() -> list:
    try:
        import forward_test
        return list(forward_test.UNIVERSE)
    except Exception:
        return []


def resolve_universe_lists(stored: list, doc: dict | None) -> dict:
    """The engine's universe lists from the stored minute names and the data layer's universe file (or None)."""
    stored = [s for s in stored if not s.startswith("_")]
    if doc and doc.get("base"):
        want = set(doc["base"])
        base = [s for s in stored if s in want] or list(doc["base"])
        extra = sorted(set(doc.get("extra") or []) | (set(stored) - set(base)))
        meta = {k: doc.get(k) for k in ("mode", "membership", "written", "sources") if k in doc}
        meta["source"] = "universe file"
        return {"base": base, "extra": [s for s in extra if s not in set(base)],
                "aliases": list(doc.get("aliases") or []), "meta": meta}
    want = set(default_base())
    base = [s for s in stored if s in want]
    if not base:
        return {"base": list(stored), "extra": [], "aliases": [], "meta": {"source": "all stored names"}}
    return {"base": base, "extra": [s for s in stored if s not in set(base)], "aliases": [],
            "meta": {"source": "stored names vs forward_test.UNIVERSE (no universe file)"}}


class ServerSource:
    kind = "server"

    def __init__(self, root: str | None = None, upto: int | None = None):
        if root is None:
            try:
                import builder_data
                root = str(builder_data.default_root())
            except Exception:
                root = os.environ.get("BUILDER_DATA_DIR", "data/builder_data")
        self.root = str(root)
        self.upto = upto
        self.nse = os.path.join(self.root, "nse")
        self.flow = os.path.join(self.root, "flow")
        self.events = os.path.join(self.root, "events")
        self.minute_root = os.path.join(self.root, "minute")
        self._ctx = None
        self._store = None

    # ── interface ──
    def store(self):
        if self._store is None:
            import forward_lab_store as FS
            self._store = FS.Store(self.minute_root)
        return self._store

    def minute_parts(self, sym: str) -> list:
        sd = self.store().load_symbol(sym, upto=self.upto)
        return [sd] if sd is not None else []

    def option_dirs(self) -> list:
        return [(self.nse, os.path.join(self.nse, "bars"), 0, int(self.upto) if self.upto else 99999999)]

    def nifty_daily(self) -> dict | None:
        return _bars_json(os.path.join(self.nse, "bars", "_NIFTY50_day.json"), self.upto)

    @property
    def end(self) -> int:
        c = self.ctx()
        return int(min(c.daily.cal[-1], c.minute.cal[-1]))

    def snapshot(self) -> dict:
        ins = {"delivery": os.path.join(self.nse, "delivery.csv.gz"),
               "fut_parts": os.path.join(self.nse, "fo_parts", "fut"),
               "optidx_parts": os.path.join(self.nse, "fo_parts", "optidx"),
               "fo_status": os.path.join(self.nse, "fo_parts", "status.csv"),
               "ban": os.path.join(self.nse, "ban.csv.gz"),
               "participant": os.path.join(self.nse, "participant_oi.csv.gz"),
               "index_bars": os.path.join(self.nse, "bars"),
               "deals": os.path.join(self.flow, "deals", "deals.csv.gz"),
               "corpactions": os.path.join(self.flow, "corpactions", "corpactions.csv.gz"),
               "results": os.path.join(self.events, "results.csv.gz"),
               "minute_manifest": os.path.join(self.minute_root, "manifest.json"),
               "minute_bars": os.path.join(self.minute_root, "bars")}
        if os.path.exists(os.path.join(self.minute_root, UNIVERSE_FILE)):
            ins["minute_universe"] = os.path.join(self.minute_root, UNIVERSE_FILE)   # (absent before the bg2 data layer)
        c = self.ctx()
        return {"source": "server", "root": self.root, "upto": self.upto, "end": self.end,
                "daily_calendar": [int(c.daily.cal[0]), int(c.daily.cal[-1])],
                "minute_calendar": [int(c.minute.cal[0]), int(c.minute.cal[-1])],
                "inputs": {k: file_id(v) for k, v in ins.items()}}

    def universe_lists(self) -> dict:
        """{"base", "extra", "aliases", "meta"}: the minute store's universe file (builder_universe.py) when present;
        else base = the stored names among forward_test.UNIVERSE (all stored names when none is), extra = the rest."""
        stored = [s for s in self.store().symbols() if not s.startswith("_")]
        return resolve_universe_lists(stored, read_universe_file(self.minute_root))

    # ── panels ──
    def ctx(self):
        if self._ctx is None:
            self._point_feature_paths()
            daily, extra = self.daily_panel()
            minute = self.minute_panel()
            ctx = F.Ctx(daily, minute)
            ctx._c["d:dp"] = extra["dp"]
            ctx._c["d:doi"] = extra["doi"]
            ctx._c["m:exp"] = extra["exp"]
            self._ctx = ctx
        return self._ctx

    def _point_feature_paths(self) -> None:
        """datagen_features reads its inputs from module paths: point them at the data layer (this process only)."""
        F.NSE_NEW = self.nse
        F.OLD_VIEW = self.nse
        F.NSE_OLD = self.nse
        F.FLOWS = self.flow
        F.EVENTS = self.events
        F.CACHE = os.path.join(_cache_root(), "datagen_server")
        if not os.path.exists(os.path.join(self.events, "raw", "ref", "symbolchange.csv")):
            F._CH.clear()
            F._CH.update({"list": [], "by": {}})

    def _delivery_frame(self):
        """The cash table (date, symbol, OHLC, prev_close, volume, deliv_pct), one row per (symbol, date). Read in
        chunks and without a Python string per row: at full scale (2012 ->, every listed stock: millions of rows) the
        one-shot read with per-row string work took several GB, and the snapshot process died at its address-space
        limit before any panel existed. Same rows and values as before."""
        import sys
        import pandas as pd
        cols = ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "deliv_pct"]
        parts, keys, ids = [], [], {}
        for ch in pd.read_csv(os.path.join(self.nse, "delivery.csv.gz"), dtype={"symbol": str}, usecols=cols,
                              chunksize=DELIVERY_CHUNK):
            d = ch["date"]
            if d.dtype.kind in "iu":
                dates = d.to_numpy(np.int64)
            else:                                        # "2012-01-02" style: convert each distinct value once
                codes, uniq = pd.factorize(d.astype(str))
                dates = np.array([int(str(u).replace("-", "")) for u in uniq], np.int64)[codes]
            ch["date"] = dates
            if self.upto is not None:
                ch = ch[ch["date"] <= int(self.upto)].copy()
            for k in cols[2:]:
                ch[k] = pd.to_numeric(ch[k], errors="coerce")
            # strip / upper the distinct names only; the column then points at those few shared (interned) strings
            codes, uniq = pd.factorize(ch["symbol"])
            names = np.array([sys.intern(str(u).strip().upper()) for u in uniq], dtype=object)
            sym = np.empty(len(ch), dtype=object)
            ok = codes >= 0
            sym[ok] = names[codes[ok]]
            sym[~ok] = np.nan
            ch["symbol"] = sym
            parts.append(ch)
            # an integer key per (symbol, date) row, to find duplicate rows without hashing strings (-1: no symbol)
            nid = np.array([ids.setdefault(n, len(ids)) for n in names], np.int64)
            sid = np.where(ok, nid[np.where(ok, codes, 0)] if len(nid) else 0, -1)
            keys.append(sid * 100_000_000 + ch["date"].to_numpy(np.int64))
        if not parts:
            return pd.DataFrame(columns=cols)
        df = pd.concat(parts, ignore_index=True)
        del parts
        dup = pd.Series(np.concatenate(keys)).duplicated(keep="last").to_numpy()
        del keys
        return df[~dup] if dup.any() else df             # (symbol, date) unique, the last row kept, as before

    def daily_panel(self):
        import pandas as pd
        sig = _sig([file_id(os.path.join(self.nse, "delivery.csv.gz")),
                    file_id(os.path.join(self.nse, "fo_parts", "fut")), self.upto, "dp1"])
        cpath = os.path.join(_cache_root(), f"server_daily_{sig}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=True)
            syms, cal = list(z["syms"]), z["cal"]
            arr = {k: z[k] for k in ("o", "h", "l", "c", "v", "dp", "doi")}
            member, fo = z["member"], z["fo"]
            excl = json.loads(str(z["excl"]))
        else:
            df = self._delivery_frame()
            cal = np.array(sorted(df["date"].unique()), np.int64)
            # symbol x session matrices by direct indexing (one cell per (symbol, date) row: the frame is unique on
            # that pair). The same numbers as DataFrame.pivot, without pivot's copies of a multi-million-row table
            ok = df["symbol"].notna().to_numpy()
            all_syms = sorted(set(df["symbol"].to_numpy()[ok]))
            six = pd.Series(np.arange(len(all_syms)), index=all_syms)
            si = six.reindex(df["symbol"].to_numpy()[ok]).to_numpy(np.int64)
            ti = np.searchsorted(cal, df["date"].to_numpy(np.int64)[ok])

            def grid(values, rows=None):
                """values per table row -> (symbols x sessions); rows = new row of each symbol (-1: left out)."""
                v = np.asarray(values, float)[ok]
                if rows is None:
                    out = np.full((len(all_syms), len(cal)), np.nan)
                    out[si, ti] = v
                    return out
                r = rows[si]
                take = r >= 0
                out = np.full((int((rows >= 0).sum()), len(cal)), np.nan)
                out[r[take], ti[take]] = v[take]
                return out
            tv = pd.DataFrame(grid(df["close"].to_numpy(float) * df["volume"].to_numpy(float)), index=all_syms,
                              columns=cal)
            m = tv.T.rolling(TV_WINDOW, min_periods=TV_MIN).mean().shift(1).T     # previous 60 sessions only
            rk = m.rank(axis=0, ascending=False, method="first")
            mem = (rk <= TOP_N).to_numpy()
            keep = mem.any(axis=1)
            syms = [s for s, k in zip(all_syms, keep) if k]
            member = mem[keep]
            row_of = np.full(len(all_syms), -1, np.int64)
            row_of[keep] = np.arange(int(keep.sum()))
            arr = {}
            for k, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume"),
                           ("pc", "prev_close"), ("dp", "deliv_pct")):
                arr[k] = grid(df[col].to_numpy(float), row_of)
            # backward adjustment for capital changes (NSE's adjusted prev_close)
            c_prev = np.full_like(arr["c"], np.nan)
            c_prev[:, 1:] = arr["c"][:, :-1]
            with np.errstate(invalid="ignore", divide="ignore"):
                f = arr["pc"] / c_prev
            f = np.where(np.isfinite(f) & (np.abs(f - 1.0) > CA_FACTOR_TOL), f, 1.0)
            cum = np.cumprod(f[:, ::-1], axis=1)[:, ::-1]                      # product of factors after t
            adj = np.ones_like(cum)
            adj[:, :-1] = cum[:, 1:]
            for k in ("o", "h", "l", "c"):
                arr[k] = arr[k] * adj
            arr["v"] = arr["v"] / adj
            excl = {}
            c_prev = np.full_like(arr["c"], np.nan)
            c_prev[:, 1:] = arr["c"][:, :-1]
            with np.errstate(invalid="ignore", divide="ignore"):
                gap = np.abs(arr["o"] / c_prev - 1.0)
            for i, t in zip(*np.nonzero(np.nan_to_num(gap, nan=0.0) > BAD_GAP)):
                excl.setdefault(str(int(i)), []).append(int(cal[t]))
            # free the cash table and its index arrays before the futures table is read
            del df, tv, m, rk, mem, c_prev, f, cum, adj, gap, si, ti, ok, six, row_of
            fo, doi = self._fut_panels(syms, cal)
            arr["doi"] = doi
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            np.savez_compressed(cpath, syms=np.array(syms, dtype=object), cal=cal, member=member, fo=fo,
                                excl=json.dumps(excl), **{k: arr[k] for k in ("o", "h", "l", "c", "v", "dp", "doi")})
        nf = self.nifty_daily()
        vx = _bars_json(os.path.join(self.nse, "bars", "_VIX_day.json"), self.upto)
        T = len(cal)

        def _align(a, key):
            out = np.full(T, np.nan)
            if a is None:
                return out
            j = np.searchsorted(cal, a["date"])
            ok = (j < T) & (cal[np.minimum(j, T - 1)] == a["date"])
            out[j[ok]] = a[key][ok]
            return out
        nifty = {"o": _align(nf, "o"), "c": _align(nf, "c")}
        vix = {"c": _align(vx, "c")}
        ex = {int(k): np.asarray(sorted(v), np.int64) for k, v in excl.items()}
        P = F.DailyPanel(syms, cal, arr["o"], arr["h"], arr["l"], arr["c"], arr["v"], np.asarray(member, bool),
                         np.asarray(fo, bool), nifty, vix, ex)
        return P, {"dp": np.asarray(arr["dp"], float), "doi": np.asarray(arr["doi"], float),
                   "exp": self._expiries()}

    def _fut_parts(self):
        """Every day's futures rows (day, exp, symbol, instrument, oi). Each small daily file is converted to numbers
        on its own and its names are interned, so the full table never holds a Python string per cell."""
        import sys
        import pandas as pd
        out = []
        for p in sorted(glob.glob(os.path.join(self.nse, "fo_parts", "fut", "*.csv.gz"))):
            d = int(os.path.basename(p)[:8])
            if self.upto is not None and d > int(self.upto):
                continue
            df = pd.read_csv(p, usecols=["date", "symbol", "expiry", "instrument", "oi"], dtype=str)
            out.append(pd.DataFrame({
                "day": df["date"].str.replace("-", "").astype(np.int64).to_numpy(),
                "exp": df["expiry"].str.replace("-", "").astype(np.int64).to_numpy(),
                "oi": pd.to_numeric(df["oi"], errors="coerce").fillna(0.0).to_numpy(),
                "symbol": np.array([sys.intern(x.strip().upper()) if isinstance(x, str) else x
                                    for x in df["symbol"]], dtype=object),
                "instrument": np.array([sys.intern(x) if isinstance(x, str) else x for x in df["instrument"]],
                                       dtype=object)}))
        if not out:
            return None
        return pd.concat(out, ignore_index=True)

    def _fut_panels(self, syms, cal):
        S, T = len(syms), len(cal)
        fo = np.zeros((S, T), bool)
        doi = np.full((S, T), np.nan)
        df = self._fut_parts()
        if df is None:
            return fo, doi
        stk = df[df["instrument"].isin(["FUTSTK", "STF"])]
        six = {s: i for i, s in enumerate(syms)}
        tix = {int(d): i for i, d in enumerate(cal)}
        fdays = sorted(stk["day"].unique())
        per = stk.groupby("day")["symbol"].apply(lambda x: set(x))
        for t, d in enumerate(cal):
            k = np.searchsorted(fdays, d, side="right")
            if k:
                for s in per[fdays[k - 1]]:
                    i = six.get(s)
                    if i is not None:
                        fo[i, t] = True
        g = F.oi_change(stk[["day", "symbol", "exp", "oi"]])
        for s, d, v in zip(g["symbol"], g["day"], g["doi"]):
            i, t = six.get(s), tix.get(int(d))
            if i is not None and t is not None:
                doi[i, t] = v
        return fo, doi

    def _expiries(self) -> dict:
        """Index expiry sessions: days on which an index option traded on its own expiry date (optidx parts)."""
        import pandas as pd
        acc = {"NIFTY": set(), "BANKNIFTY": set()}
        for p in sorted(glob.glob(os.path.join(self.nse, "fo_parts", "optidx", "*.csv.gz"))):
            d = int(os.path.basename(p)[:8])
            if self.upto is not None and d > int(self.upto):
                continue
            df = pd.read_csv(p, usecols=["date", "symbol", "expiry"], dtype=str)
            df = df[(df["date"] == df["expiry"]) & df["symbol"].isin(list(acc))]
            for s in set(df["symbol"]):
                acc[s].add(d)
        return {k: sorted(v) for k, v in acc.items()}

    def minute_panel(self):
        st = self.store()
        names = [s for s in st.symbols() if not s.startswith("_")]
        names = sorted(names) + [x for x in F.INDEX_MIN if x in set(st.symbols())]
        man = os.path.join(self.minute_root, "manifest.json")
        base = set(self.universe_lists()["base"])
        sig = _sig([file_id(man), file_id(os.path.join(self.minute_root, "bars")), self.upto, names, sorted(base),
                    "mp2"])
        cpath = os.path.join(_cache_root(), f"server_minsum_{sig}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=True)
            syms, cal = list(z["syms"]), z["cal"]
            fields = {k: z[k] for k in F.MIN_FIELDS}
            special = set(int(x) for x in z["special"])
            return F.MinutePanel(syms, cal, fields, special, {})
        per = {}
        for s in names:
            sd = st.load_symbol(s, upto=self.upto)
            if sd is not None:
                per[s] = F.session_summary(sd)
        nidx = per.get("_NIFTY50_min")
        cal = set()
        for v in per.values():
            cal.update(int(x) for x in v["days"])
        cal = np.asarray(sorted(cal), np.int64)
        syms = [s for s in names if s in per and not s.startswith("_")] + [x for x in F.INDEX_MIN if x in per]
        S, T = len(syms), len(cal)
        fields = {k: np.full((S, T), np.nan) for k in F.MIN_FIELDS}
        for i, s in enumerate(syms):
            z = per[s]
            j = np.searchsorted(cal, z["days"])
            for k in F.MIN_FIELDS:
                fields[k][i, j] = z[k]
        # special sessions from the BASE names only (the 97): names added later (NIFTY 200, recent listings with no
        # bars before their listing) never change which sessions are special
        brows = [i for i, s in enumerate(syms) if not s.startswith("_") and s in base]
        if not brows:
            brows = [i for i, s in enumerate(syms) if not s.startswith("_")]
        nst = len(brows)
        nbs = fields["nb"][brows]
        med = np.nanmedian(np.where(np.isfinite(nbs), nbs, 0.0), axis=0) if nst else np.zeros(T)
        special = {int(d) for d, m in zip(cal, med) if m < F.SPECIAL_MEDIAN_BARS}
        del nidx
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.savez_compressed(cpath, syms=np.array(syms, dtype=object), cal=cal, special=np.array(sorted(special)),
                            **fields)
        return F.MinutePanel(syms, cal, fields, special, {})


class SyntheticSource(ServerSource):
    """A ServerSource over make_synthetic_tree(root) (built on first use)."""
    kind = "synthetic"

    def __init__(self, root: str, upto: int | None = None, **kw):
        if not os.path.exists(os.path.join(root, "nse", "delivery.csv.gz")):
            make_synthetic_tree(root, **kw)
        super().__init__(root, upto)


# ══════════════════════════════════════════════════════════════════════════
# synthetic data in the data layer's layout (tests)
# ══════════════════════════════════════════════════════════════════════════
def _weekdays(start: int, end: int) -> list:
    from datetime import date, timedelta
    d = date(start // 10000, start // 100 % 100, start % 100)
    e = date(end // 10000, end // 100 % 100, end % 100)
    out = []
    while d <= e:
        if d.weekday() < 5:
            out.append(d.year * 10000 + d.month * 100 + d.day)
        d += timedelta(days=1)
    return out


def _iso(d: int) -> str:
    return f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}"


def make_synthetic_tree(root: str, start: int = 20210104, end: int = 20261109, n_stocks: int = 6,
                        seed: int = 11, bars: int = 375, n_extra: int = 0, late: dict | None = None) -> dict:
    """Random-walk market in the builder data layer's layout: minute store, delivery, fo_parts (stock + index
    futures, index options with Black-Scholes prices at 15 % vol), ban, participant OI, index bars, deals,
    corporate actions, results, and the minute store's universe file (the last `n_extra` stocks are the "added"
    names of the NIFTY200 universe; `late` = {stock: first day}: that stock's minute bars start late, i.e. it has
    incomplete minute data). Deterministic in `seed`."""
    import math
    import forward_lab_store as FS
    rng = np.random.default_rng(seed)
    days = _weekdays(start, end)
    T = len(days)
    stocks = [f"SYN{k:02d}" for k in range(n_stocks)]
    t_min = np.arange(555, 555 + bars, dtype=np.int16)
    store = FS.Store(os.path.join(root, "minute"))
    os.makedirs(os.path.join(root, "nse", "bars"), exist_ok=True)
    daily = {}

    def walk(name, p0, vol_d):
        r = rng.normal(0, vol_d / math.sqrt(bars), (T, bars))
        gaps = rng.normal(0, vol_d * 0.5, T)
        lc = math.log(p0)
        o_ = np.empty((T, bars))
        for t in range(T):
            path = lc + gaps[t] + np.cumsum(r[t])
            o_[t] = np.exp(np.r_[lc + gaps[t], path[:-1]])
            lc = path[-1]
        c_ = np.empty_like(o_)
        c_[:, :-1] = o_[:, 1:]
        c_[:, -1] = o_[:, -1] * np.exp(r[:, -1])
        hi = np.maximum(o_, c_) * (1 + np.abs(rng.normal(0, 0.0005, o_.shape)))
        lo = np.minimum(o_, c_) * (1 - np.abs(rng.normal(0, 0.0005, o_.shape)))
        v = rng.integers(1000, 5000, o_.shape).astype(float)
        z = {"date": np.repeat(np.array(days, np.int32), bars), "t": np.tile(t_min, T),
             "o": np.round(o_.ravel(), 2), "h": np.round(hi.ravel(), 2), "l": np.round(lo.ravel(), 2),
             "c": np.round(c_.ravel(), 2), "v": v.ravel()}
        z["h"] = np.maximum.reduce([z["h"], z["o"], z["c"]])
        z["l"] = np.minimum.reduce([z["l"], z["o"], z["c"]])
        if late and name in late:
            keep = z["date"] >= int(late[name])
            store.write_sessions(name, {k: v[keep] for k, v in z.items()})
        else:
            store.write_sessions(name, z)
        daily[name] = {"o": z["o"].reshape(T, bars)[:, 0], "h": z["h"].reshape(T, bars).max(1),
                       "l": z["l"].reshape(T, bars).min(1), "c": z["c"].reshape(T, bars)[:, -1],
                       "v": v.sum(1)}

    for k, s in enumerate(stocks):
        walk(s, 200.0 + 50 * k, 0.018)
    walk("_NIFTY50_min", 15000.0, 0.009)
    walk("_NIFTYBANK_min", 35000.0, 0.011)
    vix = np.clip(15 * np.exp(np.cumsum(rng.normal(0, 0.05, T)) - np.linspace(0, 0, T)), 9, 45)

    def bars_json(name, o, h, l, c, sym):
        b = [[f"{_iso(d)}T00:00:00", float(o[i]), float(h[i]), float(l[i]), float(c[i]), 0] for i, d in enumerate(days)]
        with open(os.path.join(root, "nse", "bars", name), "w") as fh:
            json.dump({"symbol": sym, "interval": "day", "bars": b}, fh)
    n = daily["_NIFTY50_min"]
    bars_json("_NIFTY50_day.json", n["o"], n["h"], n["l"], n["c"], "NIFTY 50")
    bn = daily["_NIFTYBANK_min"]
    bars_json("_NIFTYBANK_min.json", bn["o"], bn["h"], bn["l"], bn["c"], "NIFTY BANK")
    bars_json("_VIX_day.json", vix, vix, vix, vix, "INDIA VIX")
    # delivery (official daily OHLC = the minute bars' day)
    with gzip.open(os.path.join(root, "nse", "delivery.csv.gz"), "wt") as fh:
        fh.write("date,symbol,open,high,low,close,prev_close,avg_price,volume,turnover_lacs,trades,deliv_qty,"
                 "deliv_pct,bars_symbol\n")
        for i, d in enumerate(days):
            for s in stocks:
                x = daily[s]
                pc = x["c"][i - 1] if i else x["o"][i]
                dp = float(np.clip(rng.normal(45, 12), 5, 95))
                fh.write(f"{d},{s},{x['o'][i]:.2f},{x['h'][i]:.2f},{x['l'][i]:.2f},{x['c'][i]:.2f},{pc:.2f},"
                         f"{x['c'][i]:.2f},{x['v'][i]:.0f},{x['v'][i] * x['c'][i] / 1e5:.2f},100,"
                         f"{x['v'][i] * dp / 100:.0f},{dp:.2f},{s}\n")
    # F&O parts: stock + index futures, index options (weekly Thursday expiries)
    for sub in ("fut", "optidx"):
        os.makedirs(os.path.join(root, "nse", "fo_parts", sub), exist_ok=True)
    exps = [d for d in days if __import__("datetime").date(d // 10000, d // 100 % 100, d % 100).weekday() == 3]
    from statistics import NormalDist
    N = NormalDist()

    def bs(S, K, tau, typ, vol=0.15):
        if tau <= 0:
            return max(0.0, S - K) if typ == "CE" else max(0.0, K - S)
        d1 = (math.log(S / K) + 0.5 * vol * vol * tau) / (vol * math.sqrt(tau))
        d2 = d1 - vol * math.sqrt(tau)
        return S * N.cdf(d1) - K * N.cdf(d2) if typ == "CE" else K * N.cdf(-d2) - S * N.cdf(-d1)
    status = ["date,status,fmt,n_raw,n_fut,n_optidx,n_optstk_agg"]
    oi_state = {s: 1_000_000.0 for s in stocks}
    for i, d in enumerate(days):
        alive = [e for e in exps if e >= d][:2]
        futl = ["date,symbol,expiry,instrument,open,high,low,close,settle,contracts,oi,chg_oi,underlying,lot"]
        for s in stocks:
            oi_state[s] *= float(np.exp(rng.normal(0, 0.08)))
            px = daily[s]["c"][i]
            for e in alive:
                futl.append(f"{_iso(d)},{s},{_iso(e)},FUTSTK,{px:.2f},{px:.2f},{px:.2f},{px:.2f},{px:.2f},100,"
                            f"{oi_state[s] / len(alive):.0f},0,{px:.2f},500")
        optl = ["date,symbol,expiry,strike,type,open,high,low,close,settle,contracts,oi,chg_oi,underlying,lot"]
        for sym, key, step in (("NIFTY", "_NIFTY50_min", 50.0), ("BANKNIFTY", "_NIFTYBANK_min", 100.0)):
            S = daily[key]["c"][i]
            for e in alive:
                futl.append(f"{_iso(d)},{sym},{_iso(e)},FUTIDX,{S:.2f},{S:.2f},{S:.2f},{S:.2f},{S:.2f},1000,1000,0,"
                            f"{S:.2f},50")
                from datetime import date as _dt
                tau = (_dt(e // 10000, e // 100 % 100, e % 100) - _dt(d // 10000, d // 100 % 100, d % 100)).days / 365
                k0 = round(S / step) * step
                for kk in np.arange(k0 - 10 * step, k0 + 10.5 * step, step):
                    for typ in ("CE", "PE"):
                        p = max(0.05, round(bs(S, float(kk), tau, typ), 2))
                        optl.append(f"{_iso(d)},{sym},{_iso(e)},{kk:.1f},{typ},{p},{p},{p},{p},{p},50,5000,0,"
                                    f"{S:.2f},50")
        for sub, lines in (("fut", futl), ("optidx", optl)):
            with gzip.open(os.path.join(root, "nse", "fo_parts", sub, f"{d}.csv.gz"), "wt") as fh:
                fh.write("\n".join(lines) + "\n")
        status.append(f"{_iso(d)},ok,udiff,0,0,0,0")
    with open(os.path.join(root, "nse", "fo_parts", "status.csv"), "w") as fh:
        fh.write("\n".join(status) + "\n")
    with gzip.open(os.path.join(root, "nse", "ban.csv.gz"), "wt") as fh:
        fh.write("date,symbol\n")
        for i, d in enumerate(days):
            if rng.random() < 0.03:
                fh.write(f"{_iso(d)},{stocks[int(rng.integers(0, n_stocks))]}\n")
    with gzip.open(os.path.join(root, "nse", "participant_oi.csv.gz"), "wt") as fh:
        cols = ["fut_idx_long", "fut_idx_short", "fut_stk_long", "fut_stk_short"]
        fh.write("date,client_type," + ",".join(cols) + "\n")
        L = {w: 100000.0 for w in ("Client", "FII", "DII", "Pro")}
        for d in days:
            for w in L:
                L[w] *= float(np.exp(rng.normal(0, 0.04)))
                fh.write(f"{_iso(d)},{w},{L[w]:.0f},{L[w] * float(np.exp(rng.normal(0, 0.2))):.0f},1000,1000\n")
    os.makedirs(os.path.join(root, "flow", "deals"), exist_ok=True)
    os.makedirs(os.path.join(root, "flow", "corpactions"), exist_ok=True)
    os.makedirs(os.path.join(root, "events"), exist_ok=True)
    with gzip.open(os.path.join(root, "flow", "deals", "deals.csv.gz"), "wt") as fh:
        fh.write("date,kind,symbol,lab_symbol,in_lab,security,client,side,qty,price,value_cr,remarks\n")
        for d in days:
            if rng.random() < 0.05:
                s = stocks[int(rng.integers(0, n_stocks))]
                who = "ALPHA MUTUAL FUND" if rng.random() < 0.5 else "RAMESH KUMAR"
                fh.write(f"{d},bulk,{s},{s},1,{s} Ltd,{who},{'BUY' if rng.random() < 0.5 else 'SELL'},100000,200,"
                         f"{rng.uniform(5, 40):.2f},\n")
    with gzip.open(os.path.join(root, "flow", "corpactions", "corpactions.csv.gz"), "wt") as fh:
        fh.write("ex_date,symbol,symbol_then,lab_symbol,in_lab,series,isin,face_value,kind,ratio_a,ratio_b,fv_from,"
                 "fv_to,dividend,rights_premium,factor,record_date,first_listed,known_date,subject\n")
    with gzip.open(os.path.join(root, "events", "results.csv.gz"), "wt") as fh:
        fh.write("symbol,isin,period_end,relating_to,first_ts,src_first,fr_ts,ann_ts,n_fr,n_ann\n")
        for s in stocks:
            for i in range(10, T, 63):
                ts = f"{_iso(days[i])} 17:30:00"
                fh.write(f"{s},X,{_iso(days[i])},Q,{ts},ann,{ts},{ts},1,1\n")
    nb = n_stocks - int(n_extra)
    with open(os.path.join(root, "minute", UNIVERSE_FILE), "w") as fh:
        json.dump({"version": 1, "mode": "nifty200", "base": stocks[:nb], "extra": stocks[nb:], "aliases": [],
                   "missing_on_kite": [], "membership": "synthetic", "sources": {}}, fh)
    return {"days": T, "stocks": stocks, "base": stocks[:nb], "extra": stocks[nb:]}
