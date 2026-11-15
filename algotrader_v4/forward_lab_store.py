"""
forward_lab_store.py — the forward lab's bar store: the lab's 1-minute bars, kept on the volume, one session at a time.

WHAT IS STORED (the exact list the lab uses, `forward_test.UNIVERSE` = the 97 NIFTY-100 symbols of bars3y, plus
`_NIFTY50_min`, `_NIFTYBANK_min` and the daily series `_NIFTY50_day`, `_VIX_day`):
  <root>/bars/<SYMBOL>/<YYYYMM>.npz   1-minute bars of the sessions of one month, np.savez_compressed (deflate),
                                      prices as integer paise (exact: every NSE price has <= 2 decimals; a file falls
                                      back to float64 when a price is not a whole paisa), the close as a running
                                      difference, open / high / low as offsets from the close, volume as int64
  <root>/aux/<NAME>.npz               the two daily series in the lab's cache format (date o h l c v)
  <root>/manifest.json                sessions written, bootstrap, corporate-action refetches
Bars are parsed and cleaned by the lab's own functions (`stratlab.data._parse_bars` + `_clean_minute`: 09:15-15:29,
finite positive prices, sorted, duplicate minutes dropped) and loaded back through `stratlab.data._sessionize` (the
lab's < 200-bar session rule), so `load_symbol()` returns the same SymbolData the lab's loader builds from its cache.

WARM-UP: the store is ANCHORED. The first run fetches WARMUP_SESSIONS (420) sessions of minute history before the
first forward session; afterwards one session is appended per trading day and nothing is ever deleted, so every
indicator (EMA / RSI / ATR / VWAP run continuously from the store's first bar, as in the lab) is a pure function of the
bars up to the session being run: a session computed today or re-computed later gives the same result.
Daily bars of the positional strategies are derived from the minute bars (`SymbolData.daily()`), exactly as in the
lab (Kite's daily candles carry the official close, the lab's daily close is the last 1-minute close): 420 sessions
of minute history = 420 daily candles. Only the NIFTY 50 / INDIA VIX daily series (`_NIFTY50_day`, `_VIX_day`) are
Kite day candles, as in the lab's data set.

COMPLETENESS (pre-registered in forward_lab_rule.json "data"): a session is COMPLETE when every instrument of the
universe that Kite lists has bars that day with >= MIN_COMPLETE_BARS bars, its first bar at 09:15 and its last bar at
or after LAST_BAR_MIN (15:14), and both daily series have that day. Anything else (missing symbol, short session,
Kite failure) = INCOMPLETE; a holiday / special session is never run.

CORPORATE ACTIONS (the lab's convention: as-traded prices adjusted for capital changes only, dividends not adjusted;
Kite's history is adjusted for splits / bonuses): every fetch re-reads the previous stored session; when Kite's bars
for it differ from the stored ones by more than CA_TOLERANCE (median close ratio), Kite has back-adjusted its history
(a capital change went ex), and the symbol's WHOLE stored window is refetched from Kite and replaced. The same refetch
is forced when the NSE corporate-action calendar (event_calendar) lists a price-adjusting action with ex-date on the
session. Booked forward trades are never changed by a refetch; open virtual positions are re-simulated on the
adjusted bars.

DISK CAP: FORWARD_LAB_MAX_MB (default 300) for the whole forward-lab directory; at or above it nothing more is written
(the run is reported as "disk cap reached"); nothing is ever deleted.
"""
from __future__ import annotations

import io
import json
import os
import threading
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from stratlab import data as _SD

UNIVERSE_STOCKS: tuple[str, ...] = tuple(__import__("forward_test").UNIVERSE)
INDEX_MIN = {"_NIFTY50_min": "NIFTY 50", "_NIFTYBANK_min": "NIFTY BANK"}       # store name -> Kite NSE index
DAILY_AUX = {"_NIFTY50_day": "NIFTY 50", "_VIX_day": "INDIA VIX"}
MINUTE_SYMBOLS: tuple[str, ...] = UNIVERSE_STOCKS + tuple(INDEX_MIN)
WARMUP_SESSIONS = 420
MIN_COMPLETE_BARS = 360
FIRST_BAR_MIN = 9 * 60 + 15
LAST_BAR_MIN = 15 * 60 + 14
CA_TOLERANCE = 0.0005           # 0.05 % median close ratio between the stored and the re-fetched previous session
DEFAULT_MAX_MB = 300.0
_LOCK = threading.RLock()


# ── paths / disk ──────────────────────────────────────────────────────────────────────────────────────────────────
def default_root() -> Path:
    """FORWARD_LAB_DIR, else <dir of PERFORMANCE_FILE>/forward_lab (the Railway volume), else ./data/forward_lab."""
    p = os.getenv("FORWARD_LAB_DIR")
    if p:
        return Path(p)
    perf = os.getenv("PERFORMANCE_FILE")
    base = Path(perf).parent if perf else Path(__file__).resolve().parent / "data"
    return base / "forward_lab"


def max_bytes() -> int:
    try:
        from config import settings
        mb = float(getattr(settings, "forward_lab_max_mb", DEFAULT_MAX_MB))
    except Exception:
        mb = float(os.getenv("FORWARD_LAB_MAX_MB", DEFAULT_MAX_MB))
    return int(mb * 1024 * 1024)


def dir_bytes(root) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


class DiskCapReached(RuntimeError):
    pass


# ── encoding ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _paise(x: np.ndarray):
    p = np.round(np.asarray(x, float) * 100.0)
    if not np.all(np.isfinite(p)) or np.any(np.abs(p) > 2 ** 52):
        return None
    p = p.astype(np.int64)
    return p if np.array_equal(p / 100.0, np.asarray(x, float)) else None


def encode(z: dict) -> dict:
    """Minute arrays {date, t, o, h, l, c, v} (one or more whole sessions, sorted) -> compact arrays."""
    date_ = np.asarray(z["date"], np.int32)
    if len(date_):
        cut = np.flatnonzero(date_[1:] != date_[:-1]) + 1
        st = np.concatenate(([0], cut))
        days, nb = date_[st], np.diff(np.concatenate((st, [len(date_)])))
    else:
        days, nb = np.zeros(0, np.int32), np.zeros(0, np.int64)
    out = {"days": days.astype(np.int32), "n": nb.astype(np.int32), "t": np.asarray(z["t"], np.int16)}
    P = {k: _paise(z[k]) for k in "ohlc"}
    if all(v is not None for v in P.values()):
        c = P["c"]
        out["dc"] = np.diff(c, prepend=0).astype(np.int64)
        for k in "ohl":
            out["d" + k] = (P[k] - c).astype(np.int64)
    else:
        for k in "ohlc":
            out["f" + k] = np.asarray(z[k], np.float64)
    out["v"] = np.asarray(z["v"], np.float64)
    if np.array_equal(np.round(out["v"]), out["v"]) and np.all(np.abs(out["v"]) < 2 ** 62):
        out["v"] = out["v"].astype(np.int64)
    return out


def decode(e) -> dict:
    days, nb = np.asarray(e["days"]), np.asarray(e["n"])
    z = {"date": np.repeat(days.astype(np.int32), nb.astype(np.int64)), "t": np.asarray(e["t"], np.int16)}
    if "dc" in e:
        c = np.cumsum(np.asarray(e["dc"], np.int64))
        z["c"] = c / 100.0
        for k in "ohl":
            z[k] = (np.asarray(e["d" + k], np.int64) + c) / 100.0
    else:
        for k in "ohlc":
            z[k] = np.asarray(e["f" + k], np.float64)
    z["v"] = np.asarray(e["v"]).astype(np.float64)
    return z


def kite_rows_to_lab(rows: list[dict]) -> list[list]:
    """Kite candles {"date": datetime, open, high, low, close, volume} -> the lab's bar rows [iso_ts, o, h, l, c, v]
    (IST wall clock, as the lab's bar files)."""
    out = []
    for r in rows or []:
        d = r.get("date")
        if isinstance(d, str):
            d = datetime.fromisoformat(d)
        if d is None:
            continue
        if getattr(d, "tzinfo", None) is not None:
            from datetime import timezone
            d = d.astimezone(timezone(timedelta(hours=5, minutes=30))).replace(tzinfo=None)
        out.append([d.strftime("%Y-%m-%dT%H:%M:%S"), r.get("open"), r.get("high"), r.get("low"), r.get("close"),
                    r.get("volume") or 0])
    return out


def clean_minute_rows(rows: list[list]) -> Optional[dict]:
    """The lab's parse + clean of 1-minute bar rows (stratlab.data._parse_bars / _clean_minute)."""
    p = _SD._parse_bars(rows)
    return None if p is None else _SD._clean_minute(p)


def daily_rows(rows: list[list]) -> Optional[dict]:
    """Daily rows -> the lab's daily cache arrays (date o h l c v), as stratlab.data.prepare stores `_*_day` files."""
    p = _SD._parse_bars(rows)
    if p is None:
        return None
    order = np.argsort(p["date"], kind="stable")
    d, px = p["date"][order], p["px"][order]
    keep = np.ones(len(d), bool)
    keep[:-1] = d[1:] != d[:-1]
    d, px = d[keep], px[keep]
    return {"date": d, "o": px[:, 0], "h": px[:, 1], "l": px[:, 2], "c": px[:, 3], "v": px[:, 4]}


# ── the store ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _npz_bytes(arrs: dict, compressed: bool = True) -> bytes:
    b = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(b, **arrs)
    return b.getvalue()


class Store:
    def __init__(self, root=None, cap_root=None) -> None:
        self.root = Path(root) if root is not None else default_root() / "bars"
        self.cap_root = Path(cap_root) if cap_root is not None else self.root.parent

    # ── manifest ──
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text())
        except Exception:
            return {"sessions": {}, "refetch": [], "bootstrap": None}

    def _save_manifest(self, m: dict) -> None:
        _atomic_bytes(self.manifest_path, json.dumps(m, sort_keys=True, indent=0).encode())

    def sessions(self) -> list[int]:
        return sorted(int(d) for d in self.manifest().get("sessions", {}))

    def last_session(self) -> Optional[int]:
        s = self.sessions()
        return s[-1] if s else None

    def session_status(self, day: int) -> Optional[str]:
        return (self.manifest().get("sessions", {}).get(str(int(day))) or {}).get("status")

    def check_cap(self, extra: int = 0) -> None:
        used = dir_bytes(self.cap_root)
        if used + extra >= max_bytes():
            raise DiskCapReached(f"disk cap reached ({used / 1048576:.1f} MB >= {max_bytes() / 1048576:.0f} MB)")

    # ── minute bars ──
    def _sym_dir(self, sym: str) -> Path:
        return self.root / "bars" / sym.replace("/", "_").replace(" ", "_")

    def symbols(self) -> list[str]:
        d = self.root / "bars"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []

    def _read_month(self, path: Path) -> dict:
        with np.load(path) as e:
            return decode({k: e[k] for k in e.files})

    def raw(self, sym: str, upto: Optional[int] = None, lo: Optional[int] = None) -> dict:
        """Flat minute arrays of the stored sessions (date <= upto, >= lo)."""
        parts = []
        d = self._sym_dir(sym)
        if d.exists():
            for f in sorted(d.glob("*.npz")):
                ym = int(f.stem)
                if upto is not None and ym > int(upto) // 100:
                    continue
                if lo is not None and ym < int(lo) // 100:
                    continue
                parts.append(self._read_month(f))
        keys = ("date", "t", "o", "h", "l", "c", "v")
        if not parts:
            return {k: np.zeros(0, np.int32 if k == "date" else (np.int16 if k == "t" else float)) for k in keys}
        z = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        keep = np.ones(len(z["date"]), bool)
        if upto is not None:
            keep &= z["date"] <= int(upto)
        if lo is not None:
            keep &= z["date"] >= int(lo)
        return {k: v[keep] for k, v in z.items()}

    def load_symbol(self, sym: str, upto: Optional[int] = None, min_bars: int = _SD.MIN_SESSION_BARS):
        """SymbolData of the stored sessions <= upto, exactly as stratlab.data.load_symbol builds it."""
        z = self.raw(sym, upto)
        if len(z["date"]) == 0:
            return None
        sd = _SD._sessionize(sym, z, min_bars)
        return sd if sd.n_sessions else None

    def write_sessions(self, sym: str, z: dict) -> int:
        """Replace / add the sessions present in `z` (clean minute arrays) for `sym`; returns bytes written."""
        if z is None or len(z["date"]) == 0:
            return 0
        written = 0
        months = sorted(set(int(x) // 100 for x in np.unique(z["date"])))
        with _LOCK:
            for ym in months:
                path = self._sym_dir(sym) / f"{ym}.npz"
                new = {k: v[(z["date"] // 100) == ym] for k, v in z.items()}
                if path.exists():
                    old = self._read_month(path)
                    keep = ~np.isin(old["date"], np.unique(new["date"]))
                    merged = {k: np.concatenate([old[k][keep], new[k]]) for k in new}
                else:
                    merged = new
                key = merged["date"].astype(np.int64) * 10000 + merged["t"].astype(np.int64)
                order = np.argsort(key, kind="stable")
                merged = {k: v[order] for k, v in merged.items()}
                data = _npz_bytes(encode(merged))
                _atomic_bytes(path, data)
                written += len(data)
        return written

    def replace_symbol(self, sym: str, z: dict) -> int:
        """Corporate-action refetch: the symbol's stored window is replaced by `z` (months outside z are kept only
        when z does not cover them; the refetch always covers the stored window)."""
        with _LOCK:
            d = self._sym_dir(sym)
            covered = set(int(x) // 100 for x in np.unique(z["date"])) if len(z["date"]) else set()
            if d.exists():
                for f in d.glob("*.npz"):
                    if int(f.stem) in covered:
                        f.unlink()
            return self.write_sessions(sym, z)

    # ── daily series ──
    def aux_path(self, name: str) -> Path:
        return self.root / "aux" / f"{name}.npz"

    def load_aux(self, name: str, upto: Optional[int] = None) -> Optional[dict]:
        p = self.aux_path(name)
        if not p.exists():
            return None
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        if upto is not None:
            keep = d["date"] <= int(upto)
            d = {k: v[keep] for k, v in d.items()}
        return d

    def write_aux(self, name: str, new: dict) -> int:
        if new is None or len(new["date"]) == 0:
            return 0
        with _LOCK:
            old = self.load_aux(name)
            if old is not None:
                keep = ~np.isin(old["date"], new["date"])
                merged = {k: np.concatenate([old[k][keep], np.asarray(new[k])]) for k in ("date", "o", "h", "l", "c", "v")}
            else:
                merged = {k: np.asarray(new[k]) for k in ("date", "o", "h", "l", "c", "v")}
            order = np.argsort(merged["date"], kind="stable")
            merged = {k: v[order] for k, v in merged.items()}
            merged["date"] = merged["date"].astype(np.int32)
            data = _npz_bytes(merged, compressed=False)        # the lab's cache format (np.savez)
            _atomic_bytes(self.aux_path(name), data)
            return len(data)

    def write_lab_cache(self, cache_dir, upto: int) -> str:
        """A lab-format cache directory holding the daily series truncated at `upto` (what sim.run_symbol reads
        for positional strategies). Nothing after `upto` is ever copied."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name in DAILY_AUX:
            a = self.load_aux(name, upto)
            if a is not None:
                np.savez(cache_dir / f"{name}.npz", **a)
        return str(cache_dir)

    # ── session bookkeeping ──
    def mark_session(self, day: int, status: str, **info) -> None:
        with _LOCK:
            m = self.manifest()
            m.setdefault("sessions", {})[str(int(day))] = {"status": status, **info}
            self._save_manifest(m)

    def note(self, key: str, value) -> None:
        with _LOCK:
            m = self.manifest()
            if isinstance(m.get(key), list):
                m[key].append(value)
            else:
                m[key] = value
            self._save_manifest(m)


# ── completeness ──────────────────────────────────────────────────────────────────────────────────────────────────
def session_complete(day: int, minute: dict, daily: dict, expected: list[str]) -> tuple[bool, list[str]]:
    """minute: {symbol: clean arrays for the day or None}; daily: {name: arrays or None}. (complete, reasons)."""
    reasons = []
    for sym in expected:
        z = minute.get(sym)
        if z is None:
            reasons.append(f"{sym}: no data")
            continue
        sel = z["date"] == int(day)
        n = int(sel.sum())
        if n == 0:
            reasons.append(f"{sym}: no bars on {day}")
            continue
        t = z["t"][sel]
        if n < MIN_COMPLETE_BARS:
            reasons.append(f"{sym}: short session ({n} bars)")
        elif int(t[0]) != FIRST_BAR_MIN or int(t[-1]) < LAST_BAR_MIN:
            reasons.append(f"{sym}: session {int(t[0]) // 60:02d}:{int(t[0]) % 60:02d}-"
                           f"{int(t[-1]) // 60:02d}:{int(t[-1]) % 60:02d}")
    for name in DAILY_AUX:
        a = daily.get(name)
        if a is None or int(day) not in set(int(x) for x in a["date"]):
            reasons.append(f"{name}: no daily candle on {day}")
    return (not reasons), reasons


def ca_changed(stored: dict, fetched: dict, day: int, tol: float = CA_TOLERANCE) -> bool:
    """True when Kite's bars for an already stored session differ from the stored ones (median close ratio off by
    more than `tol`): Kite back-adjusted its history (a split / bonus / other capital change went ex)."""
    a = stored["date"] == int(day)
    b = fetched["date"] == int(day)
    if not a.any() or not b.any():
        return False
    ka = dict(zip(stored["t"][a].tolist(), stored["c"][a].tolist()))
    kb = dict(zip(fetched["t"][b].tolist(), fetched["c"][b].tolist()))
    common = sorted(set(ka) & set(kb))
    if len(common) < 10:
        return False
    r = np.array([kb[t] / ka[t] for t in common if ka[t] > 0])
    return bool(len(r) and abs(float(np.median(r)) - 1.0) > tol)


# ── Kite fetch (server process only; read-only history calls through kite_client's limiter) ─────────────────────────
def _ymd(d) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def _d(x: int) -> date:
    return date(int(x) // 10000, int(x) // 100 % 100, int(x) % 100)


def instrument_tokens(kite) -> dict:
    """{store name: instrument token} for the universe (stocks, the two indices, VIX)."""
    names = list(UNIVERSE_STOCKS) + list(set(INDEX_MIN.values()) | set(DAILY_AUX.values()))
    toks = kite.get_instrument_tokens(names, "NSE") or {}
    out = {s: toks[s] for s in UNIVERSE_STOCKS if s in toks}
    for k, v in INDEX_MIN.items():
        if v in toks:
            out[k] = toks[v]
    for k, v in DAILY_AUX.items():
        if v in toks:
            out[k] = toks[v]
    return out


def fetch_minute(kite, token: int, frm: date, to: date) -> dict | None:
    rows = kite.historical_data(int(token), datetime.combine(frm, dtime(9, 0)), datetime.combine(to, dtime(15, 30)),
                                "minute")
    return clean_minute_rows(kite_rows_to_lab(rows))


def fetch_daily(kite, token: int, frm: date, to: date) -> dict | None:
    rows = kite.historical_data(int(token), datetime.combine(frm, dtime(0, 0)), datetime.combine(to, dtime(23, 59)), "day")
    return daily_rows(kite_rows_to_lab(rows))


def keep_last_sessions(z: dict, n: int, upto: int) -> dict:
    sel = z["date"] <= int(upto)
    z = {k: v[sel] for k, v in z.items()}
    days = np.unique(z["date"])
    if len(days) > n:
        z = {k: v[z["date"] >= days[-n]] for k, v in z.items()}
    return z


def bootstrap(store: Store, kite, end_day: date, tokens: dict, sessions: int = WARMUP_SESSIONS,
              log: Callable = print) -> dict:
    """First run: WARMUP_SESSIONS sessions of minute history (<= end_day) for every instrument, and the daily
    series. ~11 requests per symbol (Kite serves 60 days of minute data per request)."""
    frm = end_day - timedelta(days=int(sessions * 1.55) + 20)
    out = {"symbols": 0, "missing": [], "bytes": 0}
    for sym in MINUTE_SYMBOLS:
        tok = tokens.get(sym)
        if not tok:
            out["missing"].append(sym)
            continue
        store.check_cap()
        z = fetch_minute(kite, tok, frm, end_day)
        if z is None or not len(z["date"]):
            out["missing"].append(sym)
            continue
        out["bytes"] += store.write_sessions(sym, keep_last_sessions(z, sessions, _ymd(end_day)))
        out["symbols"] += 1
    for name in DAILY_AUX:
        tok = tokens.get(name)
        if tok:
            out["bytes"] += store.write_aux(name, fetch_daily(kite, tok, frm - timedelta(days=400), end_day))
    store.note("bootstrap", {"end": _ymd(end_day), "sessions": sessions, "from": _ymd(frm), **out})
    log(f"[forward-lab] bootstrap: {out['symbols']} symbols, {out['bytes'] / 1048576:.1f} MB")
    return out


def fetch_session(store: Store, kite, day: date, tokens: dict, ca_actions: Optional[Callable] = None,
                  log: Callable = print) -> dict:
    """Fetch `day` (and re-read the previous stored session for the corporate-action check) for every instrument.
    Writes only a COMPLETE session's bars (plus corporate-action refetches); returns the completeness verdict."""
    dk = _ymd(day)
    prev = [s for s in store.sessions() if s < dk]
    prev_day = _d(prev[-1]) if prev else day - timedelta(days=7)
    minute, refetch = {}, []
    expected = [s for s in MINUTE_SYMBOLS if tokens.get(s)]
    not_listed = [s for s in MINUTE_SYMBOLS if not tokens.get(s)]
    for sym in expected:
        try:
            z = fetch_minute(kite, tokens[sym], prev_day, day)
        except Exception as exc:
            log(f"[forward-lab] {sym}: fetch failed: {exc}")
            z = None
        minute[sym] = z
        if z is None or sym in INDEX_MIN:
            continue
        stored = store.raw(sym, lo=_ymd(prev_day), upto=_ymd(prev_day))
        forced = bool(ca_actions and ca_actions(sym, day))
        if forced or ca_changed(stored, z, _ymd(prev_day)):
            refetch.append((sym, "calendar" if forced else "kite_adjusted"))
    daily = {}
    for name in DAILY_AUX:
        tok = tokens.get(name)
        try:
            daily[name] = fetch_daily(kite, tok, day - timedelta(days=10), day) if tok else None
        except Exception as exc:
            log(f"[forward-lab] {name}: fetch failed: {exc}")
            daily[name] = None
    ok, reasons = session_complete(dk, minute, daily, expected)
    res = {"day": dk, "complete": ok, "reasons": reasons[:50], "n_reasons": len(reasons), "not_listed": not_listed,
           "refetch": [], "bytes": 0}
    if not ok:
        return res
    store.check_cap()
    # corporate actions first: the symbol's whole window from Kite (adjusted), then today's bars on top
    first = store.sessions()[0] if store.sessions() else dk
    for sym, why in refetch:
        try:
            z = fetch_minute(kite, tokens[sym], _d(first), day)
        except Exception as exc:
            log(f"[forward-lab] {sym}: corporate-action refetch failed: {exc}")
            continue
        if z is not None and len(z["date"]):
            z = {k: v[z["date"] >= first] for k, v in z.items()}
            res["bytes"] += store.replace_symbol(sym, z)
            res["refetch"].append({"symbol": sym, "reason": why, "day": dk})
            store.note("refetch", {"symbol": sym, "reason": why, "day": dk})
            log(f"[forward-lab] {sym}: corporate action ({why}) — window refetched from Kite")
    for sym, z in minute.items():
        if z is None:
            continue
        today = {k: v[z["date"] == dk] for k, v in z.items()}
        res["bytes"] += store.write_sessions(sym, today)
    for name, a in daily.items():
        res["bytes"] += store.write_aux(name, a)
    return res
