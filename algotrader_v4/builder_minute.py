"""
builder_minute.py — deep 1-minute history for the strategy builder: the forward lab's universe (forward_test.UNIVERSE,
97 stocks, plus NIFTY 50 and NIFTY BANK) back to 2016-01-01 from Kite, and the NIFTY 50 / INDIA VIX daily series;
with BUILDER_MINUTE_UNIVERSE=nifty200 (default) also the ADDED names of builder_universe.py (current NIFTY 200 list +
historical NIFTY 100 members still on Kite), fetched AFTER the 99 instruments, same store / format / budget / guards
(builder_data.minute_job passes the list and the tokens; names Kite does not list are recorded as "missing").

STORE: a SIBLING of the forward lab's bar store with the SAME format and the SAME reader — it is a
forward_lab_store.Store rooted at <builder>/minute (month files bars/<SYMBOL>/<YYYYMM>.npz in integer paise, the
lab's parse / clean (09:15-15:29, finite positive prices, duplicates dropped), loaded back through
stratlab.data._sessionize by Store.load_symbol). The forward lab's own store (<volume>/forward_lab/bars) is never
read or written here, so the forward lab behaves exactly as before.

BACKFILL: a fixed grid of 60-day windows starting 2016-01-01 (Kite serves at most 60 days of minute data per request;
one window = ONE historical_data call through kite_client's shared history limiter `_hist_bucket` / `_with_retry`).
Per symbol the windows are fetched newest first; a window is FINAL once it was fetched after its last day. The newest
(still open) window is re-fetched on every run: that is the daily update (one call per instrument per run).
  resumable / idempotent  the window state lives in <builder>/minute/backfill.json; Store.write_sessions replaces the
                          sessions it is given, so re-fetching a window rewrites identical bytes
  call budget             BUILDER_KITE_MAX_CALLS_PER_RUN (default 6,000; 1,500 until Nov 2026) per run; ~67 windows
                          x 99 instruments ~= 6,630 calls + the NIFTY 200 extension's ~67 x the added names Kite lists
                          (~125 -> ~8,400 calls): ~15,000 calls in all, about 3 runs (two a night) at Kite's history
                          rate (~35-60 min a run, every call still behind the market-hours guard)
  market-hours guard      never between 09:00 and 15:45 IST on a trading day (checked before every call)
  locks                   the option-bar archiver's lock (the one the forward lab holds for its Kite calls) is held
                          per instrument, so the three Kite users never interleave
  corporate actions       Kite's history is adjusted for splits / bonuses as of the fetch. The first window fetched
                          for an instrument in a run is its newest one; its overlap with the stored bars is compared
                          with forward_lab_store.ca_changed (median close ratio > 0.05 %): a difference means Kite
                          back-adjusted (a capital change went ex) and EVERY window of that instrument is re-fetched
                          (newest first), exactly the forward lab's corporate-action convention. When that fetch
                          shares no session with the store (a new window began, possibly on the ex-date, or a
                          resumed backfill), one short request (OVERLAP_DAYS) ending at the last stored session is
                          compared instead; with no overlap at all nothing is written and the instrument is retried
                          next run (never ruled clean without a comparison)
  disk cap                BUILDER_DATA_MAX_MB for the whole builder directory: at the cap nothing more is written
                          (the run reports "disk_cap"); nothing is ever deleted
Read-only towards trading: the only Kite calls are get_instrument_tokens and historical_data.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import forward_lab_store as FS

START = date(2016, 1, 1)
WINDOW_DAYS = 60
OVERLAP_DAYS = 6                        # the short re-read ending at the last stored session (CA check)
DAILY_FROM = date(2015, 6, 1)           # day candles: a little before the minute history (warm-up of daily features)
_LOCK = threading.RLock()


def _ymd(d) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def _d(k) -> date:
    k = int(k)
    return date(k // 10000, k // 100 % 100, k % 100)


def windows(end: date, start: date = START) -> list[tuple[date, date]]:
    """The fixed 60-day grid [start + 60k, start + 60k + 59] covering start..end (oldest first)."""
    out, s = [], start
    while s <= end:
        out.append((s, s + timedelta(days=WINDOW_DAYS - 1)))
        s += timedelta(days=WINDOW_DAYS)
    return out


# ── Kite (server) ─────────────────────────────────────────────────────────────────────────────────────────────────
def kite_fetch(token: int, frm: date, to: date, interval: str = "minute") -> list[dict]:
    """ONE historical_data request through kite_client's shared history limiter; raises on failure (unlike
    kite_client.historical_data, which logs and returns [] for a failed chunk)."""
    import kite_client as KC
    from config import settings
    if not settings.uses_kite_market_data:
        raise RuntimeError("no Kite market-data session (PAPER_DATA_SOURCE is not KITE)")
    kite = KC.kite_client.ensure_session()
    if interval == "minute":
        a, b = datetime.combine(frm, dtime(9, 0)), datetime.combine(to, dtime(15, 30))
    else:
        a, b = datetime.combine(frm, dtime(0, 0)), datetime.combine(to, dtime(23, 59))
    return KC._with_retry(lambda: kite.historical_data(instrument_token=int(token), from_date=a, to_date=b,
                                                       interval=interval, continuous=False, oi=False),
                          bucket=KC._hist_bucket, label="builder_history")


def kite_tokens() -> dict:
    from kite_client import kite_client
    return FS.instrument_tokens(kite_client)


# ── state ─────────────────────────────────────────────────────────────────────────────────────────────────────────
class MinuteBackfill:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.store = FS.Store(self.root, cap_root=self.root.parent)
        self.state_path = self.root / "backfill.json"

    def state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            return {"symbols": {}, "aux": {}, "runs": [], "missing": []}

    def save(self, st: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, sort_keys=True, separators=(",", ":")))
        os.replace(tmp, self.state_path)

    @staticmethod
    def final(rec: Optional[dict], w_end: date) -> bool:
        return bool(rec) and int(rec.get("at", 0)) > _ymd(w_end)

    @staticmethod
    def last_close(now: datetime) -> datetime:
        """The most recent 15:45 IST at or before `now` (after it, the day's session is complete in Kite)."""
        c = datetime.combine(now.date(), dtime(15, 45))
        return c if now >= c else c - timedelta(days=1)

    def pending(self, st: dict, sym: str, end: date, now: Optional[datetime] = None) -> list[tuple[date, date]]:
        """Windows to fetch, newest first: never fetched, or not final and fetched before the last session close
        (the open window is re-read once per closed session: the daily update)."""
        done = (st["symbols"].get(sym) or {}).get("windows", {})
        lc = self.last_close(now or datetime.combine(end, dtime(23, 59))).isoformat(timespec="seconds")
        out = []
        for w in reversed(windows(end)):
            rec = done.get(str(_ymd(w[0])))
            if self.final(rec, w[1]):
                continue
            if rec and str(rec.get("ts", "")) >= lc:
                continue
            out.append(w)
        return out

    def progress(self, st: dict, end: date, symbols=None, now: Optional[datetime] = None) -> dict:
        """Windows done (final, or the open window read after the last session close) of all windows."""
        symbols = list(symbols or FS.MINUTE_SYMBOLS)
        total = len(windows(end)) * len(symbols)
        pend = sum(len(self.pending(st, sym, end, now)) for sym in symbols)
        first = min((s["first"] for s in st["symbols"].values() if s.get("first")), default=None)
        last = max((s["last"] for s in st["symbols"].values() if s.get("last")), default=None)
        fin = total - pend
        return {"windows_final": fin, "windows_total": total,
                "progress_pct": round(100.0 * fin / total, 2) if total else 0.0, "from": first, "to": last}

    # ── one run ──
    def run(self, now: datetime, fetch: Callable = kite_fetch, tokens: Optional[dict] = None,
            max_calls: int = 1500, cap_check: Optional[Callable[[], None]] = None,
            market_hours: Optional[Callable[[datetime], bool]] = None, clock: Optional[Callable[[], datetime]] = None,
            lock=None, symbols=None, log: Callable = print, unresolved=None) -> dict:
        """Fetch pending windows (newest first per instrument) until the call budget, the market-hours guard or the
        disk cap stops the run. Returns the run summary (also appended to the state)."""
        from builder_data import DiskCapReached
        clock = clock or (lambda: now)
        market_hours = market_hours or (lambda t: False)
        cap_check = cap_check or (lambda: None)
        symbols = list(symbols or FS.MINUTE_SYMBOLS)
        today = now.date()
        end = today
        out = {"started": now.isoformat(timespec="seconds"), "calls": 0, "windows": 0, "sessions": 0, "bytes": 0,
               "symbols_done": 0, "ca_refetch": [], "errors": [], "stopped": None}
        with _LOCK:
            st = self.state()
            if tokens is None:
                tokens = kite_tokens()
            unres = set(unresolved or [])           # token lookup failed this run: not missing, retried next run
            st["missing"] = [s for s in symbols if not tokens.get(s) and s not in unres]
            st["unresolved"] = sorted(s for s in symbols if not tokens.get(s) and s in unres)
            consecutive_err = 0
            for sym in symbols:
                if out["stopped"]:
                    break
                tok = tokens.get(sym)
                if not tok:
                    continue
                pend = self.pending(st, sym, end, now)
                if not pend:
                    out["symbols_done"] += 1
                    continue
                got = lock.acquire(timeout=600) if lock is not None else True
                if not got:
                    out["stopped"] = "busy (archiver / forward-lab lock held)"
                    break
                try:
                    s = st["symbols"].setdefault(sym, {"windows": {}})
                    first_in_run = True
                    i = 0
                    while i < len(pend):
                        w0, w1 = pend[i]
                        if out["calls"] >= int(max_calls):
                            out["stopped"] = "call budget"
                            break
                        if market_hours(clock()):
                            out["stopped"] = "market hours"
                            break
                        try:
                            cap_check()
                        except DiskCapReached as exc:
                            out["stopped"] = f"disk_cap: {exc}"
                            break
                        out["calls"] += 1
                        try:
                            rows = fetch(tok, w0, min(w1, end), "minute")
                            consecutive_err = 0
                        except Exception as exc:
                            consecutive_err += 1
                            out["errors"].append(f"{sym} {w0}..{w1}: {type(exc).__name__}: {str(exc)[:160]}")
                            s["last_error"] = out["errors"][-1]
                            if consecutive_err >= 3:
                                out["stopped"] = "Kite errors"
                            break
                        z = FS.clean_minute_rows(FS.kite_rows_to_lab(rows))
                        if z is not None and len(z["date"]):
                            keep = z["date"] <= _ymd(today)
                            z = {k: v[keep] for k, v in z.items()}
                        if first_in_run and s.get("last") and z is not None and len(z["date"]):
                            # corporate-action check against an OVERLAPPING stored session (forward lab's rule)
                            stored = self.store.raw(sym, lo=_ymd(w0), upto=_ymd(min(w1, end)))
                            sdays = sorted(set(int(x) for x in stored["date"]) & set(int(x) for x in z["date"]))
                            ref_z = z
                            if not sdays:
                                # the newest fetch shares no session with the store (e.g. a new window just began,
                                # possibly on an ex-date): re-read a short window ending at the last stored session
                                if out["calls"] >= int(max_calls):
                                    out["stopped"] = "call budget"
                                    break
                                last = _d(s["last"])
                                lo = last - timedelta(days=OVERLAP_DAYS)
                                out["calls"] += 1
                                try:
                                    ref_z = FS.clean_minute_rows(FS.kite_rows_to_lab(fetch(tok, lo, last, "minute")))
                                except Exception as exc:
                                    ref_z = None
                                    out["errors"].append(f"{sym} overlap {lo}..{last}: {type(exc).__name__}: "
                                                         f"{str(exc)[:120]}")
                                stored = self.store.raw(sym, lo=_ymd(lo), upto=_ymd(last))
                                sdays = sorted(set(int(x) for x in stored["date"]) &
                                               set(int(x) for x in (ref_z["date"] if ref_z is not None else [])))
                            if not sdays:
                                # no overlap obtainable: the instrument is NOT ruled clean; nothing is written (a new
                                # window on another price basis would hide the adjustment) — retried next run
                                msg = f"{sym}: no overlapping session for the corporate-action check; retried next run"
                                out["errors"].append(msg)
                                s["last_error"] = msg
                                break
                            if FS.ca_changed(stored, ref_z, sdays[-1]):
                                out["ca_refetch"].append(sym)
                                s["windows"] = {}
                                s.setdefault("ca_refetch", []).append(_ymd(today))
                                pend = self.pending(st, sym, end, now)  # everything again, newest first
                                i = 0
                            first_in_run = False
                        elif not s.get("last"):
                            first_in_run = False            # nothing stored yet: nothing to compare with
                        n = 0
                        if z is not None and len(z["date"]):
                            out["bytes"] += self.store.write_sessions(sym, z)
                            days = np.unique(z["date"])
                            n = int(len(days))
                            s["first"] = int(days[0]) if not s.get("first") else min(int(s["first"]), int(days[0]))
                            s["last"] = int(days[-1]) if not s.get("last") else max(int(s["last"]), int(days[-1]))
                        s["windows"][str(_ymd(w0))] = {"at": _ymd(today), "ts": now.isoformat(timespec="seconds"), "sessions": n}
                        out["windows"] += 1
                        out["sessions"] += n
                        i += 1
                        if out["windows"] % 10 == 0:
                            self.save(st)
                    else:
                        out["symbols_done"] += 1
                finally:
                    if lock is not None and got:
                        lock.release()
                self.save(st)
            # daily series (NIFTY 50, INDIA VIX): one request per <= 2,000 days, re-read every run
            if not out["stopped"]:
                for name in FS.DAILY_AUX:
                    tok = tokens.get(name)
                    if not tok or out["calls"] >= int(max_calls) or market_hours(clock()):
                        continue
                    try:
                        cap_check()
                    except DiskCapReached as exc:
                        out["stopped"] = f"disk_cap: {exc}"
                        break
                    a = DAILY_FROM
                    while a <= today and out["calls"] < int(max_calls):
                        b = min(today, a + timedelta(days=1999))
                        out["calls"] += 1
                        try:
                            arr = FS.daily_rows(FS.kite_rows_to_lab(fetch(tok, a, b, "day")))
                            if arr is not None:
                                out["bytes"] += self.store.write_aux(name, arr)
                                st["aux"][name] = {"from": int(self.store.load_aux(name)["date"][0]),
                                                   "to": int(self.store.load_aux(name)["date"][-1]),
                                                   "at": _ymd(today)}
                        except Exception as exc:
                            out["errors"].append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
                        a = b + timedelta(days=1)
            out["finished"] = clock().isoformat(timespec="seconds")
            st["runs"] = (st.get("runs") or [])[-30:] + [out]
            st["last_run"] = out
            if out["errors"]:
                st["last_error"] = {"at": out["finished"], "error": out["errors"][-1]}
            self.save(st)
        log(f"[builder-minute] {out['calls']} calls, {out['windows']} windows, {out['sessions']} sessions, "
            f"{out['bytes'] / 1048576:.1f} MB, stopped: {out['stopped']}")
        return out
