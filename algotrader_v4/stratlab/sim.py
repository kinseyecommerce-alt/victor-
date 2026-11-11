"""
stratlab/sim.py — the one simulator every strategy in the lab is run through.

Intraday (kinds intraday, scalping, futures with horizon="intraday")
  * a signal at the close of bar i is filled at the OPEN of bar i+1, 5 bps worse than the open
  * stop and target are checked on every later bar's low/high (the entry bar included);
    if a bar touches both, the STOP is assumed; a bar that opens beyond the stop fills at its open
  * stop / time / square-off exits pay 5 bps slippage; a target is a resting limit and fills at its price
  * everything still open is squared off at the open of the 15:15 bar (or the last close if the data ends first)
  * ₹1,00,000 notional per trade, whole shares; one open position per symbol; `max_per_day` trades per day
Positional (swing, futures with horizon="positional")
  * the strategy sees daily bars up to day d and is filled at the next session's open, or - when it declares a
    decision_time - sees day d only up to that minute and is filled at that minute's bar open
  * stop / target are checked on daily high/low, stop first; a gap through the stop fills at the open
  * closed at the close of the `max_hold`-th session
Costs: costs.round_trip_cost - equity_mis for intraday/scalping, equity_cnc for swing, future for futures
(futures are simulated on the cash price: no basis, no lot size, no rollover).
Swing is delivery and therefore long-only; shorts belong under futures.

No look-ahead: "bar" strategies only ever receive a View truncated at the current bar. "vector" strategies
receive the whole session for speed, so every run re-executes them on truncated views of sampled sessions and
disqualifies any strategy whose earlier signals change (`audit_lookahead`).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402

from . import data as _data  # noqa: E402
from .api import Entry, Spec, Swing  # noqa: E402
from .indicators import DailyHist, Session, View, build_sessions  # noqa: E402

SLIPPAGE = 0.0005               # 5 bps on every market fill
NOTIONAL = 100_000.0            # ₹ per trade
SQUAREOFF = 15 * 60 + 15        # 15:15
AUDIT_EVERY = 29                # audit one session in this many (per symbol)

TRADE_FIELDS = ("id", "symbol", "date", "side", "entry_t", "entry", "exit_date", "exit_t", "exit", "qty",
                "gross", "cost", "net", "reason")


def _hm(minute: int) -> str:
    return f"{int(minute) // 60:02d}:{int(minute) % 60:02d}"


def _settle(spec: Spec, symbol, date, side, entry_t, fill, exit_date, exit_t, exit_px, reason) -> tuple:
    qty = max(1, int(NOTIONAL // fill))
    gross = side * (exit_px - fill) * qty
    buy_v, sell_v = (fill * qty, exit_px * qty) if side > 0 else (exit_px * qty, fill * qty)
    cost = round_trip_cost(buy_v, sell_v, spec.cost_kind)
    return (spec.id, symbol, int(date), int(side), entry_t, round(fill, 4), int(exit_date), exit_t,
            round(exit_px, 4), qty, round(gross, 2), cost, round(gross - cost, 2), reason)


# ── intraday ────────────────────────────────────────────────────────────────
def fill_intraday(s: Session, e: Entry):
    """Simulate one entry. Returns (exit_index, fill, exit_price, reason) or None if it cannot be taken."""
    n = len(s.t)
    j = int(e.i) + 1
    if e.side not in (1, -1) or j < 1 or j >= n or s.t[j] >= SQUAREOFF:
        return None
    side = e.side
    raw = float(s.o[j])
    fill = raw * (1.0 + side * SLIPPAGE)
    stop = float(e.stop) if e.stop is not None else np.nan
    if not np.isfinite(stop) or side * (raw - stop) <= 0:
        return None                                     # no stop, or already through it at the open
    risk = abs(fill - stop)
    target = e.target
    if target is None and e.rr:
        target = fill + side * e.rr * risk
    if target is not None:
        target = float(target)
        if not np.isfinite(target) or side * (target - fill) <= 0:
            return None
    lim = s.idx(SQUAREOFF)
    reason = "squareoff"
    if e.time_stop:
        ts = s.idx(int(s.t[j]) + int(e.time_stop))
        if ts < lim:
            lim, reason = ts, "time"
    h, l, o = s.h, s.l, s.o
    if e.trail is None:
        if side > 0:
            hs = l[j:lim] <= stop
            ht = h[j:lim] >= target if target is not None else np.zeros(max(lim - j, 0), bool)
        else:
            hs = h[j:lim] >= stop
            ht = l[j:lim] <= target if target is not None else np.zeros(max(lim - j, 0), bool)
        hit = hs | ht
        if hit.size and hit.any():
            r = int(hit.argmax())
            k = j + r
            if hs[r]:                                   # stop wins when both are touched
                px = min(stop, float(o[k])) if side > 0 else max(stop, float(o[k]))
                return k, fill, px * (1.0 - side * SLIPPAGE), "stop"
            return k, fill, target, "target"
    else:
        mode, x = e.trail[0], float(e.trail[1])
        a = float(s.atr5[e.i]) if mode == "atr" else 0.0
        if mode == "atr" and not np.isfinite(a):
            a = risk
        best = fill
        for k in range(j, lim):
            if side * (float(o[k]) - stop) <= 0 or (l[k] <= stop if side > 0 else h[k] >= stop):
                px = min(stop, float(o[k])) if side > 0 else max(stop, float(o[k]))
                return k, fill, px * (1.0 - side * SLIPPAGE), "stop"
            if target is not None and (h[k] >= target if side > 0 else l[k] <= target):
                return k, fill, target, "target"
            best = max(best, float(h[k])) if side > 0 else min(best, float(l[k]))     # effective next bar
            if mode == "pct":
                new = best * (1.0 - side * x / 100.0)
            elif mode == "atr":
                new = best - side * x * a
            elif mode == "be":
                new = fill if side * (best - fill) >= x * risk else stop
            else:
                raise ValueError(f"unknown trail rule {e.trail!r}")
            if side * (new - stop) > 0:
                stop = new
    if lim < n:
        return lim, fill, float(o[lim]) * (1.0 - side * SLIPPAGE), reason
    return n - 1, fill, float(s.c[n - 1]) * (1.0 - side * SLIPPAGE), "squareoff"


def _take(spec: Spec, s: Session, e: Entry, trades: list) -> int | None:
    r = fill_intraday(s, e)
    if r is None:
        return None
    k, fill, px, reason = r
    trades.append(_settle(spec, s.symbol, s.date, e.side, _hm(s.t[e.i + 1]), fill, s.date, _hm(s.t[k]), px, reason))
    return k


def sim_session(spec: Spec, s: Session) -> tuple[list, int]:
    """All trades of one strategy in one symbol-session. Returns (trades, skipped_entries)."""
    trades: list = []
    skipped = 0
    p = spec.p
    if spec.style == "vector":
        ents = spec.fn(s, p) or []
        free = 0
        for e in sorted(ents, key=lambda x: x.i):
            if len(trades) >= spec.max_per_day:
                break
            if e.i < free:
                continue                                # a position was still open at this bar
            k = _take(spec, s, e, trades)
            if k is None:
                skipped += 1
            else:
                free = k
        return trades, skipped
    n = len(s.t)
    i = 0
    while i < n - 1 and len(trades) < spec.max_per_day:
        if s.t[i + 1] >= SQUAREOFF:
            break
        e = spec.fn(View(s, i + 1), p)                  # the strategy cannot see past bar i
        if e is not None:
            e.i = i
            k = _take(spec, s, e, trades)
            if k is None:
                skipped += 1
            else:
                i = k
                continue
        i += 1
    return trades, skipped


def _canon(ents, upto: int) -> list:
    out = []
    for e in sorted(ents or [], key=lambda x: x.i):
        if e.i <= upto:
            f = lambda x: None if x is None else round(float(x), 6)
            out.append(repr((int(e.i), int(e.side), f(e.stop), f(e.target), f(e.rr), e.time_stop, e.trail)))
    return out


def audit_lookahead(spec: Spec, s: Session, cuts=None) -> str | None:
    """Re-run a vector strategy on truncated views; its signals up to each cut must not change.
    Returns None if clean, else a description of the first difference."""
    if spec.style != "vector":
        return None
    n = len(s.t)
    full = spec.fn(s, spec.p) or []
    for m in (cuts or (n // 5, n // 3, n // 2, (3 * n) // 4)):
        if m < 2 or m >= n:
            continue
        a, b = _canon(full, m - 1), _canon(spec.fn(View(s, m), spec.p), m - 1)
        if a != b:
            return (f"{s.symbol} {s.date}: signals up to {_hm(s.t[m - 1])} differ when later bars are hidden "
                    f"(full={a[:2]} truncated={b[:2]})")
    return None


# ── positional ──────────────────────────────────────────────────────────────
class DailyView(DailyHist):
    """Daily bars a positional strategy may use: arrays date o h l c v (index -1 = the decision day; with a
    decision_time that last bar is the day so far), plus symbol, and nifty / vix daily closes aligned to the
    same days (None if not in the data; NaN where unknown)."""
    __slots__ = ("symbol", "nifty", "vix")


def _aligned(dates: np.ndarray, aux: dict | None):
    if aux is None:
        return None
    pos = np.searchsorted(aux["date"], dates)
    ok = (pos < len(aux["date"])) & (aux["date"][np.minimum(pos, len(aux["date"]) - 1)] == dates)
    return np.where(ok, aux["c"][np.minimum(pos, len(aux["c"]) - 1)], np.nan)


def sim_positional(spec: Spec, sd, lo: int, hi: int, nifty_day: dict | None = None,
                   vix_day: dict | None = None) -> tuple[list, int]:
    """All trades of one positional strategy in one symbol; entries are taken only on dates in [lo, hi]."""
    d = sd.daily()
    nd = len(d["c"])
    nifty, vix = _aligned(d["date"], nifty_day), _aligned(d["date"], vix_day)
    trades: list = []
    skipped = 0
    p = spec.p
    slip = SLIPPAGE
    day = 0
    while day < nd:
        intr = spec.decision_time is not None
        e_day = day if intr else day + 1
        if e_day >= nd or d["date"][e_day] > hi:
            break
        if d["date"][e_day] < lo:
            day += 1
            continue
        m = None
        if intr:                                        # day `day` is visible only up to the decision minute
            a, b = int(sd.starts[day]), int(sd.ends[day])
            m = a + int(np.searchsorted(sd.t[a:b], spec.decision_time))
            if m >= b or m == a:
                day += 1
                continue
            last = (d["o"][day], sd.h[a:m].max(), sd.l[a:m].min(), sd.c[m - 1], sd.v[a:m].sum())
            cols = [np.append(d[k][:day], x) for k, x in zip("ohlcv", last)]
        else:
            cols = [d[k][:day + 1] for k in "ohlcv"]
        v = DailyView(d["date"][:day + 1], *cols)
        v.symbol = sd.symbol
        if intr:                                        # today's index / VIX close is not known yet
            v.nifty = None if nifty is None else np.append(nifty[:day], np.nan)
            v.vix = None if vix is None else np.append(vix[:day], np.nan)
        else:
            v.nifty = None if nifty is None else nifty[:day + 1]
            v.vix = None if vix is None else vix[:day + 1]
        sig = spec.fn(v, p)
        if sig is None:
            day += 1
            continue
        if not isinstance(sig, Swing):
            raise TypeError(f"{spec.id}: positional strategies return Swing or None")
        side = sig.side
        if spec.kind == "swing" and side != 1:
            raise ValueError(f"{spec.id}: swing is delivery (long-only); put shorts under kind='futures'")
        raw = float(sd.o[m]) if intr else float(d["o"][e_day])
        fill = raw * (1.0 + side * slip)
        stop = sig.stop if sig.stop is not None else (fill * (1 - side * sig.stop_pct / 100.0) if sig.stop_pct else None)
        target = sig.target if sig.target is not None else (
            fill * (1 + side * sig.target_pct / 100.0) if sig.target_pct else None)
        if target is None and sig.rr and stop is not None:
            target = fill + side * sig.rr * abs(fill - stop)
        if (stop is not None and side * (raw - stop) <= 0) or (target is not None and side * (target - fill) <= 0) \
                or sig.max_hold < 1:
            skipped += 1
            day += 1
            continue
        last_day = min(e_day + sig.max_hold - (0 if intr else 1), nd - 1)
        out = None
        for k in range(e_day, last_day + 1):
            if k == e_day:
                if intr:
                    b = int(sd.ends[k])
                    kh, kl, ko = float(sd.h[m:b].max()), float(sd.l[m:b].min()), raw
                else:
                    kh, kl, ko = float(d["h"][k]), float(d["l"][k]), raw
            else:
                kh, kl, ko = float(d["h"][k]), float(d["l"][k]), float(d["o"][k])
            if stop is not None and (kl <= stop if side > 0 else kh >= stop):
                px = min(stop, ko) if side > 0 else max(stop, ko)        # a gap through the stop fills at the open
                out = (k, px * (1.0 - side * slip), "stop")
                break
            if target is not None and (kh >= target if side > 0 else kl <= target):
                out = (k, float(target), "target")
                break
        if out is None:
            out = (last_day, float(d["c"][last_day]) * (1.0 - side * slip),
                   "max_hold" if last_day == e_day + sig.max_hold - (0 if intr else 1) else "data_end")
        k, px, reason = out
        et = _hm(sd.t[m]) if intr else "09:15"
        trades.append(_settle(spec, sd.symbol, d["date"][e_day], side, et, fill, d["date"][k],
                              "15:29" if reason in ("max_hold", "data_end") else "", px, reason))
        day = max(k, day + 1)                           # flat again from the exit day's close
    return trades, skipped


# ── one symbol, many strategies ─────────────────────────────────────────────
def run_symbol(cache_dir: str, symbol: str, specs: list[Spec], lo: int, hi: int, audit: bool = True,
               sd=None, index=None) -> dict:
    """Run every spec on one symbol for entry dates in [lo, hi].
    Returns {id: {"trades": [...], "skipped": n, "lookahead": str|None, "error": str|None, "sessions": n}}."""
    sd = sd if sd is not None else _data.load_symbol(cache_dir, symbol)
    out = {sp.id: {"trades": [], "skipped": 0, "lookahead": None, "error": None, "sessions": 0, "secs": 0.0}
           for sp in specs}
    if sd is None:
        return out
    intr = [sp for sp in specs if sp.horizon == "intraday"]
    pos = [sp for sp in specs if sp.horizon == "positional"]
    if intr:
        if index is None and cache_dir:
            index = _data.load_symbol(cache_dir, "_NIFTY50_min")
        sessions = build_sessions(sd, index)
        for sp in intr:
            res = out[sp.id]
            t0 = time.perf_counter()
            try:
                for s in sessions:
                    if s.date < lo or s.date > hi:
                        continue
                    if sp.needs_index and s.nifty is None:
                        continue
                    res["sessions"] += 1
                    tr, sk = sim_session(sp, s)
                    res["trades"].extend(tr)
                    res["skipped"] += sk
                    if audit and res["lookahead"] is None and s.day_index % AUDIT_EVERY == 0:
                        res["lookahead"] = audit_lookahead(sp, s)
            except Exception as exc:
                res["error"] = f"{symbol}: {type(exc).__name__}: {str(exc)[:160]}"
            res["secs"] = time.perf_counter() - t0
    if pos:
        nifty_day = _data.load_daily_aux(cache_dir, "_NIFTY50_day") if cache_dir else None
        vix_day = _data.load_daily_aux(cache_dir, "_VIX_day") if cache_dir else None
        for sp in pos:
            res = out[sp.id]
            t0 = time.perf_counter()
            try:
                tr, sk = sim_positional(sp, sd, lo, hi, nifty_day, vix_day)
                res["trades"].extend(tr)
                res["skipped"] += sk
                res["sessions"] = int(((sd.days >= lo) & (sd.days <= hi)).sum())
            except Exception as exc:
                res["error"] = f"{symbol}: {type(exc).__name__}: {str(exc)[:160]}"
            res["secs"] = time.perf_counter() - t0
    return out
