"""
stratlab/sim.py — the one simulator every strategy in the lab is run through.

Slippage is a per-run setting (`slippage=` on run_symbol / the CLI's --slippage-bps): default 2 bps per side
for NIFTY-100 stocks and 1 bps for NIFTY / BANKNIFTY index futures. Why 2 bps: the cost audit estimated the quoted
spread of the 97 NIFTY-100 stocks from 1-minute bars with the Abdi-Ranaldo close-high-low estimator: median
2.78 bps (interquartile 2.33-3.05) over the last 12 months, and median 3.26 bps (2.87-3.72) over the 3-year
data's DISCOVERY sessions alone (to 2025-12-16), i.e. a half-spread of 1.4-1.6 bps per side. A Rs 1 lakh order is
a median 1.4 % of a minute's volume, well inside the top of book. 2 bps per side = that half-spread plus a margin
for impact and latency; the old 5 bps was ~2.4x it. 5 bps is kept as a sensitivity column in every report, so a
strategy that passes only at 2 bps is visibly fragile. Thinner names (spread estimate up to ~8 bps) pay more than
2 bps in reality: the sensitivity column is the check for them.

Intraday (kinds intraday, scalping, futures with horizon="intraday")
  * a signal at the close of bar i is filled at the OPEN of bar i+1, `slippage` worse than the open
  * stop and target are checked on every later bar's low/high (the entry bar included);
    if a bar touches both, the STOP is assumed; a bar that opens beyond the stop fills at its OPEN (not the stop)
  * stop / time / square-off / data-end exits are market orders and pay `slippage`
  * a target is a resting limit: it fills only if the bar trades THROUGH it by at least one tick (a bare touch
    is not a fill: the queue at that price may not have been reached); if the bar opens beyond the target
    by a tick or more it fills at that open (price improvement), otherwise at the target price
  * everything still open is squared off at the open of the 15:15 bar (or the last close if the data ends first)
  * ₹1,00,000 notional per trade, whole shares; one open position per symbol; `max_per_day` trades per day
Index futures (kind="futures", underlying="NIFTY" | "BANKNIFTY")
  * simulated on the index's own minute bars (_NIFTY50_min / _NIFTYBANK_min) with futures costs, `lots` x the
    lot size (LOT_SIZES) per trade and `index_slippage` (default 1 bps); tick 0.05. Basis, roll and margin are
    NOT modelled: the spot index stands in for the near-month future (the basis is a few tenths of a percent and
    drifts slowly; a positional hold across expiry would really pay a roll spread and a second round trip).
Positional (swing, futures with horizon="positional")
  * the strategy sees daily bars up to day d and is filled at the next session's open, or - when it declares a
    decision_time - sees day d only up to that minute and is filled at that minute's bar open
  * stop / target are checked on daily high/low, stop first; a gap through the stop fills at the open;
    a target needs a trade one tick through it (a gap beyond it fills at the open)
  * closed at the close of the `max_hold`-th session (paying slippage)
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
import costs as _costs  # noqa: E402
from costs import round_trip_cost  # noqa: E402

from . import data as _data  # noqa: E402
from .api import Entry, Spec, Swing  # noqa: E402
from .indicators import DailyHist, Session, View, build_sessions  # noqa: E402

DEFAULT_SLIPPAGE_BPS = 2.0      # per side, NIFTY-100 stocks at ₹1 lakh (see the module docstring)
SENSITIVITY_SLIPPAGE_BPS = 5.0  # the old default, reported as a sensitivity column
INDEX_SLIPPAGE_BPS = 1.0        # per side, NIFTY / BANKNIFTY futures
SLIPPAGE = DEFAULT_SLIPPAGE_BPS / 1e4        # default fraction on every market fill (override per run)
INDEX_SLIPPAGE = INDEX_SLIPPAGE_BPS / 1e4
# NSE lot sizes from the January 2026 expiries (NSE circular FAOP70616, 3 Oct 2025): NIFTY 75 -> 65,
# BANKNIFTY 35 -> 30. Used for every date: P&L per lot scales with the lot size, so older dates are simply
# reported in 2026 lots. (The live app's _FON_LOT_SIZES still says NIFTY 75; the lab does not read it.)
LOT_SIZES = {"NIFTY": 65, "BANKNIFTY": 30}
INDEX_DATA = {"NIFTY": "_NIFTY50_min", "BANKNIFTY": "_NIFTYBANK_min"}
INDEX_TICK = 0.05
# Bump when a fill rule changes. Together with costs.COST_MODEL_VERSION it names the cost/fill model the FINAL
# ledger is keyed on (protocol.ledger_key). Slippage is NOT part of it: choosing a different slippage after seeing
# FINAL must not buy a fresh look.
FILL_MODEL_VERSION = "f2"
COST_MODEL = f"{_costs.COST_MODEL_VERSION}/{FILL_MODEL_VERSION}"
NOTIONAL = 100_000.0            # ₹ per trade
SQUAREOFF = 15 * 60 + 15        # 15:15
AUDIT_EVERY = 29                # audit one session in this many (per symbol)

TRADE_FIELDS = ("id", "symbol", "date", "side", "entry_t", "entry", "exit_date", "exit_t", "exit", "qty",
                "gross", "cost", "net", "reason")


def _hm(minute: int) -> str:
    return f"{int(minute) // 60:02d}:{int(minute) % 60:02d}"


def equity_tick(price: float, date: int = 0) -> float:
    """NSE equity tick size: ₹0.05, and ₹0.01 for stocks under ₹250 from 16 June 2025 (NSE tick-size revision)."""
    return 0.01 if (price < 250.0 and int(date) >= 20250616) else 0.05


def spec_slippage(spec: Spec, slippage: float | None = None, index_slippage: float | None = None) -> float:
    """The per-side slippage fraction a spec pays in this run."""
    if getattr(spec, "underlying", None):
        return INDEX_SLIPPAGE if index_slippage is None else float(index_slippage)
    return SLIPPAGE if slippage is None else float(slippage)


def _tick(spec, price, date) -> float:
    return INDEX_TICK if getattr(spec, "underlying", None) else equity_tick(price, date)


def target_fill(side: int, target: float, o: float, extreme: float, tick: float):
    """A resting limit at `target`: None unless the bar trades at least one tick through it (`extreme` = the
    bar's high for a long, its low for a short); fills at the open when the open is already a tick or more beyond."""
    eps = 1e-9
    if side * (extreme - target) < tick - eps:
        return None
    return float(o) if side * (o - target) >= tick - eps else float(target)


def _settle(spec: Spec, symbol, date, side, entry_t, fill, exit_date, exit_t, exit_px, reason, lots=None) -> tuple:
    if getattr(spec, "underlying", None):
        qty = int(spec.lots if lots is None else lots) * LOT_SIZES[spec.underlying]
    else:
        qty = max(1, int(NOTIONAL // fill))
    gross = side * (exit_px - fill) * qty
    buy_v, sell_v = (fill * qty, exit_px * qty) if side > 0 else (exit_px * qty, fill * qty)
    cost = round_trip_cost(buy_v, sell_v, spec.cost_kind)
    return (spec.id, symbol, int(date), int(side), entry_t, round(fill, 4), int(exit_date), exit_t,
            round(exit_px, 4), qty, round(gross, 2), cost, round(gross - cost, 2), reason)


# ── intraday ────────────────────────────────────────────────────────────────
def fill_intraday(s: Session, e: Entry, slip: float | None = None, tick: float | None = None):
    """Simulate one entry. Returns (exit_index, fill, exit_price, reason) or None if it cannot be taken.
    slip = per-side slippage fraction (default SLIPPAGE); tick = price tick for the target rule (default: the
    NSE equity tick at the target price on the session's date)."""
    sl = SLIPPAGE if slip is None else float(slip)
    n = len(s.t)
    j = int(e.i) + 1
    if e.side not in (1, -1) or j < 1 or j >= n or s.t[j] >= SQUAREOFF:
        return None
    side = e.side
    raw = float(s.o[j])
    fill = raw * (1.0 + side * sl)
    br = getattr(e, "bracket", None)
    if br is not None:                                  # stop / target at a distance from the FILL (round 5)
        kind_, d_stop, d_tgt = br
        f_ = (lambda x: fill * float(x) / 100.0) if kind_ == "pct" else (lambda x: float(x))
        stop = fill - side * f_(d_stop) if d_stop is not None else (float(e.stop) if e.stop is not None else np.nan)
        tgt_ = fill + side * f_(d_tgt) if d_tgt is not None else None
    else:
        stop = float(e.stop) if e.stop is not None else np.nan
        tgt_ = e.target
    if not np.isfinite(stop) or side * (raw - stop) <= 0:
        return None                                     # no stop, or already through it at the open
    risk = abs(fill - stop)
    target = tgt_
    if target is None and e.rr:
        target = fill + side * e.rr * risk
    if target is not None:
        target = float(target)
        if not np.isfinite(target) or side * (target - fill) <= 0:
            return None
        if tick is None:
            tick = equity_tick(target, s.date)
    lim = s.idx(SQUAREOFF)
    reason = "squareoff"
    if e.time_stop:
        ts = s.idx(int(s.t[j]) + int(e.time_stop))
        if ts < lim:
            lim, reason = ts, "time"
    xs = getattr(e, "exit_sig", None)
    if xs is not None:                                  # exit decided at the close of bar k >= j, filled at open k+1
        xs = np.asarray(xs, bool)[j:max(j, min(lim - 1, len(xs)))]
        if xs.size and xs.any():
            lim, reason = j + int(xs.argmax()) + 1, "signal"
    h, l, o = s.h, s.l, s.o
    if e.trail is None:
        if side > 0:
            hs = l[j:lim] <= stop
            ht = h[j:lim] >= target + tick - 1e-9 if target is not None else np.zeros(max(lim - j, 0), bool)
        else:
            hs = h[j:lim] >= stop
            ht = l[j:lim] <= target - tick + 1e-9 if target is not None else np.zeros(max(lim - j, 0), bool)
        hit = hs | ht
        if hit.size and hit.any():
            r = int(hit.argmax())
            k = j + r
            if hs[r]:                                   # stop wins when both are touched
                px = min(stop, float(o[k])) if side > 0 else max(stop, float(o[k]))
                return k, fill, px * (1.0 - side * sl), "stop"
            return k, fill, target_fill(side, target, float(o[k]), float(h[k] if side > 0 else l[k]), tick), "target"
    else:
        mode, x = e.trail[0], float(e.trail[1])
        a = float(s.atr5[e.i]) if mode == "atr" else 0.0
        if mode == "atr" and not np.isfinite(a):
            a = risk
        best = fill
        for k in range(j, lim):
            if side * (float(o[k]) - stop) <= 0 or (l[k] <= stop if side > 0 else h[k] >= stop):
                px = min(stop, float(o[k])) if side > 0 else max(stop, float(o[k]))
                return k, fill, px * (1.0 - side * sl), "stop"
            if target is not None:
                tp = target_fill(side, target, float(o[k]), float(h[k] if side > 0 else l[k]), tick)
                if tp is not None:
                    return k, fill, tp, "target"
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
        return lim, fill, float(o[lim]) * (1.0 - side * sl), reason
    return n - 1, fill, float(s.c[n - 1]) * (1.0 - side * sl), "squareoff"


def _take(spec: Spec, s: Session, e: Entry, trades: list, slip: float | None = None) -> int | None:
    tick = INDEX_TICK if getattr(spec, "underlying", None) else None
    r = fill_intraday(s, e, spec_slippage(spec) if slip is None else slip, tick)
    if r is None:
        return None
    k, fill, px, reason = r
    trades.append(_settle(spec, s.symbol, s.date, e.side, _hm(s.t[e.i + 1]), fill, s.date, _hm(s.t[k]), px, reason))
    # after an exit at the open of bar k decided at bar k-1 (signal), an entry decided at bar k-1 fills at that
    # same open (stop-and-reverse); every other exit frees the symbol from bar k on, as before
    return k - 1 if reason == "signal" else k


def sim_session(spec: Spec, s: Session, slip: float | None = None) -> tuple[list, int]:
    """All trades of one strategy in one symbol-session. Returns (trades, skipped_entries).
    slip = per-side slippage fraction for this run (default: spec_slippage(spec))."""
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
            k = _take(spec, s, e, trades, slip)
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
            k = _take(spec, s, e, trades, slip)
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
            key = (int(e.i), int(e.side), f(e.stop), f(e.target), f(e.rr), e.time_stop, e.trail)
            br, xs = getattr(e, "bracket", None), getattr(e, "exit_sig", None)
            if br is not None:
                key = key + (("bracket",) + tuple(br),)
            if xs is not None:                          # the exit signal up to the cut must not change either
                key = key + (np.packbits(np.asarray(xs, bool)[:upto + 1]).tobytes(),)
            out.append(repr(key))
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


def _order_trade(spec: Spec, sd, d: dict, sig: Swing, e_day: int, nd: int, hi_day: int, slip: float,
                 nifty, vix) -> tuple[list, int, int]:
    """Round 7: a Swing with a resting entry order (see api.Swing.entry), on daily bars. Returns (trades, skipped,
    next decision day). Used only when Swing.entry is set; nothing else in the simulator depends on it."""
    side = int(sig.side)
    kind, level = str(sig.entry[0]), float(sig.entry[1])
    arm = float(sig.entry[2]) if len(sig.entry) > 2 and sig.entry[2] is not None else None
    if kind not in ("stop", "limit"):
        raise ValueError(f"{spec.id}: Swing.entry kind must be 'stop' or 'limit'")
    k_fill = fill = None
    for k in range(e_day, min(e_day + max(1, int(sig.valid)), hi_day + 1)):  # never fills past the period
        o, h, l = float(d["o"][k]), float(d["h"][k]), float(d["l"][k])
        if kind == "stop":
            if arm is not None:                         # active only after trading beyond `arm`: fill at the level
                hit = ((l < arm) and (h >= level)) if side > 0 else ((h > arm) and (l <= level))
                raw = level
            else:
                hit = (h >= level) if side > 0 else (l <= level)
                raw = max(o, level) if side > 0 else min(o, level)
            if hit:
                k_fill, fill = k, raw * (1.0 + side * slip)
                break
        else:
            px = target_fill(-side, level, o, l if side > 0 else h, _tick(spec, level, d["date"][k]))
            if px is not None:
                k_fill, fill = k, float(px)
                break
    if k_fill is None:
        return [], 0, e_day                             # not filled: ask again on the next session
    stop = sig.stop if sig.stop is not None else (fill * (1 - side * sig.stop_pct / 100.0) if sig.stop_pct else None)
    target = sig.target if sig.target is not None else (
        fill * (1 + side * sig.target_pct / 100.0) if sig.target_pct else None)
    if target is None and sig.rr and stop is not None:
        target = fill + side * sig.rr * abs(fill - stop)
    if sig.entry_stop_ticks is not None:                # "one tick under today's low", fixed at the fill session
        tk = int(sig.entry_stop_ticks) * _tick(spec, fill, d["date"][k_fill])
        if side > 0:
            stop = min(np.inf if stop is None else float(stop), float(d["l"][k_fill])) - tk
        else:
            stop = max(-np.inf if stop is None else float(stop), float(d["h"][k_fill])) + tk
    if (stop is not None and side * (fill - stop) <= 0) or (target is not None and side * (target - fill) <= 0) \
            or sig.max_hold < 1:
        return [], 1, k_fill
    natural = k_fill + sig.max_hold - 1
    last_day = min(natural, nd - 1, hi_day)
    out = None
    for k in range(k_fill, last_day + 1):
        kh, kl, ko = float(d["h"][k]), float(d["l"][k]), float(d["o"][k])
        if k == k_fill:                                 # fill session: the stop only, taken as after the fill
            if stop is not None and (kl <= stop if side > 0 else kh >= stop):
                out = (k, float(stop) * (1.0 - side * slip), "stop")
                break
        else:
            if stop is not None and (kl <= stop if side > 0 else kh >= stop):
                px = min(stop, ko) if side > 0 else max(stop, ko)
                out = (k, px * (1.0 - side * slip), "stop")
                break
            if target is not None:
                tp = target_fill(side, float(target), ko, kh if side > 0 else kl,
                                 _tick(spec, float(target), d["date"][k]))
                if tp is not None:
                    out = (k, tp, "target")
                    break
        xf = getattr(sig, "exit_fn", None)
        if xf is not None and k < last_day:
            xv = DailyView(d["date"][:k + 1], *[d[c][:k + 1] for c in "ohlcv"])
            xv.symbol = sd.symbol
            xv.nifty = None if nifty is None else nifty[:k + 1]
            xv.vix = None if vix is None else vix[:k + 1]
            if xf(xv):
                out = (k + 1, float(d["o"][k + 1]) * (1.0 - side * slip), "signal")
                break
    if out is None:
        out = (last_day, float(d["c"][last_day]) * (1.0 - side * slip),
               "max_hold" if last_day == natural else ("period_end" if last_day < nd - 1 else "data_end"))
    k, px, reason = out
    tr = _settle(spec, sd.symbol, d["date"][k_fill], side, "order", fill, d["date"][k],
                 "15:29" if reason in ("max_hold", "data_end", "period_end") else ("09:15" if reason == "signal" else ""),
                 px, reason, getattr(sig, "lots", None))
    return [tr], 0, (k - 1 if reason == "signal" else k)


def sim_positional(spec: Spec, sd, lo: int, hi: int, nifty_day: dict | None = None,
                   vix_day: dict | None = None, slip: float | None = None) -> tuple[list, int]:
    """All trades of one positional strategy in one symbol; entries are taken only on dates in [lo, hi].
    slip = per-side slippage fraction for this run (default: spec_slippage(spec))."""
    d = sd.daily()
    nd = len(d["c"])
    nifty, vix = _aligned(d["date"], nifty_day), _aligned(d["date"], vix_day)
    trades: list = []
    skipped = 0
    p = spec.p
    slip = spec_slippage(spec) if slip is None else float(slip)
    # A trade entered in a period is also EXITED inside it: at the latest at the close of the last session <= hi,
    # so DISCOVERY trades never use FINAL-period prices (reason "period_end").
    hi_day = int(np.searchsorted(d["date"], hi, side="right")) - 1
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
        if getattr(sig, "entry", None) is not None:     # round 7: resting entry order (next-open specs only)
            if intr:
                raise ValueError(f"{spec.id}: Swing.entry needs a next-open spec (no decision_time)")
            tr, sk, nxt = _order_trade(spec, sd, d, sig, e_day, nd, hi_day, slip, nifty, vix)
            trades.extend(tr)
            skipped += sk
            day = max(nxt, day + 1)
            continue
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
        natural = e_day + sig.max_hold - (0 if intr else 1)
        last_day = min(natural, nd - 1, hi_day)         # never walks past the period being simulated
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
            if target is not None:
                tp = target_fill(side, float(target), ko, kh if side > 0 else kl,
                                 _tick(spec, float(target), d["date"][k]))
                if tp is not None:
                    out = (k, tp, "target")
                    break
            xf = getattr(sig, "exit_fn", None)
            if xf is not None and k < last_day:         # decided at the close of k, exit at the open of k+1
                xv = DailyView(d["date"][:k + 1], *[d[c][:k + 1] for c in "ohlcv"])
                xv.symbol = sd.symbol
                xv.nifty = None if nifty is None else nifty[:k + 1]
                xv.vix = None if vix is None else vix[:k + 1]
                if xf(xv):
                    out = (k + 1, float(d["o"][k + 1]) * (1.0 - side * slip), "signal")
                    break
        if out is None:
            out = (last_day, float(d["c"][last_day]) * (1.0 - side * slip),
                   "max_hold" if last_day == natural else ("period_end" if last_day < nd - 1 else "data_end"))
        k, px, reason = out
        et = _hm(sd.t[m]) if intr else "09:15"
        trades.append(_settle(spec, sd.symbol, d["date"][e_day], side, et, fill, d["date"][k],
                              "15:29" if reason in ("max_hold", "data_end", "period_end") else
                              ("09:15" if reason == "signal" else ""), px, reason, getattr(sig, "lots", None)))
        if reason == "signal":                          # next-open specs: ask again on the signal day, so a
            day = max(k if intr else k - 1, day + 1)    # reversal fills at the same open; decision_time: the exit day
            continue
        day = max(k, day + 1)                           # flat again from the exit day's close
    return trades, skipped


# ── one symbol, many strategies ─────────────────────────────────────────────
def applies(spec: Spec, symbol: str) -> bool:
    """Index-futures specs run only on their index's minute bars; every other spec only on stocks."""
    u = getattr(spec, "underlying", None)
    if u:
        return symbol == INDEX_DATA[u]
    return symbol not in INDEX_DATA.values()


def run_symbol(cache_dir: str, symbol: str, specs: list[Spec], lo: int, hi: int, audit: bool = True,
               sd=None, index=None, slippage: float | None = None, index_slippage: float | None = None) -> dict:
    """Run every spec that applies to `symbol` (see `applies`) for entry dates in [lo, hi].
    slippage / index_slippage = per-side fractions for this run (default SLIPPAGE / INDEX_SLIPPAGE).
    Returns {id: {"trades": [...], "skipped": n, "lookahead": str|None, "error": str|None, "sessions": n}}."""
    out = {sp.id: {"trades": [], "skipped": 0, "lookahead": None, "error": None, "sessions": 0, "secs": 0.0}
           for sp in specs}
    specs = [sp for sp in specs if applies(sp, symbol)]
    if not specs:
        return out
    sd = sd if sd is not None else _data.load_symbol(cache_dir, symbol)
    if sd is None:
        return out
    slip = {sp.id: spec_slippage(sp, slippage, index_slippage) for sp in specs}
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
                    tr, sk = sim_session(sp, s, slip[sp.id])
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
                tr, sk = sim_positional(sp, sd, lo, hi, nifty_day, vix_day, slip[sp.id])
                res["trades"].extend(tr)
                res["skipped"] += sk
                res["sessions"] = int(((sd.days >= lo) & (sd.days <= hi)).sum())
            except Exception as exc:
                res["error"] = f"{symbol}: {type(exc).__name__}: {str(exc)[:160]}"
            res["secs"] = time.perf_counter() - t0
    return out
