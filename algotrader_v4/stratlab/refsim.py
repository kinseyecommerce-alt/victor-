"""
stratlab/refsim.py — an INDEPENDENT reference backtester, written for the simulator audit (Nov 2026).

It does not import sim.py, data.py, indicators.py, candles.py, costs.py or any catalogue module. It reads the raw
bar JSON files itself and implements the execution rules exactly as the README states them, so a trade-by-trade
comparison with sim.py tests the simulator (and data.py's parsing) against a second, separately written engine.

Execution rules (README "Simulator assumptions", re-implemented here from the text):
  data       1-minute bars 09:15..15:29 IST, positive prices; a duplicated minute keeps the last row; sessions with
             fewer than 200 bars are dropped (so "previous session" = previous FULL session)
  entry      decided at the close of bar i -> filled at the OPEN of bar i+1, `slip` worse (long pays more, short
             receives less); no fill at or after 15:15; an entry whose stop is already through at the fill open,
             or whose target is on the wrong side of the fill, is skipped
  exits      checked on every bar from the fill bar on, in this order (TIE RULE: stop first):
               stop    bar low <= stop (long) / high >= stop (short): fill at min(open, stop) (long) /
                       max(open, stop) (short) - a gap through the stop fills at the open - and pays `slip`
               target  a resting limit: needs a trade at least one tick THROUGH it (high >= target + tick for a
                       long); fills at the open if the bar opens a tick or more beyond it, else at the target; no slip
             time stop = open of the first bar >= fill minute + N; an exit signal at the close of bar k exits at the
             open of k+1; square-off at the open of the first bar >= 15:15 (or the last close); these pay `slip`
  tick       Rs 0.05; Rs 0.01 for a target price under Rs 250 from 2025-06-16 (lab rule); index 0.05
  size       floor(Rs 1,00,000 / fill) shares; one position per symbol; max_per_day trades per symbol per day
  positional decision at the daily close -> fill at the next session's open (or, with a decision minute, that
             minute's open); stop / target on daily high / low, stop first, gap through the stop fills at the open;
             closed at the close of the max_hold-th session counted from the fill session, or at the close of the
             last session of the period
Costs from first principles, Zerodha's published schedule (zerodha.com/charges, fetched 2026-11-15):
  brokerage  intraday & futures: min(0.03 % of the order value, Rs 20) per executed order; delivery: 0
  STT        intraday 0.025 % sell side; delivery 0.1 % buy and sell; futures 0.05 % sell side
  exchange   NSE transaction charge 0.00307 % (equity), 0.00183 % (futures) of turnover
  SEBI       Rs 10 per crore of turnover;  IPFT Rs 0.01 per crore (ignored: < Rs 0.01 per trade)
  stamp      buy side: 0.003 % intraday, 0.015 % delivery, 0.002 % futures
  GST        18 % on (brokerage + exchange charge + SEBI fee)
  DP         Rs 15.34 per scrip per delivery sell day, GST included
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SQ = 15 * 60 + 15
OPEN = 9 * 60 + 15
LAST = 15 * 60 + 29
MIN_BARS = 200
NOTIONAL = 100_000.0

# ── costs ────────────────────────────────────────────────────────────────────
ZERODHA = {   # product: (brokerage pct, brokerage cap per order, stt buy, stt sell, exchange, stamp buy, dp per sell)
    "MIS": (0.0003, 20.0, 0.0, 0.00025, 0.0000307, 0.00003, 0.0),
    "CNC": (0.0, 0.0, 0.001, 0.001, 0.0000307, 0.00015, 15.34),
    "FUT": (0.0003, 20.0, 0.0, 0.0005, 0.0000183, 0.00002, 0.0),
}


def zerodha_charges(buy_value: float, sell_value: float, product: str) -> float:
    bp, cap, stb, sts, ex, st, dp = ZERODHA[product]
    brok = sum(min(v * bp, cap) for v in (buy_value, sell_value) if v > 0)
    turnover = buy_value + sell_value
    exch = turnover * ex
    sebi = turnover * 10.0 / 1e7
    total = brok + buy_value * stb + sell_value * sts + exch + sebi + buy_value * st + 0.18 * (brok + exch + sebi)
    return round(total + (dp if sell_value > 0 else 0.0), 2)


def tick_for(price: float, date: int, index: bool = False) -> float:
    if index:
        return 0.05
    return 0.01 if (price < 250.0 and date >= 20250616) else 0.05


# ── data ─────────────────────────────────────────────────────────────────────
@dataclass
class Sess:
    date: int
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    k: int = 0                       # session ordinal within the symbol
    extra: dict = field(default_factory=dict)

    def idx(self, minute: int) -> int:
        """first bar at or after `minute` (len if none)"""
        return int(np.searchsorted(self.t, minute, side="left"))


def read_bars(path: str) -> pd.DataFrame:
    """Raw JSON (one symbol per file, {"bars": [[ts, o, h, l, c, v], ...]}) -> cleaned minute frame."""
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt") as fh:
            js = json.load(fh)
    else:
        with open(path) as fh:
            js = json.load(fh)
    rows = js["bars"] if isinstance(js, dict) and "bars" in js else []
    if not rows:
        return pd.DataFrame(columns=["date", "t", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"][:len(rows[0])])
    if "v" not in df:
        df["v"] = 0.0
    ts = df["ts"].astype(str)
    df["date"] = (ts.str.slice(0, 4) + ts.str.slice(5, 7) + ts.str.slice(8, 10)).astype(int)
    df["t"] = ts.str.slice(11, 13).astype(int) * 60 + ts.str.slice(14, 16).astype(int)
    for k in "ohlcv":
        df[k] = pd.to_numeric(df[k], errors="coerce").astype(float)
    df["v"] = df["v"].fillna(0.0)
    ok = df["t"].between(OPEN, LAST) & (df[["o", "h", "l", "c"]] > 0).all(axis=1) \
        & np.isfinite(df[["o", "h", "l", "c"]]).all(axis=1)
    df = df[ok].sort_values(["date", "t"], kind="stable").drop_duplicates(["date", "t"], keep="last")
    return df[["date", "t", "o", "h", "l", "c", "v"]].reset_index(drop=True)


def sessions(df: pd.DataFrame) -> list:
    out = []
    for d, g in df.groupby("date", sort=True):
        if len(g) < MIN_BARS:
            continue
        out.append(Sess(int(d), g["t"].to_numpy(int), *(g[k].to_numpy(float) for k in "ohlcv")))
    for k, s in enumerate(out):
        s.k = k
    return out


def daily(ss: list) -> pd.DataFrame:
    return pd.DataFrame({"date": [s.date for s in ss], "o": [s.o[0] for s in ss], "h": [s.h.max() for s in ss],
                         "l": [s.l.min() for s in ss], "c": [s.c[-1] for s in ss], "v": [s.v.sum() for s in ss]})


# ── textbook indicators (own code) ──────────────────────────────────────────
def ema_rec(x, n):
    a = 2.0 / (n + 1.0)
    out = np.empty(len(x))
    m = float(x[0])
    for i, xi in enumerate(x):
        m = a * xi + (1 - a) * m if i else float(xi)
        out[i] = m
    return out


def wilder_atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    a = 1.0 / n
    out = np.empty(len(tr))
    m = tr[0]
    for i, x in enumerate(tr):
        m = a * x + (1 - a) * m if i else x
        out[i] = m
    return out


def vwap(s: Sess):
    tp = (s.h + s.l + s.c) / 3.0
    cv = np.cumsum(s.v)
    return np.where(cv > 0, np.cumsum(tp * s.v) / np.where(cv > 0, cv, 1), tp)


# ── engines ──────────────────────────────────────────────────────────────────
@dataclass
class E:                    # an intraday entry decided at the close of bar i
    i: int
    side: int
    stop: float
    target: float | None = None
    rr: float | None = None
    hold: int | None = None
    xsig: np.ndarray | None = None


def _trade(sid, sym, date, side, et, fill, xd, xt, xp, product, qty_px=None):
    qty = max(1, int(NOTIONAL // (fill if qty_px is None else qty_px)))
    gross = side * (xp - fill) * qty
    bv, sv = (fill * qty, xp * qty) if side > 0 else (xp * qty, fill * qty)
    cost = zerodha_charges(bv, sv, product)
    return {"id": sid, "symbol": sym, "date": int(date), "side": int(side), "entry_t": et, "entry": fill,
            "exit_date": int(xd), "exit_t": xt, "exit": xp, "qty": qty, "gross": gross, "cost": cost,
            "net": gross - cost}


def hm(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def intraday_one(s: Sess, e: E, slip: float, index: bool = False):
    n = len(s.t)
    j = e.i + 1
    if e.side not in (1, -1) or j >= n or s.t[j] >= SQ:
        return None
    side = e.side
    raw = s.o[j]
    fill = raw * (1 + side * slip)
    stop = e.stop
    if stop is None or not np.isfinite(stop) or side * (raw - stop) <= 0:
        return None
    tgt = e.target if e.target is not None else (fill + side * e.rr * abs(fill - stop) if e.rr else None)
    if tgt is not None and side * (tgt - fill) <= 0:
        return None
    lim = s.idx(SQ)
    why = "squareoff"
    if e.hold:
        ts = s.idx(int(s.t[j]) + int(e.hold))
        if ts < lim:
            lim, why = ts, "time"
    if e.xsig is not None:
        for k in range(j, min(lim - 1, len(e.xsig))):
            if e.xsig[k]:
                lim, why = k + 1, "signal"
                break
    tk = tick_for(tgt, s.date, index) if tgt is not None else 0.0
    for k in range(j, lim):
        if (s.l[k] <= stop) if side > 0 else (s.h[k] >= stop):
            px = (min(s.o[k], stop) if side > 0 else max(s.o[k], stop)) * (1 - side * slip)
            return j, k, fill, px, "stop"
        if tgt is not None:
            ext = s.h[k] if side > 0 else s.l[k]
            if side * (ext - tgt) >= tk - 1e-9:
                px = s.o[k] if side * (s.o[k] - tgt) >= tk - 1e-9 else tgt
                return j, k, fill, px, "target"
    if lim < n:
        return j, lim, fill, s.o[lim] * (1 - side * slip), why
    return j, n - 1, fill, s.c[n - 1] * (1 - side * slip), "squareoff"


def run_intraday(sid, sym, s: Sess, ents: list, max_per_day: int, slip: float, product: str, index=False):
    out = []
    free = 0
    for e in sorted(ents, key=lambda x: x.i):
        if len(out) >= max_per_day:
            break
        if e.i < free:
            continue
        r = intraday_one(s, e, slip, index)
        if r is None:
            continue
        j, k, fill, px, why = r
        out.append(_trade(sid, sym, s.date, e.side, hm(s.t[j]), fill, s.date, hm(s.t[k]), px, product))
        free = k - 1 if why == "signal" else k
    return out


@dataclass
class P:                     # a positional entry
    side: int
    stop_pct: float | None = None
    target_pct: float | None = None
    stop: float | None = None
    target: float | None = None
    rr: float | None = None
    max_hold: int = 10


def run_positional(sid, sym, ss: list, decide, lo: int, hi: int, slip: float, product: str,
                   decision_minute: int | None = None):
    """decide(day_index) -> P or None. Without decision_minute the decision uses the close of day d and fills at
    the open of d+1; with it, the decision is taken on day d at that minute (using only earlier days) and fills
    at the open of that minute's bar on day d."""
    D = daily(ss)
    dt, o, h, l, c = (D[k].to_numpy() for k in ("date", "o", "h", "l", "c"))
    nd = len(dt)
    last = int(np.searchsorted(dt, hi, side="right")) - 1
    out = []
    d = 0
    while d < nd:
        e = d if decision_minute is not None else d + 1
        if e >= nd or dt[e] > hi:
            break
        if dt[e] < lo:
            d += 1
            continue
        if decision_minute is not None:
            s = ss[d]
            m = s.idx(decision_minute)
            if m == 0 or m >= len(s.t):
                d += 1
                continue
        sig = decide(d)
        if sig is None:
            d += 1
            continue
        side = sig.side
        raw = float(ss[d].o[m]) if decision_minute is not None else float(o[e])
        fill = raw * (1 + side * slip)
        stop = sig.stop if sig.stop is not None else (fill * (1 - side * sig.stop_pct / 100) if sig.stop_pct else None)
        tgt = sig.target if sig.target is not None else (fill * (1 + side * sig.target_pct / 100) if sig.target_pct else None)
        if tgt is None and sig.rr and stop is not None:
            tgt = fill + side * sig.rr * abs(fill - stop)
        if (stop is not None and side * (raw - stop) <= 0) or (tgt is not None and side * (tgt - fill) <= 0):
            d += 1
            continue
        natural = e + sig.max_hold - 1
        end = min(natural, nd - 1, last)
        res = None
        for k in range(e, end + 1):
            if k == e:
                if decision_minute is not None:
                    kh, kl = ss[k].h[m:].max(), ss[k].l[m:].min()
                else:
                    kh, kl = h[k], l[k]
                ko = raw
            else:
                kh, kl, ko = h[k], l[k], o[k]
            if stop is not None and ((kl <= stop) if side > 0 else (kh >= stop)):
                res = (k, (min(ko, stop) if side > 0 else max(ko, stop)) * (1 - side * slip), "stop")
                break
            if tgt is not None:
                tk = tick_for(tgt, int(dt[k]))
                ext = kh if side > 0 else kl
                if side * (ext - tgt) >= tk - 1e-9:
                    res = (k, ko if side * (ko - tgt) >= tk - 1e-9 else tgt, "target")
                    break
        if res is None:
            res = (end, c[end] * (1 - side * slip), "max_hold" if end == natural else "period_end")
        k, px, why = res
        et = hm(ss[d].t[m]) if decision_minute is not None else "09:15"
        out.append(_trade(sid, sym, dt[e], side, et, fill, dt[k], "", px, product))
        d = max(k, d + 1)
    return out


# ── the strategies, re-implemented from their one-line descriptions ─────────
def _orb(s: Sess, minutes: int):
    k = s.idx(OPEN + minutes)
    if k == 0:
        return None
    return s.h[:k].max(), s.l[:k].min(), k


def _first(mask, a, b):
    m = np.flatnonzero(mask[a:b])
    return a + int(m[0]) if len(m) else -1


def _orb_entries(s, minutes, rr, last, long_ok=True, short_ok=True, hold=None, mid=False):
    r = _orb(s, minutes)
    if r is None:
        return []
    hi, lo, k = r
    if not hi > lo:
        return []
    end = s.idx(last)
    stop_l = (hi + lo) / 2 if mid else lo
    stop_s = (hi + lo) / 2 if mid else hi
    out = []
    i = _first(s.c > hi, k, end)
    if i >= 0 and long_ok:
        out.append(E(i, 1, stop_l, rr=rr, hold=hold))
    i = _first(s.c < lo, k, end)
    if i >= 0 and short_ok:
        out.append(E(i, -1, stop_s, rr=rr, hold=hold))
    return out


def H(x):
    a, b = x.split(":")
    return int(a) * 60 + int(b)


def ref_orb15_break(ctx, s):
    return _orb_entries(s, 15, 1.5, H("12:00"))


def ref_fut_orb15_break(ctx, s):
    return _orb_entries(s, 15, 1.5, H("12:00"))


def ref_gap_and_go(ctx, s):
    pdc = ctx["prev_close"].get(s.k)
    if pdc is None:
        return []
    g = (s.o[0] / pdc - 1) * 100
    if not 0.5 <= abs(g) <= 3.0:
        return []
    return _orb_entries(s, 5, 2.0, H("10:30"), long_ok=g > 0, short_ok=g < 0)


def ref_pdh_pdl_reject(ctx, s):
    pd_ = ctx["prev_day"].get(s.k)
    if pd_ is None:
        return []
    pdh, pdl = pd_
    win = (s.t >= H("09:30")) & (s.t <= H("14:00"))
    dh, dl = np.maximum.accumulate(s.h), np.minimum.accumulate(s.l)
    out = []
    i = _first(win & (s.h > pdh) & (s.c < pdh) & (s.c < s.o), 0, len(s.t))
    if i >= 0:
        out.append(E(i, -1, dh[i] * 1.0005, rr=1.5))
    i = _first(win & (s.l < pdl) & (s.c > pdl) & (s.c > s.o), 0, len(s.t))
    if i >= 0:
        out.append(E(i, 1, dl[i] * 0.9995, rr=1.5))
    return out


def ref_close_momentum(ctx, s):
    i = s.idx(H("14:29"))
    if i >= len(s.t):
        return []
    vw = vwap(s)
    mv = (s.c[i] / s.o[0] - 1) * 100
    if mv >= 1.0 and s.c[i] > vw[i]:
        return [E(i, 1, vw[i])]
    if mv <= -1.0 and s.c[i] < vw[i]:
        return [E(i, -1, vw[i])]
    return []


def ref_rs_vs_nifty(ctx, s):
    nx = ctx["nifty"].get(s.date)
    i = s.idx(H("10:29"))
    if nx is None or i >= len(s.t):
        return []
    nt, nc = nx

    def at(m):
        p = int(np.searchsorted(nt, m, side="right")) - 1
        return nc[p] if p >= 0 else np.nan
    n0, ni = at(s.t[0]), at(s.t[i])
    if not (np.isfinite(n0) and np.isfinite(ni)):
        return []
    vw = vwap(s)
    rs = (s.c[i] / s.o[0] - ni / n0) * 100
    if rs >= 1.0 and s.c[i] > vw[i]:
        return [E(i, 1, vw[i], rr=2.0)]
    if rs <= -1.0 and s.c[i] < vw[i]:
        return [E(i, -1, vw[i], rr=2.0)]
    return []


def ref_rev_1000(ctx, s):
    k = s.idx(H("10:00"))
    if k >= len(s.t) or k < 1:
        return []
    e21 = ctx["ema21"][s.k]
    mv = (s.c[k - 1] / s.o[0] - 1) * 100
    end = s.idx(H("10:45"))
    if mv >= 0.75:
        i = _first(s.c < e21, k, end)
        return [E(i, -1, np.maximum.accumulate(s.h)[i] * 1.0005, rr=1.5)] if i >= 0 else []
    if mv <= -0.75:
        i = _first(s.c > e21, k, end)
        return [E(i, 1, np.minimum.accumulate(s.l)[i] * 0.9995, rr=1.5)] if i >= 0 else []
    return []


def ref_scalp_orb5(ctx, s):
    return _orb_entries(s, 5, 1.0, H("10:30"), hold=20, mid=True)


def ref_scalp_open_drive(ctx, s):
    if len(s.t) < 3 or s.t[2] != H("09:17"):
        return []
    c, o = s.c[:3], s.o[:3]
    if (c > o).all() and c[2] > c[1] > c[0]:
        return [E(2, 1, s.o[0], rr=1.0, hold=15)]
    if (c < o).all() and c[2] < c[1] < c[0]:
        return [E(2, -1, s.o[0], rr=1.0, hold=15)]
    return []


def _candle(s: Sess, a: int, b: int):
    """candle from minute a (incl.) to b (excl.): (o, h, l, c, v, last bar index) or None"""
    i0, i1 = s.idx(a), s.idx(b)
    if i1 <= i0:
        return None
    return s.o[i0], s.h[i0:i1].max(), s.l[i0:i1].min(), s.c[i1 - 1], s.v[i0:i1].sum(), i1 - 1


def ref_r3_tf5m_open5_rvol(ctx, s):
    pdc = ctx["prev_close"].get(s.k)
    if pdc is None or s.t[0] != OPEN:
        return []
    g = s.o[0] / pdc - 1
    hist = ctx["first5"][max(0, s.k - 10):s.k]
    hist = [x for x in hist if x is not None]
    if s.k < 10 or len(hist) < 5:
        return []
    cd = _candle(s, OPEN, OPEN + 5)
    o, h, l, c, v, j = cd
    mv = float(np.mean(hist))
    if mv <= 0 or v < 3.0 * mv or h - l <= 0 or abs(c - o) < 0.6 * (h - l) or g == 0:
        return []
    side = 1 if c > o else -1
    if np.sign(g) != side:
        return []
    return [E(j, side, l if side > 0 else h)]


def ref_r3_tf15m_slotvol_shock(ctx, s):
    out = []
    prior_hi = prior_lo = None
    for a in range(OPEN, SQ, 15):
        cd = _candle(s, a, a + 15)
        if cd is None:
            continue
        o, h, l, c, v, j = cd
        if H("09:45") <= a <= H("14:00") and prior_hi is not None:
            hist = [x for x in ctx["slot15"][a][max(0, s.k - 10):s.k] if x is not None]
            if len(hist) >= 10:
                mv = float(np.mean(hist))
                rng = h - l
                if mv > 0 and v >= 3.0 * mv and rng > 0 and abs(c - o) >= 0.6 * rng:
                    if c > o and c > prior_hi:
                        out.append(E(j, 1, l))
                    elif c < o and c < prior_lo:
                        out.append(E(j, -1, h))
        prior_hi = h if prior_hi is None else max(prior_hi, h)
        prior_lo = l if prior_lo is None else min(prior_lo, l)
    return out


def ref_r3_tf60m_hour_volshock(ctx, s):
    out = []
    for a in range(OPEN, H("14:15") + 1, 60):
        end = a + 60
        if not H("11:15") <= end <= H("14:15"):
            continue
        cd = _candle(s, a, end)
        if cd is None:
            continue
        o, h, l, c, v, j = cd
        hist = [x for x in ctx["slot60"][a][max(0, s.k - 20):s.k] if x is not None]
        atr = ctx["atr60"].get((s.k, a))
        if len(hist) < 20 or atr is None or not np.isfinite(atr) or h <= l:
            continue
        mv = float(np.mean(hist))
        if mv <= 0 or v < 3.0 * mv:
            continue
        pos = (c - l) / (h - l)
        if c > o and pos >= 0.75:
            out.append(E(j, 1, min(l, c - 0.5 * atr)))
        elif c < o and pos <= 0.25:
            out.append(E(j, -1, max(h, c + 0.5 * atr)))
    return out


def ref_r5_kite_tt_orb15_stock(ctx, s):
    r = _orb(s, 15)
    if r is None:
        return []
    hi, lo, k = r
    if not hi > lo:
        return []
    rng = hi - lo
    out = []
    i = _first(s.c > hi, k, s.idx(SQ))
    if i >= 0:
        out.append(("bracket", E(i, 1, lo), rng))
    i = _first(s.c < lo, k, s.idx(SQ))
    if i >= 0:
        out.append(("bracket", E(i, -1, hi), rng))
    return out


def ref_r7_gh2_oops_stock(ctx, s):
    pd_ = ctx["prev_day"].get(s.k)
    if pd_ is None:
        return []
    pdh, pdl = pd_
    o0 = s.o[0]
    if o0 < pdl:
        side, lvl = 1, pdl
    elif o0 > pdh:
        side, lvl = -1, pdh
    else:
        return []
    hit = (s.h >= lvl) if side > 0 else (s.l <= lvl)
    m = _first(hit, 0, len(s.t))
    if m < 0 or s.t[m] + 1 >= SQ:
        return []
    tk = tick_for(lvl, s.date)
    stop = s.l[:m + 1].min() - tk if side > 0 else s.h[:m + 1].max() + tk
    return [E(m, side, stop)]


# ── "@code" variants: the SAME engine, with the reading of an ambiguous one-line description that the
#    catalogue code actually uses (found after the first comparison; see the audit report). They show whether a
#    mismatch is a definition question or an engine difference.
def ref_pdh_pdl_reject_code(ctx, s):
    pd_ = ctx["prev_day"].get(s.k)
    if pd_ is None:
        return []
    pdh, pdl = pd_
    win = (s.t >= H("09:30")) & (s.t < H("14:00"))          # [09:30, 14:00): end exclusive
    dh, dl = np.maximum.accumulate(s.h), np.minimum.accumulate(s.l)
    out = []
    i = _first(win & (s.h > pdh) & (s.c < pdh) & (s.c < s.o), 0, len(s.t))
    if i >= 0:
        out.append(E(i, -1, dh[i] * 1.0005, rr=1.5))
    i = _first(win & (s.l < pdl) & (s.c > pdl) & (s.c > s.o), 0, len(s.t))
    if i >= 0:
        out.append(E(i, 1, dl[i] * 0.9995, rr=1.5))
    return out


def ref_r3_tf15m_slotvol_shock_code(ctx, s):
    """candle END in [09:45, 14:00]; slot mean over the positive volumes of the previous 10 sessions (>= 5)"""
    out = []
    prior_hi = prior_lo = None
    for a in range(OPEN, SQ, 15):
        cd = _candle(s, a, a + 15)
        if cd is None:
            continue
        o, h, l, c, v, j = cd
        if H("09:45") <= a + 15 <= H("14:00") and prior_hi is not None:
            hist = [x for x in ctx["slot15"][a][max(0, s.k - 10):s.k] if x is not None and x > 0]
            if len(hist) >= 5:
                mv = float(np.mean(hist))
                rng = h - l
                if mv > 0 and v >= 3.0 * mv and rng > 0 and abs(c - o) >= 0.6 * rng:
                    if c > o and c > prior_hi:
                        out.append(E(j, 1, l))
                    elif c < o and c < prior_lo:
                        out.append(E(j, -1, h))
        prior_hi = h if prior_hi is None else max(prior_hi, h)
        prior_lo = l if prior_lo is None else min(prior_lo, l)
    return out


def ref_r5_kite_tt_orb15_stock_code(ctx, s):
    """target = the broken edge +- one range (not fill +- range); no entry deadline; 2 trades a day"""
    r = _orb(s, 15)
    if r is None:
        return []
    hi, lo, k = r
    if not hi > lo:
        return []
    out = []
    i = _first(s.c > hi, k, len(s.t))
    if i >= 0:
        out.append(E(i, 1, lo, target=hi + (hi - lo)))
    i = _first(s.c < lo, k, len(s.t))
    if i >= 0:
        out.append(E(i, -1, hi, target=lo - (hi - lo)))
    return out


def ref_r7_gh2_oops_stock_code(ctx, s):
    """the resting stop is ARMED by the gap bar; that bar triggers only if it CLOSES back at / beyond the level
    (inside one bar the order of high and low is unknown), else the first later bar trading at the level"""
    pd_ = ctx["prev_day"].get(s.k)
    if pd_ is None:
        return []
    pdh, pdl = pd_
    o0 = s.o[0]
    if o0 < pdl:
        side, lvl = 1, pdl
    elif o0 > pdh:
        side, lvl = -1, pdh
    else:
        return []
    armed = (s.l < lvl) if side > 0 else (s.h > lvl)
    k = _first(armed, 0, len(s.t))
    if k < 0:
        return []
    if (s.c[k] >= lvl) if side > 0 else (s.c[k] <= lvl):
        m = k
    else:
        hit = (s.h >= lvl) if side > 0 else (s.l <= lvl)
        m = _first(hit, k + 1, len(s.t))
    if m < 0 or s.t[m] + 1 >= SQ:
        return []
    tk = tick_for(lvl, s.date)
    stop = s.l[:m + 1].min() - tk if side > 0 else s.h[:m + 1].max() + tk
    return [E(m, side, stop)]


INTRADAY = {   # id: (function, max_per_day, product)
    "pdh_pdl_reject@code": (ref_pdh_pdl_reject_code, 1, "MIS"),
    "r3_tf15m_slotvol_shock@code": (ref_r3_tf15m_slotvol_shock_code, 1, "MIS"),
    "r5_kite_tt_orb15_stock@code": (ref_r5_kite_tt_orb15_stock_code, 2, "MIS"),
    "r7_gh2_oops_stock@code": (ref_r7_gh2_oops_stock_code, 1, "MIS"),
    "orb15_break": (ref_orb15_break, 1, "MIS"),
    "fut_orb15_break": (ref_fut_orb15_break, 1, "FUT"),
    "gap_and_go": (ref_gap_and_go, 1, "MIS"),
    "pdh_pdl_reject": (ref_pdh_pdl_reject, 1, "MIS"),
    "close_momentum": (ref_close_momentum, 1, "MIS"),
    "rs_vs_nifty": (ref_rs_vs_nifty, 1, "MIS"),
    "rev_1000": (ref_rev_1000, 1, "MIS"),
    "scalp_orb5": (ref_scalp_orb5, 1, "MIS"),
    "scalp_open_drive": (ref_scalp_open_drive, 1, "MIS"),
    "r3_tf5m_open5_rvol": (ref_r3_tf5m_open5_rvol, 1, "MIS"),
    "r3_tf15m_slotvol_shock": (ref_r3_tf15m_slotvol_shock, 1, "MIS"),
    "r3_tf60m_hour_volshock": (ref_r3_tf60m_hour_volshock, 1, "MIS"),
    "r5_kite_tt_orb15_stock": (ref_r5_kite_tt_orb15_stock, 1, "MIS"),
    "r7_gh2_oops_stock": (ref_r7_gh2_oops_stock, 1, "MIS"),
}


# positional
def pos_swing_20d_breakout(ctx, d):
    c, h = ctx["D"]["c"], ctx["D"]["h"]
    if d < 20:
        return None
    return P(1, stop_pct=5.0, target_pct=10.0, max_hold=10) if c[d] > h[d - 20:d].max() else None


def pos_fut_donchian20(ctx, d):
    c, h, l = ctx["D"]["c"], ctx["D"]["h"], ctx["D"]["l"]
    if d < 20:
        return None
    if c[d] > h[d - 20:d].max():
        return P(1, stop_pct=4.0, target_pct=8.0, max_hold=10)
    if c[d] < l[d - 20:d].min():
        return P(-1, stop_pct=4.0, target_pct=8.0, max_hold=10)
    return None


def pos_r3_swing_tom(ctx, d):
    dt = ctx["D"]["date"]
    if d < 1:
        return None
    if dt[d] % 100 >= 25 and (dt[d - 1] // 100 != dt[d] // 100 or dt[d - 1] % 100 < 25):
        return P(1, stop_pct=10.0, max_hold=6)
    return None


def pos_cs_bull_engulfing_d(ctx, d):
    D = ctx["D"]
    o, h, l, c = D["o"], D["h"], D["l"], D["c"]
    e20, atr = ctx["ema20d"], ctx["atr14d"]
    if d < 7:
        return None
    p = d - 2                                           # candle before the pattern's first candle
    if not (c[p] < e20[p] and c[p] < c[p - 5]):
        return None
    if not (c[d - 1] < o[d - 1] and c[d] > o[d] and o[d] <= c[d - 1] and c[d] >= o[d - 1]
            and (o[d] < c[d - 1] or c[d] > o[d - 1])):
        return None
    a = atr[d]
    stop = min(min(l[d - 1], l[d]) - 0.05, c[d] - 0.3 * a)
    return P(1, stop=stop, rr=2.0, max_hold=10)


def _week(x):
    import datetime as dtm
    return (dtm.date(x // 10000, x // 100 % 100, x % 100).toordinal() - 1) // 7


def pos_r3_tfweekly_wk_loser_rev(ctx, d):
    dt = ctx["D"]["date"]
    if d < 2 or _week(int(dt[d])) == _week(int(dt[d - 1])):
        return None
    wk = np.array([_week(int(x)) for x in dt[:d]])
    starts = np.flatnonzero(np.r_[True, wk[1:] != wk[:-1]])
    if len(starts) < 4:
        return None
    ends = np.r_[starts[1:], len(wk)] - 1
    wc = ctx["D"]["c"][ends][1:]                      # completed weeks, the first (possibly partial) dropped
    if len(wc) < 2:
        return None
    return P(1, stop_pct=10.0, max_hold=10) if (wc[-1] / wc[-2] - 1) * 100 <= -10.0 else None


def pos_cs_bull_engulfing_d_code(ctx, d):
    """EMA20 used only once 20 daily candles exist before the trend candle"""
    if d - 2 < 19:
        return None
    return pos_cs_bull_engulfing_d(ctx, d)


def pos_r3_tfweekly_wk_loser_rev_code(ctx, d):
    """with a decision minute the fill session is not counted: closed at the close of fill day + max_hold"""
    r = pos_r3_tfweekly_wk_loser_rev(ctx, d)
    if r is not None:
        r.max_hold += 1
    return r


POSITIONAL = {  # id: (function, product, decision minute)
    "cs_bull_engulfing_d@code": (pos_cs_bull_engulfing_d_code, "CNC", None),
    "r3_tfweekly_wk_loser_rev@code": (pos_r3_tfweekly_wk_loser_rev_code, "CNC", 9 * 60 + 16),
    "swing_20d_breakout": (pos_swing_20d_breakout, "CNC", None),
    "fut_donchian20": (pos_fut_donchian20, "FUT", None),
    "r3_swing_tom": (pos_r3_swing_tom, "CNC", None),
    "cs_bull_engulfing_d": (pos_cs_bull_engulfing_d, "CNC", None),
    "r3_tfweekly_wk_loser_rev": (pos_r3_tfweekly_wk_loser_rev, "CNC", 9 * 60 + 16),
}


def context(ss: list, nifty: dict | None = None) -> dict:
    """Per-symbol inputs the reference strategies use (all causal by construction: a value for session k is
    built from sessions < k, or from session k's own bars up to the bar it is read at)."""
    ctx = {"prev_close": {}, "prev_day": {}, "first5": [], "nifty": nifty or {}}
    for k, s in enumerate(ss):
        if k:
            p = ss[k - 1]
            ctx["prev_close"][k] = p.c[-1]
            ctx["prev_day"][k] = (p.h.max(), p.l.min())
        cd = _candle(s, OPEN, OPEN + 5) if s.t[0] == OPEN else None
        ctx["first5"].append(cd[4] if cd else None)
    allc = np.concatenate([s.c for s in ss]) if ss else np.zeros(0)
    e21 = ema_rec(allc, 21) if len(allc) else allc
    off = 0
    ctx["ema21"] = []
    for s in ss:
        ctx["ema21"].append(e21[off:off + len(s.t)])
        off += len(s.t)
    ctx["slot15"] = {a: [(_candle(s, a, a + 15) or [None] * 5)[4] for s in ss] for a in range(OPEN, SQ, 15)}
    ctx["slot60"] = {a: [(_candle(s, a, a + 60) or [None] * 5)[4] for s in ss] for a in range(OPEN, SQ, 60)}
    # ATR(14) on the continuous series of full hourly candles (09:15..15:15), value at each candle
    rows, keys = [], []
    for s in ss:
        for a in range(OPEN, SQ, 60):
            cd = _candle(s, a, a + 60)
            if cd:
                rows.append(cd[:4])
                keys.append((s.k, a))
    if rows:
        r = np.array(rows, float)
        at = wilder_atr(r[:, 1], r[:, 2], r[:, 3], 14)
        ctx["atr60"] = {k: (at[i] if i >= 13 else np.nan) for i, k in enumerate(keys)}
    else:
        ctx["atr60"] = {}
    D = daily(ss)
    ctx["D"] = {k: D[k].to_numpy() for k in D.columns}
    if len(D):
        ctx["ema20d"] = ema_rec(ctx["D"]["c"], 20)
        ctx["atr14d"] = wilder_atr(ctx["D"]["h"], ctx["D"]["l"], ctx["D"]["c"], 14)
    return ctx


def run_symbol(sym: str, ss: list, ids: list, lo: int, hi: int, slip: float = 2e-4, nifty=None) -> dict:
    ctx = context(ss, nifty)
    out = {i: [] for i in ids}
    for sid in ids:
        if sid in INTRADAY:
            fn, mpd, prod = INTRADAY[sid]
            for s in ss:
                if not lo <= s.date <= hi:
                    continue
                raw = fn(ctx, s)
                ents = []
                for e in raw:
                    if isinstance(e, tuple):            # ("bracket", entry, distance): target = fill +- distance
                        _, ee, dist = e
                        j = ee.i + 1
                        if j < len(s.t):
                            f = s.o[j] * (1 + ee.side * slip)
                            ee.target = f + ee.side * dist
                        ents.append(ee)
                    else:
                        ents.append(e)
                out[sid].extend(run_intraday(sid, sym, s, ents, mpd, slip, prod))
        elif sid in POSITIONAL:
            fn, prod, dm = POSITIONAL[sid]
            out[sid].extend(run_positional(sid, sym, ss, lambda d: fn(ctx, d), lo, hi, slip, prod, dm))
    return out


# ── module strategies: opening-auction fade and attention overnight ──────────
def auction_rows(sym: str, ss: list, min_gap: float = 0.02) -> list:
    rows = []
    for k in range(1, len(ss)):
        s, pc = ss[k], ss[k - 1].c[-1]
        if s.t[0] != OPEN:
            continue
        g = s.o[0] / pc - 1
        if not (min_gap + 1e-9 < abs(g) <= 0.20):
            continue
        j = s.idx(9 * 60 + 20)
        if j >= len(s.t) or s.t[j] >= 9 * 60 + 30:
            continue
        rows.append((s.date, sym, -1 if g > 0 else 1, g, s.o[0], int(s.t[j]), s.o[j]))
    return rows


def auction_trades(rows, sid, min_gap, lo, hi, slip=2.1e-4) -> list:
    by = {}
    for r in rows:
        if lo <= r[0] <= hi and abs(r[3]) > min_gap + 1e-9:
            by.setdefault(r[0], []).append(r)
    out = []
    for d in sorted(by):
        for date, sym, side, g, apx, xt, xp in sorted(by[d], key=lambda r: (-abs(r[3]), r[1]))[:10]:
            f, x = apx * (1 + side * slip), xp * (1 - side * slip)
            out.append(_trade(sid, sym, date, side, "09:15", f, date, hm(xt), x, "MIS", qty_px=apx))
    return out


def attention_rows(sym: str, ss: list) -> list:
    rows = []
    dv = [s.v.sum() for s in ss]
    for k in range(20, len(ss)):
        s = ss[k]
        j = int(np.searchsorted(s.t, 15 * 60 + 20, side="right")) - 1
        if j < 0 or s.t[j] < 15 * 60 + 15 or j + 1 >= len(s.t) or s.t[j + 1] > 15 * 60 + 24:
            continue
        pc = ss[k - 1].c[-1]
        base = float(np.mean(dv[k - 20:k]))
        if not base > 0:
            continue
        ratio = s.v[:j + 1].sum() / base
        if not (ratio > 3.0 + 1e-12 and s.c[j] > pc):
            continue
        r = {"date": s.date, "symbol": sym, "ratio": ratio, "entry_t": int(s.t[j + 1]), "entry_px": s.o[j + 1],
             "exit_date": None}
        if k + 1 < len(ss) and ss[k + 1].t[0] < 9 * 60 + 30:
            r["exit_date"], r["exit_t"], r["exit_px"] = ss[k + 1].date, int(ss[k + 1].t[0]), ss[k + 1].o[0]
        rows.append(r)
    return rows


def attention_trades(rows, counts: dict, lo, hi, ca=(("SIEMENS", 20250407), ("VEDL", 20260430)), slip=2e-4):
    dates = sorted(counts)
    special = {d for d in dates if float(np.median(counts[d])) < 300}
    nxt = {a: b for a, b in zip(dates, dates[1:])}
    by = {}
    for r in rows:
        if r["exit_date"] is None or not (lo <= r["date"] <= hi and r["exit_date"] <= hi):
            continue
        if nxt.get(r["date"]) != r["exit_date"] or r["date"] in special or r["exit_date"] in special:
            continue
        if (r["symbol"], r["date"]) in ca or (r["symbol"], r["exit_date"]) in ca:
            continue
        by.setdefault(r["date"], []).append(r)
    out = []
    for d in sorted(by):
        for r in sorted(by[d], key=lambda r: (-r["ratio"], r["symbol"]))[:10]:
            out.append(_trade("SWING-ATTN-ON-3X", r["symbol"], r["date"], 1, hm(r["entry_t"]), r["entry_px"] * (1 + slip),
                              r["exit_date"], hm(r["exit_t"]), r["exit_px"] * (1 - slip), "FUT",
                              qty_px=r["entry_px"]))
    return out
