"""
stratlab/builder_engine.py — evaluates builder candidates (stratlab/builder_grammar.py) on a data source
(stratlab/builder_source.py) under the frozen rule builder_rule.json. Research only: no order, no agent.

Reused, not copied: the signal blocks are datagen_features' rule functions (plus four small blocks below), the fill /
cost model is datagen_run._trade (2 bps per side stocks / 1 bps index, 5 / 2.5 bps stress, costs.round_trip_cost:
equity_mis, equity_cnc for delivery longs, future for stock / index futures), the gates are datagen's DISCOVERY gates
and protocol.screen / protocol.final_verdict, DSR is trials.dsr, options run through nsedata's real-option simulator
(stratlab/builder_options.py).

EXECUTION
  minute grid  entry at the open of the first bar at / after the entry minute (the summary field, datagen's rule);
               exits: a summary field of the same session (time), the next 09:15 auction (overnight), 15:21 h sessions
               later (hold_m), or a WALK over that session's 1-minute bars: time_min (the first bar at / after entry + m),
               bracket (stop / target at a distance from the entry price; the stop wins when one bar touches both; a
               bar opening through the stop fills at its open; a target needs one tick through, sim.target_fill),
               signal (a 1-minute close on the wrong side of the session VWAP -> the next bar's open); every walk ends
               at 15:15 (intraday) or entry + 30 minutes (scalping) at that bar's open. Slippage on every fill.
  daily grid   decided in the evening of t, entered at the open of t+1; hold h = the close of t+h; bracket on the
               daily high / low (stop first; a gap through the stop fills at the open; target at its price, or the open
               when it opens beyond); signal = a close through its 10-session mean against the trade -> the next open;
               a hold passing the window end exits at the window's last close (datagen's rule).
  index        NIFTY official daily bars (2012 ->) / BANKNIFTY daily bars from its 1-minute bars (2016 ->) on the daily
               grid; the index minute rows on the minute grid; 1 lot (sim.LOT_SIZES), 1 bps.
  eligibility  datagen's: point-in-time universe member on the decision day, a listed stock future for futures /
               shorts, events_daily's corporate-action exclusions, the dirty-link rule and special sessions on the
               minute grid, <= 10 entries a day (strongest first), one open position per symbol.
MARKET TYPE    market_types(): per daily session from closes up to that session; a daily-grid decision on t uses t's,
               a minute-grid decision on t uses the previous daily session's.
UNIVERSE       (grammar bg2; a bg1 candidate uses its implicit one) the stock rows a candidate may trade, applied to
               its trigger cells and to the eligible cells of its controls (src.universe_lists(): "base" = the 97
               NIFTY-100 names, "extra" = the data layer's added NIFTY 200 names):
                 minute grid  NIFTY100 = the base rows; NIFTY200 = the base rows + every extra row with COMPLETE minute
                              data over the split (complete_rows: bars on >= 98 % of the minute sessions from the
                              calendar's first session to T, incl. one of the first 5 and one of the last 5)
                 daily grid   TOP500 = every point-in-time top-500 member (no filter); NIFTY100 / NIFTY200 = members
                              whose symbol is in the base / base + extra lists (+ the base's old symbols)
               a forward run uses the names frozen at the proof (Job.names), never a recomputed list.
SIZE           the rupee size of a stock trade comes from the candidate's size regime (grammar bg3: Candidate.size) and
               the cost kind the trade is booked at (cost_kind): SIZE_TABLES. "" (every bg1 / bg2 candidate, bg3's
               unchanged twins): Rs 1 lakh everywhere (builder/v1). "c10" (builder/v3, Rs 10 lakh capital per agent):
               Rs 5 lakh per MIS trade (<= 10 entries a day x Rs 5 L x 20 % intraday margin = Rs 10 L) and per stock
               trade at futures costs (about one real lot, ~20 % margin), Rs 1 lakh per CNC delivery long (delivery
               fees are proportional). Index futures stay 1 real lot. Option lots under "c10" follow the capital
               (builder_options.option_lots / OPTION_SIZES: Rs 1 lakh of premium or of maximum loss, Rs 5 lakh of
               naked margin, decided from the entry fills); a bg1 / bg2 option candidate trades 1 lot. A candidate
               carries its regime in its key and JSON, so a pending v1 / v2 batch and a promoted v1 / v2 strategy keep
               their size; apply_rule() checks the batch rule's costs.size_rs against the tables (a mismatch is
               refused).
"""
from __future__ import annotations

import dataclasses
import math
import os
import time

import numpy as np

from . import builder_grammar as G
from . import datagen_features as F
from . import datagen_run as R
from . import protocol as P
from . import sim
from . import trials as TR

MT_CODE = {m: i for i, m in enumerate(G.MARKET_TYPES)}
SMA_N, SLOPE_N, VIX_N, VIX_MIN, VIX_Q = 50, 10, 252, 200, 0.75
MAX_PER_DAY = R.MAX_PER_DAY
NOTIONAL = R.NOTIONAL
# Rs per stock trade by size regime (Candidate.size) and cost kind; the c10 row is builder_rule_v3.json costs.size_rs
SIZE_TABLES = {
    G.SIZE_LEGACY: {"equity_mis": NOTIONAL, "equity_cnc": NOTIONAL, "future": NOTIONAL},
    G.SIZE_C10: {"equity_mis": 5.0 * NOTIONAL, "equity_cnc": NOTIONAL, "future": 5.0 * NOTIONAL},
}


class SizeMismatch(RuntimeError):
    pass


def trade_notional(c: G.Candidate, kind: str, sizes: dict | None = None) -> float:
    """Rs of one stock trade of candidate c booked at cost kind `kind`."""
    return float((sizes or SIZE_TABLES)[c.size][kind])
ENTRY_MIN = G.ENTRY_MIN
SQUARE_MIN = 915
SCALP_MAX_MIN = 30


# ══════════════════════════════════════════════════════════════════════════
# market types (point-in-time)
# ══════════════════════════════════════════════════════════════════════════
def market_types(close: np.ndarray, vix: np.ndarray) -> np.ndarray:
    """int8 per session: 0 RISING, 1 FALLING, 2 SIDEWAYS, 3 HIGH_VOL, -1 unknown. Session t uses closes <= t only."""
    c = np.asarray(close, float)
    v = np.asarray(vix, float)
    T = len(c)
    out = np.full(T, -1, np.int8)
    sma = np.full(T, np.nan)
    for t in range(SMA_N - 1, T):
        w = c[t - SMA_N + 1:t + 1]
        if np.isfinite(w).all():
            sma[t] = w.mean()
    for t in range(T):
        if not np.isfinite(c[t]):
            continue
        hv = False
        if np.isfinite(v[t]):
            w = v[max(0, t - VIX_N):t]
            w = w[np.isfinite(w)]
            if len(w) >= VIX_MIN and float((w < v[t]).mean()) >= VIX_Q:
                hv = True
        if hv:
            out[t] = 3
            continue
        if t < SLOPE_N or not (np.isfinite(sma[t]) and np.isfinite(sma[t - SLOPE_N])):
            continue
        if c[t] > sma[t] and sma[t] > sma[t - SLOPE_N]:
            out[t] = 0
        elif c[t] < sma[t] and sma[t] < sma[t - SLOPE_N]:
            out[t] = 1
        else:
            out[t] = 2
    return out


# ══════════════════════════════════════════════════════════════════════════
# extra blocks (same conventions as datagen_features: value at t uses columns <= t)
# ══════════════════════════════════════════════════════════════════════════
def r_vol_shock(ctx, k: float, day: str):
    """Volume >= k x its 20-session mean on a >= 2 % up / down day: trade with the move (attention continuation)."""
    vr, r1 = ctx.d("vr"), ctx.d("ret1")
    if day == "up":
        return F._out((vr >= k) & (r1 >= 0.02), 1, vr)
    return F._out((vr >= k) & (r1 <= -0.02), -1, vr)


def r_vix_spike_index(ctx, x: float):
    vc = ctx.d("vix_chg")
    return F._out((vc >= x - 1e-12)[None, :], 1, 1.0)


def r_nifty_move(ctx, x: float, mode: str):
    """NIFTY 50 falls >= x close to close: long the index (rebound)."""
    r = ctx.d("nifty_ret1")
    return F._out((r <= -x + 1e-12)[None, :], 1, np.abs(r)[None, :])


def r_index_gap(ctx, g: float, mode: str):
    """Index rows of the minute grid: |opening gap| >= g (<= 5 %): fade (against) or go (with)."""
    M = ctx.minute
    gap = ctx.m("gap")
    m = np.zeros(M.shape, bool)
    for nm in F.INDEX_MIN:
        m[M.six[nm]] = True
    trig = m & (np.abs(gap) >= g - 1e-12) & (np.abs(gap) <= 0.05) & ctx.m("prev_ok")
    side = -F._sgn(gap) if mode == "fade" else F._sgn(gap)
    return F._out(trig, side, np.abs(gap))


# ── builder/v6 (bg6): price-history blocks on the daily grid (stocks) ──────────────────────────────────────────────
def _roll(x, n: int, how: str, min_count: int, include_current: bool = True) -> np.ndarray:
    """Rolling max / min / mean over the last n sessions per symbol (rows), NaN-aware; NaN with < min_count values."""
    import pandas as pd
    df = pd.DataFrame(np.asarray(x, float).T)
    if not include_current:
        df = df.shift(1)
    return getattr(df.rolling(n, min_periods=min_count), how)().to_numpy().T


def ca_clean(P, n: int) -> np.ndarray:
    """[symbols x sessions] True where no corporate-action exclusion day of the symbol falls in the last n sessions
    (today included): a price-history signal never compares across a split / bonus / large dividend."""
    T = len(P.cal)
    E = np.zeros(P.shape, float)
    for s, ds in (P.excl or {}).items():
        ds = np.asarray(ds, np.int64)
        if not len(ds):
            continue
        idx = np.searchsorted(P.cal, ds)
        ok = idx < T
        idx = idx[ok]
        E[int(s), idx] = 1.0          # an exclusion between sessions counts on the next session
    cs = np.cumsum(E, axis=1)
    w = cs.copy()
    w[:, n:] -= cs[:, :-n]
    return w == 0


def _fin(x) -> np.ndarray:
    return np.where(np.isfinite(x), x, np.nan)


def r_st_reversal(ctx, x: float, mode: str):
    """Short-term reversal: the 5-session return <= -x (losers: long) / >= x (winners: short)."""
    P = ctx.daily
    r5 = ctx.d("ret5")
    ok = ca_clean(P, 6)
    if mode == "losers":
        return F._out(ok & (r5 <= -x + 1e-12), 1, np.abs(r5))
    return F._out(ok & (r5 >= x - 1e-12), -1, np.abs(r5))


def r_high52(ctx, x: float):
    """52-week high: the close is within x of its highest close of the last 252 sessions (today included): long."""
    P = ctx.daily
    hi = _roll(P.c, 252, "max", 200)
    ok = ca_clean(P, 253)
    near = P.c / hi
    return F._out(ok & np.isfinite(near) & (near >= 1.0 - x - 1e-12), 1, _fin(ctx.d("ret20")))


def r_mom12_1(ctx, m: float):
    """12-1 momentum: the return from 252 to 21 sessions ago >= m: long."""
    P = ctx.daily
    mom = F.lag(P.c, 21) / F.lag(P.c, 252) - 1.0
    ok = ca_clean(P, 253)
    return F._out(ok & np.isfinite(mom) & (mom >= m - 1e-12), 1, _fin(mom))


def r_ma200(ctx, k: float, mode: str):
    """The close crosses its 200-session mean today (up: from below; down: from above) on volume >= k x its 20-session
    mean: with the cross."""
    P = ctx.daily
    sma = _roll(P.c, 200, "mean", 180)
    ok = ca_clean(P, 201)
    vr = ctx.d("vr")
    c1, s1 = F.lag(P.c, 1), F.lag(sma, 1)
    if mode == "up":
        return F._out(ok & (P.c > sma) & (c1 <= s1) & (vr >= k), 1, _fin(vr))
    return F._out(ok & (P.c < sma) & (c1 >= s1) & (vr >= k), -1, _fin(vr))


def r_breakout20(ctx, k: float, mode: str):
    """20-session breakout: the close above the highest high (up: long) / below the lowest low (down: short) of the
    previous 20 sessions on volume >= k x its 20-session mean."""
    P = ctx.daily
    ok = ca_clean(P, 21)
    vr = ctx.d("vr")
    if mode == "up":
        hi = _roll(P.h, 20, "max", 18, include_current=False)
        return F._out(ok & (P.c > hi) & (vr >= k), 1, _fin(vr))
    lo = _roll(P.l, 20, "min", 18, include_current=False)
    return F._out(ok & (P.c < lo) & (vr >= k), -1, _fin(vr))


# ── builder/v7 (bg7): long-only daily mean-reversion / continuation blocks (stocks) ──────────────────────────────
def _sma200(P):
    return _roll(P.c, 200, "mean", 180)


def rsi2(c) -> np.ndarray:
    """Wilder RSI(2) of each row's closes (NaN until defined)."""
    import pandas as pd
    df = pd.DataFrame(np.asarray(c, float).T)
    d = df.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 100.0 - 100.0 / (1.0 + up / dn)
    r = r.where(~((dn == 0) & (up > 0)), 100.0).where(~((dn == 0) & (up == 0)), 50.0)
    return r.to_numpy().T


def r_rsi2_dip(ctx, thr: float):
    """RSI(2) <= thr while the close is above its 200-session mean (a dip inside an uptrend): long."""
    P = ctx.daily
    r = rsi2(P.c)
    ok = ca_clean(P, 201)
    return F._out(ok & np.isfinite(r) & (r <= thr + 1e-12) & (P.c > _sma200(P)), 1, _fin(100.0 - r))


def r_ibs_low(ctx, x: float):
    """Internal bar strength (close - low) / (high - low) <= x above the 200-session mean: long."""
    P = ctx.daily
    rng = P.h - P.l
    with np.errstate(invalid="ignore", divide="ignore"):
        ibs = (P.c - P.l) / rng
    ok = ca_clean(P, 201) & (rng > 0)
    return F._out(ok & np.isfinite(ibs) & (ibs <= x + 1e-12) & (P.c > _sma200(P)), 1, _fin(1.0 - ibs))


def r_down_streak(ctx, n: int):
    """n consecutive lower closes above the 200-session mean: long."""
    P = ctx.daily
    dn = np.ones(P.shape, bool)
    for k in range(int(n)):
        cur = P.c if k == 0 else F.lag(P.c, k)
        dn &= cur < F.lag(P.c, k + 1)
    ok = ca_clean(P, 201)
    move = np.abs(P.c / F.lag(P.c, n) - 1.0)
    return F._out(ok & dn & (P.c > _sma200(P)), 1, _fin(move))


def r_nr7_up(ctx, p: float):
    """NR7: today's high-low range is the narrowest of the last 7 sessions and the close sits in the top (1 - p) of
    it: long (volatility contraction resolving up)."""
    P = ctx.daily
    rng = P.h - P.l
    mn = _roll(rng, 7, "min", 7)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (P.c - P.l) / rng
    ok = ca_clean(P, 8) & (rng > 0)
    return F._out(ok & np.isfinite(pos) & (rng <= mn + 1e-12) & (pos >= p - 1e-12), 1, _fin(P.c / rng))


def r_gap_up_hold(ctx, g: float):
    """The stock opens >= g above yesterday's close and closes at / above its open (the gap holds): long."""
    P = ctx.daily
    gap = P.o / F.lag(P.c, 1) - 1.0
    ok = ca_clean(P, 2)
    return F._out(ok & np.isfinite(gap) & (gap >= g - 1e-12) & (gap <= 0.20) & (P.c >= P.o), 1, _fin(gap))


# the SELL side of bg7: the mirror images (a spike inside a downtrend, a gap down that holds ...): short
def r_rsi2_spike(ctx, thr: float):
    """RSI(2) >= 100 - thr while the close is below its 200-session mean (a rally inside a downtrend): short."""
    P = ctx.daily
    r = rsi2(P.c)
    ok = ca_clean(P, 201)
    return F._out(ok & np.isfinite(r) & (r >= 100.0 - thr - 1e-12) & (P.c < _sma200(P)), -1, _fin(r))


def r_ibs_high(ctx, x: float):
    """Internal bar strength >= 1 - x below the 200-session mean: short."""
    P = ctx.daily
    rng = P.h - P.l
    with np.errstate(invalid="ignore", divide="ignore"):
        ibs = (P.c - P.l) / rng
    ok = ca_clean(P, 201) & (rng > 0)
    return F._out(ok & np.isfinite(ibs) & (ibs >= 1.0 - x - 1e-12) & (P.c < _sma200(P)), -1, _fin(ibs))


def r_up_streak(ctx, n: int):
    """n consecutive higher closes below the 200-session mean: short."""
    P = ctx.daily
    upm = np.ones(P.shape, bool)
    for k in range(int(n)):
        cur = P.c if k == 0 else F.lag(P.c, k)
        upm &= cur > F.lag(P.c, k + 1)
    ok = ca_clean(P, 201)
    move = np.abs(P.c / F.lag(P.c, n) - 1.0)
    return F._out(ok & upm & (P.c < _sma200(P)), -1, _fin(move))


def r_nr7_down(ctx, p: float):
    """NR7 with the close in the bottom (1 - p) of the range: short."""
    P = ctx.daily
    rng = P.h - P.l
    mn = _roll(rng, 7, "min", 7)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (P.c - P.l) / rng
    ok = ca_clean(P, 8) & (rng > 0)
    return F._out(ok & np.isfinite(pos) & (rng <= mn + 1e-12) & (pos <= 1.0 - p + 1e-12), -1, _fin(P.c / rng))


def r_gap_down_hold(ctx, g: float):
    """The stock opens >= g below yesterday's close and closes at / below its open (the gap holds): short."""
    P = ctx.daily
    gap = P.o / F.lag(P.c, 1) - 1.0
    ok = ca_clean(P, 2)
    return F._out(ok & np.isfinite(gap) & (gap <= -g + 1e-12) & (gap >= -0.20) & (P.c <= P.o), -1, _fin(-gap))


# ── bg7: index blocks (options / index futures) on NIFTY 50's daily open / close and India VIX ──────────────────────
def _nifty(ctx):
    P = ctx.daily
    return np.asarray(P.nifty["o"], float), np.asarray(P.nifty["c"], float)


def _row(x):
    return np.asarray(x, float)[None, :]


def r_idx_rsi2(ctx, thr: float, mode: str):
    """NIFTY 50 RSI(2) <= thr above its 200-session mean (dip: long) / >= 100 - thr below it (spike: short)."""
    _o, c = _nifty(ctx)
    r = rsi2(c[None, :])[0]
    sma = _roll(c[None, :], 200, "mean", 180)[0]
    if mode == "dip":
        return F._out(_row(np.isfinite(r) & (r <= thr + 1e-12) & (c > sma)) > 0, 1, _row(100.0 - r))
    return F._out(_row(np.isfinite(r) & (r >= 100.0 - thr - 1e-12) & (c < sma)) > 0, -1, _row(r))


def r_idx_streak(ctx, n: int, mode: str):
    """n lower closes in a row of NIFTY 50 above its 200-session mean (down: long) / n higher closes below it (up:
    short)."""
    _o, c = _nifty(ctx)
    c2 = c[None, :]
    sma = _roll(c2, 200, "mean", 180)[0]
    m = np.ones(len(c), bool)
    for k in range(int(n)):
        cur = c2 if k == 0 else F.lag(c2, k)
        m &= ((cur < F.lag(c2, k + 1)) if mode == "down" else (cur > F.lag(c2, k + 1)))[0]
    move = np.abs(c / F.lag(c2, int(n))[0] - 1.0)
    if mode == "down":
        return F._out(_row(m & (c > sma)) > 0, 1, _row(move))
    return F._out(_row(m & (c < sma)) > 0, -1, _row(move))


def r_idx_tom(ctx):
    """Turn of the month: the last session of a calendar month (entered at the next open = the first session of the
    new month): long."""
    cal = np.asarray(ctx.daily.cal, np.int64)
    last = np.zeros(len(cal), bool)
    last[:-1] = (cal[:-1] // 100) != (cal[1:] // 100)
    return F._out(_row(last) > 0, 1, _row(np.ones(len(cal))))


def r_idx_ma200(ctx, mode: str):
    """NIFTY 50 closes across its 200-session mean today (up: long / down: short)."""
    _o, c = _nifty(ctx)
    c2 = c[None, :]
    sma = _roll(c2, 200, "mean", 180)[0]
    c1, s1 = F.lag(c2, 1)[0], np.r_[np.nan, sma[:-1]]
    if mode == "up":
        return F._out(_row((c > sma) & (c1 <= s1)) > 0, 1, _row(np.abs(c / sma - 1.0)))
    return F._out(_row((c < sma) & (c1 >= s1)) > 0, -1, _row(np.abs(c / sma - 1.0)))


def r_idx_breakout(ctx, mode: str):
    """NIFTY 50 closes above the highest (up: long) / below the lowest (down: short) close of the previous 20
    sessions."""
    _o, c = _nifty(ctx)
    c2 = c[None, :]
    if mode == "up":
        hi = _roll(c2, 20, "max", 18, include_current=False)[0]
        return F._out(_row(c > hi) > 0, 1, _row(np.abs(c / hi - 1.0)))
    lo = _roll(c2, 20, "min", 18, include_current=False)[0]
    return F._out(_row(c < lo) > 0, -1, _row(np.abs(c / lo - 1.0)))


def r_idx_gap_hold(ctx, g: float, mode: str):
    """NIFTY 50 opens >= g above (up: long) / below (down: short) yesterday's close and closes beyond its open."""
    o, c = _nifty(ctx)
    gap = o / F.lag(c[None, :], 1)[0] - 1.0
    if mode == "up":
        return F._out(_row(np.isfinite(gap) & (gap >= g - 1e-12) & (c >= o)) > 0, 1, _row(np.abs(gap)))
    return F._out(_row(np.isfinite(gap) & (gap <= -g + 1e-12) & (c <= o)) > 0, -1, _row(np.abs(gap)))


def r_vix_drop(ctx, x: float):
    """India VIX falls >= x today (fear unwinding): long the index."""
    vc = ctx.d("vix_chg")
    return F._out((np.isfinite(vc) & (vc <= -x + 1e-12))[None, :], 1, np.abs(vc)[None, :])


# ── bg7: intraday momentum (minute grid, stocks): the day's move so far predicts the last hour ──────────────────
def r_halfhour_mom(ctx, x: float, src: str):
    """The stock's return from yesterday's close to 09:44 (open30) / to 14:29 (day) is >= x in size: trade WITH it
    from 14:30 (intraday momentum, Gao-Han-Li-Zhou 2018)."""
    M = ctx.minute
    Fm = M.f
    r = (Fm["c0944"] if src == "open30" else Fm["c1429"]) / Fm["pc"] - 1.0
    trig = F._stock_rows(M, np.isfinite(r) & (np.abs(r) >= x - 1e-12) & (np.abs(r) <= F.MAX_GAP) & ctx.m("prev_ok"))
    return F._out(trig, F._sgn(r), np.abs(r))


EXTRA = {"r_vol_shock": r_vol_shock, "r_vix_spike_index": r_vix_spike_index, "r_nifty_move": r_nifty_move,
         "r_index_gap": r_index_gap, "r_st_reversal": r_st_reversal, "r_high52": r_high52, "r_mom12_1": r_mom12_1,
         "r_ma200": r_ma200, "r_breakout20": r_breakout20, "r_rsi2_dip": r_rsi2_dip, "r_ibs_low": r_ibs_low,
         "r_down_streak": r_down_streak, "r_nr7_up": r_nr7_up, "r_gap_up_hold": r_gap_up_hold,
         "r_rsi2_spike": r_rsi2_spike, "r_ibs_high": r_ibs_high, "r_up_streak": r_up_streak, "r_nr7_down": r_nr7_down,
         "r_gap_down_hold": r_gap_down_hold, "r_idx_rsi2": r_idx_rsi2, "r_idx_streak": r_idx_streak,
         "r_idx_tom": r_idx_tom, "r_idx_ma200": r_idx_ma200, "r_idx_breakout": r_idx_breakout,
         "r_idx_gap_hold": r_idx_gap_hold, "r_vix_drop": r_vix_drop, "r_halfhour_mom": r_halfhour_mom}


def block_fn(name: str):
    return EXTRA[name[3:]] if name.startswith("be:") else getattr(F, name)


# ══════════════════════════════════════════════════════════════════════════
# the engine context
# ══════════════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self, src):
        self.src = src
        self.ctx = src.ctx()
        P_ = self.ctx.daily
        M = self.ctx.minute
        self.dcal = P_.cal
        self.mcal = M.cal
        self.regime_d = market_types(P_.nifty["c"], P_.vix["c"])
        self.prev_d = np.searchsorted(self.dcal, self.mcal, side="left") - 1          # last daily session < day
        self.regime_m = np.where(self.prev_d >= 0, self.regime_d[np.maximum(self.prev_d, 0)], -1).astype(np.int8)
        self.idx = self._index_daily()
        self._sig: dict = {}
        self._walk_cache: dict = {}
        self.stats = {"walk_requests": 0, "walk_symbols": 0}
        self.ulists = universe_lists_of(src, M)
        self._umask: dict = {}
        self.sizes = {k: dict(v) for k, v in SIZE_TABLES.items()}

    def apply_rule(self, rule: dict) -> None:
        """The batch rule's stock trade sizes. builder/v1 / v2 carry no costs.size_rs (Rs 1 lakh, the legacy row);
        builder/v3's must equal the engine's frozen table for its regime (SizeMismatch otherwise: an edited table or
        rule is refused, never silently applied)."""
        rs = ((rule or {}).get("costs") or {}).get("size_rs")
        if not rs:
            return
        from . import builder_options as BO
        regime = rs.get("regime")
        want = {k: float(rs[k]) for k in ("equity_mis", "equity_cnc", "future") if k in rs}
        if regime not in SIZE_TABLES or want != {k: float(v) for k, v in SIZE_TABLES[regime].items()}:
            raise SizeMismatch(f"rule {rule.get('version')} costs.size_rs {rs} != engine table "
                               f"{SIZE_TABLES.get(regime)}")
        if rs.get("options") != BO.OPTION_SIZES.get(regime):
            raise SizeMismatch(f"rule {rule.get('version')} costs.size_rs.options {rs.get('options')} != "
                               f"{BO.OPTION_SIZES.get(regime)}")
        self.sizes[regime] = want
        self.option_sizes = {**getattr(self, "option_sizes", {}), regime: dict(rs["options"])}

    # ── index daily bars aligned to the daily calendar ──
    def _index_daily(self) -> dict:
        T = len(self.dcal)
        out = {}
        nd = self.src.nifty_daily()
        a = {k: np.full(T, np.nan) for k in "ohlc"}
        if nd is not None:
            j = np.searchsorted(self.dcal, nd["date"].astype(np.int64))
            ok = (j < T) & (self.dcal[np.minimum(j, T - 1)] == nd["date"])
            for k in "ohlc":
                a[k][j[ok]] = nd[k][ok]
        else:
            a["o"], a["c"] = self.ctx.daily.nifty["o"].copy(), self.ctx.daily.nifty["c"].copy()
            a["h"] = np.fmax(a["o"], a["c"])
            a["l"] = np.fmin(a["o"], a["c"])
        out["NIFTY"] = a
        b = {k: np.full(T, np.nan) for k in "ohlc"}
        for sd in self.src.minute_parts("_NIFTYBANK_min"):
            dd = sd.daily()
            j = np.searchsorted(self.dcal, dd["date"].astype(np.int64))
            ok = (j < T) & (self.dcal[np.minimum(j, T - 1)] == dd["date"])
            for k in "ohlc":
                b[k][j[ok]] = dd[k][ok]
        out["BANKNIFTY"] = b
        return out

    def calendar(self, c: G.Candidate) -> np.ndarray:
        return self.mcal if c.grid in ("minute", "idx_minute", "prev_eod") else self.dcal

    def names(self, c: G.Candidate) -> list:
        if c.grid in ("minute", "idx_minute", "prev_eod"):
            return [{"_NIFTY50_min": "NIFTY", "_NIFTYBANK_min": "BANKNIFTY"}.get(x, x) for x in self.ctx.minute.syms]
        if c.grid == "idx_daily":
            return [c.instrument]
        return self.ctx.daily.syms

    # ── universes ──
    def minute_base_rows(self) -> list:
        M = self.ctx.minute
        base = set(self.ulists["base"])
        rows = [i for i in M.stocks if M.syms[i] in base]
        return rows if rows else list(M.stocks)

    def complete_rows(self, rows: list, lo: int | None = None, hi: int | None = None) -> tuple:
        """(complete, incomplete) minute rows over [lo, hi] (default: the calendar's first session .. T)."""
        M = self.ctx.minute
        lo = int(self.mcal[0]) if lo is None else int(lo)
        hi = int(self.src.end) if hi is None else int(hi)
        idx = np.nonzero((self.mcal >= lo) & (self.mcal <= hi))[0]
        ok, bad = [], []
        if not len(idx):
            return ok, list(rows)
        for r in rows:
            h = np.isfinite(M.f["nb"][r, idx])
            n = len(h)
            if h[:min(5, n)].any() and h[max(0, n - 5):].any() and float(h.mean()) >= COMPLETE_SHARE:
                ok.append(r)
            else:
                bad.append(r)
        return ok, bad

    def _extra_rows(self) -> list:
        M = self.ctx.minute
        base = set(self.minute_base_rows())
        extra = set(self.ulists["extra"]) - set(self.ulists["base"])
        return [i for i in M.stocks if M.syms[i] in extra and i not in base]

    def universe_rows(self, universe: str) -> list:
        """Minute rows of a stock universe (NIFTY100 / NIFTY200; anything else: the base rows)."""
        key = ("m", universe)
        if key not in self._umask:
            base = self.minute_base_rows()
            if universe == "NIFTY200":
                ok, _bad = self.complete_rows(self._extra_rows())
                rows = sorted(set(base) | set(ok))
            else:
                rows = base
            self._umask[key] = rows
        return self._umask[key]

    def daily_universe_names(self, universe: str) -> set:
        base = set(self.ulists["base"]) | set(self.ulists.get("aliases") or [])
        if universe == "NIFTY100":
            return base
        return base | set(self.ulists["extra"])

    def row_mask(self, c: G.Candidate, names=None):
        """Bool mask over the rows of the candidate's EXECUTION grid (None = no filter)."""
        g = c.grid
        if c.single or g in ("idx_minute", "idx_daily"):
            return None
        u = c.stock_universe
        if g in ("minute", "prev_eod"):
            M = self.ctx.minute
            key = ("mm", u) if names is None else ("mn", tuple(names))
            if key not in self._umask:
                m = np.zeros(len(M.syms), bool)
                if names is None:
                    m[self.universe_rows(u)] = True
                else:
                    want = set(names)
                    m[[i for i in M.stocks if M.syms[i] in want]] = True
                self._umask[key] = m
            return self._umask[key]
        if names is None and u == "TOP500":
            return None
        key = ("dm", u) if names is None else ("dn", tuple(names))
        if key not in self._umask:
            want = set(names) if names is not None else self.daily_universe_names(u)
            self._umask[key] = np.array([s in want for s in self.ctx.daily.syms], bool)
        return self._umask[key]

    def universe_report(self) -> dict:
        """What NIFTY100 / NIFTY200 / TOP500 resolve to on this data (recorded in the batch summary; the names are
        the list a promoted strategy is frozen with)."""
        M = self.ctx.minute
        base = self.minute_base_rows()
        extra = set(self.ulists["extra"]) - set(self.ulists["base"])
        ok, bad = self.complete_rows(self._extra_rows())
        stored = {M.syms[i] for i in M.stocks}
        both = sorted(set(base) | set(ok))
        return {"split": [int(self.mcal[0]), int(self.src.end)], "complete_share": COMPLETE_SHARE,
                "NIFTY100": {"names": sorted(M.syms[i] for i in base), "n": len(base)},
                "NIFTY200": {"names": sorted(M.syms[i] for i in both), "n": len(both),
                             "added_complete": sorted(M.syms[i] for i in ok),
                             "added_incomplete": sorted(M.syms[i] for i in bad),
                             "added_not_stored": sorted(extra - stored),
                             "daily_names": len(self.daily_universe_names("NIFTY200"))},
                "TOP500": {"daily_symbols": len(self.ctx.daily.syms)},
                "meta": self.ulists.get("meta", {})}

    def frozen_names(self, c: G.Candidate) -> list | None:
        """The name list a promoted candidate is frozen with (None: no stock universe or TOP500 = no filter)."""
        if c.single or c.grid in ("idx_minute", "idx_daily"):
            return None
        u = c.stock_universe
        if c.grid in ("minute", "prev_eod"):
            M = self.ctx.minute
            return sorted(M.syms[i] for i in self.universe_rows(u))
        if u == "TOP500":
            return None
        return sorted(self.daily_universe_names(u))

    # ── signals ──
    def raw_signal(self, c: G.Candidate):
        """Sparse trigger cells (S, T, side, rank, hold) of the candidate's block on its own grid, before the
        market-type filter (cached per block + parameters, sparse: the dense arrays are dropped at once)."""
        b = G.BLOCK[c.block]
        key = (b.fn, c.params)
        if key not in self._sig:
            kw = dict(c.params)
            if b.name == "expiry" and kw.get("side") == 0:
                kw["side"] = 1
            trig, side, rank, hold = block_fn(b.fn)(self.ctx, **kw)
            S, T_ = np.nonzero(trig)
            HD = np.asarray(hold)[S, T_].astype(np.int64) if hold is not None else np.zeros(len(S), np.int64)
            self._sig[key] = (S.astype(np.int64), T_.astype(np.int64), np.asarray(side)[S, T_].astype(np.int64),
                              np.asarray(rank, float)[S, T_], HD)
            del trig, side, rank, hold
        return self._sig[key]

    def signals(self, c: G.Candidate, mtype: str | None = "own", names=None) -> dict:
        """Sparse trigger cells {s, t, side, rank, hold} sorted by (t, -rank, symbol) on the candidate's EXECUTION
        grid (prev_eod -> minute grid), filtered to the candidate's market type (mtype=None: no filter) and to its
        universe (names: an explicit frozen name list instead)."""
        S0, T0, SD0, RK0, HD0 = self.raw_signal(c)
        code = MT_CODE[c.mtype] if mtype == "own" else (None if mtype is None else MT_CODE[mtype])
        g = c.grid
        if g == "prev_eod":
            s_d, t_d = S0, T0
            M = self.ctx.minute
            msix = M.six
            dnames = self.ctx.daily.syms
            tm = np.searchsorted(self.mcal, self.dcal[t_d], side="right")
            ok = tm < len(self.mcal)
            nxt = np.where(t_d + 1 < len(self.dcal), self.dcal[np.minimum(t_d + 1, len(self.dcal) - 1)], -1)
            ok &= self.mcal[np.minimum(tm, len(self.mcal) - 1)] == nxt
            sm = np.array([msix.get(dnames[i], -1) for i in s_d], np.int64) if len(s_d) else np.zeros(0, np.int64)
            ok &= sm >= 0
            S, T_, SD, RK = sm[ok], tm[ok], SD0[ok], RK0[ok]
            HD = np.zeros(len(S), np.int64)
            if code is not None:
                k = self.regime_m[T_] == code
                S, T_, SD, RK, HD = S[k], T_[k], SD[k], RK[k], HD[k]
        else:
            S, T_, SD, RK, HD = S0, T0, SD0, RK0, HD0
            if g == "idx_minute":
                M = self.ctx.minute
                row = M.six["_NIFTY50_min" if c.instrument == "NIFTY" else "_NIFTYBANK_min"]
                k = S == row
                S, T_, SD, RK, HD = S[k], T_[k], SD[k], RK[k], HD[k]
            if code is not None:
                reg = self.regime_m if g in ("minute", "idx_minute") else self.regime_d
                k = reg[T_] == code
                S, T_, SD, RK, HD = S[k], T_[k], SD[k], RK[k], HD[k]
        um = self.row_mask(c, names)
        if um is not None and len(S):
            k = um[S]
            S, T_, SD, RK, HD = S[k], T_[k], SD[k], RK[k], HD[k]
        return self.sort_sig(c, S, T_, SD, RK, HD)

    def sort_sig(self, c, S, T_, SD, RK, HD) -> dict:
        S, T_ = np.asarray(S, np.int64), np.asarray(T_, np.int64)
        RK = np.asarray(RK, float)
        names = self.names(c)
        nr = np.empty(len(names), np.int64)
        nr[np.argsort(np.array(names, dtype=object), kind="stable")] = np.arange(len(names))
        order = np.lexsort((nr[S] if len(S) else S, -RK, T_)) if len(S) else np.arange(0)
        return {"s": S[order], "t": T_[order], "side": np.asarray(SD, np.int64)[order], "rank": RK[order],
                "hold": np.asarray(HD, np.int64)[order]}

    # ── eligible cells (the pool of the shuffle / placebo / planted controls) ──
    def eligible(self, c: G.Candidate, lo: int, hi: int, mtype: str | None = "own") -> tuple:
        code = MT_CODE[c.mtype] if mtype == "own" else (None if mtype is None else MT_CODE[mtype])
        g = c.grid
        if g == "daily":
            Pn = self.ctx.daily
            cal = self.dcal
            e_ok = np.zeros(len(cal), bool)
            e_ok[:-1] = (cal[1:] >= lo) & (cal[1:] <= hi)
            if code is not None:
                e_ok &= self.regime_d == code
            o1 = np.full(Pn.shape, np.nan)
            o1[:, :-1] = Pn.o[:, 1:]
            m = Pn.member & np.isfinite(o1) & e_ok[None, :]
            if c.product == "fut" or c.agent == "futures":
                m &= Pn.fo
            um = self.row_mask(c)
            if um is not None:
                m &= um[:, None]
            return np.nonzero(m)
        if g == "idx_daily":
            cal = self.dcal
            nxt = np.r_[cal[1:], 0]
            ok = (nxt >= lo) & (nxt <= hi) & np.isfinite(np.r_[self.idx[c.instrument]["o"][1:], np.nan])
            if code is not None:
                ok &= self.regime_d == code
            t = np.nonzero(ok)[0]
            return np.zeros(len(t), np.int64), t
        M = self.ctx.minute
        if g == "idx_minute":
            rows = [M.six["_NIFTY50_min" if c.instrument == "NIFTY" else "_NIFTYBANK_min"]]
        else:
            rows = self.universe_rows(c.stock_universe)
        inwin = (self.mcal >= lo) & (self.mcal <= hi)
        if code is not None:
            inwin &= self.regime_m == code
        m = np.zeros(M.shape, bool)
        m[rows] = np.isfinite(M.f["nb"][rows]) & inwin[None, :]
        return np.nonzero(m)

    # ══════════════════════════════════════════════════════════════════════
    # planning (entries + exits known without a walk)
    # ══════════════════════════════════════════════════════════════════════
    def plan(self, c: G.Candidate, sig: dict, lo: int, hi: int) -> list:
        """[(s, t_dec, side, t_entry, t_exit, px_in, px_out | None, entry_label, exit_label, walk | None)]."""
        if c.grid in ("daily", "idx_daily"):
            return self._plan_daily(c, sig, lo, hi)
        return self._plan_minute(c, sig, lo, hi)

    def _plan_daily(self, c, sig, lo, hi) -> list:
        Pn = self.ctx.daily
        cal = self.dcal
        T = len(cal)
        hi_i = int(np.searchsorted(cal, hi, side="right")) - 1
        out, busy, per_day = [], {}, {}
        ex = c.exit
        stock = c.grid == "daily"
        for s, t, side, h in zip(sig["s"].tolist(), sig["t"].tolist(), sig["side"].tolist(), sig["hold"].tolist()):
            e = t + 1
            if e >= T or e > hi_i or cal[e] < lo:
                continue
            if per_day.get(e, 0) >= (1 if not stock else MAX_PER_DAY):
                continue
            if busy.get(s, -1) >= e:
                continue
            if stock:
                if not Pn.member[s, t]:
                    continue
                needs_fo = c.agent == "futures" or (c.agent == "swing" and side < 0)
                if needs_fo and not Pn.fo[s, t]:
                    continue
                O, H, L, C = Pn.o[s], Pn.h[s], Pn.l[s], Pn.c[s]
            else:
                a = self.idx[c.instrument]
                O, H, L, C = a["o"], a["h"], a["l"], a["c"]
            pin = O[e]
            if not (np.isfinite(pin) and pin > 0):
                continue
            kind = ex[0]
            if kind == "hold":
                mh = int(ex[1])
            elif kind == "event":
                mh = int(h)
                if mh < 1:
                    continue
            elif kind in ("bracket_d", "signal_d"):
                mh = int(ex[-1])
            else:
                raise ValueError(f"daily exit {ex}")
            xmax = min(t + mh, hi_i)
            xs, pout, xlab = None, None, "c"
            if kind in ("hold", "event"):
                x = xmax
                while x >= e and not np.isfinite(C[x]):
                    x -= 1
                if x >= e:
                    xs, pout = x, float(C[x])
            elif kind == "bracket_d":
                sp, rr = float(ex[1]), float(ex[2])
                stop = pin * (1 - side * sp)
                tgt = pin * (1 + side * sp * rr)
                for k in range(e, xmax + 1):
                    o_, h_, l_, c_ = O[k], H[k], L[k], C[k]
                    if not np.isfinite(c_):
                        continue
                    if k > e and np.isfinite(o_) and side * (o_ - stop) <= 0:
                        xs, pout, xlab = k, float(o_), "o"
                        break
                    if (side > 0 and l_ <= stop) or (side < 0 and h_ >= stop):
                        xs, pout, xlab = k, float(stop), "stop"
                        break
                    if (side > 0 and h_ >= tgt) or (side < 0 and l_ <= tgt):
                        px = float(o_) if (k > e and np.isfinite(o_) and side * (o_ - tgt) >= 0) else float(tgt)
                        xs, pout, xlab = k, px, "target"
                        break
                    xs, pout, xlab = k, float(c_), "c"
            elif kind == "signal_d":
                cc = C
                for k in range(e, xmax + 1):
                    if not np.isfinite(cc[k]):
                        continue
                    xs, pout, xlab = k, float(cc[k]), "c"
                    w = cc[max(0, k - 9):k + 1]
                    if k < xmax and len(w) == 10 and np.isfinite(w).all() and side * (cc[k] - w.mean()) < 0:
                        k2 = k + 1
                        if np.isfinite(O[k2]):
                            xs, pout, xlab = k2, float(O[k2]), "o"
                        break
            if xs is None or pout is None or not (pout > 0):
                continue
            if xlab == "c" and xs == xmax and t + mh > hi_i:
                xlab = "end"                    # cut at the window end (still open in a forward run)
            if stock and Pn.excluded(s, int(cal[t]), int(cal[xs])):
                continue
            out.append((s, t, side, e, xs, float(pin), pout, "o", xlab, None))
            busy[s] = xs
            per_day[e] = per_day.get(e, 0) + 1
        return out

    def pending_daily(self, c, sig, busy_syms=()) -> list:
        """Live support (daily stock grid): the entries _plan_daily takes at the NEXT session's open for the signals
        of the LAST session of the data - which it cannot plan itself (e = T). Same filters in the same order: the
        signal order (t, -rank, name), point-in-time membership, F&O for futures / swing shorts, a hold >= 1 for
        event exits, one position per symbol (busy_syms: symbols with an open trade) and MAX_PER_DAY.
        [(symbol, side, rank)]. Unknown here (decided at the open): the open price itself and a corporate action
        inside the holding period (the live agent's own guards)."""
        if c.grid != "daily":
            raise ValueError(f"pending_daily: {c.grid}")
        Pn = self.ctx.daily
        t_last = len(self.dcal) - 1
        names = self.names(c)
        busy = set(busy_syms)
        out = []
        for s, t, side, h, rk in zip(sig["s"].tolist(), sig["t"].tolist(), sig["side"].tolist(), sig["hold"].tolist(),
                                     sig["rank"].tolist()):
            if t != t_last or len(out) >= MAX_PER_DAY:
                continue
            sym = names[s]
            if sym in busy or not Pn.member[s, t]:
                continue
            needs_fo = c.agent == "futures" or (c.agent == "swing" and side < 0)
            if needs_fo and not Pn.fo[s, t]:
                continue
            if c.exit[0] == "event" and int(h) < 1:
                continue
            out.append((sym, int(side), float(rk)))
            busy.add(sym)
        return out

    def _plan_minute(self, c, sig, lo, hi) -> list:
        M = self.ctx.minute
        cal = self.mcal
        T = len(cal)
        has = np.isfinite(M.f["nb"])
        out, busy, per_day = [], {}, {}
        ex = c.exit
        single = c.single
        fi = c.entry
        for s, t, side, _h in zip(sig["s"].tolist(), sig["t"].tolist(), sig["side"].tolist(), sig["hold"].tolist()):
            d = int(cal[t])
            if d < lo or d > hi:
                continue
            if per_day.get(t, 0) >= (1 if single else MAX_PER_DAY):
                continue
            if busy.get(s, -1) >= t:
                continue
            if d in M.special:
                continue
            pin = M.f[fi][s, t]
            if not (np.isfinite(pin) and pin > 0):
                continue
            kind = ex[0]
            walk, pout, x, xl = None, None, t, None
            if kind == "time":
                if ENTRY_MIN[ex[1]] <= ENTRY_MIN[fi]:
                    continue
                pout, xl = M.f[ex[1]][s, t], ex[1]
            elif kind == "overnight":
                x = t + 1
                if x >= T or not has[s, x] or int(cal[x]) > hi or int(cal[x]) in M.special:
                    continue
                pout, xl = M.f["o0915"][s, x], "o0915"
            elif kind == "hold_m":
                x = t + int(ex[1])
                if x >= T or not has[s, x] or int(cal[x]) > hi or int(cal[x]) in M.special:
                    continue
                pout, xl = M.f["o1521"][s, x], "o1521"
            elif kind in ("time_min", "bracket", "bracket_min", "signal", "signal_min") + G.TRAIL_KINDS + G.ATR_KINDS:
                walk = (ENTRY_MIN[fi], kind) + tuple(ex[1:])
            else:
                raise ValueError(f"minute exit {ex}")
            if walk is None and not (np.isfinite(pout) and pout > 0):
                continue
            if R._dirty(self.ctx, s, t, x):
                continue
            out.append((s, t, side, t, x, float(pin), None if walk else float(pout), fi, xl, walk))
            busy[s] = x
            per_day[t] = per_day.get(t, 0) + 1
        return out

    # ══════════════════════════════════════════════════════════════════════
    # minute walks (grouped by symbol: each symbol's bars are loaded once per resolve)
    # ══════════════════════════════════════════════════════════════════════
    def resolve_walks(self, requests: list) -> dict:
        """requests: [(key, sym_row, t, side, walk)] -> {key: (px_in, px_out, exit_hhmm, reason) | None}."""
        by: dict = {}
        for r in requests:
            by.setdefault(r[1], []).append(r)
        res = {}
        M = self.ctx.minute
        self.stats["walk_requests"] += len(requests)
        for srow, reqs in by.items():
            sym = M.syms[srow]
            parts = self.src.minute_parts(sym)
            self.stats["walk_symbols"] += 1
            index = {}
            for pi, sd in enumerate(parts):
                for k, d in enumerate(sd.days.tolist()):
                    index[int(d)] = (pi, k)
            days = sorted(index)
            ohlc: dict = {}
            for key, _s, t, side, walk in reqs:
                loc = index.get(int(self.mcal[t]))
                if loc is None:
                    res[key] = None
                    continue
                if walk[1] in G.ATR_KINDS:
                    d0 = int(self.mcal[t])
                    prev = [d for d in days[:int(np.searchsorted(days, d0))] if d not in M.special][-(ATR_N + 1):]
                    for d in prev:
                        if d not in ohlc:
                            pp = parts[index[d][0]]
                            a_, b_ = int(pp.starts[index[d][1]]), int(pp.ends[index[d][1]])
                            ohlc[d] = session_hlc(pp.h[a_:b_], pp.l[a_:b_], pp.c[a_:b_])
                    walk = atr_walk(walk, [ohlc[d] for d in prev])
                    if walk is None:
                        res[key] = None
                        continue
                sd = parts[loc[0]]
                a, b = int(sd.starts[loc[1]]), int(sd.ends[loc[1]])
                res[key] = walk_session(sd.t[a:b], sd.o[a:b], sd.h[a:b], sd.l[a:b], sd.c[a:b], sd.v[a:b], side, walk,
                                        int(self.mcal[t]), index_tick=sym.startswith("_"))
            del parts
        return res


COMPLETE_SHARE = 0.98


def universe_lists_of(src, M) -> dict:
    """The source's universe lists ({"base", "extra", "aliases", "meta"}); a source without them (the lab's): every
    minute stock is base, nothing extra."""
    fn = getattr(src, "universe_lists", None)
    if fn is not None:
        d = fn()
        if d and d.get("base"):
            return {"base": list(d["base"]), "extra": list(d.get("extra") or []),
                    "aliases": list(d.get("aliases") or []), "meta": dict(d.get("meta") or {})}
    return {"base": [M.syms[i] for i in M.stocks], "extra": [], "aliases": [], "meta": {"source": "all minute stocks"}}


def _hm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


ATR_N = 14                                   # builder/v5: ATR(14) from the 14 sessions before the entry day


def session_hlc(h, l, c) -> tuple:
    """One session's (high, low, close) from its 1-minute bars."""
    return float(np.max(h)), float(np.min(l)), float(c[-1])


def atr_fraction(prev: list, k: float):
    """k x ATR(14) / previous close from the sessions before the entry day [(high, low, close)], oldest first (the
    last 15 are used: 14 true ranges); None without 15 sessions or a non-positive ATR / close."""
    if len(prev) < ATR_N + 1:
        return None
    s = prev[-(ATR_N + 1):]
    tr = [max(hh - ll, abs(hh - pc), abs(ll - pc)) for (hh, ll, _c), (_h, _l, pc) in zip(s[1:], s[:-1])]
    atr, pc = sum(tr) / ATR_N, s[-1][2]
    if not (atr > 0 and pc > 0):
        return None
    return float(k) * atr / pc


def atr_walk(walk: tuple, prev: list):
    """A bg5 ATR walk (entry_min, kind, k[, minutes]) as the bg4 walk it fills like, at x = atr_fraction(prev, k);
    None when the ATR is unknown (the trade is not taken)."""
    x = atr_fraction(prev, walk[2])
    if x is None:
        return None
    return (walk[0], G.ATR_BASE[walk[1]], x) + tuple(walk[3:])


def walk_session(tt, o, h, l, c, v, side: int, walk: tuple, day: int, index_tick: bool = False):
    """One trade over one session's 1-minute bars. walk = (entry_min, kind, params...). Returns (px_in, px_out,
    exit_hhmm, reason) at raw prices (slippage is applied by the caller on both fills) or None."""
    tt = np.asarray(tt, np.int64)
    n = len(tt)
    m0, kind = int(walk[0]), walk[1]
    j = int(np.searchsorted(tt, m0))
    if j >= n or tt[j] >= m0 + 10:
        return None
    raw = float(o[j])
    if not raw > 0:
        return None
    if kind in ("time_min",):
        deadline = m0 + int(walk[2])
    elif kind in ("bracket_min",):
        deadline = m0 + int(walk[4])
    elif kind in ("signal_min",):
        deadline = m0 + int(walk[3])
    elif kind in ("trail_min", "trail_be_min"):
        deadline = m0 + int(walk[3])
    else:
        deadline = SQUARE_MIN
    deadline = min(deadline, SQUARE_MIN)
    lim = int(np.searchsorted(tt, deadline))
    if lim <= j:
        return None

    def at_lim(reason):
        if lim < n:
            return raw, float(o[lim]), _hm(int(tt[lim])), reason
        return raw, float(c[n - 1]), _hm(int(tt[n - 1])), reason

    if kind == "time_min":
        return at_lim("time")
    if kind in ("bracket", "bracket_min"):
        sp, rr = float(walk[2]), float(walk[3])
        stop = raw * (1 - side * sp)
        tgt = raw * (1 + side * sp * rr)
        tick = sim.INDEX_TICK if index_tick else sim.equity_tick(tgt, day)
        for k in range(j, lim):
            if k > j and side * (float(o[k]) - stop) <= 0:
                return raw, float(o[k]), _hm(int(tt[k])), "stop"
            if (side > 0 and l[k] <= stop) or (side < 0 and h[k] >= stop):
                return raw, float(stop), _hm(int(tt[k])), "stop"
            tp = sim.target_fill(side, tgt, float(o[k]), float(h[k] if side > 0 else l[k]), tick)
            if tp is not None:
                return raw, float(tp), _hm(int(tt[k])), "target"
        return at_lim("time")
    if kind in G.TRAIL_KINDS:
        # builder/v4: the stop is checked against each bar (gap through it at the open: filled at the open; touched
        # inside the bar: filled at the stop), and only then moved with that bar's best price - never inside a bar
        x = float(walk[2])
        be = kind in ("trail_be", "trail_be_min")
        best = raw
        stop = raw * (1 - side * x)
        armed = not be                               # trail: trails from the first bar; trail_be: after +x
        for k in range(j, lim):
            if k > j and side * (float(o[k]) - stop) <= 0:
                return raw, float(o[k]), _hm(int(tt[k])), "stop"
            if (side > 0 and l[k] <= stop) or (side < 0 and h[k] >= stop):
                return raw, float(stop), _hm(int(tt[k])), "stop"
            fav = float(h[k]) if side > 0 else float(l[k])
            if side * (fav - best) > 0:
                best = fav
            if be and not armed and side * (best / raw - 1.0) >= x:
                armed = True
            if armed:
                trail = best * (1 - side * x)
                if be:
                    trail = max(trail, raw) if side > 0 else min(trail, raw)
                if side * (trail - stop) > 0:
                    stop = trail
        return at_lim("time")
    if kind in ("signal", "signal_min"):
        a0 = 0
        typ = (np.asarray(h, float) + np.asarray(l, float) + np.asarray(c, float)) / 3.0
        vv = np.asarray(v, float)
        cv = np.cumsum(typ[a0:] * vv[a0:])
        cn = np.cumsum(vv[a0:])
        with np.errstate(invalid="ignore", divide="ignore"):
            vwap = np.where(cn > 0, cv / cn, np.nan)
        for k in range(j, lim - 1):
            if np.isfinite(vwap[k]) and side * (float(c[k]) - vwap[k]) < 0:
                return raw, float(o[k + 1]), _hm(int(tt[k + 1])), "signal"
        return at_lim("time")
    raise ValueError(kind)


# ══════════════════════════════════════════════════════════════════════════
# trades
# ══════════════════════════════════════════════════════════════════════════
def cost_kind(c: G.Candidate, side: int) -> str:
    if c.agent in ("intraday", "scalping"):
        return "equity_mis"
    if c.agent == "swing":
        return "equity_cnc" if side > 0 else "future"
    return "future"


def slip_bps(c: G.Candidate, stress: bool = False) -> float:
    if c.single:
        return R.IDX_SENS_BPS if stress else R.IDX_SLIP_BPS
    return R.SENS_BPS if stress else R.SLIP_BPS


def build_trades(E: Engine, cid: str, c: G.Candidate, rows: list, walks: dict, stress: bool = False,
                 side_override: dict | None = None) -> list:
    """Trade tuples (sim.TRADE_FIELDS) from planned rows (+ resolved walks). side_override {row index: side}."""
    names = E.names(c)
    cal = E.calendar(c)
    slip = slip_bps(c, stress) / 1e4
    out = []
    for k, (s, t, side, e, x, pin, pout, fi, fo_, walk) in enumerate(rows):
        if side_override is not None:
            side = side_override.get(k, side)
            if side == 0:
                continue
        et = G.HHMM.get(fi, "09:15") if fi != "o" else "09:15"
        if walk is not None:
            w = walks.get((cid, k))
            if w is None:
                continue
            pin, pout, xt, _reason = w
        elif fo_ in ("c", "stop", "target", "end"):
            xt = "15:29"
        elif fo_ == "o":
            xt = "09:15"
        else:
            xt = G.HHMM.get(fo_, "15:15")
        sym = names[s]
        kind = cost_kind(c, side)
        qty = sim.LOT_SIZES[sym] if c.single else max(1, int(trade_notional(c, kind, E.sizes) // pin))
        tr = R._trade(cid, sym, cal[e], side, et, pin, cal[x], xt, pout, qty, kind, slip)
        reason = w[3] if walk is not None else {"end": "window_end", "stop": "stop", "target": "target"}.get(fo_,
                                                                                                         "time")
        out.append(tr[:13] + (reason,))
    out.sort(key=lambda t: (t[2], t[1]))
    return out


# ══════════════════════════════════════════════════════════════════════════
# running many jobs (one walk pass)
# ══════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class Job:
    key: str
    cand: G.Candidate
    sig: dict | None           # None = the candidate's own signals (options: entry-day set)
    lo: int
    hi: int
    stress: bool = True        # also compute the 5 bps (2.5 bps) version
    planted_p: float | None = None    # planted control: side = sign of the realised move with probability p
    seed: int = 0
    names: tuple | None = None        # a frozen universe name list (promoted strategies in forward runs)


def run_jobs(E: Engine, jobs: list) -> dict:
    """{job key: {"trades": [...], "trades5": [...]}}; minute walks of every job resolved in one pass."""
    from . import builder_options as BO
    plans, reqs, out = {}, [], {}
    for j in jobs:
        if j.cand.agent == "options":
            out[j.key] = BO.run_option_job(E, j)
            continue
        sig = j.sig if j.sig is not None else E.signals(j.cand, names=j.names)
        rows = E.plan(j.cand, sig, j.lo, j.hi)
        plans[j.key] = rows
        for k, r in enumerate(rows):
            if r[9] is not None:
                reqs.append(((j.key, k), r[0], r[1], r[2] if j.planted_p is None else 1, r[9]))
    walks = E.resolve_walks(reqs) if reqs else {}
    for j in jobs:
        if j.key not in plans:
            continue
        rows = plans[j.key]
        ov = None
        if j.planted_p is not None:
            rng = np.random.default_rng(j.seed)
            ov = {}
            for k, r in enumerate(rows):
                pin, pout = r[5], r[6]
                if r[9] is not None:
                    w = walks.get((j.key, k))
                    if w is None:
                        continue
                    pin, pout = w[0], w[1]
                mv = 1 if pout > pin else -1
                sd = mv if rng.random() < j.planted_p else -mv
                if j.cand.agent == "swing" and sd < 0 and not E.ctx.daily.fo[r[0], r[1]]:
                    sd = 0
                ov[k] = sd
        tr = build_trades(E, j.key, j.cand, rows, walks, False, ov)
        tr5 = build_trades(E, j.key, j.cand, rows, walks, True, ov) if j.stress else []
        out[j.key] = {"trades": tr, "trades5": tr5}
    return out


# ══════════════════════════════════════════════════════════════════════════
# windows, gates
# ══════════════════════════════════════════════════════════════════════════
def months_back(day: int, m: int) -> int:
    y, mo, d = day // 10000, day // 100 % 100, day % 100
    k = y * 12 + (mo - 1) - m
    y2, mo2 = k // 12, k % 12 + 1
    import calendar
    d2 = min(d, calendar.monthrange(y2, mo2)[1])
    return y2 * 10000 + mo2 * 100 + d2


def windows(T: int, rule: dict) -> dict:
    """The rule's rolling splits: DISCOVERY = data start .. T - 24 months, VALIDATION = (T-24m, T-12m], FINAL =
    (T-12m, T]."""
    sp = rule["splits"]
    v0 = months_back(T, int(sp["validation_months_before_end"]))
    f0 = months_back(T, int(sp["final_months_before_end"]))
    return {"discovery": (0, v0), "validation": (v0 + 1, f0), "final": (f0 + 1, T), "T": T}


def halves_of(cal, lo: int, hi: int):
    days = [int(d) for d in cal if lo <= int(d) <= hi]
    if len(days) < 2:
        return (lo, hi), (hi, hi)
    n = len(days) // 2
    return (days[0], days[n - 1]), (days[n], days[-1])


def judge_discovery(c: G.Candidate, tr2: list, tr5: list, h1, h2, rule: dict) -> dict:
    """datagen's DISCOVERY gates (rule["discovery_gates"])."""
    g = rule["discovery_gates"]
    b1 = [t for t in tr2 if h1[0] <= t[2] <= h1[1]]
    b2 = [t for t in tr2 if h2[0] <= t[2] <= h2[1]]
    s1, s2, sa = R.block_stats(b1), R.block_stats(b2), R.block_stats(b1 + b2)
    f1 = sum(t[12] for t in tr5 if h1[0] <= t[2] <= h1[1])
    f2 = sum(t[12] for t in tr5 if h2[0] <= t[2] <= h2[1])
    by: dict = {}
    for t in b1 + b2:
        by[t[1]] = by.get(t[1], 0.0) + t[12]
    top, top_net = max(by.items(), key=lambda kv: kv[1]) if by else ("", 0.0)
    share = top_net / sa["net"] if sa["net"] > 0 else float("nan")
    need = int(g["min_trades_per_half"][c.horizon])
    why = []
    if s1["net"] <= 0:
        why.append("H1 net <= 0")
    if s2["net"] <= 0:
        why.append("H2 net <= 0")
    if sa["pf"] < float(g["min_pf"]):
        why.append(f"PF {sa['pf']:.3f} < {g['min_pf']}")
    if s1["trades"] < need:
        why.append(f"H1 {s1['trades']} trades < {need}")
    if s2["trades"] < need:
        why.append(f"H2 {s2['trades']} trades < {need}")
    if sa["net"] > 0 and share > float(g["max_symbol_share"]) and not c.single:
        why.append(f"{top} is {share:.0%} of net")
    if f1 <= 0 or f2 <= 0:
        why.append(f"stress: H1 {f1:,.0f} / H2 {f2:,.0f}")
    return {"passed": not why, "why": why, "h1": s1, "h2": s2, "disc": sa, "top_symbol": top,
            "top_share": None if share != share else round(share, 3), "stress_h1": round(f1, 2),
            "stress_h2": round(f2, 2)}


def screen_window(c: G.Candidate, trades: list, cal, lo: int, hi: int) -> dict:
    """protocol.screen on a window (its own halves by sessions)."""
    h1, h2 = halves_of(cal, lo, hi)
    n = sum(1 for d in cal if lo <= int(d) <= hi)
    split = P.Split(h1, h2, (hi, hi), n, n, 0)
    r = P.screen(trades, split, c.horizon, single_underlying=c.single)
    return {"passed": r["passed"], "reasons": r["reasons"], "h1": r["h1"], "h2": r["h2"], "disc": r["disc"],
            "top_symbol": r["top_symbol"],
            "top_share": None if r["top_share"] != r["top_share"] else round(float(r["top_share"]), 3)}


def series(cal, trades: list, lo: int, hi: int) -> dict:
    st = TR.series_stats(R._tdicts(trades), [int(x) for x in cal], lo, hi)
    return {k: st[k] for k in ("trades", "net", "pf", "mean_net", "std_net", "sr_trade", "mean_gross", "mean_cost",
                               "n_days", "sr_daily", "skew_daily", "kurt_daily")}


def dsr_of(st: dict, n: int, v: float) -> dict:
    return R.dsr_of(st, n, v)


def robust_var(values) -> float:
    return R.robust_var(values)


def clean(o):
    """JSON-safe (NaN / inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


__all__ = ["Engine", "Job", "run_jobs", "market_types", "windows", "judge_discovery", "screen_window", "series",
           "walk_session", "months_back", "halves_of", "os"]
