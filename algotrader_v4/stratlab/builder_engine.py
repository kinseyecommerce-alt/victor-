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


EXTRA = {"r_vol_shock": r_vol_shock, "r_vix_spike_index": r_vix_spike_index, "r_nifty_move": r_nifty_move,
         "r_index_gap": r_index_gap}


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

    # ── signals ──
    def raw_signal(self, c: G.Candidate):
        """(trig, side, rank, hold) on the candidate's grid, before the market-type filter (cached per block)."""
        b = G.BLOCK[c.block]
        key = (b.fn, c.params)
        if key not in self._sig:
            kw = dict(c.params)
            if b.name == "expiry" and kw.get("side") == 0:
                kw["side"] = 1
            self._sig[key] = block_fn(b.fn)(self.ctx, **kw)
        return self._sig[key]

    def signals(self, c: G.Candidate, mtype: str | None = "own") -> dict:
        """Sparse trigger cells {s, t, side, rank, hold} sorted by (t, -rank, symbol) on the candidate's EXECUTION
        grid (prev_eod -> minute grid), filtered to the candidate's market type (mtype=None: no filter)."""
        trig, side, rank, hold = self.raw_signal(c)
        code = MT_CODE[c.mtype] if mtype == "own" else (None if mtype is None else MT_CODE[mtype])
        g = c.grid
        if g == "prev_eod":
            s_d, t_d = np.nonzero(trig)
            M = self.ctx.minute
            msix = M.six
            dnames = self.ctx.daily.syms
            tm = np.searchsorted(self.mcal, self.dcal[t_d], side="right")
            ok = tm < len(self.mcal)
            nxt = np.where(t_d + 1 < len(self.dcal), self.dcal[np.minimum(t_d + 1, len(self.dcal) - 1)], -1)
            ok &= self.mcal[np.minimum(tm, len(self.mcal) - 1)] == nxt
            sm = np.array([msix.get(dnames[i], -1) for i in s_d], np.int64) if len(s_d) else np.zeros(0, np.int64)
            ok &= sm >= 0
            S, T_, SD, RK = sm[ok], tm[ok], side[s_d, t_d][ok], rank[s_d, t_d][ok]
            HD = np.zeros(len(S), np.int64)
            if code is not None:
                k = self.regime_m[T_] == code
                S, T_, SD, RK, HD = S[k], T_[k], SD[k], RK[k], HD[k]
        else:
            if g == "idx_minute":
                M = self.ctx.minute
                row = M.six["_NIFTY50_min" if c.instrument == "NIFTY" else "_NIFTYBANK_min"]
                keep = np.zeros(M.shape, bool)
                keep[row] = True
                trig = np.asarray(trig, bool) & keep
            S, T_ = np.nonzero(trig)
            SD, RK = side[S, T_], rank[S, T_]
            if hold is not None:
                HD = np.asarray(hold)[S, T_].astype(np.int64)
            else:
                HD = np.zeros(len(S), np.int64)
            if code is not None:
                reg = self.regime_m if g in ("minute", "idx_minute") else self.regime_d
                k = reg[T_] == code
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
            rows = M.stocks
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
            elif kind in ("time_min", "bracket", "bracket_min", "signal", "signal_min"):
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
            for key, _s, t, side, walk in reqs:
                loc = index.get(int(self.mcal[t]))
                if loc is None:
                    res[key] = None
                    continue
                sd = parts[loc[0]]
                a, b = int(sd.starts[loc[1]]), int(sd.ends[loc[1]])
                res[key] = walk_session(sd.t[a:b], sd.o[a:b], sd.h[a:b], sd.l[a:b], sd.c[a:b], sd.v[a:b], side, walk,
                                        int(self.mcal[t]), index_tick=sym.startswith("_"))
            del parts
        return res


def _hm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


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
        qty = sim.LOT_SIZES[sym] if c.single else max(1, int(NOTIONAL // pin))
        tr = R._trade(cid, sym, cal[e], side, et, pin, cal[x], xt, pout, qty, cost_kind(c, side), slip)
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


def run_jobs(E: Engine, jobs: list) -> dict:
    """{job key: {"trades": [...], "trades5": [...]}}; minute walks of every job resolved in one pass."""
    from . import builder_options as BO
    plans, reqs, out = {}, [], {}
    for j in jobs:
        if j.cand.agent == "options":
            out[j.key] = BO.run_option_job(E, j)
            continue
        sig = j.sig if j.sig is not None else E.signals(j.cand)
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
