"""
stratlab/r6_portfolio.py — round-6 portfolio strategies (monthly / quarterly / weekly / daily cross-sectional and the
two-ETF switch) on factor_momentum's data, costs, metrics and gates. Research only; one look per strategy.

Pre-registration: stratlab/prereg/github_v1.json (byte-identical to <scratchpad>/PREREGISTERED_GITHUB.json). A
strategy runs only if it is listed there with the same fingerprint; the ledger refuses a second run of an id.

Engine (generalises factor_momentum.backtest; its rules unchanged unless stated):
  * decision at the close of a schedule session t (monthly = the month's last session; quarterly = last session of
    Mar/Jun/Sep/Dec; q_nov = last session of Nov/Feb/May/Aug; weekly = last session of each ISO week; daily = every
    session); the weights function reads rows <= t only; execution at the NEXT session's OPEN.
  * target weights may be negative (shorts = stock futures, simulated on the adjusted cash price: no basis / roll /
    margin interest, as the lab's stock-futures convention); cash earns 0.
  * a trade toward a non-zero target in a stock that did not trade that day is skipped; a full exit of such a stock
    is done at its last close (stale exit). No-trade band 0.5 % of equity for names that stay on the same side.
  * costs: costs.round_trip_cost per side ("equity_cnc" for long-only books, "future" for long / short books) plus
    10 bps slippage per side on traded value (factor_momentum.SLIP_BPS).
  * universe / data: era 1 = nse_old 2012-2023 point-in-time top-100 by turnover (delisted kept), era 2 = bars3y
    2023-11..2026-11 on the current ~97 NIFTY-100 names (SURVIVORSHIP-BIASED; history spliced from nse_old).
  * metrics / gates = factor_momentum.segment_metrics / judge: each era-1 half CAGR > 0 and > NIFTY, era-1 Sharpe
    >= 0.5, era-2 CAGR > 0 and > NIFTY (price index). PASS needs all seven.

    python -m stratlab.r6_portfolio --list
    python -m stratlab.r6_portfolio --smoke [--ids a,b]              # errors only (no metric is printed)
    python -m stratlab.r6_portfolio --prereg <PREREGISTERED_GITHUB.json> --ledger <ledger.json> --out <dir> \
        [--ids a,b] [--trials <trials_v2.csv>]
    python -m stratlab.r6_portfolio --january --out <dir>            # January barometer report (no verdict)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

from . import factor_momentum as FM
from . import kite_ind as K
from . import r6_ind as R

try:
    from costs import round_trip_cost
except ImportError:                     # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from costs import round_trip_cost

PREREG_SHA256 = "b283cda39dd84f6afd65175bc34f7901d9de542bc4bda079848297ec2ecfaf09"   # prereg/github_v1.json
SLIP_BPS = FM.SLIP_BPS
BAND = FM.BAND
CAPITAL = FM.CAPITAL
MIN_REAL_FRAC = FM.MIN_REAL_FRAC
ERA1 = dict(FM.ERA1)
ERA2 = dict(FM.ERA2)
ETFS = ("NIFTYBEES", "GOLDBEES")
_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
OLD_DIR = os.path.join(_SCRATCH, "nse_old")
NSE_DIR = os.path.join(_SCRATCH, "nse")
BARS3Y = os.path.join(_SCRATCH, "bars3y")
WORK = os.path.join(_SCRATCH, "github", "_cache")


# ══════════════════════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class XPanel(FM.Panel):
    high: object = None                  # (N, M) real highs / lows / volumes (NaN where not traded)
    low: object = None
    vol: object = None

    def __post_init__(self):
        super().__post_init__()
        N, M = self.close.shape
        for k in ("high", "low", "vol"):
            v = getattr(self, k)
            setattr(self, k, np.full((N, M), np.nan) if v is None else np.asarray(v, float))
        self.hf = FM.ffill_2d(np.where(np.isfinite(self.high), self.high, self.close))
        self.lf = FM.ffill_2d(np.where(np.isfinite(self.low), self.low, self.close))
        self.month_end = _month_end_index(self.dates)


def _month_end_index(dates) -> dict:
    """yyyymm -> index of the month's last session (months whose last session has a following session)."""
    d = np.asarray(dates, np.int64)
    out = {}
    for i in range(len(d) - 1):
        if d[i] // 100 != d[i + 1] // 100:
            out[int(d[i] // 100)] = i
    return out


def _read_old_eq(old_dir: str, cache: str):
    import pandas as pd
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    cols = ["date", "symbol", "series", "open", "high", "low", "close", "volume"]
    eq = pd.read_csv(os.path.join(old_dir, "eq_daily_adj.csv.gz"), usecols=cols)
    eq = eq.assign(pri=(eq.series != "EQ").astype(int)).sort_values(["symbol", "date", "pri"])
    eq = eq.drop_duplicates(["symbol", "date"], keep="first").drop(columns=["pri"])
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        eq.to_pickle(cache)
    return eq


def _pivot(eq, dates, syms, col):
    return eq.pivot(index="date", columns="symbol", values=col).reindex(index=dates, columns=syms).to_numpy(float)


def load_era1(old_dir: str = OLD_DIR, work: str = WORK):
    """Era-1 XPanel (nse_old, adjusted, point-in-time universe), {date: NIFTY close}, the old equity frame."""
    import pandas as pd
    uni = pd.read_csv(os.path.join(old_dir, "universe_monthly.csv"), usecols=["month", "symbol"])
    universe = {int(m): set(g.symbol) for m, g in uni.groupby("month")}
    names = sorted(set(uni.symbol))
    idx = pd.read_csv(os.path.join(old_dir, "index_daily.csv.gz"), usecols=["date", "index", "close"])
    nf = idx[idx["index"] == "NIFTY"].sort_values("date")
    dates = nf.date.to_numpy(np.int64)
    eq = _read_old_eq(old_dir, os.path.join(work, "eq_old_ohlcv.pkl"))
    sub = eq[eq.symbol.isin(names)]
    arr = {c: _pivot(sub, dates, names, c) for c in ("open", "high", "low", "close", "volume")}
    p = XPanel(dates, names, arr["close"], arr["open"], nf.close.to_numpy(float), universe=universe,
               label="era1 nse_old 2012-2023 point-in-time top-100", high=arr["high"], low=arr["low"],
               vol=arr["volume"])
    return p, dict(zip(dates.tolist(), p.nifty.tolist())), eq


def load_era2(old_eq, bars_dir: str = BARS3Y, old_dir: str = OLD_DIR, nse_dir: str = NSE_DIR):
    """Era-2 XPanel: bars3y daily OHLCV of the current names (SURVIVORSHIP-BIASED) with history spliced from nse_old
    exactly as factor_momentum.load_era2 (junction return from the 2023-11-10 bhavcopy; CA_GAPS back-adjusted);
    spliced volumes are divided by the price factor."""
    import pandas as pd
    from . import data as D
    from .ltr import CA_GAPS
    man = D.prepare(bars_dir)
    cdir = man["cache_dir"]
    daily = {}
    for s in sorted(man["symbols"]):
        sd = D.load_symbol(cdir, s)
        if sd is not None:
            daily[s] = sd.daily()
    syms = sorted(daily)
    new_dates = sorted({int(x) for d in daily.values() for x in d["date"] if x >= ERA2["junction"]})
    idx = pd.read_csv(os.path.join(old_dir, "index_daily.csv.gz"), usecols=["date", "index", "close"])
    nf_old = idx[idx["index"] == "NIFTY"].sort_values("date")
    old_dates = [int(x) for x in nf_old.date if x <= ERA2["splice_last_old"]]
    dates = np.array(old_dates + new_dates, np.int64)
    pos = {int(d): i for i, d in enumerate(dates)}
    N, M = len(dates), len(syms)
    A = {k: np.full((N, M), np.nan) for k in "ohlcv"}
    bh = pd.read_csv(os.path.join(nse_dir, "delivery.csv.gz"), usecols=["date", "symbol", "close", "prev_close"])
    bh = bh[bh.date == ERA2["junction"]].set_index("symbol")
    oe = old_eq[old_eq.date <= ERA2["splice_last_old"]]
    oe_by = {s: g for s, g in oe.groupby("symbol")}
    for j, s in enumerate(syms):
        d = daily[s]
        m = d["date"] >= ERA2["junction"]
        rows = [pos[int(x)] for x in d["date"][m]]
        for k in "ohlcv":
            A[k][rows, j] = d[k][m]
        old_sym = FM.SPLICE_MAP.get(s, s)
        g = oe_by.get(old_sym)
        if g is None or g.empty:
            continue
        jd = d["date"][m]
        if len(jd) == 0 or int(jd[0]) != ERA2["junction"] or old_sym not in bh.index \
                or int(g.date.iloc[-1]) != ERA2["splice_last_old"]:
            continue
        r_j = float(bh.loc[old_sym, "close"]) / float(bh.loc[old_sym, "prev_close"])
        kf = float(d["c"][m][0]) / (r_j * float(g.close.iloc[-1]))
        keep = g[g.date.isin(pos.keys())]
        rr = [pos[int(x)] for x in keep.date]
        A["o"][rr, j] = keep.open.to_numpy(float) * kf
        A["h"][rr, j] = keep.high.to_numpy(float) * kf
        A["l"][rr, j] = keep.low.to_numpy(float) * kf
        A["c"][rr, j] = keep.close.to_numpy(float) * kf
        A["v"][rr, j] = keep.volume.to_numpy(float) / kf
    gaps = {}
    for (s, gd), _why in CA_GAPS.items():
        if s in syms and gd in pos:
            j, gi = syms.index(s), pos[gd]
            prev = [i for i in range(gi) if np.isfinite(A["c"][i, j])][-1]
            f = A["o"][gi, j] / A["c"][prev, j]
            for k in "ohlc":
                A[k][:gi, j] *= f
            A["v"][:gi, j] /= f
            gaps.setdefault(s, []).append(int(gd))
    nifty = _nifty_era2(bars_dir, nf_old, dates)
    p = XPanel(dates, syms, A["c"], A["o"], nifty, universe=None, ca_gaps=gaps,
               label="era2 bars3y 2023-11..2026-11, current NIFTY-100 names (SURVIVORSHIP-BIASED)",
               high=A["h"], low=A["l"], vol=A["v"])
    return p, dict(zip(dates.tolist(), p.nifty.tolist()))


def _nifty_era2(bars_dir, nf_old, dates):
    from . import data as D
    man = D.prepare(bars_dir)
    aux = D.load_daily_aux(man["cache_dir"], "_NIFTY50_day")
    nmap = dict(zip(nf_old.date.astype(int), nf_old.close.astype(float)))
    first_aux = int(aux["date"][0])
    nmap = {k: v for k, v in nmap.items() if k < first_aux}
    nmap.update({int(a): float(b) for a, b in zip(aux["date"], aux["c"])})
    return np.array([nmap.get(int(x), np.nan) for x in dates])


def load_etf(era: int, old_eq, old_dir: str = OLD_DIR, nse_dir: str = NSE_DIR, bars_dir: str = BARS3Y):
    """NIFTYBEES / GOLDBEES panel. Era 1: nse_old adjusted EQ rows on the NIFTY calendar 2012-2023. Era 2: the same
    history spliced (junction return from the 2023-11-10 bhavcopy) to the NSE bhavcopy (sec_bhavdata, unadjusted: no
    split / bonus in either ETF 2023-11..2026-11) on the bars3y NIFTY calendar."""
    import pandas as pd
    idx = pd.read_csv(os.path.join(old_dir, "index_daily.csv.gz"), usecols=["date", "index", "close"])
    nf_old = idx[idx["index"] == "NIFTY"].sort_values("date")
    syms = list(ETFS)
    sub = old_eq[old_eq.symbol.isin(syms)]
    if era == 1:
        dates = nf_old.date.to_numpy(np.int64)
        o, c = _pivot(sub, dates, syms, "open"), _pivot(sub, dates, syms, "close")
        p = XPanel(dates, syms, c, o, nf_old.close.to_numpy(float), label="era1 NIFTYBEES/GOLDBEES (nse_old)")
        return p, dict(zip(dates.tolist(), p.nifty.tolist()))
    from . import data as D
    man = D.prepare(bars_dir)
    aux = D.load_daily_aux(man["cache_dir"], "_NIFTY50_day")
    new_dates = [int(x) for x in aux["date"] if x >= ERA2["junction"]]
    old_dates = [int(x) for x in nf_old.date if x <= ERA2["splice_last_old"]]
    dates = np.array(old_dates + new_dates, np.int64)
    pos = {int(d): i for i, d in enumerate(dates)}
    bh = pd.read_csv(os.path.join(nse_dir, "delivery.csv.gz"), usecols=["date", "symbol", "open", "close", "prev_close"])
    bh = bh[bh.symbol.isin(syms)]
    o = np.full((len(dates), 2), np.nan)
    c = np.full((len(dates), 2), np.nan)
    for j, s in enumerate(syms):
        g = sub[(sub.symbol == s) & (sub.date <= ERA2["splice_last_old"])]
        b = bh[(bh.symbol == s) & (bh.date >= ERA2["junction"])]
        for dt, oo, cc in zip(b.date, b.open, b.close):
            if int(dt) in pos:
                o[pos[int(dt)], j], c[pos[int(dt)], j] = oo, cc
        j0 = b[b.date == ERA2["junction"]]
        kf = float(j0.close.iloc[0]) / (float(j0.close.iloc[0]) / float(j0.prev_close.iloc[0]) * float(g.close.iloc[-1]))
        for dt, oo, cc in zip(g.date, g.open, g.close):
            if int(dt) in pos:
                o[pos[int(dt)], j], c[pos[int(dt)], j] = oo * kf, cc * kf
    nifty = _nifty_era2(bars_dir, nf_old, dates)
    p = XPanel(dates, syms, c, o, nifty, label="era2 NIFTYBEES/GOLDBEES (bhavcopy, spliced)")
    return p, dict(zip(dates.tolist(), p.nifty.tolist()))


# ══════════════════════════════════════════════════════════════════════════
# engine
# ══════════════════════════════════════════════════════════════════════════
def schedule(kind: str, dates) -> list:
    d = np.asarray(dates, np.int64)
    n = len(d)
    if kind == "daily":
        return list(range(n - 1))
    if kind == "weekly":
        wk = [_dt.date(int(x) // 10000, int(x) // 100 % 100, int(x) % 100).isocalendar()[:2] for x in d]
        return [i for i in range(n - 1) if wk[i] != wk[i + 1]]
    ms = FM.decision_sessions(d)
    if kind == "monthly":
        return ms
    months = {"quarterly": (3, 6, 9, 12), "q_nov": (11, 2, 5, 8)}[kind]
    return [i for i in ms if int(d[i]) // 100 % 100 in months]


def trade_costs(kind: str):
    if kind == "cnc":
        return FM.trade_costs

    def fut(value: float, side: str):
        if value <= 0:
            return 0.0, 0.0
        stat = round_trip_cost(value, 0.0, "future") if side == "BUY" else round_trip_cost(0.0, value, "future")
        return float(stat), value * SLIP_BPS / 1e4
    return fut


@dataclass
class Ctx:
    t: int
    exec_month: int
    cur: dict                 # j -> current weight (value at t's close / equity)
    equity: float
    state: dict


@dataclass
class PSpec:
    id: str
    desc: str
    weights: object           # fn(p, t, ctx, params) -> {j: w} or None (= keep the book unchanged)
    params: dict
    schedule: str
    book: str                 # "cnc" (long-only delivery) | "futures" (long / short stock futures)
    panel: str = "stocks"     # "stocks" | "etf"
    era1_first: int = ERA1["first_decision"]

    @property
    def fingerprint(self) -> str:
        src = "".join(inspect.getsource(f) for f in (self.weights, backtest, elig, trade_costs, schedule) + _HELPERS())
        blob = repr((self.id, sorted(self.params.items()), self.schedule, self.book, self.panel, self.era1_first, src))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _HELPERS() -> tuple:
    """Shared helpers every portfolio fingerprint covers (a change to any of them is a new strategy)."""
    return (_resid_scores, _ls, _rank_sel, _month_ret, _atr_last, R.residual_score, R.alpha101,
            R.clenow_momentum, R.pct_rank_desc, K.atr, K.rma, K.true_range)


def backtest(p: XPanel, spec: PSpec, first_decision: int, end: int | None = None, capital: float = CAPITAL) -> dict:
    """One strategy on one panel. Same output layout as factor_momentum.backtest (+ gross exposure)."""
    dates = p.dates
    last = len(dates) - 1 if end is None else int(np.searchsorted(dates, end, side="right")) - 1
    decs = [t for t in schedule(spec.schedule, dates[: last + 1]) if dates[t] >= first_decision]
    if not decs:
        raise ValueError("no decision session in range")
    t0 = decs[0]
    decset = set(decs)
    M = len(p.symbols)
    sh = np.zeros(M)
    cash = capital
    cost_fn = trade_costs(spec.book)
    state: dict = {}
    eq_dates, eq, gross = [int(dates[t0])], [capital], [0.0]
    stat_tot = slip_tot = traded = 0.0
    stale_exits, n_rebal = 0, 0
    log = []

    def decide(t):
        val = sh * np.nan_to_num(p.cf[t])
        e = cash + val.sum()
        cur = {int(j): float(val[j] / e) for j in np.flatnonzero(sh)} if e > 0 else {}
        return spec.weights(p, t, Ctx(t, int(dates[t + 1]) // 100, cur, float(e), state), spec.params)

    pending = decide(t0)
    for i in range(t0 + 1, last + 1):
        if pending is not None:
            px = p.open[i].copy()
            stale = ~np.isfinite(px)
            px[stale] = p.cf[i - 1][stale]
            val = sh * np.nan_to_num(px)
            equity = cash + val.sum()
            todo = []
            for j in sorted(set(int(x) for x in np.flatnonzero(sh)) | set(pending)):
                w = float(pending.get(j, 0.0))
                tgt, cur = w * equity, val[j]
                d = tgt - cur
                if abs(d) < 1e-9:
                    continue
                if tgt != 0 and cur != 0 and np.sign(tgt) == np.sign(cur) and abs(d) < BAND * equity:
                    continue
                if stale[j] and tgt != 0:
                    continue
                todo.append((j, d, tgt == 0))
            for j, d, to_zero in todo:
                v = abs(d)
                stat, slip = cost_fn(v, "BUY" if d > 0 else "SELL")
                sh[j] = 0.0 if to_zero else sh[j] + d / px[j]
                cash -= d + stat + slip
                stat_tot += stat
                slip_tot += slip
                traded += v
                se = bool(stale[j]) and to_zero
                stale_exits += int(se)
                log.append((int(dates[i]), p.symbols[j], "BUY" if d > 0 else "SELL", round(v, 2),
                            round(stat + slip, 2), se))
            n_rebal += 1
            pending = None
        mark = sh * np.nan_to_num(p.cf[i])
        e = cash + float(mark.sum())
        eq_dates.append(int(dates[i]))
        eq.append(e)
        gross.append(float(np.abs(mark).sum() / e) if e > 0 else float("nan"))
        if e <= 0:
            raise RuntimeError(f"{spec.id}: equity <= 0 on {dates[i]}")
        if i in decset and i < last:
            pending = decide(i)
    return {"dates": np.array(eq_dates), "equity": np.array(eq), "gross": np.array(gross), "statutory": stat_tot,
            "slippage": slip_tot, "traded": traded, "stale_exits": stale_exits, "rebalances": n_rebal,
            "cash_switches": 0, "trades": log, "first_decision": int(dates[t0]), "last": int(dates[last])}


def elig(p: XPanel, t: int, exec_month: int, lb: int) -> np.ndarray:
    """factor_momentum.eligible with a strategy-specific lookback: traded at t, real on >= 90 % of the lb+1 sessions
    t-lb..t, a close lb sessions back, universe member for the execution month, no CA_GAPS date in the window."""
    M = len(p.symbols)
    ok = np.zeros(M, bool)
    if t < lb:
        return ok
    frac = p.real[t - lb: t + 1].sum(axis=0) / (lb + 1)
    ok = p.real[t] & (frac >= MIN_REAL_FRAC) & np.isfinite(p.cf[t - lb])
    if p.universe is not None:
        mem = p.universe.get(int(exec_month), set())
        ok &= np.array([s in mem for s in p.symbols])
    if p.ca_gaps:
        lo, hi = p.dates[t - lb], p.dates[t]
        for j, s in enumerate(p.symbols):
            if ok[j] and any(lo < g <= hi for g in p.ca_gaps.get(s, ())):
                ok[j] = False
    return ok


def _rank_sel(p, score: dict, n: int, top: bool) -> list:
    """n symbols with the highest (top) / lowest score; ties broken by symbol name."""
    keys = sorted(score, key=lambda j: ((-score[j]) if top else score[j], p.symbols[j]))
    return keys[:n]


def _ls(p, score: dict, n: int, gross_side: float = 0.5) -> dict:
    if n < 1 or len(score) < 2 * n:
        return {}
    w = {j: gross_side / n for j in _rank_sel(p, score, n, True)}
    for j in _rank_sel(p, score, n, False):
        w[j] = -gross_side / n
    return w


# ══════════════════════════════════════════════════════════════════════════
# 1. residual momentum (QuantConnect Strategy Library 136)
# ══════════════════════════════════════════════════════════════════════════
def _resid_scores(p, t, ctx):
    me = sorted(v for v in p.month_end.values() if v <= t)
    if t not in p.month_end.values() or len(me) < 37:
        return {}
    ends = np.array(me[-37:])
    ok = elig(p, t, ctx.exec_month, int(t - ends[0]))
    cm = p.cf[ends]
    sr = cm[1:] / cm[:-1] - 1.0
    mk = p.nifty[ends]
    mr = mk[1:] / mk[:-1] - 1.0
    out = {}
    for j in np.flatnonzero(ok):
        sc = R.residual_score(sr[:, j], mr)
        if np.isfinite(sc):
            out[int(j)] = sc
    return out


def w_resid_long(p, t, ctx, prm):
    sc = _resid_scores(p, t, ctx)
    if not sc:
        return {}
    n = max(1, int(len(sc) * prm["decile"]))
    return {j: 1.0 / n for j in _rank_sel(p, sc, n, True)}


def w_resid_ls(p, t, ctx, prm):
    sc = _resid_scores(p, t, ctx)
    return _ls(p, sc, max(1, int(len(sc) * prm["decile"])), prm["gross_side"]) if sc else {}


# ══════════════════════════════════════════════════════════════════════════
# 2. same-calendar-month seasonality (QuantConnect 269 / Heston-Sadka)
# ══════════════════════════════════════════════════════════════════════════
def _month_ret(p, yyyymm: int) -> np.ndarray:
    y, m = yyyymm // 100, yyyymm % 100
    prev = (y * 100 + m - 1) if m > 1 else ((y - 1) * 100 + 12)
    a, b = p.month_end.get(prev), p.month_end.get(yyyymm)
    M = len(p.symbols)
    if a is None or b is None:
        return np.full(M, np.nan)
    ok = p.real[a] & p.real[b]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = p.close[b] / p.close[a] - 1.0
    return np.where(ok, r, np.nan)


def w_season(p, t, ctx, prm):
    em = ctx.exec_month
    y, m = em // 100, em % 100
    ok = elig(p, t, em, 21)
    rs = np.array([_month_ret(p, (y - k) * 100 + m) for k in range(1, prm["years"] + 1)])
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(rs).sum(axis=0)
        avg = np.where(cnt >= 1, np.nansum(rs, axis=0) / np.maximum(cnt, 1), np.nan)
    sc = {int(j): float(avg[j]) for j in np.flatnonzero(ok) if np.isfinite(avg[j])}
    return _ls(p, sc, prm["n"], prm["gross_side"])


# ══════════════════════════════════════════════════════════════════════════
# 3. momentum x turnover (QuantConnect 66, Lee-Swaminathan), 3 overlapping monthly tranches
# ══════════════════════════════════════════════════════════════════════════
def w_mom_turnover(p, t, ctx, prm):
    lb = prm["roc"]
    ok = elig(p, t, ctx.exec_month, lb)
    val = p.close * p.vol                                   # traded value (Rs, close x volume); NaN if no trade
    win = val[t - prm["turn_avg"] + 1: t + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.nanmean(win, axis=0)
        turn = val[t] / avg
        roc = p.cf[t] / p.cf[t - lb] - 1.0
    cand = [int(j) for j in np.flatnonzero(ok) if np.isfinite(roc[j]) and np.isfinite(turn[j])]
    tr = ctx.state.setdefault("tranches", [])
    if len(cand) >= 5:
        order = sorted(cand, key=lambda j: (-roc[j], p.symbols[j]))
        k = int(len(order) * prm["mom_frac"])
        top, bot = order[:k], order[len(order) - k:]
        pick = lambda xs: sorted(xs, key=lambda j: (-turn[j], p.symbols[j]))[:max(1, int(len(xs) / prm["turn_split"]))]
        lng, sht = pick(top), pick(bot)
        new = {}
        for j in lng:
            new[j] = new.get(j, 0.0) + prm["tranche"] / len(lng)
        for j in sht:
            new[j] = new.get(j, 0.0) - prm["tranche"] / len(sht)
    else:
        new = {}
    tr.append(new)
    if len(tr) > prm["hold"]:
        tr.pop(0)
    out: dict = {}
    for d in tr:
        for j, w in d.items():
            out[j] = out.get(j, 0.0) + w
    return {j: w for j, w in out.items() if abs(w) > 1e-12}


# ══════════════════════════════════════════════════════════════════════════
# 5. NIFTYBEES vs GOLDBEES quarterly switch (marketcalls dual_momentum_backtest.py; QuantConnect 33)
# ══════════════════════════════════════════════════════════════════════════
def w_etf_switch(p, t, ctx, prm):
    ym = int(p.dates[t]) // 100
    y, m = ym // 100, ym % 100 - 3
    if m < 1:
        y, m = y - 1, m + 12
    q = p.month_end.get(y * 100 + m)                        # the previous quarter-end session
    if q is None:
        return None
    r = p.cf[t] / p.cf[q] - 1.0
    if not np.isfinite(r).all():
        return None
    win = int(np.argmax(r))                                 # idxmax: the first on a tie (NIFTYBEES)
    if ctx.state.get("held") == win:
        return None                                         # rebalance only on a switch (as coded)
    ctx.state["held"] = win
    return {win: prm["alloc"]}


# ══════════════════════════════════════════════════════════════════════════
# 12. Alpha101 subset, cross-sectional daily quintiles (long / short stock futures)
# ══════════════════════════════════════════════════════════════════════════
def w_alpha(p, t, ctx, prm):
    if t < 6:
        return {}
    ok = elig(p, t, ctx.exec_month, 21)
    a = R.alpha101(prm["alpha"], p.open, p.high, p.low, p.close, p.vol, t)
    sc = {int(j): float(a[j]) for j in np.flatnonzero(ok) if np.isfinite(a[j])}
    return _ls(p, sc, int(len(sc) * prm["quantile"]), prm["gross_side"])


# ══════════════════════════════════════════════════════════════════════════
# 20. cross-sectional IBS, 1 session (long lowest-IBS quintile, short highest)
# ══════════════════════════════════════════════════════════════════════════
def w_ibs(p, t, ctx, prm):
    ok = elig(p, t, ctx.exec_month, 21)
    h, l, c = p.high[t], p.low[t], p.close[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        ibs = np.where(h > l, (c - l) / (h - l), np.nan)
    sc = {int(j): -float(ibs[j]) for j in np.flatnonzero(ok) if np.isfinite(ibs[j])}   # high score = low IBS
    return _ls(p, sc, int(len(sc) * prm["quantile"]), prm["gross_side"])


# ══════════════════════════════════════════════════════════════════════════
# 16. Clenow "Stocks on the Move" (Suchismit4/NiftyOnTheMove strategy.py; book rules where the repo deviates)
# ══════════════════════════════════════════════════════════════════════════
def _atr_last(p, j, t, n, tail=250):
    a = max(0, t - tail)
    at = K.atr(p.hf[a: t + 1, j], p.lf[a: t + 1, j], p.cf[a: t + 1, j], n)
    return float(at[-1])


def w_clenow(p, t, ctx, prm):
    L, S = prm["lookback"], prm["sma_short"]
    st = ctx.state
    st["week"] = st.get("week", -1) + 1
    ok = elig(p, t, ctx.exec_month, max(L, S))
    ranked, info = [], {}
    for j in np.flatnonzero(ok):
        cl = p.cf[t - L: t + 1, j]
        if np.any(np.abs(np.diff(cl) / cl[:-1]) > prm["gap"]):
            continue
        sma = float(p.cf[t - S + 1: t + 1, j].mean())
        px = float(p.cf[t, j])
        if px < sma:
            continue
        mom = float(R.clenow_momentum(p.cf[t - L + 1: t + 1, j][:, None])[0])
        if not np.isfinite(mom) or mom < prm["min_momentum"]:
            continue
        atr = _atr_last(p, j, t, prm["atr"])
        if not (atr > 0):
            continue
        ranked.append((int(j), mom))
        info[int(j)] = (px, sma, atr)
    ranked.sort(key=lambda x: (-x[1], p.symbols[x[0]]))
    top = [j for j, _m in ranked[: int(len(ranked) * prm["top_frac"])]]
    tops = set(top)
    tgt = {j: w for j, w in ctx.cur.items() if w > 0 and j in tops}          # sell what left the top 20 %
    size = lambda j: prm["risk"] * info[j][0] / info[j][2]                   # shares = equity x 0.1 % / ATR
    if st["week"] % 2 == 0:                                                  # position re-size every other week
        for j, w in list(tgt.items()):
            want = size(j)
            if abs(want - w) / w > prm["resize_tol"]:
                if want > w and info[j][0] > info[j][1]:
                    tgt[j] = want
                elif want < w:
                    tgt[j] = want
    sma200 = float(np.mean(p.nifty[t - prm["sma_long"] + 1: t + 1])) if t + 1 >= prm["sma_long"] else np.nan
    if np.isfinite(sma200) and p.nifty[t] > sma200:
        avail = 1.0 - sum(tgt.values())
        for j in top:
            if j in tgt:
                continue
            if len(tgt) >= prm["max_stocks"]:
                break
            w = size(j)
            if w > avail:
                break
            tgt[j] = w
            avail -= w
    return tgt


# ══════════════════════════════════════════════════════════════════════════
# 17. QMOM (quantrocket-codeload qmom/qmom.py), quarterly before the Feb/May/Aug/Nov month-ends
# ══════════════════════════════════════════════════════════════════════════
def w_qmom(p, t, ctx, prm):
    W, X = prm["window"], prm["skip"]
    ok = elig(p, t, ctx.exec_month, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = (p.cf[t - X] - p.cf[t - W]) / p.cf[t - W]
        posd = (np.diff(p.cf[t - W: t + 1], axis=0) > 0).sum(axis=0).astype(float)
    ret = np.where(ok & np.isfinite(ret), ret, np.nan)
    have = R.pct_rank_desc(ret) <= prm["top_pct"]
    pd_m = np.where(have, posd, np.nan)
    smooth = R.pct_rank_desc(pd_m) <= prm["smooth_pct"]
    pick = [int(j) for j in np.flatnonzero(smooth)]
    return {j: 1.0 / len(pick) for j in pick} if pick else {}


# ══════════════════════════════════════════════════════════════════════════
# 19. per-stock TSMOM, volatility-scaled (rkohli3/TSMOM core.tsmom / get_tsmom, MomentumIndia notebook)
# ══════════════════════════════════════════════════════════════════════════
def w_tsmom(p, t, ctx, prm):
    me = sorted(v for v in p.month_end.values() if v <= t)
    if t not in p.month_end.values() or len(me) < 13:
        return {}
    t12 = me[-13]
    ok = elig(p, t, ctx.exec_month, int(t - t12))
    a = max(1, t - prm["vol_tail"])
    import pandas as pd
    rets = p.cf[a - 1: t + 1, :][1:] / p.cf[a - 1: t + 1, :][:-1] - 1.0
    vol = pd.DataFrame(rets).ewm(com=prm["com"]).std().to_numpy()[-1] * math.sqrt(261)
    with np.errstate(invalid="ignore", divide="ignore"):
        r12 = p.cf[t] / p.cf[t12] - 1.0
    use = [int(j) for j in np.flatnonzero(ok) if np.isfinite(r12[j]) and np.isfinite(vol[j]) and vol[j] > 0
           and r12[j] != 0]
    if not use:
        return {}
    n = len(use)
    return {j: float(np.sign(r12[j])) * (prm["scale"] / vol[j]) / n for j in use}


# ══════════════════════════════════════════════════════════════════════════
# registry
# ══════════════════════════════════════════════════════════════════════════
SPECS = [
    PSpec("r6_gh_resid_mom_long", "QC 136 residual momentum (NIFTY-only regression, 36 months), long top decile, monthly, CNC",
          w_resid_long, {"months": 36, "score": "t-11..t-1", "decile": 0.10}, "monthly", "cnc", era1_first=20150130),
    PSpec("r6_gh_resid_mom_ls", "QC 136 residual momentum, long top / short bottom decile (stock futures), monthly",
          w_resid_ls, {"months": 36, "score": "t-11..t-1", "decile": 0.10, "gross_side": 0.5}, "monthly", "futures",
          era1_first=20150130),
    PSpec("r6_gh_season_1y", "QC 269 same-calendar-month % return 1 year ago: long top 10 / short bottom 10, monthly",
          w_season, {"years": 1, "n": 10, "gross_side": 0.5}, "monthly", "futures"),
    PSpec("r6_gh_season_5y", "Heston-Sadka same-month return averaged over years 1-5 (available): long top 10 / short bottom 10",
          w_season, {"years": 5, "n": 10, "gross_side": 0.5}, "monthly", "futures"),
    PSpec("r6_gh_mom_turnover", "QC 66 momentum x turnover proxy: top/bottom 20 % ROC(252), highest-turnover third; 3 x 1/3 tranches held 3 months",
          w_mom_turnover, {"roc": 252, "mom_frac": 0.2, "turn_split": 3, "turn_avg": 60, "tranche": 1.0 / 3, "hold": 3},
          "monthly", "futures"),
    PSpec("r6_gh_etf_switch", "marketcalls dual momentum: quarter-end winner of NIFTYBEES / GOLDBEES (fixed lag) at 75 %, switch only",
          w_etf_switch, {"alloc": 0.75, "assets": "NIFTYBEES,GOLDBEES"}, "quarterly", "cnc", panel="etf"),
    PSpec("r6_gh_alpha101_a101", "Alpha#101 (close-open)/(high-low+.001): daily long top / short bottom quintile (stock futures)",
          w_alpha, {"alpha": "a101", "quantile": 0.2, "gross_side": 0.5}, "daily", "futures"),
    PSpec("r6_gh_alpha101_a33", "Alpha#33 rank(-(1-open/close)): daily long top / short bottom quintile (stock futures)",
          w_alpha, {"alpha": "a33", "quantile": 0.2, "gross_side": 0.5}, "daily", "futures"),
    PSpec("r6_gh_alpha101_a12", "Alpha#12 sign(dvolume)*(-dclose): daily long top / short bottom quintile (stock futures)",
          w_alpha, {"alpha": "a12", "quantile": 0.2, "gross_side": 0.5}, "daily", "futures"),
    PSpec("r6_gh_alpha101_a54", "Alpha#54 -(low-close)*open^5/((low-high)*close^5): daily long top / short bottom quintile",
          w_alpha, {"alpha": "a54", "quantile": 0.2, "gross_side": 0.5}, "daily", "futures"),
    PSpec("r6_gh_alpha101_a9", "Alpha#9 (5-day delta-close trend / reversal switch): daily long top / short bottom quintile",
          w_alpha, {"alpha": "a9", "quantile": 0.2, "gross_side": 0.5}, "daily", "futures"),
    PSpec("r6_gh_ibs_xs", "cross-sectional IBS: long lowest-IBS quintile / short highest, rebalanced every session (stock futures)",
          w_ibs, {"quantile": 0.2, "gross_side": 0.5}, "daily", "futures"),
    PSpec("r6_gh_clenow", "Clenow Stocks on the Move: slope*R^2 (90d), SMA100 / 15 % gap filters, NIFTY > SMA200, 0.1 % ATR risk, max 15, weekly",
          w_clenow, {"lookback": 90, "sma_short": 100, "sma_long": 200, "atr": 20, "risk": 0.001, "gap": 0.15,
                     "min_momentum": 0.0, "top_frac": 0.2, "max_stocks": 15, "resize_tol": 0.05}, "weekly", "cnc"),
    PSpec("r6_gh_qmom", "QMOM: top 10 % by 12-1 return, keep the 50 % with most positive days, equal weight, quarterly (Q-NOV)",
          w_qmom, {"window": 252, "skip": 22, "top_pct": 0.10, "smooth_pct": 0.50}, "q_nov", "cnc"),
    PSpec("r6_gh_tsmom_stock", "per-stock TSMOM: sign of 12-month return, weight 0.5 / ex-ante vol (EWM com 60) / N, monthly (stock futures)",
          w_tsmom, {"lookback_months": 12, "scale": 0.5, "com": 60, "vol_tail": 750}, "monthly", "futures"),
]
BY_ID = {s.id: s for s in SPECS}


# ══════════════════════════════════════════════════════════════════════════
# January barometer (report only)
# ══════════════════════════════════════════════════════════════════════════
def january_barometer(dates, nifty) -> dict:
    """Hold NIFTY every January; stay long Feb-Dec only if January's return (last December close -> last January
    close) is positive, else cash until the next January. Index closes, no costs: a REPORT (about 14 decisions)."""
    me = _month_end_index(dates)
    rows = []
    for y in range(int(dates[0]) // 10000 + 1, int(dates[-1]) // 10000 + 1):
        a, b, e = me.get((y - 1) * 100 + 12), me.get(y * 100 + 1), me.get(y * 100 + 12, len(dates) - 1)
        if a is None or b is None:
            continue
        jan = nifty[b] / nifty[a] - 1
        rest = nifty[e] / nifty[b] - 1
        rows.append({"year": y, "jan": jan, "rest": rest, "invested": jan > 0, "to": int(dates[e])})
    strat = float(np.prod([(1 + r["jan"]) * ((1 + r["rest"]) if r["invested"] else 1.0) for r in rows]))
    bh = float(np.prod([(1 + r["jan"]) * (1 + r["rest"]) for r in rows]))
    return {"rows": rows, "strategy_growth": strat, "buy_hold_growth": bh,
            "years_out": sum(1 for r in rows if not r["invested"]),
            "out_years_rest_return": [round(r["rest"], 4) for r in rows if not r["invested"]]}


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _daily_sr(res, lo, hi):
    d, e = res["dates"], res["equity"]
    m = (d >= lo) & (d <= hi)
    x = e[m]
    r = x[1:] / x[:-1] - 1
    return float(r.mean() / r.std()) if len(r) > 2 and r.std() > 0 else float("nan")


def run_one(spec: PSpec, panels: dict) -> dict:
    p1, n1 = panels["etf1" if spec.panel == "etf" else "era1"]
    p2, n2 = panels["etf2" if spec.panel == "etf" else "era2"]
    r1 = backtest(p1, spec, max(spec.era1_first, ERA1["first_decision"]), ERA1["end"])
    h1_end = int(r1["dates"][r1["dates"] <= ERA1["h1_end"]][-1])
    era1 = {"h1": FM.segment_metrics(r1, n1, r1["dates"][0], h1_end),
            "h2": FM.segment_metrics(r1, n1, h1_end, r1["dates"][-1]),
            "whole": FM.segment_metrics(r1, n1, r1["dates"][0], r1["dates"][-1]),
            "stale_exits": r1["stale_exits"], "rebalances": r1["rebalances"], "trades": len(r1["trades"]),
            "mean_gross": float(np.nanmean(r1["gross"][1:])), "sr_daily": _daily_sr(r1, r1["dates"][0], r1["dates"][-1])}
    r2 = backtest(p2, spec, ERA2["first_decision"])
    era2 = {"whole": FM.segment_metrics(r2, n2, r2["dates"][0], r2["dates"][-1]), "stale_exits": r2["stale_exits"],
            "rebalances": r2["rebalances"], "trades": len(r2["trades"]), "label": p2.label,
            "mean_gross": float(np.nanmean(r2["gross"][1:])), "sr_daily": _daily_sr(r2, r2["dates"][0], r2["dates"][-1])}
    return {"era1": era1, "era2": era2, **FM.judge(era1, era2), "_r1": r1, "_r2": r2}


def load_panels(need_etf: bool, need_stocks: bool) -> dict:
    os.makedirs(WORK, exist_ok=True)
    out = {}
    p1, n1, eq = load_era1()
    if need_stocks:
        out["era1"] = (p1, n1)
        out["era2"] = load_era2(eq)
    if need_etf:
        out["etf1"] = load_etf(1, eq)
        out["etf2"] = load_etf(2, eq)
    return out


def _trial_rows(spec, res, prereg_sha):
    e1, e2 = res["era1"], res["era2"]
    base = {"strategy_id": spec.id, "family": "r6_gh", "group": "gh_portfolio", "params_fp": spec.fingerprint,
            "cost_model": f"{'equity_cnc' if spec.book == 'cnc' else 'future'}+{SLIP_BPS:g}bps",
            "source": "github/portfolio_ledger.json", "verdict": res["verdict"]}
    return [dict(base, era="2012-23", protocol="r6 portfolio (factor_momentum gates)", stage="oos", dataset="nse_old",
                 period=f"{e1['whole']['from']}-{e1['whole']['to']}", trades=e1["trades"], sr_daily=e1["sr_daily"],
                 note=f"CAGR {e1['whole']['cagr']:.4f} excess H1 {e1['h1']['excess']:.4f} H2 {e1['h2']['excess']:.4f} "
                      f"Sharpe {e1['whole']['sharpe']:.2f} prereg {prereg_sha[:12]}"),
            dict(base, era="2023-26", protocol="r6 portfolio (factor_momentum gates)", stage="screen", dataset="bars3y",
                 period=f"{e2['whole']['from']}-{e2['whole']['to']}", trades=e2["trades"], sr_daily=e2["sr_daily"],
                 note=f"CAGR {e2['whole']['cagr']:.4f} excess {e2['whole']['excess']:.4f} (survivorship-biased)")]


def main(argv=None) -> int:
    from .data import assert_outside_repo
    ap = argparse.ArgumentParser(prog="python -m stratlab.r6_portfolio")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="run each spec on a short window; report errors only")
    ap.add_argument("--january", action="store_true")
    ap.add_argument("--prereg", default="")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--trials", default=os.environ.get("STRATLAB_TRIALS", ""))
    a = ap.parse_args(argv)
    ids = [x for x in a.ids.split(",") if x] or [s.id for s in SPECS]
    specs = [BY_ID[i] for i in ids]
    if a.list:
        for s in SPECS:
            print(f"{s.id:<26} {s.fingerprint}  {s.schedule:<9} {s.book:<8} {s.desc}")
        return 0
    if a.january:
        out = assert_outside_repo(a.out)
        os.makedirs(out, exist_ok=True)
        _p1, _n1, eq = load_era1()
        rep = {}
        for era, (p, _n) in (("era1", (_p1, _n1)), ("era2", load_era2(eq))):
            rep[era] = january_barometer(p.dates, p.nifty)
        with open(os.path.join(out, "january_barometer.json"), "w") as f:
            json.dump(rep, f, indent=1, default=float)
        print(json.dumps({k: {kk: v[kk] for kk in ("strategy_growth", "buy_hold_growth", "years_out")}
                          for k, v in rep.items()}, indent=1))
        return 0
    if a.smoke:
        panels = load_panels(any(s.panel == "etf" for s in specs), any(s.panel == "stocks" for s in specs))
        bad = 0
        for s in specs:
            try:
                p, _n = panels["etf1" if s.panel == "etf" else "era1"]
                backtest(p, s, max(s.era1_first, ERA1["first_decision"]), 20151231 if s.era1_first > 20140101 else 20140630)
                p, _n = panels["etf2" if s.panel == "etf" else "era2"]
                backtest(p, s, ERA2["first_decision"], 20240531)
                print(f"smoke ok   {s.id}")
            except Exception as exc:                                  # noqa: BLE001
                bad += 1
                print(f"smoke FAIL {s.id}: {type(exc).__name__}: {exc}")
        return 1 if bad else 0
    if not (a.prereg and a.ledger and a.out):
        ap.error("--prereg, --ledger and --out are required for a run")
    sha = _sha256(a.prereg)
    if PREREG_SHA256 and sha != PREREG_SHA256:
        raise SystemExit(f"pre-registration sha256 {sha} != registered {PREREG_SHA256}: refusing to run")
    pr = json.load(open(a.prereg))
    for s in specs:
        reg = pr.get("strategies", {}).get(s.id)
        if reg is None or reg.get("fingerprint") != s.fingerprint:
            raise SystemExit(f"{s.id} not pre-registered with fingerprint {s.fingerprint}: refusing to run")
    ledger_path = assert_outside_repo(a.ledger)
    out = assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    led = FM.load_ledger(ledger_path)
    todo = []
    for s in specs:
        if s.id in led:
            print(f"REFUSED: {s.id} already judged {led[s.id]['when']} (one look). Stored verdict: "
                  f"{led[s.id]['result']['verdict']}")
        else:
            todo.append(s)
    if not todo:
        return 0
    panels = load_panels(any(s.panel == "etf" for s in todo), any(s.panel == "stocks" for s in todo))
    lines = []
    for s in todo:
        res = run_one(s, panels)
        r1, r2 = res.pop("_r1"), res.pop("_r2")
        led[s.id] = {"when": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                     "fingerprint": s.fingerprint, "prereg_sha256": sha, "params": s.params, "result": res}
        FM.save_ledger(ledger_path, led)                              # written immediately: one look
        for name, r in (("era1", r1), ("era2", r2)):
            with open(os.path.join(out, f"{s.id}_{name}_equity.csv"), "w") as f:
                f.write("date,equity,gross\n")
                for d_, e_, g_ in zip(r["dates"], r["equity"], r["gross"]):
                    f.write(f"{d_},{e_:.2f},{g_:.4f}\n")
            with open(os.path.join(out, f"{s.id}_{name}_trades.csv"), "w") as f:
                f.write("date,symbol,side,value,cost,stale_exit\n")
                for t in r["trades"]:
                    f.write(",".join(str(x) for x in t) + "\n")
        e1, e2 = res["era1"], res["era2"]
        lines.append(f"{s.id}  ->  {res['verdict']}  {('failed: ' + ', '.join(res['failed'])) if res['failed'] else ''}")
        for k in ("h1", "h2", "whole"):
            lines.append(f"  era1 {k:5s} {e1[k]['from']}..{e1[k]['to']}  {FM._fmt(e1[k])}")
        lines.append(f"  era2 whole {e2['whole']['from']}..{e2['whole']['to']}  {FM._fmt(e2['whole'])}  [survivorship-biased]")
        lines.append(f"  trades {e1['trades']}/{e2['trades']}, rebalances {e1['rebalances']}/{e2['rebalances']}, "
                     f"stale exits {e1['stale_exits']}/{e2['stale_exits']}, mean gross exposure "
                     f"{e1['mean_gross']:.2f}/{e2['mean_gross']:.2f}")
        print("\n".join(lines[-6:]), flush=True)
        if a.trials:
            from .trials import TrialsLedger
            TrialsLedger(a.trials).append(_trial_rows(s, res, sha))
    with open(os.path.join(out, "summary.txt"), "a") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
