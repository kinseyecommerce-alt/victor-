"""
stratlab/kite_spreads.py — Zerodha Varsity's spread trades (Trading Systems module), tested as published with the
plain defaults fixed in stratlab/prereg/kite_ecosystem_v1.json before any run. Research only.

  r5_kite_vars_ptm2      pair trade, regression residual (chapter "trade-identification"): regress Y on X over the
                         trailing 200 sessions (both orientations; Y/X = the one with the lower error ratio
                         SE(intercept)/SE(residual)); pair qualifies if the ADF p-value of those residuals < 0.05;
                         z = residual / SE(residual). z <= -2 (and > -3): long pair (buy Y, sell beta*X); z >= +2 (and
                         < +3): short pair. The regression is FROZEN while the trade is open. Exit when z crosses 0,
                         stop at |z| >= 3. Decision at the daily close, both legs at the next open.
  r5_kite_vars_ptm1      pair trade, price-ratio density (chapter "the-pair-trade"): ratio A/B (A = alphabetically
                         first), density = normal CDF of the ratio vs its trailing 496-session mean / SD (recomputed
                         daily). Long ratio (buy A, sell B) at density 0.003..0.025; exit when back inside 2 SD
                         (>= 0.025); stop <= 0.003. Short mirror (0.975..0.997; exit <= 0.975; stop >= 0.997).
                         Rupee-neutral legs (Rs 1 lakh each).
  r5_kite_vars_calendar  calendar spread (chapter "calendar-spreads"): spread = (near - next) / near on stock futures
                         closes; trailing 200 observations (sessions where both contracts traded). Spread > mean + 1 SD:
                         sell near / buy next; < mean - 1 SD: buy near / sell next; exit when it returns to the
                         (rolling) mean; forced exit at the close of the session before the near expiry. Fills at the
                         NEXT session's EOD close of both contracts (next-month futures are illiquid at the open).

Pairs: same NSE industry (ind_nifty500list.csv "Industry", sha256 in the prereg), both stocks with listed stock
futures that day and, in 2012-23, both in that month's point-in-time top-100 universe. Prices: split/bonus-adjusted
cash closes / opens (the lab's stock-futures convention: futures simulated on the cash price with futures costs,
no basis / roll), Rs 1 lakh on the Y leg (PTM2: beta * qty_Y shares of X), 2 bps slippage per leg per side
(5 bps sensitivity), costs.round_trip_cost(..., "future") per leg. One open trade per pair / per stock.

Judged on 2012-01-01..2023-11-09 by protocol_oos (one look, oldera_oos_ledger.json) and on the 2023-26 data by
protocol.evaluate (DISCOVERY / FINAL split, FINAL once, bars3y final_ledger.json; era-2 names are the current
NIFTY-100: survivorship-biased).

    python -m stratlab.kite_spreads --era 2012-23 --prereg <PREREGISTERED_KITE.json> --out <dir>
    python -m stratlab.kite_spreads --era 2023-26 --prereg <PREREGISTERED_KITE.json> --out <dir> [--screen-only]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import factor_momentum as FM
from .kite_stats import adf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402

SCRATCH = os.environ.get("STRATLAB_SCRATCH",
                         "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad")
OLD_DIR = os.path.join(SCRATCH, "nse_old")
NSE_DIR = os.path.join(SCRATCH, "nse")
BARS3Y = os.path.join(SCRATCH, "bars3y")
INDUSTRY_CSV = os.path.join(SCRATCH, "kite", "ref", "ind_nifty500list.csv.nsearchives.nseindia.com")
WORK = os.path.join(SCRATCH, "kite", "_spread_cache")

NOTIONAL = 100_000.0
SLIP_BPS = 2.0
SENS_BPS = 5.0
FILL_MODEL = "ks1"
ERA1 = (20120101, 20231109)
ERA2_FIRST = 20231110

PTM2 = {"lookback": 200, "adf_p": 0.05, "entry_z": 2.0, "stop_z": 3.0, "exit_z": 0.0}
PTM1 = {"lookback": 496, "entry_lo": 0.003, "entry_hi": 0.025, "exit": 0.025, "stop": 0.003}
CAL = {"lookback": 200, "k_sd": 1.0}


# ══════════════════════════════════════════════════════════════════════════
# data
# ══════════════════════════════════════════════════════════════════════════
def industry_map() -> dict:
    """symbol -> NSE industry (today's NIFTY 500 list), old symbols mapped through nse_old/symbol_changes.csv."""
    df = pd.read_csv(INDUSTRY_CSV)
    ind = dict(zip(df["Symbol"], df["Industry"]))
    ch = pd.read_csv(os.path.join(OLD_DIR, "symbol_changes.csv"))
    new_of = dict(zip(ch["old_symbol"], ch["new_symbol"]))
    for old in list(new_of):
        cur, seen = old, set()
        while cur in new_of and cur not in seen:
            seen.add(cur)
            cur = new_of[cur]
        if cur in ind and old not in ind:
            ind[old] = ind[cur]
    for a, b in FM.SPLICE_MAP.items():            # ETERNAL <- ZOMATO, UNITDSPR <- MCDOWELL-N
        if a in ind:
            ind.setdefault(b, ind[a])
        if b in ind:
            ind.setdefault(a, ind[b])
    return ind


def futures_table() -> pd.DataFrame:
    """Stock futures rows 2012-01 .. 2026-11 (nse_old/fut.csv.gz then nse/fut.csv.gz from 2023-11-10)."""
    os.makedirs(WORK, exist_ok=True)
    pk = os.path.join(WORK, "futstk.pkl")
    if os.path.exists(pk):
        return pd.read_pickle(pk)
    cols = ["date", "symbol", "expiry", "instrument", "close", "contracts"]
    a = pd.read_csv(os.path.join(OLD_DIR, "fut.csv.gz"), usecols=cols)
    b = pd.read_csv(os.path.join(NSE_DIR, "fut.csv.gz"), usecols=cols)
    out = []
    for df, keep in ((a, lambda d: d < ERA2_FIRST), (b, lambda d: d >= ERA2_FIRST)):
        df = df[df.instrument == "FUTSTK"].copy()
        df["date"] = df["date"].str.replace("-", "").astype(np.int64)
        df["expiry"] = df["expiry"].str.replace("-", "").astype(np.int64)
        out.append(df[keep(df["date"])])
    f = pd.concat(out, ignore_index=True)
    rev = {b_: a_ for a_, b_ in FM.SPLICE_MAP.items()}
    f["symbol"] = f["symbol"].map(lambda s: rev.get(s, s))   # ZOMATO -> ETERNAL, MCDOWELL-N -> UNITDSPR
    f = f[["date", "symbol", "expiry", "close", "contracts"]].sort_values(["symbol", "date", "expiry"])
    f.to_pickle(pk)
    return f


def fno_matrix(panel: FM.Panel, fut: pd.DataFrame) -> np.ndarray:
    """(N, M) bool: the stock had a listed stock future on that session."""
    pos = {int(d): i for i, d in enumerate(panel.dates)}
    col = {s: j for j, s in enumerate(panel.symbols)}
    m = np.zeros((len(panel.dates), len(panel.symbols)), bool)
    g = fut[["date", "symbol"]].drop_duplicates()
    for d, s in zip(g["date"].to_numpy(), g["symbol"].to_numpy()):
        i, j = pos.get(int(d)), col.get(s)
        if i is not None and j is not None:
            m[i, j] = True
    return m


def load_era(era: str):
    """(panel, fno matrix, industry map). Panels from factor_momentum's loaders (era 1: point-in-time top-100,
    delisted kept; era 2: the bars3y names with history spliced from nse_old)."""
    os.makedirs(WORK, exist_ok=True)
    pk = os.path.join(WORK, f"panel_{era}.pkl")
    if os.path.exists(pk):
        with open(pk, "rb") as fh:
            p = pickle.load(fh)
    else:
        p1, _n1, eq = FM.load_oldera(OLD_DIR, cache=os.path.join(WORK, "_eqadj_cache.pkl"))
        if era == "2012-23":
            p = p1
        else:
            p, _n2, _notes = FM.load_era2(BARS3Y, eq, OLD_DIR, NSE_DIR)
        with open(pk, "wb") as fh:
            pickle.dump(p, fh)
    return p, fno_matrix(p, futures_table()), industry_map()


def eligible(panel: FM.Panel, fno: np.ndarray, t: int) -> np.ndarray:
    ok = fno[t] & panel.real[t]
    if panel.universe is not None:
        mem = panel.universe.get(int(panel.dates[t]) // 100, set())
        ok &= np.array([s in mem for s in panel.symbols])
    return ok


def pairs_of(panel: FM.Panel, ind: dict) -> list:
    by: dict = {}
    for j, s in enumerate(panel.symbols):
        if s in ind:
            by.setdefault(ind[s], []).append(j)
    out = []
    for g in by.values():
        g = sorted(g, key=lambda j: panel.symbols[j])
        out += [(g[a], g[b]) for a in range(len(g)) for b in range(a + 1, len(g))]
    return out


# ══════════════════════════════════════════════════════════════════════════
# shared accounting
# ══════════════════════════════════════════════════════════════════════════
def _leg_cost(qty, p_in, p_out, side):
    buy, sell = (p_in * qty, p_out * qty) if side > 0 else (p_out * qty, p_in * qty)
    return round_trip_cost(buy, sell, "future")


def _trade(sid, name, d_in, side, legs, d_out, reason, slip):
    """legs: [(qty, side, raw_in, raw_out)] -> lab trade tuple. Slippage is applied to every fill."""
    gross = cost = 0.0
    for q, s, a, b in legs:
        fin, fout = a * (1 + s * slip), b * (1 - s * slip)
        gross += s * (fout - fin) * q
        cost += _leg_cost(q, fin, fout, s)
    q0, s0, a0, b0 = legs[0]
    return (sid, name, int(d_in), int(side), "", round(a0, 4), int(d_out), "", round(b0, 4), int(q0),
            round(gross, 2), round(cost, 2), round(gross - cost, 2), reason)


def _rolling_ok(x, n):
    return pd.Series(np.isfinite(x).astype(float)).rolling(n).sum().to_numpy() == n


# ══════════════════════════════════════════════════════════════════════════
# PTM2 — regression residual pair trade
# ══════════════════════════════════════════════════════════════════════════
def _reg_stats(y, x, n):
    """Rolling OLS y = a + b x over n sessions: (a, b, se_resid, err_ratio) arrays (population moments)."""
    sy, sx = pd.Series(y), pd.Series(x)
    mx, my = sx.rolling(n).mean().to_numpy(), sy.rolling(n).mean().to_numpy()
    vx = sx.rolling(n).var(ddof=0).to_numpy()
    vy = sy.rolling(n).var(ddof=0).to_numpy()
    cxy = sx.rolling(n).cov(sy, ddof=0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        b = cxy / vx
        a = my - b * mx
        ssr = np.maximum(n * (vy - cxy * cxy / vx), 0.0)
        se = np.sqrt(ssr / (n - 2))
        er = np.sqrt(1.0 / n + mx * mx / (n * vx))           # SE(intercept) / SE(residual)
    return a, b, se, er


def sim_ptm2(panel, fno, ind, lo, hi, slip, sid="r5_kite_vars_ptm2", p=PTM2):
    n = p["lookback"]
    C, O = panel.close, panel.open
    cf = panel.cf
    D = panel.dates
    i_lo = int(np.searchsorted(D, lo, "left"))
    i_hi = int(np.searchsorted(D, hi, "right")) - 1
    elig = np.array([eligible(panel, fno, t) for t in range(len(D))]) if len(D) else np.zeros((0, 0), bool)
    trades = []
    for (i, j) in pairs_of(panel, ind):
        if not (elig[:, i] & elig[:, j]).any():
            continue
        xi, xj = C[:, i], C[:, j]
        both = _rolling_ok(xi, n) & _rolling_ok(xj, n)
        A1, B1, S1, E1 = _reg_stats(xj, xi, n)              # j on i
        A2, B2, S2, E2 = _reg_stats(xi, xj, n)              # i on j
        pos = None
        t = max(i_lo - 1, n - 1)
        while t < i_hi:
            if pos is None:
                if both[t] and elig[t, i] and elig[t, j]:
                    if E1[t] <= E2[t]:
                        yk, xk, a, b, se = j, i, A1[t], B1[t], S1[t]
                    else:
                        yk, xk, a, b, se = i, j, A2[t], B2[t], S2[t]
                    if b > 0 and se > 0:
                        z = (C[t, yk] - a - b * C[t, xk]) / se
                        if p["entry_z"] <= abs(z) < p["stop_z"]:
                            res = C[t - n + 1:t + 1, yk] - a - b * C[t - n + 1:t + 1, xk]
                            if adf(res)[1] < p["adf_p"]:
                                e = t + 1
                                if e <= i_hi and np.isfinite(O[e, yk]) and np.isfinite(O[e, xk]) \
                                        and D[e] >= lo:
                                    side = 1 if z < 0 else -1              # long pair = buy Y, sell X
                                    qy = max(1, int(NOTIONAL // O[e, yk]))
                                    qx = max(1, int(round(b * qy)))
                                    pos = (e, yk, xk, a, b, se, side, qy, qx)
                                    t = e
                                    continue
                t += 1
                continue
            e, yk, xk, a, b, se, side, qy, qx = pos
            z = (cf[t, yk] - a - b * cf[t, xk]) / se
            reason = None
            if side > 0 and z >= -p["exit_z"]:
                reason = "target"
            elif side < 0 and z <= p["exit_z"]:
                reason = "target"
            elif side > 0 and z <= -p["stop_z"]:
                reason = "stop"
            elif side < 0 and z >= p["stop_z"]:
                reason = "stop"
            if reason is not None and t + 1 <= i_hi:
                x_ = t + 1
                py = O[x_, yk] if np.isfinite(O[x_, yk]) else cf[t, yk]
                px = O[x_, xk] if np.isfinite(O[x_, xk]) else cf[t, xk]
                legs = [(qy, side, O[e, yk], py), (qx, -side, O[e, xk], px)]
                trades.append(_trade(sid, f"{panel.symbols[yk]}/{panel.symbols[xk]}", D[e], side, legs, D[x_],
                                     reason, slip))
                pos = None
                t = x_                                       # flat from the exit session (may re-enter at its close)
                continue
            if t + 1 > i_hi:
                break
            t += 1
        if pos is not None:                                   # closed at the period's last close
            e, yk, xk, a, b, se, side, qy, qx = pos
            legs = [(qy, side, O[e, yk], cf[i_hi, yk]), (qx, -side, O[e, xk], cf[i_hi, xk])]
            trades.append(_trade(sid, f"{panel.symbols[yk]}/{panel.symbols[xk]}", D[e], side, legs, D[i_hi],
                                 "period_end", slip))
    return trades


# ══════════════════════════════════════════════════════════════════════════
# PTM1 — ratio density pair trade
# ══════════════════════════════════════════════════════════════════════════
def _ncdf(z):
    from math import erf
    return 0.5 * (1.0 + np.vectorize(erf)(z / math.sqrt(2.0)))


def sim_ptm1(panel, fno, ind, lo, hi, slip, sid="r5_kite_vars_ptm1", p=PTM1):
    n = p["lookback"]
    C, O, cf, D = panel.close, panel.open, panel.cf, panel.dates
    i_lo = int(np.searchsorted(D, lo, "left"))
    i_hi = int(np.searchsorted(D, hi, "right")) - 1
    elig = np.array([eligible(panel, fno, t) for t in range(len(D))])
    trades = []
    for (i, j) in pairs_of(panel, ind):                       # i = A (alphabetically first), j = B
        if not (elig[:, i] & elig[:, j]).any():
            continue
        r = C[:, i] / C[:, j]
        ok = _rolling_ok(r, n)
        sr = pd.Series(r)
        mu, sd = sr.rolling(n).mean().to_numpy(), sr.rolling(n).std(ddof=1).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            dens = np.where(ok & (sd > 0), _ncdf((r - mu) / sd), np.nan)
        rf = cf[:, i] / cf[:, j]
        pos = None
        t = max(i_lo - 1, n - 1)
        while t < i_hi:
            if pos is None:
                dv = dens[t]
                if np.isfinite(dv) and elig[t, i] and elig[t, j]:
                    side = 1 if p["entry_lo"] <= dv <= p["entry_hi"] else (
                        -1 if (1 - p["entry_hi"]) <= dv <= (1 - p["entry_lo"]) else 0)
                    e = t + 1
                    if side and e <= i_hi and np.isfinite(O[e, i]) and np.isfinite(O[e, j]) and D[e] >= lo:
                        qa, qb = max(1, int(NOTIONAL // O[e, i])), max(1, int(NOTIONAL // O[e, j]))
                        pos = (e, side, qa, qb)
                        t = e
                        continue
                t += 1
                continue
            e, side, qa, qb = pos
            # density with today's rolling mean / SD (recomputed daily); a missing close is forward-filled
            dv = dens[t] if np.isfinite(dens[t]) else (
                float(_ncdf((rf[t] - mu[t]) / sd[t])) if np.isfinite(mu[t]) and sd[t] > 0 else np.nan)
            reason = None
            if np.isfinite(dv):
                if side > 0 and dv >= p["exit"]:
                    reason = "target"
                elif side < 0 and dv <= 1 - p["exit"]:
                    reason = "target"
                elif side > 0 and dv <= p["stop"]:
                    reason = "stop"
                elif side < 0 and dv >= 1 - p["stop"]:
                    reason = "stop"
            if reason is not None and t + 1 <= i_hi:
                x_ = t + 1
                pa = O[x_, i] if np.isfinite(O[x_, i]) else cf[t, i]
                pb = O[x_, j] if np.isfinite(O[x_, j]) else cf[t, j]
                legs = [(qa, side, O[e, i], pa), (qb, -side, O[e, j], pb)]
                trades.append(_trade(sid, f"{panel.symbols[i]}/{panel.symbols[j]}", D[e], side, legs, D[x_],
                                     reason, slip))
                pos = None
                t = x_
                continue
            if t + 1 > i_hi:
                break
            t += 1
        if pos is not None:
            e, side, qa, qb = pos
            legs = [(qa, side, O[e, i], cf[i_hi, i]), (qb, -side, O[e, j], cf[i_hi, j])]
            trades.append(_trade(sid, f"{panel.symbols[i]}/{panel.symbols[j]}", D[e], side, legs, D[i_hi],
                                 "period_end", slip))
    return trades


# ══════════════════════════════════════════════════════════════════════════
# calendar spread on stock futures
# ══════════════════════════════════════════════════════════════════════════
def calendar_frame(fut: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, session): near / next expiry (near = nearest expiry >= date), closes, traded flags."""
    f = fut[fut.expiry >= fut.date].sort_values(["symbol", "date", "expiry"]).copy()
    f["rk"] = f.groupby(["symbol", "date"]).cumcount()
    a = f[f.rk == 0].set_index(["symbol", "date"])
    b = f[f.rk == 1].set_index(["symbol", "date"])
    m = a[["expiry", "close", "contracts"]].join(b[["expiry", "close", "contracts"]], how="inner", lsuffix="1",
                                                 rsuffix="2")
    return m.reset_index()


def calendar_series(cf: pd.DataFrame, sym: str):
    g = cf[cf.symbol == sym].sort_values("date")
    if g.empty:
        return None
    return {"date": g.date.to_numpy(np.int64), "e1": g.expiry1.to_numpy(np.int64), "e2": g.expiry2.to_numpy(np.int64),
            "c1": g.close1.to_numpy(float), "c2": g.close2.to_numpy(float), "t1": (g.contracts1 > 0).to_numpy(),
            "t2": (g.contracts2 > 0).to_numpy()}


def _last_px(px: dict, D, f: int, x: int, e: int) -> float:
    """Close of contract e on session index x; if that row is missing (no bhavcopy row that day), the last close
    on or before x and after the fill session f (the fill price itself at worst)."""
    for k in range(x, f - 1, -1):
        r = px.get((int(D[k]), int(e)))
        if r is not None:
            return r[0]
    raise ValueError("no close for a held contract")


def sim_calendar(fut, symbols_by_day, lo, hi, slip, sid="r5_kite_vars_calendar_v2", p=CAL):
    """symbols_by_day(date) -> set of eligible symbols that session (universe gate on the decision day)."""
    n, k = p["lookback"], p["k_sd"]
    trades = []
    cfr = calendar_frame(fut)
    for sym in sorted(fut.symbol.unique()):
        cs = calendar_series(cfr, sym)
        if cs is None:
            continue
        g = fut[fut.symbol == sym]
        px = {(int(d), int(e)): (float(c), bool(v > 0)) for d, e, c, v in
              zip(g.date.to_numpy(), g.expiry.to_numpy(), g.close.to_numpy(), g.contracts.to_numpy())}
        D = cs["date"]
        valid = cs["t1"] & cs["t2"] & (D != cs["e1"]) & (cs["c1"] > 0) & (cs["c2"] > 0)
        sp = np.where(valid, (cs["c1"] - cs["c2"]) / np.where(cs["c1"] > 0, cs["c1"], 1.0), np.nan)
        vi = np.flatnonzero(valid)
        mu, sd = np.full(len(D), np.nan), np.full(len(D), np.nan)
        if len(vi) >= n:
            s = pd.Series(sp[vi])
            mu[vi] = s.rolling(n).mean().to_numpy()
            sd[vi] = s.rolling(n).std(ddof=1).to_numpy()
        i_hi = int(np.searchsorted(D, hi, "right")) - 1
        pos = None
        t = 0
        while t < i_hi:
            if pos is None:
                if valid[t] and np.isfinite(mu[t]) and sd[t] > 0 and D[t + 1] >= lo and sym in symbols_by_day(int(D[t])):
                    side = -1 if sp[t] > mu[t] + k * sd[t] else (1 if sp[t] < mu[t] - k * sd[t] else 0)
                    e1, e2 = int(cs["e1"][t]), int(cs["e2"][t])
                    f = t + 1
                    # fill at the next session's close; never on / after the last session before the near expiry
                    if side and f < i_hi and int(D[f + 1]) < e1:
                        a, b = px.get((int(D[f]), e1)), px.get((int(D[f]), e2))
                        if a and b and a[1] and b[1]:
                            q = max(1, int(NOTIONAL // a[0]))
                            pos = (f, side, e1, e2, a[0], b[0], q)
                            t = f
                            continue
                t += 1
                continue
            f, side, e1, e2, a0, b0, q = pos
            reason = None
            if t + 1 <= i_hi and int(D[t + 1]) >= e1:
                reason = "pre_expiry"                       # exit today: the next session is the near expiry
                x_ = t
            elif valid[t] and int(cs["e1"][t]) == e1 and np.isfinite(mu[t]) and \
                    ((side < 0 and sp[t] <= mu[t]) or (side > 0 and sp[t] >= mu[t])):
                reason, x_ = "target", t + 1
            if reason is not None and x_ <= i_hi:
                a1, b1 = _last_px(px, D, f, x_, e1), _last_px(px, D, f, x_, e2)
                legs = [(q, side, a0, a1), (q, -side, b0, b1)]   # long spread = buy near, sell next
                trades.append(_trade(sid, sym, D[f], side, legs, D[x_], reason, slip))
                pos = None
                t = x_
                continue
            if t + 1 > i_hi:
                break
            t += 1
        if pos is not None:
            f, side, e1, e2, a0, b0, q = pos
            a1, b1 = _last_px(px, D, f, i_hi, e1), _last_px(px, D, f, i_hi, e2)
            trades.append(_trade(sid, sym, D[f], side, [(q, side, a0, a1), (q, -side, b0, b1)], D[i_hi],
                                 "period_end", slip))
    return trades


# ══════════════════════════════════════════════════════════════════════════
# specs / protocol
# ══════════════════════════════════════════════════════════════════════════
_SIMS = {"r5_kite_vars_ptm2": (sim_ptm2, PTM2), "r5_kite_vars_ptm1": (sim_ptm1, PTM1),
         "r5_kite_vars_calendar_v2": (sim_calendar, CAL)}


@dataclass
class SpreadSpec:
    id: str
    desc: str
    params: dict = field(default_factory=dict)
    kind: str = "futures"
    family: str = "kite_varsity_spread"
    horizon: str = "positional"
    underlying: str | None = None

    @property
    def fingerprint(self) -> str:
        fn, _p = _SIMS[self.id]
        src = "".join(inspect.getsource(f) for f in (fn, _trade, _leg_cost, eligible, pairs_of, _reg_stats, adf,
                                                     calendar_frame, calendar_series, industry_map))
        if self.id == "r5_kite_vars_calendar_v2":
            src += inspect.getsource(_last_px)
        blob = repr((self.id, sorted(self.params.items()), NOTIONAL, FILL_MODEL, src))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


SPECS = [
    SpreadSpec("r5_kite_vars_ptm2", "Varsity PTM2 regression-residual pair trade (200d, ADF p<0.05, +-2 SD in, 0 out, 3 SD stop)",
               dict(PTM2)),
    SpreadSpec("r5_kite_vars_ptm1", "Varsity PTM1 ratio-density pair trade (496d, 0.003-0.025 / 0.975-0.997 in, back inside 2 SD out)",
               dict(PTM1)),
    SpreadSpec("r5_kite_vars_calendar_v2", "Varsity calendar spread on stock futures (near-next, 200 obs, mean+-1 SD in, mean out)",
               dict(CAL)),
]
BY_ID = {s.id: s for s in SPECS}


def cost_model() -> str:
    from .sim import COST_MODEL
    return f"{COST_MODEL}/{FILL_MODEL}"


def make_simulator(era: str, slip_bps: float):
    cache: dict = {}

    def simulate(specs, lo, hi):
        if "data" not in cache:
            cache["data"] = load_era(era)
            cache["fut"] = futures_table()
        panel, fno, ind = cache["data"]
        fut = cache["fut"]
        out = {}
        for sp in specs:
            try:
                fn, prm = _SIMS[sp.id]
                if sp.id == "r5_kite_vars_calendar_v2":
                    names = set(panel.symbols)
                    pos = {int(d): i for i, d in enumerate(panel.dates)}

                    def by_day(d, _p=panel, _pos=pos, _f=fno):
                        i = _pos.get(int(d))
                        if i is None:
                            return set()
                        ok = eligible(_p, _f, i)
                        return {s for s, o in zip(_p.symbols, ok) if o}
                    f = fut[fut.symbol.isin(names)]
                    tr = fn(f, by_day, lo, hi, slip_bps / 1e4, p=prm)
                else:
                    tr = fn(panel, fno, ind, lo, hi, slip_bps / 1e4, p=prm)
                out[sp.id] = {"trades": sorted(tr, key=lambda x: (x[2], x[1])), "skipped": 0, "lookahead": None,
                              "error": None}
            except Exception as exc:
                import traceback
                traceback.print_exc()
                out[sp.id] = {"trades": [], "skipped": 0, "lookahead": None, "error": repr(exc)[:300]}
        return out
    return simulate


def _write_trades(out, sid, trades):
    from .sim import TRADE_FIELDS
    os.makedirs(os.path.join(out, "trades"), exist_ok=True)
    with open(os.path.join(out, "trades", sid + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRADE_FIELDS)
        w.writerows(trades)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.kite_spreads")
    ap.add_argument("--era", required=True, choices=("2012-23", "2023-26"))
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--screen-only", action="store_true")
    ap.add_argument("--amend", default="", help="pre-registration amendment(s), comma-separated: their strategies are "
                                                "added (never replace an id of the original)")
    a = ap.parse_args(argv)
    from .data import assert_outside_repo
    out = assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    prereg = json.load(open(a.prereg))
    for am in [x for x in a.amend.split(",") if x]:
        for k, v in json.load(open(am)).get("strategies", {}).items():
            if k in prereg["strategies"]:
                raise SystemExit(f"amendment {am} redefines {k}: refused")
            prereg["strategies"][k] = v
    specs = [BY_ID[x] for x in (a.ids.split(",") if a.ids else BY_ID)]
    for sp in specs:
        reg = prereg.get("strategies", {}).get(sp.id)
        if reg is None or reg.get("fingerprint") != sp.fingerprint:
            raise SystemExit(f"{sp.id}: not pre-registered with this fingerprint ({sp.fingerprint}): refusing to run")
    if a.era == "2012-23":
        from . import protocol_oos as PO
        res = PO.evaluate(specs, make_simulator(a.era, SLIP_BPS), make_simulator(a.era, SENS_BPS),
                          cost_model=cost_model(), prereg=prereg, dataset=OLD_DIR)
        print(PO._write(out, res, f"kite_spreads era 2012-23, cost model {cost_model()}, prereg {a.prereg}"))
        return 0
    from . import protocol as P
    from . import data as D
    from . import run as R
    panel = load_era(a.era)[0]
    cal = [int(d) for d in panel.dates if d >= ERA2_FIRST]
    cache = D.assert_outside_repo(D.default_cache_dir(BARS3Y))
    results, trades = P.evaluate(specs, cal, make_simulator(a.era, SLIP_BPS), n_symbols=len(panel.symbols),
                                 ledger_path=os.path.join(cache, "final_ledger.json"),
                                 dataset=os.path.realpath(BARS3Y) + "#kite_spreads", screen_only=a.screen_only,
                                 cost_model=cost_model())
    split = results[0]["split"]
    sens = make_simulator(a.era, SENS_BPS)(specs, *split.discovery)
    for r in results:
        sc = P.screen(sens[r["id"]]["trades"], split, "positional")
        r["sens"] = {"disc": sc["disc"], "passed": sc["passed"]}
    meta = {"kind": "futures", "data": os.path.realpath(BARS3Y), "n_symbols": len(panel.symbols),
            "n_sessions": len(cal), "first": cal[0], "last": cal[-1], "jobs": 1, "prep_secs": 0.0,
            "quality_notes": [], "slippage_bps": SLIP_BPS, "index_slippage_bps": "", "sens_bps": SENS_BPS,
            "sens_index_bps": 0, "screen_only": a.screen_only, "cost_model": cost_model()}
    print(R.write_outputs(out, results, trades, {}, meta, [], []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
