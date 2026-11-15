"""
stratlab/r7_portfolio.py — round 7 portfolio items (second GitHub shortlist). Research only; one look per strategy.
Pre-registration: stratlab/prereg/github_v2.json (byte-identical to <scratchpad>/PREREGISTERED_GITHUB2.json); a
strategy runs only if listed there with the same fingerprint, and the ledger refuses a second run of an id.

  r7_gh2_volmanaged_nifty  Moreira & Muir (2017) volatility-managed NIFTY (coolgan vol-timing.R, bugs fixed):
      at each month's last close, RV = sum of the month's squared daily NIFTY returns; next month's weight
      w = min(CAP, c / RV); c = EXPANDING-window constant making the uncapped managed series r_d / RV_prev as volatile
      as NIFTY itself over all history to date (the source divides by the full-sample std = look-ahead, and its
      monthly weight 1/vol[1] uses the 22-day variance ending ON the month's first day, i.e. that day's own return).
      Index-futures exposure (spot index, cash at 0 %, futures statutory costs + 1 bps per side, executed at the next
      open). Judged against buy-and-hold NIFTY with the same costs: PASS needs a higher Sharpe in era-1 H1, era-1 H2
      and era 2 AND a shallower max drawdown in era 1 and era 2. CAGR reported.
  r7_gh2_carry_xs  cross-sectional stock-futures carry, weekly: carry = (front settle / second settle - 1) x 365 /
      days between their expiries (front = earliest expiry after the decision day, both contracts traded that day);
      long the top quintile / short the bottom quintile (+-0.5 gross), decided at the last session of each ISO week,
      executed at the next open. P&L on the FUTURES contracts themselves (the lab's cash-price convention would
      delete the roll yield being tested): held contract = earliest expiry after the day, rolled at the close of
      the eve of expiry (two legs of costs); futures statutory costs + 10 bps per side per leg; a contract return
      more than 10 % away from the adjusted cash return that day (corporate-action adjustment) is replaced by it.
      factor_momentum gates (as round 6's long / short books).
  r7_gh2_sector_rot  Alvarez ETF sector rotation (TuringTrader Alvarez_EtfSectorRotation.cs) on 11 NSE sector
      indices (marketcalls sector-rotation-map list minus NIFTYPVTBANK, which starts 2016): at the close of each
      month's FIRST session rank by 252- and 126-session momentum, score = 1.001 x rank252 + rank126, hold the best 2
      at 50 % each if its close > its close 252 sessions earlier, else that half in cash (0 %, standing in for TLT);
      executed at the next open; index levels stand in for sector ETFs (CNC costs + 10 bps). The porter warns his
      coded version (top 2) does NOT reproduce Alvarez's published results and that choosing the 2 WORST would; the
      described / coded version is tested, the reversal one is not. factor_momentum gates.

    python -m stratlab.r7_portfolio --list
    python -m stratlab.r7_portfolio --smoke [--ids a,b]             # errors only (no metric printed)
    python -m stratlab.r7_portfolio --prereg <PREREGISTERED_GITHUB2.json> --ledger <ledger.json> --out <dir> \
        [--ids a,b] [--trials <trials_v2.csv>]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

from . import factor_momentum as FM
from . import r6_portfolio as RP

try:
    from costs import round_trip_cost
except ImportError:                     # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from costs import round_trip_cost

PREREG_SHA256 = "1745b3e24092cae2d9ea7009ee6380c322a5fcc67a3f35617a8e6e840e2b7b92"                      # filled in after the pre-registration is written (checked by the CLI)
_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
OLD_DIR = os.path.join(_SCRATCH, "nse_old")
NSE_DIR = os.path.join(_SCRATCH, "nse")
BARS3Y = os.path.join(_SCRATCH, "bars3y")
IND2 = os.path.join(_SCRATCH, "github2_r7", "ind_close_all")
CAPITAL = FM.CAPITAL
ERA1 = dict(FM.ERA1)
ERA2 = dict(FM.ERA2)

VOL = {"cap": 1.5, "min_months": 12, "index_slip_bps": 1.0, "rebalance": "month-end close -> next open"}
CARRY = {"quantile": 0.2, "gross_side": 0.5, "slip_bps": 10.0, "ca_tol": 0.10, "schedule": "weekly"}
SECTORS = {"Auto": ("CNX Auto", "Nifty Auto"), "IT": ("CNX IT", "Nifty IT"), "Pharma": ("CNX Pharma", "Nifty Pharma"),
           "Energy": ("CNX Energy", "Nifty Energy"), "FMCG": ("CNX FMCG", "Nifty FMCG"),
           "Metal": ("CNX Metal", "Nifty Metal"), "Realty": ("CNX Realty", "Nifty Realty"),
           "PSUBank": ("CNX PSU Bank", "Nifty PSU Bank"), "Media": ("CNX Media", "Nifty Media"),
           "Infra": ("CNX Infrastructure", "Nifty Infrastructure"),
           "Commodities": ("CNX Commodities", "Nifty Commodities")}
SECT = {"rank1": 252, "rank2": 126, "top": 2, "weight": 0.5, "abs_mom": 252, "tie": 1.001,
        "sectors": sorted(SECTORS)}


# ══════════════════════════════════════════════════════════════════════════
# shared: costs and metrics
# ══════════════════════════════════════════════════════════════════════════
def fut_cost(value: float, side: str, slip_bps: float) -> tuple:
    """(statutory, slippage) rupees for one futures buy / sell of `value`."""
    if value <= 0:
        return 0.0, 0.0
    stat = round_trip_cost(value, 0.0, "future") if side == "BUY" else round_trip_cost(0.0, value, "future")
    return float(stat), value * slip_bps / 1e4


def _nifty_map(dates, nifty) -> dict:
    return dict(zip([int(x) for x in dates], [float(x) for x in nifty]))


# ══════════════════════════════════════════════════════════════════════════
# 5. volatility-managed NIFTY
# ══════════════════════════════════════════════════════════════════════════
def vm_weights(dates, close, prm: dict = VOL) -> dict:
    """{decision index t: weight for the next month} at every month-end t with >= min_months monthly RVs.
    Reads close[:t+1] only (expanding c)."""
    dates = np.asarray(dates, np.int64)
    c = np.asarray(close, float)
    r = np.full(len(c), np.nan)
    r[1:] = c[1:] / c[:-1] - 1.0
    ym = dates // 100
    ends = [i for i in range(len(dates) - 1) if ym[i] != ym[i + 1]]
    out = {}
    rv_by_month = {}
    first = None
    for t in ends:
        m = ym[t]
        idx = np.flatnonzero(ym[: t + 1] == m)
        rr = r[idx]
        rr = rr[np.isfinite(rr)]
        if len(rr) < 10:
            continue
        rv_by_month[int(m)] = float(np.sum(rr ** 2))
        if first is None:
            first = int(idx[0])
        months = sorted(rv_by_month)
        if len(months) < prm["min_months"] + 1:
            continue
        # managed (uncapped, c = 1) daily returns r_d / RV(previous month) for every month after the first one
        num, den = [], []
        for a, b in zip(months[:-1], months[1:]):
            ii = np.flatnonzero(ym[: t + 1] == b)
            x = r[ii]
            x = x[np.isfinite(x)]
            num.append(x)
            den.append(x / rv_by_month[a])
        raw = np.concatenate(num)
        man = np.concatenate(den)
        if len(raw) < 2 or not np.std(man) > 0:
            continue
        cc = float(np.std(raw, ddof=1) / np.std(man, ddof=1))
        out[t] = float(min(prm["cap"], cc / rv_by_month[int(m)]))
    return out


def vm_backtest(dates, opn, close, weights: dict, first_decision: int, end: int | None, slip_bps: float,
                capital: float = CAPITAL) -> dict:
    """Index-futures exposure that changes only at the next open after a decision (weights: {t: w}); units held
    between rebalances (the weight drifts); cash at 0 %; futures statutory costs + slip_bps per side."""
    dates = np.asarray(dates, np.int64)
    last = len(dates) - 1 if end is None else int(np.searchsorted(dates, end, side="right")) - 1
    decs = [t for t in sorted(weights) if dates[t] >= first_decision and t < last]
    if not decs:
        raise ValueError("no decision in range")
    t0 = decs[0]
    units, cash = 0.0, capital
    eq_d, eq, gross, log = [int(dates[t0])], [capital], [0.0], []
    pending = weights.get(t0)
    decset = set(decs)
    for i in range(t0 + 1, last + 1):
        if pending is not None:
            px = float(opn[i]) if np.isfinite(opn[i]) and opn[i] > 0 else float(close[i - 1])
            e = cash + units * px
            tgt = pending * e / px
            d = tgt - units
            if abs(d) > 1e-12:
                v = abs(d) * px
                stat, slip = fut_cost(v, "BUY" if d > 0 else "SELL", slip_bps)
                cash -= d * px + stat + slip
                units = tgt
                log.append((int(dates[i]), "NIFTY", "BUY" if d > 0 else "SELL", round(v, 2), round(stat + slip, 2),
                            False))
            pending = None
        e = cash + units * float(close[i])
        if e <= 0:
            raise RuntimeError(f"equity <= 0 on {dates[i]}")
        eq_d.append(int(dates[i]))
        eq.append(e)
        gross.append(abs(units * float(close[i])) / e)
        if i in decset:
            pending = weights[i]
    return {"dates": np.array(eq_d), "equity": np.array(eq), "gross": np.array(gross), "trades": log,
            "first_decision": int(dates[t0]), "last": int(dates[last])}


def load_nifty_daily():
    """(era1 dates, open, close), (era2 dates, open, close): era 1 = nse_old official NIFTY 2012-01..2023-11-09;
    era 2 = the same history to 2023-11-09 + the lab's bars3y NIFTY daily bars (09:15 open, 15:29 close) after."""
    import pandas as pd
    from . import data as D
    idx = pd.read_csv(os.path.join(OLD_DIR, "index_daily.csv.gz"), usecols=["date", "index", "open", "close"])
    nf = idx[idx["index"] == "NIFTY"].sort_values("date")
    d1, o1, c1 = nf.date.to_numpy(np.int64), nf.open.to_numpy(float), nf.close.to_numpy(float)
    aux = D.load_daily_aux(D.prepare(BARS3Y)["cache_dir"], "_NIFTY50_day")
    from .r7_tps import calendar_3y
    cal = set(calendar_3y())                    # the lab's 2023-26 sessions (special / short sessions and days after
    m = (aux["date"] >= ERA2["junction"]) & np.isin(aux["date"], list(cal))   # 2026-11-09 excluded)
    k = d1 <= ERA2["splice_last_old"]
    d2 = np.concatenate([d1[k], aux["date"][m].astype(np.int64)])
    o2 = np.concatenate([o1[k], aux["o"][m].astype(float)])
    c2 = np.concatenate([c1[k], aux["c"][m].astype(float)])
    return (d1, o1, c1), (d2, o2, c2)


def run_volmanaged(spec) -> dict:
    (d1, o1, c1), (d2, o2, c2) = load_nifty_daily()
    prm = spec.params
    res = {}
    for era, (d, o, c), first, end in (("era1", (d1, o1, c1), ERA1["first_decision"], ERA1["end"]),
                                       ("era2", (d2, o2, c2), ERA2["first_decision"], None)):
        w = vm_weights(d, c, prm)
        bh = {t: 1.0 for t in w}
        r_vm = vm_backtest(d, o, c, w, first, end, prm["index_slip_bps"])
        first_bh = r_vm["first_decision"]
        r_bh = vm_backtest(d, o, c, {min(t for t in bh if d[t] >= first_bh): 1.0}, first_bh, end,
                           prm["index_slip_bps"])
        nm = _nifty_map(d, c)
        segs = {}
        lo, hi = r_vm["dates"][0], r_vm["dates"][-1]
        cuts = [("whole", lo, hi)]
        if era == "era1":
            h1_end = int(r_vm["dates"][r_vm["dates"] <= ERA1["h1_end"]][-1])
            cuts = [("h1", lo, h1_end), ("h2", h1_end, hi), ("whole", lo, hi)]
        for name, a, b in cuts:
            segs[name] = {"managed": FM.segment_metrics(r_vm, nm, a, b), "buy_hold": FM.segment_metrics(r_bh, nm, a, b)}
        ws = [w[t] for t in w if lo <= d[t] <= hi]
        segs["mean_weight"] = float(np.mean(ws)) if ws else float("nan")
        segs["max_weight_share_at_cap"] = float(np.mean([x >= prm["cap"] - 1e-12 for x in ws])) if ws else float("nan")
        segs["_r"] = (r_vm, r_bh)
        res[era] = segs
    e1, e2 = res["era1"], res["era2"]
    checks = {
        "era1 H1 Sharpe > buy-and-hold": e1["h1"]["managed"]["sharpe"] > e1["h1"]["buy_hold"]["sharpe"],
        "era1 H2 Sharpe > buy-and-hold": e1["h2"]["managed"]["sharpe"] > e1["h2"]["buy_hold"]["sharpe"],
        "era2 Sharpe > buy-and-hold": e2["whole"]["managed"]["sharpe"] > e2["whole"]["buy_hold"]["sharpe"],
        "era1 maxDD shallower than buy-and-hold": e1["whole"]["managed"]["max_dd"] > e1["whole"]["buy_hold"]["max_dd"],
        "era2 maxDD shallower than buy-and-hold": e2["whole"]["managed"]["max_dd"] > e2["whole"]["buy_hold"]["max_dd"],
    }
    failed = [k for k, v in checks.items() if not v]
    return {"era1": e1, "era2": e2, "checks": {k: bool(v) for k, v in checks.items()},
            "verdict": "PASS" if not failed else "FAIL", "failed": failed}


# ══════════════════════════════════════════════════════════════════════════
# 10. cross-sectional stock-futures carry (weekly)
# ══════════════════════════════════════════════════════════════════════════
def _ymd(s) -> int:
    return int(str(s).replace("-", ""))


def futures_frame(path: str, symbols: set, rename: dict | None = None):
    """FUTSTK rows of fut.csv.gz for `symbols` (rename: old symbol -> panel symbol) as a DataFrame with int dates."""
    import pandas as pd
    f = pd.read_csv(path, usecols=["date", "symbol", "expiry", "instrument", "open", "settle", "contracts"])
    f = f[f.instrument == "FUTSTK"].copy()
    if rename:
        f["symbol"] = f.symbol.map(lambda s: rename.get(s, s))
    f = f[f.symbol.isin(symbols)]
    f["date"] = f.date.map(_ymd).astype(np.int64)
    f["expiry"] = f.expiry.map(_ymd).astype(np.int64)
    return f.drop(columns=["instrument"])


def carry_arrays(f, dates, symbols) -> dict:
    """Per (session, symbol): front = earliest expiry strictly after the session, second = the next one. Arrays of
    their expiry, settle, open and traded contracts (NaN / 0 where absent)."""
    pos = {int(d): i for i, d in enumerate(dates)}
    col = {s: j for j, s in enumerate(symbols)}
    N, M = len(dates), len(symbols)
    A = {k: np.full((N, M), np.nan) for k in ("f_exp", "f_set", "f_open", "f_con", "s_exp", "s_set", "s_con")}
    g = f[f.expiry > f.date].sort_values(["date", "symbol", "expiry"])
    g = g.assign(rk=g.groupby(["date", "symbol"]).cumcount())
    for rk, pre in ((0, "f"), (1, "s")):
        h = g[g.rk == rk]
        ii = h.date.map(pos)
        jj = h.symbol.map(col)
        ok = ii.notna() & jj.notna()
        ii, jj, h = ii[ok].astype(int).to_numpy(), jj[ok].astype(int).to_numpy(), h[ok]
        A[pre + "_exp"][ii, jj] = h.expiry.to_numpy(float)
        A[pre + "_set"][ii, jj] = h.settle.to_numpy(float)
        A[pre + "_con"][ii, jj] = h.contracts.to_numpy(float)
        if pre == "f":
            A["f_open"][ii, jj] = h.open.to_numpy(float)
    return A


def _days(a: int, b: int) -> int:
    da = _dt.date(a // 10000, a // 100 % 100, a % 100)
    db = _dt.date(b // 10000, b // 100 % 100, b % 100)
    return (db - da).days


def carry_score(A: dict, t: int) -> np.ndarray:
    """Annualised front-over-second carry at session t (NaN unless both contracts traded and settled > 0)."""
    fs, ss = A["f_set"][t], A["s_set"][t]
    ok = (A["f_con"][t] > 0) & (A["s_con"][t] > 0) & (fs > 0) & (ss > 0) & np.isfinite(A["f_exp"][t]) \
        & np.isfinite(A["s_exp"][t])
    out = np.full(len(fs), np.nan)
    for j in np.flatnonzero(ok):
        dd = _days(int(A["f_exp"][t, j]), int(A["s_exp"][t, j]))
        if dd > 0:
            out[j] = (fs[j] / ss[j] - 1.0) * 365.0 / dd
    return out


def contract_returns(A: dict, cash_open, cash_close, tol: float) -> tuple:
    """(gap_r, day_r) per (session, symbol) of the contract held over (close t-1, close t] = the front of session t:
    gap = open / previous settle - 1, day = settle / open - 1, from the held contract's settles (its previous-day
    settle is the front's or, across an expiry, the second's). Where the contract return is unavailable or more than
    `tol` away from the adjusted cash return (corporate-action adjustment of the contract), the cash returns are
    used. roll[t, j] = True when the held contract changes at the close of t."""
    N, M = A["f_set"].shape
    gap = np.full((N, M), np.nan)
    day = np.full((N, M), np.nan)
    for t in range(1, N):
        same = A["f_exp"][t - 1] == A["f_exp"][t]
        prev = np.where(same, A["f_set"][t - 1], np.where(A["s_exp"][t - 1] == A["f_exp"][t], A["s_set"][t - 1], np.nan))
        st, op = A["f_set"][t], A["f_open"][t]
        with np.errstate(divide="ignore", invalid="ignore"):
            r_f = st / prev - 1.0
            g_f = op / prev - 1.0
            d_f = st / op - 1.0
            cg = cash_open[t] / cash_close[t - 1] - 1.0
            cd = cash_close[t] / cash_open[t] - 1.0
            rc = cash_close[t] / cash_close[t - 1] - 1.0
        good = np.isfinite(r_f) & (prev > 0) & (st > 0) & (np.abs(r_f - np.where(np.isfinite(rc), rc, r_f)) <= tol)
        good_open = good & np.isfinite(op) & (op > 0)
        gap[t] = np.where(good_open, g_f, np.where(good, r_f, cg))
        day[t] = np.where(good_open, d_f, np.where(good, 0.0, cd))
    roll = np.zeros((N, M), bool)
    roll[:-1] = np.isfinite(A["f_exp"][:-1]) & np.isfinite(A["f_exp"][1:]) & (A["f_exp"][:-1] != A["f_exp"][1:])
    return np.nan_to_num(gap), np.nan_to_num(day), roll


def carry_weights(p, A, t: int, exec_month: int, prm: dict) -> dict:
    sc = carry_score(A, t)
    ok = p.real[t] & np.isfinite(sc)
    if p.universe is not None:
        mem = p.universe.get(int(exec_month), set())
        ok &= np.array([s in mem for s in p.symbols])
    score = {int(j): float(sc[j]) for j in np.flatnonzero(ok)}
    return RP._ls(p, score, int(len(score) * prm["quantile"]), prm["gross_side"])


def carry_backtest(p, A, gap, day, roll, prm: dict, first_decision: int, end: int | None = None,
                   capital: float = CAPITAL) -> dict:
    """Signed rupee exposures in the held contracts; marks at settles; weekly decisions, next-open execution; rolls
    at the close of the expiry eve cost two legs. Output layout of factor_momentum.backtest."""
    dates = p.dates
    last = len(dates) - 1 if end is None else int(np.searchsorted(dates, end, side="right")) - 1
    decs = [t for t in RP.schedule("weekly", dates[: last + 1]) if dates[t] >= first_decision]
    t0 = decs[0]
    decset = set(decs)
    M = len(p.symbols)
    E = np.zeros(M)
    eq_val = capital
    log = []
    eq_d, eq, gross = [int(dates[t0])], [capital], [0.0]
    pending = carry_weights(p, A, t0, int(dates[t0 + 1]) // 100, prm)
    slip = prm["slip_bps"]
    for i in range(t0 + 1, last + 1):
        eq_val += float(np.sum(E * gap[i]))
        E = E * (1.0 + gap[i])
        if pending is not None:
            stale = ~(np.isfinite(A["f_open"][i]) & (A["f_open"][i] > 0))
            for j in sorted(set(int(x) for x in np.flatnonzero(E)) | set(pending)):
                tgt = float(pending.get(j, 0.0)) * eq_val
                d = tgt - E[j]
                if abs(d) < 1e-9:
                    continue
                if tgt != 0 and E[j] != 0 and np.sign(tgt) == np.sign(E[j]) and abs(d) < RP.BAND * eq_val:
                    continue
                if stale[j] and tgt != 0:
                    continue
                stat, sl = fut_cost(abs(d), "BUY" if d > 0 else "SELL", slip)
                eq_val -= stat + sl
                E[j] = tgt
                log.append((int(dates[i]), p.symbols[j], "BUY" if d > 0 else "SELL", round(abs(d), 2),
                            round(stat + sl, 2), bool(stale[j])))
            pending = None
        eq_val += float(np.sum(E * day[i]))
        E = E * (1.0 + day[i])
        if i < last:
            for j in np.flatnonzero((E != 0) & roll[i]):
                v = abs(E[j])
                c1 = fut_cost(v, "SELL" if E[j] > 0 else "BUY", slip)
                c2 = fut_cost(v, "BUY" if E[j] > 0 else "SELL", slip)
                cost = sum(c1) + sum(c2)
                eq_val -= cost
                log.append((int(dates[i]), p.symbols[j], "ROLL", round(2 * v, 2), round(cost, 2), False))
        if eq_val <= 0:
            raise RuntimeError(f"equity <= 0 on {dates[i]}")
        eq_d.append(int(dates[i]))
        eq.append(eq_val)
        gross.append(float(np.abs(E).sum() / eq_val))
        if i in decset and i < last:
            pending = carry_weights(p, A, i, int(dates[i + 1]) // 100, prm)
    return {"dates": np.array(eq_d), "equity": np.array(eq), "gross": np.array(gross), "trades": log,
            "first_decision": int(dates[t0]), "last": int(dates[last]), "stale_exits": 0,
            "rebalances": len(decs)}


def carry_panels():
    p1, n1, eq = RP.load_era1()
    p2, n2 = RP.load_era2(eq)
    f1 = futures_frame(os.path.join(OLD_DIR, "fut.csv.gz"), set(p1.symbols))
    # era 2: the NSE F&O bhavcopy 2023-11-01.. (carry needs no lookback: decisions start 2023-11-30); old names of
    # renamed companies are mapped to the panel's current names, as in the cash splice
    inv = {v: k for k, v in FM.SPLICE_MAP.items()}
    f2 = futures_frame(os.path.join(NSE_DIR, "fut.csv.gz"), set(p2.symbols), rename=inv)
    return (p1, n1, f1), (p2, n2, f2)


def run_carry(spec, panels=None) -> dict:
    (p1, n1, f1), (p2, n2, f2) = panels or carry_panels()
    prm = spec.params
    out = {}
    for era, p, nmap, f, first, end in (("era1", p1, n1, f1, ERA1["first_decision"], ERA1["end"]),
                                        ("era2", p2, n2, f2, ERA2["first_decision"], None)):
        A = carry_arrays(f, p.dates, p.symbols)
        gap, day, roll = contract_returns(A, p.open, p.close, prm["ca_tol"])
        r = carry_backtest(p, A, gap, day, roll, prm, first, end)
        out[era] = r
    return _judge_two_eras(out["era1"], out["era2"], p1, n1, n2)


def _judge_two_eras(r1, r2, p1, n1, n2) -> dict:
    h1_end = int(r1["dates"][r1["dates"] <= ERA1["h1_end"]][-1])
    era1 = {"h1": FM.segment_metrics(r1, n1, r1["dates"][0], h1_end),
            "h2": FM.segment_metrics(r1, n1, h1_end, r1["dates"][-1]),
            "whole": FM.segment_metrics(r1, n1, r1["dates"][0], r1["dates"][-1]),
            "trades": len(r1["trades"]), "mean_gross": float(np.nanmean(r1["gross"][1:])),
            "sr_daily": RP._daily_sr(r1, r1["dates"][0], r1["dates"][-1])}
    era2 = {"whole": FM.segment_metrics(r2, n2, r2["dates"][0], r2["dates"][-1]), "trades": len(r2["trades"]),
            "mean_gross": float(np.nanmean(r2["gross"][1:])), "sr_daily": RP._daily_sr(r2, r2["dates"][0], r2["dates"][-1])}
    return {"era1": era1, "era2": era2, **FM.judge(era1, era2), "_r1": r1, "_r2": r2}


# ══════════════════════════════════════════════════════════════════════════
# 11. sector rotation (Alvarez via TuringTrader)
# ══════════════════════════════════════════════════════════════════════════
def _read_ind2(path_glob: str):
    import pandas as pd
    rows = []
    for p in sorted(glob.glob(path_glob)):
        x = pd.read_csv(p)
        x.columns = [c.strip() for c in x.columns]
        x = x.rename(columns={"Index Name": "index", "Index Date": "d", "Open Index Value": "open",
                              "Closing Index Value": "close"})
        x["date"] = x.d.map(lambda s: int(_dt.datetime.strptime(str(s).strip(), "%d-%m-%Y").strftime("%Y%m%d")))
        for k in ("open", "close"):                      # "-" for an index without a value that day -> NaN
            x[k] = pd.to_numeric(x[k], errors="coerce")
        rows.append(x[["date", "index", "open", "close"]])
    return pd.concat(rows, ignore_index=True) if rows else None


def sector_panels():
    """Era 1: nse_old ind_close_all 2012-02-21..2023-11-09 (CNX -> Nifty renames of 2015-11-09 spliced: the same
    index levels). Era 2: that history + the round-7 ind_close_all download (2023-11-01..2026-11) on the bars3y NIFTY
    calendar after 2023-11-09. Opens: the official open (NaN -> the engine's stale rule)."""
    import pandas as pd
    x = pd.read_csv(os.path.join(OLD_DIR, "index_all_daily.csv.gz"), usecols=["date", "index", "open", "close"])
    want = {n: k for k, (a, b) in SECTORS.items() for n in (a, b)}
    nifty_names = {"S&P CNX Nifty", "CNX Nifty", "Nifty 50"}
    y = _read_ind2(os.path.join(IND2, "ind_close_all_*.csv"))
    out = {}
    (d1, _o1, c1), (d2, _o2, c2) = load_nifty_daily()
    for era in ("era1", "era2"):
        src = x if era == "era1" else pd.concat([x[x.date <= ERA2["splice_last_old"]],
                                                 y[y.date >= ERA2["junction"]]], ignore_index=True)
        s = src[src["index"].isin(want)].assign(sec=lambda z: z["index"].map(want))
        dates, nifty = (d1, c1) if era == "era1" else (d2, c2)
        dates = np.asarray(dates, np.int64)
        keep = dates >= int(s.date.min())
        dates, nifty = dates[keep], nifty[keep]
        syms = sorted(SECTORS)
        cl = s.pivot_table(index="date", columns="sec", values="close", aggfunc="last").reindex(index=dates, columns=syms)
        op = s.pivot_table(index="date", columns="sec", values="open", aggfunc="last").reindex(index=dates, columns=syms)
        op = op.where(op > 0)
        p = RP.XPanel(dates, syms, cl.to_numpy(float), op.to_numpy(float), nifty, universe=None,
                      label=f"{era} NSE sector indices (11)")
        out[era] = (p, _nifty_map(dates, nifty))
    return out


def w_sector(p, t, ctx, prm):
    """Decision only at the close of a month's FIRST session (the port trades when SimTime[0].Month !=
    SimTime[1].Month); None (book unchanged) on every other session."""
    if t < 1 or int(p.dates[t]) // 100 == int(p.dates[t - 1]) // 100:
        return None
    n1, n2, na = prm["rank1"], prm["rank2"], prm["abs_mom"]
    if t < max(n1, n2, na):
        return None
    c = p.cf
    ok = [j for j in range(len(p.symbols)) if np.isfinite(c[t, j]) and np.isfinite(c[t - n1, j])
          and np.isfinite(c[t - n2, j]) and np.isfinite(c[t - na, j])]
    if len(ok) < len(p.symbols):                    # the port waits until every instrument is valid
        return None
    m1 = {j: math.log(c[t, j] / c[t - n1, j]) / n1 for j in ok}
    m2 = {j: math.log(c[t, j] / c[t - n2, j]) / n2 for j in ok}
    r1 = {j: k for k, j in enumerate(sorted(ok, key=lambda j: (-m1[j], p.symbols[j])))}
    r2 = {j: k for k, j in enumerate(sorted(ok, key=lambda j: (-m2[j], p.symbols[j])))}
    top = sorted(ok, key=lambda j: (prm["tie"] * r1[j] + r2[j], p.symbols[j]))[:prm["top"]]
    return {j: prm["weight"] for j in top if c[t, j] > c[t - na, j]}


def run_sector(spec, panels=None) -> dict:
    pan = panels or sector_panels()
    p1, n1 = pan["era1"]
    p2, n2 = pan["era2"]
    first1 = int(p1.dates[max(spec.params["rank1"], spec.params["abs_mom"])])
    r1 = RP.backtest(p1, spec.pspec, max(first1, ERA1["first_decision"]), ERA1["end"])
    r2 = RP.backtest(p2, spec.pspec, 20231201)
    return _judge_two_eras(r1, r2, p1, n1, n2)


# ══════════════════════════════════════════════════════════════════════════
# registry
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class R7Spec:
    id: str
    desc: str
    params: dict
    runner: object
    parts: tuple                      # functions whose source the fingerprint covers
    pspec: object = None              # r6_portfolio.PSpec for engine-based specs

    @property
    def fingerprint(self) -> str:
        src = "".join(inspect.getsource(f) for f in self.parts + (self.runner, _judge_two_eras, fut_cost))
        if self.pspec is not None:
            src += self.pspec.fingerprint
        blob = repr((self.id, sorted((k, repr(v)) for k, v in self.params.items()), src))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


_SECT_PSPEC = RP.PSpec("r7_gh2_sector_rot", "Alvarez sector rotation on 11 NSE sector indices", w_sector, dict(SECT),
                       "daily", "cnc")
SPECS = [
    R7Spec("r7_gh2_volmanaged_nifty", "Moreira-Muir volatility-managed NIFTY: w = min(1.5, c / RV_prev month), expanding c; vs buy-and-hold",
           dict(VOL), run_volmanaged, (vm_weights, vm_backtest, load_nifty_daily)),
    R7Spec("r7_gh2_carry_xs", "Weekly cross-sectional stock-futures carry (front vs second, annualised): long top / short bottom quintile, P&L on the contracts",
           dict(CARRY), run_carry, (futures_frame, carry_arrays, carry_score, contract_returns, carry_weights,
                                    carry_backtest, carry_panels, _days)),
    R7Spec("r7_gh2_sector_rot", "Alvarez ETF sector rotation on 11 NSE sector indices: 252/126 rank, top 2 at 50 % if above 252d ago, monthly (first session)",
           dict(SECT), run_sector, (w_sector, sector_panels, _read_ind2), pspec=_SECT_PSPEC),
]
BY_ID = {s.id: s for s in SPECS}


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if not str(k).startswith("_")}
    return o


def _trial_rows(spec, res, prereg_sha):
    base = {"strategy_id": spec.id, "family": "r7_gh2", "group": "gh2_portfolio", "params_fp": spec.fingerprint,
            "source": "github2_r7/portfolio_ledger.json", "verdict": res["verdict"]}
    if spec.id == "r7_gh2_volmanaged_nifty":
        e1, e2 = res["era1"], res["era2"]
        cm = f"future+{spec.params['index_slip_bps']:g}bps"
        return [dict(base, era="2012-23", protocol="r7 vol-managed vs buy-and-hold", stage="oos", dataset="nse_old",
                     period=f"{e1['whole']['managed']['from']}-{e1['whole']['managed']['to']}", cost_model=cm,
                     note=f"Sharpe {e1['whole']['managed']['sharpe']:.2f} vs B&H {e1['whole']['buy_hold']['sharpe']:.2f} "
                          f"maxDD {e1['whole']['managed']['max_dd']:.3f} vs {e1['whole']['buy_hold']['max_dd']:.3f} "
                          f"prereg {prereg_sha[:12]}"),
                dict(base, era="2023-26", protocol="r7 vol-managed vs buy-and-hold", stage="screen", dataset="bars3y",
                     period=f"{e2['whole']['managed']['from']}-{e2['whole']['managed']['to']}", cost_model=cm,
                     note=f"Sharpe {e2['whole']['managed']['sharpe']:.2f} vs B&H {e2['whole']['buy_hold']['sharpe']:.2f}")]
    e1, e2 = res["era1"], res["era2"]
    cm = "future+10bps" if spec.id == "r7_gh2_carry_xs" else "equity_cnc+10bps"
    return [dict(base, era="2012-23", protocol="r7 portfolio (factor_momentum gates)", stage="oos", dataset="nse_old",
                 period=f"{e1['whole']['from']}-{e1['whole']['to']}", trades=e1["trades"], sr_daily=e1["sr_daily"],
                 cost_model=cm, note=f"CAGR {e1['whole']['cagr']:.4f} excess H1 {e1['h1']['excess']:.4f} H2 "
                                     f"{e1['h2']['excess']:.4f} Sharpe {e1['whole']['sharpe']:.2f} prereg {prereg_sha[:12]}"),
            dict(base, era="2023-26", protocol="r7 portfolio (factor_momentum gates)", stage="screen", dataset="bars3y",
                 period=f"{e2['whole']['from']}-{e2['whole']['to']}", trades=e2["trades"], sr_daily=e2["sr_daily"],
                 cost_model=cm, note=f"CAGR {e2['whole']['cagr']:.4f} excess {e2['whole']['excess']:.4f}")]


def _lines(spec, res) -> list:
    L = [f"{spec.id}  ->  {res['verdict']}  {('failed: ' + ', '.join(res['failed'])) if res['failed'] else ''}"]
    if spec.id == "r7_gh2_volmanaged_nifty":
        for era, segs in (("era1", ("h1", "h2", "whole")), ("era2", ("whole",))):
            for k in segs:
                m, b = res[era][k]["managed"], res[era][k]["buy_hold"]
                L.append(f"  {era} {k:5s} {m['from']}..{m['to']}  managed: {FM._fmt(m)}")
                L.append(f"  {era} {k:5s} {b['from']}..{b['to']}  buy-hold: {FM._fmt(b)}")
            L.append(f"  {era} mean weight {res[era]['mean_weight']:.2f}, months at the cap {res[era]['max_weight_share_at_cap']:.0%}")
        return L
    e1, e2 = res["era1"], res["era2"]
    for k in ("h1", "h2", "whole"):
        L.append(f"  era1 {k:5s} {e1[k]['from']}..{e1[k]['to']}  {FM._fmt(e1[k])}")
    L.append(f"  era2 whole {e2['whole']['from']}..{e2['whole']['to']}  {FM._fmt(e2['whole'])}")
    L.append(f"  trades {e1['trades']}/{e2['trades']}, mean gross exposure {e1['mean_gross']:.2f}/{e2['mean_gross']:.2f}")
    return L


def main(argv=None) -> int:
    from .data import assert_outside_repo
    ap = argparse.ArgumentParser(prog="python -m stratlab.r7_portfolio")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true")
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
            print(f"{s.id:<26} {s.fingerprint}  {s.desc}")
        return 0
    if a.smoke:
        bad = 0
        for s in specs:
            try:
                if s.id == "r7_gh2_sector_rot":
                    pan = sector_panels()
                    for era in ("era1", "era2"):
                        p, _n = pan[era]
                        RP.backtest(p, s.pspec, int(p.dates[300]), int(p.dates[420]))
                elif s.id == "r7_gh2_carry_xs":
                    (p1, _n1, f1), (p2, _n2, f2) = carry_panels()
                    for p, f, lo, hi in ((p1, f1, 20130104, 20130628), (p2, f2, 20231201, 20240531)):
                        A = carry_arrays(f, p.dates, p.symbols)
                        g, dd, rl = contract_returns(A, p.open, p.close, s.params["ca_tol"])
                        carry_backtest(p, A, g, dd, rl, s.params, lo, hi)
                else:
                    (d1, o1, c1), (d2, o2, c2) = load_nifty_daily()
                    w = vm_weights(d1[:600], c1[:600], s.params)
                    vm_backtest(d1, o1, c1, w, 20130101, 20140630, 1.0)
                print(f"smoke ok   {s.id}")
            except Exception as exc:                                  # noqa: BLE001
                import traceback
                traceback.print_exc()
                bad += 1
                print(f"smoke FAIL {s.id}: {type(exc).__name__}: {exc}")
        return 1 if bad else 0
    if not (a.prereg and a.ledger and a.out):
        ap.error("--prereg, --ledger and --out are required for a run")
    sha = _sha256(a.prereg)
    if not PREREG_SHA256 or sha != PREREG_SHA256:
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
    lines = []
    for s in specs:
        if s.id in led:
            print(f"REFUSED: {s.id} already judged {led[s.id]['when']} (one look). Stored verdict: "
                  f"{led[s.id]['result']['verdict']}")
            continue
        res = s.runner(s)
        curves = {}
        if s.id == "r7_gh2_volmanaged_nifty":
            for era in ("era1", "era2"):
                vm, bh = res[era]["_r"]
                curves[f"{era}_managed"], curves[f"{era}_buyhold"] = vm, bh
        else:
            curves["era1"], curves["era2"] = res.pop("_r1"), res.pop("_r2")
        res = _strip(res)
        led[s.id] = {"when": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                     "fingerprint": s.fingerprint, "prereg_sha256": sha, "params": s.params, "result": res}
        FM.save_ledger(ledger_path, led)                               # written immediately: one look
        for name, r in curves.items():
            with open(os.path.join(out, f"{s.id}_{name}_equity.csv"), "w") as f:
                f.write("date,equity,gross\n")
                for d_, e_, g_ in zip(r["dates"], r["equity"], r["gross"]):
                    f.write(f"{d_},{e_:.2f},{g_:.4f}\n")
            with open(os.path.join(out, f"{s.id}_{name}_trades.csv"), "w") as f:
                f.write("date,symbol,side,value,cost,stale_exit\n")
                for t in r["trades"]:
                    f.write(",".join(str(x) for x in t) + "\n")
        L = _lines(s, res)
        lines += L
        print("\n".join(L), flush=True)
        if a.trials:
            from .trials import TrialsLedger
            TrialsLedger(a.trials).append(_trial_rows(s, res, sha))
    with open(os.path.join(out, "summary.txt"), "a") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
