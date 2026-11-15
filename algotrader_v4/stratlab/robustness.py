"""
stratlab/robustness.py — REPORT-ONLY overfitting and robustness diagnostics borrowed from established practice.

Nothing here changes a protocol verdict, writes to a FINAL ledger or the trials ledger, adds a trial to N, or may be
used to pick or tune a strategy. It reads trade files and bars that already exist and writes a report to a scratch
directory (refused inside the repository).

    python -m stratlab.robustness --scratch <scratchpad> --out <scratch dir> [--parts pbo,spa,mc,wf,lookahead]

Parts
  pbo        Probability of Backtest Overfitting by Combinatorially Symmetric Cross-Validation (Bailey, Borwein,
             Lopez de Prado, Zhu 2014/2017): per trial family, the T x N matrix of DISCOVERY daily net P&L (one
             column per variant that has a 3-year trade file) is cut into S = 16 blocks; for each of the C(16, 8)
             halves the in-sample best variant (daily Sharpe) is ranked out of sample; PBO = share of halves where
             it lands at or below the OOS median (logit <= 0). Also reported: P(OOS Sharpe of the IS-best < 0) and
             the slope of OOS on IS Sharpe of the selected variant (performance degradation).
  spa        White's Reality Check (2000) and Hansen's Superior Predictive Ability test (2005; l / c / u p-values)
             for "the best variant of the family beats the benchmark", stationary bootstrap (Politis-Romano,
             mean block 10 sessions, 2,000 draws, one shared index set for all variants). Benchmarks: zero, and
             NIFTY buy-and-hold on the variant's average capital in use (Rs 1 lakh x mean open positions per day).
  mc         Monte Carlo on the trade list (Jesse / Build Alpha style): trade-order permutation (path risk at a
             fixed total), trade bootstrap with replacement (total and drawdown), and a stationary block bootstrap
             of DAILY P&L (keeps same-day clustering, which order shuffling destroys).
  wf         Walk-forward VIEW: net P&L in rolling 12-month windows stepped monthly over every trade file of the
             strategy (old era + 2023-26). It does not re-fit anything (the lab's rules have no fitted
             parameters); it shows how stable the P&L of the fixed rule was through time.
  lookahead  freqtrade-style lookahead-analysis and recursive-analysis on catalogue strategies:
             (a) intraday: for sampled signal sessions, rebuild the bars TRUNCATED at each signal bar (a new
                 SymbolData ending at that minute, every indicator recomputed by build_sessions) and require the
                 strategy's decisions up to that bar to be identical to the full-data run; also compare every
                 Session indicator array on the truncated vs the full data.
             (b) positional: decisions from a SymbolData truncated at the decision minute vs the simulator's own
                 view, and the recursive (warm-up) check: the decision with history starting only 250 / 120
                 sessions before the decision day vs the full history.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

from . import data as D

# ───────────────────────────── statistics ─────────────────────────────


def daily_sharpe(x: np.ndarray, axis=0) -> np.ndarray:
    m = x.mean(axis=axis)
    s = x.std(axis=axis, ddof=1)
    return np.where(s > 0, m / np.where(s > 0, s, 1.0), 0.0)


def cscv_pbo(M: np.ndarray, S: int = 16, max_combos: int | None = None, seed: int = 0) -> dict:
    """PBO by CSCV. M = T x N performance matrix (rows = time). Performance = Sharpe of the rows in a half.
    Returns {"pbo", "n_combos", "prob_oos_loss", "slope", "median_logit", "N", "T"}."""
    M = np.asarray(M, float)
    T, N = M.shape
    if N < 2 or T < S * 2:
        raise ValueError(f"CSCV needs N >= 2 and T >= 2S (N={N}, T={T}, S={S})")
    cut = np.array_split(np.arange(T), S)
    bsum = np.array([M[ix].sum(axis=0) for ix in cut])            # S x N
    bsq = np.array([(M[ix] ** 2).sum(axis=0) for ix in cut])
    bn = np.array([len(ix) for ix in cut], float)
    combos = list(itertools.combinations(range(S), S // 2))
    if max_combos and len(combos) > max_combos:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in sorted(rng.choice(len(combos), max_combos, replace=False))]
    C = np.zeros((len(combos), S), bool)
    for k, cb in enumerate(combos):
        C[k, list(cb)] = True

    def sharpe(mask):
        n = mask.astype(float) @ bn
        s1 = mask.astype(float) @ bsum
        s2 = mask.astype(float) @ bsq
        mean = s1 / n[:, None]
        var = (s2 - n[:, None] * mean ** 2) / (n[:, None] - 1)
        sd = np.sqrt(np.maximum(var, 0))
        return np.where(sd > 0, mean / np.where(sd > 0, sd, 1.0), 0.0)

    is_sr, oos_sr = sharpe(C), sharpe(~C)
    best = is_sr.argmax(axis=1)
    sel_oos = oos_sr[np.arange(len(C)), best]
    # relative OOS rank of the IS-best (1 = worst .. N = best), ties averaged
    ranks = np.array([(row < row[b]).sum() + 0.5 * ((row == row[b]).sum() - 1) + 1 for row, b in zip(oos_sr, best)])
    w = ranks / (N + 1)
    logit = np.log(w / (1 - w))
    sel_is = is_sr[np.arange(len(C)), best]
    slope = float(np.polyfit(sel_is, sel_oos, 1)[0]) if np.ptp(sel_is) > 0 else float("nan")
    return {"pbo": float((logit <= 0).mean()), "n_combos": len(C), "prob_oos_loss": float((sel_oos < 0).mean()),
            "slope": slope, "median_logit": float(np.median(logit)), "N": int(N), "T": int(T), "S": S}


def stationary_bootstrap_indices(T: int, B: int, mean_block: float, rng) -> np.ndarray:
    """B x T indices of the Politis-Romano stationary bootstrap (geometric block lengths, wrap-around)."""
    p = 1.0 / mean_block
    idx = np.empty((B, T), np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    new = rng.random((B, T)) < p
    starts = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def spa_test(Dm: np.ndarray, B: int = 2000, mean_block: float = 10.0, seed: int = 0) -> dict:
    """White's Reality Check and Hansen's SPA for H0: no column of Dm (T x K relative performance, e.g. strategy
    P&L minus benchmark P&L per day) has a positive mean. Returns p-values {rc, spa_l, spa_c, spa_u} and the best."""
    Dm = np.asarray(Dm, float)
    T, K = Dm.shape
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, B, mean_block, rng)
    dbar = Dm.mean(axis=0)
    boot = np.empty((B, K))
    for b in range(B):
        boot[b] = Dm[idx[b]].mean(axis=0)
    st = math.sqrt(T)
    omega = np.sqrt(np.maximum(((st * (boot - dbar)) ** 2).mean(axis=0), 1e-300))
    # White RC (not studentised)
    v = st * dbar.max()
    v_star = (st * (boot - dbar)).max(axis=1)
    p_rc = float((v_star >= v).mean())
    # Hansen SPA (studentised; lower / consistent / upper re-centring)
    t_spa = max(0.0, float((st * dbar / omega).max()))
    thr = -np.sqrt((omega ** 2 / T) * 2 * math.log(math.log(T)))
    mu = {"l": np.maximum(dbar, 0.0), "c": np.where(dbar >= thr, dbar, 0.0), "u": dbar}
    out = {}
    for k, m in mu.items():
        z = st * (boot - m) / omega
        out[f"spa_{k}"] = float((np.maximum(0.0, z.max(axis=1)) >= t_spa).mean())
    kbest = int(dbar.argmax())
    out.update({"rc": p_rc, "best": kbest, "best_mean": float(dbar[kbest]), "t_spa": t_spa, "T": T, "K": K, "B": B,
                "mean_block": mean_block})
    return out


def max_drawdown(pnl) -> float:
    """Largest peak-to-trough fall of the cumulative sum (a positive number; 0 if never below a peak)."""
    c = np.cumsum(np.asarray(pnl, float))
    if c.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate(([0.0], c)))[1:]
    return float(max(0.0, (peak - c).max()))


def mc_trades(nets, daily=None, n: int = 10_000, seed: int = 0, mean_block: float = 10.0) -> dict:
    """Monte Carlo on a trade list: permutation (path only), bootstrap with replacement (total + path) and, if
    `daily` (P&L per session) is given, a stationary block bootstrap of the daily series."""
    x = np.asarray(nets, float)
    rng = np.random.default_rng(seed)
    pct = (5, 50, 95, 99)
    perm = np.array([max_drawdown(rng.permutation(x)) for _ in range(n)])
    bs = rng.integers(0, len(x), (n, len(x)))
    bs_tot = x[bs].sum(axis=1)
    bs_dd = np.array([max_drawdown(x[r]) for r in bs])
    out = {"trades": int(len(x)), "net": float(x.sum()), "observed_dd": max_drawdown(x),
           "perm_dd": {f"p{p}": float(np.percentile(perm, p)) for p in pct},
           "perm_dd_rank_of_observed": float((perm <= max_drawdown(x)).mean()),
           "boot_total": {f"p{p}": float(np.percentile(bs_tot, p)) for p in pct},
           "boot_p_loss": float((bs_tot <= 0).mean()),
           "boot_dd": {f"p{p}": float(np.percentile(bs_dd, p)) for p in pct}}
    if daily is not None and len(daily) > 20:
        dd_ = np.asarray(daily, float)
        idx = stationary_bootstrap_indices(len(dd_), min(n, 2000), mean_block, rng)
        tot = dd_[idx].sum(axis=1)
        ddd = np.array([max_drawdown(dd_[r]) for r in idx])
        out.update({"observed_daily_dd": max_drawdown(dd_),
                    "block_total": {f"p{p}": float(np.percentile(tot, p)) for p in pct},
                    "block_p_loss": float((tot <= 0).mean()),
                    "block_dd": {f"p{p}": float(np.percentile(ddd, p)) for p in pct}})
    return out


def _month_index(d: int) -> int:
    return (d // 10000) * 12 + (d // 100) % 100 - 1


def walk_forward(trades, window_months: int = 12, step_months: int = 1) -> list:
    """Rolling windows over trades [{"entry", "exit", "net"}] booked on the exit month. Returns rows
    (start yyyymm, end yyyymm, trades, net, pf). Windows only where the trade list covers the whole window."""
    if not trades:
        return []
    by_m = defaultdict(list)
    for t in trades:
        by_m[_month_index(t["exit"])].append(t["net"])
    m0 = _month_index(min(t["entry"] for t in trades))
    m1 = _month_index(max(t["exit"] for t in trades))
    rows = []
    for s in range(m0, m1 - window_months + 2, step_months):
        nets = [x for m in range(s, s + window_months) for x in by_m.get(m, [])]
        gp = sum(x for x in nets if x > 0)
        gl = -sum(x for x in nets if x < 0)
        e = s + window_months - 1
        rows.append({"start": (s // 12) * 100 + s % 12 + 1, "end": (e // 12) * 100 + e % 12 + 1,
                     "trades": len(nets), "net": round(sum(nets), 2),
                     "pf": round(gp / gl, 3) if gl > 0 else (float("inf") if gp > 0 else 0.0)})
    return rows


def wf_summary(rows) -> dict:
    if not rows:
        return {"windows": 0}
    nets = np.array([r["net"] for r in rows])
    worst = rows[int(nets.argmin())]
    return {"windows": len(rows), "pct_positive": float((nets > 0).mean()), "min": float(nets.min()),
            "median": float(np.median(nets)), "max": float(nets.max()),
            "worst_window": f"{worst['start']}-{worst['end']}", "cv": float(nets.std() / abs(nets.mean())) if nets.mean() else float("inf")}


# ───────────────────────────── trade files ─────────────────────────────

def load_trades_any(path: str) -> list:
    """stratlab.trials.load_trades for plain or .gz CSV files of any lab layout."""
    from .trials import load_trades
    if not path.endswith(".gz"):
        return load_trades(path)
    import tempfile
    with gzip.open(path, "rt", newline="") as fh, tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(fh.read())
        name = tmp.name
    try:
        return load_trades(name)
    finally:
        os.unlink(name)


def capital_in_use(trades, calendar, lo, hi) -> float:
    """Mean number of open Rs 1 lakh positions per session (entry..exit sessions inclusive)."""
    days = [d for d in calendar if lo <= d <= hi]
    if not days:
        return 0.0
    import bisect
    n = 0
    for t in trades:
        if lo <= t["entry"] <= hi:
            a = bisect.bisect_left(days, t["entry"])
            b = min(bisect.bisect_left(days, t["exit"]), len(days) - 1)
            n += max(1, b - a + 1)
    return n / len(days)


# ───────────────────────────── families (PBO / SPA) ─────────────────────────────

DISC = (20231110, 20251216)


def family_variants(ledger_path: str, min_trades: int = 20) -> dict:
    """{family: [(strategy_id, params_fp, trades_file)]} for 2023-26 DISCOVERY screens on bars3y with an existing
    trade file; one row per (id, params) variant (current cost model preferred, then the latest row)."""
    best = {}
    with open(ledger_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["era"] != "2023-26" or r["stage"] != "screen" or r["dataset"] != "bars3y":
                continue
            if r["period"] != f"{DISC[0]}-{DISC[1]}" or not r["trades_file"] or not os.path.exists(r["trades_file"]):
                continue
            try:
                if int(float(r["trades"] or 0)) < min_trades:
                    continue
            except ValueError:
                continue
            k = (r["strategy_id"], r["params_fp"])
            score = (1 if r["cost_model"] else 0, r["recorded_at"])
            if k not in best or score > best[k][0]:
                best[k] = (score, r)
    fam = defaultdict(list)
    for (sid, fp), (_s, r) in best.items():
        fam[r["family"]].append((sid, fp, r["trades_file"]))
    return dict(fam)


def nifty_daily(scratch: str) -> dict:
    for cache in sorted(os.listdir(os.path.join(scratch, "stratlab_cache"))):
        if cache.startswith("bars3y_"):
            a = D.load_daily_aux(os.path.join(scratch, "stratlab_cache", cache), "_NIFTY50_day")
            if a is not None:
                return {int(d): float(c) for d, c in zip(a["date"], a["c"])}
    return {}


def run_families(scratch: str, ledger: str, min_variants: int = 8, log=print) -> dict:
    from .trials import calendar_3y, daily_pnl
    cal = calendar_3y(scratch)
    days = [d for d in cal if DISC[0] <= d <= DISC[1]]
    nifty = nifty_daily(scratch)
    nr = np.array([(nifty.get(d, np.nan) / nifty.get(p, np.nan) - 1.0) if (d in nifty and p in nifty) else 0.0
                   for p, d in zip([days[0]] + days[:-1], days)])
    nr[0] = 0.0
    fams = family_variants(ledger)
    out = {}
    all_cols, all_caps, all_names = [], [], []
    for f, vs in sorted(fams.items()):
        cols, caps, names = [], [], []
        for sid, fp, path in vs:
            try:
                tr = load_trades_any(path)
            except Exception:
                continue
            pnl = daily_pnl(tr, cal, *DISC)
            if not any(pnl):
                continue
            cols.append(pnl)
            caps.append(capital_in_use(tr, cal, *DISC))
            names.append(sid)
        if not cols:
            continue
        M = np.array(cols).T
        all_cols.extend(cols); all_caps.extend(caps); all_names.extend(names)
        res = {"variants": len(names), "enough": len(names) >= min_variants}
        if len(names) >= 2:
            res["pbo"] = cscv_pbo(M)
        bench = np.outer(nr, np.array(caps) * 1e5)
        res["spa_zero"] = spa_test(M)
        res["spa_nifty"] = spa_test(M - bench)
        res["best_variant"] = names[res["spa_zero"]["best"]]
        res["best_variant_vs_nifty"] = names[res["spa_nifty"]["best"]]
        res["best_disc_net"] = float(M[:, res["spa_zero"]["best"]].sum())
        out[f] = res
        log(f"  {f}: N={len(names)} PBO={res.get('pbo', {}).get('pbo', float('nan')):.2f} "
            f"SPA_c(0)={res['spa_zero']['spa_c']:.3f} SPA_c(NIFTY)={res['spa_nifty']['spa_c']:.3f}")
    if all_cols:
        M = np.array(all_cols).T
        bench = np.outer(nr, np.array(all_caps) * 1e5)
        res = {"variants": len(all_names), "enough": True, "pbo": cscv_pbo(M, max_combos=4000),
               "spa_zero": spa_test(M, B=1000), "spa_nifty": spa_test(M - bench, B=1000)}
        res["best_variant"] = all_names[res["spa_zero"]["best"]]
        res["best_variant_vs_nifty"] = all_names[res["spa_nifty"]["best"]]
        res["best_disc_net"] = float(M[:, res["spa_zero"]["best"]].sum())
        out["_ALL_"] = res
        log(f"  ALL: N={len(all_names)} PBO={res['pbo']['pbo']:.2f} SPA_c(0)={res['spa_zero']['spa_c']:.3f}")
    return out


# ───────────────────────────── named strategies (MC / WF) ─────────────────────────────

def named_sources(scratch: str) -> dict:
    """{strategy: {"old": trade file of the old-era run, "2023-26": the 3-year DISCOVERY + FINAL trade file}}."""
    j = lambda *p: os.path.join(scratch, *p)  # noqa: E731
    om = lambda run, sid: j("oldera_min", run, "trades", f"{sid}.csv.gz")  # noqa: E731
    return {
        "AUC-FADE-2PCT": {"old": om("run_b", "AUC-FADE-2PCT"), "2023-26": j("dd", "auction_final", "trades_AUC-FADE-2PCT.csv")},
        "AUC-FADE-3PCT": {"old": om("run_a", "AUC-FADE-3PCT"), "2023-26": j("dd", "auction_final", "trades_AUC-FADE-3PCT.csv")},
        "SWING-ATTN-ON-3X": {"old": om("run_a", "SWING-ATTN-ON-3X"), "2023-26": j("dd", "r2_swing_out", "trades_SWING-ATTN-ON-3X.csv")},
        "VRP-STRANGLE-016-WINGED": {"old": om("run_a", "VRP-STRANGLE-016-WINGED"),
                                    "2023-26": j("dd", "rev18", "repro", "trades", "VRP-STRANGLE-016-WINGED.csv")},
        "ev_pead_long_h60": {"old": j("events", "run1", "trades", "ev_pead_long_h60_2012-23.csv"),
                             "2023-26": j("events", "run1", "trades", "ev_pead_long_h60_2023-26.csv")},
        "fl_inst_buy_h60": {"old": j("flows", "run1", "trades", "fl_inst_buy_h60_2012-23.csv"),
                            "2023-26": j("flows", "run1", "trades", "fl_inst_buy_h60_2023-26.csv")},
    }


def t_stat(daily) -> float:
    """t statistic of the mean daily P&L (iid; Harvey-Liu-Zhu 2016 ask for t > 3.0 after a large search)."""
    x = np.asarray(daily, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def run_named(scratch: str, n_mc: int = 10_000, log=print) -> dict:
    from .trials import calendar_3y, calendar_old, daily_pnl
    cal3 = calendar_3y(scratch)
    out = {}
    for sid, paths in named_sources(scratch).items():
        eras = {k: sorted(load_trades_any(p), key=lambda t: (t["exit"], t["entry"]))
                for k, p in paths.items() if os.path.exists(p)}
        res = {}
        if "2023-26" in eras:
            tr = eras["2023-26"]
            lo, hi = cal3[0], cal3[-1]
            res["mc_2023_26"] = mc_trades([t["net"] for t in tr], daily_pnl(tr, cal3, lo, hi), n=n_mc)
            res["t_daily_2023_26"] = t_stat(daily_pnl(tr, cal3, lo, hi))
        if "old" in eras:
            tr = eras["old"]
            calo = calendar_old(scratch)
            lo_o, hi_o = min(t["entry"] for t in tr), max(t["exit"] for t in tr)
            dpo = daily_pnl(tr, calo, lo_o, hi_o) if calo else None
            res["mc_old"] = mc_trades([t["net"] for t in tr], dpo, n=n_mc)
            res["old_range"] = f"{lo_o}-{hi_o}"
            if dpo:
                res["t_daily_old"] = t_stat(dpo)
        allt = sorted([t for v in eras.values() for t in v], key=lambda t: t["exit"])
        rows = walk_forward(allt)
        res["wf"] = wf_summary(rows)
        res["wf_rows"] = rows
        yr = defaultdict(float)
        for t in allt:
            yr[t["exit"] // 10000] += t["net"]
        res["by_year"] = {int(k): round(v, 0) for k, v in sorted(yr.items())}
        out[sid] = res
        m = res.get("mc_2023_26", {})
        log(f"  {sid}: 2023-26 trades {m.get('trades')} DD obs {m.get('observed_dd', 0):,.0f} perm p95 "
            f"{m.get('perm_dd', {}).get('p95', 0):,.0f}; WF {res['wf'].get('pct_positive', 0):.0%} of "
            f"{res['wf'].get('windows')} windows > 0")
    return out


# ───────────────────────────── look-ahead / recursive ─────────────────────────────

INTRADAY_CHECK = ["r3_tf5m_macd_zero_adx", "r3_tf5m_keltner_break", "r3_tf15m_fh_break", "r3_scalping_rsi5_extreme",
                  "cs_hammer_15m", "r3_intraday_vwap2sd_quietidx", "scalp_orb5", "r5_kite_streak_orb5_long"]
POSITIONAL_CHECK = ["r3_tfdaily_ema20_50_cross", "r3_tfdaily_bb_upper_short_downtrend", "cs_hammer_d",
                    "r5_kite_vars_ema25_50", "r3_tfweekly_wk_loser_rev", "r3_swing_close_down4"]
SESSION_ARRAYS = ("t", "o", "h", "l", "c", "v", "vwap", "vsd", "ema9", "ema21", "ema50", "rsi", "atr", "e9_5",
                  "e21_5", "e50_5", "rsi5", "atr5", "vr", "day_hi", "day_lo", "nifty")
SESSION_SCALARS = ("pdo", "pdh", "pdl", "pdc", "gap", "datr", "dsma20", "dvol20", "nifty_pdc")


def truncate(sd, k: int, m: int):
    """A new SymbolData holding every bar before session k plus the first m bars of session k (as if the data
    file ended at that minute)."""
    n = int(sd.starts[k]) + int(m)
    ends = sd.ends[:k + 1].copy()
    ends[k] = n
    return D.SymbolData(sd.symbol, sd.date[:n], sd.t[:n], sd.o[:n], sd.h[:n], sd.l[:n], sd.c[:n], sd.v[:n],
                        sd.starts[:k + 1].copy(), ends, sd.days[:k + 1].copy(), [])


FIVE_MIN_FIELDS = ("e9_5", "e21_5", "e50_5", "rsi5", "atr5")


def hide_partial_5m(ts) -> int:
    """build_sessions treats the LAST 5-minute block of its input as complete. On data cut in the middle of a
    5-minute candle that block is still forming, so a recompute would show the half-built candle that the full-data
    run (correctly) does not use yet. Put the truncated session back on the information a live system has: the
    5-minute fields of the trailing partial block take the value of the last complete block, and bars5() drops the
    partial candle. Returns the number of bars adjusted (0 when the cut is on a 5-minute boundary)."""
    from .indicators import Bars5
    from .data import SESSION_OPEN
    t = np.asarray(ts.t)
    m = len(t)
    if m == 0 or (int(t[-1]) - SESSION_OPEN) % 5 == 4:
        return 0
    blk = (t.astype(int) - SESSION_OPEN) // 5
    b0 = int(np.searchsorted(blk, blk[-1]))
    for name in FIVE_MIN_FIELDS:
        arr = np.array(getattr(ts, name), float)
        arr[b0:] = arr[b0 - 1] if b0 > 0 else np.nan
        setattr(ts, name, arr)
    b = ts._b5
    k = int(np.searchsorted(b.end, b0 - 1, side="right"))
    ts._b5 = Bars5(b.o[:k], b.h[:k], b.l[:k], b.c[:k], b.v[:k], b.end[:k])
    return m - b0


def _canon_entries(ents, upto):
    f = lambda x: None if x is None else round(float(x), 6)  # noqa: E731
    out = []
    for e in sorted(ents or [], key=lambda x: x.i):
        if e.i <= upto:
            xs = getattr(e, "exit_sig", None)
            out.append((int(e.i), int(e.side), f(e.stop), f(e.target), f(e.rr), e.time_stop, e.trail,
                        getattr(e, "bracket", None),
                        None if xs is None else tuple(np.flatnonzero(np.asarray(xs, bool)[:upto + 1]).tolist())))
    return out


def lookahead_intraday(spec, sd, index, lo, hi, max_sessions: int = 25, seed: int = 0) -> dict:
    """Truncate-and-recompute check for one intraday spec on one symbol (see module docstring)."""
    from .indicators import build_sessions
    full = build_sessions(sd, index)
    rng = np.random.default_rng(seed)
    cand = []
    for s in full:
        if lo <= s.date <= hi and not (spec.needs_index and s.nifty is None):
            ents = spec.fn(s, spec.p) or []
            if ents:
                cand.append((s, ents))
    if len(cand) > max_sessions:
        cand = [cand[i] for i in sorted(rng.choice(len(cand), max_sessions, replace=False))]
    res = {"sessions": len(cand), "cuts": 0, "partial_5m_cuts": 0, "signal_diffs": 0, "indicator_diffs": defaultdict(int),
           "indicator_diffs_last_bar_only": defaultdict(int), "examples": []}
    for s, ents in cand:
        n = len(s.t)
        cuts = sorted({int(e.i) + 1 for e in ents if int(e.i) + 1 < n} | {n // 3 + 2, n // 2 + 1})   # + two cuts
        # that are not on a 5-minute boundary (a half-built 5-minute candle at the cut)
        for m in cuts:
            if m < 2:
                continue
            tsd = truncate(sd, s.day_index, m)
            ts = build_sessions(tsd, index)[-1]
            assert ts.date == s.date and len(ts.t) == m
            res["partial_5m_cuts"] += int(hide_partial_5m(ts) > 0)
            res["cuts"] += 1
            a = _canon_entries(spec.fn(s, spec.p), m - 1)
            b = _canon_entries(spec.fn(ts, spec.p), m - 1)
            if a != b:
                res["signal_diffs"] += 1
                if len(res["examples"]) < 3:
                    res["examples"].append(f"{s.symbol} {s.date} cut {m}: full {a[-1:] } truncated {b[-1:]}")
            for name in SESSION_ARRAYS:
                x, y = getattr(s, name, None), getattr(ts, name, None)
                if x is None or y is None:
                    continue
                x = np.asarray(x[:m], float)
                y = np.asarray(y, float)
                bad = ~((x == y) | (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-9 * np.maximum(1, np.abs(x))))
                if bad.any():
                    if bad[:-1].any():
                        res["indicator_diffs"][name] += 1
                    else:
                        res["indicator_diffs_last_bar_only"][name] += 1
            for name in SESSION_SCALARS:
                x, y = getattr(s, name, None), getattr(ts, name, None)
                if x is not None and y is not None and not (x == y or (np.isnan(x) and np.isnan(y))):
                    res["indicator_diffs"][name] += 1
    res["indicator_diffs"] = dict(res["indicator_diffs"])
    res["indicator_diffs_last_bar_only"] = dict(res["indicator_diffs_last_bar_only"])
    return res


def _positional_view(spec, sd, day, nifty_day, vix_day, hist_start: int = 0):
    """The simulator's decision view for decision day `day` (sim.sim_positional), optionally with the daily
    history starting at session `hist_start` (recursive / warm-up check)."""
    from .sim import DailyView, _aligned
    d = sd.daily()
    nifty, vix = _aligned(d["date"], nifty_day), _aligned(d["date"], vix_day)
    a0 = hist_start
    if spec.decision_time is not None:
        a, b = int(sd.starts[day]), int(sd.ends[day])
        m = a + int(np.searchsorted(sd.t[a:b], spec.decision_time))
        if m >= b or m == a:
            return None
        last = (d["o"][day], sd.h[a:m].max(), sd.l[a:m].min(), sd.c[m - 1], sd.v[a:m].sum())
        cols = [np.append(d[k][a0:day], x) for k, x in zip("ohlcv", last)]
        v = DailyView(d["date"][a0:day + 1], *cols)
        v.nifty = None if nifty is None else np.append(nifty[a0:day], np.nan)
        v.vix = None if vix is None else np.append(vix[a0:day], np.nan)
    else:
        v = DailyView(d["date"][a0:day + 1], *[d[k][a0:day + 1] for k in "ohlcv"])
        v.nifty = None if nifty is None else nifty[a0:day + 1]
        v.vix = None if vix is None else vix[a0:day + 1]
    v.symbol = sd.symbol
    return v


def _sig_key(sig):
    if sig is None:
        return None
    r = lambda x: None if x is None else round(float(x), 6)  # noqa: E731
    return (int(sig.side), r(sig.stop), r(sig.target), r(sig.stop_pct), r(sig.target_pct), r(sig.rr), int(sig.max_hold))


def lookahead_positional(spec, sd, nifty_day, vix_day, lo, hi, max_days: int = 40, seed: int = 0) -> dict:
    """(1) decision from a SymbolData truncated at the decision minute (end of day for next-open specs) vs the
    simulator's view; (2) recursive check: the decision with the history cut to start 250 / 120 sessions before."""
    rng = np.random.default_rng(seed)
    days = [k for k, d in enumerate(sd.days) if lo <= int(d) <= hi]
    sig_days, all_checked = [], 0
    full = {}
    for k in days:
        v = _positional_view(spec, sd, k, nifty_day, vix_day)
        if v is None:
            continue
        full[k] = _sig_key(spec.fn(v, spec.p))
        if full[k] is not None:
            sig_days.append(k)
    pick = sig_days if len(sig_days) <= max_days else [sig_days[i] for i in sorted(rng.choice(len(sig_days), max_days, replace=False))]
    nosig = [k for k in days if full.get(k) is None]
    pick_no = [nosig[i] for i in sorted(rng.choice(len(nosig), min(max_days, len(nosig)), replace=False))] if nosig else []
    res = {"signal_days": len(sig_days), "checked": 0, "trunc_diffs": 0, "recursive": {}, "examples": []}
    for k in pick + pick_no:
        if spec.decision_time is not None:
            a, b = int(sd.starts[k]), int(sd.ends[k])
            m = int(np.searchsorted(sd.t[a:b], spec.decision_time))
            tsd = truncate(sd, k, min(m + 1, b - a))  # data end at the fill bar (its open is the fill price;
            #                                           the decision view never contains it)
        else:
            tsd = truncate(sd, k, int(sd.ends[k] - sd.starts[k]))
        tv = _positional_view(spec, tsd, k, nifty_day, vix_day)
        got = None if tv is None else _sig_key(spec.fn(tv, spec.p))
        res["checked"] += 1
        if got != full[k]:
            res["trunc_diffs"] += 1
            if len(res["examples"]) < 3:
                res["examples"].append(f"{sd.symbol} {int(sd.days[k])}: sim {full[k]} truncated-file {got}")
    for warm in (250, 120):
        changed = checked = 0
        for k in pick + pick_no:
            if k - warm <= 0:
                continue
            v = _positional_view(spec, sd, k, nifty_day, vix_day, hist_start=k - warm)
            g = None if v is None else _sig_key(spec.fn(v, spec.p))
            checked += 1
            changed += int(g != full[k])
        res["recursive"][f"hist_{warm}"] = {"checked": checked, "changed": changed}
    return res


def run_lookahead(scratch: str, symbols=("INFY", "TATASTEEL", "PAYTM", "ADANIENT"), log=print) -> dict:
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    cache = [os.path.join(scratch, "stratlab_cache", c) for c in sorted(os.listdir(os.path.join(scratch, "stratlab_cache")))
             if c.startswith("bars3y_")][0]
    index = D.load_symbol(cache, "_NIFTY50_min")
    nifty_day = D.load_daily_aux(cache, "_NIFTY50_day")
    vix_day = D.load_daily_aux(cache, "_VIX_day")
    out = {}
    for sid in INTRADAY_CHECK + POSITIONAL_CHECK:
        if sid not in REGISTRY:
            continue
        sp = REGISTRY[sid]
        agg = None
        t0 = time.time()
        for sym in symbols:
            sd = D.load_symbol(cache, sym)
            if sp.horizon == "intraday":
                r = lookahead_intraday(sp, sd, index, *DISC)
            else:
                r = lookahead_positional(sp, sd, nifty_day, vix_day, *DISC)
            if agg is None:
                agg = r
            else:
                for k, v in r.items():
                    if isinstance(v, int):
                        agg[k] += v
                    elif isinstance(v, list):
                        agg[k] = (agg[k] + v)[:4]
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, dict):
                                for k3, v3 in vv.items():
                                    agg[k].setdefault(kk, {}).setdefault(k3, 0)
                                    agg[k][kk][k3] += v3
                            else:
                                agg[k][kk] = agg[k].get(kk, 0) + vv
        agg["horizon"] = sp.horizon
        agg["secs"] = round(time.time() - t0, 1)
        out[sid] = agg
        log(f"  {sid}: {json.dumps({k: v for k, v in agg.items() if k != 'examples'})}")
    return out


# ───────────────────────────── CLI ─────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--parts", default="pbo,spa,mc,wf,lookahead")
    ap.add_argument("--mc", type=int, default=10_000)
    a = ap.parse_args(argv)
    out = D.assert_outside_repo(a.out)
    os.makedirs(out, exist_ok=True)
    parts = set(a.parts.split(","))
    ledger = a.ledger or os.path.join(a.scratch, "stratlab_cache", "trials_v2.csv")
    res = {"note": "REPORT ONLY: no protocol verdict, ledger or trial count is changed by these diagnostics"}
    if parts & {"pbo", "spa"}:
        print("families (PBO / SPA) ...", flush=True)
        res["families"] = run_families(a.scratch, ledger)
    if parts & {"mc", "wf"}:
        print("named strategies (Monte Carlo / walk-forward) ...", flush=True)
        res["named"] = run_named(a.scratch, a.mc)
    if "lookahead" in parts:
        print("look-ahead / recursive ...", flush=True)
        res["lookahead"] = run_lookahead(a.scratch)
    with open(os.path.join(out, "robustness.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote", os.path.join(out, "robustness.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
