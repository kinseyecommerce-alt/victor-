"""
stratlab/tests_robustness.py — synthetic tests for stratlab/robustness.py (PBO by CSCV, White RC / Hansen SPA,
Monte Carlo trade resampling, walk-forward windows, freqtrade-style look-ahead / recursive checks, report-only).
Run by test_pipeline.py (section "STRATLAB: robustness diagnostics"), or alone: python -m stratlab.tests_robustness
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from . import data as D
from . import robustness as R
from .api import LONG, Entry, Spec, Swing


def t_pbo_noise_near_half():
    """20 zero-edge variants: the in-sample best is no better than a coin flip out of sample (PBO ~ 0.5)."""
    rng = np.random.default_rng(7)
    M = rng.normal(0, 1, (480, 20))
    r = R.cscv_pbo(M, S=16)
    assert r["n_combos"] == 12870 and r["N"] == 20 and r["T"] == 480
    assert 0.3 <= r["pbo"] <= 0.75, r


def t_pbo_real_edge_low():
    """One variant with a real edge among 19 noise variants: it is picked in sample and stays on top (PBO ~ 0)."""
    rng = np.random.default_rng(8)
    M = rng.normal(0, 1, (480, 20))
    M[:, 3] += 0.4
    r = R.cscv_pbo(M, S=16)
    assert r["pbo"] < 0.05 and r["prob_oos_loss"] < 0.05, r


def t_pbo_hand_case():
    """Two variants, S = 2: one half each; the IS winner is the OOS loser in both splits -> PBO = 1."""
    a = np.array([1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0])
    b = -a
    r = R.cscv_pbo(np.c_[a + np.array([0, .1, 0, .1, 0, .1, 0, .1]), b], S=2)
    assert r["n_combos"] == 2 and r["pbo"] == 1.0, r


def t_spa_noise_and_edge():
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 1, (500, 10))
    rn = R.spa_test(noise, B=600, seed=1)
    assert rn["spa_c"] > 0.05 and rn["rc"] > 0.05, rn
    assert rn["spa_l"] <= rn["spa_c"] + 1e-12 <= rn["spa_u"] + 2e-12, rn
    edge = noise.copy()
    edge[:, 4] += 0.3
    re_ = R.spa_test(edge, B=600, seed=1)
    assert re_["best"] == 4 and re_["spa_c"] < 0.01 and re_["rc"] < 0.05, re_
    # many useless losers do not hide the good one under SPA (studentised, re-centred), they do under RC less so
    bad = np.c_[edge[:, 4:5], rng.normal(-0.5, 3, (500, 40))]
    rb = R.spa_test(bad, B=600, seed=1)
    assert rb["spa_c"] < 0.01, rb


def t_stationary_bootstrap_shape():
    idx = R.stationary_bootstrap_indices(50, 200, 5.0, np.random.default_rng(0))
    assert idx.shape == (200, 50) and idx.min() >= 0 and idx.max() < 50
    runs = (np.diff(idx, axis=1) == 1) | (np.diff(idx, axis=1) == -49)
    assert 0.7 < runs.mean() < 0.9                            # mean block 5 -> 80 % continuation


def t_mc_properties():
    assert R.max_drawdown([1, -2, 3, -4, 1]) == 4.0
    assert R.max_drawdown([1, 2, 3]) == 0.0
    assert R.max_drawdown([-1, -1]) == 2.0                    # from the zero start
    rng = np.random.default_rng(3)
    x = rng.normal(10, 100, 300)
    r = R.mc_trades(x, daily=x, n=500, seed=1)
    assert abs(r["net"] - x.sum()) < 1e-6 and r["trades"] == 300
    p = r["perm_dd"]
    assert p["p5"] <= p["p50"] <= p["p95"] <= p["p99"]
    assert r["boot_total"]["p5"] < x.sum() < r["boot_total"]["p95"]
    assert 0.0 <= r["perm_dd_rank_of_observed"] <= 1.0 and "block_dd" in r
    r2 = R.mc_trades(np.ones(50), n=100)
    assert r2["perm_dd"]["p99"] == 0.0 and r2["boot_p_loss"] == 0.0


def t_walk_forward_windows():
    trades = [{"entry": y * 10000 + m * 100 + 1, "exit": y * 10000 + m * 100 + 15, "net": 1.0}
              for y in (2024, 2025) for m in range(1, 13)]
    trades.append({"entry": 20240301, "exit": 20240302, "net": -5.0})
    rows = R.walk_forward(trades)
    assert len(rows) == 13 and rows[0]["start"] == 202401 and rows[0]["end"] == 202412 and rows[-1]["end"] == 202512
    assert rows[0]["net"] == 7.0 and rows[0]["trades"] == 13 and rows[3]["net"] == 12.0
    s = R.wf_summary(rows)
    assert s["windows"] == 13 and s["min"] == 7.0 and s["max"] == 12.0 and abs(s["pct_positive"] - 1.0) < 1e-12


def _synthetic(days=30, seed=5):
    rng = np.random.default_rng(seed)
    dates, ts, o, h, l, c, v = [], [], [], [], [], [], []
    px = 500.0
    d0 = np.datetime64("2024-01-01")
    k = 0
    while len(set(dates)) < days:
        day = d0 + np.timedelta64(k, "D")
        k += 1
        if day.astype("datetime64[D]").item().weekday() >= 5:
            continue
        ymd = int(str(day).replace("-", ""))
        for t in range(555, 930):
            op = px
            px = max(10.0, px * (1 + rng.normal(0, 0.001)))
            hi, lo = max(op, px) * (1 + abs(rng.normal(0, 3e-4))), min(op, px) * (1 - abs(rng.normal(0, 3e-4)))
            dates.append(ymd); ts.append(t); o.append(op); h.append(hi); l.append(lo); c.append(px)
            v.append(float(rng.integers(100, 1000)))
    return D.from_arrays("SYN", dates, ts, o, h, l, c, v)


def _honest(s, p):
    out = []
    for i in range(30, len(s.t), 37):
        if np.isfinite(s.e9_5[i]) and s.c[i] > s.e9_5[i] and s.c[i] > s.vwap[i]:
            out.append(Entry(i, LONG, stop=float(s.day_lo[i]) - 0.5, rr=2.0))
    return out[:2]


def _cheat(s, p):
    i = int(np.argmax(s.c[: len(s.t) - 1]))                   # the day's high close: needs the whole day
    j = max(0, i - 3)
    return [Entry(j, LONG, stop=float(s.l[j]) - 1.0, rr=1.0)]


def t_lookahead_intraday_truncate_recompute():
    sd = _synthetic()
    lo, hi = int(sd.days[10]), int(sd.days[-1])
    good = Spec("t_honest", "intraday", "t", "honest", _honest, {})
    bad = Spec("t_cheat", "intraday", "t", "cheat", _cheat, {})
    rg = R.lookahead_intraday(good, sd, None, lo, hi, max_sessions=6)
    assert rg["sessions"] == 6 and rg["cuts"] >= 6 and rg["signal_diffs"] == 0, rg
    assert not rg["indicator_diffs"], rg                     # every indicator recomputes identically on truncated data
    rb = R.lookahead_intraday(bad, sd, None, lo, hi, max_sessions=6)
    assert rb["signal_diffs"] > 0, rb


def t_truncate_symboldata():
    sd = _synthetic(days=5)
    t = R.truncate(sd, 3, 40)
    assert t.n_sessions == 4 and len(t.c) == int(sd.starts[3]) + 40 and int(t.ends[-1]) == len(t.c)
    assert np.array_equal(t.c, sd.c[: len(t.c)])


def _ema_rule(d, p):
    if len(d) < 5:
        return None
    a = d.c.astype(float)
    e = a[0]
    for x in a[1:]:
        e += 0.05 * (x - e)                                    # recursive EMA seeded at the first bar
    return Swing(LONG, stop_pct=5.0, max_hold=3) if a[-1] > e else None


def _peek_rule(d, p):
    return Swing(LONG, stop_pct=5.0, max_hold=3) if len(d) % 2 == 0 else None


def t_lookahead_positional_recursive():
    sd = _synthetic(days=300, seed=9)
    sp = Spec("t_ema_rule", "swing", "t", "ema", _ema_rule, {}, horizon="positional")
    r = R.lookahead_positional(sp, sd, None, None, int(sd.days[260]), int(sd.days[-1]), max_days=40)
    assert r["checked"] > 0 and r["trunc_diffs"] == 0, r
    total = sum(v["changed"] for v in r["recursive"].values())
    assert r["recursive"]["hist_250"]["checked"] > 0 and total >= 0
    # a rule that depends on how much history it is given (len(d)) is flagged by the recursive check
    sp2 = Spec("t_len_rule", "swing", "t", "len", _peek_rule, {}, horizon="positional")
    r2 = R.lookahead_positional(sp2, sd, None, None, int(sd.days[260]), int(sd.days[-1]), max_days=40)
    assert r2["recursive"]["hist_250"]["changed"] > 0 or r2["recursive"]["hist_120"]["changed"] > 0, r2


def t_report_only():
    """The module never writes a ledger or records trials, and refuses an output directory inside the repo."""
    src = open(R.__file__).read()
    for word in ("record_evaluation", "final_ledger.json", "_load_ledger", "ledger_key", "backfill("):
        assert word not in src, word
    repo_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(R.__file__))), "robust_out_test")
    try:
        R.main(["--scratch", tempfile.gettempdir(), "--out", repo_out, "--parts", "none"])
        raise AssertionError("output inside the repository was accepted")
    except ValueError:
        pass
    assert not os.path.exists(repo_out)


TESTS = [
    ("robustness: PBO (CSCV, S=16, 12,870 halves) of 20 noise variants is near 0.5", t_pbo_noise_near_half),
    ("robustness: PBO is ~0 when one variant has a real edge", t_pbo_real_edge_low),
    ("robustness: PBO hand case (IS winner is OOS loser in both halves -> 1.0)", t_pbo_hand_case),
    ("robustness: White RC / Hansen SPA: noise not rejected, a real edge is; p_l <= p_c <= p_u", t_spa_noise_and_edge),
    ("robustness: stationary bootstrap indices (shape, mean block length)", t_stationary_bootstrap_shape),
    ("robustness: Monte Carlo trade resampling (drawdown hand values, percentiles, totals)", t_mc_properties),
    ("robustness: rolling 12-month walk-forward windows", t_walk_forward_windows),
    ("robustness: truncated SymbolData keeps every earlier bar", t_truncate_symboldata),
    ("robustness: truncate-and-recompute look-ahead check passes an honest rule, catches a cheat", t_lookahead_intraday_truncate_recompute),
    ("robustness: positional truncated-file check and recursive (warm-up) check", t_lookahead_positional_recursive),
    ("robustness: report-only (no ledger writes, no output inside the repository)", t_report_only),
]


if __name__ == "__main__":
    for name, fn in TESTS:
        fn()
        print("ok", name)
