"""
stratlab/tests_datagen.py — tests for the data-driven discovery study (datagen_features.py, datagen_run.py) on
SYNTHETIC data only: the engine's fills, slippage, costs, holding periods, period-end exits, daily cap, one position
per symbol, universe / F&O / exclusion gates, the minute executions (intraday, overnight, multi-day) with special
sessions and the dirty-link rule; causality of every feature family (changing data after t never changes a decision at
or before t; the ban rule reads only the next day's ban file); the OI-change rule over the same contracts; results
timing; the walk-forward model never sees later data; the shuffle control keeps each symbol's count; the selection
rule (gates, DSR ranking, K and the family cap); the frozen pre-registration matches the code.
Run by test_pipeline.py (section "STRATLAB: data-driven discovery") or directly: python -m stratlab.tests_datagen
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import datagen_features as F
from . import datagen_run as R
from . import sim
from costs import round_trip_cost


def _daily(S=3, T=30, seed=0):
    rng = np.random.default_rng(seed)
    cal = np.array([20200101 + i for i in range(T)], np.int64)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (S, T)), axis=1))
    o = c * (1 + rng.normal(0, 0.002, (S, T)))
    h = np.maximum(o, c) * 1.005
    l = np.minimum(o, c) * 0.995
    v = rng.uniform(1e5, 2e5, (S, T))
    member = np.ones((S, T), bool)
    fo = np.ones((S, T), bool)
    nifty = {"o": 1000 + np.arange(T, dtype=float), "c": 1000.5 + np.arange(T, dtype=float)}
    vix = {"c": 15 + rng.normal(0, 1, T)}
    return F.DailyPanel([f"S{i}" for i in range(S)], cal, o, h, l, c, v, member, fo, nifty, vix, {})


def _sig(s, t, side, rank=None, hold=None):
    n = len(s)
    return {"s": np.asarray(s, np.int64), "t": np.asarray(t, np.int64), "side": np.asarray(side, np.int64),
            "rank": np.asarray(rank if rank is not None else [1.0] * n, float),
            "hold": np.asarray(hold if hold is not None else [0] * n, np.int64)}


def _cand(grid="daily", ex=("d_hold", 3), prod="cnc", id="t_x"):
    return R.Cand(id, "T", grid, "r_ml", (), ex, prod, "test")


def t_daily_fill_slippage_cost_hold():
    """d_hold: decided on t, filled at t+1's open +2 bps, exit at t+h's close -2 bps, qty floor(1 lakh / open), CNC costs"""
    P = _daily()
    ctx = F.Ctx(P, None)
    c = _cand(ex=("d_hold", 3))
    tr = R.execute(c, ctx, 20200101, 20200130, 2.0, _sig([1], [5], [1], hold=[3]))
    assert len(tr) == 1
    t = tr[0]
    o, cl = P.o[1, 6], P.c[1, 8]
    qty = int(100000 // o)
    fill, xp = o * 1.0002, cl * 0.9998
    assert t[2] == int(P.cal[6]) and t[6] == int(P.cal[8]) and t[9] == qty
    assert abs(t[5] - round(fill, 4)) < 1e-9 and abs(t[8] - round(xp, 4)) < 1e-9
    assert abs(t[10] - round((xp - fill) * qty, 2)) < 0.011
    assert abs(t[11] - round_trip_cost(fill * qty, xp * qty, "equity_cnc")) < 1e-9
    assert t[4] == "09:15" and t[7] == "15:29"
    # a short through futures: sells at the open -2 bps, buys back at the close +2 bps, futures costs
    c2 = _cand(ex=("d_hold", 1), prod="fut")
    t2 = R.execute(c2, ctx, 20200101, 20200130, 2.0, _sig([0], [3], [-1], hold=[1]))[0]
    fill, xp = P.o[0, 4] * 0.9998, P.c[0, 4] * 1.0002
    q = int(100000 // P.o[0, 4])
    assert abs(t2[10] - round((fill - xp) * q, 2)) < 0.011
    assert abs(t2[11] - round_trip_cost(xp * q, fill * q, "future")) < 1e-9
    # a CNC short is refused; an intraday (mis) short is allowed
    assert R.execute(_cand(prod="cnc"), ctx, 20200101, 20200130, 2.0, _sig([0], [3], [-1], hold=[3])) == []
    assert len(R.execute(_cand(ex=("d_hold", 1), prod="mis"), ctx, 20200101, 20200130, 2.0,
                         _sig([0], [3], [-1], hold=[1]))) == 1


def t_daily_period_end_and_entry_window():
    """a hold crossing the period end exits at the period's last close; entries outside [lo, hi] are not taken"""
    P = _daily()
    ctx = F.Ctx(P, None)
    c = _cand(ex=("d_hold", 10))
    hi = int(P.cal[12])
    tr = R.execute(c, ctx, int(P.cal[0]), hi, 2.0, _sig([0, 1], [8, 15], [1, 1], hold=[10, 10]))
    assert len(tr) == 1 and tr[0][6] == hi
    assert R.execute(c, ctx, int(P.cal[10]), int(P.cal[20]), 2.0, _sig([0], [8], [1], hold=[10])) == []
    # a missing close on the exit day: the last close before it
    P.c[0, 12] = np.nan
    tr = R.execute(_cand(ex=("d_hold", 4)), ctx, int(P.cal[0]), int(P.cal[29]), 2.0, _sig([0], [8], [1], hold=[4]))
    assert tr[0][6] == int(P.cal[11])


def t_daily_cap_busy_gates():
    """10 a day by rank, one open position per symbol, member / F&O / corporate-action exclusions"""
    P = _daily(S=15)
    ctx = F.Ctx(P, None)
    c = _cand(ex=("d_hold", 5), prod="fut")
    s = list(range(15))
    sig = R._sorted_sig(c, ctx, s, [3] * 15, [1] * 15, [float(i) for i in s], [5] * 15)
    tr = R.execute(c, ctx, 20200101, 20200130, 2.0, sig)
    assert len(tr) == 10 and {t[1] for t in tr} == {f"S{i}" for i in range(5, 15)}
    sig = _sig([2, 2, 2], [3, 5, 9], [1, 1, 1], hold=[5, 5, 5])
    tr = R.execute(c, ctx, 20200101, 20200130, 2.0, sig)
    assert [t[2] for t in tr] == [int(P.cal[4]), int(P.cal[10])]
    P.member[4, 3] = False
    P.fo[5, 3] = False
    P.excl = {6: np.array([int(P.cal[6])], np.int64)}
    tr = R.execute(c, ctx, 20200101, 20200130, 2.0, _sig([4, 5, 6, 7], [3] * 4, [1] * 4, hold=[5] * 4))
    assert [t[1] for t in tr] == ["S7"]


def _minute(S=3, T=12):
    rng = np.random.default_rng(1)
    syms = [f"M{i}" for i in range(S)] + list(F.INDEX_MIN)
    n = len(syms)
    cal = np.array([20210101 + i for i in range(T)], np.int64)
    f = {}
    for k in F.MIN_FIELDS:
        f[k] = 100 + rng.normal(0, 1, (n, T))
    f["nb"] = np.full((n, T), 375.0)
    return F.MinutePanel(syms, cal, f, set(), {})


def t_minute_executions_special_dirty():
    """m_intra uses the named bar opens of the session; m_on the 15:21 open -> next 09:15; m_hold(h) 15:21 -> 15:21 h
    later; special sessions and dirty links are excluded; index rows trade one lot"""
    M = _minute()
    ctx = F.Ctx(None, M)
    c = _cand("minute", ("m_intra", "o0920", "o1515"), "mis")
    t = R.execute(c, ctx, 20210101, 20210112, 2.0, _sig([0], [3], [1]))[0]
    assert abs(t[5] - round(M.f["o0920"][0, 3] * 1.0002, 4)) < 1e-9 and t[4] == "09:20" and t[7] == "15:15"
    assert abs(t[8] - round(M.f["o1515"][0, 3] * 0.9998, 4)) < 1e-9
    c = _cand("minute", ("m_on",), "fut")
    t = R.execute(c, ctx, 20210101, 20210112, 2.0, _sig([1], [4], [-1]))[0]
    assert t[6] == int(M.cal[5]) and abs(t[8] - round(M.f["o0915"][1, 5] * 1.0002, 4)) < 1e-9
    assert R.execute(c, ctx, 20210101, int(M.cal[4]), 2.0, _sig([1], [4], [-1])) == []      # exit beyond the period
    c = _cand("minute", ("m_hold", 3), "fut")
    t = R.execute(c, ctx, 20210101, 20210112, 2.0, _sig([2], [2], [1]))[0]
    assert t[6] == int(M.cal[5]) and abs(t[8] - round(M.f["o1521"][2, 5] * 0.9998, 4)) < 1e-9
    M.special.add(int(M.cal[5]))
    assert R.execute(c, ctx, 20210101, 20210112, 2.0, _sig([2], [2], [1])) == []
    M.special.clear()
    M.dirty[2] = np.array([int(M.cal[4])], np.int64)
    assert R.execute(c, ctx, 20210101, 20210112, 2.0, _sig([2], [2], [1])) == []
    ci = _cand("idx_minute", ("i_on",), "fut")
    t = R.execute(ci, ctx, 20210101, 20210112, 1.0, _sig([3], [2], [1]))[0]
    assert t[1] == "NIFTY" and t[9] == sim.LOT_SIZES["NIFTY"]


def _perturb_after(P: F.DailyPanel, k: int, seed: int = 9) -> F.DailyPanel:
    rng = np.random.default_rng(seed)
    Q = F.DailyPanel(P.syms, P.cal, P.o.copy(), P.h.copy(), P.l.copy(), P.c.copy(), P.v.copy(), P.member, P.fo,
                     {k2: v.copy() for k2, v in P.nifty.items()}, {"c": P.vix["c"].copy()}, P.excl)
    for a in (Q.o, Q.h, Q.l, Q.c):
        a[:, k:] *= rng.uniform(0.8, 1.2, a[:, k:].shape)
    Q.v[:, k:] *= rng.uniform(0.1, 10, Q.v[:, k:].shape)
    Q.vix["c"][k:] *= rng.uniform(0.5, 2.0, len(Q.vix["c"]) - k)
    Q.nifty["c"][k:] *= 1.1
    return Q


def t_daily_features_causal():
    """every daily feature / rule at t is unchanged when all data after t change (delivery, OI, deals, participant, ban
    included); the ban rule at t reads the ban file of t+1 only"""
    P = _daily(S=6, T=300, seed=3)
    rng = np.random.default_rng(4)
    extra = {"d:dp": rng.uniform(10, 90, P.shape), "d:doi": rng.normal(0, 0.15, P.shape),
             "d:ban": rng.random(P.shape) < 0.1,
             "d:deals": {"inst": rng.normal(0, 15, P.shape), "other": rng.normal(0, 15, P.shape)},
             "d:part": {k: rng.uniform(1e5, 3e5, P.shape[1]) for k in
                        ("fii_fut_idx_long", "fii_fut_idx_short", "client_fut_idx_long", "client_fut_idx_short")},
             "d:react": rng.random(P.shape) < 0.05}
    k = 280
    a, b = F.Ctx(P, None), F.Ctx(_perturb_after(P, k + 1), None)
    for ctx, shift in ((a, False), (b, True)):
        for name, v in extra.items():
            v2 = v
            if shift:
                if isinstance(v, dict):
                    v2 = {kk: _noise_after(vv, k + 1) for kk, vv in v.items()}
                elif name == "d:ban":
                    v2 = v.copy()
                    v2[:, k + 2:] = ~v2[:, k + 2:]          # the ban file of t+1 is legitimately known at t
                else:
                    v2 = _noise_after(v, k + 1)
            ctx._c[name] = v2
    rules = [("r_vix_spike_losers", {"x": 0.05}), ("r_vix_high_losers", {}), ("r_delivery_shock", {"pp": 5.0, "day": "up"}),
             ("r_delivery_shock", {"pp": 5.0, "day": "down"}), ("r_low_delivery_spike", {}), ("r_delivery_trend", {}),
             ("r_fii_flow", {"z": 1.0}), ("r_client_contrarian", {}), ("r_ban", {"event": "entry", "side": 1}),
             ("r_ban", {"event": "exit", "side": 1}), ("r_oi_quadrant", {"quad": "long_buildup", "thr": 0.1}),
             ("r_oi_quadrant", {"quad": "long_unwinding", "thr": 0.1}), ("r_deals", {"cls": "inst", "flow": "buy", "side": 1}),
             ("r_results_reaction", {"a": 0.01, "mode": "fade"}), ("r_vix_high_index", {})]
    for rule, kw in rules:
        ta, sa, ra, _ = getattr(F, rule)(a, **kw)
        tb, sb, rb, _ = getattr(F, rule)(b, **kw)
        assert (ta[:, :k + 1] == tb[:, :k + 1]).all(), rule
        assert (sa[:, :k + 1] == sb[:, :k + 1]).all(), rule
        assert np.allclose(np.nan_to_num(ra[:, :k + 1]), np.nan_to_num(rb[:, :k + 1])), rule


def _noise_after(x, k):
    x = np.array(x, dtype=float if np.asarray(x).dtype != bool else bool, copy=True)
    if x.dtype == bool:
        x[..., k:] = ~x[..., k:]
    else:
        x[..., k:] = x[..., k:] * 3.0 + 7.0
    return x


def t_minute_features_causal():
    """minute rules at session t are unchanged when every later session changes"""
    M = _minute(S=5, T=60)
    rng = np.random.default_rng(5)
    for kf in F.MIN_FIELDS:
        if kf.startswith("v_"):
            M.f[kf] = rng.uniform(1e4, 1e6, M.shape)
    M.f["o0915"] = M.f["pc"] * (1 + rng.normal(0, 0.03, M.shape))
    k = 50
    M2 = F.MinutePanel(M.syms, M.cal, {kk: v.copy() for kk, v in M.f.items()}, set(), {})
    for kk in M2.f:
        if kk != "nb":
            M2.f[kk][:, k + 1:] *= rng.uniform(0.5, 1.5, M2.f[kk][:, k + 1:].shape)
    pre = rng.random(M.shape) < 0.1
    exp = {"NIFTY": [int(M.cal[i]) for i in range(0, 60, 5)], "BANKNIFTY": [int(M.cal[i]) for i in range(2, 60, 7)]}
    a, b = F.Ctx(None, M), F.Ctx(None, M2)
    for ctx in (a, b):
        ctx._c["m:pre"] = pre
        ctx._c["m:exp"] = exp
    rules = [("r_gap_path", {"g": 0.01, "mode": "go"}), ("r_gap_path", {"g": 0.01, "mode": "fade_day"}),
             ("r_open5_shock", {"r": 0.001, "mode": "cont"}), ("r_idio_gap", {"g": 0.01}),
             ("r_early_vol", {"k": 1.0, "mode": "cont"}), ("r_down_attention", {"mode": "rev"}),
             ("r_lasthour_vol", {"k": 1.0, "mode": "fade"}), ("r_attention_swing", {"day": "up"}),
             ("r_gap_volume", {"mode": "hivol_go"}), ("r_results_gap", {"g": 0.01, "mode": "cont"}),
             ("r_expiry", {"when": "on", "side": 1})]
    for rule, kw in rules:
        ta, sa, ra, _ = getattr(F, rule)(a, **kw)
        tb, sb, rb, _ = getattr(F, rule)(b, **kw)
        assert (ta[:, :k + 1] == tb[:, :k + 1]).all() and (sa[:, :k + 1] == sb[:, :k + 1]).all(), rule


def t_oi_change_same_contracts():
    """OI change uses only contracts expiring after the session, compared with their own OI the previous session"""
    import pandas as pd
    df = pd.DataFrame([
        (1, "X", 5, 100.0), (1, "X", 10, 50.0),          # day 1: near (exp 5) + next (exp 10)
        (2, "X", 5, 90.0), (2, "X", 10, 60.0),
        (5, "X", 5, 0.0), (5, "X", 10, 80.0), (5, "X", 15, 10.0),   # expiry day 5: near excluded, new far listed
    ], columns=["day", "symbol", "exp", "oi"])
    g = F.oi_change(df).set_index("day")
    assert abs(g.loc[2, "doi"] - (150.0 / 150.0 - 1)) < 1e-12
    assert abs(g.loc[5, "doi"] - (90.0 / 60.0 - 1)) < 1e-12        # (80 + 10) vs exp-10's 60 on day 2


def t_results_timing():
    """reaction day = the first close after the result; pre-market = after the previous close, before 09:15"""
    cal = np.array([20240102, 20240103, 20240104], np.int64)
    tab = {"A": np.array([202401021600, 202401040900], np.int64), "B": np.array([202401031530], np.int64)}
    r = F.results_reaction(["A", "B"], cal, tab)
    assert r[0].tolist() == [False, True, True] and r[1].tolist() == [False, False, True]
    p = F.results_premarket(["A", "B"], cal, tab)
    assert p[0].tolist() == [False, True, True] and p[1].tolist() == [False, False, True]
    tab2 = {"A": np.array([202401040916], np.int64)}
    assert F.results_premarket(["A"], cal, tab2)[0].tolist() == [False, False, False]


def t_ml_walk_forward_causal():
    """model scores before day k are unchanged when every price after k changes (labels of later windows are never
    used for earlier predictions); the frozen model is fitted on DISCOVERY windows only"""
    T = 900
    cal = []
    d = 20120102
    import datetime as dt
    day = dt.date(2012, 1, 2)
    while len(cal) < T:
        if day.weekday() < 5:
            cal.append(int(day.strftime("%Y%m%d")))
        day += dt.timedelta(days=1)
    P = _daily(S=20, T=T, seed=7)
    P.cal = np.asarray(cal, np.int64)
    P.tix = {int(x): i for i, x in enumerate(P.cal)}
    rng = np.random.default_rng(8)
    split = int(P.cal[700])
    k = 650

    def run(Pn):
        ctx = F.Ctx(Pn, None)
        ctx._c["d:dp"] = rng_dp
        ctx._c["d:doi"] = rng_oi
        return F.ml_scores(ctx, split_day=split), ctx
    rng_dp = rng.uniform(10, 90, P.shape)
    rng_oi = rng.normal(0, 0.1, P.shape)
    s1, c1 = run(P)
    s2, c2 = run(_perturb_after(P, k))
    assert np.isfinite(s1[:, 600]).any()
    assert np.allclose(np.nan_to_num(s1[:, :k]), np.nan_to_num(s2[:, :k]))
    assert c1._c["ml_fits"]["frozen"] != [0.0] * (len(F.ML_FEATURES) + 1)


def t_shuffle_keeps_counts():
    """the shuffle control moves each symbol's trigger cells to that symbol's own eligible dates (same count)"""
    P = _daily(S=4, T=200, seed=2)
    ctx = F.Ctx(P, None)
    c = _cand(ex=("d_hold", 5), prod="fut", id="t_shuf")
    sig = _sig([0, 0, 1, 3, 3, 3], [10, 50, 60, 70, 80, 90], [1, -1, 1, 1, 1, -1], hold=[5] * 6)
    ctx._c["sig:t_shuf"] = sig
    lo, hi = int(P.cal[1]), int(P.cal[199])
    sh = R.shuffled_signals(c, ctx, lo, hi, 11)
    from collections import Counter
    assert Counter(sh["s"].tolist()) == Counter(sig["s"].tolist())
    assert sorted(sh["side"].tolist()) == sorted(sig["side"].tolist())
    assert (sh["t"] != np.sort(sig["t"])).any()
    pl = R.placebo_signals(c, ctx, lo, hi, 30, 0.5, 5, 3)
    assert len(pl["s"]) == 30 and set(pl["side"].tolist()) <= {1, -1}


def t_selection_rule():
    """DISCOVERY gates (both halves, PF, trades per half, top symbol, 5 bps) and the DSR ranking with K and the
    family cap"""
    c = _cand(ex=("d_hold", 5), prod="fut")
    mk = lambda d, sym, net: ("x", sym, d, 1, "09:15", 100.0, d, "15:29", 101.0, 1000, net + 50, 50.0, net, "time")
    tr = [mk(20130101 + i, f"S{i % 10}", 100.0) for i in range(70)] + \
         [mk(20190101 + i, f"S{i % 10}", 100.0) for i in range(70)]
    j = R.judge_discovery(c, tr, tr)
    assert j["passed"], j["why"]
    j = R.judge_discovery(c, tr[:59] + tr[70:], tr)
    assert not j["passed"] and any("H1" in w for w in j["why"])
    bad5 = [t[:12] + (-1.0,) + t[13:] for t in tr]
    assert not R.judge_discovery(c, tr, bad5)["passed"]
    one = [mk(20130101 + i, "S0" if i % 2 else f"S{i % 10}", 100.0) for i in range(140)]
    assert any("of net" in w for w in R.judge_discovery(c, one[:70] + [mk(20190101 + i, "S0", 100.0)
                                                                        for i in range(70)], tr)["why"])
    rows = [{"id": f"a{i}", "family": "A", "passed": True, "dsr": 0.9 - i / 100} for i in range(5)] + \
           [{"id": f"b{i}", "family": "B", "passed": True, "dsr": 0.5 - i / 100} for i in range(10)] + \
           [{"id": "z", "family": "C", "passed": False, "dsr": 0.99}]
    sel = R.select(rows, 50, 0.01, k=10, cap=3)
    assert sel[:3] == ["a0", "a1", "a2"] and "a3" not in sel and "z" not in sel and len(sel) == 6


def t_candidates_declared():
    """at most 200 declared candidates, unique ids, every rule exists, intraday rules use MIS, shorts never CNC"""
    cs = R.candidates()
    assert len(cs) <= 200 and len({c.id for c in cs}) == len(cs)
    for c in cs:
        assert hasattr(F, c.rule), c.id
        if c.horizon == "intraday":
            assert c.product == "mis", c.id
        sides = dict(c.params).get("side")
        if sides == -1 or dict(c.params).get("mode") in ("short", "down"):
            assert c.product != "cnc", c.id


def _weekdays(a: str, b: str) -> np.ndarray:
    import datetime as dt
    d, e = dt.date(int(a[:4]), int(a[4:6]), int(a[6:])), dt.date(int(b[:4]), int(b[4:6]), int(b[6:]))
    out = []
    while d <= e:
        if d.weekday() < 5:
            out.append(int(d.strftime("%Y%m%d")))
        d += dt.timedelta(days=1)
    return np.asarray(out, np.int64)


def _synthetic_ctx(seed=12):
    """A 12-stock daily panel 2012-2026 with a planted long-build-up drift, a 6-stock minute panel 2016-2026 and
    synthetic event tables."""
    rng = np.random.default_rng(seed)
    cal = _weekdays("20120102", "20261109")
    S, T = 12, len(cal)
    r = rng.normal(0.0, 0.015, (S, T))
    doi = rng.normal(0.0, 0.08, (S, T))
    for s_ in range(S):
        for t in range(T - 6):
            if r[s_, t] >= 0.02 and doi[s_, t] >= 0.10:
                r[s_, t + 1:t + 6] += 0.008
    c = 100 * np.exp(np.cumsum(r, axis=1))
    o = np.empty_like(c)
    o[:, 0] = c[:, 0]
    o[:, 1:] = c[:, :-1] * (1 + rng.normal(0, 0.001, (S, T - 1)))
    h, l = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996
    v = rng.uniform(1e5, 3e5, (S, T))
    nifty = {"o": 1e4 * np.exp(np.cumsum(rng.normal(0, 0.008, T))), "c": None}
    nifty["c"] = nifty["o"] * (1 + rng.normal(0, 0.004, T))
    P = F.DailyPanel([f"S{i:02d}" for i in range(S)], cal, o, h, l, c, v, np.ones((S, T), bool),
                     np.ones((S, T), bool), nifty, {"c": 15 * np.exp(np.cumsum(rng.normal(0, 0.05, T)))}, {})
    mcal = _weekdays("20160101", "20261109")
    msyms = [f"M{i}" for i in range(6)] + list(F.INDEX_MIN)
    n, Tm = len(msyms), len(mcal)
    pc = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, Tm)), axis=1))
    f = {"pc": pc, "o0915": pc * (1 + rng.normal(0, 0.02, (n, Tm))), "nb": np.full((n, Tm), 375.0)}
    for k in F.MIN_FIELDS:
        if k.startswith("v_"):
            f[k] = rng.uniform(1e4, 1e6, (n, Tm))
        elif k not in f:
            f[k] = f["o0915"] * (1 + rng.normal(0, 0.01, (n, Tm)))
    M = F.MinutePanel(msyms, mcal, f, set(), {})
    ctx = F.Ctx(P, M)
    ctx._c.update({
        "d:dp": rng.uniform(10, 90, (S, T)), "d:doi": doi, "d:ban": rng.random((S, T)) < 0.03,
        "d:deals": {"inst": np.where(rng.random((S, T)) < 0.02, rng.normal(0, 30, (S, T)), 0.0),
                    "other": np.where(rng.random((S, T)) < 0.02, rng.normal(0, 30, (S, T)), 0.0)},
        "d:part": {k: rng.uniform(1e5, 3e5, T) for k in ("fii_fut_idx_long", "fii_fut_idx_short",
                                                       "client_fut_idx_long", "client_fut_idx_short")},
        "d:react": rng.random((S, T)) < 0.02,
        "d:ca": {"splitbonus": [(int(rng.integers(S)), int(cal[t]), int(cal[t + 10])) for t in range(20, T - 40, 25)],
                 "dividend": [(int(rng.integers(S)), int(cal[t]), int(cal[t + 12]), 3.0) for t in range(20, T - 40, 9)]},
        "m:pre": rng.random((n, Tm)) < 0.03,
        "m:exp": {"NIFTY": [int(x) for x in mcal[::5]], "BANKNIFTY": [int(x) for x in mcal[2::5]]},
    })
    return ctx


def t_pipeline_end_to_end_synthetic():
    """the whole study on synthetic data in a temporary directory: availability counts, pre-registration (sha256,
    repo copy), DISCOVERY once (refuses to recompute), selection, controls, VALIDATION once, the one-look FINAL ledger,
    the proof gates and the trials rows; a planted edge is found and carried; changing the code after the
    pre-registration is refused"""
    import tempfile
    from . import protocol as P_
    names = ("DG", "PREREG", "PREREG_SHA_FILE", "REPO_PREREG", "DATA_CHECK", "DISC_OUT", "VAL_OUT", "FIN_OUT",
             "FINAL_LEDGER", "PROGRESS", "N_SHUFFLE", "N_PLACEBO", "BOOT_N", "validation_split")
    saved = {k: getattr(R, k) for k in names}
    saved_ctx = R._CTX.get("ctx")
    tmp = tempfile.mkdtemp(prefix="dg_test_")
    try:
        R.DG = tmp
        for k, fn in (("PREREG", "PRE.json"), ("PREREG_SHA_FILE", "PRE.sha256"), ("REPO_PREREG", "repo_pre.json"),
                      ("DATA_CHECK", "dc.json"), ("DISC_OUT", "disc.json"), ("VAL_OUT", "val.json"),
                      ("FIN_OUT", "fin.json"), ("FINAL_LEDGER", "fl.json"), ("PROGRESS", "progress.txt")):
            setattr(R, k, os.path.join(tmp, fn))
        R.N_SHUFFLE, R.N_PLACEBO, R.BOOT_N = 1, 3, 50
        ctx = _synthetic_ctx()
        R._CTX["ctx"] = ctx
        cal3 = [int(x) for x in ctx.daily.cal if 20231110 <= x <= 20261109]
        R.validation_split = lambda: (P_.split_sessions(cal3), cal3)
        dc = R.data_check()
        assert dc["n_search"] >= 5
        R.preregister()
        with open(R.PREREG_SHA_FILE) as f:
            assert f.read().strip() == R._sha(R.REPO_PREREG)
        disc = R.discovery()
        ids = [c.id for c in R.search_space()]
        assert [r["id"] for r in disc["rows"]] == ids
        assert "dg_h_long_buildup_10_h5" in disc["passed"] and disc["carried"]
        assert len(disc["carried"]) <= R.K_CARRY
        again = R.discovery()
        assert again["run"] == disc["run"]
        ctl = R.controls_discovery()
        assert ctl["planted"]["ctl_planted_daily_h5_p100"]["passed"]
        val = R.validate()
        assert {r["id"] for r in val["rows"]} == set(disc["carried"])
        fin = R.final()
        led = os.path.join(tmp, "trials.csv")
        pr = R.proof(led)
        assert set(pr["candidates"]) == set(disc["carried"])
        rows = R.TR.TrialsLedger(led).read()
        assert sum(1 for x in rows if x["stage"] == "datagen-discovery") == len(ids)
        assert sum(1 for x in rows if x["stage"] == "screen") == len(val["rows"])
        assert sum(1 for x in rows if x["stage"] == "final") == len(fin["rows"])
        with open(R.FINAL_LEDGER) as f:
            assert len(json.load(f)) == len(fin["rows"])
        doc = json.load(open(R.PREREG))
        doc["code_sha256"]["datagen_run.py"] = "0" * 64
        txt = json.dumps(doc)
        open(R.PREREG, "w").write(txt)
        try:
            R.check_prereg()
            raise AssertionError("a changed pre-registration was accepted")
        except SystemExit:
            pass
    finally:
        for k, v in saved.items():
            setattr(R, k, v)
        if saved_ctx is None:
            R._CTX.pop("ctx", None)
        else:
            R._CTX["ctx"] = saved_ctx
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def t_prereg_matches_code():
    """the committed pre-registration lists exactly the kept candidates with matching fingerprints and code"""
    if not os.path.exists(R.REPO_PREREG):
        return
    with open(R.REPO_PREREG) as f:
        pre = json.load(f)
    fp = {c.id: c.spec().fingerprint for c in R.candidates()}
    assert pre["n_search"] == len(pre["candidates"]) <= 200
    for row in pre["candidates"]:
        assert fp[row["id"]] == row["fingerprint"], row["id"]
    assert pre["selection"]["K"] == R.K_CARRY and pre["code_sha256"]["datagen_features.py"] == \
        R.code_sha()["datagen_features.py"]
    assert pre["code_sha256"]["datagen_run.py"] == R.code_sha()["datagen_run.py"]


TESTS = [(f.__name__, f) for f in (
    t_daily_fill_slippage_cost_hold, t_daily_period_end_and_entry_window, t_daily_cap_busy_gates,
    t_minute_executions_special_dirty, t_daily_features_causal, t_minute_features_causal, t_oi_change_same_contracts,
    t_results_timing, t_ml_walk_forward_causal, t_shuffle_keeps_counts, t_selection_rule, t_candidates_declared,
    t_pipeline_end_to_end_synthetic, t_prereg_matches_code)]


if __name__ == "__main__":
    bad = 0
    for n, f in TESTS:
        try:
            f()
            print("PASS", n)
        except Exception as exc:
            bad += 1
            import traceback
            traceback.print_exc()
            print("FAIL", n, exc)
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
