"""
stratlab/tests_r6.py — tests for the round-6 GitHub-sourced strategies: formulas in r6_ind against hand-computed
values (TD Sequential counts and perfection, Williams VIX Fix, Dynamic Breakout II lookback, pysystemtrade EWMAC /
mr_wings, Meeus moon phases, R-Breaker pivots and state machine, Clenow momentum, residual-momentum score, Alpha101
#9 #12 #33 #54 #101), the causality of every point function, the portfolio engine (next-open execution, costs,
shorts, stale names, schedules) and a look-ahead check of every portfolio weights function and of the intraday
specs. Run by test_pipeline.py (section "STRATLAB: round-6 GitHub strategies") or directly.
"""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np

from . import data as D
from . import indicators as I
from . import r6_ind as R
from . import sim


def _close(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)))


def _rw(n, seed, s0=100.0, sig=0.015):
    rng = np.random.default_rng(seed)
    c = s0 * np.exp(np.cumsum(rng.normal(0, sig, n)))
    o = np.concatenate(([s0], c[:-1])) * np.exp(rng.normal(0, sig / 3, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, sig / 2, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sig / 2, n)))
    v = rng.integers(1000, 9000, n).astype(float)
    return o, h, l, c, v


# ── TD Sequential ───────────────────────────────────────────────────────────
def t_td_counts_hand():
    c = np.arange(100.0, 87.0, -1.0)                 # 13 falling closes: count = i - 3 from i = 4
    l = c - 1.0
    h = c + 1.0
    sb, ss, xl, xh = R.td_counts(c, h, l)
    assert list(sb) == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], sb
    assert not ss.any() and not xh.any()
    assert xl[11] and xl[12] and not xl[10], xl      # bar 8 (i=11) low 88 < min(low6 90, low7 89); bar 9 inherits
    assert R.td_at(c, h, l, 12) == (9, 0, True, False)
    # perfection fix: bar-8 / bar-9 lows below bar 7's low but NOT below bar 6's low
    l2 = l.copy()
    l2[9] = 80.0                                      # bar 6 low very low
    _sb, _ss, xl_fix, _ = R.td_counts(c, h, l2)
    _sb, _ss, xl_src, _ = R.td_counts(c, h, l2, as_coded=True)
    assert not xl_fix[12] and xl_src[12], (xl_fix, xl_src)   # the source's "below either" fires, DeMark's does not
    assert R.td_at(c, h, l2, 12)[2] is False


def t_td_point_equals_arrays_and_causal():
    o, h, l, c, _v = _rw(700, 3, sig=0.01)
    sb, ss, xl, xh = R.td_counts(c, h, l)
    for i in range(0, 700, 7):
        assert R.td_at(c, h, l, i) == (sb[i], ss[i], bool(xl[i]), bool(xh[i])), i
        assert R.td_at(c[:i + 1], h[:i + 1], l[:i + 1]) == R.td_at(c, h, l, i), i
    assert sb.max() >= 9 and xl.any()


# ── Williams VIX Fix ────────────────────────────────────────────────────────
def t_wvf_hand():
    c = np.array([10.0, 11.0, 12.0, 11.0, 9.0])
    l = np.array([9.5, 10.5, 11.0, 10.0, 8.0])
    wvf, sig = R.wvf_signal(c, l, pd_=3, bbl=2, mult=1.0, lb=2, ph=0.85)
    # highest(close, 3) = -, -, 12, 12, 12 ; wvf = (hc - low) / hc * 100
    assert _close(wvf, [np.nan, np.nan, 100 / 12, 200 / 12, 400 / 12]), wvf
    # BB(2, 1.0): mean +- pop std of the last 2 wvf; rangeHigh = max(last 2) * 0.85
    up3 = 150 / 12 + 50 / 12
    assert abs(up3 - (np.mean([100 / 12, 200 / 12]) + np.std([100 / 12, 200 / 12]))) < 1e-12
    assert sig[3] and sig[4] and not sig[2] and not sig[0], sig       # 200/12 >= 200/12 * 0.85 etc.
    wvf2, sig2 = R.wvf_signal(c, l, pd_=3, bbl=2, mult=1.0, lb=2, ph=1.5)
    assert sig2[3] and sig2[4]                        # BB leg alone: wvf equals the upper band at the high


# ── Dynamic Breakout II ─────────────────────────────────────────────────────
def t_dbo2_lookback_hand_and_resume():
    o, h, l, c, _v = _rw(200, 11)
    N = R.dbo2_lookback(c)
    assert (N[:30] == 0).all() and N[30] == 20 and ((N[31:] >= 20) & (N[31:] <= 60)).all()
    v_t, v_y = np.std(c[2:32]), np.std(c[1:31])
    assert N[31] == int(min(max(round(20 * (1 + (v_t - v_y) / v_t)), 20), 60)), (N[31], v_t, v_y)
    assert (R.dbo2_lookback(c, n0=N[:120]) == N).all()            # resumed prefix = full recursion
    assert (R.dbo2_lookback(c[:150]) == N[:150]).all()           # causal
    lv = R.dbo2_at(c, h, l, N, 150)
    n = int(N[150])
    assert lv["n"] == n and lv["hh"] == h[150 - n:150].max() and abs(lv["ma"] - c[151 - n:151].mean()) < 1e-12
    assert abs(lv["ub"] - lv["ma"] - 2 * c[151 - n:151].std()) < 1e-9


# ── pysystemtrade ───────────────────────────────────────────────────────────
def t_ewmac_and_mr_wings_hand():
    price = np.array([1.0, 2.0, 4.0])
    vol = np.array([1.0, 1.0, 2.0])
    # pandas ewm(span, adjust=True): weights (1-a)^k, a = 2 / (span + 1)
    def ew(x, span):
        a = 2.0 / (span + 1)
        w = (1 - a) ** np.arange(len(x))[::-1]
        return float((w * x).sum() / w.sum())
    exp = [(ew(price[:k + 1], 4) - ew(price[:k + 1], 16)) / vol[k] for k in range(3)]
    assert _close(R.ewmac(price, vol, 4, 16), exp, 1e-12), (R.ewmac(price, vol, 4, 16), exp)
    o, h, l, c, _v = _rw(600, 5)
    f = R.mr_wings(c)
    vol = R.robust_vol(c)
    e = R.ewmac(c, vol, 4, 16)
    import pandas as pd
    sd = pd.Series(e).rolling(5000, min_periods=3).std().to_numpy()
    nz = np.isfinite(sd) & (np.abs(e) >= 3 * sd)
    assert _close(f[nz], -e[nz]) and (f[~nz & np.isfinite(sd)] == 0).all()
    assert _close(R.mr_wings(c[:400]), f[:400], 1e-9)                  # causal
    vv = pd.Series(c).diff().ewm(span=35, min_periods=10).std().to_numpy()
    assert _close(vol[10:99], vv[10:99], 1e-12)                          # floor not active before 100 obs


# ── Moon phases (Meeus ch. 49) ──────────────────────────────────────────────
def t_moon_phases_meeus_and_usno():
    assert abs(R.moon_phase_jde(-283) - 2443192.65118) < 2e-5           # Meeus example 49.a (new moon 1977 Feb)
    assert abs(R.moon_phase_jde(544.75) - 2467636.49186) < 2e-5         # Meeus example 49.b (last quarter 2044 Jan)
    usno = {297.75 - 1.0: (2024, 1, 4, 3, 30), 297.0: (2024, 1, 11, 11, 57), 297.25: (2024, 1, 18, 3, 52),
            297.5: (2024, 1, 25, 17, 54)}
    for k, (y, m, d, hh, mm) in usno.items():
        got = R.jd_to_datetime(R.moon_phase_jde(k) - R.DELTA_T_DAYS)
        assert abs((got - _dt.datetime(y, m, d, hh, mm)).total_seconds()) < 120, (k, got)
    assert R.lunar_state(20240105) == "LQ" and R.lunar_state(20240117) == "LQ"
    assert R.lunar_state(20240118) == "FQ"                              # FQ at 03:52 UTC, before the 10:00 UTC close


# ── R-Breaker ───────────────────────────────────────────────────────────────
def t_rbreaker_levels_and_flips():
    lv = R.rbreaker_levels(110.0, 90.0, 100.0)
    assert lv == {"pivot": 100.0, "r1": 110.0, "r2": 120.0, "r3": 130.0, "s1": 90.0, "s2": 80.0, "s3": 70.0}, lv
    c = np.array([125.0, 131.0, 125.0, 105.0, 101.0, 135.0])
    hi = np.maximum.accumulate(np.array([126.0, 132.0, 126.0, 126.0, 102.0, 136.0]))
    lo = np.minimum.accumulate(np.array([124.0, 124.0, 124.0, 104.0, 100.0, 100.0]))
    enter, flip = R.rbreaker_day(c, hi, lo, lv)
    # 131 > r3 -> long; 105 with day high 132 > r2 and < r1 -> short; 135 > r3 -> long again
    assert list(enter) == [0, 1, 0, -1, 0, 1] and list(flip) == [False, False, False, True, False, True], (enter, flip)


# ── cross-sectional formulas ────────────────────────────────────────────────
def t_clenow_residual_alpha_hand():
    c = 100 * np.exp(0.001 * np.arange(90))
    m = R.clenow_momentum(c[:, None])
    assert abs(m[0] - (math.exp(0.252) - 1) * 100) < 1e-9, m
    rng = np.random.default_rng(2)
    x = rng.normal(0.01, 0.05, 36)
    X = np.column_stack([np.ones(36), x])
    e = rng.normal(0, 0.03, 36)
    e = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]                     # orthogonal to [1, x]: OLS residuals = e
    y = 0.004 + 1.3 * x + e
    w = e[-12:-1]
    assert abs(R.residual_score(y, x) - w.sum() / w.std()) < 1e-9
    assert np.isnan(R.residual_score(y[:35], x[:35]))
    O = np.array([[10.0, 20.0]] * 6)
    H = np.array([[12.0, 21.0]] * 6)
    L = np.array([[9.0, 19.0]] * 6)
    C = np.array([[10.0, 20.0], [10.5, 20.0], [11.0, 20.5], [11.5, 20.0], [12.0, 19.0], [11.0, 20.0]])
    V = np.array([[100.0, 100.0]] * 5 + [[150.0, 90.0]])
    a = R.alpha101("a101", O, H, L, C, V, 5)
    assert _close(a, [1.0 / 3.001, 0.0 / 2.001]), a
    assert _close(R.alpha101("a33", O, H, L, C, V, 5), [-(1 - 10 / 11), -(1 - 20 / 20)])
    # a12: stock 0 volume up, close down 1 -> +1 * 1 = 1; stock 1 volume down, close up 1 -> -1 * -1 = 1
    assert _close(R.alpha101("a12", O, H, L, C, V, 5), [1.0, 1.0])
    a54 = R.alpha101("a54", O, H, L, C, V, 5)
    assert _close(a54, [-(9 - 11) * 10 ** 5 / ((9 - 12) * 11 ** 5), -(19 - 20) * 20 ** 5 / ((19 - 21) * 20 ** 5)])
    # a9: stock 0 deltas .5,.5,.5,.5,-1 (min < 0, max > 0) -> -delta = 1 ; stock 1 deltas 0,.5,-.5,-1,1 -> -1
    assert _close(R.alpha101("a9", O, H, L, C, V, 5), [1.0, -1.0])
    C2 = C.copy()
    C2[:, 0] = [10, 11, 12, 13, 14, 15]
    assert R.alpha101("a9", O, H, L, C2, V, 5)[0] == 1.0                 # all deltas > 0: momentum, +delta
    assert _close(R.pct_rank_desc(np.array([3.0, 1.0, 2.0, np.nan])), [1 / 3, 1.0, 2 / 3, np.nan])


# ── ghost trader ────────────────────────────────────────────────────────────
def t_ghost_replay_rules_and_causal():
    o, h, l, c, _v = _rw(900, 7, sig=0.012)
    enter, ex = R.ghost_replay(h, l, c)
    assert (enter != 0).sum() >= 5, (enter != 0).sum()
    e3, e21, r9 = (R.K.ema_sma_seed(c, 3), R.K.ema_sma_seed(c, 21), R.K.rsi(c, 9))
    pos = 0
    for i in range(len(c)):
        if enter[i] == 1:
            assert pos == 0 and e3[i] > e21[i] and r9[i] < 70 and h[i] > h[i - 1]
        if enter[i] == -1:
            assert pos == 0 and e3[i] < e21[i] and r9[i] > 30 and l[i] < l[i - 1]
        if ex[i]:
            assert pos != 0
            assert (c[i] <= l[i - 21:i].min()) if pos > 0 else (h[i] >= h[i - 21:i].max())
        pos = enter[i] if enter[i] else (0 if ex[i] else pos)
    e2, x2 = R.ghost_replay(h[:600], l[:600], c[:600])
    assert (e2 == enter[:600]).all() and (x2 == ex[:600]).all()
    ea, _xa = R.ghost_replay(h, l, c, as_coded_window=True)
    assert (ea != enter).any()                                           # the window fix changes the trades


# ── portfolio engine ────────────────────────────────────────────────────────
def _days(n, start=_dt.date(2024, 1, 1)):
    out, x = [], start
    while len(out) < n:
        if x.weekday() < 5:
            out.append(int(x.strftime("%Y%m%d")))
        x += _dt.timedelta(days=1)
    return np.array(out, np.int64)


def t_schedules():
    from . import r6_portfolio as RP
    d = _days(300)
    wk = RP.schedule("weekly", d)
    assert all(_dt.date(d[i] // 10000, d[i] // 100 % 100, d[i] % 100).weekday() == 4 for i in wk[:-1])
    q = RP.schedule("quarterly", d)
    assert [int(d[i]) for i in q] == [20240329, 20240628, 20240930, 20241231]
    qn = RP.schedule("q_nov", d)
    assert [int(d[i]) for i in qn] == [20240229, 20240531, 20240830, 20241129]
    assert RP.schedule("daily", d) == list(range(len(d) - 1))


def t_engine_long_short_costs_stale():
    from . import r6_portfolio as RP
    d = _days(70)
    n = len(d)
    t0 = RP.schedule("monthly", d)[0]                 # 2024-01-31: executed at the 2024-02-01 open
    c = np.ones((n, 3)) * 100.0
    o = np.ones((n, 3)) * 100.0
    c[t0 + 2:, 0] = o[t0 + 2:, 0] = 110.0             # stock 0 +10 % from the session after the execution
    c[t0 + 2:, 1] = o[t0 + 2:, 1] = 90.0              # stock 1 -10 %
    o[t0 + 1, 2] = c[t0 + 1, 2] = np.nan              # stock 2 does not trade on the execution day
    p = RP.XPanel(d, ["A", "B", "C"], c, o, np.full(n, 1000.0))

    def w(p_, t, ctx, prm):
        return {0: 0.5, 1: -0.5, 2: 0.2} if not ctx.state.get("done") and ctx.state.update(done=1) is None else None
    sp = RP.PSpec("T", "t", w, {}, "monthly", "futures")
    r = RP.backtest(p, sp, int(d[0]))
    assert r["first_decision"] == int(d[t0]) == 20240131
    buys = [x for x in r["trades"]]
    assert {x[1] for x in buys} == {"A", "B"} and all(x[0] == int(d[t0 + 1]) for x in buys)   # C stale: skipped
    cf = RP.trade_costs("futures")
    cost = sum(sum(cf(5e5, s)) for s in ("BUY", "SELL"))
    eq_end = r["equity"][-1]
    assert abs(eq_end - (1e6 + 5e5 * 0.10 + 5e5 * 0.10 - cost)) < 1e-6, (eq_end, cost)    # long +10 %, short of -10 %
    assert abs(r["gross"][-1] - (5000 * 110 + 5000 * 90) / eq_end) < 1e-9
    cnc = RP.trade_costs("cnc")
    assert cnc(1e5, "BUY") == RP.FM.trade_costs(1e5, "BUY")


def t_portfolio_weights_causal():
    """Every portfolio weights function gives the same target at t when every row after t is replaced by noise."""
    from . import r6_portfolio as RP
    rng = np.random.default_rng(9)
    d = _days(1300, _dt.date(2019, 1, 1))
    N, M = len(d), 30
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, (N, M)), axis=0))
    o = c * np.exp(rng.normal(0, 0.004, (N, M)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, (N, M))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, (N, M))))
    v = rng.integers(10_000, 90_000, (N, M)).astype(float)
    nifty = 1000 * np.exp(np.cumsum(rng.normal(0.0008, 0.005, N)))
    syms = [f"S{j:02d}" for j in range(M)]
    full = RP.XPanel(d, syms, c, o, nifty, high=h, low=l, vol=v)
    sched = {k: RP.schedule(k, d) for k in ("monthly", "weekly", "quarterly", "q_nov", "daily")}
    etf = RP.XPanel(d, ["N", "G"], c[:, :2], o[:, :2], nifty)
    for sp in RP.SPECS:
        P0 = etf if sp.panel == "etf" else full
        ts = [t for t in sched[sp.schedule] if t > 800][:4]
        for t in ts:
            cut = {}
            for k in ("close", "open", "high", "low", "vol"):
                a = getattr(P0, k).copy()
                if a is not None:
                    a[t + 1:] = a[t + 1:] * np.exp(rng.normal(0, 0.2, a[t + 1:].shape))
                cut[k] = a
            nf = P0.nifty.copy()
            nf[t + 1:] *= 1.3
            P1 = RP.XPanel(d, P0.symbols, cut["close"], cut["open"], nf, high=cut["high"], low=cut["low"], vol=cut["vol"])
            mk = lambda: RP.Ctx(t, int(d[t + 1]) // 100, {}, 1e6, {})
            w0, w1 = sp.weights(P0, t, mk(), sp.params), sp.weights(P1, t, mk(), sp.params)
            assert w0 == w1, (sp.id, t)
        assert any(sp.weights(P0, t, RP.Ctx(t, int(d[t + 1]) // 100, {}, 1e6, {}), sp.params) for t in ts), sp.id


def t_january_barometer_hand():
    from . import r6_portfolio as RP
    d = np.array([20221230, 20230131, 20230201, 20231229, 20240131, 20240201, 20241231, 20250102], np.int64)
    nf = np.array([100.0, 110.0, 110.0, 121.0, 115.0, 115.0, 92.0, 92.0])
    rep = RP.january_barometer(d, nf)
    y = {r["year"]: r for r in rep["rows"]}
    assert y[2023]["invested"] and not y[2024]["invested"]
    assert abs(rep["strategy_growth"] - 1.1 * 1.1 * (115 / 121)) < 1e-12, rep
    assert abs(rep["buy_hold_growth"] - 92 / 100) < 1e-12


# ── intraday specs: dense look-ahead audit ──────────────────────────────────
def t_r6_intraday_audit_clean():
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    rng = np.random.default_rng(6)
    n, nd = 375, 20
    days = [20260100 + k for k in range(1, nd + 1)]
    date = np.repeat(np.array(days, np.int32), n)
    t = np.tile(555 + np.arange(n), nd).astype(np.int16)
    mu = np.repeat(np.where(np.arange(nd) % 2 == 0, 0.00015, -0.00015), n)   # trend days reach r3 / s3
    c = 20000.0 * np.exp(np.cumsum(rng.normal(0, 0.0012, n * nd) + mu))
    o = np.concatenate(([20000.0], c[:-1]))
    h = np.maximum(o, c) * 1.0003
    l = np.minimum(o, c) * 0.9997
    sd = D.from_arrays("RW", date, t, o, h, l, c, np.zeros(len(c)))
    sessions = I.build_sessions(sd)
    specs = [sp for i, sp in REGISTRY.items() if i.startswith("r6_gh_") and sp.horizon == "intraday"]
    assert len(specs) == 4, [s.id for s in specs]
    for sp in specs:
        k = 0
        for s in sessions[2:]:
            tr, _ = sim.sim_session(sp, s)
            k += len(tr)
            for x in tr:
                assert x[4] < "15:15" and x[7] <= "15:15", (sp.id, x)
            bad = sim.audit_lookahead(sp, s, cuts=(20, 46, 75, 110, 170, 226, 290, 330))
            assert bad is None, f"{sp.id}: {bad}"
        assert k > 0, sp.id


def t_r6_daily_specs_run():
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    o, h, l, c, v = _rw(420, 21)
    dd = _days(420, _dt.date(2023, 1, 2))
    date = np.repeat(dd.astype(np.int32), 2)
    tt = np.tile(np.array([555, 929]), 420).astype(np.int16)
    oo = np.column_stack([o, c]).ravel()                 # bar 1 carries open / high / low, bar 2 the close
    hh = np.column_stack([h, c]).ravel()
    ll = np.column_stack([l, c]).ravel()
    cc = np.column_stack([o, c]).ravel()
    sd = D.from_arrays("SYN", date, tt, oo, hh, ll, cc, np.repeat(v, 2))
    specs = [sp for i, sp in REGISTRY.items() if i.startswith("r6_gh_") and sp.horizon == "positional"]
    assert len(specs) == 10, len(specs)
    n_tr = 0
    for sp in specs:
        tr, _sk = sim.sim_positional(sp, sd, 0, 99999999)
        n_tr += len(tr)
        for x in tr:
            assert x[6] >= x[2]
    assert n_tr > 10, n_tr


TESTS = [
    ("r6_ind: TD Sequential counts and perfection (fixed: below BOTH bar-6/7 lows) by hand", t_td_counts_hand),
    ("r6_ind: TD point evaluation = full arrays, causal", t_td_point_equals_arrays_and_causal),
    ("r6_ind: Williams VIX Fix against hand-computed values", t_wvf_hand),
    ("r6_ind: Dynamic Breakout II adaptive lookback (hand step, resume, causal) and levels", t_dbo2_lookback_hand_and_resume),
    ("r6_ind: pysystemtrade EWMAC by hand, mr_wings zeroing / sign, robust vol", t_ewmac_and_mr_wings_hand),
    ("r6_ind: Meeus moon phases (examples 49.a / 49.b, USNO Jan 2024) and the lunar state", t_moon_phases_meeus_and_usno),
    ("r6_ind: R-Breaker pivots and flip state machine by hand", t_rbreaker_levels_and_flips),
    ("r6_ind: Clenow momentum, residual-momentum score, Alpha101 #9/#12/#33/#54/#101 by hand", t_clenow_residual_alpha_hand),
    ("r6_ind: ghost trader replay obeys its rules (fixed stop window), causal", t_ghost_replay_rules_and_causal),
    ("r6_portfolio: weekly / quarterly / Q-NOV / daily schedules", t_schedules),
    ("r6_portfolio: next-open execution, long / short P&L, futures costs, stale name skipped", t_engine_long_short_costs_stale),
    ("r6_portfolio: every weights function is causal (rows after t replaced by noise)", t_portfolio_weights_causal),
    ("r6_portfolio: January barometer by hand", t_january_barometer_hand),
    ("r6_github: intraday specs run and pass the dense look-ahead audit", t_r6_intraday_audit_clean),
    ("r6_github: daily specs run on synthetic bars", t_r6_daily_specs_run),
]


if __name__ == "__main__":
    bad = 0
    for name, f in TESTS:
        try:
            f()
            print("ok  ", name)
        except Exception as exc:            # pragma: no cover
            bad += 1
            import traceback
            print("FAIL", name, repr(exc)[:400])
            traceback.print_exc(limit=3)
    raise SystemExit(1 if bad else 0)
