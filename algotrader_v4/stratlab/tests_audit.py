"""
stratlab/tests_audit.py — tests of the Nov 2026 simulator audit and the corrected-data overlay, on SYNTHETIC data
(plus one optional regression on the real 3-year bars when they are on disk): the independent engine (refsim.py)
agrees with sim.py fill for fill on random entries (stop first on a two-sided bar, a gap through the stop fills at
the open, a target needs a tick through, time stops, square-off); refsim's own Zerodha table equals costs.py on
random order values; the positional engines agree; the mirror and one-bar-shift wrappers do what they say; the
aggregated screen equals protocol.screen; the corrected-view maps preserve intraday returns (ratio), repair them
(affine), drop a stale first bar and excluded sessions; install()/uninstall() route data.load_symbol.
Run by test_pipeline.py (section "STRATLAB: simulator audit + corrected data") or `python -m stratlab.tests_audit`.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

from . import data as D
from . import refsim as R
from . import sim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402


def _days(n, start=dt.date(2025, 7, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(int(d.strftime("%Y%m%d")))
        d += dt.timedelta(days=1)
    return out


def _synth(n_days=5, seed=1, px=400.0, vol=0.0015):
    rng = np.random.default_rng(seed)
    days = _days(n_days)
    T = np.arange(555, 930)
    date, t = np.repeat(days, len(T)), np.tile(T, n_days)
    c = px * np.exp(np.cumsum(rng.normal(0, vol, len(date))))
    o = np.r_[px, c[:-1]] * np.exp(rng.normal(0, vol / 3, len(c)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol, len(c))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol, len(c))))
    o, h, lo, c = (np.round(x / 0.05) * 0.05 for x in (o, h, lo, c))
    h = np.maximum.reduce([h, o, c])
    lo = np.minimum.reduce([lo, o, c])
    v = rng.integers(1000, 9000, len(c)).astype(float)
    return D.from_arrays("SYN", date, t, o, h, lo, c, v, min_bars=200)


def _ref_sessions(sd):
    out = []
    for k in range(sd.n_sessions):
        a, b = sd.starts[k], sd.ends[k]
        out.append(R.Sess(int(sd.days[k]), sd.t[a:b].astype(int), sd.o[a:b], sd.h[a:b], sd.l[a:b], sd.c[a:b],
                          sd.v[a:b], k))
    return out


def t_refsim_matches_sim_fill_for_fill():
    """2,000 random entries (both sides, stop, target or rr, time stops) fill identically in refsim and sim.py"""
    from .api import Entry
    from .indicators import build_sessions
    rng = np.random.default_rng(7)
    sd = _synth(8, seed=11)
    S, Rs = build_sessions(sd), _ref_sessions(sd)
    n_ok = n = 0
    for _ in range(2000):
        k = int(rng.integers(0, len(S)))
        s, rs = S[k], Rs[k]
        i = int(rng.integers(0, len(s.t) - 2))
        side = int(rng.choice([1, -1]))
        c = float(s.c[i])
        stop = c * (1 - side * rng.uniform(0.001, 0.01))
        tgt = c * (1 + side * rng.uniform(0.001, 0.02)) if rng.random() < 0.5 else None
        rr = None if tgt is not None else float(rng.choice([1.0, 1.5, 2.0]))
        hold = int(rng.choice([0, 5, 20])) or None
        a = sim.fill_intraday(s, Entry(i, side, stop, tgt, rr, hold), 2e-4)
        b = R.intraday_one(rs, R.E(i, side, stop, tgt, rr, hold), 2e-4)
        n += 1
        if a is None or b is None:
            assert a is None and b is None, (a, b)
            n_ok += 1
            continue
        assert a[0] == b[1] and abs(a[1] - b[2]) < 1e-9 and abs(a[2] - b[3]) < 1e-9 and a[3] == b[4], (a, b)
        n_ok += 1
    assert n_ok == n


def t_tie_gap_and_tick_rules():
    """hand-built bars: a bar touching stop and target exits at the stop; an open through the stop fills at the
    open; a bare touch of the target is not a fill, one tick through is"""
    days = _days(1)
    T = np.arange(555, 930)
    n = len(T)
    o = np.full(n, 100.0); h = np.full(n, 100.1); lo = np.full(n, 99.9); c = np.full(n, 100.0)
    h[3], lo[3] = 101.5, 98.5                      # two-sided bar
    sd = D.from_arrays("SYN", np.repeat(days, n), T, o, h, lo, c, np.full(n, 1e3), min_bars=200)
    rs = _ref_sessions(sd)[0]
    r = R.intraday_one(rs, R.E(0, 1, 99.0, target=101.0), 0.0)
    assert r[4] == "stop" and r[1] == 3 and abs(r[3] - 99.0) < 1e-9
    o2, lo2, h2 = o.copy(), np.full(n, 99.9), np.full(n, 100.1)
    o2[4], lo2[4], h2[4] = 98.0, 97.9, 98.1         # gap through the stop
    sd2 = D.from_arrays("SYN", np.repeat(days, n), T, o2, h2, lo2, c, np.full(n, 1e3), min_bars=200)
    r = R.intraday_one(_ref_sessions(sd2)[0], R.E(0, 1, 99.0, target=103.0), 0.0)
    assert r[4] == "stop" and abs(r[3] - 98.0) < 1e-9
    h3, lo2b = np.full(n, 100.1), np.full(n, 99.9)
    h3[3] = 101.0                                   # bare touch of 101
    sd3 = D.from_arrays("SYN", np.repeat(days, n), T, o, h3, lo2b, c, np.full(n, 1e3), min_bars=200)
    r = R.intraday_one(_ref_sessions(sd3)[0], R.E(0, 1, 99.0, target=101.0), 0.0)
    assert r[4] == "squareoff"
    h3[3] = 101.05                                  # one tick through
    sd4 = D.from_arrays("SYN", np.repeat(days, n), T, o, h3, lo2b, c, np.full(n, 1e3), min_bars=200)
    r = R.intraday_one(_ref_sessions(sd4)[0], R.E(0, 1, 99.0, target=101.0), 0.0)
    assert r[4] == "target" and abs(r[3] - 101.0) < 1e-9


def t_zerodha_table_equals_costs_py():
    """refsim's own Zerodha table (zerodha.com/charges, fetched 2026-11-15) = costs.round_trip_cost, every product"""
    rng = np.random.default_rng(3)
    for _ in range(3000):
        bv, sv = float(rng.uniform(1e3, 5e6)), float(rng.uniform(1e3, 5e6))
        for prod, kind in (("MIS", "equity_mis"), ("CNC", "equity_cnc"), ("FUT", "future")):
            assert abs(R.zerodha_charges(bv, sv, prod) - round_trip_cost(bv, sv, kind)) <= 0.011, (prod, bv, sv)
    # Rs 1 lakh intraday round trip ~ Rs 82-83 (the figure quoted in the README)
    assert 82.0 < R.zerodha_charges(1e5, 1e5, "MIS") < 83.5


def t_positional_engines_agree():
    """swing next-open fills, % stop / target and max-hold close: refsim.run_positional = sim.sim_positional"""
    from .api import Spec, Swing
    sd = _synth(60, seed=5, vol=0.002)
    ss = _ref_sessions(sd)
    D_ = R.daily(ss)

    def rule(d, p):
        return Swing(1, stop_pct=2.0, target_pct=3.0, max_hold=4) if len(d) > 3 and d.c[-1] > d.c[-3] else None
    sp = Spec("T_POS", "swing", "t", "test", rule, {}, horizon="positional")
    a, _ = sim.sim_positional(sp, sd, int(sd.days[0]), int(sd.days[-1]), None, None, 2e-4)
    c = D_["c"].to_numpy()

    def dec(d):
        return R.P(1, stop_pct=2.0, target_pct=3.0, max_hold=4) if d >= 3 and c[d] > c[d - 2] else None
    b = R.run_positional("T_POS", "SYN", ss, dec, int(sd.days[0]), int(sd.days[-1]), 2e-4, "CNC")
    assert len(a) == len(b) and len(a) > 5
    for x, y in zip(a, b):
        assert x[2] == y["date"] and x[6] == y["exit_date"] and abs(x[8] - y["exit"]) < 1e-3 and \
            abs(x[12] - y["net"]) < 0.02, (x, y)


def t_mirror_and_shift_wrappers():
    """mirror_spec flips the side and reflects stop / target around the decision close; shift_spec decides one bar
    (one session) later"""
    from .api import Entry, Spec, Swing
    from .audit_run import mirror_spec, shift_spec
    from .indicators import build_sessions
    sd = _synth(3, seed=2)
    s = build_sessions(sd)[1]

    def f(ss, p):
        return [Entry(10, 1, stop=float(ss.c[10]) - 1.0, target=float(ss.c[10]) + 2.0, time_stop=15)]
    sp = Spec("T_M", "intraday", "t", "t", f, {})
    e = mirror_spec(sp).fn(s, sp.p)[0]
    assert e.side == -1 and abs(e.stop - (s.c[10] + 1.0)) < 1e-9 and abs(e.target - (s.c[10] - 2.0)) < 1e-9
    assert e.time_stop == 15 and shift_spec(sp).fn(s, sp.p)[0].i == 11
    from .sim import DailyView
    dv = DailyView(np.arange(5), *(np.arange(5.0) + 1 for _ in range(5)))
    dv.symbol, dv.nifty, dv.vix = "SYN", None, None
    seen = []

    def g(d, p):
        seen.append(len(d))
        return Swing(1, max_hold=2)
    sp2 = Spec("T_S", "swing", "t", "t", g, {}, horizon="positional")
    shift_spec(sp2).fn(dv, sp2.p)
    assert seen == [4]
    m = mirror_spec(sp2)
    assert m.kind == "futures" and m.fn(dv, sp2.p).side == -1


def t_aggregated_screen_equals_protocol_screen():
    """audit_run.screen_from on per-date sums gives protocol.screen's verdict, numbers and reasons"""
    from . import protocol as P
    from .audit_run import DISC, H1, H2, FINAL, screen_from
    rng = np.random.default_rng(9)
    cal = [d for d in _days(800, dt.date(2023, 11, 10)) if d <= FINAL[1]]
    split = P.Split(H1, H2, FINAL, 726, 508, 218)
    for trial in range(30):
        tr = []
        for _ in range(int(rng.integers(50, 400))):
            d = int(rng.choice(cal))
            net = float(rng.normal(20 if trial % 3 == 0 else -10, 500))
            tr.append(("X", f"S{int(rng.integers(0, 6))}", d, 1, "09:30", 1.0, d, "10:00", 1.0, 1, net + 80, 80.0,
                       net, "x"))
        ref = P.screen(tr, split, "intraday")
        dates = np.array(sorted({t[2] for t in tr}), np.int64)
        A = np.zeros((len(dates), 6))
        pos = {d: k for k, d in enumerate(dates)}
        persym = {}
        for t in tr:
            k = pos[t[2]]
            A[k] += [1, t[12], max(t[12], 0), min(t[12], 0), t[10], t[11]]
            if DISC[0] <= t[2] <= DISC[1]:
                persym[t[1]] = persym.get(t[1], 0.0) + t[12]
        got = screen_from(dates, A, persym, "intraday", False)
        assert got["passed"] == ref["passed"] and got["reasons"] == ref["reasons"], (got["reasons"], ref["reasons"])
        assert abs(got["disc"]["net"] - ref["disc"]["net"]) < 0.02 and got["disc"]["trades"] == ref["disc"]["trades"]


def _map_for(sd, mult, off=None, exclude=None, drop_first=None):
    n = sd.n_sessions
    return {"days": sd.days.astype(np.int32), "off": np.zeros(n) if off is None else off, "mult": mult,
            "vmult": np.ones(n), "exclude": np.zeros(n, bool) if exclude is None else exclude,
            "drop_first": np.zeros(n, bool) if drop_first is None else drop_first}


def t_corrected_ratio_map_keeps_intraday_returns():
    """a per-session ratio map changes levels and overnight ratios only: every intraday return is unchanged"""
    from . import corrected as C
    sd = _synth(4, seed=4)
    m = _map_for(sd, np.array([0.5, 0.5, 1.0, 1.0]))
    c = C.apply_map(sd, m)
    for k in range(4):
        a, b = sd.starts[k], sd.ends[k]
        assert np.allclose(np.diff(np.log(c.c[a:b])), np.diff(np.log(sd.c[a:b])))
    d0, d1 = sd.daily(), c.daily()
    assert abs(d1["o"][2] / d1["c"][1] - 2 * d0["o"][2] / d0["c"][1]) < 1e-9     # the step lands on the overnight


def t_corrected_affine_map_repairs_returns():
    """bars = a x true + b (dividends subtracted in rupees): the affine map restores the true intraday returns"""
    from . import corrected as C
    sd = _synth(3, seed=6)
    a_, b_ = 0.9, -12.0
    bad = D.from_arrays("SYN", sd.date, sd.t, a_ * sd.o + b_, a_ * sd.h + b_, a_ * sd.l + b_, a_ * sd.c + b_, sd.v,
                        min_bars=200)
    m = _map_for(bad, np.full(3, 1.0 / a_), off=np.full(3, b_))
    c = C.apply_map(bad, m)
    assert np.allclose(c.c, sd.c) and np.allclose(c.h, sd.h)
    assert not np.allclose(np.diff(np.log(bad.c)), np.diff(np.log(sd.c)))


def t_corrected_drops_stale_first_bar_and_excluded_sessions():
    """drop_first removes only that session's first bar; an excluded session disappears (and is listed in dropped)"""
    from . import corrected as C
    sd = _synth(4, seed=8)
    m = _map_for(sd, np.ones(4), exclude=np.array([False, False, True, False]),
                 drop_first=np.array([False, True, False, False]))
    c = C.apply_map(sd, m)
    assert c.n_sessions == 3 and int(sd.days[2]) in c.dropped
    assert (c.ends - c.starts)[1] == (sd.ends - sd.starts)[1] - 1 and c.t[c.starts[1]] == sd.t[sd.starts[1] + 1]


def t_corrected_install_routes_loader():
    """install() wraps data.load_symbol (and data_daily.load_symbol); uninstall() restores both"""
    from . import corrected as C
    from . import data_daily as DD
    orig_m, orig_d = D.load_symbol, DD.load_symbol
    try:
        C.install()
        assert D.load_symbol is C.load_symbol_corrected and DD.load_symbol is C.load_daily_corrected
        C.install()                                 # idempotent
        assert C._ORIG["min"] is orig_m
    finally:
        C.uninstall()
    assert D.load_symbol is orig_m and DD.load_symbol is orig_d
    assert C.load_symbol_corrected("/nonexistent", "X") is None


def t_corrected_daily_fix():
    """a daily-file split that was not applied: rows before the ex-date are scaled by official / applied factor"""
    from . import corrected as C
    sd = D.from_arrays("ZZZ", [20200101, 20200102, 20200103], [555] * 3, [100.0, 100.0, 50.0],
                       [101.0, 101.0, 51.0], [99.0, 99.0, 49.0], [100.0, 100.0, 50.0], [10.0, 10.0, 20.0])
    C._ORIG.setdefault("daily", lambda c, s: sd)
    saved = C.daily_fixes
    try:
        C.daily_fixes = lambda: {"ZZZ": [(20200103, 0.5)]}
        out = C.load_daily_corrected("x", "ZZZ")
        assert np.allclose(out.c, [50.0, 50.0, 50.0]) and np.allclose(out.v, [20.0, 20.0, 20.0])
    finally:
        C.daily_fixes = saved
        if C._ORIG.get("daily") is not None and getattr(C._ORIG["daily"], "__name__", "") == "<lambda>":
            C._ORIG.pop("daily")


def t_corrected_rerun_rows_are_not_trials():
    """trials.count_trials ignores corrected_* stages: a corrected re-run never adds a variant to N"""
    from . import trials as T
    rows = [{"strategy_id": "a", "era": "2023-26", "stage": "screen", "params_fp": "x"},
            {"strategy_id": "b", "era": "2023-26", "stage": "corrected_rerun", "params_fp": "x"},
            {"strategy_id": "a", "era": "2023-26", "stage": "corrected_final", "params_fp": "p:zz"}]
    assert T.count_trials(rows, "2023-26") == 1


def t_regression_real_bars_one_symbol():
    """(skipped without the 3-year bars) refsim = sim.py on ABB for five strategies over DISCOVERY"""
    from .audit_run import BARS3Y, CACHE3Y, DISC
    if not (os.path.exists(os.path.join(BARS3Y, "ABB.json")) and os.path.isdir(CACHE3Y)):
        return
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    ids = ["orb15_break", "gap_and_go", "scalp_orb5", "swing_20d_breakout", "fut_donchian20"]
    ss = R.sessions(R.read_bars(os.path.join(BARS3Y, "ABB.json")))
    ref = R.run_symbol("ABB", ss, ids, DISC[0], DISC[1])
    res = sim.run_symbol(CACHE3Y, "ABB", [REGISTRY[i] for i in ids], DISC[0], DISC[1], audit=False)
    for i in ids:
        a, b = ref[i], res[i]["trades"]
        assert len(a) == len(b) and len(a) > 5, i
        for x, y in zip(a, b):
            assert (x["date"], x["entry_t"], x["exit_date"]) == (y[2], y[4], y[6]), (i, x, y)
            assert abs(x["exit"] - y[8]) < 1e-3 and abs(x["net"] - y[12]) < 0.02, (i, x, y)


TESTS = [(f.__name__, f) for f in (t_refsim_matches_sim_fill_for_fill, t_tie_gap_and_tick_rules,
                                     t_zerodha_table_equals_costs_py, t_positional_engines_agree,
                                     t_mirror_and_shift_wrappers, t_aggregated_screen_equals_protocol_screen,
                                     t_corrected_ratio_map_keeps_intraday_returns,
                                     t_corrected_affine_map_repairs_returns,
                                     t_corrected_drops_stale_first_bar_and_excluded_sessions,
                                     t_corrected_install_routes_loader, t_corrected_daily_fix,
                                     t_corrected_rerun_rows_are_not_trials,
                                     t_regression_real_bars_one_symbol)]


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
