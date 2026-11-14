"""
stratlab/tests_r7.py — tests for round 7 (second GitHub shortlist): r7_ind formulas against hand-computed values
(Clenow counter-trend on price differences, Turtle Soup / +1 setups, Smash Day with the MT5 SMA ATR, Holy Grail
gating, Double 7s, Bensdorp ATR% and MRL levels, the TPS state machine, 80-20 and the intraday stop re-entry), the
round-7 resting-order fill in sim.py (stop, armed stop, limit, 2-session validity, "one tick under today's low",
pessimistic fill-session stop, targets from the next session), the TPS trade arithmetic, the portfolio engines
(volatility-managed exposure, futures carry with rolls) and look-ahead checks (portfolio functions with the rows after
t replaced by noise; dense audit of the intraday specs). Run by test_pipeline.py (section "STRATLAB: round-7 GitHub
strategies") or directly: python -m stratlab.tests_r7
"""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as I
from . import kite_ind as K
from . import r7_ind as R7
from . import sim
from .api import LONG, SHORT, Spec, Swing


def _days(n, start=_dt.date(2024, 1, 1)):
    out, x = [], start
    while len(out) < n:
        if x.weekday() < 5:
            out.append(int(x.strftime("%Y%m%d")))
        x += _dt.timedelta(days=1)
    return np.array(out, np.int64)


def _rw(n, seed, s0=100.0, sig=0.015):
    rng = np.random.default_rng(seed)
    c = s0 * np.exp(np.cumsum(rng.normal(0, sig, n)))
    o = np.concatenate(([s0], c[:-1])) * np.exp(rng.normal(0, sig / 3, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, sig / 2, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sig / 2, n)))
    return o, h, l, c


def _daily_sd(o, h, l, c, sym="SYN", start=_dt.date(2024, 1, 1)):
    d = _days(len(c), start)
    return D.from_arrays(sym, d.astype(np.int32), np.full(len(c), 555, np.int16), o, h, l, c, np.ones(len(c)))


# ── formulas ────────────────────────────────────────────────────────────────
def t_clenow_ct_hand():
    c = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136,
                  138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176,
                  178, 180, 150.0])
    prm = dict(R7.CLENOW)
    trend, pb = R7.clenow_ct(c, prm)
    s = pd.Series(c)
    assert trend == bool(s.ewm(span=40).mean().iloc[-1] > s.ewm(span=80).mean().iloc[-1]) and trend
    diffs = np.diff(c)[-40:]                          # 39 diffs of +2 and one of -30
    m = (39 * 2 - 30) / 40
    sd = math.sqrt((39 * (2 - m) ** 2 + (-30 - m) ** 2) / 39)
    assert abs(np.std(diffs, ddof=1) - sd) < 1e-12
    assert abs(pb - (150 - 180) / sd) < 1e-12, pb
    # price DIFFERENCES, not % changes: adding a constant to every price leaves the pullback unchanged
    assert abs(R7.clenow_ct(c + 5000.0, prm)[1] - pb) < 1e-9
    assert abs(R7.ewm_adjust_mean([1.0, 2.0, 3.0], 3) - pd.Series([1.0, 2.0, 3.0]).ewm(span=3).mean().iloc[-1]) < 1e-12


def t_turtle_soup_hand():
    l = np.array([50.0] * 30)
    h = l + 2
    l[-6] = 45.0                                       # the 20-day low, made 5 sessions before the decision day
    assert (1, 45.0) in R7.turtle_soup(h, l)           # age 5 >= 3: the trade session is 6 sessions after it
    l2 = l.copy()
    l2[-6] = 50.0
    l2[-3] = 44.0                                      # age 2 on the decision day: only 3 sessions before the trade day
    assert not any(s == 1 for s, _x in R7.turtle_soup(h, l2))
    l3 = l.copy()
    l3[-4] = 45.0                                      # tie with l[-6]: the most recent occurrence (age 3) counts
    assert (1, 45.0) in R7.turtle_soup(h, l3)
    l3[-3] = 45.0
    assert not any(s == 1 for s, _x in R7.turtle_soup(h, l3))
    # Plus One: day one (last bar) breaks the prior 20-day low (made >= 3 sessions earlier) and closes at / below it
    c = np.array([51.0] * 30)
    l4 = np.array([50.0] * 30)
    h4 = l4 + 3
    l4[-5] = 47.0
    l4[-1], c[-1] = 46.0, 46.5
    assert R7.turtle_soup_plus_one(c, h4, l4) == [(1, 47.0, 46.0)]
    c[-1] = 47.5                                       # closed back above the prior low: no setup
    assert R7.turtle_soup_plus_one(c, h4, l4) == []
    c[-1] = 46.5
    l4[-5], l4[-2] = 50.0, 47.0                        # prior low only 1 session old
    assert R7.turtle_soup_plus_one(c, h4, l4) == []


def t_smash_day_hand():
    n = 30
    h = np.full(n, 105.0)
    l = np.full(n, 95.0)
    c = np.full(n, 100.0)
    o = c.copy()
    l[-1], c[-1], h[-1] = 90.0, 92.0, 101.0            # close 92 < yesterday's low 95; low 90 < 4-day low 95
    r = R7.smash_day(o, h, l, c)
    tr = K.true_range(h, l, c)                          # 21-bar SMA ATR: 20 bars of TR 10 and the last bar's 11
    atr = (20 * 10.0 + max(101 - 90, abs(101 - 100), abs(90 - 100))) / 21
    assert abs(tr[-1] - 11.0) < 1e-12
    assert r[0] == 1 and r[1] == 101.0
    assert abs(r[2] - (101 - 2.1 * atr)) < 1e-9 and abs(r[3] - (101 + 0.75 * 2.1 * atr)) < 1e-9
    l[-1] = 95.0                                        # low not below the 4-day low: no setup
    assert R7.smash_day(o, h, l, c) is None
    l[-1], c[-1], h[-1] = 99.0, 108.0, 110.0            # short: close 108 > yesterday's high 105, high 110 > 4-day high
    r = R7.smash_day(o, h, l, c)
    assert r[0] == -1 and r[1] == 99.0 and r[2] > 99.0 > r[3]


def t_holy_grail_gate():
    """On noisy trending walks: every setup satisfies the rule (checked with independently computed DMI / EMA) with
    the stated levels, and every bar that meets the rule is a setup."""
    n_hit = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        c = 100 * np.exp(np.cumsum(rng.normal(0.006 if seed % 2 == 0 else -0.006, 0.012, 400)))
        h = c * (1 + np.abs(rng.normal(0, 0.012, 400)))
        l = c * (1 - np.abs(rng.normal(0, 0.012, 400)))
        for t in range(80, 400):
            hh, ll, cc = h[:t + 1], l[:t + 1], c[:t + 1]
            pdi, mdi, adx = K.pine_dmi(hh, ll, cc, 14, 14)
            e = K.ema(cc, 20)
            gate = adx[-2] > 30 and adx[-2] > adx[-3]
            want_l = gate and pdi[-2] > mdi[-2] and cc[-2] > e[-2] and ll[-1] <= e[-1]
            want_s = gate and mdi[-2] > pdi[-2] and cc[-2] < e[-2] and hh[-1] >= e[-1]
            r = R7.holy_grail(hh, ll, cc)
            if want_l:
                assert r == (1, hh[-1], ll[-1], float(np.max(hh[-21:-1]))), (seed, t, r)
            elif want_s:
                assert r == (-1, ll[-1], hh[-1], float(np.min(ll[-21:-1]))), (seed, t, r)
            else:
                assert r is None, (seed, t, r)
            n_hit += r is not None
    assert n_hit >= 3, n_hit


def t_double7_hand():
    c = np.concatenate([np.linspace(100, 200, 200), [199, 198, 197, 196, 195, 194, 193.0]])
    assert R7.double7_entry(c)                          # 7-day closing low, above SMA200
    assert not R7.double7_entry(np.concatenate([c[:-1], [199.5]]))
    assert R7.double7_exit(np.array([1, 2, 3, 4, 5, 6, 7.0])) and not R7.double7_exit(np.array([7, 6, 5, 4, 3, 2, 1.0]))
    cl = np.concatenate([np.linspace(200, 100, 200), [99, 98, 97, 96, 95, 94, 93.0]])
    assert not R7.double7_entry(cl)                     # below SMA200


def t_bensdorp_levels():
    h = np.array([10, 11, 12, 11.0])
    l = np.array([9, 10, 10, 9.0])
    c = np.array([9.5, 10.5, 11, 10.0])
    # TR(10.5 ref) = max(11-10, |11-9.5|, |10-9.5|) = 1.5 ; TR = max(2, |12-10.5|, |10-10.5|) = 2 ; TR = max(2, 0, 2) = 2
    assert abs(R7.atr_pct(h, l, c, 3) - np.mean([1.5 / 10.5, 2 / 11, 2 / 10])) < 1e-12
    o, hh, ll, cc = _rw(1500, 4, sig=0.035)
    n_hit = 0
    for t in range(200, 1500):
        r = R7.mrl_setup(hh[:t + 1], ll[:t + 1], cc[:t + 1])
        if r is None:
            continue
        n_hit += 1
        ap = R7.atr_pct(hh[:t + 1], ll[:t + 1], cc[:t + 1], 10)
        assert cc[t] > cc[t - 149:t + 1].mean() and ap > 0.04
        assert abs(r[0] - cc[t] * 0.96) < 1e-9 and abs(r[1] - r[0] * (1 - 2.5 * ap)) < 1e-9 and abs(r[2] - r[0] * 1.03) < 1e-9
        assert R7.wilder_rsi(cc[:t + 1], 3)[-1] < 30
    assert n_hit > 0


def t_tps_machine():
    up = np.linspace(100, 160, 220)
    c = np.concatenate([up, [158, 156, 155, 154, 153, 165, 170, 175]])
    acts = R7.tps_signals(c)
    rsi = K.rsi(c, 2)
    ent = [i for i, a in enumerate(acts) if a and a[0] == "enter"]
    assert ent and acts[ent[0]] == ("enter", 1, 0.1)
    i0 = ent[0]
    assert rsi[i0] < 25 and rsi[i0 - 1] < 25 and c[i0] > c[i0 - 199:i0 + 1].mean()
    adds = [acts[i] for i in range(i0 + 1, len(c)) if acts[i] and acts[i][0] == "add"]
    assert [a[2] for a in adds] == [0.2, 0.3, 0.4][:len(adds)] and len(adds) >= 1
    ex = [i for i in range(i0 + 1, len(c)) if acts[i] and acts[i][0] == "exit"]
    assert ex and rsi[ex[0]] > 70


def t_tps_trade_arithmetic():
    from . import r7_tps as T
    seq = {"side": 1, "fills": [(20240102, 100.0, 6.5), (20240103, 98.0, 13.0)]}
    tr = T._close(seq, 20240110, 105.0, "signal", "X", "_NIFTY50_min")
    gross = 6.5 * 5 + 13.0 * 7
    cost = T.round_trip_cost(650.0, 682.5, "future") + T.round_trip_cost(1274.0, 1365.0, "future")
    assert tr[2] == 20240102 and tr[6] == 20240110 and abs(tr[9] - 19.5) < 1e-9
    assert abs(tr[10] - round(gross, 2)) < 1e-9 and abs(tr[11] - round(cost, 2)) < 1e-6
    assert abs(tr[5] - (650 + 1274) / 19.5) < 1e-3
    # a full run: every sequence starts on a fill date in range and ends at / after it
    o, h, l, c = _rw(900, 7, s0=20000, sig=0.012)
    d = _days(900)
    trs = T.simulate_tps(d, o, c, 65, int(d[300]), int(d[700]), 1e-4, "X", "_NIFTY50_min")
    assert trs and all(int(d[300]) <= x[2] <= int(d[700]) and x[6] >= x[2] for x in trs)
    assert all(x[6] <= int(d[700]) for x in trs)


def t_intraday_helpers():
    assert R7.eighty_twenty_side(109, 110, 100, 101) == 1     # open at 90 % of range, close at 10 %
    assert R7.eighty_twenty_side(101, 110, 100, 109) == -1
    assert R7.eighty_twenty_side(105, 110, 100, 101) == 0
    hi = np.array([101, 100, 99.5, 100.2, 101])
    lo = np.array([100, 98.9, 99.0, 99.4, 100])
    cl = np.array([100.5, 99.2, 99.3, 100.1, 100.8])
    # arm = 99.0 (traded below at bar 1), level 100: bar 1 closes below -> trigger on bar 3 (high 100.2)
    assert R7.stop_reentry_bar(hi, lo, cl, 1, 99.0, 100.0) == 3
    cl2 = cl.copy()
    cl2[1] = 100.0                                      # the arming bar closes at the level: it triggers itself
    assert R7.stop_reentry_bar(hi, lo, cl2, 1, 99.0, 100.0) == 1
    assert R7.stop_reentry_bar(hi, lo, cl, 1, 98.0, 100.0) == -1


# ── the round-7 resting-order fill (sim.py) ───────────────────────────────────
def _spec(fn, kind="futures"):
    return Spec(id="T7", kind=kind, family="t", desc="t", fn=fn, params={}, horizon="positional")


def _bars(rows):
    o = np.array([r[0] for r in rows], float)
    h = np.array([r[1] for r in rows], float)
    l = np.array([r[2] for r in rows], float)
    c = np.array([r[3] for r in rows], float)
    return _daily_sd(o, h, l, c)


def t_sim_resting_orders():
    rows = [(100, 101, 99, 100)] * 3 + [(100, 104, 98, 103), (103, 106, 102, 105), (105, 107, 104, 106),
                                        (106, 108, 105, 107)]
    sd = _bars(rows)

    def once(sig):
        def fn(d, p):
            return sig if len(d.c) == 3 else None
        return fn
    slip = 0.0
    # plain buy stop at 103.5: day 3 trades 98..104 -> fill at 103.5; max_hold 2 -> out at day 4's close
    tr, sk = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 103.5), max_hold=2))), sd, 0, 99999999, slip=slip)
    assert len(tr) == 1 and tr[0][5] == 103.5 and tr[0][8] == 105.0 and tr[0][13] == "max_hold" and tr[0][4] == "order"
    # gap above the stop: fill at the open
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 99.5), max_hold=1))), sd, 0, 99999999, slip=slip)
    assert tr[0][5] == 100.0
    # armed stop: must trade below 98.5 first (day 3 low 98 -> armed), fills AT the level even though it opened above
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 99.0, 98.5), max_hold=1))), sd, 0, 99999999, slip=slip)
    assert tr[0][5] == 99.0
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 99.0, 97.5), max_hold=1))), sd, 0, 99999999, slip=slip)
    assert tr == []                                     # never armed (low 98 > 97.5): no trade
    # limit buy: needs a trade one tick through (98.05 -> low 98 is a tick through: fill at 98.05); 98.0 is a bare touch
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("limit", 98.05), max_hold=1))), sd, 0, 99999999, slip=slip)
    assert tr[0][5] == 98.05
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("limit", 98.0), max_hold=1))), sd, 0, 99999999, slip=slip)
    assert tr == []
    # valid 2 sessions: a stop at 105.5 misses day 3 (high 104) and fills on day 4 (high 106)
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 105.5), valid=2, max_hold=1))), sd, 0, 99999999,
                               slip=slip)
    assert tr[0][2] == int(_days(7)[4]) and tr[0][5] == 105.5
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 105.5), valid=1, max_hold=1))), sd, 0, 99999999,
                               slip=slip)
    assert tr == []
    # "one tick under today's low": stop = min(given 99, fill-day low 98) - 0.05 = 97.95, fixed at the fill session
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 103.5), stop=99.0, entry_stop_ticks=1,
                                                   max_hold=3))), sd, 0, 99999999, slip=slip)
    assert tr[0][13] == "max_hold" and tr[0][8] == 106.0
    rows2 = rows[:4] + [(103, 104, 97.0, 98), (98, 99, 97, 98), (98, 99, 97, 98)]
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 103.5), entry_stop_ticks=1, max_hold=5))),
                               _bars(rows2), 0, 99999999, slip=slip)
    assert tr[0][13] == "stop" and abs(tr[0][8] - 97.95) < 1e-9 and tr[0][6] == int(_days(7)[4])
    # fill-session stop is pessimistic: a stop at 99 inside the fill day's range (low 98) is taken as hit that day
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 103.5), stop=99.0, max_hold=5))), sd, 0, 99999999,
                               slip=slip)
    assert tr[0][13] == "stop" and tr[0][6] == tr[0][2] and tr[0][8] == 99.0
    # targets start the session after the fill: 103.9 is traded through on the fill day (high 104) but not counted
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, entry=("stop", 103.5), target=105.9, max_hold=5))), sd, 0,
                               99999999, slip=slip)
    assert tr[0][13] == "target" and tr[0][6] == int(_days(7)[4]) and tr[0][8] == 105.9
    # short sell stop and slippage sign
    tr, _ = sim.sim_positional(_spec(once(Swing(SHORT, entry=("stop", 98.5), max_hold=1))), sd, 0, 99999999, slip=0.001)
    assert abs(tr[0][5] - 98.5 * 0.999) < 1e-9 and tr[0][3] == -1
    # unfilled order: the strategy is asked again the next session
    calls = []

    def fn(d, p):
        calls.append(len(d.c))
        return Swing(LONG, entry=("stop", 1e9), max_hold=1)
    sim.sim_positional(_spec(fn), sd, 0, 99999999, slip=slip)
    assert calls == [1, 2, 3, 4, 5, 6], calls
    # a Swing without `entry` is untouched by the round-7 branch (market entry at the next open)
    tr, _ = sim.sim_positional(_spec(once(Swing(LONG, max_hold=1))), sd, 0, 99999999, slip=slip)
    assert tr[0][5] == 100.0 and tr[0][4] == "09:15"


# ── portfolio engines ────────────────────────────────────────────────────────
def t_volmanaged_hand_and_causal():
    from . import r7_portfolio as RP7
    d = _days(5)
    o = np.array([100, 100, 110, 121, 121.0])
    c = np.array([100, 110, 121, 121, 133.1])
    r = RP7.vm_backtest(d, o, c, {0: 1.5}, int(d[0]), None, 0.0)
    e0 = 1e6
    units = 1.5 * e0 / 100
    stat = RP7.round_trip_cost(1.5e6, 0.0, "future")
    assert abs(r["equity"][1] - (e0 - stat + units * 10)) < 1e-6        # bought 15,000 units at the day-1 open
    assert abs(r["equity"][-1] - (e0 - stat + units * 33.1)) < 1e-6
    # weights: c / RV with the expanding c; rows after t replaced by noise do not change the weight at t
    rng = np.random.default_rng(3)
    dd = _days(900, _dt.date(2012, 1, 2))
    cc = 5000 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, 900)))
    w = RP7.vm_weights(dd, cc)
    ts = sorted(w)
    assert len(ts) > 20 and all(0 < w[t] <= 1.5 for t in ts)
    t = ts[10]
    cc2 = cc.copy()
    cc2[t + 1:] *= np.exp(rng.normal(0, 0.1, 900 - t - 1))
    assert abs(RP7.vm_weights(dd, cc2)[t] - w[t]) < 1e-12
    # by hand for the first weight: c = std(raw) / std(raw / RV_prev); w = min(1.5, c / RV_t)
    r_ = cc[1:] / cc[:-1] - 1
    ym = dd // 100
    months = sorted(set(ym.tolist()))
    rv = {}
    for m in months:
        ii = np.flatnonzero(ym == m)
        x = np.array([r_[i - 1] for i in ii if i >= 1])
        rv[m] = float(np.sum(x ** 2))
    t0 = ts[0]
    m_t = int(ym[t0])
    ms = [m for m in months if m <= m_t]
    raw = np.concatenate([[r_[i - 1] for i in np.flatnonzero(ym == b) if i >= 1] for b in ms[1:]])
    man = np.concatenate([np.array([r_[i - 1] for i in np.flatnonzero(ym == b) if i >= 1]) / rv[a]
                          for a, b in zip(ms[:-1], ms[1:])])
    exp = min(1.5, (np.std(raw, ddof=1) / np.std(man, ddof=1)) / rv[m_t])
    assert abs(w[t0] - exp) < 1e-12, (w[t0], exp)


def t_carry_engine_hand():
    from . import r6_portfolio as RP
    from . import r7_portfolio as RP7
    d = _days(12, _dt.date(2024, 1, 1))                  # 2024-01-01 (Mon) .. 2024-01-16
    N = len(d)
    syms = ["A", "B", "C", "D", "E"]
    rows = []
    exp1, exp2, exp3 = 20240111, 20240125, 20240229
    for t, day in enumerate(d):
        for j, s in enumerate(syms):
            px = 100.0 + (j == 0) * (t >= 6) * 10.0       # A's futures +10 % from session 6 on
            for e, prem in ((exp1, 0.0), (exp2, 1.0 - 0.5 * j), (exp3, 2.0)):
                if e >= day:
                    rows.append({"date": int(day), "symbol": s, "expiry": e, "open": px + prem, "settle": px + prem,
                                 "contracts": 10.0})
    f = pd.DataFrame(rows)
    close = np.full((N, 5), 100.0)
    close[6:, 0] = 110.0
    p = RP.XPanel(d, syms, close, close.copy(), np.full(N, 1000.0))
    A = RP7.carry_arrays(f, d, syms)
    t = 2                                                # 2024-01-03: front = Jan 11, second = Jan 25
    assert A["f_exp"][t, 0] == exp1 and A["s_exp"][t, 0] == exp2
    sc = RP7.carry_score(A, t)
    assert abs(sc[4] - ((100.0 / (100.0 + 1.0 - 2.0)) - 1) * 365 / 14) < 1e-12     # E: second below front: highest
    gap, day, roll = RP7.contract_returns(A, p.open, p.close, 0.10)
    assert roll[7].all() and not roll[6].any()           # the held contract changes at the close of Jan 10
    assert abs(gap[6, 0] - 0.10) < 1e-12                 # A's front jumps 100 -> 110 on session 6
    w = RP7.carry_weights(p, A, t, 202401, {"quantile": 0.2, "gross_side": 0.5})
    assert w == {4: 0.5, 0: -0.5}                        # long E (highest carry), short A (lowest)
    prm = {"quantile": 0.2, "gross_side": 0.5, "slip_bps": 0.0}
    r = RP7.carry_backtest(p, A, gap, day, roll, prm, int(d[4]))   # first weekly decision Fri 2024-01-05
    assert r["first_decision"] == 20240105
    rolls = [x for x in r["trades"] if x[2] == "ROLL"]
    assert rolls and all(x[0] == int(d[7]) for x in rolls)


def t_r7_portfolio_functions_causal():
    """Sector weights and carry weights at t do not change when rows after t are replaced by noise."""
    from . import r6_portfolio as RP
    from . import r7_portfolio as RP7
    rng = np.random.default_rng(11)
    d = _days(700, _dt.date(2019, 1, 1))
    N, M = len(d), 11
    c = 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (N, M)), axis=0))
    p = RP.XPanel(d, sorted(RP7.SECTORS), c, c.copy(), np.full(N, 1000.0))
    firsts = [t for t in range(300, N) if d[t] // 100 != d[t - 1] // 100][:6]
    for t in firsts:
        w0 = RP7.w_sector(p, t, None, RP7.SECT)
        c2 = c.copy()
        c2[t + 1:] *= np.exp(rng.normal(0, 0.3, c2[t + 1:].shape))
        w1 = RP7.w_sector(RP.XPanel(d, sorted(RP7.SECTORS), c2, c2.copy(), np.full(N, 1000.0)), t, None, RP7.SECT)
        assert w0 == w1 and w0 is not None and len(w0) <= 2 and all(v == 0.5 for v in w0.values())
    assert RP7.w_sector(p, firsts[0] + 1, None, RP7.SECT) is None       # not a month's first session


def t_r7_intraday_audit_clean():
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    rng = np.random.default_rng(17)
    n, nd = 375, 24
    days = [20260100 + k for k in range(1, nd + 1)]
    date = np.repeat(np.array(days, np.int32), n)
    t = np.tile(555 + np.arange(n), nd).astype(np.int16)
    parts, px = [], 20000.0
    for k in range(nd):                                 # alternate: strong down day closing on its low, gap, reversal
        drift = -0.00025 if k % 3 == 0 else (0.00025 if k % 3 == 1 else 0.0)
        gap = 0.997 if k % 3 == 1 else (1.003 if k % 3 == 2 else 1.0)
        px *= gap
        seg = px * np.exp(np.cumsum(rng.normal(drift, 0.0009, n)))
        parts.append(seg)
        px = seg[-1]
    c = np.concatenate(parts)
    o = np.concatenate(([20000.0], c[:-1]))
    o[::n] = c[::n] / np.exp(rng.normal(0, 0.0005, nd))
    h = np.maximum(o, c) * 1.0003
    l = np.minimum(o, c) * 0.9997
    specs = [sp for i, sp in REGISTRY.items() if i.startswith("r7_gh2_") and sp.horizon == "intraday"]
    assert len(specs) == 6, [s.id for s in specs]
    for sym in ("_NIFTY50_min", "_NIFTYBANK_min", "RW"):
        sd = D.from_arrays(sym, date, t, o, h, l, c, np.zeros(len(c)))
        sessions = I.build_sessions(sd)
        for sp in specs:
            if not sim.applies(sp, sym):
                continue
            for s in sessions[2:]:
                tr, _ = sim.sim_session(sp, s)
                for x in tr:
                    assert x[4] < "15:15" and x[7] <= "15:15", (sp.id, x)
                bad = sim.audit_lookahead(sp, s, cuts=(5, 20, 46, 75, 110, 170, 226, 290, 330))
                assert bad is None, f"{sp.id}: {bad}"


def t_r7_daily_specs_run():
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    o, h, l, c = _rw(700, 21, sig=0.02)
    specs = [sp for i, sp in REGISTRY.items() if i.startswith("r7_gh2_") and sp.horizon == "positional"]
    assert len(specs) == 19, len(specs)
    n_tr = 0
    for sym in ("SYN", "_NIFTY50_min"):
        sd = _daily_sd(o, h, l, c, sym=sym)
        for sp in specs:
            tr, _sk = sim.sim_positional(sp, sd, 0, 99999999)
            n_tr += len(tr)
            for x in tr:
                assert x[6] >= x[2] and x[0] == sp.id
    assert n_tr > 30, n_tr


TESTS = [
    ("r7_ind: Clenow counter-trend by hand (std of price DIFFERENCES, pandas ewm adjust=True)", t_clenow_ct_hand),
    ("r7_ind: Turtle Soup / Turtle Soup +1 setups by hand (age rules, ties, close at / below)", t_turtle_soup_hand),
    ("r7_ind: Smash Day rule list and MT5 SMA ATR stop / target by hand", t_smash_day_hand),
    ("r7_ind: Holy Grail gate (ADX(14) > 30 and rising, first EMA20 touch) and levels", t_holy_grail_gate),
    ("r7_ind: Double 7s closing low / high and SMA200 filter", t_double7_hand),
    ("r7_ind: Bensdorp ATR% by hand and MRL levels on every setup of a random walk", t_bensdorp_levels),
    ("r7_ind: Connors TPS state machine (enter 10 %, adds 20/30/40 %, exit RSI2 > 70)", t_tps_machine),
    ("r7_tps: scale-in sequence arithmetic by hand; sequences inside the period", t_tps_trade_arithmetic),
    ("r7_ind: 80-20 setup and the intraday stop re-entry bar by hand", t_intraday_helpers),
    ("sim: round-7 resting orders (stop, armed stop, limit, validity, tick-under-low, pessimistic fill day)",
     t_sim_resting_orders),
    ("r7_portfolio: volatility-managed engine by hand; expanding c by hand; causal", t_volmanaged_hand_and_causal),
    ("r7_portfolio: futures carry arrays, score, rolls and weights by hand", t_carry_engine_hand),
    ("r7_portfolio: sector-rotation weights causal and only on a month's first session", t_r7_portfolio_functions_causal),
    ("r7_github: intraday specs pass the dense look-ahead audit", t_r7_intraday_audit_clean),
    ("r7_github: daily specs run on synthetic bars (stock and index)", t_r7_daily_specs_run),
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
            traceback.print_exc(limit=4)
    raise SystemExit(1 if bad else 0)
