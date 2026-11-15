"""
stratlab/tests_kite.py — tests for the round-5 Kite-ecosystem code: indicators against hand-computed values
(kite_ind), the ADF test against statsmodels, the simulator extensions (signal exits / stop-and-reverse,
fill-relative brackets, positional exit callbacks, lots), the look-ahead audit of exit signals, and the strategy
plumbing (kite_tf). Run by test_pipeline.py (section "STRATLAB: round-5 Kite ecosystem") or with pytest.
"""
from __future__ import annotations

import math

import numpy as np

from . import data as D
from . import indicators as I
from . import kite_ind as K
from . import sim
from .api import LONG, SHORT, Entry, Spec, Swing
from .kite_stats import adf, mackinnonp


def _close(a, b, tol=1e-6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)))


H = np.array([10, 12, 11, 13, 14], float)
L = np.array([8, 9, 7, 10, 11], float)
C = np.array([9, 11, 8, 12, 12], float)


def t_aroon_hand():
    h = np.array([1, 3, 2, 5, 4], float)
    l = np.array([5, 4, 6, 1, 2], float)
    dn, up = K.talib_aroon(h, l, 2)
    assert _close(up, [np.nan, np.nan, 50, 100, 50]), up
    assert _close(dn, [np.nan, np.nan, 50, 100, 50]), dn
    _dn, up2 = K.talib_aroon(np.array([2, 2, 1.0]), np.array([1, 1, 1.0]), 2)
    assert up2[2] == 50.0, up2                                  # tie -> the most recent high (TA-Lib >=)


def t_williams_r_hand():
    w = K.williams_r(H[:4], L[:4], C[:4], 3)
    assert _close(w, [np.nan, np.nan, -80.0, -100.0 / 6.0]), w


def t_stochastic_hand():
    k = K.pine_stoch(C, H, L, 3)
    assert _close(k, [np.nan, np.nan, 20.0, 500.0 / 6.0, 500.0 / 7.0]), k
    sk, sd = K.talib_stoch(H, L, C, 3, 2, 2)
    assert _close(sk[3:], [(20 + 500 / 6) / 2, (500 / 6 + 500 / 7) / 2]), sk
    assert _close(sd[4], ((20 + 500 / 6) / 2 + (500 / 6 + 500 / 7) / 2) / 2), sd


def t_ichimoku_as_coded_hand():
    a, b = K.ichimoku_algotest(H, L, conv=1, base=2, span_b=3, disp=1)
    # hl2 = 9, 10.5, 9, 11.5, 12.5 ; kijun(2) = -, 9.75, 9.75, 10.25, 12 ; (tenkan+kijun)/2 shifted by 1
    assert _close(a, [np.nan, np.nan, 10.125, 9.375, 10.875]), a
    assert _close(b, [np.nan, np.nan, np.nan, 9.5, 31.0 / 3.0]), b


_STH = np.array([11, 12, 13, 12, 10], float)
_STL = np.array([9, 10, 11, 10, 8], float)
_STC = np.array([10, 11, 12, 11, 9], float)


def t_supertrend_and_double_hand():
    d1, line = K.supertrend(_STH, _STL, _STC, 1, 1.0)
    assert list(d1) == [0, 1, 1, 1, -1] and _close(line, [np.nan, 9, 10, 10, 12]), (d1, line)
    d2, _ = K.supertrend(_STH, _STL, _STC, 1, 2.0)
    assert list(d2) == [0, 1, 1, 1, 1], d2
    st = K.double_supertrend(_STH, _STL, _STC, a=(1, 1.0), b=(1, 2.0))
    assert list(st) == [0, 1, 1, 1, 0], st                     # last bar: the two disagree -> 0 (exit)


def t_simplified_supertrend_fires_both_ways():
    lg, sh = K.supertrend_simplified_algotest(_STH, _STL, _STC, 1, 1.0)
    assert list(lg) == [False, True, True, True, True] and list(sh) == list(lg), (lg, sh)


def t_psar_dmi_sanity():
    n = 60
    c = 100 + np.arange(n, dtype=float)
    h, l = c + 0.5, c - 0.5
    sar = K.pine_sar(h, l, c)
    assert np.all(sar[2:] < l[2:]), sar[:6]                     # a steady uptrend keeps SAR under the lows
    pdi, mdi, adx = K.pine_dmi(h, l, c, 14, 25)
    assert pdi[-1] > 50 and mdi[-1] < 1 and adx[-1] > 50, (pdi[-1], mdi[-1], adx[-1])


def t_rsi_ema_macd_bb_basics():
    x = np.arange(1, 11, dtype=float)
    assert _close(K.ema_sma_seed(x, 3), [np.nan, np.nan, 2, 3, 4, 5, 6, 7, 8, 9])
    assert K.rsi(x, 3)[-1] == 100.0 and np.isnan(K.rsi(x, 3)[2])
    m, up, lo = K.bollinger(np.array([1, 2, 3.0]), 3, 2.0)
    assert _close([m[2], up[2], lo[2]], [2, 2 + 2 * math.sqrt(2 / 3), 2 - 2 * math.sqrt(2 / 3)])
    x = K.algobulls_cross([1.0, 1.004, 1.02, 0.99], [1.0, 1.0, 1.0, 1.0])
    assert list(x) == [0, 0, 1, -1], x                        # 1.004 - 1 rounds to 0.00: no cross yet


def t_vwap_restarts_daily():
    tp = np.array([10, 20, 30, 40.0])
    v = np.array([1, 3, 1, 1.0])
    vw = K.session_vwap(tp, v, np.array([True, False, True, False]))
    assert _close(vw, [10, 17.5, 30, 35]), vw


def t_adf_matches_statsmodels():
    # statsmodels 0.15.0 adfuller(x) (regression="c", autolag="AIC"), computed in an isolated venv
    ref = [(1, 0.9, -4.328102017375453, 0.0003969858371982833, 2),
           (3, 1.0, -0.8589962172651996, 0.8011755939505768, 0),
           (5, 0.995, -2.0992076653830214, 0.24483332257526003, 8)]
    for seed, phi, stat, p, lag in ref:
        rng = np.random.default_rng(seed)
        e = rng.normal(size=200)
        x = np.zeros(200)
        for i in range(1, 200):
            x[i] = phi * x[i - 1] + e[i]
        t, pv, lg = adf(x)
        assert abs(t - stat) < 1e-8 and abs(pv - p) < 1e-8 and lg == lag, (seed, t, pv, lg)
    assert abs(mackinnonp(-2.86) - 0.0502) < 1e-3 and abs(mackinnonp(-3.43) - 0.00998) < 1e-3


# ── simulator extensions ────────────────────────────────────────────────────
def _flat_session(n=375, px=100.0, days=(20260105,)):
    date = np.repeat(np.array(days, np.int32), n)
    t = np.tile(555 + np.arange(n), len(days)).astype(np.int16)
    a = {k: np.full(len(date), px) for k in "ohlc"}
    return date, t, a, np.full(len(date), 1000.0)


def _sess(a, date, t, v, k=0):
    return I.build_sessions(D.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v))[k]


def _spec(fn, kind="intraday", horizon=None, **kw):
    hz = horizon or ("positional" if kind == "swing" else "intraday")
    return Spec(id=kw.pop("id", "t_" + fn.__name__), kind=kind, family="test", desc="test", fn=fn, horizon=hz, **kw)


def t_exit_signal_and_stop_and_reverse():
    date, t, a, v = _flat_session()
    for k in range(21, 40):                                     # price steps to 105 from bar 21
        a["o"][k] = a["h"][k] = a["l"][k] = a["c"][k] = 105.0
    s = _sess(a, date, t, v)
    xl = np.zeros(len(s), bool)
    xl[20] = True                                               # long exit decided at the close of bar 20
    xs = np.zeros(len(s), bool)
    xs[30] = True

    def fn(s_, p):
        return [Entry(10, LONG, stop=50.0, exit_sig=xl[:len(s_)]), Entry(20, SHORT, stop=200.0, exit_sig=xs[:len(s_)])]
    sp = _spec(fn, max_per_day=5)
    tr, sk = sim.sim_session(sp, s, slip=0.0)
    assert sk == 0 and len(tr) == 2, tr
    a1, a2 = tr
    assert a1[13] == "signal" and a1[7] == "09:36" and a1[8] == 105.0 and a1[4] == "09:26", a1   # out at bar 21's open
    assert a2[4] == "09:36" and a2[5] == 105.0 and a2[13] == "signal" and a2[7] == "09:46", a2  # reversed at that open
    assert sim.audit_lookahead(sp, s) is None


def t_exit_signal_audit_catches_future_reads():
    date, t, a, v = _flat_session()
    s = _sess(a, date, t, v)

    def cheat(s_, p):
        m = np.zeros(len(s_), bool)
        m[min(len(s_) - 1, 5)] = len(s_) > 300                   # bar 5 knows whether the session reaches bar 300
        return [Entry(3, LONG, stop=50.0, exit_sig=m)]
    msg = sim.audit_lookahead(_spec(cheat), s)
    assert msg and "differ" in msg, msg


def t_bracket_points_from_fill():
    date, t, a, v = _flat_session()
    a["o"][11] = 100.0
    a["l"][30] = 49.0                                            # 51 points below the fill: stop (50 pts) hit
    s = _sess(a, date, t, v)
    k, fill, px, reason = sim.fill_intraday(s, Entry(10, LONG, stop=None, bracket=("pts", 50, 100)), slip=0.0)
    assert (k, fill, px, reason) == (30, 100.0, 50.0, "stop"), (k, fill, px, reason)
    a["l"][30] = 100.0
    a["h"][40] = 101.0                                          # 1 % target, bracket in % of the fill
    s = _sess(a, date, t, v)
    k, fill, px, reason = sim.fill_intraday(s, Entry(10, LONG, stop=None, bracket=("pct", 1.0, 0.5)), slip=0.0)
    assert reason == "target" and abs(px - 100.5) < 1e-9 and k == 40, (k, px, reason)
    k, fill, px, reason = sim.fill_intraday(s, Entry(10, LONG, stop=99.0, bracket=("pts", None, 0.5)), slip=0.0)
    assert reason == "target" and abs(px - 100.5) < 1e-9, (k, px, reason)    # fixed stop, fill-relative target


def _daily_sd(closes, opens=None):
    n = len(closes)
    days = []
    d = 20260105
    import datetime as _dt
    x = _dt.date(2026, 1, 5)
    while len(days) < n:
        if x.weekday() < 5:
            days.append(int(x.strftime("%Y%m%d")))
        x += _dt.timedelta(days=1)
    date = np.repeat(np.array(days, np.int32), 2)
    t = np.tile(np.array([555, 929]), n).astype(np.int16)
    o = np.repeat(np.asarray(opens if opens is not None else closes, float), 2)
    c = np.repeat(np.asarray(closes, float), 2)
    return D.from_arrays("SYN", date, t, o, np.maximum(o, c), np.minimum(o, c), c, np.full(len(c), 1.0))


def t_positional_exit_fn_reversal_and_lots():
    closes = [100, 101, 102, 103, 104, 99, 98, 97, 96, 95]
    opens = [100, 100.5, 101.5, 102.5, 103.5, 100, 98.5, 97.5, 96.5, 95.5]
    sd = _daily_sd(closes, opens)

    def fn(d, p):
        if len(d) == 1:
            return Swing(LONG, max_hold=1000, exit_fn=lambda v: v.c[-1] < v.c[-2], lots=2)
        if d.c[-1] < d.c[-2]:
            return Swing(SHORT, max_hold=1000, exit_fn=lambda v: False)
        return None
    sp = _spec(fn, kind="futures", horizon="positional", underlying="NIFTY", lots=1)
    tr, sk = sim.sim_positional(sp, sd, 0, 99999999, slip=0.0)
    assert len(tr) == 2, tr
    t1, t2 = tr
    # long from day 2's open (100.5), exit decided at day 6's close (99 < 104), out at day 7's open (98.5)
    assert t1[5] == 100.5 and t1[8] == 98.5 and t1[13] == "signal" and t1[9] == 2 * sim.LOT_SIZES["NIFTY"], t1
    # the short decided on the same close fills at that same open (stop-and-reverse), 1 lot
    assert t2[2] == t1[6] and t2[5] == 98.5 and t2[3] == -1 and t2[9] == sim.LOT_SIZES["NIFTY"], t2


def t_kite_machine_entries_audit_clean():
    """Every round-5 intraday strategy runs on a random walk and passes the dense look-ahead audit."""
    from . import catalog  # noqa: F401
    from .api import REGISTRY
    rng = np.random.default_rng(5)
    n, nd = 375, 25
    days = [20260100 + d for d in range(1, nd + 1)]
    date = np.repeat(np.array(days, np.int32), n)
    t = np.tile(555 + np.arange(n), nd).astype(np.int16)
    c = 500.0 * np.exp(np.cumsum(rng.normal(0, 0.0015, n * nd)))
    o = np.concatenate(([500.0], c[:-1]))
    h = np.maximum(o, c) * 1.0004
    l = np.minimum(o, c) * 0.9996
    sd = D.from_arrays("RW", date, t, o, h, l, c, rng.integers(500, 5000, len(c)).astype(float))
    sessions = I.build_sessions(sd)
    specs = [sp for i, sp in REGISTRY.items() if i.startswith("r5_kite_") and sp.horizon == "intraday"]
    assert len(specs) == 29, len(specs)
    traded = 0
    for sp in specs:
        k = 0
        for s in sessions[5:]:
            tr, _ = sim.sim_session(sp, s)
            k += len(tr)
            for x in tr:
                assert x[4] < "15:15" and x[7] <= "15:15", (sp.id, x)
            bad = sim.audit_lookahead(sp, s, cuts=(20, 46, 75, 110, 170, 226, 290, 330))
            assert bad is None, f"{sp.id}: {bad}"
        traded += k > 0
    assert traded >= len(specs) - 4, traded


TESTS = [
    ("kite_ind: Aroon (TA-Lib) against hand-computed values", t_aroon_hand),
    ("kite_ind: Williams %R (AlgoTest template) against hand-computed values", t_williams_r_hand),
    ("kite_ind: stochastic (Pine raw %K; TA-Lib STOCH SMA) against hand-computed values", t_stochastic_hand),
    ("kite_ind: Ichimoku as coded by AlgoTest (SMA-of-hl2 lines, displaced) against hand values", t_ichimoku_as_coded_hand),
    ("kite_ind: Supertrend and Tradetron double Supertrend against hand-computed values", t_supertrend_and_double_hand),
    ("kite_ind: AlgoTest simplified Supertrend fires long and short on the same bars (as coded)",
     t_simplified_supertrend_fires_both_ways),
    ("kite_ind: Pine PSAR / DMI behave on a steady uptrend", t_psar_dmi_sanity),
    ("kite_ind: EMA seed, RSI, Bollinger, AlgoBulls 2-dp crossover", t_rsi_ema_macd_bb_basics),
    ("kite_ind: AlgoBulls VWAP restarts every day", t_vwap_restarts_daily),
    ("kite_stats: ADF statistic / p-value / lag equal statsmodels adfuller", t_adf_matches_statsmodels),
    ("sim: exit signal fills at the next open; opposite entry reverses at that open", t_exit_signal_and_stop_and_reverse),
    ("sim: the look-ahead audit also checks exit signals", t_exit_signal_audit_catches_future_reads),
    ("sim: bracket stop / target measured from the fill (points, percent, fixed stop)", t_bracket_points_from_fill),
    ("sim: positional exit callback, same-open reversal, per-trade lots", t_positional_exit_fn_reversal_and_lots),
    ("r5_kite: every intraday strategy runs and passes the dense look-ahead audit", t_kite_machine_entries_audit_clean),
]


if __name__ == "__main__":
    bad = 0
    for name, f in TESTS:
        try:
            f()
            print("ok  ", name)
        except Exception as exc:            # pragma: no cover
            bad += 1
            print("FAIL", name, repr(exc)[:300])
    raise SystemExit(1 if bad else 0)
