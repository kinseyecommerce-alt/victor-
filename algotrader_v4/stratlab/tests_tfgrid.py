"""
stratlab/tests_tfgrid.py — tests for the timeframe grid (tfgrid.py) and the Rs 10 lakh account (capital.py) on
SYNTHETIC data only: sizing and margin arithmetic (MIS 5x, CNC 1x, futures lots + margin, below one lot, the minimum
position value), open-position slots and buying power, time-ordered priority with the previous-session traded-value
tie-break, the daily loss stop, the 1:3 bracket (target distance = 3 x stop distance on both sides, account net =
gross - costs.py costs), parity of the grid engine with sim.fill_intraday, candles of every timeframe, no look-ahead
across timeframes (cutting the data later never changes earlier candidates), the native 1:3 wrapper and the trials
ledger append. Run by test_pipeline.py (section "STRATLAB: timeframe grid + Rs 10 lakh account") or directly:
python -m stratlab.tests_tfgrid
"""
from __future__ import annotations

import datetime as dt
import math
import os
import tempfile

import numpy as np

from . import capital as K
from . import data as D
from . import tfgrid as G

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402


def _ag(**kw):
    base = dict(name="t", max_pos=2, max_trades_day=None, daily_loss=None, min_value=0.0)
    base.update(kw)
    return K.AgentConfig(**base)


def _c(sym, ek, fill, stop, xk, xpx, side=1, prio=0.0, inst="mis", lot=1):
    return K.cand(sym, side, ek, fill, stop, xk, xpx, "x", prio, inst, lot, 1, "T")


def _synth(n_days=60, seed=3, start=dt.date(2024, 1, 1), vol=0.0008, sym="SYN"):
    rng = np.random.default_rng(seed)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(int(d.strftime("%Y%m%d")))
        d += dt.timedelta(days=1)
    T = np.arange(555, 930)
    date = np.repeat(days, len(T))
    t = np.tile(T, len(days))
    c = 500 * np.exp(np.cumsum(rng.normal(0, vol, len(date))))
    o = np.concatenate(([500.0], c[:-1]))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0003, len(c))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0003, len(c))))
    v = rng.integers(1000, 5000, len(c)).astype(float)
    return D.from_arrays(sym, date, t, o, h, l, c, v, min_bars=200), days


# ── sizing ────────────────────────────────────────────────────────────────
def t_sizing_mis_cnc():
    """the account sizes by 1 % risk, capped by MIS 5x / CNC 1x buying power and the minimum position value"""
    ag = _ag()
    q, m, why = K.size(_c("A", 0, 1000.0, 990.0, 0, 0), ag, 1e6, 1e6)
    assert why is None and q == 1000 and abs(m - 1000 * 1000 / 5.0) < 1e-6, (q, m, why)   # 10,000 / 10
    q, m, why = K.size(_c("A", 0, 1000.0, 999.0, 0, 0), ag, 1e6, 1e6)                      # risk wants 10,000 sh
    assert q == 5000 and abs(m - 1e6) < 1e-6, (q, m)                                        # capped at 5x capital
    q, m, why = K.size(_c("A", 0, 1000.0, 999.0, 0, 0, inst="cnc"), ag, 1e6, 1e6)
    assert q == 1000 and abs(m - 1e6) < 1e-6, (q, m)                                        # CNC 1x
    q, m, why = K.size(_c("A", 0, 1000.0, 999.0, 0, 0), ag, 1e6, 3e5)
    assert q == 1500, q                                                                     # free 3 L x 5
    q, m, why = K.size(_c("A", 0, 1000.0, float("nan"), 0, 0), ag, 1e6, 1e6)
    assert q == 100, q                                                                      # no stop: Rs 1 lakh
    q, m, why = K.size(_c("A", 0, 1000.0, 700.0, 0, 0), _ag(min_value=25000.0), 1e6, 1e6)
    assert why is None and q == 33, (q, why)
    q, m, why = K.size(_c("A", 0, 1000.0, 500.0, 0, 0), _ag(min_value=25000.0), 1e6, 1e6)
    assert q == 0 and why == "too_small", (q, why)


def t_sizing_futures():
    ag = _ag(fut_margin_pct=15.0)
    nocap = _ag(fut_margin_pct=15.0, max_fut_order_value=1e12)
    # index future: lot 65, price 25,000, stop 50 points -> risk lots = 10,000 / (50 x 65) = 3
    q, m, why = K.size(_c("NIFTY", 0, 25000.0, 24950.0, 0, 0, inst="fut", lot=65), nocap, 1e6, 1e6)
    assert why is None and q == 3 * 65 and abs(m - 3 * 65 * 25000 * 0.15) < 1e-6, (q, m, why)
    # margin 2.44 L per lot: 1e6 free fits 4 lots; stop 5 points wants 30 lots -> capped to the order-value cap
    q, m, why = K.size(_c("NIFTY", 0, 25000.0, 24950.0, 0, 0, inst="fut", lot=65), ag, 1e6, 1e6)
    assert q == 65, q                                     # max_fut_order_value 25 L / (65 x 25,000) = 1 lot
    q, m, why = K.size(_c("NIFTY", 0, 25000.0, 24995.0, 0, 0, inst="fut", lot=65), nocap, 1e6, 1e6)
    assert q == 4 * 65, q                                 # no order cap: the margin binds (4 lots)
    # stock future: lot 500 at 1,400, stop 30 -> 10,000 / 15,000 < 1 lot -> skipped
    q, m, why = K.size(_c("RELIANCE", 0, 1400.0, 1370.0, 0, 0, inst="fut", lot=500), ag, 1e6, 1e6)
    assert q == 0 and why == "below_one_lot", (q, why)


# ── account rules ─────────────────────────────────────────────────────────
def t_slots_and_capital():
    cs = [_c("A", 202401010920, 1000, 990, 202401011000, 1005),
          _c("B", 202401010921, 1000, 990, 202401011000, 1005),
          _c("C", 202401010922, 1000, 990, 202401011000, 1005),       # third concurrent: no slot
          _c("D", 202401011000, 1000, 990, 202401011100, 1005)]       # at the exit minute: slot freed first
    r = K.simulate(cs, _ag(max_pos=2))
    taken = [t[1] for t in r["trades"]]
    assert taken == ["A", "B", "D"], taken
    assert r["skipped"]["slots"] == 1
    # capital: two MIS positions at 5x with 1 rupee stops use it all (5,000 sh x 1,000 = 50 L, margin 10 L)
    cs = [_c("A", 202401010920, 1000, 999, 202401011000, 1001), _c("B", 202401010921, 1000, 999, 202401011000, 1001)]
    r = K.simulate(cs, _ag(max_pos=5))
    assert [t[1] for t in r["trades"]] == ["A"] and r["skipped"]["capital"] == 1, r["skipped"]
    # one position per symbol
    cs = [_c("A", 202401010920, 1000, 990, 202401011000, 1001), _c("A", 202401010930, 1000, 990, 202401011100, 1001)]
    r = K.simulate(cs, _ag(max_pos=5))
    assert r["taken"] == 1 and r["skipped"]["symbol_busy"] == 1


def t_priority_time_then_value():
    # same minute, one slot: the higher previous-session traded value wins, whatever the list order
    cs = [_c("LOW", 202401010920, 1000, 990, 202401011000, 1001, prio=1e6),
          _c("HIGH", 202401010920, 1000, 990, 202401011000, 1001, prio=9e9)]
    r = K.simulate(cs, _ag(max_pos=1))
    assert [t[1] for t in r["trades"]] == ["HIGH"], r["trades"]
    # an earlier fill beats a higher value (no hindsight)
    cs = [_c("HIGH", 202401010921, 1000, 990, 202401011000, 1001, prio=9e9),
          _c("EARLY", 202401010920, 1000, 990, 202401011000, 1001, prio=1.0)]
    r = K.simulate(cs, _ag(max_pos=1))
    assert [t[1] for t in r["trades"]] == ["EARLY"], r["trades"]


def t_daily_loss_and_trades_per_day():
    # a 10,000 loss realised at 10:00 stops new entries that day (limit 10,000), not the next day
    cs = [_c("A", 202401010920, 1000, 990, 202401011000, 990.0),          # 1,000 sh x -10 = -10,000 - costs
          _c("B", 202401011030, 1000, 990, 202401011100, 1001),
          _c("C", 202401020920, 1000, 990, 202401021000, 1001)]
    r = K.simulate(cs, _ag(max_pos=5, daily_loss=10000.0))
    assert [t[1] for t in r["trades"]] == ["A", "C"] and r["skipped"]["daily_loss"] == 1, r["trades"]
    cs = [_c(s, 202401010920 + i, 1000, 990, 202401010920 + i + 1, 1001) for i, s in enumerate("ABCD")]
    r = K.simulate(cs, _ag(max_pos=5, max_trades_day=2))
    assert r["taken"] == 2 and r["skipped"]["max_trades_day"] == 2


def t_costs_and_compounding():
    cs = [_c("A", 202401010920, 1000.0, 990.0, 202401011000, 1030.0)]
    r = K.simulate(cs, _ag())
    t = r["trades"][0]
    want_cost = round_trip_cost(1000.0 * 1000, 1030.0 * 1000, "equity_mis")
    assert t[9] == 1000 and abs(t[10] - 30000.0) < 1e-6 and abs(t[11] - want_cost) < 1e-6
    assert abs(t[12] - (30000.0 - want_cost)) < 1e-6
    # compounding: risk measured on capital + realised P&L
    cs = [_c("A", 202401010920, 1000.0, 990.0, 202401011000, 1100.0), _c("B", 202401020920, 1000.0, 990.0,
                                                                         202401021000, 1001.0)]
    r = K.simulate(cs, _ag(compound=True))
    q2 = r["trades"][1][9]
    assert q2 > 1000, q2
    r = K.simulate(cs, _ag(compound=False))
    assert r["trades"][1][9] == 1000


def t_metrics():
    cs = [_c("A", 202401010920, 1000.0, 990.0, 202401011000, 1010.0), _c("A", 202401020920, 1000.0, 990.0,
                                                                         202401021000, 985.0)]
    r = K.simulate(cs, _ag())
    m = K.metrics(r, [20240101, 20240102, 20240103], 20240101, 20240103)
    assert m["trades"] == 2 and abs(m["net"] - sum(t[12] for t in r["trades"])) < 0.01
    assert m["max_dd_rs"] > 0 and m["worst_month"] == "202401"
    assert m["win_pct"] == 50.0


# ── the 1:3 bracket ───────────────────────────────────────────────────────
def _one_bar_rule(i, side, stop_off):
    def fn(C):
        n = len(C.c)
        li, si = np.zeros(n, bool), np.zeros(n, bool)
        (li if side > 0 else si)[i] = True
        st = C.c - side * stop_off
        return G.Sig(li, si, np.zeros(n, bool), np.zeros(n, bool), st if side > 0 else None,
                     st if side < 0 else None, None)
    return fn


def t_rr_bracket_both_sides():
    sd, days = _synth(5)
    B = G.Bars(sd)
    C = G.candles(B, "1m")
    for side in (1, -1):
        S = _one_bar_rule(30, side, 2.0)(C)
        g = G.gen(B, C, S, positional=False, index=False, lo=days[0], hi=days[-1])
        assert len(g["f"]) == 1
        fill, stop = float(g["fill"][0]), float(g["stop"][0])
        assert abs(fill - B.o[31] * (1 + side * G.SLIP_STOCK)) < 1e-9          # next minute's open + slippage
        dist = abs(fill - stop)
        tgt = fill + side * G.RR * dist
        assert abs((tgt - fill) - side * 3.0 * dist) < 1e-9 and side * (fill - stop) > 0
        why = G.REASONS[int(g["reason"][0])]
        xpx = float(g["xpx"][0])
        if why == "target":
            assert side * (xpx - fill) >= 3.0 * dist - 1e-9                    # at the target or better
        elif why == "stop":
            assert side * (xpx - fill) <= -dist + 1e-9                         # at the stop or worse (+ slippage)
        # the account's net = gross - costs.py, and an R multiple of 3 before costs at the target
        cols = G.to_columns(B, g, 0)
        r = G.run_account(cols, ["SYN"], "stocks", "intraday", {}, "T", agcfg=_ag())
        t = r["trades"][0]
        q = t[9]
        assert q == int(10000.0 // dist) or q == int(1e6 * 5 // fill)
        bv, sv = (fill * q, xpx * q) if side > 0 else (xpx * q, fill * q)
        assert abs(t[11] - round_trip_cost(bv, sv, "equity_mis")) < 1e-6
        assert abs(t[12] - (side * (xpx - fill) * q - t[11])) < 0.02


def t_rr_target_hit_hand_built():
    # hand-built bars: long fill at 100.02 (100 open + 2 bps), stop 99.02 -> risk 1.00, target 103.02
    date = [20240102] * 200
    t = list(range(555, 755))
    o = [100.0] * 200
    h = [100.1] * 200
    l = [99.9] * 200
    c = [100.0] * 200
    h[50] = 103.2                                    # trades 0.18 above the target (> one tick): target fill
    sd = D.from_arrays("H", date, t, o, h, l, c, [1000.0] * 200)
    B = G.Bars(sd)
    C = G.candles(B, "1m")
    S = _one_bar_rule(10, 1, 0.98)(C)               # stop = close 100 - 0.98 = 99.02
    g = G.gen(B, C, S, positional=False, index=False, lo=20240102, hi=20240102)
    fill = float(g["fill"][0])
    assert abs(fill - 100.02) < 1e-9 and abs(float(g["stop"][0]) - 99.02) < 1e-9
    assert G.REASONS[int(g["reason"][0])] == "target" and abs(float(g["xpx"][0]) - 103.02) < 1e-9
    # short mirror: target = fill - 3 x risk
    l2 = list(l)
    l2[50] = 96.7
    sd = D.from_arrays("H", date, t, o, [100.1] * 200, l2, c, [1000.0] * 200)
    B = G.Bars(sd)
    C = G.candles(B, "1m")
    S = _one_bar_rule(10, -1, 0.98)(C)               # stop 100.98, fill 99.98 -> risk 1.00, target 96.98
    g = G.gen(B, C, S, positional=False, index=False, lo=20240102, hi=20240102)
    assert abs(float(g["fill"][0]) - 99.98) < 1e-9
    assert G.REASONS[int(g["reason"][0])] == "target" and abs(float(g["xpx"][0]) - 96.98) < 1e-9


def t_engine_matches_sim_fill_intraday():
    """The grid engine's 1-minute fills / stop / target / square-off equal sim.fill_intraday's for the same entry."""
    from . import sim
    from .api import Entry
    from .indicators import build_sessions
    sd, days = _synth(4, seed=11)
    sess = build_sessions(sd)
    B = G.Bars(sd)
    C = G.candles(B, "1m")
    checked = 0
    for k, s in enumerate(sess):
        for i in (20, 120, 250, 340):
            for side in (1, -1):
                stop = float(s.c[i]) - side * 0.6
                e = Entry(i, side, stop=stop, rr=3.0)
                r = sim.fill_intraday(s, e)
                gi = int(B.starts[k]) + i
                n = len(C.c)
                li, si = np.zeros(n, bool), np.zeros(n, bool)
                (li if side > 0 else si)[gi] = True
                st = np.full(n, stop)
                S = G.Sig(li, si, np.zeros(n, bool), np.zeros(n, bool), st, st, None)
                g = G.gen(B, C, S, positional=False, index=False, lo=days[0], hi=days[-1])
                if r is None:
                    assert len(g["f"]) == 0
                    continue
                kx, fill, px, reason = r
                assert len(g["f"]) == 1
                assert abs(float(g["fill"][0]) - fill) < 1e-9
                assert int(g["x"][0]) - int(B.starts[k]) == kx, (int(g["x"][0]) - int(B.starts[k]), kx, reason)
                assert abs(float(g["xpx"][0]) - px) < 1e-9, (float(g["xpx"][0]), px)
                checked += 1
    assert checked >= 20


# ── candles and look-ahead ────────────────────────────────────────────────
def t_candles_every_tf():
    sd, days = _synth(12)
    B = G.Bars(sd)
    for tf in G.TFS:
        C = G.candles(B, tf)
        assert (C.first <= C.last).all() and C.first[0] == 0 and C.last[-1] == len(B) - 1
        assert np.allclose(C.h, [B.h[a:b + 1].max() for a, b in zip(C.first, C.last)])
        if tf in G.TF_MIN:
            per = len(C.c) / len(days)
            assert abs(per - math.ceil(375 / G.TF_MIN[tf])) < 1e-9, (tf, per)
    assert len(G.candles(B, "1d").c) == 12
    assert len(G.candles(B, "1w").c) == len({G._isoweek(d) for d in days})


def t_no_lookahead_across_tfs():
    """Cutting the data at a later date never changes a candidate that was decided and closed before the cut: every
    rule, every timeframe (intraday: candidates of earlier sessions identical; positional: entries before the cut
    identical, exits may only move to the cut's period end)."""
    sd, days = _synth(130, seed=5)
    cut = days[100]
    Bf = G.Bars(sd)
    Bc = G.Bars(sd, hi=cut)
    for tf in G.TFS:
        Cf, Cc = G.candles(Bf, tf), G.candles(Bc, tf)
        for rid, r in G.RULES.items():
            with np.errstate(all="ignore"):
                Sf, Sc = r.fn(Cf), r.fn(Cc)
            gf = G.to_columns(Bf, G.gen(Bf, Cf, Sf, positional=tf in G.POSITIONAL_TFS, index=False, lo=days[0],
                                        hi=days[-1]), 0)
            gc = G.to_columns(Bc, G.gen(Bc, Cc, Sc, positional=tf in G.POSITIONAL_TFS, index=False, lo=days[0],
                                        hi=cut), 0)
            if tf in G.POSITIONAL_TFS:
                kf = {(int(a), int(b), round(float(c), 6), round(float(d), 6)) for a, b, c, d in
                      zip(gf["ekey"], gf["side"], gf["fill"], gf["stop"]) if a // 10000 <= cut}
                kc = {(int(a), int(b), round(float(c), 6), round(float(d), 6)) for a, b, c, d in
                      zip(gc["ekey"], gc["side"], gc["fill"], gc["stop"])}
            else:
                kf = {(int(a), int(b), round(float(c), 6), int(x), round(float(y), 6)) for a, b, c, x, y in
                      zip(gf["ekey"], gf["side"], gf["fill"], gf["xkey"], gf["xpx"]) if a // 10000 < cut}
                kc = {(int(a), int(b), round(float(c), 6), int(x), round(float(y), 6)) for a, b, c, x, y in
                      zip(gc["ekey"], gc["side"], gc["fill"], gc["xkey"], gc["xpx"]) if a // 10000 < cut}
            assert kf == kc, (tf, rid, len(kf ^ kc))


def t_controls_detect_planted_edge():
    sd, days = _synth(80, seed=9)
    B = G.Bars(sd)
    C = G.candles(B, "15m")
    out = {}
    for rid in ("CTRL_RANDOM_1", "CTRL_PLANTED_100"):
        S = G.CONTROL_RULES[rid].fn(C)
        g = G.gen(B, C, S, positional=False, index=False, lo=days[0], hi=days[-1], slip=0.0)
        out[rid] = float(np.mean(g["side"] * (g["xpx"] - g["fill"]))) if len(g["f"]) else 0.0
    assert out["CTRL_PLANTED_100"] > out["CTRL_RANDOM_1"], out


def t_native_wrapper_rr3():
    from .api import Entry, Spec, Swing
    from types import SimpleNamespace
    sp = Spec(id="t_nat", kind="intraday", family="t", desc="", fn=lambda s, p: [Entry(5, 1, stop=99.0, target=105.0)],
              params={})
    w = G.wrap_rr(sp)
    s = SimpleNamespace(atr5=np.full(10, 1.0), c=np.full(10, 100.0))
    e = w.fn(s, sp.p)[0]
    assert e.target is None and e.rr == 3.0 and e.stop == 99.0
    sp2 = Spec(id="t_nat2", kind="intraday", family="t", desc="", fn=lambda s, p: [Entry(5, -1, stop=None)], params={})
    e = G.wrap_rr(sp2).fn(s, sp2.p)[0]
    assert abs(e.stop - 101.5) < 1e-9 and e.rr == 3.0                   # 1.5 x the 5-minute ATR above the close
    sp3 = Spec(id="t_nat3", kind="swing", family="t", desc="", fn=lambda d, p: Swing(1, target_pct=10.0), params={},
               horizon="positional")
    d = SimpleNamespace(h=np.full(30, 101.0), l=np.full(30, 99.0), c=np.full(30, 100.0), date=np.arange(30),
                        symbol="X")
    g = G.wrap_rr(sp3).fn(d, sp3.p)
    assert g.target_pct is None and g.rr == 3.0 and abs(g.stop - (100.0 - 1.5 * 2.0)) < 1e-9


def t_trials_append():
    from . import trials as T
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "trials.csv")
        tr = [("x", "A", 20240102, 1, "09:20", 100.0, 20240102, "10:00", 101.0, 10, 10.0, 2.0, 8.0, "target")]
        row = G.trial_row("tfg_ema_9_21@1m:stocks", era="2023-26", stage="tfgrid-screen", dataset="bars3y",
                          period=(20231110, 20251216), trades=tr, daily=[8.0, 0.0, -1.0], verdict="FAILED SCREEN",
                          fp="f:abc")
        assert G.append_trials([row], p, backup=False) == 1
        assert G.append_trials([row], p, backup=False) == 0              # same evaluation: no second row
        rows = T.TrialsLedger(p).read()
        assert len(rows) == 1 and rows[0]["stage"] == "tfgrid-screen" and rows[0]["era"] == "2023-26"
        assert T.count_trials(rows, "2023-26") == 1


def t_grid_registry():
    cs = G.combos()
    ids = [c[0] for c in cs]
    assert len(ids) == len(set(ids))
    for c in cs:
        assert c[2] in G.TFS and c[4] in ("scalping", "intraday", "swing", "futures")
        if c[3] == "index":
            assert not G.RULES[c[1]].needs_volume
    assert G.instrument("stocks", "swing", 1) == "cnc" and G.instrument("stocks", "swing", -1) == "fut"
    assert G.instrument("stocks", "scalping", -1) == "mis" and G.instrument("index", "futures", 1) == "fut"


TESTS = [(f.__name__, f) for f in (t_sizing_mis_cnc, t_sizing_futures, t_slots_and_capital,
                                     t_priority_time_then_value, t_daily_loss_and_trades_per_day,
                                     t_costs_and_compounding, t_metrics, t_rr_bracket_both_sides,
                                     t_rr_target_hit_hand_built, t_engine_matches_sim_fill_intraday,
                                     t_candles_every_tf, t_no_lookahead_across_tfs, t_controls_detect_planted_edge,
                                     t_native_wrapper_rr3, t_trials_append, t_grid_registry)]


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
