"""
stratlab/tests_ltr.py — tests for stratlab/ltr.py (look-ahead, costs, sizing, corporate actions, protocol).
Run by test_pipeline.py (section "STRATLAB: long-term reversal basket").
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import tempfile

import numpy as np

from . import ltr as _ltr


def _days(n, start=20230102):
    """n consecutive weekday dates (yyyymmdd) from start."""
    d = _dt.date(start // 10000, start // 100 % 100, start % 100)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(int(d.strftime("%Y%m%d")))
        d += _dt.timedelta(days=1)
    return out


def _panel(n_sym=30, n_hist=260, n_min=420, seed=5, paths=None):
    """Synthetic panel: n_hist history-only days (close only) then n_min minute days (open, close, 15:14 px)."""
    rng = np.random.default_rng(seed)
    days = _days(n_hist + n_min)
    hd, md = days[:n_hist], days[n_hist:]
    per, hist = {}, {}
    for k in range(n_sym):
        s = f"S{k:02d}"
        if paths is not None and s in paths:
            c = np.asarray(paths[s], float)
        else:
            c = 100.0 * np.exp(np.cumsum(rng.normal(0.0004 * (k % 5 - 2), 0.015, len(days))))
        hist[s] = {d: (float(c[i]), float(c[i])) for i, d in enumerate(hd)}
        m = c[n_hist:]
        per[s] = {"date": np.array(md, np.int64), "o": m * 0.999, "c": m.copy(), "px": m * 1.0005}
    return _ltr.build_panel(per, hist), md


def t_costs():
    b = _ltr.cost_side(100_000.0, "buy")
    s = _ltr.cost_side(100_000.0, "sell")
    exch_sebi = 2.97 + 0.10
    assert abs(b - (100.0 + exch_sebi + 0.18 * exch_sebi + 15.0)) < 1e-6, b          # STT, NSE, SEBI, GST, stamp
    assert abs(s - (100.0 + exch_sebi + 0.18 * exch_sebi + 15.34 * 1.18)) < 1e-6, s  # ... + DP 15.34 + GST, no stamp
    assert _ltr.cost_side(0.0, "sell") == 0.0
    from costs import round_trip_cost
    assert abs(_ltr.cost_side(1e5, "sell", "costs_py") - round_trip_cost(0.0, 1e5, "equity_cnc")) < 1e-9
    # in the simulation: buys fill 2.1 bps above the 15:14 price, sells 2.1 bps below, costs charged per side
    p, md = _panel()
    r = _ltr.simulate_period(p, md[0], md[-1])
    first = [t for t in r["absolute"] if t[2] == md[0]]
    assert len(first) == 10 and all(t[13].startswith("bought") for t in first)
    j = int(np.searchsorted(p.dates, md[0]))
    for t in first:
        i = p.symbols.index(t[1])
        assert abs(t[5] - p.px[i, j] * (1 + 2.1e-4)) < 1e-3, t
        assert abs(t[11] - _ltr.cost_side(t[9] * t[5], "buy") - (_ltr.cost_side(t[9] * t[8], "sell")
                                                                 if t[13].endswith("sold") else 0.0)) < 0.02, t
    sold = [t for t in r["absolute"] if t[13].endswith("sold")]
    assert sold
    for t in sold:
        i = p.symbols.index(t[1])
        jj = int(np.searchsorted(p.dates, t[6]))
        assert abs(t[8] - p.px[i, jj] * (1 - 2.1e-4)) < 1e-3, t
        assert t[11] >= _ltr.cost_side(t[9] * t[8], "sell") - 0.01                  # sell cost (with DP) included
    r5 = _ltr.simulate_period(p, md[0], md[-1], slip_bps=5.0)
    assert sum(t[12] for t in r5["absolute"]) < sum(t[12] for t in r["absolute"])


def t_signal_skips_latest_month():
    n_hist, n_min = 260, 60
    n = n_hist + n_min
    paths = {f"S{k:02d}": np.full(n, 100.0 + k) for k in range(12)}
    a = np.full(n, 100.0)
    a[n_hist - 10:] = 70.0              # falls 30 % inside the skipped month only: not a 12-1 loser
    paths["S00"] = a
    b = np.full(n, 100.0)
    b[n_hist - 240:] = 80.0             # fell 20 % about a year ago and stayed: the biggest 12-1 loser
    paths["S01"] = b
    p, md = _panel(n_sym=12, n_hist=n_hist, n_min=n_min, paths=paths)
    j = int(np.searchsorted(p.dates, md[0]))
    sig = _ltr.signal(p, j)
    assert abs(sig["S00"]) < 1e-12 and abs(sig["S01"] - (80.0 / 100.0 - 1)) < 1e-12, (sig["S00"], sig["S01"])
    assert _ltr.select(sig, 1) == ["S01"]
    # today's close is never read
    before = _ltr.signal(p, j)
    p.close[:, j] *= 3.0
    assert _ltr.signal(p, j) == before
    p.close[:, j] /= 3.0
    # fewer than 252 valid closes before the day: not eligible
    p.close[2, :j - 251] = np.nan
    s2 = _ltr.signal(p, j)
    assert "S02" not in s2 and "S03" in s2
    # no 15:14 price on the day: not eligible (cannot be traded)
    p.px[4, j] = np.nan
    assert "S04" not in _ltr.signal(p, j)


def t_no_lookahead():
    p, md = _panel(seed=9)
    lo, hi = md[0], md[300]
    base = _ltr.simulate_period(p, lo, hi)
    rng = np.random.default_rng(1)
    q = _ltr.Panel(p.symbols, p.dates, p.close.copy(), p.px.copy(), p.opn.copy(), p.sessions, [])
    k = int(np.searchsorted(q.dates, hi, side="right"))
    for arr in (q.close, q.px, q.opn):                      # 1) data after hi is never read
        arr[:, k:] *= rng.uniform(0.3, 3.0, arr[:, k:].shape)
    assert _ltr.simulate_period(q, lo, hi) == base
    x = md[150]                                             # 2) the future after day x changes nothing up to x
    k = int(np.searchsorted(q.dates, x, side="right"))
    for arr in (q.close, q.px, q.opn):
        arr[:, k:] *= rng.uniform(0.3, 3.0, arr[:, k:].shape)
    alt = _ltr.simulate_period(q, lo, hi)
    rb = [r for r in base["rebalances"] if r["date"] <= x]
    assert rb and rb == [r for r in alt["rebalances"] if r["date"] <= x]
    tb = [t for t in base["excess"] if t[6] <= x]
    assert tb and tb == [t for t in alt["excess"] if t[6] <= x]
    assert [t for t in base["excess"] if t[6] > x] != [t for t in alt["excess"] if t[6] > x]
    # 3) a run trades only inside [lo, hi] and ends flat on hi
    assert all(lo <= t[2] <= hi and lo <= t[6] <= hi for t in base["absolute"])
    assert base["periods"][-1]["end"] == hi and base["rebalances"][-1]["date"] < hi


def t_sizing_and_schedule():
    p, md = _panel(seed=13)
    r = _ltr.simulate_period(p, md[0], md[-1])
    dates = [x["date"] for x in r["rebalances"]]
    assert dates == md[0:len(md) - 1:21] and len(dates) == math.ceil((len(md) - 1) / 21), dates
    assert all(len(x["held"]) <= 10 and x["cash"] >= 0 for x in r["rebalances"])
    qty = {}
    for t in sorted(r["absolute"], key=lambda t: t[2]):
        if t[13].startswith("bought"):
            assert t[9] * t[5] <= 100_000.0 + 1e-6, t                                   # never above Rs 1 lakh
            qty[t[1]] = t[9]
        else:
            assert t[9] == qty[t[1]], t                                                # kept: not topped up
    nav = r["periods"][-1]["nav_end"]
    total = 1_000_000.0 + sum(t[12] for t in r["absolute"])
    assert abs(nav - total) < 1.0, (nav, total)                                       # the books add up
    for per in r["periods"]:
        assert abs(per["excess"] - (per["ret"] - per["universe"])) < 1e-12


def t_corporate_action_gap_excluded():
    n_hist, n_min = 260, 80
    n = n_hist + n_min
    days = _days(n)
    paths = {f"S{k:02d}": np.full(n, 100.0 + k) for k in range(12)}
    a = np.full(n, 100.0)
    a[n_hist - 240:] = 60.0                       # a 12-1 loser, bought on day 1 ...
    g = n_hist + 30
    a[g:] = 30.0                                  # ... then a 50 % unadjusted demerger gap
    paths["S00"] = a
    p, md = _panel(n_sym=12, n_hist=n_hist, n_min=n_min, paths=paths)
    i = p.symbols.index("S00")
    j = int(np.searchsorted(p.dates, days[g]))
    p.opn[i, j] = 30.0                            # it opens at the post-gap price
    raw = _ltr.simulate_period(p, md[0], md[-1], gaps={})
    adj = _ltr.simulate_period(p, md[0], md[-1], gaps={("S00", days[g]): "test demerger"})

    def pnl(r):
        return sum(t[10] for t in r["absolute"] if t[1] == "S00")
    assert pnl(raw) < -25_000 and abs(pnl(adj)) < 1_000, (pnl(raw), pnl(adj))
    assert adj["gaps"] and adj["gaps"][0][:2] == ("S00", days[g])
    r2 = _ltr.simulate_period(p, md[0], md[-1], gaps={("S00", md[21]): "test"})   # a rebalance on the gap day
    rb = [x for x in r2["rebalances"] if x["date"] == md[21]][0]
    assert "S00" in rb["skipped_gap"] and "S00" not in rb["sold"] and "S00" not in rb["bought"]
    assert set(_ltr.CA_GAPS) == {("SIEMENS", 20250407), ("VEDL", 20260430)}


def t_protocol_final_only_for_a_pass():
    p, md = _panel(n_sym=20, n_min=420, seed=21)
    with tempfile.TemporaryDirectory() as td:
        ledger = os.path.join(td, "ledger.json")
        o = _ltr.run_protocol(p, ledger)
        sp = o["split"]
        assert (sp.discovery[0], sp.final[0]) == (md[0], md[int(len(md) * 0.7)])
        if o["result"]["verdict"] == "FAILED SCREEN":
            assert o["calls"] == [sp.discovery] and "final" not in o and not os.path.exists(ledger)
        else:
            assert o["calls"] == [sp.discovery, sp.final] and os.path.exists(ledger)
        assert all(t[6] <= sp.discovery[1] for t in o["disc"]["excess"])     # DISCOVERY never uses FINAL prices
        o2 = _ltr.run_protocol(p, os.path.join(td, "l2.json"), screen_only=True)
        assert o2["calls"] == [sp.discovery] and "final" not in o2
        assert o["sens"]["abs"]["trades"] == len(o["disc"]["absolute"])
    # a book whose excess is all in one symbol fails the 35 % gate even if everything else passes
    from . import protocol as P
    rows = [(_ltr.STRATEGY_ID, "ONE" if k % 10 == 0 else f"X{k % 10}", md[k % 290], 1, "", 0, md[k % 290], "", 0, 1,
             0, 0, 1000.0 if k % 10 == 0 else 1.0, "") for k in range(300)]
    sc = P.screen(rows, sp, "positional")
    assert not sc["passed"] and any("ONE" in x for x in sc["reasons"]), sc["reasons"]


TESTS = [
    ("ltr: CNC costs (STT, NSE, stamp, SEBI, GST, DP+GST) and 2.1 bps slippage per side", t_costs),
    ("ltr: signal = C[t-21]/C[t-252]-1, latest month skipped, >= 252 closes, today's close unread",
     t_signal_skips_latest_month),
    ("ltr: no look-ahead (data after hi unread; perturbing the future keeps past decisions)", t_no_lookahead),
    ("ltr: every 21 sessions, <= Rs 1 lakh per buy, kept names not topped up, books add up", t_sizing_and_schedule),
    ("ltr: unadjusted corporate-action gap excluded; no trade in the name on the gap day",
     t_corporate_action_gap_excluded),
    ("ltr: protocol screen on DISCOVERY excess; FINAL simulated only for a pass", t_protocol_final_only_for_a_pass),
]
