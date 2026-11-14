"""
stratlab/tests_factor_momentum.py — tests for stratlab/factor_momentum.py (no look-ahead, execution timing, cost
accounting, trend filter, schedule, eligibility, one-look ledger). Run by test_pipeline.py (section "STRATLAB:
factor momentum") or with pytest.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import tempfile

import numpy as np

from costs import round_trip_cost

from . import factor_momentum as fm


def _days(n, start=20200102):
    d = _dt.date(start // 10000, start // 100 % 100, start % 100)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(int(d.strftime("%Y%m%d")))
        d += _dt.timedelta(days=1)
    return out


def _panel(n_sym=30, n_days=700, seed=3, nifty=None, universe=None):
    rng = np.random.default_rng(seed)
    dates = _days(n_days)
    drift = np.linspace(-0.0008, 0.0012, n_sym)
    lr = rng.normal(drift, 0.015, (n_days, n_sym))
    close = 100 * np.exp(np.cumsum(lr, axis=0))
    opn = close * np.exp(rng.normal(0, 0.004, close.shape))
    nf = np.full(n_days, 10000.0) if nifty is None else np.asarray(nifty, float)
    return fm.Panel(dates, [f"S{k:02d}" for k in range(n_sym)], close, opn, nf, universe=universe)


def test_no_lookahead_scores():
    """Scores at decision t are unchanged when every row after t is replaced by garbage."""
    p = _panel()
    for t in fm.decision_sessions(p.dates)[13:20]:
        a = fm.scores(p, t, int(p.dates[t + 1]) // 100)
        c, o = p.close.copy(), p.open.copy()
        c[t + 1:] *= np.random.default_rng(t).uniform(0.2, 5.0, c[t + 1:].shape)
        o[t + 1:] = np.nan
        q = fm.Panel(p.dates, p.symbols, c, o, p.nifty)
        b = fm.scores(q, t, int(p.dates[t + 1]) // 100)
        assert np.allclose(np.nan_to_num(a, nan=-99), np.nan_to_num(b, nan=-99)), t


def test_no_lookahead_backtest_prefix():
    """The equity curve up to a date does not depend on data after it (prefix invariance)."""
    p = _panel()
    first = int(p.dates[fm.decision_sessions(p.dates)[12]])
    full = fm.backtest(p, "monthly", True, first)
    cut = 560
    c = p.close.copy(); c[cut:] *= 3.0
    o = p.open.copy(); o[cut:] *= 0.3
    q = fm.Panel(p.dates, p.symbols, c, o, p.nifty * np.r_[np.ones(cut), np.full(len(p.dates) - cut, 0.5)])
    alt = fm.backtest(q, "monthly", True, first)
    k = int(np.searchsorted(full["dates"], p.dates[cut - 1], side="right"))
    assert np.allclose(full["equity"][:k], alt["equity"][:k])
    assert not np.allclose(full["equity"][k:], alt["equity"][k:])


def test_scores_formula():
    p = _panel(n_sym=5)
    t = fm.decision_sessions(p.dates)[14]
    sc = fm.scores(p, t, int(p.dates[t + 1]) // 100)
    cf = p.cf
    sig = np.diff(np.log(cf[t - 252: t + 1]), axis=0).std(axis=0) * math.sqrt(252)
    v6 = (cf[t - 21] / cf[t - 126] - 1) / sig
    v12 = (cf[t - 21] / cf[t - 252] - 1) / sig
    z = lambda v: np.clip((v - v.mean()) / v.std(), -3, 3)
    assert np.allclose(sc, (z(v6) + z(v12)) / 2)


def test_executes_at_next_open():
    """Every trade is dated on the session after a decision session and is valued at that session's open."""
    p = _panel()
    decs = fm.decision_sessions(p.dates)
    first = int(p.dates[decs[12]])
    r = fm.backtest(p, "monthly", False, first, cost_fn=lambda v, s: (0.0, 0.0))
    exec_days = {int(p.dates[t + 1]) for t in decs}
    assert r["trades"] and all(t[0] in exec_days for t in r["trades"])
    # first rebalance: 20 names x 1/20 of capital at the open, so equity at that open equals capital
    first_trades = [t for t in r["trades"] if t[0] == r["trades"][0][0]]
    assert len(first_trades) == 20 and abs(sum(t[3] for t in first_trades) - fm.CAPITAL) < 1e-6


def test_cost_accounting():
    """Equity loss vs a zero-cost run on the first rebalance equals statutory + slippage exactly, and each buy's
    cost is costs.round_trip_cost(value, 0, 'equity_cnc') + 10 bps."""
    stat, slip = fm.trade_costs(50_000.0, "BUY")
    assert abs(stat - round_trip_cost(50_000.0, 0.0, "equity_cnc")) < 1e-9 and abs(slip - 50.0) < 1e-9
    s_stat, _ = fm.trade_costs(50_000.0, "SELL")
    assert abs(s_stat - round_trip_cost(0.0, 50_000.0, "equity_cnc")) < 1e-9
    assert s_stat > stat - 50_000 * 0.00015 + 15.0 - 1e-6            # DP charge on the sell, stamp only on the buy
    p = _panel()
    first = int(p.dates[fm.decision_sessions(p.dates)[12]])
    a = fm.backtest(p, "monthly", False, first)
    b = fm.backtest(p, "monthly", False, first, cost_fn=lambda v, s: (0.0, 0.0))
    d0 = a["trades"][0][0]
    i = int(np.searchsorted(a["dates"], d0))
    cost0 = sum(t[4] for t in a["trades"] if t[0] == d0)
    expect = sum(sum(fm.trade_costs(t[3], t[2])) for t in a["trades"] if t[0] == d0)
    assert abs(cost0 - expect) < 0.05
    assert abs((b["equity"][i] - a["equity"][i]) - cost0) < 0.05     # same holdings at day one, minus the costs
    assert abs(a["statutory"] + a["slippage"] - sum(t[4] for t in a["trades"])) < 1.0
    assert a["equity"][-1] < b["equity"][-1]


def test_trend_filter_goes_to_cash():
    n = 700
    nifty = np.r_[np.linspace(10000, 15000, 450), np.linspace(15000, 9000, n - 450)]
    p = _panel(nifty=nifty)
    first = int(p.dates[fm.decision_sessions(p.dates)[12]])
    r = fm.backtest(p, "monthly", True, first)
    assert r["cash_switches"] >= 1
    e = r["equity"]
    assert np.allclose(e[-30:], e[-1])                                # flat in cash at the end of the downtrend
    r0 = fm.backtest(p, "monthly", False, first)
    assert not np.allclose(r0["equity"][-30:], r0["equity"][-1])


def test_semiannual_schedule():
    p = _panel()
    decs = fm.decision_sessions(p.dates)
    first = int(p.dates[decs[12]])
    r = fm.backtest(p, "semiannual", False, first)
    days = sorted({t[0] for t in r["trades"]})
    allowed = {int(p.dates[decs[12 + 6 * k] + 1]) for k in range(10) if 12 + 6 * k < len(decs)}
    assert days and set(days) <= allowed


def test_universe_and_history_eligibility():
    p = _panel(n_sym=25)
    t = fm.decision_sessions(p.dates)[14]
    m = int(p.dates[t + 1]) // 100
    uni = {m: {f"S{k:02d}" for k in range(10)}}
    q = fm.Panel(p.dates, p.symbols, p.close, p.open, p.nifty, universe=uni)
    ok = fm.eligible(q, t, m)
    assert ok[:10].all() and not ok[10:].any()
    c = p.close.copy(); c[t - 100: t - 60, 0] = np.nan                 # 40 missing of 253 -> < 90 % real
    q2 = fm.Panel(p.dates, p.symbols, c, p.open, p.nifty)
    assert not fm.eligible(q2, t, m)[0] and fm.eligible(q2, t, m)[1]
    q3 = fm.Panel(p.dates, p.symbols, p.close, p.open, p.nifty, ca_gaps={"S01": [int(p.dates[t - 5])]})
    assert not fm.eligible(q3, t, m)[1]
    assert fm.eligible(p, 100, m).sum() == 0                          # not enough history


def test_cap_and_delisted_exit():
    p = _panel(n_sym=6)
    t = fm.decision_sessions(p.dates)[14]
    sel = fm.select(p, fm.scores(p, t, int(p.dates[t + 1]) // 100))
    assert len(sel) == 6 and all(abs(w - 0.10) < 1e-12 for w in sel.values())   # 6 names -> capped at 10 %
    p = _panel()
    decs = fm.decision_sessions(p.dates)
    first = int(p.dates[decs[12]])
    r = fm.backtest(p, "monthly", False, first, cost_fn=lambda v, s: (0.0, 0.0))
    held0 = {t[1] for t in r["trades"] if t[0] == r["trades"][0][0]}
    j = p.symbols.index(sorted(held0)[0])
    c, o = p.close.copy(), p.open.copy()
    k = decs[13] - 3
    c[k:, j] = np.nan; o[k:, j] = np.nan                               # delisted mid-month
    r2 = fm.backtest(fm.Panel(p.dates, p.symbols, c, o, p.nifty), "monthly", False, first,
                     cost_fn=lambda v, s: (0.0, 0.0))
    assert r2["stale_exits"] == 1
    assert not any(t[1] == p.symbols[j] and t[0] > int(p.dates[decs[13] + 1]) for t in r2["trades"])


def test_prereg_and_ledger_one_look():
    here = os.path.dirname(os.path.abspath(__file__))
    pr = os.path.join(here, "prereg", "momentum_v1.json")
    assert fm.sha256_file(pr) == fm.PREREG_SHA256
    fm.check_prereg(pr)
    with tempfile.TemporaryDirectory() as d:
        led = os.path.join(d, "ledger.json")
        fm.save_ledger(led, {"fm_monthly_filter": {"when": "x", "result": {"verdict": "FAIL"}}})
        bad = os.path.join(d, "pr.json")
        with open(bad, "w") as f:
            json.dump(json.load(open(pr)), f)
        try:
            fm.check_prereg(bad)
            raise AssertionError("a modified prereg file must be refused")
        except SystemExit:
            pass
        rc = fm.main(["--prereg", pr, "--old", d, "--bars", d, "--nse", d, "--ledger", led, "--out",
                      os.path.join(d, "out"), "--variants", "fm_monthly_filter"])
        assert rc == 0 and json.load(open(led))["fm_monthly_filter"]["when"] == "x"   # refused, nothing loaded


TESTS = [
    ("factor momentum: scores ignore every row after the decision close", test_no_lookahead_scores),
    ("factor momentum: equity up to a date is independent of later data", test_no_lookahead_backtest_prefix),
    ("factor momentum: score = mean of clipped z of vol-adjusted 12-1 and 6-1", test_scores_formula),
    ("factor momentum: trades execute at the next session's open", test_executes_at_next_open),
    ("factor momentum: cost accounting (CNC statutory + 10 bps) matches costs.py exactly", test_cost_accounting),
    ("factor momentum: NIFTY < SMA200 -> cash", test_trend_filter_goes_to_cash),
    ("factor momentum: semi-annual variant trades only every 6th month-end", test_semiannual_schedule),
    ("factor momentum: universe, 90 % history and CA-gap eligibility", test_universe_and_history_eligibility),
    ("factor momentum: 10 % cap and stale exit of a delisted holding", test_cap_and_delisted_exit),
    ("factor momentum: prereg sha256 enforced, ledger refuses a re-run", test_prereg_and_ledger_one_look),
]
