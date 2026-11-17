"""Tests of the Rs 10 lakh-per-agent size rule (stratlab/size10l_run.py). No market data needed.

    cd algotrader_v4 && python -m stratlab.tests_size10l
"""
from __future__ import annotations

from . import sim
from . import size10l_run as Z
from .api import Spec


def _spec(kind, underlying=None, horizon="intraday"):
    return Spec(f"T_{kind}_{underlying}", kind, "test", "test", lambda s, p: [], {}, horizon=horizon,
                underlying=underlying)


def t_size_by_cost_kind():
    assert Z.notional_for("equity_mis") == 500_000.0
    assert Z.notional_for("future") == 500_000.0                    # a STOCK at futures costs
    assert Z.notional_for("equity_cnc") == 100_000.0                # delivery: unchanged
    assert Z.notional_for("future", "NIFTY") is None                # index future: lots, not a notional
    assert Z.size_changed("equity_mis") and Z.size_changed("future")
    assert not Z.size_changed("equity_cnc") and not Z.size_changed("future", "BANKNIFTY")
    try:
        Z.notional_for("bond")
        raise AssertionError("an unknown cost kind must be refused")
    except ValueError:
        pass


def t_wrapper_changes_only_the_quantity():
    args = ("X", 20240102, 1, "09:30", 1234.56, 20240102, "10:00", 1240.1, "time")
    for kind, mult in (("intraday", 5), ("scalping", 5), ("futures", 5), ("swing", 1)):
        sp = _spec(kind, horizon="intraday" if kind in ("intraday", "scalping") else "positional")
        base = sim._settle(sp, *args)
        Z.install()
        try:
            t = sim._settle(sp, *args)
        finally:
            Z.uninstall()
        assert t[:9] == base[:9] and t[13] == base[13], "fills, exits and the reason never change"
        assert t[9] == max(1, int(mult * 100_000 // 1234.56)), (kind, t[9])
        assert abs(t[10] - (1240.1 - 1234.56) * t[9]) < 0.01
        assert t.l1 == (base[9], base[10], base[11], base[12]), "the Rs 1 lakh twin is sim._settle's own trade"
        if mult == 1:
            assert tuple(t) == base
    assert sim._settle.__module__ == "stratlab.sim", "uninstall restores the lab's function"


def t_forced_one_lakh_is_the_lab_trade():
    sp = _spec("intraday")
    args = ("X", 20240102, -1, "09:30", 250.05, 20240102, "09:45", 248.4, "target")
    base = sim._settle(sp, *args)
    Z.install(force=Z.LAKH)
    try:
        t = sim._settle(sp, *args)
    finally:
        Z.uninstall()
    assert tuple(t) == base


def t_fee_per_rupee_falls_with_size():
    sp = _spec("intraday")
    Z.install()
    try:
        t = sim._settle(sp, "X", 20240102, 1, "09:30", 1000.0, 20240102, "10:00", 1000.0, "time")
    finally:
        Z.uninstall()
    q1, _g1, c1, _n1 = t.l1
    assert t[9] == 5 * q1 and c1 < t[11] < 5 * c1, "the flat Rs 20 per order is spread over five times the value"
    assert t[11] / (t[9] * 1000.0) < 0.6 * c1 / (q1 * 1000.0)


def t_index_future_untouched():
    sp = _spec("futures", underlying="NIFTY", horizon="positional")
    args = ("_NIFTY50_min", 20240102, 1, "09:30", 21000.0, 20240103, "15:29", 21100.0, "max_hold")
    base = sim._settle(sp, *args)
    Z.install()
    try:
        t = sim._settle(sp, *args)
    finally:
        Z.uninstall()
    assert t == base and t[9] == sim.LOT_SIZES["NIFTY"]


def t_daily_series_books_on_the_exit_session():
    cal = [20240101, 20240102, 20240103, 20240105]
    pnl = Z.daily_series({20240102: 10.0, 20240104: 5.0, 20240110: -3.0}, cal, 20240101, 20240105)
    assert pnl == [0.0, 10.0, 0.0, 2.0]          # a missing exit day -> the next session; after hi -> the last


def t_catalogue_split():
    ch, un = Z.split_ids()
    from .api import REGISTRY
    assert len(ch) + len(un) == len(REGISTRY) and not set(ch) & set(un)
    assert all(REGISTRY[i].cost_kind in ("equity_mis", "future") and not REGISTRY[i].underlying for i in ch)
    assert all(REGISTRY[i].cost_kind == "equity_cnc" or REGISTRY[i].underlying for i in un)


def t_option_lot_rule():
    from . import size10l_options as O
    ce, pe = "CE", "PE"
    # long straddle: 190 points x 75 = Rs 14,250 of premium per lot -> 7 lots
    assert O.new_lots([(1, 1, 20000, ce, 100, 1), (1, 1, 20000, pe, 90, 1)], 75) == (7, "premium")
    # debit spread: 35 points x 75 = Rs 2,625 -> 38 lots
    assert O.new_lots([(1, 1, 20000, ce, 60, 1), (-1, 1, 20200, ce, 25, 1)], 75) == (38, "premium")
    # iron condor: credit 29, wings 400 -> max risk 371 x 75 = Rs 27,825 -> 3 lots
    ic = [(-1, 1, 20400, ce, 20, 1), (-1, 1, 19600, pe, 18, 1), (1, 1, 20800, ce, 5, 1), (1, 1, 19200, pe, 4, 1)]
    assert O.new_lots(ic, 75) == (3, "max_risk")
    # credit spread: width 200 - credit 35 = 165 x 75 = Rs 12,375 -> 8 lots
    assert O.new_lots([(-1, 1, 20000, ce, 60, 1), (1, 1, 20200, ce, 25, 1)], 75) == (8, "max_risk")
    # naked strangle: the record's margin, else 12 % of the mean short strike x lot (Rs 1,80,000) -> 2 lots
    ns = [(-1, 1, 20400, ce, 20, 1), (-1, 1, 19600, pe, 18, 1)]
    assert O.new_lots(ns, 75, 174220.0) == (2, "margin") and O.new_lots(ns, 75) == (2, "margin")
    assert O.new_lots(ns, 75, 600000.0) == (1, "margin"), "never below one lot"
    # a credit structure with legs in two expiries is sized as naked
    assert O.new_lots([(-1, 1, 20000, ce, 60, 1), (1, 1, 20200, ce, 25, 2)], 75)[1] == "margin"


def t_option_fees_flat_part_does_not_scale():
    from . import size10l_options as O
    assert O.flat_orders([(23.7, 112.68), (21.38, 0.0)]) == 3          # a leg settled at zero pays one order
    f1 = 81.10
    assert O.resize_fees(f1, 3, 1.0) == f1
    f2 = O.resize_fees(f1, 3, 2.0)
    assert f1 < f2 < 2 * f1 and abs(f2 - (3 * 23.6 + 2 * (f1 - 3 * 23.6))) < 0.01


TESTS = [
    ("size10l: Rs 5 lakh for MIS and stock-futures trades, Rs 1 lakh delivery, index lots untouched", t_size_by_cost_kind),
    ("size10l: the wrapper changes the quantity only and carries the Rs 1 lakh twin", t_wrapper_changes_only_the_quantity),
    ("size10l: forced to Rs 1 lakh the wrapper returns sim._settle's trade", t_forced_one_lakh_is_the_lab_trade),
    ("size10l: the fee per rupee falls at Rs 5 lakh (flat brokerage)", t_fee_per_rupee_falls_with_size),
    ("size10l: index futures are not resized", t_index_future_untouched),
    ("size10l: daily series by exit session (trials.daily_pnl convention)", t_daily_series_books_on_the_exit_session),
    ("size10l: catalogue split into re-tested and unchanged ids", t_catalogue_split),
    ("size10l options: lots from a premium / max-risk / margin budget, entry prices only", t_option_lot_rule),
    ("size10l options: the flat brokerage per order does not scale with the lots", t_option_fees_flat_part_does_not_scale),
]


if __name__ == "__main__":
    for _n, _f in TESTS:
        _f()
        print("ok  ", _n)
