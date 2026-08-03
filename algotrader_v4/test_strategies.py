"""
test_strategies.py — Tests for the positional trend-following strategies
package (strategies/): Donchian S1/S2, MA crossover, TSMOM, sizing,
portfolio caps, cost model, kill criteria, and the EOD engine.
Run: cd algotrader_v4 && python test_strategies.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from strategies import (
    Action,
    DailyBar,
    DonchianBreakoutStrategy,
    KillCriteria,
    MACrossoverStrategy,
    PortfolioRiskBook,
    PositionalEngine,
    TSMOMStrategy,
    Verdict,
    atr,
    donchian,
    fixed_fractional_lots,
    get_contract,
    order_cost,
    round_trip_cost,
    turtle_unit_lots,
    vol_target_lots,
)

# ── Test harness (same style as test_pipeline.py) ──────────────────────────

_results: list[tuple[str, bool, str]] = []

def ok(name: str):
    _results.append((name, True, ""))
    print(f"  ✅  {name}")

def fail(name: str, err: str):
    _results.append((name, False, err))
    print(f"  ❌  {name}: {err}")

def run(name: str, fn):
    try:
        fn()
        ok(name)
    except Exception as exc:
        fail(name, str(exc)[:140])

def section(title: str):
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}")

def summary():
    passed = sum(1 for _, k, _ in _results if k)
    failed = sum(1 for _, k, _ in _results if not k)
    print(f"\n{'═'*60}\n  RESULTS: {len(_results)} tests — "
          f"✅ {passed} passed  ❌ {failed} failed\n{'═'*60}")
    if failed:
        for name, k, err in _results:
            if not k:
                print(f"  ❌ {name}: {err}")
    return failed


# ── Bar builders ───────────────────────────────────────────────────────────

def make_bars(closes: list[float], spread: float = 1.0,
              start: date = date(2025, 1, 1)) -> list[DailyBar]:
    return [DailyBar(date=start + timedelta(days=i), open=c, high=c + spread,
                     low=c - spread, close=c) for i, c in enumerate(closes)]


def drive(strat, symbol: str, bars: list[DailyBar], warmup: int):
    """Feed growing prefixes day-by-day; return every non-HOLD signal."""
    out = []
    for i in range(warmup, len(bars) + 1):
        sig = strat.evaluate(symbol, bars[:i])
        if sig.action != Action.HOLD:
            out.append((i - 1, sig))
    return out


# ═══════════════════════════════════════════════════════════════════════════
section("INDICATORS")
# ═══════════════════════════════════════════════════════════════════════════

def t_atr_constant_range():
    bars = make_bars([100.0] * 30, spread=2.0)   # TR = 4 every bar
    a = atr(bars, 20)
    assert a is not None and abs(a - 4.0) < 1e-9, f"ATR {a} != 4.0"

def t_atr_needs_warmup():
    assert atr(make_bars([100.0] * 10), 20) is None

def t_donchian_excludes_today():
    closes = [100.0] * 25 + [200.0]
    hi, lo = donchian(make_bars(closes), 20)
    assert hi == 101.0 and lo == 99.0, f"channel {hi}/{lo} leaked today's bar"

run("ATR(20) on constant 4-pt range = 4.0",        t_atr_constant_range)
run("ATR returns None during warm-up",             t_atr_needs_warmup)
run("Donchian channel excludes the current bar",   t_donchian_excludes_today)


# ═══════════════════════════════════════════════════════════════════════════
section("POSITION SIZING — spec worked example (Crude Oil)")
# ═══════════════════════════════════════════════════════════════════════════
# Equity ₹10L, 1% risk, ATR(20)=₹150, stop 2N=₹300:
#   CRUDEOIL  (100 bbl): risk/lot ₹30,000 → 0 lots
#   CRUDEOILM (10 bbl):  risk/lot  ₹3,000 → 3 lots

def t_std_crude_doesnt_fit():
    lots = fixed_fractional_lots(1_000_000, 150, get_contract("CRUDEOIL"),
                                 risk_fraction=0.01, atr_multiple=2.0)
    assert lots == 0, f"expected 0 lots, got {lots}"

def t_mini_crude_3_lots():
    lots = fixed_fractional_lots(1_000_000, 150, get_contract("CRUDEOILM"),
                                 risk_fraction=0.01, atr_multiple=2.0)
    assert lots == 3, f"expected 3 lots, got {lots}"

def t_turtle_unit_mini_crude():
    # Unit = floor(10,000 / (150 × 1 × 10)) = 6
    lots = turtle_unit_lots(1_000_000, 150, get_contract("CRUDEOILM"))
    assert lots == 6, f"expected 6 lots, got {lots}"

def t_vol_target_inverse_vol():
    c = get_contract("GOLDM")
    hi = vol_target_lots(10_000_000, 70_000, 0.10, c)   # 200k / 70k  → 2 lots
    lo = vol_target_lots(10_000_000, 70_000, 0.30, c)   # 200k / 210k → 0 lots
    assert hi > lo, f"higher vol must mean smaller position ({hi} vs {lo})"

def t_zero_inputs_zero_lots():
    c = get_contract("CRUDEOILM")
    assert fixed_fractional_lots(0, 150, c) == 0
    assert turtle_unit_lots(1_000_000, 0, c) == 0
    assert vol_target_lots(1_000_000, 0, 0.2, c) == 0

run("CRUDEOIL standard: ₹10L @ 1% risk → 0 lots",   t_std_crude_doesnt_fit)
run("CRUDEOILM mini: ₹10L @ 1% risk → 3 lots",      t_mini_crude_3_lots)
run("Turtle unit CRUDEOILM: N=150 → 6 lots",         t_turtle_unit_mini_crude)
run("Vol-target sizing ∝ 1/volatility",              t_vol_target_inverse_vol)
run("Degenerate inputs size to 0 lots",              t_zero_inputs_zero_lots)


# ═══════════════════════════════════════════════════════════════════════════
section("PORTFOLIO RISK CAPS (Turtle-derived)")
# ═══════════════════════════════════════════════════════════════════════════

def t_market_cap_4_units():
    book = PortfolioRiskBook(equity=10_000_000)
    c = get_contract("CRUDEOILM")
    for _ in range(4):
        allowed, _ = book.can_add(c, "LONG", 3000)
        assert allowed
        book.add(c, "LONG", 3000)
    allowed, why = book.can_add(c, "LONG", 3000)
    assert not allowed and "market cap" in why

def t_cluster_cap_6_units():
    book = PortfolioRiskBook(equity=10_000_000)
    crude, gas = get_contract("CRUDEOILM"), get_contract("NATURALGAS")
    for _ in range(4):
        book.add(crude, "LONG", 3000)
    for _ in range(2):
        book.add(gas, "LONG", 3000)
    allowed, why = book.can_add(gas, "LONG", 3000)
    assert not allowed and "cluster cap" in why

def t_direction_cap_10_units():
    book = PortfolioRiskBook(equity=100_000_000)
    for sym, n in (("CRUDEOILM", 4), ("NATURALGAS", 2), ("GOLDM", 4)):
        book.add(get_contract(sym), "LONG", 1000, units=n)
    allowed, why = book.can_add(get_contract("NIFTY"), "LONG", 1000)
    assert not allowed and "direction cap" in why
    allowed, _ = book.can_add(get_contract("NIFTY"), "SHORT", 1000)
    assert allowed, "opposite direction must not be capped"

def t_heat_cap_6_pct():
    book = PortfolioRiskBook(equity=1_000_000)
    c = get_contract("GOLDM")
    allowed, _ = book.can_add(c, "LONG", 30_000)   # 3%
    assert allowed
    book.add(c, "LONG", 30_000)
    allowed, _ = book.can_add(get_contract("SILVERM"), "LONG", 30_000)  # → 6% exactly
    assert allowed
    book.add(get_contract("SILVERM"), "LONG", 30_000)
    allowed, why = book.can_add(get_contract("NIFTY"), "LONG", 30_000)  # → 9%
    assert not allowed and "heat" in why

def t_remove_by_owner():
    book = PortfolioRiskBook(equity=1_000_000)
    c = get_contract("GOLDM")
    book.add(c, "LONG", 1000, owner="donchian")
    book.add(c, "LONG", 1000, owner="tsmom")
    book.remove("GOLDM", owner="donchian")
    assert book.units_in("GOLDM") == 1, "must only remove donchian's block"

run("Single market capped at 4 units",               t_market_cap_4_units)
run("Energy cluster capped at 6 units/direction",    t_cluster_cap_6_units)
run("Direction capped at 10 units",                  t_direction_cap_10_units)
run("Portfolio heat capped at 6% of equity",         t_heat_cap_6_pct)
run("remove(owner=) leaves other strategies' risk",  t_remove_by_owner)


# ═══════════════════════════════════════════════════════════════════════════
section("STRATEGY 1 — DONCHIAN / TURTLE S1-S2")
# ═══════════════════════════════════════════════════════════════════════════

FLAT60 = [100.0] * 60

def t_donchian_s1_long_entry():
    strat = DonchianBreakoutStrategy()
    bars = make_bars(FLAT60 + [105.0])
    sig = strat.evaluate("CRUDEOILM", bars)
    assert sig.action == Action.ENTER_LONG and sig.system == "S1", sig
    n = atr(bars, 20)
    assert abs(sig.stop - (105.0 - 2 * n)) < 1e-6, f"stop {sig.stop} != entry-2N"

def t_donchian_s1_short_entry():
    strat = DonchianBreakoutStrategy()
    sig = strat.evaluate("CRUDEOILM", make_bars(FLAT60 + [95.0]))
    assert sig.action == Action.ENTER_SHORT and sig.system == "S1", sig

def t_donchian_no_entry_inside_channel():
    strat = DonchianBreakoutStrategy()
    sig = strat.evaluate("CRUDEOILM", make_bars(FLAT60 + [100.5]))
    assert sig.action == Action.HOLD, sig

def t_donchian_s1_skip_after_winner():
    strat = DonchianBreakoutStrategy()
    strat._last_s1_won["CRUDEOILM"] = True
    # Old 110-high keeps the 55-day channel above: 105 breaks ONLY the 20-day
    closes = [110.0] * 30 + [100.0] * 30 + [105.0]
    sig = strat.evaluate("CRUDEOILM", make_bars(closes))
    assert sig.action == Action.HOLD and "skip" in sig.reason.lower(), sig

def t_donchian_s2_takes_skipped_signal():
    strat = DonchianBreakoutStrategy()
    strat._last_s1_won["GOLDM"] = True
    # Flat history: 105 breaks the 20-day AND 55-day highs → S2 safety net
    sig = strat.evaluate("GOLDM", make_bars(FLAT60 + [105.0]))
    assert sig.action == Action.ENTER_LONG and sig.system == "S2", sig

def t_donchian_pyramid_and_max_units():
    strat = DonchianBreakoutStrategy()
    closes = FLAT60 + [105.0]
    bars = make_bars(closes)
    assert strat.evaluate("CRUDEOILM", bars).action == Action.ENTER_LONG
    n = strat.get_position("CRUDEOILM").entry_n
    adds = 0
    px = 105.0
    for _ in range(8):
        px += 0.6 * n                     # > 0.5N favorable move each day
        closes = closes + [px]
        sig = strat.evaluate("CRUDEOILM", make_bars(closes))
        if sig.action == Action.PYRAMID:
            adds += 1
    assert adds == 3, f"expected 3 adds (4 units total), got {adds}"
    assert strat.get_position("CRUDEOILM").units == 4

def t_donchian_stop_exit():
    strat = DonchianBreakoutStrategy()
    closes = FLAT60 + [105.0]
    strat.evaluate("CRUDEOILM", make_bars(closes))
    stop = strat.get_position("CRUDEOILM").stop
    bars = make_bars(closes + [stop - 5])  # closes (and lows) below the stop
    sig = strat.evaluate("CRUDEOILM", bars)
    assert sig.action == Action.EXIT and "stop" in sig.reason.lower(), sig
    assert strat.get_position("CRUDEOILM") is None

def t_donchian_channel_exit():
    strat = DonchianBreakoutStrategy()
    # Steep rise (each close clears the prior day's high) then a fall
    # through the 10-day low, well above the 2N stop
    closes = [100.0 + 3.0 * i for i in range(70)]
    bars = make_bars(closes)
    sigs = drive(strat, "SILVERM", bars, warmup=57)
    assert any(s.action in (Action.ENTER_LONG, Action.PYRAMID) for _, s in sigs)
    pos = strat.get_position("SILVERM")
    assert pos is not None
    # Drop to just below the prior 10-day low but above the stop
    ten_low = min(b.low for b in bars[-10:])
    exit_px = ten_low - 1
    assert exit_px > pos.stop, "test setup: channel exit must precede stop"
    sig = strat.evaluate("SILVERM", make_bars(closes + [exit_px]))
    assert sig.action == Action.EXIT and "10-day" in sig.reason, sig

def t_donchian_tracker_records_loss():
    strat = DonchianBreakoutStrategy()
    closes = FLAT60 + [105.0]
    strat.evaluate("NIFTY", make_bars(closes))         # S1 breakout tracked
    stop = strat._s1_tracker["NIFTY"]["stop"]
    strat.evaluate("NIFTY", make_bars(closes + [stop - 2]))
    assert strat._last_s1_won.get("NIFTY") is False, "2N stop-out must record a loss"

run("S1: close > prior 20-day high → LONG, stop=2N",  t_donchian_s1_long_entry)
run("S1: close < prior 20-day low → SHORT",           t_donchian_s1_short_entry)
run("Inside channel → HOLD",                          t_donchian_no_entry_inside_channel)
run("S1 skipped after a winning S1 breakout",         t_donchian_s1_skip_after_winner)
run("S2 55-day safety net takes skipped signals",     t_donchian_s2_takes_skipped_signal)
run("Pyramids every 0.5N to a max of 4 units",        t_donchian_pyramid_and_max_units)
run("2N stop hit → EXIT",                             t_donchian_stop_exit)
run("Opposite 10-day channel break → EXIT (S1)",      t_donchian_channel_exit)
run("Theoretical S1 tracker records stop-out loss",   t_donchian_tracker_records_loss)


# ═══════════════════════════════════════════════════════════════════════════
section("STRATEGY 2 — MA CROSSOVER + 200-SMA TREND FILTER")
# ═══════════════════════════════════════════════════════════════════════════

def _xover_bars() -> list[DailyBar]:
    # Flat base, mild dip, then strong rise → golden cross above the 200-SMA
    closes = [100.0] * 200 + [98.0 - 0.2 * i for i in range(25)] \
             + [93.0 + 1.2 * i for i in range(45)]
    return make_bars(closes)

def t_ma_enters_long_above_200sma():
    strat = MACrossoverStrategy()
    sigs = drive(strat, "GOLDM", _xover_bars(), warmup=201)
    entries = [s for _, s in sigs if s.action == Action.ENTER_LONG]
    assert entries, "no long entry on golden cross above 200-SMA"
    assert entries[0].system == "MA_XOVER"
    assert entries[0].stop is not None and entries[0].stop < entries[0].price

def t_ma_trend_filter_blocks_below_200sma():
    strat = MACrossoverStrategy()
    # Long decline then a weak bounce: golden cross happens far below 200-SMA
    closes = [200.0 - 0.5 * i for i in range(240)] \
             + [80.0 + 0.8 * i for i in range(30)]
    sigs = drive(strat, "GOLDM", make_bars(closes), warmup=201)
    assert not any(s.action == Action.ENTER_LONG for _, s in sigs), \
        "trend filter must block longs below the 200-SMA"

def t_ma_trailing_stop_ratchets():
    strat = MACrossoverStrategy()
    bars = _xover_bars()
    drive(strat, "GOLDM", bars, warmup=201)
    pos = strat.get_position("GOLDM")
    assert pos is not None, "expected an open long"
    stop_before = pos.stop
    higher = make_bars([b.close for b in bars] + [bars[-1].close + 15])
    strat.evaluate("GOLDM", higher)
    assert strat.get_position("GOLDM").stop > stop_before, "stop must ratchet up"

def t_ma_exit_on_stop():
    strat = MACrossoverStrategy()
    bars = _xover_bars()
    drive(strat, "GOLDM", bars, warmup=201)
    pos = strat.get_position("GOLDM")
    sig = strat.evaluate("GOLDM", make_bars(
        [b.close for b in bars] + [pos.stop - 10]))
    assert sig.action == Action.EXIT and "stop" in sig.reason.lower(), sig

def t_ma_short_requires_flag():
    # Long decline, then a bounce long enough for a real golden cross,
    # then a steep fall → fresh death cross while price < 200-SMA
    closes = [200.0 - 0.5 * i for i in range(240)] \
             + [80.0 + 1.0 * i for i in range(40)] \
             + [119.0 - 2.0 * i for i in range(30)]
    bars = make_bars(closes)
    no_short = MACrossoverStrategy(allow_short=False)
    sigs = drive(no_short, "NIFTY", bars, warmup=201)
    assert not any(s.action == Action.ENTER_SHORT for _, s in sigs)
    shorts_on = MACrossoverStrategy(allow_short=True)
    sigs2 = drive(shorts_on, "NIFTY", bars, warmup=201)
    assert any(s.action == Action.ENTER_SHORT for _, s in sigs2), \
        "death cross below 200-SMA should short when enabled"

run("Golden cross above 200-SMA → ENTER_LONG",        t_ma_enters_long_above_200sma)
run("Golden cross below 200-SMA → blocked",           t_ma_trend_filter_blocks_below_200sma)
run("ATR trailing stop ratchets up, never loosens",   t_ma_trailing_stop_ratchets)
run("Trailing stop hit → EXIT",                       t_ma_exit_on_stop)
run("Shorts only when allow_short=True",              t_ma_short_requires_flag)


# ═══════════════════════════════════════════════════════════════════════════
section("STRATEGY 3 — TSMOM (12-month momentum)")
# ═══════════════════════════════════════════════════════════════════════════

def _up_bars(n=300, start=100.0, step=0.3) -> list[float]:
    return [start + step * i for i in range(n)]

def t_tsmom_long_on_positive_12mo():
    strat = TSMOMStrategy()
    sig = strat.evaluate("GOLDM", make_bars(_up_bars()))
    assert sig.action == Action.ENTER_LONG and sig.system == "TSMOM", sig
    assert sig.weight and sig.weight > 0, "vol-target weight missing"

def t_tsmom_monthly_rebalance_only():
    strat = TSMOMStrategy()
    bars = make_bars(_up_bars())
    strat.evaluate("GOLDM", bars)
    sig = strat.evaluate("GOLDM", bars + [DailyBar(
        bars[-1].date + timedelta(days=1), 190, 191, 189, 190)])
    assert sig.action == Action.HOLD and "rebalanced" in sig.reason, sig

def t_tsmom_exit_on_flip():
    strat = TSMOMStrategy()
    closes = _up_bars()
    bars = make_bars(closes)
    strat.evaluate("GOLDM", bars)                     # long
    # Next month: price collapsed → 252-day return negative
    crash = [closes[-1] - 2.0 * i for i in range(40)]
    sig = strat.evaluate("GOLDM", make_bars(closes + crash))
    assert sig.action == Action.EXIT and "flip" in sig.reason.lower(), sig
    assert strat.get_position("GOLDM") is None

def t_tsmom_long_only_stays_flat():
    strat = TSMOMStrategy(long_only=True)
    closes = [300.0 - 0.5 * i for i in range(300)]    # steady decline
    sig = strat.evaluate("NIFTY", make_bars(closes))
    assert sig.action == Action.HOLD, "long-only must stay flat on negative momentum"

def t_tsmom_shorts_when_enabled():
    strat = TSMOMStrategy(long_only=False)
    closes = [300.0 - 0.5 * i for i in range(300)]
    sig = strat.evaluate("NIFTY", make_bars(closes))
    assert sig.action == Action.ENTER_SHORT, sig

run("Positive 12-mo return → ENTER_LONG + weight",    t_tsmom_long_on_positive_12mo)
run("Second signal same month → HOLD (monthly)",      t_tsmom_monthly_rebalance_only)
run("Momentum sign flip → EXIT",                      t_tsmom_exit_on_flip)
run("long_only stays flat on negative momentum",      t_tsmom_long_only_stays_flat)
run("Shorts on negative momentum when enabled",       t_tsmom_shorts_when_enabled)


# ═══════════════════════════════════════════════════════════════════════════
section("COST MODEL (MCX / NSE futures)")
# ═══════════════════════════════════════════════════════════════════════════

def t_cost_mcx_buy_breakdown():
    c = get_contract("CRUDEOILM")
    b = order_cost(5000, 1, c, "BUY", slippage_ticks=1)   # turnover ₹50,000
    assert abs(b.brokerage - 15.0) < 1e-9, f"brokerage {b.brokerage}"   # 0.03% < ₹20
    assert b.stt_ctt == 0.0, "CTT must not apply on the buy side"
    assert abs(b.stamp - 1.0) < 1e-9, f"stamp {b.stamp}"
    assert abs(b.slippage - 10.0) < 1e-9                  # 1 tick × ₹1 × 10 bbl
    assert b.gst > 0 and b.total > 25

def t_cost_mcx_ctt_sell_only():
    c = get_contract("CRUDEOILM")
    s = order_cost(5000, 1, c, "SELL")
    assert abs(s.stt_ctt - 5.0) < 1e-9, f"CTT {s.stt_ctt} != 0.01% of 50k"
    assert s.stamp == 0.0, "stamp duty is buy-side only"

def t_cost_nse_stt_higher():
    n = get_contract("NIFTY")
    s = order_cost(24000, 1, n, "SELL")
    turnover = 24000 * 75
    assert abs(s.stt_ctt - turnover * 0.0005) < 1e-6, "NSE futures STT 0.05% sell"

def t_cost_brokerage_capped_at_20():
    c = get_contract("GOLD")   # big contract → 0.03% > ₹20
    b = order_cost(70000, 1, c, "BUY")
    assert b.brokerage == 20.0

def t_cost_round_trip_positive():
    c = get_contract("CRUDEOILM")
    rt = round_trip_cost(5000, 5100, 3, c, long=True)
    assert rt > 0

run("MCX buy: brokerage/stamp/slippage per spec",     t_cost_mcx_buy_breakdown)
run("MCX CTT 0.01% on sell side only",                t_cost_mcx_ctt_sell_only)
run("NSE futures STT 0.05% (post-Apr-2026) on sell",  t_cost_nse_stt_higher)
run("Brokerage capped at ₹20/order",                  t_cost_brokerage_capped_at_20)
run("Round trip cost is positive",                    t_cost_round_trip_positive)


# ═══════════════════════════════════════════════════════════════════════════
section("KILL CRITERIA (pre-committed)")
# ═══════════════════════════════════════════════════════════════════════════

def _kc(**kw) -> KillCriteria:
    base = dict(strategy="donchian", backtest_max_dd=0.10,
                backtest_sharpe=1.2, live_start=date(2026, 1, 1))
    base.update(kw)
    return KillCriteria(**base)

def t_kill_dd_halts_even_early():
    kc = _kc()
    v, why = kc.evaluate(date(2026, 2, 1), live_drawdown=0.16)   # > 10%×1.5
    assert v == Verdict.AUTO_HALT, (v, why)

def t_kill_no_kill_inside_window():
    kc = _kc()
    kc.trade_pnls = [-100.0] * 40          # ugly, but window incomplete (2 mo)
    v, why = kc.evaluate(date(2026, 3, 1), live_drawdown=0.05)
    assert v == Verdict.OK and "window" in why, (v, why)

def t_kill_sharpe_floor():
    kc = _kc()
    kc.trade_pnls = [100.0] * 30
    v, why = kc.evaluate(date(2026, 8, 3), live_drawdown=0.05, live_sharpe=0.4)
    assert v == Verdict.KILL and "Sharpe" in why, (v, why)

def t_kill_expectancy_review():
    kc = _kc()
    kc.trade_pnls = [500.0] * 10 + [-200.0] * 30   # rolling 30 all losers
    v, why = kc.evaluate(date(2026, 8, 3), live_drawdown=0.05, live_sharpe=1.0)
    assert v == Verdict.REVIEW and "expectancy" in why, (v, why)

def t_kill_all_clear():
    kc = _kc()
    kc.trade_pnls = [300.0, -100.0] * 20
    v, _ = kc.evaluate(date(2026, 8, 3), live_drawdown=0.05, live_sharpe=1.0)
    assert v == Verdict.OK

run("DD > backtest×1.5 → AUTO_HALT even in window",   t_kill_dd_halts_even_early)
run("No KILL before 6 months + 30 trades",            t_kill_no_kill_inside_window)
run("Live Sharpe < 50% of backtest → KILL",           t_kill_sharpe_floor)
run("Rolling 30-trade expectancy < 0 → REVIEW",       t_kill_expectancy_review)
run("Healthy metrics → OK",                           t_kill_all_clear)


# ═══════════════════════════════════════════════════════════════════════════
section("POSITIONAL ENGINE (EOD orchestration)")
# ═══════════════════════════════════════════════════════════════════════════

def _crude_breakout_bars() -> list[float]:
    return [5000.0] * 60 + [5200.0]

def t_engine_entry_plan():
    eng = PositionalEngine(equity=3_000_000)
    closes = _crude_breakout_bars()
    plans = eng.run_eod({"CRUDEOILM": make_bars(closes, spread=75.0)})
    entries = [p for p in plans if p.action == "BUY"]
    assert entries, "expected a Donchian entry plan"
    p = entries[0]
    assert p.exchange == "MCX" and p.product == "NRML"
    assert p.lots > 0 and p.quantity == p.lots * 10
    assert p.stop is not None and p.stop < 5200
    assert p.tag.startswith("maran_donchian")
    assert eng.book.units_in("CRUDEOILM") >= 1

def t_engine_exit_unwinds():
    eng = PositionalEngine(equity=3_000_000)
    closes = _crude_breakout_bars()
    plans = eng.run_eod({"CRUDEOILM": make_bars(closes, spread=75.0)})
    entry = [p for p in plans if p.action == "BUY"][0]
    # Crash through the 2N stop next day
    plans2 = eng.run_eod({"CRUDEOILM": make_bars(closes + [4300.0], spread=75.0)})
    exits = [p for p in plans2 if p.action == "SELL"]
    assert exits, "expected an exit plan after the stop was breached"
    assert exits[0].lots == entry.lots, "exit must unwind all entered lots"
    assert eng.book.units_in("CRUDEOILM") == 0

def t_engine_respects_heat_cap():
    # Tiny equity → turtle sizing rounds to 0 lots → no plan, no crash
    eng = PositionalEngine(equity=10_000)
    plans = eng.run_eod({"CRUDEOILM": make_bars(_crude_breakout_bars(), spread=75.0)})
    assert all(p.lots > 0 for p in plans), "0-lot plans must be suppressed"

run("EOD run produces sized MCX entry plan",          t_engine_entry_plan)
run("Stop breach next day → full exit plan",          t_engine_exit_unwinds)
run("Unaffordable sizing suppressed cleanly",         t_engine_respects_heat_cap)


# ═══════════════════════════════════════════════════════════════════════════
section("STATE PERSISTENCE (restart-safe)")
# ═══════════════════════════════════════════════════════════════════════════

def t_engine_state_round_trip():
    eng = PositionalEngine(equity=3_000_000)
    closes = [5000.0] * 60 + [5200.0]
    plans = eng.run_eod({"CRUDEOILM": make_bars(closes, spread=75.0)})
    assert plans, "setup: need an open position"
    state = eng.to_state()

    eng2 = PositionalEngine(equity=3_000_000)
    eng2.load_state(state)
    assert eng2.net_quantities() == eng.net_quantities()
    assert eng2.book.units_in("CRUDEOILM") == eng.book.units_in("CRUDEOILM")
    don = eng2.strategies[0]
    pos = don.get_position("CRUDEOILM")
    assert pos is not None and pos.side == "LONG" and pos.stop > 0

    # The restored engine must manage the position (stop-hit exit works)
    plans2 = eng2.run_eod({"CRUDEOILM": make_bars(closes + [4300.0], spread=75.0)})
    assert any(p.action == "SELL" for p in plans2), "restored engine must exit"

def t_state_store_idempotent_queue():
    from strategies.state_store import PositionalStateStore
    store = PositionalStateStore(":memory:")
    from strategies.positional_engine import OrderPlan
    plan = OrderPlan(symbol="CRUDEOILM", exchange="MCX", action="BUY",
                     lots=3, quantity=30, stop=4900.0,
                     strategy="donchian", tag="maran_donchian-entry")
    assert store.queue_plans("2026-08-03", [plan]) == 1
    assert store.queue_plans("2026-08-03", [plan]) == 0, "re-queue must be a no-op"
    pending = store.pending_plans()
    assert len(pending) == 1 and pending[0][1].quantity == 30

def t_state_store_lifecycle():
    from strategies.state_store import PositionalStateStore
    from strategies.positional_engine import OrderPlan
    store = PositionalStateStore(":memory:")
    p = OrderPlan(symbol="GOLDM", exchange="MCX", action="BUY", lots=1,
                  quantity=10, strategy="tsmom", tag="maran_tsmom-entry")
    store.queue_plans("2026-08-01", [p])
    pid = store.pending_plans()[0][0]
    store.mark_plan(pid, "placed", order_id="OID1", gtt_id="G1")
    assert store.pending_plans() == [], "placed plan must leave the queue"
    store.queue_plans("2026-08-02", [p])
    assert store.cancel_stale_pending("2026-08-03") == 1, "old pending → cancelled"

def t_state_store_engine_snapshot():
    from strategies.state_store import PositionalStateStore
    store = PositionalStateStore(":memory:")
    assert store.load_engine_state() is None
    store.save_engine_state({"equity": 1.0})
    store.save_engine_state({"equity": 2.0})
    assert store.load_engine_state() == {"equity": 2.0}, "snapshot must upsert"

def t_state_store_trade_log():
    from strategies.state_store import PositionalStateStore
    store = PositionalStateStore(":memory:")
    store.record_trade("2026-08-01", "donchian", "CRUDEOILM", -500.0)
    store.record_trade("2026-08-02", "donchian", "CRUDEOILM", 1500.0)
    assert store.trade_pnls("donchian") == [-500.0, 1500.0]
    assert store.trade_pnls("tsmom") == []

run("Engine state round-trips through to_state/load_state", t_engine_state_round_trip)
run("Plan queue is idempotent (unique client tag)",         t_state_store_idempotent_queue)
run("Plan lifecycle: placed leaves queue, stale cancelled", t_state_store_lifecycle)
run("Engine snapshot upserts single row",                   t_state_store_engine_snapshot)
run("Trade log feeds kill criteria per strategy",           t_state_store_trade_log)


# ═══════════════════════════════════════════════════════════════════════════
section("LIVE RUNNER HELPERS (rollover / reconcile / data sanity)")
# ═══════════════════════════════════════════════════════════════════════════

from positional_runner import pick_near_month, reconcile, sane_bars

def _fut(name, tsym, expiry):
    return {"name": name, "tradingsymbol": tsym, "expiry": expiry,
            "instrument_type": "FUT", "instrument_token": hash(tsym) % 10**6}

def t_rollover_picks_near_month():
    instruments = [
        _fut("CRUDEOILM", "CRUDEOILM26AUGFUT", date(2026, 8, 18)),
        _fut("CRUDEOILM", "CRUDEOILM26SEPFUT", date(2026, 9, 18)),
        _fut("CRUDEOILM", "CRUDEOILM26OCTFUT", date(2026, 10, 19)),
        _fut("GOLDM",     "GOLDM26AUGFUT",     date(2026, 8, 5)),
        {"name": "CRUDEOILM", "instrument_type": "CE", "expiry": date(2026, 8, 18)},
    ]
    inst = pick_near_month(instruments, "CRUDEOILM", date(2026, 8, 3), 3)
    assert inst["tradingsymbol"] == "CRUDEOILM26AUGFUT"

def t_rollover_respects_buffer():
    instruments = [
        _fut("CRUDEOILM", "CRUDEOILM26AUGFUT", date(2026, 8, 18)),
        _fut("CRUDEOILM", "CRUDEOILM26SEPFUT", date(2026, 9, 18)),
    ]
    # 3 days (or fewer) to expiry → roll to next month
    inst = pick_near_month(instruments, "CRUDEOILM", date(2026, 8, 15), 3)
    assert inst["tradingsymbol"] == "CRUDEOILM26SEPFUT", \
        "within the rollover buffer the next month must be picked"
    assert pick_near_month(instruments, "CRUDEOILM", date(2026, 12, 1), 3) is None

def t_reconcile_match_and_mismatch():
    universe = ["CRUDEOILM", "GOLDM", "NIFTY"]
    broker   = {"CRUDEOILM26AUGFUT": 30, "GOLDM26SEPFUT": -10,
                "RELIANCE": 100}                      # non-universe → ignored
    expected = {"CRUDEOILM": 30, "GOLDM": -10}
    ok, mm = reconcile(broker, expected, universe)
    assert ok and not mm, mm
    expected_bad = {"CRUDEOILM": 30, "GOLDM": -10, "NIFTY": 75}
    ok, mm = reconcile(broker, expected_bad, universe)
    assert not ok and any("NIFTY" in m for m in mm)

def t_sane_bars_rejects_bad_data():
    good = [{"date": date(2026, 8, 1), "open": 100, "high": 101,
             "low": 99, "close": 100, "volume": 10},
            {"date": date(2026, 8, 2), "open": 100, "high": 102,
             "low": 99, "close": 101, "volume": 12}]
    bars = sane_bars(good)
    assert bars and len(bars) == 2 and bars[1].close == 101
    zero = [dict(good[0], close=0)]
    assert sane_bars(zero) is None, "zero close must reject the symbol"
    outlier = [good[0], dict(good[1], close=160)]     # +60% day → bad feed
    assert sane_bars(outlier) is None

run("Near-month FUT contract resolution",             t_rollover_picks_near_month)
run("Rollover buffer skips expiring contract",        t_rollover_respects_buffer)
run("Reconcile matches roots, flags mismatches",      t_reconcile_match_and_mismatch)
run("EOD bar sanity: zero/outlier data rejected",     t_sane_bars_rejects_bad_data)


failed = summary()
sys.exit(1 if failed else 0)
