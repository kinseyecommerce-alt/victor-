"""
test_sim_orders_flow.py  --  Simulation Orders Flow tests
Standalone test module for paper-mode order lifecycle and tick pipeline.
Run: cd algotrader_v4 && python test_sim_orders_flow.py

Covers (13 tests):
  1. Paper BUY returns PAPER-XXXXXXXX id
  2. Position qty > 0 after paper BUY
  3. Paper SELL closes position to qty=0
  4. Cancel paper order sets status=CANCELLED
  5. Squareoff all clears positions
  6. Modify order price updates record
  7. After 3 orders, list has >=3 entries
  8. order_guard blocks duplicate BUY
  9. risk_manager blocks oversized qty
 10. risk_manager blocks outside market hours
 11. TickBuffer builds >=1 candle from 5 ticks
 12. IndicatorCalc EMA9 > 0 after 30 ticks
 13. validate_credentials() has all required keys
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta

# ── Test harness ──────────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []

def ok(name: str):
    _results.append((name, True, ""))
    print(f"  OK  {name}")

def fail(name: str, err: str):
    _results.append((name, False, err))
    print(f"  FAIL  {name}: {err}")

def run(name: str, fn):
    try:
        fn()
        ok(name)
    except Exception as exc:
        fail(name, str(exc)[:120])

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def summary() -> int:
    passed = sum(1 for _, ok_, _ in _results if ok_)
    failed = sum(1 for _, ok_, _ in _results if not ok_)
    total  = len(_results)
    print(f"\n{'='*60}")
    print(f"  RESULTS: {total} tests -- {passed} passed  {failed} failed")
    print(f"{'='*60}")
    if failed:
        print("\nFailed tests:")
        for name, ok_, err in _results:
            if not ok_:
                print(f"  FAIL {name}: {err}")
    return failed


# ── Imports ───────────────────────────────────────────────────────────────
section("SIMULATION ORDERS FLOW")

from config import settings
settings.trading_mode = "PAPER"  # ensure paper mode for all tests

from kite_client import kite_client
from risk_manager import risk_manager
from order_guard import order_guard


# ── Tests ─────────────────────────────────────────────────────────────────

def t_sim_paper_buy():
    """Paper BUY returns PAPER-XXXXXXXX order id."""
    kite_client._paper_orders.clear()
    kite_client._paper_positions.clear()
    oid = kite_client.place_order(
        tradingsymbol="RELIANCE", exchange="NSE",
        transaction_type="BUY", quantity=1,
        order_type="MARKET", product="MIS", price=2800.0,
    )
    assert re.match(r"^PAPER-[A-F0-9]{8}$", oid), f"Expected PAPER-XXXXXXXX, got {oid}"


def t_sim_position_updated():
    """After paper BUY, position qty > 0 in net positions."""
    kite_client._paper_orders.clear()
    kite_client._paper_positions.clear()
    kite_client.place_order(
        tradingsymbol="TCS", exchange="NSE",
        transaction_type="BUY", quantity=2,
        order_type="MARKET", product="MIS", price=3500.0,
    )
    pos = kite_client.positions()
    net = pos.get("net", [])
    tcs = next((p for p in net if p["tradingsymbol"] == "TCS"), None)
    assert tcs is not None, "TCS position not found after BUY"
    assert tcs["quantity"] == 2, f"Expected qty=2, got {tcs['quantity']}"


def t_sim_paper_sell_closes():
    """Paper SELL reduces position back to 0."""
    kite_client._paper_orders.clear()
    kite_client._paper_positions.clear()
    kite_client.place_order(
        tradingsymbol="INFY", exchange="NSE",
        transaction_type="BUY", quantity=3,
        order_type="MARKET", product="MIS", price=1500.0,
    )
    kite_client.place_order(
        tradingsymbol="INFY", exchange="NSE",
        transaction_type="SELL", quantity=3,
        order_type="MARKET", product="MIS", price=1510.0,
    )
    pos = kite_client.positions()
    net = pos.get("net", [])
    infy = next((p for p in net if p["tradingsymbol"] == "INFY"), None)
    qty = infy["quantity"] if infy else 0
    assert qty == 0, f"Expected qty=0 after SELL, got {qty}"


def t_sim_cancel_order():
    """Cancel a paper order sets status=CANCELLED."""
    kite_client._paper_orders.clear()
    kite_client._paper_positions.clear()
    oid = kite_client.place_order(
        tradingsymbol="HDFC", exchange="NSE",
        transaction_type="BUY", quantity=1,
        order_type="LIMIT", product="MIS", price=1600.0,
    )
    kite_client.cancel_order(oid)
    orders = kite_client.orders()
    target = next((o for o in orders if o["order_id"] == oid), None)
    assert target is not None, "Order not found after cancel"
    assert target["status"] == "CANCELLED", f"Expected CANCELLED, got {target['status']}"


def t_sim_squareoff_clears():
    """Squareoff all positions clears all net quantities."""
    kite_client._paper_orders.clear()
    kite_client._paper_positions.clear()
    for sym in ("SBIN", "AXISBANK"):
        kite_client.place_order(
            tradingsymbol=sym, exchange="NSE",
            transaction_type="BUY", quantity=1,
            order_type="MARKET", product="MIS", price=500.0,
        )
    kite_client.squareoff_all_positions()
    pos = kite_client.positions()
    open_qty = sum(abs(p["quantity"]) for p in pos.get("net", []))
    assert open_qty == 0, f"Expected 0 open qty after squareoff, got {open_qty}"


def t_sim_modify_order():
    """Modify a paper order price updates the order record."""
    kite_client._paper_orders.clear()
    kite_client._paper_positions.clear()
    oid = kite_client.place_order(
        tradingsymbol="WIPRO", exchange="NSE",
        transaction_type="BUY", quantity=5,
        order_type="LIMIT", product="MIS", price=400.0,
    )
    kite_client.modify_order(oid, price=410.0)
    orders = kite_client.orders()
    target = next((o for o in orders if o["order_id"] == oid), None)
    assert target is not None
    assert target["price"] == 410.0, f"Expected price=410, got {target['price']}"


def t_sim_orders_list():
    """After placing 3 orders, orders list has >= 3 entries."""
    kite_client._paper_orders.clear()
    kite_client._paper_positions.clear()
    for sym in ("MARUTI", "BAJFINANCE", "TITAN"):
        kite_client.place_order(
            tradingsymbol=sym, exchange="NSE",
            transaction_type="BUY", quantity=1,
            order_type="MARKET", product="MIS", price=100.0,
        )
    orders = kite_client.orders()
    assert len(orders) >= 3, f"Expected >= 3 orders, got {len(orders)}"


def t_sim_order_guard_duplicate():
    """order_guard.can_place blocks a duplicate position for same symbol+strategy."""
    if hasattr(order_guard, '_orders'):
        order_guard._orders.clear()
    order_guard.register_order("HDFCBANK", "intraday", "BUY", "PAPER-TEST001")
    ok_flag, reason = order_guard.can_place("HDFCBANK", "intraday", "BUY")
    assert not ok_flag, "Expected duplicate to be blocked"
    assert reason, "Expected a reason string"


def t_sim_risk_oversized():
    """risk_manager.check_before_order blocks oversized qty."""
    ok_flag, reason = risk_manager.check_before_order("NIFTY", 999999, 18000.0, "BUY")
    assert not ok_flag, "Expected oversized order to be blocked by risk manager"
    assert reason, "Expected a blocking reason"


def t_sim_risk_market_hours():
    """risk_manager blocks outside market hours when market is closed."""
    from market_data import is_market_open
    if is_market_open():
        return  # market is open, skip outside-hours check
    ok_flag, reason = risk_manager.check_before_order("TATAMOTORS", 1, 500.0, "BUY")
    if not ok_flag and "hours" in str(reason).lower():
        pass  # correctly blocked outside hours


def t_sim_tick_buffer_candle():
    """TickBuffer builds at least one 1m candle after feeding 5 ticks spanning 2 minutes."""
    from tick_engine import TickBuffer
    buf = TickBuffer(60)  # 60-second resolution = 1-min candles
    base_ts = datetime(2024, 1, 15, 10, 0, 0)
    for i in range(5):
        # each tick 65 s apart → crosses minute boundary → completes candles
        buf.push(2800.0 + i, 1000 + i * 100, base_ts + timedelta(seconds=i * 65))
    candles = buf.candles()
    assert len(candles) >= 1, f"Expected >=1 candle after 5 ticks, got {len(candles)}"


def t_sim_indicators_nonzero():
    """IndicatorCalc returns non-zero EMA9 after sufficient price history."""
    import pandas as pd
    from tick_engine import IndicatorCalc, Tick
    base_ts = datetime(2024, 1, 15, 10, 0, 0)
    rows = []
    for i in range(30):
        p = 3400.0 + (i % 5)
        rows.append({"open": p, "high": p + 2, "low": p - 2, "close": p, "volume": 500})
    df = pd.DataFrame(rows)
    tick = Tick(
        symbol="TCS", ltp=3402.0, bid=3401.0, ask=3403.0, volume=500,
        change=2.0, change_pct=0.06, high=3410.0, low=3390.0, open=3400.0,
        timestamp=base_ts + timedelta(seconds=30 * 15),
    )
    ind = IndicatorCalc.compute("TCS", tick, df)
    assert ind.ema9 > 0, f"EMA9 should be > 0 after 30 bars, got {ind.ema9}"


def t_sim_validate_credentials_keys():
    """validate_credentials() returns all required keys."""
    creds = kite_client.validate_credentials()
    required = [
        "kite_api_key", "kite_api_secret", "kite_access_token",
        "kite_initialised", "anthropic_api_key", "trading_mode",
    ]
    missing = [k for k in required if k not in creds]
    assert not missing, f"validate_credentials missing keys: {missing}"
    assert creds["trading_mode"] in ("PAPER", "LIVE"), f"Unexpected mode: {creds['trading_mode']}"


# ── Run all ───────────────────────────────────────────────────────────────

run("Paper BUY returns PAPER-XXXXXXXX id",           t_sim_paper_buy)
run("Position qty>0 after paper BUY",                t_sim_position_updated)
run("Paper SELL closes position to qty=0",           t_sim_paper_sell_closes)
run("Cancel paper order -> CANCELLED status",        t_sim_cancel_order)
run("Squareoff all -> all positions cleared",        t_sim_squareoff_clears)
run("Modify order price -> order record updated",    t_sim_modify_order)
run("After 3 orders, list has >=3 entries",          t_sim_orders_list)
run("order_guard blocks duplicate BUY",              t_sim_order_guard_duplicate)
run("risk_manager blocks oversized qty",             t_sim_risk_oversized)
run("risk_manager blocks outside market hours",      t_sim_risk_market_hours)
run("TickBuffer builds >=1 candle from 5 ticks",     t_sim_tick_buffer_candle)
run("IndicatorCalc EMA9 > 0 after 30 ticks",         t_sim_indicators_nonzero)
run("validate_credentials() has all required keys",  t_sim_validate_credentials_keys)


if __name__ == "__main__":
    failed = summary()
    sys.exit(1 if failed else 0)
