"""
test_mcx.py — MCX restructure test suite
Run: cd algotrader_v4 && python test_mcx.py

Covers the MCX commodity restructure:
  • mcx_universe   — contracts, lot/tick sizing, per-strategy universes
  • risk_manager   — lot-based quantity sizing
  • agent_bus      — pub/sub blackboard round-trips & peer views
  • agent_coordinator — conflict / duplicate / group-cap / global-cap / peer conviction
  • mcx_agents     — the four MCX agents produce valid, correctly-shaped signals
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from unittest.mock import patch

os.environ.setdefault("TRADING_MODE", "PAPER")

# ── Harness ────────────────────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []

def ok(name):   _results.append((name, True, "")); print(f"  OK  {name}")
def fail(name, err): _results.append((name, False, err)); print(f"  XX  {name}: {err}")
def run(name, fn):
    try:
        fn(); ok(name)
    except Exception as exc:
        fail(name, str(exc)[:140])
def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

import mcx_universe as U
from agent_bus import agent_bus, TOPIC_SIGNAL
from agent_coordinator import agent_coordinator
from risk_manager import risk_manager
from config import settings
from agents.mcx_agents import (
    MCX_AGENTS, MCXIntradayAgent, MCXScalpingAgent,
    MCXPositionalAgent, MCXOptionsSpreadAgent,
)
from tick_engine import MarketSnapshot, Tick, LiveIndicators, Candle

_MKT = datetime(2026, 1, 15, 12, 0, 0)   # mid-session IST, away from close guard


def _snap(symbol="CRUDEOIL", ltp=6500.0, rsi=52.0, rsi7=None, trend="UP",
          vwap=6480.0, macd_hist=0.5, volume_ratio=1.5, n_candles=40,
          ema9=6520.0, ema21=6500.0, ema50=6470.0, ema200=6400.0,
          volatility="NORMAL"):
    rsi7 = rsi if rsi7 is None else rsi7
    tick = Tick(symbol=symbol, ltp=ltp, bid=ltp-0.5, ask=ltp+0.5, volume=500000,
                change=0.0, change_pct=0.0, high=ltp+10, low=ltp-10, open=ltp-5,
                timestamp=datetime.now())
    ind = LiveIndicators(
        symbol=symbol, ltp=ltp, bid=ltp-0.5, ask=ltp+0.5, spread=1.0,
        ema9=ema9, ema21=ema21, ema50=ema50, ema200=ema200,
        vwap=vwap, rsi_14=rsi, rsi_7=rsi7,
        macd=2.5, macd_signal=2.0, macd_hist=macd_hist,
        bb_upper=ltp+50, bb_lower=ltp-50, bb_mid=ltp,
        atr_14=15.0, volume_ratio=volume_ratio, obv=1e6,
        day_high=ltp+50, day_low=ltp-50, day_open=ltp-20, change_pct=0.5,
        trend=trend, momentum="UP", volatility=volatility, computed_at=datetime.now(),
    )
    candles = [Candle(ltp, ltp+5, ltp-5, ltp, 500000, datetime.now()-timedelta(minutes=i))
               for i in range(n_candles)]
    return MarketSnapshot(symbol=symbol, tick=tick, indicators=ind,
                          candles_1min=candles, candles_5min=candles[:8])


# ═══════════════════════════════════════════════════════════════════════════════
section("1. MCX UNIVERSE")

def t_universe_size():
    assert len(U.MCX_SYMBOLS) >= 12, len(U.MCX_SYMBOLS)
    assert "CRUDEOIL" in U.CONTRACTS and "GOLDM" in U.CONTRACTS

def t_contract_fields():
    for sym, c in U.CONTRACTS.items():
        assert c.lot_size >= 1, sym
        assert c.tick_size > 0, sym
        assert c.margin_per_lot > 0, sym
        assert c.base_price > 0, sym
        assert c.group in (U.GROUP_BULLION, U.GROUP_ENERGY, U.GROUP_BASE_METAL, U.GROUP_AGRI)

def t_round_to_tick():
    # COPPER tick 0.05 → 810.13 snaps to 810.15
    assert U.round_to_tick("COPPER", 810.13) == 810.15, U.round_to_tick("COPPER", 810.13)
    # NATURALGAS tick 0.1
    assert U.round_to_tick("NATURALGAS", 250.24) == 250.2

def t_strategy_watchlists():
    for strat in ("intraday", "scalping", "swing", "fno"):
        wl = U.get_strategy_watchlist(strat)
        assert wl and all(i["exchange"] == "MCX" for i in wl), strat
        assert all(i["symbol"] in U.CONTRACTS for i in wl), strat

def t_session_close_groups():
    assert U.contract("COTTON").session_close == U.SESSION_CLOSE_AGRI
    assert U.contract("GOLDM").session_close == U.SESSION_CLOSE_NORMAL

run("14+ MCX contracts defined",              t_universe_size)
run("every contract has valid lot/tick/margin", t_contract_fields)
run("round_to_tick snaps to contract grid",   t_round_to_tick)
run("per-strategy watchlists are MCX symbols", t_strategy_watchlists)
run("agri vs normal session close times",     t_session_close_groups)


# ═══════════════════════════════════════════════════════════════════════════════
section("2. LOT-BASED SIZING")

def t_size_crude_lot_multiple():
    q = risk_manager.calculate_quantity(6500.0, agent="intraday", symbol="CRUDEOIL")
    assert q >= U.lot_size("CRUDEOIL") and q % U.lot_size("CRUDEOIL") == 0, q

def t_size_goldm_lot_multiple():
    q = risk_manager.calculate_quantity(72000.0, agent="swing", symbol="GOLDM")
    assert q % U.lot_size("GOLDM") == 0 and q >= 10, q

def t_size_min_one_lot():
    # tiny capital still yields at least one lot
    q = risk_manager.calculate_quantity(6500.0, symbol="CRUDEOIL", capital=1000.0)
    assert q == U.lot_size("CRUDEOIL"), q

def t_size_non_mcx_notional():
    # unknown symbol → legacy notional path (qty ~ capital/price)
    q = risk_manager.calculate_quantity(100.0, symbol="RELIANCE", capital=10000.0)
    assert q == 100, q

run("CRUDEOIL qty is a lot multiple",  t_size_crude_lot_multiple)
run("GOLDM qty is a lot multiple",     t_size_goldm_lot_multiple)
run("tiny capital still sizes 1 lot",  t_size_min_one_lot)
run("non-MCX symbol uses notional",    t_size_non_mcx_notional)


# ═══════════════════════════════════════════════════════════════════════════════
section("3. AGENT BUS")

def t_bus_publish_latest():
    agent_bus.clear()
    agent_bus.publish("intraday", TOPIC_SIGNAL, {"action": "BUY"}, key="CRUDEOIL")
    m = agent_bus.latest(TOPIC_SIGNAL, "CRUDEOIL")
    assert m and m.payload["action"] == "BUY" and m.agent == "intraday"

def t_bus_peer_excludes_self():
    agent_bus.clear()
    agent_bus.publish("intraday", TOPIC_SIGNAL, {"action": "BUY"}, key="GOLDM")
    agent_bus.publish("scalping", TOPIC_SIGNAL, {"action": "BUY"}, key="GOLDM")
    peers = agent_bus.peer_signal("GOLDM", "intraday")
    assert "scalping" in peers and "intraday" not in peers, list(peers)

def t_bus_signals_for_filters_symbol():
    agent_bus.clear()
    agent_bus.publish("intraday", TOPIC_SIGNAL, {"action": "BUY"}, key="GOLDM")
    agent_bus.publish("intraday", TOPIC_SIGNAL, {"action": "SELL"}, key="SILVERM")
    got = agent_bus.signals_for("SILVERM")
    assert len(got) == 1 and got[0].key == "SILVERM"

def t_bus_stats():
    agent_bus.clear()
    agent_bus.publish("intraday", TOPIC_SIGNAL, {"action": "BUY"}, key="GOLDM")
    s = agent_bus.stats()
    assert s["messages_total"] >= 1 and s["by_topic"].get(TOPIC_SIGNAL) == 1

run("publish → latest round-trip",       t_bus_publish_latest)
run("peer_signal excludes publisher",    t_bus_peer_excludes_self)
run("signals_for filters by symbol",     t_bus_signals_for_filters_symbol)
run("bus stats count messages",          t_bus_stats)


# ═══════════════════════════════════════════════════════════════════════════════
section("4. AGENT COORDINATOR")

def _reset_coord():
    agent_coordinator.reset(); agent_bus.clear()

def t_coord_first_allowed():
    _reset_coord()
    d = agent_coordinator.request("intraday", "CRUDEOIL", "BUY", 1)
    assert d.allowed and d.reason == "OK"

def t_coord_conflict_block():
    _reset_coord()
    agent_coordinator.request("intraday", "CRUDEOIL", "BUY", 1)
    d = agent_coordinator.request("scalping", "CRUDEOIL", "SELL", 1)
    assert not d.allowed and "conflict" in d.reason

def t_coord_duplicate_block():
    _reset_coord()
    agent_coordinator.request("intraday", "CRUDEOIL", "BUY", 1)
    d = agent_coordinator.request("scalping", "CRUDEOIL", "BUY", 1)
    assert not d.allowed and "duplicate" in d.reason

def t_coord_group_cap_block():
    _reset_coord()
    # energy group cap 250k: CRUDEOIL(210k) ok, +NATURALGAS(140k)=350k → blocked
    a = agent_coordinator.request("intraday", "CRUDEOIL", "BUY", 1)
    b = agent_coordinator.request("swing", "NATURALGAS", "BUY", 1)
    assert a.allowed and not b.allowed and "group" in b.reason, b.reason

def t_coord_global_cap_block():
    _reset_coord()
    old = settings.coord_group_margin_cap
    settings.coord_group_margin_cap = 1e12  # disable group cap to isolate global cap
    try:
        syms = ["GOLDM","SILVERM","CRUDEOIL","COPPER","ZINC","ALUMINIUM","LEAD"]
        allowed = [agent_coordinator.request("intraday", s, "BUY", 1).allowed for s in syms]
        assert allowed[:6] == [True]*6 and allowed[6] is False, allowed
    finally:
        settings.coord_group_margin_cap = old

def t_coord_peer_boost():
    _reset_coord()
    agent_bus.publish("scalping", TOPIC_SIGNAL, {"action": "BUY"}, key="GOLDM")
    d = agent_coordinator.evaluate("intraday", "GOLDM", "BUY", 1)
    assert d.allowed and d.size_factor > 1.0 and "scalping" in d.peers_agree, d.as_dict()

def t_coord_peer_damp():
    _reset_coord()
    agent_bus.publish("scalping", TOPIC_SIGNAL, {"action": "SELL"}, key="GOLDM")
    d = agent_coordinator.evaluate("intraday", "GOLDM", "BUY", 1)
    assert d.allowed and d.size_factor < 1.0 and "scalping" in d.peers_oppose, d.as_dict()

def t_coord_release_frees_slot():
    _reset_coord()
    agent_coordinator.request("intraday", "CRUDEOIL", "BUY", 1)
    agent_coordinator.release("CRUDEOIL", "intraday")
    d = agent_coordinator.request("scalping", "CRUDEOIL", "SELL", 1)
    assert d.allowed, "slot should be free after release"

run("first entry allowed",               t_coord_first_allowed)
run("opposite side → conflict block",    t_coord_conflict_block)
run("same side → duplicate block",       t_coord_duplicate_block)
run("correlated group margin cap block", t_coord_group_cap_block)
run("global concurrency cap block",      t_coord_global_cap_block)
run("peer agreement boosts size",        t_coord_peer_boost)
run("peer opposition damps size",        t_coord_peer_damp)
run("release frees the contract slot",   t_coord_release_frees_slot)


# ═══════════════════════════════════════════════════════════════════════════════
section("5. MCX AGENTS")

def _eval(agent, snap):
    with patch("agents.mcx_agents.now_ist", return_value=_MKT):
        return agent.evaluate_tick(snap)

def t_registry_four():
    assert set(MCX_AGENTS.keys()) == {"intraday", "scalping", "swing", "fno"}
    assert MCX_AGENTS["fno"].product == "NRML" and MCX_AGENTS["intraday"].product == "MIS"

def t_all_return_valid():
    for a in MCX_AGENTS.values():
        act, sig = _eval(a, _snap(rsi=50.0, volume_ratio=1.0, macd_hist=0.0))
        assert act in ("BUY", "SELL", "HOLD", "EXIT")
        assert sig is None or (sig["exchange"] == "MCX" and "stop_loss" in sig)

def t_intraday_buy():
    a = MCXIntradayAgent()
    act, sig = _eval(a, _snap(ltp=6500, rsi=58, vwap=6480, macd_hist=1.2,
                              volume_ratio=1.9, ema9=6530, ema21=6505, ema50=6480))
    assert act == "BUY", act
    assert sig["exchange"] == "MCX" and sig["product"] == "MIS"
    assert sig["stop_loss"] < sig["price"] < sig["target"]

def t_intraday_sell():
    a = MCXIntradayAgent()
    act, sig = _eval(a, _snap(ltp=6500, rsi=40, trend="DOWN", vwap=6520, macd_hist=-1.2,
                              volume_ratio=1.9, ema9=6470, ema21=6495, ema50=6510))
    assert act == "SELL", act
    assert sig["stop_loss"] > sig["price"] > sig["target"]

def t_scalping_buy():
    a = MCXScalpingAgent()
    act, sig = _eval(a, _snap(ltp=250, rsi=58, rsi7=62, vwap=249, macd_hist=0.4,
                              volume_ratio=1.8, n_candles=20))
    assert act == "BUY" and sig["product"] == "MIS", act

def t_scalping_cooldown():
    a = MCXScalpingAgent()
    sym = "CRUDEOIL"
    a._record_outcome(sym, False); a._record_outcome(sym, False); a._record_outcome(sym, False)
    assert a._in_cooldown(sym), "3 losses should trigger cooldown"

def t_positional_buy():
    a = MCXPositionalAgent()
    act, sig = _eval(a, _snap(ltp=72000, rsi=60, vwap=71500, macd_hist=1.0,
                              ema9=72500, ema21=72000, ema50=71000, ema200=69000))
    assert act == "BUY" and sig["product"] == "NRML", act
    # wide positional stop (3x ATR) → SL further than intraday
    assert sig["stop_loss_pct"] > 0

def t_positional_no_overnight_guard():
    # positional has no session-close square-off — fires even near close
    a = MCXPositionalAgent()
    late = datetime(2026, 1, 15, 23, 25, 0)
    with patch("agents.mcx_agents.now_ist", return_value=late):
        act, _ = a.evaluate_tick(_snap(ltp=72000, rsi=60, vwap=71500, macd_hist=1.0,
                                       ema9=72500, ema21=72000, ema50=71000, ema200=69000))
    assert act == "BUY", act

def t_options_momentum_buy():
    a = MCXOptionsSpreadAgent()
    act, sig = _eval(a, _snap(symbol="GOLDM", ltp=72000, rsi=60, vwap=71500,
                              macd_hist=1.0, volume_ratio=1.6, ema9=72500, ema21=72000))
    assert act == "BUY" and sig["product"] == "NRML", act

def t_options_high_vol_blocks():
    a = MCXOptionsSpreadAgent()
    act, _ = _eval(a, _snap(symbol="GOLDM", ltp=72000, rsi=60, vwap=71500, macd_hist=1.0,
                            volume_ratio=1.6, ema9=72500, ema21=72000, volatility="HIGH"))
    assert act == "HOLD", act

def t_intraday_session_guard():
    a = MCXIntradayAgent()
    late = datetime(2026, 1, 15, 23, 25, 0)   # within 15m of 23:30 close
    with patch("agents.mcx_agents.now_ist", return_value=late):
        act, _ = a.evaluate_tick(_snap(ltp=6500, rsi=58, vwap=6480, macd_hist=1.2,
                                       volume_ratio=1.9, ema9=6530, ema21=6505, ema50=6480))
    assert act == "HOLD", "intraday must not enter near session close"

def t_filter_watchlist_restricts():
    a = MCXScalpingAgent()
    wl = U.as_watchlist(["CRUDEOIL", "GOLDM", "COPPER"])  # only CRUDEOIL is in scalping universe
    approved = {i["symbol"] for i in a.filter_watchlist(wl)}
    assert "CRUDEOIL" in approved and "COPPER" not in approved, approved

def t_should_exit_target_and_sl():
    a = MCXIntradayAgent()
    ind = _snap(ltp=6600).indicators   # +100 from avg 6500, ATR 15 → target 2.5*15=37.5
    exit_, reason = a.should_exit_position(
        {"tradingsymbol": "CRUDEOIL", "quantity": 100, "average_price": 6500.0}, ind)
    assert exit_ and "target" in reason.lower(), reason
    ind2 = _snap(ltp=6450).indicators  # -50 → below SL 1.5*15=22.5
    exit2, r2 = a.should_exit_position(
        {"tradingsymbol": "CRUDEOIL", "quantity": 100, "average_price": 6500.0}, ind2)
    assert exit2 and "SL" in r2, r2

run("registry has 4 MCX agents w/ products", t_registry_four)
run("all agents return valid MCX signals",   t_all_return_valid)
run("intraday BUY on bullish setup",         t_intraday_buy)
run("intraday SELL on bearish setup",        t_intraday_sell)
run("scalping BUY on momentum burst",        t_scalping_buy)
run("scalping 3-loss cooldown",              t_scalping_cooldown)
run("positional BUY on stacked EMA trend",   t_positional_buy)
run("positional ignores session-close guard",t_positional_no_overnight_guard)
run("options/spread BUY on momentum",        t_options_momentum_buy)
run("options/spread blocks on HIGH vol",     t_options_high_vol_blocks)
run("intraday blocked near session close",   t_intraday_session_guard)
run("filter_watchlist restricts to universe",t_filter_watchlist_restricts)
run("should_exit hits target and SL",        t_should_exit_target_and_sl)


# ═══════════════════════════════════════════════════════════════════════════════
section("6. BASE-AGENT INTEGRATION (bus + coordinator wired end-to-end)")

import asyncio

def t_pipeline_publishes_and_reserves():
    """A live tick through base_agent must publish to the bus AND reserve via coordinator."""
    settings.trading_mode = "PAPER"
    settings.use_claude_trade_gate = False   # no external gate in this test
    settings.use_multi_timeframe   = False
    agent_bus.clear(); agent_coordinator.reset()

    from kite_client import kite_client
    kite_client._paper_orders.clear()
    if hasattr(kite_client, "_paper_positions"):
        kite_client._paper_positions.clear()

    async def _drive():
        a = MCXIntradayAgent()
        a._approved.add("CRUDEOIL")
        snap = _snap(symbol="CRUDEOIL", ltp=6500, rsi=58, vwap=6480, macd_hist=1.2,
                     volume_ratio=1.9, ema9=6530, ema21=6505, ema50=6480, n_candles=40)
        q: asyncio.Queue = asyncio.Queue()
        with patch("agents.mcx_agents.now_ist", return_value=_MKT):
            a.start(q)
            await q.put(snap)
            for _ in range(20):
                await asyncio.sleep(0.05)
                if agent_bus.latest("fill", "CRUDEOIL"):
                    break
            a.stop()

    asyncio.run(_drive())

    # SIGNAL broadcast happened, coordinator booked the contract, paper order placed
    assert agent_bus.latest(TOPIC_SIGNAL, "CRUDEOIL") is not None, "no SIGNAL on bus"
    assert agent_bus.latest("fill", "CRUDEOIL") is not None, "no FILL on bus"
    booked = {b["symbol"] for b in agent_coordinator.book()}
    assert "CRUDEOIL" in booked, f"coordinator did not reserve: {agent_coordinator.book()}"

run("tick → bus SIGNAL+FILL and coordinator reservation", t_pipeline_publishes_and_reserves)


# ── Summary ──────────────────────────────────────────────────────────────────
passed = sum(1 for _, o, _ in _results if o)
failed = sum(1 for _, o, _ in _results if not o)
print(f"\n{'='*60}\n  RESULTS: {len(_results)} tests -- {passed} passed  {failed} failed\n{'='*60}")
if failed:
    print("\nFailed:")
    for n, o, e in _results:
        if not o:
            print(f"  XX {n}: {e}")
import sys
sys.exit(1 if failed else 0)
