"""
paper_trade_sim.py
Full paper trading simulation across 5 symbols and all 4 agents.
Runs the complete pipeline: signal → risk check → order guard → SEBI check → paper order.
"""
import sys
sys.path.insert(0, '/home/user/JAG/algotrader_v4')

from datetime import datetime, time, timedelta
from collections import defaultdict
import random

# ─── Module imports ─────────────────────────────────────────────────────────
from tick_engine import MarketSnapshot, LiveIndicators, Tick, Candle
from agents.strategy_agents import IntradayAgent, FnOAgent, SwingAgent, ScalpingAgent
from risk_manager import risk_manager
from order_guard import order_guard
from sebi_compliance import sebi_compliance
from kite_client import kite_client
from market_regime import regime_detector
from config import settings

print("=" * 70)
print("PAPER TRADE SIMULATION — 5 Symbols × 4 Agents")
print(f"Mode: {settings.trading_mode} | Time: {datetime.now().strftime('%H:%M:%S IST')}")
print("=" * 70)


# ─── Helper: Build various market snapshots ───────────────────────────────────

def make_snapshot(symbol, ltp, scenario="bullish"):
    """Create a realistic market snapshot for testing."""
    if scenario == "bullish":
        bid = ltp - 0.5; ask = ltp + 0.5
        vol_ratio = 1.8
        ema9 = ltp + 4; ema21 = ltp - 8; ema50 = ltp - 25; ema200 = ltp - 80
        vwap = ltp - 12
        rsi14 = 58.0; rsi7 = 62.0
        macd_hist = 0.4
        trend = "UP"; momentum = "STRONG_UP"
        candle_dir = 1
    elif scenario == "bearish":
        bid = ltp - 0.5; ask = ltp + 0.5
        vol_ratio = 1.6
        ema9 = ltp - 4; ema21 = ltp + 8; ema50 = ltp + 25; ema200 = ltp + 80
        vwap = ltp + 12
        rsi14 = 42.0; rsi7 = 38.0
        macd_hist = -0.4
        trend = "DOWN"; momentum = "STRONG_DOWN"
        candle_dir = -1
    elif scenario == "pullback":
        bid = ltp - 0.3; ask = ltp + 0.3
        vol_ratio = 1.3
        ema9 = ltp + 3; ema21 = ltp - 12; ema50 = ltp - 35; ema200 = ltp - 90
        vwap = ltp - 5
        rsi14 = 50.0; rsi7 = 52.0
        macd_hist = 0.1
        trend = "UP"; momentum = "NEUTRAL"
        candle_dir = 1
    else:  # ranging
        bid = ltp - 0.2; ask = ltp + 0.2
        vol_ratio = 0.9
        ema9 = ltp; ema21 = ltp + 2; ema50 = ltp - 5; ema200 = ltp - 40
        vwap = ltp
        rsi14 = 50.0; rsi7 = 50.0
        macd_hist = 0.0
        trend = "NEUTRAL"; momentum = "NEUTRAL"
        candle_dir = 0

    tick = Tick(
        symbol=symbol, ltp=ltp, bid=bid, ask=ask,
        volume=200000, change=ltp*0.005*candle_dir, change_pct=0.5*candle_dir,
        high=ltp + 20, low=ltp - 20, open=ltp - 5*candle_dir,
        timestamp=datetime.now(),
    )

    ind = LiveIndicators(symbol=symbol)
    ind.ltp = ltp
    ind.ema9 = ema9; ind.ema21 = ema21; ind.ema50 = ema50; ind.ema200 = ema200
    ind.vwap = vwap
    ind.rsi_14 = rsi14; ind.rsi_7 = rsi7
    ind.macd_hist = macd_hist
    ind.volume_ratio = vol_ratio
    ind.atr_14 = ltp * 0.008
    ind.adx_14 = 28.0
    ind.bb_upper = ltp + 40; ind.bb_lower = ltp - 40; ind.bb_mid = ltp
    ind.trend = trend; ind.momentum = momentum; ind.volatility = "NORMAL"
    ind.bid = bid; ind.ask = ask

    # Build realistic candles
    candles = []
    base = ltp - 40
    for i in range(30):
        close = base + i * 1.4 * candle_dir if candle_dir != 0 else base + (random.random()-0.5)*2
        o = close - 2*candle_dir; h = close + 3; lo = o - 2
        c = Candle(
            ts=datetime.now().replace(hour=9, minute=30+i, second=0),
            open=max(o,1), high=max(h,1), low=max(lo,1), close=max(close,1),
            volume=10000 + i*300
        )
        candles.append(c)

    return MarketSnapshot(symbol=symbol, tick=tick, indicators=ind, candles_1min=candles)


# ─── Simulation setup ─────────────────────────────────────────────────────────
SYMBOLS = {
    "RELIANCE": 2850.0,
    "TCS":      3500.0,
    "NIFTY":    22500.0,
    "BANKNIFTY": 48000.0,
    "INFY":     1500.0,
}

SCENARIOS = ["bullish", "bearish", "pullback", "ranging"]

# Reset state
order_guard._active.clear()
order_guard._trade_count.clear()
risk_manager.reset_daily()
sebi_compliance._state = "ACTIVE"  # ensure not killed

agents = {
    "intraday": IntradayAgent(),
    "fno":      FnOAgent(),
    "swing":    SwingAgent(),
    "scalping": ScalpingAgent(),
}

# Set swing agent throttle to fire
for sym in SYMBOLS:
    agents["swing"]._last_eval[sym] = 0

# Track results
signals_by_agent = defaultdict(int)
signals_by_pattern = defaultdict(int)
signals_by_symbol = defaultdict(int)
blocked_risk = 0
blocked_guard = 0
blocked_sebi = 0
successful_orders = []
total_signals = 0

print("\n📊 Running simulation across 5 symbols × 4 agents × 4 scenarios...\n")
print(f"{'Symbol':<12} {'Agent':<12} {'Scenario':<12} {'Signal':<8} {'Pipeline':<40}")
print("-" * 90)

for symbol, base_ltp in SYMBOLS.items():
    for scenario in SCENARIOS:
        snap = make_snapshot(symbol, base_ltp, scenario)

        for agent_name, agent in agents.items():
            try:
                # Set swing last_eval to allow evaluation
                if agent_name == "swing":
                    agent._last_eval[symbol] = 0

                action, signal = agent.evaluate_tick(snap)

                if action == "HOLD" or signal is None:
                    print(f"  {symbol:<12} {agent_name:<12} {scenario:<12} HOLD")
                    continue

                total_signals += 1
                signals_by_agent[agent_name] += 1
                signals_by_symbol[symbol] += 1

                # Extract pattern from trigger string
                trigger = signal.get("trigger", "")
                for pattern in ["VWAP_TREND", "EMA_PULLBACK", "ORB_BREAK", "BREAKOUT",
                                 "VWAP_RECLAIM", "EMA_CROSS", "TREND_PULL", "BB_SQUEEZE",
                                 "RSI_EXTREME", "SURGE", "EMA9X", "EMA921X", "VWAP_BOUNCE",
                                 "EMA50-BOUNCE"]:
                    if pattern in trigger:
                        signals_by_pattern[pattern] += 1
                        break
                else:
                    if "FNO-CE" in trigger or "FNO-PE" in trigger:
                        opt_type = signal.get("option_type", "CE/PE")
                        signals_by_pattern[f"FNO_{opt_type}"] += 1
                    else:
                        signals_by_pattern["OTHER"] += 1

                # ─── Pipeline validation ─────────────────────────────────
                pipeline_status = []
                trade_sym = signal.get("option_symbol", symbol)
                side = signal.get("side", action)
                if side not in ("BUY", "SELL"):
                    side = action

                # Step 1: Risk manager
                ok_risk, reason_risk = risk_manager.check_before_order(
                    trade_sym, 1, snap.tick.ltp, side
                )
                if not ok_risk:
                    blocked_risk += 1
                    pipeline_status.append(f"RISK_BLOCK: {reason_risk[:30]}")
                    print(f"  {symbol:<12} {agent_name:<12} {scenario:<12} {action:<8} RISK_BLOCKED")
                    continue
                pipeline_status.append("RISK_OK")

                # Step 2: Order guard
                ok_guard, reason_guard = order_guard.can_place(symbol, agent_name, side)
                if not ok_guard:
                    blocked_guard += 1
                    pipeline_status.append(f"GUARD_BLOCK: {reason_guard[:30]}")
                    print(f"  {symbol:<12} {agent_name:<12} {scenario:<12} {action:<8} GUARD_BLOCKED")
                    continue
                pipeline_status.append("GUARD_OK")

                # Step 3: SEBI compliance
                regime_val = regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN"
                sebi_ok, algo_id, sebi_reason = sebi_compliance.pre_order_check(
                    strategy=agent_name, symbol=trade_sym,
                    exchange=signal.get("exchange", "NSE"),
                    transaction_type=side, quantity=1,
                    order_type="MARKET", price_at_signal=snap.tick.ltp,
                    signal_source=f"agent_{agent_name}",
                    regime=regime_val
                )
                if not sebi_ok:
                    blocked_sebi += 1
                    pipeline_status.append(f"SEBI_BLOCK: {sebi_reason[:30]}")
                    print(f"  {symbol:<12} {agent_name:<12} {scenario:<12} {action:<8} SEBI_BLOCKED")
                    continue
                pipeline_status.append("SEBI_OK")

                # Step 4: Place paper order
                order_id = kite_client.place_order(
                    tradingsymbol=trade_sym,
                    exchange=signal.get("exchange", "NSE"),
                    transaction_type=side, quantity=1,
                    order_type="MARKET", product=agent.product,
                    tag=algo_id or f"sim_{agent_name}"
                )
                pipeline_status.append(f"ORDER_OK:{order_id}")

                # Register
                sebi_compliance.record_order_id(agent_name, symbol, order_id)
                order_guard.register_order(symbol, agent_name, side, order_id)
                risk_manager.position_opened()

                successful_orders.append({
                    "symbol": symbol,
                    "agent": agent_name,
                    "scenario": scenario,
                    "action": action,
                    "order_id": order_id,
                    "price": snap.tick.ltp,
                })

                print(f"  {symbol:<12} {agent_name:<12} {scenario:<12} {action:<8} ✅ {order_id}")

            except Exception as e:
                print(f"  {symbol:<12} {agent_name:<12} {scenario:<12} ERROR: {str(e)[:40]}")


# ─── Print simulation report ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SIMULATION REPORT")
print("=" * 70)

print(f"\n📈 Signal Generation:")
print(f"  Total signals generated:  {total_signals}")
print(f"  Successful paper orders:  {len(successful_orders)}")
print(f"  Blocked by risk manager:  {blocked_risk}")
print(f"  Blocked by order guard:   {blocked_guard}")
print(f"  Blocked by SEBI:          {blocked_sebi}")
print(f"  Pass rate:                {len(successful_orders)/max(total_signals,1)*100:.1f}%")

print(f"\n📊 Signals by Agent:")
for agent, count in sorted(signals_by_agent.items(), key=lambda x: -x[1]):
    print(f"  {agent:<12}: {count} signals")

print(f"\n📊 Signals by Symbol:")
for sym, count in sorted(signals_by_symbol.items(), key=lambda x: -x[1]):
    print(f"  {sym:<12}: {count} signals")

print(f"\n📊 Signals by Pattern:")
for pattern, count in sorted(signals_by_pattern.items(), key=lambda x: -x[1]):
    print(f"  {pattern:<20}: {count} signals")

print(f"\n✅ Successful Paper Orders ({len(successful_orders)}):")
for o in successful_orders:
    print(f"  {o['order_id']:<20} {o['agent']:<12} {o['action']:<5} {o['symbol']:<12} "
          f"@₹{o['price']:.0f} [{o['scenario']}]")

print(f"\n🔒 Pipeline Validation Summary:")
total_attempts = total_signals
pipeline_pass = len(successful_orders)
print(f"  Total signal attempts:    {total_attempts}")
print(f"  Risk checks passed:       {total_attempts - blocked_risk}")
print(f"  Guard checks passed:      {total_attempts - blocked_risk - blocked_guard}")
print(f"  SEBI checks passed:       {total_attempts - blocked_risk - blocked_guard - blocked_sebi}")
print(f"  Orders placed (PAPER):    {pipeline_pass}")

print("\n" + "=" * 70)
print(f"Simulation complete. {len(successful_orders)} paper orders placed successfully.")
print("=" * 70)
