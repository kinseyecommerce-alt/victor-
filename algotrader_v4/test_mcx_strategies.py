"""
test_mcx_strategies.py — 20-strategies-per-agent tests
Run: cd algotrader_v4 && python test_mcx_strategies.py

Verifies each MCX agent has 20 distinct, well-formed strategies and that
representative strategies fire on textbook commodity setups.
"""
from __future__ import annotations

from datetime import datetime, timedelta, time as dtime
from unittest.mock import patch

# ── Harness ────────────────────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []
def ok(n):   _results.append((n, True, "")); print(f"  OK  {n}")
def fail(n, e): _results.append((n, False, e)); print(f"  XX  {n}: {e}")
def run(n, fn):
    try: fn(); ok(n)
    except Exception as exc: fail(n, str(exc)[:150])
def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

from agents.mcx_strategies import (
    SCtx, STRATEGY_REGISTRY, INTRADAY_STRATEGIES, SCALPING_STRATEGIES,
    POSITIONAL_STRATEGIES, OPTIONS_STRATEGIES,
)
from agents.mcx_agents import (
    MCX_AGENTS, MCXIntradayAgent, MCXScalpingAgent,
    MCXPositionalAgent, MCXOptionsSpreadAgent,
)
from tick_engine import MarketSnapshot, Tick, LiveIndicators, Candle

_MKT = datetime(2026, 1, 15, 12, 0, 0)


def _ind(**kw):
    base = dict(
        symbol="CRUDEOIL", ltp=6500.0, bid=6499.5, ask=6500.5, spread=1.0,
        ema9=6520, ema21=6500, ema50=6470, ema200=6400,
        vwap=6480, rsi_14=55, rsi_7=55, macd=2.5, macd_signal=2.0, macd_hist=0.5,
        bb_upper=6560, bb_lower=6440, bb_mid=6500, atr_14=15.0, volume_ratio=1.5,
        obv=1e6, day_high=6560, day_low=6440, day_open=6485, change_pct=0.3,
        trend="UP", momentum="UP", volatility="NORMAL",
        supertrend=6450, supertrend_dir="UP", hma=6510, hma_dir="UP",
        squeeze_on=False, squeeze_momentum=1.0,
        vwap_upper2=6540, vwap_lower2=6420, vwap_upper3=6570, vwap_lower3=6390,
        stoch_rsi_k=55, stoch_rsi_d=50, williams_r=-40,
    )
    base.update(kw)
    return LiveIndicators(**base)


def _ctx(prev=None, ltp=6500.0, candles_n=45, **kw):
    ind = _ind(ltp=ltp, **kw)
    candles = [Candle(6500, 6505, 6495, 6500, 500000, datetime.now()-timedelta(minutes=i))
               for i in range(candles_n)]
    return SCtx(sym="CRUDEOIL", ltp=ltp, ind=ind, prev=prev or {}, candles=candles,
                t=dtime(12, 0))


def _snap(symbol="CRUDEOIL", ltp=6500.0, n=45, **kw):
    ind = _ind(symbol=symbol, ltp=ltp, **kw)
    tick = Tick(symbol=symbol, ltp=ltp, bid=ltp-0.5, ask=ltp+0.5, volume=500000,
                change=0.0, change_pct=ind.change_pct, high=ltp+10, low=ltp-10,
                open=ltp-5, timestamp=datetime.now())
    candles = [Candle(ltp, ltp+5, ltp-5, ltp, 500000, datetime.now()-timedelta(minutes=i))
               for i in range(n)]
    return MarketSnapshot(symbol=symbol, tick=tick, indicators=ind,
                          candles_1min=candles, candles_5min=candles[:8])


# ═══════════════════════════════════════════════════════════════════════════════
section("1. EACH AGENT HAS 20 DISTINCT STRATEGIES")

def t_counts():
    assert len(INTRADAY_STRATEGIES) == 20, len(INTRADAY_STRATEGIES)
    assert len(SCALPING_STRATEGIES) == 20, len(SCALPING_STRATEGIES)
    assert len(POSITIONAL_STRATEGIES) == 20, len(POSITIONAL_STRATEGIES)
    assert len(OPTIONS_STRATEGIES) == 20, len(OPTIONS_STRATEGIES)

def t_names_unique_per_agent():
    for key, reg in STRATEGY_REGISTRY.items():
        names = [n for n, _ in reg]
        assert len(names) == len(set(names)) == 20, (key, names)

def t_total_80():
    total = sum(len(r) for r in STRATEGY_REGISTRY.values())
    assert total == 80, total

def t_agents_expose_20():
    for a in MCX_AGENTS.values():
        assert a.get_status()["strategy_count"] == 20, a.name
        assert len(a.strategy_names()) == 20

run("20 strategies per registry",          t_counts)
run("names unique within each agent",      t_names_unique_per_agent)
run("80 strategies total",                 t_total_80)
run("agents expose 20 strategies in status", t_agents_expose_20)


# ═══════════════════════════════════════════════════════════════════════════════
section("2. STRATEGIES ARE WELL-FORMED (valid shape, no exceptions)")

def t_all_callable_valid_shape():
    ctx = _ctx()
    for reg in STRATEGY_REGISTRY.values():
        for name, fn in reg:
            res = fn(ctx)
            assert res is None or (
                isinstance(res, tuple) and len(res) == 2
                and res[0] in ("BUY", "SELL") and isinstance(res[1], int)
            ), (name, res)

def t_neutral_ctx_mostly_holds():
    # a dead-flat market: no crossovers, neutral regime → most strategies return None
    ctx = _ctx(rsi_14=50, rsi_7=50, macd_hist=0.0, volume_ratio=1.0, change_pct=0.0,
               trend="NEUTRAL", momentum="NEUTRAL", supertrend_dir="NEUTRAL",
               hma_dir="NEUTRAL", ema9=6500, ema21=6500, ema50=6500, ema200=6500,
               stoch_rsi_k=50, williams_r=-50)
    fired = sum(1 for reg in STRATEGY_REGISTRY.values() for _, fn in reg if fn(ctx))
    assert fired <= 3, f"too many strategies fired on a flat market: {fired}"

run("all 80 strategies return valid shape", t_all_callable_valid_shape)
run("flat market → (almost) no fires",      t_neutral_ctx_mostly_holds)


# ═══════════════════════════════════════════════════════════════════════════════
section("3. REPRESENTATIVE STRATEGY FIRES")

def _eval(agent, snap):
    with patch("agents.mcx_agents.now_ist", return_value=_MKT):
        return agent.evaluate_tick(snap)

def t_intraday_supertrend_flip_fires():
    a = MCXIntradayAgent()
    a._pstate["CRUDEOIL"] = {"st": "DOWN"}   # previous tick was DOWN
    act, sig = _eval(a, _snap(supertrend_dir="UP", volume_ratio=1.5, rsi_14=58))
    assert act == "BUY" and sig["strategy"], (act, sig and sig.get("strategy"))

def t_intraday_bollinger_breakout():
    a = MCXIntradayAgent()
    act, sig = _eval(a, _snap(ltp=6570, bb_upper=6560, volume_ratio=1.8, macd_hist=1.2))
    assert act == "BUY", act

def t_scalping_stoch_reversal():
    a = MCXScalpingAgent()
    a._pstate["CRUDEOIL"] = {"stk": 10}      # was deeply oversold
    act, sig = _eval(a, _snap(stoch_rsi_k=25, ltp=6500, vwap=6490))
    assert act == "BUY", act

def t_scalping_vwap_lower2_bounce():
    a = MCXScalpingAgent()
    act, sig = _eval(a, _snap(ltp=6420, vwap_lower2=6420, rsi_7=30))
    assert act == "BUY" and sig["product"] == "MIS", act

def t_positional_full_ribbon():
    a = MCXPositionalAgent()
    act, sig = _eval(a, _snap(ema9=6600, ema21=6550, ema50=6480, ema200=6300,
                              rsi_14=60, macd_hist=1.0))
    assert act == "BUY" and sig["product"] == "NRML", act

def t_positional_supertrend_flip_swing():
    a = MCXPositionalAgent()
    a._pstate["CRUDEOIL"] = {"st": "DOWN"}
    act, sig = _eval(a, _snap(supertrend_dir="UP", ema9=6520, ema21=6500, rsi_14=55))
    assert act == "BUY", act

def t_options_squeeze_fire():
    a = MCXOptionsSpreadAgent()
    a._pstate["GOLDM"] = {"squeeze": True}   # squeeze was on, now releasing
    act, sig = _eval(a, _snap(symbol="GOLDM", squeeze_on=False, squeeze_momentum=1.5,
                              rsi_14=58, volatility="NORMAL"))
    assert act == "BUY" and sig["strategy"].startswith("SQUEEZE"), (act, sig and sig.get("strategy"))

def t_options_high_vol_blocks_all():
    a = MCXOptionsSpreadAgent()
    a._pstate["GOLDM"] = {"squeeze": True}
    act, _ = _eval(a, _snap(symbol="GOLDM", squeeze_on=False, squeeze_momentum=1.5,
                            rsi_14=58, volatility="HIGH"))
    assert act == "HOLD", "high realised vol must suppress option-buy strategies"

def t_best_score_wins():
    # ribbon (score 5) should beat a plain trend strategy (score 3)
    a = MCXPositionalAgent()
    act, sig = _eval(a, _snap(ema9=6600, ema21=6550, ema50=6480, ema200=6300,
                              rsi_14=60, macd_hist=1.0, supertrend_dir="UP"))
    assert act == "BUY" and "score=5" in sig["trigger"], sig["trigger"]

def t_last_strategy_recorded():
    a = MCXIntradayAgent()
    _eval(a, _snap(ltp=6570, bb_upper=6560, volume_ratio=1.8, macd_hist=1.2))
    assert a._last_strategy, "agent should record which strategy fired"
    assert a.get_status()["last_strategy"] == a._last_strategy

run("intraday supertrend flip fires BUY",   t_intraday_supertrend_flip_fires)
run("intraday bollinger breakout fires",    t_intraday_bollinger_breakout)
run("scalping stoch reversal fires",        t_scalping_stoch_reversal)
run("scalping VWAP lower-2σ bounce fires",  t_scalping_vwap_lower2_bounce)
run("positional full-ribbon fires",         t_positional_full_ribbon)
run("positional supertrend-flip swing",     t_positional_supertrend_flip_swing)
run("options squeeze-fire fires",           t_options_squeeze_fire)
run("options high-vol suppresses all",      t_options_high_vol_blocks_all)
run("best-score strategy wins",             t_best_score_wins)
run("agent records last fired strategy",    t_last_strategy_recorded)


# ── Summary ──────────────────────────────────────────────────────────────────
passed = sum(1 for _, o, _ in _results if o)
failed = sum(1 for _, o, _ in _results if not o)
print(f"\n{'='*60}\n  RESULTS: {len(_results)} tests -- {passed} passed  {failed} failed\n{'='*60}")
if failed:
    print("\nFailed:")
    for n, o, e in _results:
        if not o: print(f"  XX {n}: {e}")
import sys
sys.exit(1 if failed else 0)
