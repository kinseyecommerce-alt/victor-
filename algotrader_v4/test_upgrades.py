"""
test_upgrades.py — world-class upgrade tests
Run: cd algotrader_v4 && python test_upgrades.py

Covers the five upgrades:
  1. cost_model        — MCX round-trip cost, slippage, break-even move, cost gate
  2. strategy_backtest — walk-forward, cost-adjusted replay of all 20 strategies
  3. state_store + coordinator — durable book, load, reconcile against broker
  4. execution         — slippage-aware entry order-type / price resolution
  5. risk posture      — LLM gate off hot path; regime size_factor via the bus
"""
from __future__ import annotations

# ── Harness ────────────────────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []
def ok(n):   _results.append((n, True, "")); print(f"  OK  {n}")
def fail(n, e): _results.append((n, False, e)); print(f"  XX  {n}: {e}")
def run(n, fn):
    try: fn(); ok(n)
    except Exception as exc: fail(n, str(exc)[:160])
def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

import cost_model, execution, state_store
from config import settings, Settings
from agent_coordinator import agent_coordinator, AgentCoordinator
from agent_bus import agent_bus, TOPIC_REGIME


# ═══════════════════════════════════════════════════════════════════════════════
section("1. TRANSACTION COST MODEL")

def t_round_trip_positive():
    rt = cost_model.round_trip_cost("CRUDEOIL", 1, 6500.0)
    assert rt > 0 and rt < 5000, rt

def t_breakdown_sums():
    bd = cost_model.round_trip_breakdown("CRUDEOIL", 1, 6500.0)
    assert abs(bd["entry"]["total"] + bd["exit"]["total"] - bd["round_trip_total"]) < 0.01

def t_slippage_scales_with_lots():
    s1 = cost_model.slippage_cost("CRUDEOIL", 1)
    s3 = cost_model.slippage_cost("CRUDEOIL", 3)
    assert abs(s3 - 3 * s1) < 0.01 and s1 > 0

def t_min_move_and_cover():
    move = cost_model.min_profitable_move("CRUDEOIL", 1, 6500.0)
    assert move > 0
    assert cost_model.covers_cost("CRUDEOIL", 1, 6500, 6500 + move + 1)
    assert not cost_model.covers_cost("CRUDEOIL", 1, 6500, 6500 + move / 2)

def t_cost_scales_turnover():
    lo = cost_model.round_trip_cost("CRUDEOIL", 1, 6500.0)
    hi = cost_model.round_trip_cost("CRUDEOIL", 5, 6500.0)
    assert hi > lo   # more lots → more cost

run("round-trip cost is positive & sane",  t_round_trip_positive)
run("entry+exit == round-trip total",      t_breakdown_sums)
run("slippage scales linearly with lots",  t_slippage_scales_with_lots)
run("break-even move + covers_cost gate",  t_min_move_and_cover)
run("cost scales with turnover",           t_cost_scales_turnover)


# ═══════════════════════════════════════════════════════════════════════════════
section("2. WALK-FORWARD STRATEGY BACKTESTER")

import strategy_backtest as bt

def t_backtest_runs_20_metrics():
    df, source = bt.load_bars("CRUDEOIL", "5m", 15)
    assert source == "SYNTHETIC" and len(df) > bt.WARMUP + 50
    metrics = bt.backtest_agent_symbol("intraday", "CRUDEOIL", df, source, folds=4)
    assert len(metrics) == 20, len(metrics)
    keys = {"agent","strategy","trades","win_rate","net_expectancy","sharpe",
            "profit_factor","fold_consistency","approved"}
    assert keys <= set(metrics[0]), metrics[0]

def t_backtest_costs_applied():
    df, source = bt.load_bars("CRUDEOIL", "5m", 15)
    m = bt.backtest_agent_symbol("intraday", "CRUDEOIL", df, source, folds=4)
    withtrades = [r for r in m if r["trades"] > 0]
    assert withtrades, "expected at least one strategy to trade"
    r = withtrades[0]
    assert abs(r["gross_total"] - r["cost_total"] - r["net_total"]) < 1.0, r

def t_synthetic_never_approved():
    rep = bt.run(symbols=["CRUDEOIL"], agents=["intraday"], interval="5m", days=15)
    assert rep["source"] == "SYNTHETIC"
    assert all(not r["approved"] for r in rep["results"]), "synthetic data must not approve"
    approved = bt.approved_strategies(rep)
    assert sum(len(v) for v in approved.values()) == 0

def t_rank_orders_by_sharpe():
    df, source = bt.load_bars("CRUDEOIL", "5m", 15)
    m = bt.backtest_agent_symbol("scalping", "CRUDEOIL", df, source, folds=4)
    ranked = bt.rank(m, by="sharpe")
    assert ranked[0]["sharpe"] >= ranked[-1]["sharpe"]

run("backtest returns 20 strategy metrics", t_backtest_runs_20_metrics)
run("net == gross − cost per strategy",     t_backtest_costs_applied)
run("synthetic data approves nothing",      t_synthetic_never_approved)
run("rank orders by sharpe",                t_rank_orders_by_sharpe)


# ═══════════════════════════════════════════════════════════════════════════════
section("3. STATE PERSISTENCE + RECONCILIATION")

def _fresh():
    agent_coordinator.reset()
    state_store.clear("coordinator_state.json")

def t_persist_and_load():
    _fresh()
    agent_coordinator.request("intraday", "CRUDEOIL", "BUY", 1)
    agent_coordinator.request("swing", "GOLDM", "BUY", 1)
    c2 = AgentCoordinator(); c2.load()
    syms = {b["symbol"] for b in c2.book()}
    assert syms == {"CRUDEOIL", "GOLDM"}, syms

def t_reconcile_drops_stale():
    _fresh()
    agent_coordinator.request("intraday", "CRUDEOIL", "BUY", 1)
    agent_coordinator.request("swing", "GOLDM", "BUY", 1)
    # only CRUDEOIL actually open at broker (futures tradingsymbol → base match)
    rec = agent_coordinator.reconcile([{"tradingsymbol": "CRUDEOIL25JULFUT", "quantity": 100}])
    assert "GOLDM" in rec["dropped"] and "CRUDEOIL" not in rec["dropped"]
    assert {b["symbol"] for b in agent_coordinator.book()} == {"CRUDEOIL"}

def t_reconcile_adopts_untracked():
    _fresh()
    rec = agent_coordinator.reconcile([{"tradingsymbol": "SILVERM25JULFUT", "quantity": -5}])
    assert "SILVERM" in rec["adopted"]
    held = agent_coordinator._book.get("SILVERM")
    assert held and held.side == "SELL" and held.agent == "reconciled"

def t_atomic_json_roundtrip():
    state_store.save_json("t_test.json", {"a": 1, "b": [1, 2, 3]})
    assert state_store.load_json("t_test.json") == {"a": 1, "b": [1, 2, 3]}
    state_store.clear("t_test.json")
    assert state_store.load_json("t_test.json", default="X") == "X"

run("coordinator persists & reloads book",  t_persist_and_load)
run("reconcile drops stale reservations",   t_reconcile_drops_stale)
run("reconcile adopts untracked positions", t_reconcile_adopts_untracked)
run("atomic JSON save/load roundtrip",      t_atomic_json_roundtrip)
_fresh()


# ═══════════════════════════════════════════════════════════════════════════════
section("4. SLIPPAGE-AWARE EXECUTION")

def t_market_mode():
    settings.entry_order_type = "MARKET"
    assert execution.resolve_entry("CRUDEOIL", "BUY", 6500) == ("MARKET", 0.0)

def t_marketable_limit_caps_slippage():
    settings.entry_order_type = "MARKETABLE_LIMIT"
    settings.entry_limit_cross_ticks = 2   # tick 1.0 → cross 2
    ot, px = execution.resolve_entry("CRUDEOIL", "BUY", 6500, bid=6499, ask=6501)
    assert ot == "LIMIT" and px == 6503.0, (ot, px)     # ask 6501 + 2 ticks
    ot, px = execution.resolve_entry("CRUDEOIL", "SELL", 6500, bid=6499, ask=6501)
    assert px == 6497.0, px                              # bid 6499 − 2 ticks

def t_passive_limit_rests_on_near_side():
    settings.entry_order_type = "LIMIT"
    _, buy = execution.resolve_entry("CRUDEOIL", "BUY", 6500, bid=6499, ask=6501)
    _, sell = execution.resolve_entry("CRUDEOIL", "SELL", 6500, bid=6499, ask=6501)
    assert buy == 6499.0 and sell == 6501.0, (buy, sell)

def t_price_snapped_to_tick():
    settings.entry_order_type = "MARKETABLE_LIMIT"
    _, px = execution.resolve_entry("COPPER", "BUY", 810.0, bid=810.02, ask=810.07)  # tick 0.05
    assert abs(px / 0.05 - round(px / 0.05)) < 1e-6, px   # px is a multiple of the tick

run("MARKET mode → no price",               t_market_mode)
run("marketable-limit caps slippage",       t_marketable_limit_caps_slippage)
run("passive LIMIT rests on near side",     t_passive_limit_rests_on_near_side)
run("limit price snapped to tick grid",     t_price_snapped_to_tick)
settings.entry_order_type = "MARKETABLE_LIMIT"


# ═══════════════════════════════════════════════════════════════════════════════
section("5. RISK POSTURE OFF THE HOT PATH")

def t_llm_gate_off_by_default():
    # the CODE default (independent of any .env override) must be off
    assert Settings.model_fields["use_claude_trade_gate"].default is False

def t_regime_posture_scales_size():
    agent_coordinator.reset(); agent_bus.clear()
    # master publishes a risk-off posture to the bus (0.5×)
    agent_bus.publish("master", TOPIC_REGIME, {"size_factor": 0.5, "regime": "volatile"}, key="regime")
    d = agent_coordinator.evaluate("intraday", "CRUDEOIL", "BUY", 1)
    assert d.allowed and abs(d.size_factor - 0.5) < 1e-6, d.size_factor

def t_regime_neutral_no_scale():
    agent_coordinator.reset(); agent_bus.clear()
    d = agent_coordinator.evaluate("intraday", "CRUDEOIL", "BUY", 1)
    assert abs(d.size_factor - 1.0) < 1e-6, d.size_factor

run("per-trade LLM gate off by default",    t_llm_gate_off_by_default)
run("regime posture scales entry size",     t_regime_posture_scales_size)
run("neutral regime → no scaling",          t_regime_neutral_no_scale)
agent_coordinator.reset(); agent_bus.clear(); state_store.clear("coordinator_state.json")


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
