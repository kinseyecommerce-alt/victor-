"""
test_rebuild.py — tests of the core rebuild: state written by the old build loads unchanged, and the engine
upgrades (trading states, approvals, risk budget, drift monitor, price protection, readiness, v2 API).

    cd algotrader_v4 && python test_rebuild.py      (also run from test_pipeline.py)

Every test isolates its files in a temporary directory and restores the in-memory state it touches.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PERFORMANCE_FILE", os.path.join(tempfile.mkdtemp(), "performance.json"))

from config import settings  # noqa: E402
from kite_client import kite_client  # noqa: E402
import paper_book  # noqa: E402
import performance as perf_mod  # noqa: E402
from agents.strategy_agents import ALL_AGENTS  # noqa: E402
from agents.validated_base import ValidatedPosition  # noqa: E402
from ist_clock import now_ist, _IST  # noqa: E402


@contextlib.contextmanager
def isolated_book():
    """Empty paper book, private snapshot / state files, validated agents emptied; everything put back after."""
    d = tempfile.mkdtemp()
    saved = {"orders": list(kite_client._paper_orders), "positions": copy.deepcopy(kite_client._paper_positions),
             "gtts": dict(kite_client._paper_gtts),
             "agents": {n: (dict(a._positions), a.state.trades_today, a.state.pnl_today, dict(a.days_done))
                        for n, a in ALL_AGENTS.items()},
             "pb": (paper_book._restore_attempted, paper_book._last_body, paper_book._book_day),
             "mode": settings.trading_mode}
    env = {"PAPER_BOOK_FILE": os.path.join(d, "paper_book.json"),
           "TRADING_STATE_FILE": os.path.join(d, "trading_state.json"),
           "APPROVALS_FILE": os.path.join(d, "approvals.json"),
           "DRIFT_STATE_FILE": os.path.join(d, "drift_state.json")}
    with patch.dict(os.environ, env):
        try:
            import trading_state
            trading_state.reload()
        except ImportError:
            pass
        try:
            import approvals
            approvals.reload()
        except ImportError:
            pass
        kite_client._paper_orders.clear(); kite_client._paper_positions.clear(); kite_client._paper_gtts.clear()
        for a in ALL_AGENTS.values():
            a._positions.clear()
        paper_book._restore_attempted, paper_book._last_body = False, None
        settings.trading_mode = "PAPER"
        try:
            yield Path(d)
        finally:
            settings.trading_mode = saved["mode"]
            kite_client._paper_orders[:] = saved["orders"]
            kite_client._paper_positions[:] = saved["positions"]
            kite_client._paper_gtts.clear(); kite_client._paper_gtts.update(saved["gtts"])
            for n, (pos, t, p, dd) in saved["agents"].items():
                a = ALL_AGENTS[n]
                a._positions.clear(); a._positions.update(pos)
                a.state.trades_today, a.state.pnl_today = t, p
                a.days_done.clear(); a.days_done.update(dd)
            paper_book._restore_attempted, paper_book._last_body, paper_book._book_day = saved["pb"]
            for mod in ("trading_state", "approvals"):
                if mod in sys.modules:
                    sys.modules[mod].reload()


def _attn_pos(sym="NHPC", qty=1000, px=80.0) -> ValidatedPosition:
    return ValidatedPosition(oid=f"PAPER-ATT{sym}", agent="attn_on", symbol=sym, exchange="NSE", product="CNC",
                             side="BUY", qty=qty, entry_px=px, entry_ref=px,
                             opened=now_ist().replace(microsecond=0).isoformat(), overnight=True)


def _old_format_snapshot(session: str) -> dict:
    """A paper_book.json as the pre-rebuild build wrote it: legacy engine blocks, legacy agents' day state, an AI
    book, a legacy MIS row and one attention CNC holding owned by attn_on."""
    p = _attn_pos()
    return {
        "schema": 1, "session": session, "book_day": session, "mode": "PAPER", "saved_at": session + "T15:30:00",
        "kite": {"orders": [{"order_id": "PAPER-OLD1", "tradingsymbol": "RELIANCE", "status": "COMPLETE"}],
                 "positions": [
                     {"tradingsymbol": "NHPC", "exchange": "NSE", "product": "CNC", "quantity": 1000,
                      "average_price": 80.0, "last_price": 81.0, "pnl": 0.0, "buy_value": 80000.0, "sell_value": 0.0},
                     {"tradingsymbol": "RELIANCE", "exchange": "NSE", "product": "MIS", "quantity": 30,
                      "average_price": 2800.0, "last_price": 2810.0, "pnl": 0.0, "buy_value": 84000.0,
                      "sell_value": 0.0}],
                 "gtts": {"123": {"id": 123, "tradingsymbol": "TCS", "status": "active"}}},
        "tsl": [{"symbol": "RELIANCE", "strategy": "intraday", "side": "BUY", "entry_price": 2800.0, "quantity": 30,
                 "order_id": "PAPER-OLD1", "current_sl": 2780.0, "best_price": 2810.0}],
        "tracked": {"PAPER-OLD1": {"symbol": "RELIANCE", "agent": "intraday", "product": "MIS"}},
        "risk": {"daily_realised_pnl": -1200.0, "open_position_count": 1, "is_trading_halted": False},
        "guard": {"active": [["RELIANCE", "intraday", "BUY", "PAPER-OLD1", 1.0]], "trade_count": {"intraday": 1}},
        "agents": {"intraday": {"trades_today": 1, "pnl_today": -1200.0, "day_state": {"orb": []}},
                   "scalping": {"trades_today": 0, "pnl_today": 0.0},
                   "attn_on": {"trades_today": 1, "pnl_today": 0.0, "signals_fired": 1, "skip_counts": {},
                               "last_skip": "", "day_state": {}}},
        "ai": {"targets": {"X": 1}, "open": {}, "spreads": {}},
        "validated": {"attn_on": {"version": ALL_AGENTS["attn_on"].VERSION, "positions": [asdict(p)],
                                  "days_done": {}, "flags": {}, "deviations": {}, "day": {}}},
    }


# ── persistence written by the old build ───────────────────────────────────────

def test_old_format_paper_book_restores_kept_and_ignores_removed():
    with isolated_book() as d:
        today = now_ist().date().isoformat()
        raw = json.dumps(_old_format_snapshot(today))
        paper_book.path().write_text(raw)
        out = paper_book.restore()
        assert out["restored"] is True, out
        assert set(out["legacy_blocks_ignored"]) == {"tracked", "tsl", "ai", "risk", "guard"}, out
        assert out["agents_ignored"] == ["intraday", "scalping"], out
        a = ALL_AGENTS["attn_on"]
        assert [p.symbol for p in a.positions()] == ["NHPC"] and a.positions()[0].qty == 1000
        assert out["orphans"] == ["RELIANCE"], "the legacy MIS row has no kept owner"
        assert (d / "paper_book.pre-rebuild.json").read_text() == raw, "a byte-identical copy is kept"
        assert len(kite_client._paper_positions) == 2 and 123 in kite_client._paper_gtts
        # the next snapshot is the new schema; the kept agent's position is in it
        assert paper_book.save(force=True)
        snap = json.loads(paper_book.path().read_text())
        assert snap["schema"] == paper_book.SCHEMA and "tracked" not in snap and "ai" not in snap
        assert snap["validated"]["attn_on"]["positions"][0]["symbol"] == "NHPC"
        # a second restore of the new snapshot never overwrites the pre-rebuild copy
        assert (d / "paper_book.pre-rebuild.json").read_text() == raw


def test_old_format_paper_book_from_an_earlier_session():
    with isolated_book() as d:
        prev = (now_ist() - timedelta(days=3)).date().isoformat()
        paper_book.path().write_text(json.dumps(_old_format_snapshot(prev)))
        out = paper_book.restore()
        assert out["restored"] and out["stale"], out
        assert [p.symbol for p in ALL_AGENTS["attn_on"].positions()] == ["NHPC"], "CNC holding carried over"
        assert (d / f"paper_book.{prev}.json").exists() and (d / "paper_book.pre-rebuild.json").exists()


def test_corrupt_or_foreign_snapshot_never_crashes():
    with isolated_book() as d:
        paper_book.path().write_text("{not json")
        out = paper_book.restore()
        assert out["restored"] is False and (d / "paper_book.corrupt.json").exists()
    with isolated_book():
        paper_book.path().write_text(json.dumps({"session": now_ist().date().isoformat(), "kite": "garbage",
                                                 "validated": {"attn_on": {"positions": [{"bad": 1}]}},
                                                 "agents": {"attn_on": "nope"}}))
        out = paper_book.restore()                       # never raises
        assert "restored" in out
    with isolated_book():
        settings.trading_mode = "LIVE"
        paper_book.path().write_text(json.dumps(_old_format_snapshot(now_ist().date().isoformat())))
        assert paper_book.restore()["restored"] is False and not kite_client._paper_positions


def test_agent_enables_with_legacy_keys_load_and_survive_a_save():
    import bot_state
    d = tempfile.mkdtemp()
    f = Path(d) / "agent_enables.json"
    f.write_text(json.dumps({"intraday": True, "fno": False, "swing": True, "scalping": True, "auc_fade": False,
                             "attn_on": True, "vrp_winged": True}))
    saved = (bot_state._FILE, dict(bot_state._agent_enabled), dict(bot_state._extra_saved))
    try:
        bot_state._FILE = f
        bot_state._agent_enabled.clear(); bot_state._agent_enabled.update(bot_state._load())
        assert bot_state._agent_enabled == {"auc_fade": False, "attn_on": True, "vrp_winged": True, "pead_h60": True,
                                            "vrp_naked": True, "opt_ic2510": True, "opt_wcfri": True,
                                            "bld_promoted": True}
        assert not bot_state.is_agent_enabled("auc_fade")
        bot_state.set_agent_enabled("vrp_winged", False)
        on_disk = json.loads(f.read_text())
        assert on_disk["fno"] is False and on_disk["intraday"] is True, "legacy keys written back unchanged"
        assert on_disk["vrp_winged"] is False and on_disk["auc_fade"] is False
    finally:
        bot_state._FILE = saved[0]
        bot_state._agent_enabled.clear(); bot_state._agent_enabled.update(saved[1])
        bot_state._extra_saved.clear(); bot_state._extra_saved.update(saved[2])


def test_performance_ledger_of_the_old_build_loads_unchanged():
    d = tempfile.mkdtemp()
    f = Path(d) / "performance.json"
    old_rows = [{"t": "2026-11-12T10:00:00+05:30", "strategy": "intraday", "version": "v2", "symbol": "TCS",
                 "side": "BUY", "qty": 10, "entry": 4000.0, "exit": 4010.0, "gross": 100.0, "cost": 40.0,
                 "net": 60.0, "product": "MIS", "exchange": "NSE", "note": ""},
                {"t": "2026-11-13T09:20:01+05:30", "strategy": "auc_fade", "version": "AUC-FADE-3PCT/x/v1",
                 "symbol": "ABB", "side": "SELL", "qty": 961, "entry": 104.0, "exit": 103.1, "gross": 864.9,
                 "cost": 80.0, "net": 784.9, "product": "MIS", "exchange": "NSE", "note": "09:20 exit",
                 "oid": "PAPER-1", "lab_id": "AUC-FADE-3PCT"}]
    body = {"trades": old_rows, "days": {"2026-11-12": {"date": "2026-11-12", "net": 60.0, "gross": 100.0,
                                                          "costs": 40.0, "ai_cost_inr": 12.0, "trades": 1,
                                                          "nifty50": 24000.0, "nasdaq100": 20000.0}},
            "pruned": {"swing": {"since": "2026-11-01", "reason": "x", "version": "v2"}},
            "enabled_since": {"intraday": "2026-11-02T00:00:00"}, "future_key": {"keep": True}}
    f.write_text(json.dumps(body))
    with patch.object(perf_mod, "FILE", f):
        p = perf_mod.Performance()
        assert p.trades == old_rows and p.days == body["days"]
        st = p.strategy_stats()
        assert st["intraday"]["kept"] is False and st["auc_fade"]["kept"] is True
        assert p.scoreboard()["periods"]["since_start"]["net_pnl"] == 60.0
        p.record_trade("attn_on", "NHPC", "BUY", 10, 80.0, 10.0, "NSE", "CNC", cost_kind="future",
                       charge_risk=False)
        back = json.loads(f.read_text())
        assert back["trades"][:2] == old_rows and back["pruned"] == body["pruned"]
        assert back["enabled_since"] == body["enabled_since"] and back["future_key"] == {"keep": True}
        assert back["days"] == body["days"]


def test_vault_with_removed_feature_keys_starts_and_keeps_them():
    import secret_store as ss
    d = tempfile.mkdtemp()
    v = ss.SecretStore(d)
    v.save({"anthropic_api_key": "sk-old", "openai_api_key": "sk-old2", "kite_user_id": "AB1234"})
    saved = settings.kite_user_id
    try:
        applied = ss.apply_to(settings, v)
        assert applied == ["kite_user_id"] and settings.kite_user_id == "AB1234"
        assert not hasattr(settings, "anthropic_api_key")
        assert v.load()["anthropic_api_key"] == "sk-old", "never deleted from the vault"
    finally:
        settings.kite_user_id = saved


def test_squareoff_flattens_leftovers_keeps_validated_holdings():
    import platform_scheduler as ps
    with isolated_book():
        a = ALL_AGENTS["attn_on"]
        p = _attn_pos()
        a._positions[p.oid] = p
        kite_client.paper_fill_at("NHPC", "NSE", "BUY", 1000, "CNC", 80.0, "VAL-ATTN-IN", "AUCTION")
        kite_client.paper_fill_at("RELIANCE", "NSE", "BUY", 30, "MIS", 2800.0, "LEGACY", "RECONCILE")
        ids = ps.PlatformScheduler.squareoff_book()
        rows = {(r["tradingsymbol"], r["product"]): r["quantity"] for r in kite_client._paper_positions}
        assert len(ids) == 1 and rows[("RELIANCE", "MIS")] == 0 and rows[("NHPC", "CNC")] == 1000


def test_scheduler_has_no_legacy_jobs_and_keeps_the_validated_ones():
    import platform_scheduler as ps

    class _S:
        def __init__(self):
            self.jobs = []

        def add_job(self, fn, trigger, **kw):
            self.jobs.append(kw.get("id"))

        def start(self):
            pass

    s = ps.PlatformScheduler()
    s._sched = _S()
    s.start()
    ids = set(s._sched.jobs)
    for need in ("kite_refresh", "kite_refresh_last", "kite_login_startup", "validated_preflight", "validated_open",
                 "validated_auc_exit", "validated_attn_entry", "validated_store", "validated_vrp_entry",
                 "validated_vrp_settle", "validated_vrp_mark", "validated_catch_up", "forward_open", "forward_exit",
                 "daily_reset", "squareoff", "daily_scoreboard", "risk_brake", "drift_monitor"):
        assert need in ids, need
    for gone in ("allocate_morning", "allocate_midday", "regime_preopen", "universe_refresh", "pre_market",
                 "morning_data", "auto_start", "options_cache", "master__weekly_backtest"):
        assert gone not in ids, gone
    assert "validated_pead_entry" not in ids, "PEAD stays off by default"


# ── engine upgrades ────────────────────────────────────────────────────────────

@contextlib.contextmanager
def kite_paper():
    """PAPER with Kite data, validated entries allowed by settings, every agent enabled and running."""
    import bot_state
    keys = ("trading_mode", "paper_data_source", "validated_only", "approval_mode", "risk_budget_daily_vol",
            "risk_budget_allow_upscale", "risk_budget_max_scale", "risk_capital", "risk_brake_reduce_pct",
            "risk_brake_halt_pct", "drift_telegram", "validated_pead_enabled")
    saved = {k: getattr(settings, k) for k in keys}
    saved_en = dict(bot_state._agent_enabled)
    saved_run = {n: a.state.running for n, a in ALL_AGENTS.items()}
    saved_stop = bot_state.operator_stopped()
    try:
        settings.trading_mode, settings.paper_data_source, settings.validated_only = "PAPER", "KITE", True
        for n in bot_state._agent_enabled:
            bot_state._agent_enabled[n] = True
        for a in ALL_AGENTS.values():
            a.state.running = True
        bot_state._operator_stopped = False
        yield
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)
        bot_state._agent_enabled.clear(); bot_state._agent_enabled.update(saved_en)
        for n, a in ALL_AGENTS.items():
            a.state.running = saved_run[n]
        bot_state._operator_stopped = saved_stop


def _fresh_ledger():
    path = Path(tempfile.mkdtemp()) / "perf.json"
    with patch.object(perf_mod, "FILE", path):
        p = perf_mod.Performance()
    return p, path


def test_trading_states_persist_and_most_restrictive_wins():
    import trading_state as TS
    with isolated_book() as d:
        assert TS.effective("auc_fade") == "ACTIVE" and TS.entry_block("auc_fade") is None
        TS.set_strategy("attn_on", "REDUCE_ONLY", "test", "tester")
        assert TS.effective("attn_on") == "REDUCE_ONLY" and TS.effective("auc_fade") == "ACTIVE"
        TS.set_global("HALTED", "maintenance", "tester")
        assert TS.effective("auc_fade") == "HALTED" and TS.entry_block("attn_on") == "trading_state:HALTED"
        TS.reload()                                       # a restart reads the file
        assert TS.global_state() == "HALTED" and TS.strategy_state("attn_on") == "REDUCE_ONLY"
        TS.set_global("ACTIVE", "done", "tester")
        assert TS.effective("attn_on") == "REDUCE_ONLY"
        hist = TS.snapshot()["history"]
        assert [h["to"] for h in hist] == ["REDUCE_ONLY", "HALTED", "ACTIVE"] and hist[0]["by"] == "tester"
        assert (d / "trading_state.json").exists()
        try:
            TS.set_global("PAUSED")
            raise AssertionError("bad state accepted")
        except ValueError:
            pass


def test_trading_state_blocks_entries_but_never_exits():
    import trading_state as TS
    from agents.validated_base import paper_marketable, paper_fill_at
    with isolated_book(), kite_paper():
        a = ALL_AGENTS["attn_on"]
        assert a.entry_block_reason() is None and a.entry_block_reason_readonly() is None
        TS.set_strategy("attn_on", "REDUCE_ONLY", "test")
        assert a.entry_block_reason() == "trading_state:REDUCE_ONLY"
        assert a.entry_block_reason_readonly() == "trading_state:REDUCE_ONLY"
        assert ALL_AGENTS["auc_fade"].entry_block_reason() is None, "per-strategy state"
        q = {"ltp": 100.0, "bid": 99.9, "ask": 100.1}
        try:
            paper_marketable("NHPC", "NSE", "BUY", 10, "CNC", q, "VAL-ATTN-IN")
            raise AssertionError("an opening order went out under REDUCE_ONLY")
        except TS.TradingStateRefused:
            pass
        assert not kite_client._paper_orders, "nothing reached the paper broker"
        TS.set_strategy("attn_on", "ACTIVE", "test")
        oid, fill = paper_marketable("NHPC", "NSE", "BUY", 10, "CNC", q, "VAL-ATTN-IN")
        assert fill == round(100.1 * (1 + settings.paper_slippage_bps / 1e4), 2), "the quote's ask + stress slippage"
        TS.set_global("HALTED", "test")
        oid2, px = paper_fill_at("NHPC", "NSE", "SELL", 10, "CNC", 101.0, "VAL-ATTN-OUT", "AUCTION")
        assert px == round(101.0 * (1 - settings.paper_slippage_bps / 1e4), 2), "the exit still runs under HALTED"
        assert next(r for r in kite_client._paper_positions if r["tradingsymbol"] == "NHPC")["quantity"] == 0
        try:
            paper_fill_at("ABB", "NSE", "SELL", 5, "MIS", 100.0, "VAL-AUC-IN", "AUCTION")
            raise AssertionError("a short entry went out under HALTED")
        except TS.TradingStateRefused:
            pass
        paper_fill_at("NHPC", "NSE", "BUY", 3, "CNC", 0.0, "VAL-PEAD-CA", "CORPORATE_ACTION")   # book-keeping passes


def test_paper_fills_pay_the_stress_slippage_except_official_and_book_keeping_prices():
    """Every trading fill in PAPER is made worse than its ideal price by settings.paper_slippage_bps (the lab's
    5 bps cost stress): BUY higher, SELL lower, on marketable orders, auction entries, closes and stops.
    Settlements, reconciliations and corporate actions keep their exact price. The ideal price is kept on the
    order, and 0 bps gives back the old fills."""
    from agents.validated_base import paper_marketable, paper_fill_at
    assert settings.paper_slippage_bps == 0.0, "an overlay defaults to no change"
    with isolated_book(), kite_paper(), patch.object(settings, "paper_slippage_bps", 5.0):
        q = {"ltp": 200.0, "bid": 199.9, "ask": 200.1}
        oid, fill = paper_marketable("NHPC", "NSE", "BUY", 10, "CNC", q, "VAL-ATTN-IN")
        assert fill == round(200.1 * 1.0005, 2) == 200.2
        rec = next(o for o in kite_client._paper_orders if o["order_id"] == oid)
        assert rec["ideal_price"] == 200.1 and rec["average_price"] == 200.2
        _o, px = paper_fill_at("NHPC", "NSE", "SELL", 4, "CNC", 210.0, "VAL-ATTN-OUT", "CLOSE")
        assert px == round(210.0 * 0.9995, 2) == 209.9
        _o, px = paper_fill_at("NHPC", "NSE", "SELL", 3, "CNC", 210.0, "VAL-ATTN-OUT", "AUCTION")
        assert px == 209.9
        for kind in ("SETTLEMENT", "RECONCILE"):
            _o, px = paper_fill_at("NHPC", "NSE", "SELL", 1, "CNC", 210.0, "VAL-ATTN-OUT", kind)
            assert px == 210.0, kind
        _o, px = paper_fill_at("NHPC", "NSE", "BUY", 2, "CNC", 0.0, "VAL-PEAD-CA", "CORPORATE_ACTION")
        assert px == 0.0
        with patch.object(settings, "paper_slippage_bps", 0.0):
            _o, fill = paper_marketable("NHPC", "NSE", "BUY", 1, "CNC", q, "VAL-ATTN-IN")
            assert fill == 200.1, "0 bps: the old ideal fill"


def test_powered_gate_and_one_definition_of_proven():
    """22 Nov 2026: (1) no kept strategy is labelled proven under the one strict definition (luck test at the
    current N, executable as tested, positive at the held instrument's costs), each with its reason; (2) the powered
    paper-to-live gate needs n, a one-sided t >= 2.33 and gross >= 2 x cost, and readiness needs it on top of the
    agent's own bar; the strangle counts cycles, not legs; (3) the attention swing books CNC delivery costs."""
    from agents.strategy_agents import PROVEN, WHY_NOT_PROVEN
    assert not any(PROVEN.values()) and set(WHY_NOT_PROVEN) == set(ALL_AGENTS)
    a = ALL_AGENTS["auc_fade"]
    win = [{"net": 900.0 + (i % 5) * 40, "gross": 1400.0, "cost": 500.0} for i in range(120)]
    pg = a.powered_gate(win)
    assert pg["passed"] and pg["n"] == 120 and pg["t"] > 2.33 and pg["gross_to_cost"] == 2.8
    assert not a.powered_gate(win[:99])["passed"], "too few trades"
    thin = [dict(r, gross=900.0) for r in win]
    assert not a.powered_gate(thin)["passed"], "gross < 2 x cost: the kill rule"
    noise = [{"net": (600.0 if i % 2 else -590.0), "gross": 1500.0, "cost": 500.0} for i in range(200)]
    assert not a.powered_gate(noise)["passed"], "no significant edge"
    assert a.powered_gate([])["passed"] is False
    with patch.object(type(a), "live_readiness", lambda self: {"ready": True}), \
            patch.object(type(a), "powered_gate", lambda self, rows=None, net_key="net": {"passed": False}):
        r = a.readiness()
        assert r["own_criteria_ready"] and not r["ready"], "the powered gate is needed on top of the agent's bar"
    s = ALL_AGENTS["vrp_winged"]
    legs = [{"group": f"g{i}", "net": 300.0 + (i % 3) * 10, "gross": 400.0, "cost": 100.0} for i in range(20)
            for _ in range(4)]
    pg = s.powered_gate(legs)
    assert pg["n"] == 20 and not pg["passed"], "80 legs are 20 cycles: below 26"
    assert ALL_AGENTS["attn_on"].COST_KIND == "equity_cnc"
    from datetime import date as _d
    R = type(ALL_AGENTS["vrp_winged"]).settle_regime
    assert R(_d(2026, 7, 30))["model_matches"] and R(_d(2026, 12, 8))["model_matches"]
    assert not R(_d(2026, 11, 23))["model_matches"] and R(_d(2026, 12, 6))["regime"].startswith("cas_")

def test_alerts_rate_limit_health_checks_and_daily_messages():
    """alerts.py: error lines are rate limited (same text once per 30 min, a daily cap then one muted note); the
    health checks fail on a missing Kite session after 08:58 on a trading day and on a stale feed in market hours;
    /health says degraded and /health/strict answers 503 then; off by default (no sink, no jobs)."""
    import alerts
    from datetime import datetime as _dt
    from fastapi.testclient import TestClient
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
    assert settings.alerts_enabled is False and alerts.install() is False
    alerts._seen.clear()
    alerts._day.update({"date": "", "n": 0, "muted": False})
    t0 = 1_000_000.0
    assert alerts.should_send("x: job failed 12", now=t0, today="2026-11-23")
    assert alerts.should_send("x: job failed 13", now=t0 + 60, today="2026-11-23") is None, "same text, digits aside"
    assert alerts.should_send("x: job failed 14", now=t0 + 1801, today="2026-11-23"), "after 30 minutes again"
    out = [alerts.should_send(f"y{chr(65 + i % 26)}{chr(65 + i // 26)} distinct", now=t0, today="2026-11-23")
           for i in range(alerts.ALERT_MAX_PER_DAY + 5)]
    sent = [x for x in out if x]
    assert len(sent) == alerts.ALERT_MAX_PER_DAY - 2 + 1 and "muted" in sent[-1], "cap, then one muted note"
    assert alerts.should_send("w other", now=t0, today="2026-11-23") is None
    assert alerts.should_send("w other", now=t0, today="2026-11-24"), "a new day resets the cap"
    from kite_client import kite_client as KC
    from tick_engine import tick_engine as TE
    mon = _dt(2026, 11, 23, 10, 0, tzinfo=IST)
    with patch.object(KC, "_kite", None), patch.object(TE, "status", lambda: {"last_tick_age_s": 900.0}):
        h = alerts.health_summary(mon)
        assert not h["healthy"] and {"kite_session", "price_feed"} <= set(h["failing"])
        sun = _dt(2026, 11, 22, 10, 0, tzinfo=IST)
        assert {c["check"] for c in alerts.checks(sun)} == {"scheduler"}, "no session / feed check off days"
        with patch.object(alerts, "health_summary", lambda now=None: {"healthy": False, "failing": ["kite_session"],
                                                                        "checks": []}):
            import main
            c = TestClient(main.app)
            r = c.get("/health/strict")
            assert r.status_code == 503 and r.json()["failing"] == ["kite_session"]
            assert c.get("/health").json()["status"] == "degraded"
    with patch.object(alerts, "health_summary", lambda now=None: {"healthy": True, "failing": [], "checks": []}):
        assert alerts.morning_text().startswith("✅ Morning check: all clear")
        assert "Realised today" in alerts.evening_text()
    with patch.object(alerts, "telegram_ready", lambda: False):
        assert alerts.send_now("hi") is False

def test_kill_switch_survives_a_restart_until_reset():
    """A manual kill switch is written next to the ledger; a fresh process (load_persisted_kill) comes back KILLED;
    reset_kill_switch removes the file. An unreadable file counts as KILLED."""
    import tempfile as _tf
    from sebi_compliance import SEBICompliance, KillSwitchState
    d = _tf.mkdtemp()
    with patch.dict(os.environ, {"KILL_SWITCH_FILE": os.path.join(d, "kill_switch.json")}):
        a = SEBICompliance()
        assert a.load_persisted_kill() is False and a._state == KillSwitchState.ACTIVE
        a.trigger_kill_switch("owner pressed it")
        b = SEBICompliance()
        assert b.load_persisted_kill() and b._state == KillSwitchState.KILLED and "owner" in b._kill_reason
        assert b.reset_kill_switch(secret=settings.kill_switch_reset_secret)[0]
        c = SEBICompliance()
        assert c.load_persisted_kill() is False and c._state == KillSwitchState.ACTIVE
        with open(os.path.join(d, "kill_switch.json"), "w") as fh:
            fh.write("{broken")
        assert SEBICompliance().load_persisted_kill() is True

def test_live_order_path_never_duplicates_tracks_and_reconciles():
    """LIVE order safety (PAPER behaviour unchanged): every order carries a unique tag (strategy prefix kept); a
    network error whose order reached the broker returns that order instead of re-sending; an order that did not
    arrive is retried; no unprotected MARKET order is ever sent; track_order reports part-fills and final states;
    reconcile_positions lists every app / broker difference."""
    from kiteconnect.exceptions import NetworkException, InputException
    from kite_client import KiteClient
    from agents.validated_base import _agent_of
    t = KiteClient.unique_tag("VAL-ATTN-IN")
    assert len(t) <= 20 and t.startswith("VAL-ATTN-IN-") and _agent_of(t) == _agent_of("VAL-ATTN-IN")
    assert KiteClient.unique_tag("VAL-ATTN-IN") != t

    class FakeKite:
        VARIETY_REGULAR = "regular"

        def __init__(self, fail_first=0, arrives=True, depth=True):
            self.placed, self.book, self.fail_first, self.arrives, self.depth = [], [], fail_first, arrives, depth

        def quote(self, ins):
            if not self.depth:
                return {ins[0]: {"last_price": 0, "depth": {}}}
            return {ins[0]: {"last_price": 100.0, "depth": {"buy": [{"price": 99.9}], "sell": [{"price": 100.1}]}}}

        def place_order(self, **kw):
            self.placed.append(kw)
            if self.fail_first > 0:
                self.fail_first -= 1
                if self.arrives:
                    self.book.append({"order_id": f"L{len(self.placed)}", "tag": kw["tag"]})
                raise NetworkException("timeout")
            self.book.append({"order_id": f"L{len(self.placed)}", "tag": kw["tag"]})
            return f"L{len(self.placed)}"

        def orders(self):
            return list(self.book)

        def order_history(self, oid):
            return [{"status": "OPEN", "filled_quantity": 2, "pending_quantity": 3, "average_price": 100.1}]

        def margins(self):
            return {"equity": {"available": {"live_balance": 1e9}}}

        def positions(self):
            return {"net": [{"tradingsymbol": "RELIANCE", "product": "CNC", "quantity": 5},
                            {"tradingsymbol": "TCS", "product": "MIS", "quantity": -2}]}
    saved = (settings.trading_mode, settings.kite_access_token)
    try:
        settings.trading_mode, settings.kite_access_token = "LIVE", "x"
        with patch("time.sleep", lambda s: None), patch("drift_monitor.record_fill", lambda *a: None):
            kc = KiteClient()
            kc._kite = fk = FakeKite(fail_first=1, arrives=True)
            oid = kc.place_order("RELIANCE", "NSE", "BUY", 5, order_type="LIMIT", price=100.2, product="CNC",
                                 tag="VAL-ATTN-IN")
            assert oid == "L1" and len(fk.placed) == 1, "the order reached the broker: found by tag, not re-sent"
            kc._kite = fk = FakeKite(fail_first=1, arrives=False)
            oid = kc.place_order("RELIANCE", "NSE", "BUY", 5, order_type="LIMIT", price=100.2, product="CNC",
                                 tag="VAL-ATTN-IN")
            assert oid == "L2" and len(fk.placed) == 2 and fk.placed[0]["tag"] == fk.placed[1]["tag"], \
                "not in the book: retried once, with the same tag"
            kc._kite = fk = FakeKite(depth=False)
            with patch.object(KiteClient, "_paper_ltp", staticmethod(lambda s: 0.0)):
                try:
                    kc.place_order("RELIANCE", "NSE", "SELL", 5, order_type="MARKET", product="CNC", tag="X")
                    raise AssertionError("an unprotected MARKET order went out")
                except InputException:
                    pass
            assert not fk.placed
            with patch.object(KiteClient, "_paper_ltp", staticmethod(lambda s: 101.0)):
                kc.place_order("RELIANCE", "NSE", "SELL", 5, order_type="MARKET", product="CNC", tag="X")
            assert fk.placed[-1]["order_type"] == "LIMIT", "no depth: protected off the price feed"
            tr = kc.track_order("L1", timeout_s=0)
            assert tr["status"] == "OPEN" and not tr["final"] and tr["filled_qty"] == 2 and tr["pending_qty"] == 3
            rc = kc.reconcile_positions({("RELIANCE", "CNC"): 5, ("INFY", "CNC"): 3})
            assert not rc["ok"] and {(d["symbol"], d["app"], d["broker"]) for d in rc["diffs"]} == \
                {("INFY", 3, 0), ("TCS", 0, -2)}
    finally:
        settings.trading_mode, settings.kite_access_token = saved

def test_builder_promoted_agent_trades_the_registry_rules_with_their_own_exits():
    """agents/builder_promoted.py: reads the promoted strategies, keeps those of yesterday's market type, enters the
    builder's ranked gap fades at the auction print (Rs 5 lakh under c10), stops at the gross cap, leaves the 15:15
    square-off alone, and books each exit with builder_engine.walk_session on the day's minute bars."""
    import asyncio
    import numpy as np
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo
    import agents.builder_promoted as BP
    a = ALL_AGENTS["bld_promoted"]
    assert a.enabled_by_setting() is False, "off by default"
    hv = {"agent": "scalping", "mtype": "HIGH_VOL", "block": "gap_fade", "params": [["g", 0.02], ["mode", "fade_day"]],
          "entry": "o0915", "exit": ["bracket_min", 0.005, 2.0, 30], "instrument": "STOCKS", "product": "mis",
          "structure": "", "universe": "NIFTY200", "size": "c10"}
    sw = dict(hv, mtype="SIDEWAYS", params=[["g", 0.01], ["mode", "fade_day"]])
    daily = {"agent": "swing", "mtype": "HIGH_VOL", "block": "vol_shock", "params": [["k", 3.0]], "entry": "o0915",
             "exit": ["hold_d", 5], "instrument": "STOCKS", "product": "cnc", "structure": "", "universe": "NIFTY100",
             "size": "c10"}
    rules = [{"id": "bldA", "cand": hv, "universe_names": ["AAA", "BBB", "CCC"], "universe": "NIFTY200", "ok": True,
              "why": ""},
             {"id": "bldB", "cand": sw, "universe_names": ["AAA"], "universe": "NIFTY200", "ok": True, "why": ""},
             {"id": "bldC", "cand": daily, "ok": False, "why": "block vol_shock"}]
    day = _dt(2026, 11, 26, 9, 15, 5, tzinfo=ZoneInfo("Asia/Kolkata"))
    quotes = {"AAA": {"open": 103.0, "ltp": 103.0}, "BBB": {"open": 95.0, "ltp": 95.0}, "CCC": {"open": 101.0},
              "NIFTY 50": {"open": 26000.0}}
    pcs = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0, "_NIFTY50_min": 26000.0}

    def bars(sym, d):                                # AAA (short) falls 0.6 % by 09:20: target 1 %? no -> 09:45 open
        tt = np.arange(555, 600)
        o = np.full(len(tt), 103.0)
        if sym == "BBB":                             # BBB long: rises to the 1 % target at 09:18
            o = np.full(len(tt), 95.0)
            h = o.copy()
            h[3:] = 96.5
            return {"tt": tt, "o": o, "h": h, "l": o - 0.1, "c": o, "v": np.ones(len(tt))}
        return {"tt": tt, "o": o, "h": o + 0.1, "l": o - 0.1, "c": o, "v": np.ones(len(tt))}
    with isolated_book(), kite_paper(), patch.object(settings, "validated_builder_promoted_enabled", True), \
            patch.object(BP, "promoted_rules", lambda: rules), \
            patch.object(BP, "market_labels", lambda d: {20261125: "HIGH_VOL"}), \
            patch.object(BP, "prev_closes", lambda names, d: {k: v for k, v in pcs.items() if k in names}), \
            patch.object(BP.VD, "quotes", lambda ins: quotes), \
            patch.object(BP.VD, "price_adjusting_actions", lambda s, d: []), \
            patch.object(BP, "day_minutes", bars):
        a.state.running = True
        out = asyncio.run(a.enter(day))
        assert out["market_type"] == "HIGH_VOL" and out["entered"] == ["bldA:BBB", "bldA:AAA"], out   # |gap| desc
        pos = {p.symbol: p for p in a.positions()}
        assert pos["AAA"].side == "SELL" and pos["BBB"].side == "BUY", "against the gap"
        assert pos["AAA"].qty == int(500_000 // 103.0) and pos["BBB"].qty == int(500_000 // 95.0), "Rs 5 lakh (c10)"
        assert a.overnight_holdings()[("AAA", "MIS")] == -pos["AAA"].qty, "the square-off leaves them alone"
        assert asyncio.run(a.enter(day))["entered"] == [], "once a day"
        early = day.replace(minute=40)
        assert asyncio.run(a.exit_due(early)) == [], "not before the rule's deadline (09:45)"
        done = asyncio.run(a.exit_due(day.replace(minute=46)))
        assert sorted(done) == ["AAA", "BBB"] and not a.positions()
        rows = {r["symbol"]: r for r in a.ledger_rows()}
        assert rows["BBB"]["rule"] == "bldA" and rows["BBB"]["exit_reason"] == "target"
        assert rows["AAA"]["exit_reason"] == "time"
        assert a.rule_stats()["bldA"]["n"] == 2
    with isolated_book(), kite_paper(), patch.object(settings, "validated_builder_promoted_enabled", True), \
            patch.object(settings, "validated_builder_promoted_max_gross", 600_000.0), \
            patch.object(BP, "promoted_rules", lambda: rules), \
            patch.object(BP, "market_labels", lambda d: {20261125: "HIGH_VOL"}), \
            patch.object(BP, "prev_closes", lambda names, d: {k: v for k, v in pcs.items() if k in names}), \
            patch.object(BP.VD, "quotes", lambda ins: quotes), \
            patch.object(BP.VD, "price_adjusting_actions", lambda s, d: []):
        a.state.running = True
        out = asyncio.run(a.enter(day))
        assert out["entered"] == ["bldA:BBB"] and out["skipped"] == {"bldA:AAA": "gross_cap"}
    with isolated_book(), kite_paper(), patch.object(settings, "validated_builder_promoted_enabled", True), \
            patch.object(BP, "promoted_rules", lambda: rules), \
            patch.object(BP, "market_labels", lambda d: {20261125: "RISING"}):
        a.state.running = True
        out = asyncio.run(a.enter(day))
        assert out["entered"] == [] and "RISING" in out["reason"], "no promoted strategy of that market type"
    # the market type must be the PREVIOUS session's: a lagging NSE archive (newest label 24 Nov) -> no entries
    with isolated_book(), kite_paper(), patch.object(settings, "validated_builder_promoted_enabled", True), \
            patch.object(BP, "promoted_rules", lambda: rules), \
            patch.object(BP, "market_labels", lambda d: {20261124: "HIGH_VOL"}):
        a.state.running = True
        out = asyncio.run(a.enter(day))
        assert out["entered"] == [] and out["reason"].startswith("data unavailable") and "20261125" in out["reason"]
        assert "market_type" not in out
    # every entry passes the SEBI pre-order check (kill switch / pause), and its order id is recorded
    from sebi_compliance import sebi_compliance as SC
    seen = []
    with isolated_book(), kite_paper(), patch.object(settings, "validated_builder_promoted_enabled", True), \
            patch.object(BP, "promoted_rules", lambda: rules), \
            patch.object(BP, "market_labels", lambda d: {20261125: "HIGH_VOL"}), \
            patch.object(BP, "prev_closes", lambda names, d: {k: v for k, v in pcs.items() if k in names}), \
            patch.object(BP.VD, "quotes", lambda ins: quotes), \
            patch.object(BP.VD, "price_adjusting_actions", lambda s, d: []), \
            patch.object(SC, "pre_order_check", lambda **k: (k["symbol"] != "BBB", None, "kill switch active")), \
            patch.object(SC, "record_order_id", lambda strat, sym, oid: seen.append((strat, sym))):
        a.state.running = True
        out = asyncio.run(a.enter(day))
        assert out["entered"] == ["bldA:AAA"] and out["skipped"] == {"bldA:BBB": "sebi: kill switch active"}, out
        assert seen == [("bld_promoted", "AAA")]

def test_builder_promoted_shares_a_binding_cap_in_turns():
    import agents.builder_promoted as BP
    plans = {"A": [("a1", 1, 3.0), ("a2", 1, 2.0), ("a3", -1, 1.0)], "B": [("b1", -1, 5.0)], "C": []}
    rules = [({"id": k}, None) for k in ("A", "B", "C")]
    got = [(r["id"], sym) for r, _c, sym, _side, _rk in BP.fair_order(rules, lambda r, c: plans[r["id"]])]
    assert got == [("A", "a1"), ("B", "b1"), ("A", "a2"), ("A", "a3")], got
    assert BP.fair_order([], lambda r, c: []) == []


def test_builder_promoted_agent_trades_swing_rules_from_the_forward_plan():
    """Daily-grid swing rules (CNC longs): entries = the builder forward run's pending_daily at the open; exits =
    the engine's own trade of the same rule / symbol / entry day; a stale plan trades nothing; shorts unsupported."""
    import asyncio
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo
    import agents.builder_promoted as BP
    from stratlab import builder_grammar as G
    a = ALL_AGENTS["bld_promoted"]
    lng = {"agent": "swing", "mtype": "HIGH_VOL", "block": "high52", "params": [["x", 0.03]], "entry": "next_open",
           "exit": ["hold", 5], "instrument": "STOCKS", "product": "cnc/fut", "structure": "", "universe": "TOP500",
           "size": ""}
    sht = dict(lng, block="st_rev_winners", params=[["x", 0.08], ["mode", "winners"]], size="c10")
    assert BP.supported(G.Candidate.from_json(lng)) == (True, "")
    assert BP.supported(G.Candidate.from_json(sht)) == (True, ""), "swing shorts trade in PAPER too"
    rules = [{"id": "bldS", "cand": lng, "universe_names": [], "universe": "TOP500", "ok": True, "why": ""},
             {"id": "bldT", "cand": sht, "universe_names": [], "universe": "TOP500", "ok": True, "why": ""}]
    d1 = _dt(2026, 11, 26, 9, 15, 5, tzinfo=ZoneInfo("Asia/Kolkata"))
    d2 = _dt(2026, 11, 27, 9, 15, 5, tzinfo=ZoneInfo("Asia/Kolkata"))
    quotes = {"AAA": {"open": 250.0, "ltp": 251.0}, "BBB": {"open": 400.0, "ltp": 401.0},
              "CCC": {"open": 500.0, "ltp": 499.0}}
    plan1 = {"last_daily": 20261125, "pending": {"bldS": [["AAA", 1, 0.5], ["BBB", 1, 0.3]], "bldT": [["CCC", -1, 0.4]]},
             "trades": {}}

    def tr(sym, reason, xd, px, rid="bldS", side=1):
        return [rid, sym, 20261126, side, "09:15", quotes[sym]["open"], xd, "15:29", px, 1, "equity_cnc", 0, 0, reason]
    plan2 = {"last_daily": 20261126, "pending": {"bldS": [], "bldT": []},
             "trades": {"bldS": [tr("AAA", "window_end", 20261126, 255.0), tr("BBB", "stop", 20261126, 380.0)],
                        "bldT": [tr("CCC", "target", 20261126, 470.0, "bldT", -1)]}}
    with isolated_book(), kite_paper(), patch.object(settings, "validated_builder_promoted_enabled", True), \
            patch.object(BP, "promoted_rules", lambda: rules), \
            patch.object(BP.VD, "quotes", lambda ins: quotes), \
            patch.object(BP.VD, "price_adjusting_actions", lambda s, d: []):
        a.state.running = True
        with patch.object(BP, "forward_plan", lambda d: plan1):
            out = asyncio.run(a.enter(d1))
        sw = out["swing"]
        assert sw["entered"] == ["bldS:AAA", "bldS:BBB", "bldT:CCC"] and sw["exited"] == [], out
        pos = {p.symbol: p for p in a.positions()}
        assert pos["AAA"].product == "CNC" and pos["AAA"].overnight and pos["AAA"].qty == int(100_000 // 250.0)
        assert pos["BBB"].meta["kind"] == "swing" and pos["BBB"].meta["rule"] == "bldS"
        assert pos["CCC"].side == "SELL" and pos["CCC"].qty == int(500_000 // 500.0), "c10 futures size: Rs 5 lakh"
        assert pos["CCC"].meta["cost_kind"] == "future" and "paper_only" in pos["CCC"].meta
        assert pos["AAA"].meta["cost_kind"] == "equity_cnc"
        assert asyncio.run(a.exit_due(d1.replace(hour=15, minute=20))) == [], "the minute exits leave swing alone"
        with patch.object(BP, "forward_plan", lambda d: plan2):
            sw2 = asyncio.run(a._swing(d2.date(), rules))
        assert sorted(sw2["exited"]) == ["BBB", "CCC"] and sw2["entered"] == [], sw2
        assert [p.symbol for p in a.positions()] == ["AAA"], "the engine still holds AAA"
        row = [r for r in a.ledger_rows() if r["symbol"] == "BBB" and r.get("rule") == "bldS"][-1]
        assert row["exit"] == 380.0 and row.get("cost_kind", "equity_cnc") in (None, "equity_cnc"), row
        rc = [r for r in a.ledger_rows() if r["symbol"] == "CCC" and r.get("rule") == "bldT"][-1]
        assert rc["exit"] == 470.0 and rc["side"] == "SELL" and rc.get("cost_kind") == "future", rc
        assert rc["gross"] == round((500.0 - 470.0) * pos["CCC"].qty, 2), "a short gains when the price falls"
        with patch.object(BP, "forward_plan", lambda d: dict(plan2, last_daily=20261125)):
            sw3 = asyncio.run(a._swing(d2.date(), rules))
        assert sw3["reason"].startswith("data unavailable") and sw3["entered"] == [] and sw3["exited"] == []
        with patch.object(BP, "forward_plan", lambda d: None):
            assert "missing" in asyncio.run(a._swing(d2.date(), rules))["reason"]


def test_approval_mode_gates_entries_by_day_ticket():
    import approvals as AP
    import trading_state as TS
    late = {k: dtime(23, 59, 59) for k in AP.DEADLINES}      # deadlines later today, whenever the test runs
    with isolated_book(), kite_paper(), patch.dict(AP.DEADLINES, late):
        a = ALL_AGENTS["auc_fade"]
        assert AP.mode_on() is False and a.entry_block_reason() is None, "default OFF: nothing changes"
        AP.set_mode(True, "tester")
        assert a.entry_block_reason_readonly() == "awaiting_approval"
        assert AP.status()["pending"] == [], "a status read never opens a ticket"
        assert a.entry_block_reason() == "awaiting_approval"
        t = AP.status()["pending"][0]
        assert t["strategy"] == "auc_fade" and t["deadline"].endswith("23:59:59+05:30")
        assert AP.request("auc_fade")["id"] == t["id"], "idempotent while pending"
        AP.decide(t["id"], True, "tester")
        assert a.entry_block_reason() is None, "approved today"
        assert ALL_AGENTS["attn_on"].entry_block_reason() == "awaiting_approval", "per strategy"
        t2 = AP.status()["pending"][0]
        AP.decide(t2["id"], False, "tester")
        assert ALL_AGENTS["attn_on"].entry_block_reason() == "approval_rejected"
        assert ALL_AGENTS["pead_h60"].entry_block_reason(approval=False) in (None, "kill_switch"), \
            "a planning step is never gated"
        AP.request("vrp_winged", deadline=(now_ist() - timedelta(minutes=1)).isoformat())
        assert ALL_AGENTS["vrp_winged"].entry_block_reason() == "approval_expired"
        AP.request("attn_on", day=(now_ist() + timedelta(days=1)).date().isoformat())
        TS.set_global("HALTED", "test")
        assert not AP.status()["pending"] and any(x["status"] == "cancelled" for x in AP.status()["recent"])
        TS.set_global("ACTIVE", "test")
        AP.set_mode(False)
        assert ALL_AGENTS["attn_on"].entry_block_reason() is None


def test_risk_budget_defaults_to_lab_sizing_and_only_scales_down():
    import risk_budget as RB
    with isolated_book(), kite_paper():
        assert all(RB.scale(n) == 1.0 for n in ALL_AGENTS), "default: lab sizing"
        assert all(a.alloc_scale() == 1.0 for a in ALL_AGENTS.values()) and ALL_AGENTS["vrp_winged"].alloc_lots(2) == 2
        lab = {n: RB.lab_stats(n)["daily_std"] for n in ALL_AGENTS if RB.lab_stats(n).get("daily_std")}
        assert RB.scale("bld_promoted") == 1.0, "no lab volatility on file: never upsized"
        settings.risk_budget_daily_vol = 3000.0              # 3 active strategies (PEAD off) → ₹1,000 each
        active = RB._active_names()
        assert "pead_h60" not in active and len(active) == 3
        assert RB.scale("attn_on") == round(min(1.0, 1000 / lab["attn_on"]), 4)
        assert RB.scale("vrp_winged") == round(1000 / lab["vrp_winged"], 4) < 1.0
        settings.risk_budget_daily_vol = 10_000_000.0
        assert all(RB.scale(n) == 1.0 for n in active), "never above 1.0 unless upscaling is allowed"
        settings.risk_budget_allow_upscale, settings.risk_budget_max_scale = True, 1.5
        assert all(RB.scale(n) == 1.5 for n in active)
        settings.risk_budget_allow_upscale = False
        settings.risk_budget_daily_vol = 3000.0
        assert ALL_AGENTS["vrp_winged"].alloc_lots(2) == int(2 * RB.scale("vrp_winged"))
        st = RB.status()
        assert st["strategies"]["auc_fade"]["hard_caps"]["max_gross"] == settings.validated_auction_max_gross


def test_per_agent_size_scale_defaults_to_lab_size_and_scales_limits_only():
    """Each agent's own VALIDATED_<X>_NOTIONAL_SCALE: default 1.0 (lab size); it multiplies the global scale and moves
    quantities, loss stops, drawdown kills and gross caps of THAT agent only."""
    auc, attn, pead, vrp = (ALL_AGENTS[k] for k in ("auc_fade", "attn_on", "pead_h60", "vrp_winged"))
    keys = ("validated_notional_scale", "validated_auction_notional_scale", "validated_attention_notional_scale",
            "validated_pead_notional_scale")
    saved = {k: getattr(settings, k) for k in keys}
    try:
        for k in keys:
            assert getattr(type(settings)(), k) == 1.0, f"{k} must default to 1.0 (lab sizing)"
            setattr(settings, k, 1.0)
        assert auc._scale() == attn._scale() == pead._scale() == 1.0
        base = (auc.daily_loss_limit(), auc.dd_limit(), pead.max_gross(), pead.dd_limit(), attn.dd_limit())
        settings.validated_auction_notional_scale, settings.validated_pead_notional_scale = 5.0, 0.1
        assert auc._scale() == 5.0 and attn._scale() == 1.0 and abs(pead._scale() - 0.1) < 1e-12
        assert auc.daily_loss_limit() == base[0] * 5 and auc.dd_limit() == base[1] * 5
        assert abs(pead.max_gross() - base[2] * 0.1) < 1e-6 and abs(pead.dd_limit() - base[3] * 0.1) < 1e-6
        assert attn.dd_limit() == base[4], "another agent's factor never leaks"
        assert auc.get_status()["size_scale"] == 5.0 and vrp._scale() == 1.0
        settings.validated_notional_scale = 2.0                       # the global factor multiplies
        assert auc._scale() == 10.0 and attn._scale() == 2.0
        import drift_monitor as DM
        settings.validated_notional_scale = 1.0
        assert DM._size_factor(auc) == 5.0 and DM._size_factor(attn) == 1.0
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)


def test_auction_quantity_and_gross_cap_follow_the_agent_scale():
    """₹10 L capital for the auction fade = scale 5: ₹5 L per trade, gross cap ₹50 L (10 trades); same names."""
    from agents import validated_auction as VA
    from stratlab import auction as A
    ins = [VA.AuctionInput(symbol=f"S{i}", prev_close=1000.0, auction_px=1000.0 * (1.03 + i * 0.001)) for i in range(3)]
    p1, _ = VA.plan_auction(ins, 0.02, A.NOTIONAL)
    p5, _ = VA.plan_auction(ins, 0.02, A.NOTIONAL * 5)
    assert [t.symbol for t in p1] == [t.symbol for t in p5] and [t.side for t in p1] == [t.side for t in p5]
    assert len(p1) == 3
    for a, b in zip(p1, p5):
        assert 4.9 <= b.qty / a.qty <= 5.1
    capped, f = VA.apply_gross_cap(p5, 1_000_000.0 * 5)
    assert f is None or f == 1.0, "3 trades of ₹5 L are inside the ₹50 L cap"


def test_ca_calendar_parses_the_real_feed_and_falls_back_to_the_stored_copy():
    """Regression: the corporate-action calendar never loaded on the server ('ca_calendar_unavailable' every day)
    and, when it did load, matched nothing: NSE's feed sends `subject` and ex-dates like 03-Jan-2012, the parser
    read `purpose` and DD-MM-YYYY. Now (1) the parser reads the real rows, (2) when the site answers nothing the
    data layer's stored copy of the same feed is used if it was fetched in the last 3 days, never an older one."""
    import asyncio, gzip, json, tempfile
    from datetime import date, timedelta
    from unittest.mock import patch
    import event_calendar as EC
    import builder_data as BD
    today = date.today()
    ex = today + timedelta(days=2)
    rows = [{"symbol": "ABC", "subject": "Bonus 1:1", "exDate": ex.strftime("%d-%b-%Y"), "series": "EQ"},
            {"symbol": "XYZ", "subject": "Interim Dividend Rs.5/- Per Share", "exDate": ex.strftime("%d-%b-%Y")},
            {"symbol": "OLD", "subject": "Face Value Split", "exDate": (today - timedelta(days=30)).strftime("%d-%b-%Y")}]
    saved = (EC._price_actions, EC._loaded_at, dict(EC._event_cache))
    try:
        # (1) the real row format
        EC._price_actions, EC._loaded_at = {}, None
        EC._parse_corporate_actions(rows)
        assert EC.price_adjusting_actions("ABC", ex) == ["Bonus 1:1"] and EC.price_adjusting_actions("XYZ", ex) == []
        assert EC.ca_calendar_loaded()
        assert EC._parse_dmy("03-Jan-2012").date() == date(2012, 1, 3) and EC._parse_dmy("03-01-2012").date() == date(2012, 1, 3)
        # (2) the stored copy
        root = tempfile.mkdtemp(prefix="ca_copy_")
        b = BD.Builder(root)
        st = b.state()
        to = today + timedelta(days=7)
        for _s, _e, tag in BD.Builder.quarters(today, to):
            pth = b.flow / "raw" / "ca" / f"{tag}.json.gz"
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_bytes(gzip.compress(json.dumps(rows).encode()))
            st.setdefault("windows", {})[f"ca:{tag}"] = {"status": "ok", "at": int(today.strftime("%Y%m%d"))}
        b.save(st)
        got = EC._stored_corporate_actions(today, to, root=root)
        assert got is not None and {r["symbol"] for r in got} == {"ABC", "XYZ"}, "only ex-dates inside the window"

        async def nothing(_url):
            return None
        real_builder = BD.Builder
        EC._price_actions, EC._loaded_at = {}, None
        with patch.object(EC.nse_client, "get", nothing), patch.object(BD, "Builder", lambda r=None: real_builder(root)):
            BD.Builder.quarters = real_builder.quarters
            asyncio.run(EC._fetch_corporate_actions())
        assert EC.ca_calendar_loaded() and EC.price_adjusting_actions("ABC", ex) == ["Bonus 1:1"]
        # a copy older than the limit is not a calendar
        for k in list(st["windows"]):
            st["windows"][k]["at"] = int((today - timedelta(days=EC.CA_COPY_MAX_AGE_DAYS + 2)).strftime("%Y%m%d"))
        b.save(st)
        assert EC._stored_corporate_actions(today, to, root=root) is None
        EC._price_actions, EC._loaded_at = {}, None
        with patch.object(EC.nse_client, "get", nothing), patch.object(BD, "Builder", lambda r=None: real_builder(root)):
            BD.Builder.quarters = real_builder.quarters
            asyncio.run(EC._fetch_corporate_actions())
        assert not EC.ca_calendar_loaded(), "a stale copy must leave the calendar unavailable (flagged)"
    finally:
        EC._price_actions, EC._loaded_at = saved[0], saved[1]
        EC._event_cache.clear()
        EC._event_cache.update(saved[2])


def test_open_positions_outside_the_watchlist_get_marks_but_never_ticks():
    """Regression: the unrealised P&L of the PEAD small caps and the strangle's option legs stood still all day (the
    price feed follows only its watchlist). Their marks are now polled (one quote call, at most every 30 s) and used
    by the paper book ONLY for last_price / pnl: latest() stays empty for them, so a paper fill still uses the
    agent's own fresh quote."""
    import asyncio
    from unittest.mock import patch
    from tick_engine import tick_engine as TE
    from kite_client import kite_client as KC
    saved = (list(KC._paper_positions), dict(TE._marks), TE._marks_at)
    calls = []

    def fake_quote(ins):
        calls.append(list(ins))
        return {"NSE:MARSONS": {"last_price": 171.5}, "NFO:NIFTY26NOV25750CE": {"last_price": 21.4},
                "NSE:GONE": {"last_price": 0}}
    try:
        KC._paper_positions[:] = [
            {"tradingsymbol": "MARSONS", "exchange": "NSE", "product": "CNC", "quantity": 59, "average_price": 168.84,
             "last_price": 168.84, "pnl": 0.0, "buy_value": 59 * 168.84, "sell_value": 0.0},
            {"tradingsymbol": "NIFTY26NOV25750CE", "exchange": "NFO", "product": "NRML", "quantity": -130,
             "average_price": 33.7, "last_price": 33.7, "pnl": 0.0, "buy_value": 0.0, "sell_value": 130 * 33.7},
            {"tradingsymbol": "CLOSED", "exchange": "NSE", "product": "MIS", "quantity": 0, "average_price": 10.0,
             "last_price": 10.0, "pnl": 5.0, "buy_value": 100.0, "sell_value": 105.0}]
        TE._marks.clear()
        TE._marks["STALE"] = 1.0
        with patch.object(KC, "quote_kite", fake_quote), patch.object(KC, "has_session", lambda: True):
            n = asyncio.run(TE._refresh_position_marks(force=True))
            assert n == 2 and len(calls) == 1 and sorted(calls[0]) == ["NFO:NIFTY26NOV25750CE", "NSE:MARSONS"]
            assert asyncio.run(TE._refresh_position_marks()) == 0 and len(calls) == 1, "at most one call per 30 s"
        assert TE.mark("MARSONS") == 171.5 and TE.mark("NIFTY26NOV25750CE") == 21.4 and TE.mark("STALE") == 0.0
        assert TE.latest("MARSONS")[0] is None and TE.latest("NIFTY26NOV25750CE")[0] is None, "marks are not ticks"
        assert KC._paper_ltp("MARSONS") == 0.0 and KC._paper_touch("MARSONS", "SELL", 0.0) == 0.0   # fills unchanged
        # the refresh itself wrote the marks into the paper book's rows (no dashboard request needed)
        rows = {p["tradingsymbol"]: p for p in KC._paper_positions}
        assert rows["MARSONS"]["last_price"] == 171.5 and rows["MARSONS"]["pnl"] == round(59 * (171.5 - 168.84), 2)
        assert rows["NIFTY26NOV25750CE"]["last_price"] == 21.4
        assert rows["NIFTY26NOV25750CE"]["pnl"] == round(130 * 33.7 - 130 * 21.4, 2)      # short leg gains as it falls
        assert rows["CLOSED"]["last_price"] == 10.0 and rows["CLOSED"]["pnl"] == 5.0       # a flat row is untouched
        # a failing quote never raises and keeps the last marks
        def boom(_ins):
            raise RuntimeError("kite down")
        tries = []

        def boom_counted(_ins):
            tries.append(1)
            raise RuntimeError("kite down")
        with patch.object(KC, "quote_kite", boom), patch.object(KC, "has_session", lambda: True):
            assert asyncio.run(TE._refresh_position_marks(force=True)) == 0
        assert TE.mark("MARSONS") == 171.5
        # after a failed call (an expired token at night) the next try waits 15 minutes instead of 30 seconds
        import time
        import tick_engine as TEM
        assert TE._marks_failed and TEM.POSITION_MARK_RETRY_S == 900.0
        with patch.object(KC, "quote_kite", boom_counted), patch.object(KC, "has_session", lambda: True):
            TE._marks_at = time.monotonic() - 60.0                       # 60 s later: still inside the retry wait
            assert asyncio.run(TE._refresh_position_marks()) == 0 and not tries
            TE._marks_at = time.monotonic() - 901.0                      # 15 minutes later: one more try
            assert asyncio.run(TE._refresh_position_marks()) == 0 and len(tries) == 1 and TE._marks_failed
        with patch.object(KC, "quote_kite", fake_quote), patch.object(KC, "has_session", lambda: True):
            TE._marks_at = time.monotonic() - 901.0
            assert asyncio.run(TE._refresh_position_marks()) == 2 and not TE._marks_failed   # back to every 30 s
            TE._marks_at = time.monotonic() - 31.0
            assert asyncio.run(TE._refresh_position_marks()) == 2
    finally:
        KC._paper_positions[:] = saved[0]
        TE._marks.clear()
        TE._marks.update(saved[1])
        TE._marks_at = saved[2]
        TE._marks_failed = False


def test_drawdown_brake_escalates_latches_and_resets():
    import risk_budget as RB
    import trading_state as TS
    with isolated_book(), kite_paper():
        p, f = _fresh_ledger()
        settings.risk_capital, settings.risk_brake_reduce_pct, settings.risk_brake_halt_pct = 100_000.0, 10.0, 20.0
        with patch.object(perf_mod, "performance", p):
            def book(net):
                p.record_trade("auc_fade", "ABB", "SELL", 100, 100.0, net, "NSE", "MIS", cost_kind="equity_mis",
                               charge_risk=False)
            book(5000.0)
            assert RB.apply_brake() is None and TS.brake_state() == "ACTIVE"
            book(-16000.0)
            RB.apply_brake()
            assert TS.brake_state() == "REDUCE_ONLY" and TS.effective("auc_fade") == "REDUCE_ONLY"
            assert ALL_AGENTS["auc_fade"].entry_block_reason() == "trading_state:REDUCE_ONLY"
            book(-6000.0)
            RB.apply_brake()
            assert TS.brake_state() == "HALTED"
            book(30000.0)                                       # recovery never de-escalates by itself
            assert RB.apply_brake() is None and TS.brake_state() == "HALTED"
            st = RB.reset_brake("tester", "test")
            assert TS.brake_state() == "ACTIVE" and st["brake"]["drawdown"] == 0.0, st["brake"]
            assert ALL_AGENTS["auc_fade"].entry_block_reason() is None


def test_drift_monitor_warns_never_trades_and_messages_once_a_day():
    import drift_monitor as DM
    import trading_state as TS
    with isolated_book(), kite_paper():
        p, _ = _fresh_ledger()
        settings.drift_telegram = True
        fl = str(Path(tempfile.mkdtemp()) / "fills.jsonl")
        with patch.object(perf_mod, "performance", p), patch.dict(os.environ, {"FILL_LOG_FILE": fl}):
            row = DM.record_fill("attn_on", "NHPC", "BUY", 10, 100.0, 100.05, "MARKETABLE")
            assert row["slip_bps"] == 5.0 and DM.recent_fills("attn_on")[-1]["fill"] == 100.05
            a = ALL_AGENTS["attn_on"]
            ok = DM.strategy_drift("attn_on", a)
            assert ok["status"] == "ok" and ok["trades"] == 0
            for _ in range(25):                                 # 25 losers: below the lab band, deep drawdown
                p.record_trade("attn_on", "NHPC", "BUY", 1000, 80.0, -2000.0, "NSE", "CNC", cost_kind="future",
                               charge_risk=False, signal_price=79.0)
            d = DM.strategy_drift("attn_on", a)
            text = " ".join(d["warnings"] + d["alerts"])
            assert "below the lab band" in text and "drawdown" in text and "slippage" in text, d
            sent = []

            async def tg(msg):
                sent.append(msg)
            with patch("agents.base_agent.send_telegram", tg):
                asyncio.run(DM.daily_job())
                asyncio.run(DM.daily_job())
            assert len([m for m in sent if "attn_on" in m]) == 1, "at most one message per strategy per day"
            assert TS.effective("attn_on") == "ACTIVE" and not kite_client._paper_orders, "warnings only"


def test_price_protection_live_only_paper_unchanged():
    from kite_client import KiteClient
    assert KiteClient.protection_limit("BUY", 99.9, 100.1, 100.0, 0.5) == (100.6, 100.1)
    assert KiteClient.protection_limit("SELL", 99.9, 100.1, 100.0, 0.5) == (99.4, 99.9)
    assert KiteClient.protection_limit("SELL", 0, 0, 0, 0.5) == (0.0, 0.0)
    with isolated_book():
        kite_client.place_order("RELIANCE", "NSE", "BUY", 1, order_type="MARKET", product="MIS")
        assert kite_client._paper_orders[-1]["order_type"] == "MARKET", "PAPER path untouched"
    kc = KiteClient()

    class FakeKite:
        VARIETY_REGULAR = "regular"

        def __init__(self):
            self.placed = []

        def quote(self, ins):
            return {ins[0]: {"last_price": 100.0, "depth": {"buy": [{"price": 99.9}], "sell": [{"price": 100.1}]}}}

        def place_order(self, **kw):
            self.placed.append(kw)
            return "LIVE-1"

        def order_history(self, oid):
            return [{"status": "COMPLETE", "average_price": 100.15}]

        def margins(self):
            return {"equity": {"available": {"live_balance": 1e9}}}
    fk = FakeKite()
    kc._kite = fk
    saved = (settings.trading_mode, settings.kite_access_token)
    fills = []
    try:
        settings.trading_mode, settings.kite_access_token = "LIVE", "x"
        with patch("drift_monitor.record_fill", lambda *a: fills.append(a)):
            kc.place_order("RELIANCE", "NSE", "BUY", 5, order_type="MARKET", product="MIS", tag="VAL-ATTN-IN")
        o = fk.placed[-1]
        assert o["order_type"] == "LIMIT" and o["price"] == 100.6, o
        assert fills and fills[-1][4] == 100.1 and fills[-1][5] == 100.15, "quote vs fill logged"
    finally:
        settings.trading_mode, settings.kite_access_token = saved


def test_validated_agents_still_refuse_live():
    from agents.validated_base import paper_marketable
    with kite_paper():
        settings.trading_mode = "LIVE"
        for a in ALL_AGENTS.values():
            assert a.entry_block_reason() == "live_mode_refused"
        for fn in (lambda: paper_marketable("X", "NSE", "BUY", 1, "MIS", {"ask": 1.0}, "VAL-AUC-IN"),
                   lambda: kite_client.paper_fill_at("X", "NSE", "BUY", 1, "MIS", 1.0, "t", "AUCTION")):
            try:
                fn()
                raise AssertionError("a LIVE order path was reachable")
            except RuntimeError:
                pass


def test_live_readiness_reports_static_ip_and_never_ready():
    import live_readiness as LR
    saved = settings.kite_static_ip
    try:
        settings.kite_static_ip = ""
        r = LR.check()
        assert "static_ip" in r["blockers"] and r["ready_for_live"] is False
        settings.kite_static_ip = "10.0.0.5"
        assert "static_ip" in LR.check()["blockers"], "a private address is not a static egress IP"
        settings.kite_static_ip = "13.235.10.20"
        r = LR.check()
        assert "static_ip" not in r["blockers"] and r["ready_for_live"] is False
        assert "strategies_live_capable" in r["blockers"]
        assert not any(s["live_capable"] for s in r["strategies"].values())
    finally:
        settings.kite_static_ip = saved


def test_streaming_half_candle_hazard_not_on_a_live_path():
    """stratlab's build_sessions treats the last candle as complete; no kept live module uses it, and the only live
    minute-bar store (validated_data.refresh_store) keeps today's session only after 15:35."""
    import re
    import validated_data as VD
    here = Path(__file__).resolve().parent
    live = [p for p in here.glob("*.py") if not p.name.startswith(("test_", "lab_risk_build"))] + \
        list((here / "agents").glob("*.py"))
    for p in live:
        assert not re.search(r"\bbuild_sessions\b", p.read_text()), p.name
    today = now_ist().date()
    rows = []
    for d in (today - timedelta(days=1), today):
        for i in range(300):
            ts = datetime.combine(d, dtime(9, 15), tzinfo=_IST) + timedelta(minutes=i)
            rows.append({"date": ts, "open": 100, "high": 101, "low": 99, "close": 100 + i * 0.01, "volume": 10})

    class K:
        def get_instrument_tokens(self, syms, ex):
            return {s: 1 for s in syms}

        def historical_data(self, tok, frm, to, iv):
            return rows
    got = {}
    for hhmm, has_today in ((dtime(14, 0), False), (dtime(15, 40), True)):
        at = datetime.combine(today, hhmm, tzinfo=_IST)
        with patch("kite_client.kite_client", K()), patch("ist_clock.now_ist", lambda: at), \
                patch.object(VD, "set_store", lambda out, persist=False, missing=(): got.update(out=out)), \
                patch.object(VD, "UNIVERSE", ("ABB",)):
            VD.refresh_store()
        days = [s.day for s in got.pop("out")["ABB"]]
        ymd = int(today.strftime("%Y%m%d"))
        assert (ymd in days) is has_today, (hhmm, days)


def _client():
    import main
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_v2_read_endpoints_respond_and_are_json():
    with isolated_book():
        c = _client()
        for path in ("/api/v2/health", "/api/v2/overview", "/api/v2/strategies", "/api/v2/strategies/vrp_winged",
                     "/api/v2/positions", "/api/v2/trades", "/api/v2/shadow-registry", "/api/v2/archiver",
                     "/api/v2/risk-budget", "/api/v2/drift", "/api/v2/fills", "/api/v2/trading-state",
                     "/api/v2/approvals", "/api/v2/readiness", "/api/v2/system", "/health", "/validated",
                     "/shadow-registry", "/forward-test"):
            r = c.get(path)
            assert r.status_code == 200, (path, r.status_code, r.text[:200])
            r.json()
        cards = c.get("/api/v2/strategies").json()["strategies"]
        assert [x["name"] for x in cards] == ["auc_fade", "attn_on", "vrp_winged", "pead_h60", "vrp_naked",
                                              "opt_ic2510", "opt_wcfri", "bld_promoted"]
        assert {x["name"]: x["status"] for x in cards} == {n: "candidate" for n in ALL_AGENTS}, \
            "one strict definition of proven (22 Nov 2026): none of the four qualifies"
        assert all(x["why_not_proven"] for x in cards)
        assert all(x["allocator_scale"] == 1.0 for x in cards)
        assert c.get("/api/v2/strategies/nope").status_code == 404


def test_approval_deadline_without_timezone_is_ist_and_never_breaks_status():
    import approvals as AP
    with isolated_book():
        AP.set_mode(True, "tester")
        tomorrow = (now_ist() + timedelta(days=1)).date().isoformat()
        t = AP.request("auc_fade", day=tomorrow, deadline=f"{tomorrow}T09:16:00")
        assert t["deadline"] == f"{tomorrow}T09:16:00+05:30", t
        assert AP.status()["pending"], "status still works with the stored ticket"
        d = AP._load()                                     # a ticket written by the old build: naive deadline
        legacy = dict(t, id="legacy-naive", strategy="attn_on", deadline=f"{tomorrow}T09:16:00")
        d["tickets"]["legacy-naive"] = legacy
        AP._save(d)
        st = AP.status()
        assert any(x["id"] == "legacy-naive" for x in st["pending"]), st
        AP.request("vrp_winged", day=tomorrow)             # further requests keep working
        AP.set_mode(False)


def test_reenabling_a_strategy_restores_running_unless_operator_stopped():
    import bot_state
    import ist_clock
    a = ALL_AGENTS["attn_on"]
    saved = (bot_state.is_agent_enabled("attn_on"), a.state.running, bot_state.operator_stopped())
    with isolated_book(), patch.object(ist_clock, "is_trading_day", lambda *x, **k: True):
        try:
            settings.api_key, key0 = "k-test", settings.api_key
            c, h = _client(), {"X-API-Key": "k-test"}
            bot_state.set_operator_stopped(False)
            a.start()
            assert c.post("/api/v2/strategies/attn_on/enable", json={"enabled": False}, headers=h).json()["running"] \
                is False
            r = c.post("/api/v2/strategies/attn_on/enable", json={"enabled": True}, headers=h).json()
            assert r["enabled"] is True and r["running"] is True, r
            c.post("/api/v2/strategies/attn_on/enable", json={"enabled": False}, headers=h)
            bot_state.set_operator_stopped(True)                     # the operator-wide Stop still wins
            r = c.post("/api/v2/strategies/attn_on/enable", json={"enabled": True}, headers=h).json()
            assert r["enabled"] is True and r["running"] is False, r
        finally:
            settings.api_key = key0
            bot_state.set_agent_enabled("attn_on", saved[0])
            bot_state.set_operator_stopped(saved[2])
            (a.start() if saved[1] else a.stop())


def test_v2_write_endpoints_need_auth_and_act():
    import approvals as AP
    import bot_state
    import trading_state as TS
    saved = (settings.api_key, settings.jwt_secret_key, bot_state.is_agent_enabled("vrp_winged"))
    with isolated_book(), patch.dict(AP.DEADLINES, {k: dtime(23, 59, 59) for k in AP.DEADLINES}):
        try:
            settings.api_key, settings.jwt_secret_key = "k-test", "j" * 40
            c = _client()
            body = {"scope": "strategy", "strategy": "attn_on", "state": "REDUCE_ONLY", "reason": "test"}
            assert c.post("/api/v2/trading-state", json=body).status_code == 401
            assert c.get("/api/v2/overview").status_code == 401, "v2 reads need auth when API_KEY is set"
            assert c.get("/api/v2/health").status_code == 200
            h = {"X-API-Key": "k-test"}
            r = c.post("/api/v2/trading-state", json=body, headers=h)
            assert r.status_code == 200 and TS.strategy_state("attn_on") == "REDUCE_ONLY"
            assert r.json()["changed"]["by"] == "api-key"
            assert c.post("/api/v2/trading-state", json={**body, "state": "PAUSED"}, headers=h).status_code == 422
            assert c.post("/api/v2/approvals/mode", json={"on": True}, headers=h).json()["mode"] is True
            t = c.post("/api/v2/approvals", json={"strategy": "auc_fade"}, headers=h).json()["ticket"]
            r = c.post(f"/api/v2/approvals/{t['id']}/approve", json={"note": "ok"}, headers=h)
            assert r.status_code == 200 and r.json()["ticket"]["status"] == "approved"
            assert c.post(f"/api/v2/approvals/{t['id']}/reject", json={}, headers=h).status_code == 409
            assert c.post("/api/v2/approvals/zzz/approve", json={}, headers=h).status_code == 404
            r = c.post("/api/v2/strategies/vrp_winged/enable", json={"enabled": False}, headers=h)
            assert r.status_code == 200 and not bot_state.is_agent_enabled("vrp_winged")
            r = c.post("/api/v2/risk/brake/reset", json={"reason": "test"}, headers=h)
            assert r.status_code == 200 and TS.brake_state() == "ACTIVE"
            AP.set_mode(False)
        finally:
            settings.api_key, settings.jwt_secret_key = saved[0], saved[1]
            bot_state.set_agent_enabled("vrp_winged", saved[2])


def test_login_fails_closed_without_a_configured_password():
    import auth
    saved = (settings.admin_username, settings.admin_password, settings.admin_password_hash)
    try:
        settings.admin_username, settings.admin_password, settings.admin_password_hash = "admin", "", ""
        assert not auth.authenticate("admin", ""), "empty password with nothing configured must not log in"
        assert not auth.authenticate("admin", "anything"), "no password configured: password login disabled"
        settings.admin_password = "s3cret-plain"
        assert auth.authenticate("admin", "s3cret-plain") and not auth.authenticate("admin", "")
        assert not auth.authenticate("other", "s3cret-plain")
        settings.admin_password = "pässwörd-₹"                    # non-ASCII: no TypeError, still constant time
        assert auth.authenticate("admin", "pässwörd-₹") and not auth.authenticate("admin", "passwoerd")
        assert not auth.authenticate("admin", "ünïcode-wrong")
        settings.admin_password = "s3cret-plain"
        assert not auth.authenticate("admin", "ünïcode-wrong")
        settings.admin_password_hash = auth.hash_password("from-hash") if hasattr(auth, "hash_password") else \
            __import__("bcrypt").hashpw(b"from-hash", __import__("bcrypt").gensalt()).decode()
        assert auth.authenticate("admin", "from-hash") and not auth.authenticate("admin", "s3cret-plain")
        assert not auth.authenticate("admin", "")
    finally:
        settings.admin_username, settings.admin_password, settings.admin_password_hash = saved


TESTS = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


def _main() -> int:
    failed = 0
    for n, f in TESTS:
        try:
            f()
            print(f"  ok    {n}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
