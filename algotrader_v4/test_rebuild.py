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
        assert bot_state._agent_enabled == {"auc_fade": False, "attn_on": True, "vrp_winged": True, "pead_h60": True}
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
        assert fill == 100.1, "paper fill semantics unchanged: the quote's ask"
        TS.set_global("HALTED", "test")
        oid2, px = paper_fill_at("NHPC", "NSE", "SELL", 10, "CNC", 101.0, "VAL-ATTN-OUT", "AUCTION")
        assert px == 101.0, "the exit (a risk exit) still runs under HALTED"
        assert next(r for r in kite_client._paper_positions if r["tradingsymbol"] == "NHPC")["quantity"] == 0
        try:
            paper_fill_at("ABB", "NSE", "SELL", 5, "MIS", 100.0, "VAL-AUC-IN", "AUCTION")
            raise AssertionError("a short entry went out under HALTED")
        except TS.TradingStateRefused:
            pass
        paper_fill_at("NHPC", "NSE", "BUY", 3, "CNC", 0.0, "VAL-PEAD-CA", "CORPORATE_ACTION")   # book-keeping passes


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
        lab = {n: RB.lab_stats(n)["daily_std"] for n in ALL_AGENTS}
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
        assert [x["name"] for x in cards] == ["auc_fade", "attn_on", "vrp_winged", "pead_h60"]
        assert {x["name"]: x["status"] for x in cards} == {"auc_fade": "proven", "attn_on": "proven",
                                                            "vrp_winged": "candidate", "pead_h60": "candidate"}
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
