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
