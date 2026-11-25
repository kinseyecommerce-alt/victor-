"""
test_pipeline.py — Full pipeline integration test
Tests every module's internal logic and cross-module connections.
Run: cd algotrader_v4 && python test_pipeline.py
"""
from __future__ import annotations

import asyncio
import os as _os, tempfile as _tmp0
# The ledger must never write into the repo's data/ during tests, and pruning must not
# switch agents off mid-suite because of synthetic losing trades (its own tests enable it).
_os.environ.setdefault("PERFORMANCE_FILE", _os.path.join(_tmp0.mkdtemp(), "performance.json"))
import sys
import time as _time_mod
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time

# ── Test harness ───────────────────────────────────────────────────────────

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
        fail(name, str(exc)[:300] or repr(exc)[:300])

async def arun(name: str, coro):
    try:
        await coro
        ok(name)
    except Exception as exc:
        fail(name, str(exc)[:300] or repr(exc)[:300])

def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def summary():
    passed = sum(1 for _, ok_, _ in _results if ok_)
    failed = sum(1 for _, ok_, _ in _results if not ok_)
    total  = len(_results)
    print(f"\n{'═'*60}")
    print(f"  RESULTS: {total} tests — ✅ {passed} passed  ❌ {failed} failed")
    print(f"{'═'*60}")
    if failed:
        print("\nFailed tests:")
        for name, ok_, err in _results:
            if not ok_:
                print(f"  ❌ {name}: {err}")
    return failed


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════

# ── shared imports and helpers (the core rebuild removed the sections that used to define them) ──
import contextlib as _ctx_pb
import copy as _copy_pb
import json as _json_pb
import tempfile as _pftmp
from datetime import date
from pathlib import Path as _PFPath
from unittest.mock import patch as _patch

from config import settings
from config import settings as _ai_settings
from kite_client import kite_client as _te_kite_client
import agents.base_agent as _ba
import paper_book as _pb
import performance as _perf_mod
from agents.strategy_agents import ALL_AGENTS
from agents.strategy_agents import ALL_AGENTS as _ALL_pb
from costs import trade_cost as _tc, round_trip_cost as _rtc
from sebi_compliance import sebi_compliance as _sebi_x
from fastapi.testclient import TestClient

_IST_pb = _pb.now_ist().tzinfo


async def _noop_async(*a, **k):
    return None


def _fresh_perf():
    path = _PFPath(_pftmp.mkdtemp()) / "perf.json"
    with _patch.object(_perf_mod, "FILE", path):
        p = _perf_mod.Performance()
    return p, path


def _pb_wipe():
    """What a process restart does to the in-memory paper book."""
    _te_kite_client._paper_orders.clear(); _te_kite_client._paper_positions.clear(); _te_kite_client._paper_gtts.clear()
    for a in _ALL_pb.values():
        a.state.trades_today = a.state.pnl_today = a.state.signals_fired = 0
    _pb._last_body = None


@_ctx_pb.contextmanager
def _pb_isolated():
    """Run with an empty in-memory book and a private snapshot file; put the suite's state back after."""
    saved = {
        "orders": list(_te_kite_client._paper_orders), "positions": _copy_pb.deepcopy(_te_kite_client._paper_positions),
        "gtts": dict(_te_kite_client._paper_gtts),
        "agents": {n: (a.state.trades_today, a.state.pnl_today, a.state.signals_fired) for n, a in _ALL_pb.items()},
        "pb": (_pb._restore_attempted, _pb._last_body, _pb._book_day),
        "env": _os.environ.get("PAPER_BOOK_FILE"), "mode": settings.trading_mode,
    }
    d = _tmp0.mkdtemp()
    _os.environ["PAPER_BOOK_FILE"] = _os.path.join(d, "paper_book.json")
    try:
        _pb_wipe()
        _pb._restore_attempted, _pb._last_body = True, None
        _pb._book_day = _pb.session_day()
        yield _pb.path()
    finally:
        settings.trading_mode = saved["mode"]
        _te_kite_client._paper_orders[:] = saved["orders"]
        _te_kite_client._paper_positions[:] = saved["positions"]
        _te_kite_client._paper_gtts.clear(); _te_kite_client._paper_gtts.update(saved["gtts"])
        for n, (t, p, s) in saved["agents"].items():
            a = _ALL_pb[n]
            a.state.trades_today, a.state.pnl_today, a.state.signals_fired = t, p, s
        _pb._restore_attempted, _pb._last_body, _pb._book_day = saved["pb"]
        if saved["env"] is None:
            _os.environ.pop("PAPER_BOOK_FILE", None)
        else:
            _os.environ["PAPER_BOOK_FILE"] = saved["env"]


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════
section("1. CONFIG / SETTINGS")


def t_cfg_mode():        assert settings.trading_mode == "PAPER"
def t_cfg_pos_size():    assert settings.max_position_size > 0
def t_cfg_origins():     assert isinstance(settings.origins_list, list)
def t_cfg_squareoff():
    h, m = settings.squareoff_time.split(":")
    assert 0 <= int(h) <= 23 and 0 <= int(m) <= 59
def t_cfg_max_positions(): assert settings.max_open_positions >= 1


def t_cfg_engine_defaults_keep_paper_unchanged():
    """The engine upgrades default to 'no change': approvals off, no upscaling, drift Telegram off, PEAD off."""
    from config import Settings
    s = Settings(_env_file=None)
    assert s.approval_mode is False and s.risk_budget_allow_upscale is False and s.risk_budget_max_scale == 1.0
    assert s.drift_telegram is False and s.validated_pead_enabled is False and s.trading_mode == "PAPER"
    assert 0 < s.risk_brake_reduce_pct < s.risk_brake_halt_pct


def t_cfg_unknown_env_ignored():
    """Variables of the removed engine (still set on an old deployment) are ignored, never a startup error."""
    from config import Settings
    with _patch.dict(_os.environ, {"USE_CLAUDE_TRADE_GATE": "true", "AI_AGENT_MODEL": "x", "PRUNE_ENABLED": "false",
                                   "AUTO_START_STRATEGIES": "intraday,scalping"}):
        s = Settings(_env_file=None)
    assert not hasattr(s, "use_claude_trade_gate") and not hasattr(s, "ai_agent_model")


run("settings.trading_mode == PAPER",      t_cfg_mode)
run("settings.max_position_size > 0",      t_cfg_pos_size)
run("settings.origins_list is list",       t_cfg_origins)
run("squareoff_time HH:MM valid",          t_cfg_squareoff)
run("max_open_positions >= 1",             t_cfg_max_positions)
run("engine defaults: approvals off, no upscale, drift Telegram off, PEAD off", t_cfg_engine_defaults_keep_paper_unchanged)
run("unknown (removed-engine) env variables are ignored", t_cfg_unknown_env_ignored)


section("2. KITE CLIENT — rate limiter + paper trading")
from kite_client import (
    KiteClient, _TokenBucket, _with_retry,
    _KITE_ORDER_TAG_MAX, _FON_LOT_SIZES, _RETRY_MAX,
)
from kiteconnect.exceptions import InputException

def t_bucket_acquire():
    b = _TokenBucket(100)
    b.acquire()  # should not raise

def t_bucket_throttle():
    b = _TokenBucket(5)
    t0 = _time_mod.monotonic()
    for _ in range(5):
        b.acquire()
    elapsed = _time_mod.monotonic() - t0
    assert elapsed < 2.0, f"Too slow: {elapsed:.2f}s"

def t_retry_success():
    assert _with_retry(lambda: 42) == 42

def t_fno_lot_sizes():
    assert len(_FON_LOT_SIZES) >= 20
    assert _FON_LOT_SIZES["NIFTY"] == 75
    assert _FON_LOT_SIZES["BANKNIFTY"] == 30

def t_paper_place_order():
    kc = KiteClient()
    oid = kc.place_order("RELIANCE", "NSE", "BUY", 1, tag="Test")
    assert oid.startswith("PAPER-"), f"Got: {oid}"

def t_tag_truncated():
    kc = KiteClient()
    kc.place_order("RELIANCE", "NSE", "BUY", 1, tag="A" * 30)
    assert all(len(o["tag"]) <= _KITE_ORDER_TAG_MAX for o in kc._paper_orders)

def t_paper_orders_list():
    kc = KiteClient()
    assert isinstance(kc.orders(), list)

def t_paper_positions_net():
    kc = KiteClient()
    assert "net" in kc.positions()

def t_buy_updates_position():
    kc = KiteClient()
    kc.place_order("TCS", "NSE", "BUY", 5)
    pos = next(p for p in kc.positions()["net"] if p["tradingsymbol"] == "TCS")
    assert pos["quantity"] == 5

def t_buy_sell_nets_zero():
    kc = KiteClient()
    kc.place_order("INFY", "NSE", "BUY", 10)
    kc.place_order("INFY", "NSE", "SELL", 10)
    pos = next((p for p in kc.positions()["net"] if p["tradingsymbol"] == "INFY"),
               {"quantity": 0})
    assert pos["quantity"] == 0

def t_qty_zero_raises():
    kc = KiteClient()
    try:
        kc._validated_quantity("RELIANCE", "NSE", "MIS", 0)
        raise AssertionError("Should have raised InputException")
    except InputException:
        pass

def t_fno_qty_snapped():
    kc = KiteClient()
    # NIFTY lot=75; 80 → ceil(80/75)*75 = 150
    snapped = kc._validated_quantity("NIFTY", "NFO", "NRML", 80)
    assert snapped == 150, f"Expected 150, got {snapped}"

def t_valid_qty_unchanged():
    kc = KiteClient()
    assert kc._validated_quantity("RELIANCE", "NSE", "MIS", 5) == 5

def t_squareoff_all():
    kc = KiteClient()
    kc.place_order("HDFC", "NSE", "BUY", 2)
    kc.squareoff_all_positions()
    for pos in kc.positions()["net"]:
        assert pos["quantity"] == 0

def t_margins_equity():
    kc = KiteClient()
    assert "equity" in kc.margins()

def t_hist_paper_empty():
    kc = KiteClient()
    data = kc.historical_data(256265,
                               datetime.now() - timedelta(days=5),
                               datetime.now(), "day")
    assert data == []

def t_order_history():
    kc = KiteClient()
    oid = kc.place_order("WIPRO", "NSE", "BUY", 1)
    hist = kc.order_history(oid)
    assert len(hist) == 1 and hist[0]["order_id"] == oid

def t_cancel_order():
    kc = KiteClient()
    oid = kc.place_order("SBIN", "NSE", "BUY", 1)
    kc.cancel_order(oid)
    o = next(x for x in kc._paper_orders if x["order_id"] == oid)
    assert o["status"] == "CANCELLED"

def t_modify_order():
    kc = KiteClient()
    oid = kc.place_order("AXISBANK", "NSE", "BUY", 1, price=900.0)
    kc.modify_order(oid, price=910.0)
    o = next(x for x in kc._paper_orders if x["order_id"] == oid)
    assert o["price"] == 910.0

run("_TokenBucket.acquire() no raise",          t_bucket_acquire)
run("_TokenBucket throttle within 2s for 5",    t_bucket_throttle)
run("_with_retry returns value on success",      t_retry_success)
run("F&O lot sizes: >=20 entries, NIFTY=75",     t_fno_lot_sizes)
run("paper place_order returns PAPER-* id",      t_paper_place_order)
run("tag auto-truncated to 20 chars",            t_tag_truncated)
run("paper orders() returns list",               t_paper_orders_list)
run("paper positions() has 'net' key",           t_paper_positions_net)
run("BUY updates position quantity",             t_buy_updates_position)
run("BUY then SELL nets to qty=0",               t_buy_sell_nets_zero)
run("qty=0 raises InputException",               t_qty_zero_raises)
run("FNO qty snapped to lot multiple",           t_fno_qty_snapped)
run("valid MIS qty unchanged",                   t_valid_qty_unchanged)
run("squareoff_all zeros all positions",         t_squareoff_all)
run("paper margins() has 'equity' key",          t_margins_equity)
run("paper historical_data returns []",          t_hist_paper_empty)
run("order_history() finds by order_id",         t_order_history)
run("cancel_order() sets CANCELLED",             t_cancel_order)
run("modify_order() updates price",              t_modify_order)


# ══════════════════════════════════════════════════════════════════════════
# 3. RISK MANAGER
# ══════════════════════════════════════════════════════════════════════════
section("7. SEBI COMPLIANCE")
from sebi_compliance import SEBICompliance, KillSwitchState, APPROVED_ALGO_IDS

def t_sebi_5_algos():
    # the six original algos; the lab-validated agents add their own ids (ALGO-ATTN-008 ...)
    assert {"intraday", "fno", "swing", "scalping", "manual", "ai"} <= set(APPROVED_ALGO_IDS)

def t_sebi_algo_id_format():
    assert all(v.startswith("ALGO-") for v in APPROVED_ALGO_IDS.values())

def t_sebi_approves_valid():
    sc = SEBICompliance()
    ok_, algo_id, _ = sc.pre_order_check(
        "intraday","RELIANCE","NSE","BUY",1,"MARKET",2800.0,"sig","RANGING")
    assert ok_ is True and algo_id == "ALGO-INTRA-001"

def t_sebi_rejects_unknown_strategy():
    sc = SEBICompliance()
    ok_, _, _ = sc.pre_order_check(
        "ghost","RELIANCE","NSE","BUY",1,"MARKET",100.0,"sig","RANGING")
    assert ok_ is False

def t_sebi_rejects_when_paused():
    sc = SEBICompliance()
    sc.pause_trading("test")
    ok_, _, _ = sc.pre_order_check(
        "intraday","RELIANCE","NSE","BUY",1,"MARKET",100.0,"sig","RANGING")
    assert ok_ is False

def t_sebi_rejects_when_killed():
    sc = SEBICompliance()
    sc.trigger_kill_switch("crash")
    ok_, _, _ = sc.pre_order_check(
        "intraday","RELIANCE","NSE","BUY",1,"MARKET",100.0,"sig","RANGING")
    assert ok_ is False

def t_sebi_rejects_over_max_value():
    sc = SEBICompliance()
    ok_, _, reason = sc.pre_order_check(
        "intraday","RELIANCE","NSE","BUY",10000,"MARKET",99999.0,"sig","RANGING")
    assert ok_ is False and "exceeds" in reason.lower()

def t_sebi_resume_from_paused():
    sc = SEBICompliance()
    sc.pause_trading("test")
    ok_, msg = sc.resume_trading()
    assert ok_ is True and msg == "ACTIVE"

def t_sebi_resume_from_killed_fails():
    sc = SEBICompliance()
    sc.trigger_kill_switch("test")
    ok_, _ = sc.resume_trading()
    assert ok_ is False

def t_sebi_reset_reenables():
    sc = SEBICompliance()
    sc.trigger_kill_switch("test")
    # Pass the configured secret (or empty string when not set) so the reset succeeds
    sc.reset_kill_switch(secret=settings.kill_switch_reset_secret)
    ok_, _ = sc.resume_trading()
    assert ok_ is True

def t_sebi_record_executed():
    sc = SEBICompliance()
    sc.record_order_id("intraday","RELIANCE","oid-001")
    assert sc.status()["orders_executed_today"] == 1

def t_sebi_audit_log_today():
    sc = SEBICompliance()
    sc.pre_order_check("intraday","RELIANCE","NSE","BUY",1,"MARKET",100.0,"sig","RANGING")
    import datetime as dt
    recs = sc.query_audit_log(dt.date.today().isoformat())
    assert len(recs) >= 1

def t_sebi_audit_filter_strategy():
    sc = SEBICompliance()
    sc.pre_order_check("intraday","RELIANCE","NSE","BUY",1,"MARKET",100.0,"sig","RANGING")
    sc.pre_order_check("swing","TCS","NSE","BUY",1,"LIMIT",3500.0,"sig","RANGING")
    import datetime as dt
    recs = sc.query_audit_log(dt.date.today().isoformat(), strategy="intraday")
    assert all(r["strategy"] == "intraday" for r in recs)

def t_sebi_strategy_disclosure():
    sc = SEBICompliance()
    d = sc.get_strategy_logic_disclosure("intraday")
    assert d["algo_id"] == "ALGO-INTRA-001"

def t_sebi_full_disclosure():
    sc = SEBICompliance()
    doc = sc.get_disclosure_document()
    assert len(doc["algo_strategies"]) >= 6      # + the lab-validated agents' own algo ids

def t_sebi_ip_empty_allows_all():
    sc = SEBICompliance()
    assert sc.is_ip_allowed("1.2.3.4") is True

def t_sebi_ip_whitelist():
    sc = SEBICompliance()
    sc.add_whitelisted_ip("10.0.0.1")
    assert sc.is_ip_allowed("9.9.9.9") is False
    assert sc.is_ip_allowed("10.0.0.1") is True

def t_sebi_otr_tracking():
    sc = SEBICompliance()
    sc.pre_order_check("intraday","RELIANCE","NSE","BUY",1,"MARKET",100.0,"sig","RANGING")
    assert sc.status()["orders_placed_today"] == 1

run("7 approved algo IDs registered",          t_sebi_5_algos)
run("algo IDs have ALGO- prefix",              t_sebi_algo_id_format)
run("pre_order_check approves valid order",    t_sebi_approves_valid)
run("rejects unregistered strategy",           t_sebi_rejects_unknown_strategy)
run("rejects when paused",                     t_sebi_rejects_when_paused)
run("rejects when kill switch active",         t_sebi_rejects_when_killed)
run("rejects order exceeding max value",       t_sebi_rejects_over_max_value)
run("resume from PAUSED returns True+ACTIVE",  t_sebi_resume_from_paused)
run("resume from KILLED returns False",        t_sebi_resume_from_killed_fails)
run("reset_kill_switch re-enables trading",    t_sebi_reset_reenables)
run("record_order_id increments executed",     t_sebi_record_executed)
run("audit log records today's orders",        t_sebi_audit_log_today)
run("audit log filters by strategy",           t_sebi_audit_filter_strategy)
run("strategy disclosure returns algo_id",     t_sebi_strategy_disclosure)
run("full disclosure has 7 strategies",        t_sebi_full_disclosure)
run("empty whitelist allows all IPs",          t_sebi_ip_empty_allows_all)
run("whitelist blocks non-whitelisted IPs",    t_sebi_ip_whitelist)
run("OTR counter increments per check",        t_sebi_otr_tracking)


# ══════════════════════════════════════════════════════════════════════════
# 8. MARKET REGIME
# ══════════════════════════════════════════════════════════════════════════
section("17. ENCRYPTED KEY VAULT")
# ══════════════════════════════════════════════════════════════════════════
import tempfile as _vtmp
from pathlib import Path
import os
from secret_store import SecretStore as _Vault, apply_to as _vault_apply

def _vault():
    return _Vault(Path(_vtmp.mkdtemp()) / "s")

def t_vault_roundtrip_encrypted():
    v = _vault(); v.save({"anthropic_api_key": "sk-ant-secret-1234", "kite_api_key": "kitekey9876"})
    assert v.load() == {"anthropic_api_key": "sk-ant-secret-1234", "kite_api_key": "kitekey9876"}
    raw = v.vault.read_bytes()
    assert b"sk-ant" not in raw and b"kitekey" not in raw

def t_vault_file_permissions():
    v = _vault(); v.save({"kite_api_key": "abc"})
    assert v.keyfile.stat().st_mode & 0o777 == 0o600
    assert v.vault.stat().st_mode & 0o777 == 0o600
    assert v.dir.stat().st_mode & 0o777 == 0o700

def t_vault_status_has_no_values():
    v = _vault(); v.save({"kite_api_secret": "verysecretvalue42"})
    st = v.status()
    assert st["fields"]["kite_api_secret"]["stored"] and st["fields"]["kite_api_secret"]["hint"] == "…ue42"
    assert "verysecret" not in str(st)

def t_vault_rejects_unknown_fields():
    v = _vault()
    try:
        v.save({"api_key": "x"}); assert False, "should reject"
    except ValueError:
        pass

def t_vault_kite_token_expires_next_day():
    v = _vault(); v.save({"kite_access_token": "tok"})
    assert v.load()["kite_access_token"] == "tok"
    d = v._read(); d["kite_access_token"]["day"] = "2000-01-01"; v._write(d)
    assert "kite_access_token" not in v.load()
    assert v.status()["fields"]["kite_access_token"]["expired"] is True

def t_vault_delete():
    v = _vault(); v.save({"kite_api_key": "a", "kite_api_secret": "b"})
    v.delete(["kite_api_key"]); assert list(v.load()) == ["kite_api_secret"]
    v.delete(["kite_api_secret"]); assert not v.vault.exists()

def t_vault_wrong_key_does_not_crash_startup():
    from cryptography.fernet import Fernet
    v = _vault(); v.save({"kite_api_key": "a"})
    old = os.environ.get("SECRETS_MASTER_KEY")
    os.environ["SECRETS_MASTER_KEY"] = Fernet.generate_key().decode()
    try:
        class S: pass
        assert _vault_apply(S(), v) == []
        assert "cannot be decrypted" in v.status()["error"]
    finally:
        if old is None: os.environ.pop("SECRETS_MASTER_KEY", None)
        else: os.environ["SECRETS_MASTER_KEY"] = old

def t_vault_apply_overrides_settings():
    v = _vault(); v.save({"anthropic_api_key": "from-vault"})
    class S: anthropic_api_key = "from-env"
    s = S(); assert _vault_apply(s, v) == ["anthropic_api_key"] and s.anthropic_api_key == "from-vault"

run("vault: round-trip, nothing stored in plain text",       t_vault_roundtrip_encrypted)
run("vault: key 600, vault 600, dir 700",                    t_vault_file_permissions)
run("vault: status shows last 4 chars only",                 t_vault_status_has_no_values)
run("vault: unknown fields rejected",                        t_vault_rejects_unknown_fields)
run("vault: Kite access token ignored the next day",         t_vault_kite_token_expires_next_day)
run("vault: delete fields; empty vault removed",             t_vault_delete)
run("vault: wrong master key → server still starts",         t_vault_wrong_key_does_not_crash_startup)
run("vault: stored values override .env values",             t_vault_apply_overrides_settings)


# ══════════════════════════════════════════════════════════════════════════
section("20. AUTOMATIC DAILY KITE LOGIN (TOTP)")
# ══════════════════════════════════════════════════════════════════════════
import httpx as _hx
import kite_auto_login as _kal
from config import settings as _kal_settings

class _FakeZerodha:
    """Transport that answers like kite.zerodha.com's login endpoints."""
    def __init__(self, password_ok=True, totp_ok=True):
        self.password_ok, self.totp_ok, self.calls = password_ok, totp_ok, []
    def handler(self, req: _hx.Request) -> _hx.Response:
        self.calls.append((req.method, req.url.path, dict(req.url.params)))
        path = req.url.path
        if path == "/connect/login" and "sess_id" not in req.url.params:
            return _hx.Response(302, headers={"location": f"https://kite.zerodha.com/connect/login?sess_id=SESS1&api_key={req.url.params['api_key']}"})
        if path == "/connect/login":
            return _hx.Response(200, text="<html>login</html>")
        if path == "/api/login":
            form = dict(x.split("=") for x in req.content.decode().split("&"))
            if not self.password_ok or form.get("password") != "pw":
                return _hx.Response(403, json={"status": "error", "message": "Invalid username or password"})
            return _hx.Response(200, json={"status": "success", "data": {"request_id": "REQ1", "twofa_type": "totp"}})
        if path == "/api/twofa":
            form = dict(x.split("=") for x in req.content.decode().split("&"))
            if not self.totp_ok or form.get("twofa_value") != "123456" or form.get("request_id") != "REQ1":
                return _hx.Response(403, json={"status": "error", "message": "Invalid TOTP"})
            return _hx.Response(200, json={"status": "success", "data": {}})
        if path == "/connect/finish":
            return _hx.Response(302, headers={"location": "https://app.example.com/auth/kite/callback?request_token=RT42&action=login&status=success"})
        return _hx.Response(404, text="nope")

def _kal_env(fn):
    saved = {k: getattr(_kal_settings, k) for k in _kal.REQUIRED}
    try:
        for k, v in {"kite_api_key": "apikey", "kite_api_secret": "sec", "kite_user_id": "AB1234",
                     "kite_password": "pw", "kite_totp_secret": "JBSWY3DPEHPK3PXP"}.items():
            setattr(_kal_settings, k, v)
        fn()
    finally:
        for k, v in saved.items(): setattr(_kal_settings, k, v)

def t_kal_full_flow_returns_request_token():
    def body():
        z = _FakeZerodha()
        c = _hx.Client(transport=_hx.MockTransport(z.handler), follow_redirects=True)
        tok = _kal.fetch_request_token(client=c, totp_now=lambda: "123456")
        assert tok == "RT42", tok
        paths = [p for _, p, _ in z.calls]
        assert paths[:1] == ["/connect/login"] and "/api/login" in paths and "/api/twofa" in paths and paths[-1] == "/connect/finish"
        # the callback URL (our own server) is never requested from here
        assert all("example.com" not in p for p in paths)
    _kal_env(body)

def t_kal_wrong_totp_is_a_clear_error():
    def body():
        z = _FakeZerodha(totp_ok=False)
        c = _hx.Client(transport=_hx.MockTransport(z.handler), follow_redirects=True)
        try:
            _kal.fetch_request_token(client=c, totp_now=lambda: "000000"); assert False, "should fail"
        except _kal.KiteLoginError as e:
            assert "TOTP" in str(e) and "Invalid TOTP" in str(e)
    _kal_env(body)

def t_kal_wrong_password_is_a_clear_error():
    def body():
        z = _FakeZerodha(password_ok=False)
        c = _hx.Client(transport=_hx.MockTransport(z.handler), follow_redirects=True)
        try:
            _kal.fetch_request_token(client=c, totp_now=lambda: "123456"); assert False, "should fail"
        except _kal.KiteLoginError as e:
            assert "password" in str(e)
    _kal_env(body)

def t_kal_missing_credentials_named():
    saved = _kal_settings.kite_totp_secret; _kal_settings.kite_totp_secret = ""
    try:
        assert not _kal.credentials_ready() and "KITE_TOTP_SECRET" in _kal.missing_credentials()
        assert _kal.status()["enabled"] is False
        try:
            _kal.fetch_request_token(client=_hx.Client(transport=_hx.MockTransport(lambda r: _hx.Response(500)))); assert False
        except _kal.KiteLoginError as e:
            assert "KITE_TOTP_SECRET" in str(e)
    finally:
        _kal_settings.kite_totp_secret = saved

def t_kal_totp_seed_generates_codes():
    import pyotp
    code = pyotp.TOTP("JBSWY3DPEHPK3PXP").now()
    assert len(code) == 6 and code.isdigit()

def t_kal_vault_stores_login_fields():
    v = _vault(); v.save({"kite_user_id": "AB1234", "kite_password": "pw", "kite_totp_secret": "JBSWY3DPEHPK3PXP"})
    assert set(v.load()) == {"kite_user_id", "kite_password", "kite_totp_secret"}
    assert b"JBSWY3DP" not in v.vault.read_bytes() and b"AB1234" not in v.vault.read_bytes()
    st = v.status()["fields"]
    assert st["kite_totp_secret"]["stored"] and "JBSWY3DP" not in str(st)

def t_kal_endpoint_rejects_six_digit_code_as_seed():
    import main as _main
    c = TestClient(_main.app)
    prev = _ai_settings.api_key; _ai_settings.api_key = "k"; H = {"X-API-Key": "k"}
    try:
        r = c.post("/settings/credentials", json={"kite_totp_secret": "123456", "persist": False}, headers=H)
        assert r.status_code == 422 and "base32" in r.json()["detail"]
        r = c.post("/settings/credentials", json={"kite_totp_secret": "not base32 !!!!!", "persist": False}, headers=H)
        assert r.status_code == 422
        r = c.get("/auth/kite/auto-login/status", headers=H)
        assert r.status_code == 200 and "enabled" in r.json() and "schedule_ist" in r.json()
        r = c.post("/auth/kite/auto-login", headers=H)
        assert r.status_code in (422, 502)     # no credentials in the test environment
    finally:
        _ai_settings.api_key = prev

def t_kal_needed_now_false_without_credentials():
    saved = _kal_settings.kite_user_id; _kal_settings.kite_user_id = ""
    try:
        assert _kal.needed_now() is False
    finally:
        _kal_settings.kite_user_id = saved

run("auto-login: user id + password + TOTP → request_token",  t_kal_full_flow_returns_request_token)
run("auto-login: wrong TOTP reported clearly",                t_kal_wrong_totp_is_a_clear_error)
run("auto-login: wrong password reported clearly",            t_kal_wrong_password_is_a_clear_error)
run("auto-login: missing credentials are named",              t_kal_missing_credentials_named)
run("auto-login: TOTP seed produces 6-digit codes",           t_kal_totp_seed_generates_codes)
run("auto-login: vault stores user id/password/seed encrypted", t_kal_vault_stores_login_fields)
run("auto-login: endpoint rejects a 6-digit code as the seed", t_kal_endpoint_rejects_six_digit_code_as_seed)
run("auto-login: not needed without credentials",             t_kal_needed_now_false_without_credentials)

def t_kal_needed_now_weekend_and_weekday_window():
    from datetime import datetime as _dt
    import pytz as _pz
    ist = _pz.timezone("Asia/Kolkata")
    saved = (_kal.now_ist, _kal.credentials_ready, _kal.token_valid_today)
    try:
        _kal.credentials_ready = lambda: True
        _kal.token_valid_today = lambda: False
        for when, want in [(_dt(2026, 11, 14, 8, 45), True),    # Saturday morning
                           (_dt(2026, 11, 15, 20, 0), True),    # Sunday evening
                           (_dt(2026, 11, 16, 8, 45), True),    # Monday before the close
                           (_dt(2026, 11, 16, 16, 0), False)]:  # Monday after the close
            _kal.now_ist = (lambda w=when: ist.localize(w))
            assert _kal.needed_now() is want, (when, want)
        _kal.token_valid_today = lambda: True
        _kal.now_ist = lambda: ist.localize(_dt(2026, 11, 14, 8, 45))
        assert _kal.needed_now() is False                        # a working token: nothing to do
    finally:
        _kal.now_ist, _kal.credentials_ready, _kal.token_valid_today = saved

def t_kal_refresh_jobs_run_every_day():
    import platform_scheduler as _ps
    src = _inspect_kal.getsource(_ps.PlatformScheduler.start) if hasattr(_ps, "PlatformScheduler") else open(_ps.__file__).read()
    for job in ('id="kite_refresh"', 'id="kite_refresh_last"'):
        line = next(l for l in src.splitlines() if job in l)
        assert 'day_of_week="*"' in line, line

import inspect as _inspect_kal
run("auto-login: weekend login any time, weekday only before 15:30", t_kal_needed_now_weekend_and_weekday_window)
run("auto-login: 08:45/08:55/09:05 refresh jobs run every day",      t_kal_refresh_jobs_run_every_day)


# ══════════════════════════════════════════════════════════════════════════
section("21. DASHBOARD AUTH: ADMIN LOGIN (JWT) IS ENOUGH, API KEY OPTIONAL")
# ══════════════════════════════════════════════════════════════════════════
from auth import create_token as _mk_jwt

def _auth_env(fn):
    import main as _main
    prev_key, prev_jwt = _ai_settings.api_key, _ai_settings.jwt_secret_key
    _ai_settings.api_key = "server-api-key"; _ai_settings.jwt_secret_key = "x" * 40
    try:
        fn(TestClient(_main.app))
    finally:
        _ai_settings.api_key, _ai_settings.jwt_secret_key = prev_key, prev_jwt

def t_auth_post_needs_credentials():
    def body(c):
        r = c.post("/settings/credentials", json={"anthropic_api_key": "sk-x", "persist": False})
        assert r.status_code == 401 and "X-API-Key or Bearer" in r.json()["detail"]
    _auth_env(body)

def t_auth_jwt_alone_is_enough_for_a_save():
    def body(c):
        tok, _ = _mk_jwt("admin")
        prev = _ai_settings.kite_user_id
        try:
            r = c.post("/settings/credentials", json={"kite_user_id": "AB1234", "persist": False},
                       headers={"Authorization": f"Bearer {tok}"})
            assert r.status_code == 200, r.text
            assert _ai_settings.kite_user_id == "AB1234"
            r = c.get("/settings/credentials/status", headers={"Authorization": f"Bearer {tok}"})
            assert r.status_code == 200 and r.json()["credentials"]["kite_user_id"] is True
        finally:
            _ai_settings.kite_user_id = prev
    _auth_env(body)

def t_auth_bad_jwt_rejected():
    def body(c):
        r = c.post("/settings/credentials", json={"anthropic_api_key": "sk-x", "persist": False},
                   headers={"Authorization": "Bearer not.a.token"})
        assert r.status_code == 401
    _auth_env(body)


run("dashboard auth: a POST without credentials is 401",      t_auth_post_needs_credentials)
run("dashboard auth: admin JWT alone saves credentials",      t_auth_jwt_alone_is_enough_for_a_save)
run("dashboard auth: a bad JWT is refused",                   t_auth_bad_jwt_rejected)

def t_auth_password_change_persists_in_vault():
    import main as _main, secret_store as _ss
    from auth import verify_password as _vp
    v = _vault()
    prev_store, prev_hash, prev_user = _ss.secret_store, _ai_settings.admin_password_hash, _ai_settings.admin_username
    _ss.secret_store = v
    prev_key = _ai_settings.api_key; _ai_settings.api_key = "k"; H = {"X-API-Key": "k"}
    try:
        c = TestClient(_main.app)
        r = c.post("/settings/app-password", json={"new_password": "brand-new-pass-1"}, headers=H)
        assert r.status_code == 200 and r.json()["persisted"] is True, r.text
        stored = v.load()
        assert _vp("brand-new-pass-1", stored["admin_password_hash"]) and stored["admin_username"] == _ai_settings.admin_username
        assert b"brand-new" not in v.vault.read_bytes()
        # a fresh process applies the vault over the environment value
        class S: admin_password_hash = "from-env"; admin_username = "admin"
        s2 = S(); _vault_apply(s2, v)
        assert _vp("brand-new-pass-1", s2.admin_password_hash)
    finally:
        _ss.secret_store = prev_store
        _ai_settings.admin_password_hash, _ai_settings.admin_username, _ai_settings.api_key = prev_hash, prev_user, prev_key

run("dashboard auth: password change survives a restart (vault)", t_auth_password_change_persists_in_vault)






# ══════════════════════════════════════════════════════════════════════════
section("25. OPTIONS REPLAY (research only: Black-Scholes on index minute bars)")
# ══════════════════════════════════════════════════════════════════════════
import math as _ormath
from datetime import date as _ordate, datetime as _ordt, time as _ortime, timedelta as _ortd
import options_replay as _orp


def _or_bars(days: list, path=lambda i: 0.0, base: float = 24000.0) -> list:
    """Synthetic minute bars 09:15-15:29 for each date; close = base + path(global minute index)."""
    out, k = [], 0
    for d in days:
        t0 = _ordt.combine(d, _ortime(9, 15))
        for m in range(375):
            c = base + path(k)
            prev = base + path(k - 1) if k else c
            out.append((t0 + _ortd(minutes=m), prev, max(prev, c) + 1, min(prev, c) - 1, c))
            k += 1
    return out


_OR_DAYS = [_ordate(2026, 10, d) for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)]
_OR_VIX = {_ordate(2026, 10, 2) + _ortd(days=i): 13.0 for i in range(-5, 20)}


def t_or_put_call_parity():
    r = _orp.RISK_FREE
    for S, K, T, sig in ((24000, 24000, 5 / 365, 0.13), (24000, 24500, 30 / 365, 0.2),
                         (52000, 50000, 2 / 365, 0.35), (24000, 23000, 0.5, 0.1)):
        c, p = _orp.bs(S, K, T, sig, "CE"), _orp.bs(S, K, T, sig, "PE")
        assert abs((c - p) - (S - K * _ormath.exp(-r * T))) < 1e-6, (S, K, c, p)
        assert c > 0 and p > 0


def t_or_delta_range():
    T, sig = 6 / 365, 0.14
    for K in range(22000, 26001, 250):
        dc, dp = _orp.bs_delta(24000, K, T, sig, "CE"), _orp.bs_delta(24000, K, T, sig, "PE")
        assert 0.0 <= dc <= 1.0 and -1.0 <= dp <= 0.0, (K, dc, dp)
        assert abs((dc - dp) - 1.0) < 1e-9
    assert abs(_orp.bs_delta(24000, 24000, T, sig, "CE") - 0.5) < 0.05       # ATM call ~0.5
    v = _orp.MarketView("NIFTY", _ordt(2026, 10, 14, 10, 20), 24013.0, [], [], 0.13,
                        _ordate(2026, 10, 19), 50, _orp.years_between(_ordt(2026, 10, 14, 10, 20), _ordate(2026, 10, 19)))
    kc, kp = v.strike_for_delta("CE", 0.175), v.strike_for_delta("PE", 0.175)
    assert kc % 50 == 0 and kp % 50 == 0 and kc > 24013 > kp, (kc, kp)
    assert 0.1 < v.delta("CE", kc) < 0.25 and -0.25 < v.delta("PE", kp) < -0.1


def t_or_expiry_calendar():
    td = lambda d: d.weekday() < 5 and d != _ordate(2026, 10, 20)           # Dussehra (Tue) closed
    assert _orp.next_expiry("NIFTY", _ordate(2026, 10, 7), td) == _ordate(2026, 10, 13)       # Tuesday
    assert _orp.next_expiry("NIFTY", _ordate(2026, 10, 13), td) == _ordate(2026, 10, 19)      # skip today; holiday → Mon
    assert _orp.next_expiry("NIFTY", _ordate(2026, 10, 13), td, skip_today=False) == _ordate(2026, 10, 13)
    assert _orp.next_expiry("NIFTY", _ordate(2025, 6, 2), td) == _ordate(2025, 6, 5)          # Thursday era
    assert _orp.next_expiry("BANKNIFTY", _ordate(2026, 10, 7), td) == _ordate(2026, 10, 27)   # monthly, last Tue
    assert _orp.next_expiry("BANKNIFTY", _ordate(2026, 10, 28), td) == _ordate(2026, 11, 24)
    assert _orp.round_strike(24026, 50) == 24050 and _orp.round_strike(52049, 100) == 52000


class _OrSpy(_orp.OptionStrategy):
    name = "SPY"

    def __init__(self):
        self.decision_times = [_ortime(10, 20)]
        self.hold = "intraday"
        self.seen = []

    def decide(self, v):
        self.seen.append((v.now, v.spot, max(b[0] for b in v.bars), len(v.bars), v.today_bars()[-1][4]))
        return _orp.IronCondor().decide(v)


def t_or_no_lookahead():
    days = _OR_DAYS[:3]
    path = lambda i: 40 * _ormath.sin(i / 50.0)
    cfg = _orp.ReplayConfig(underlying="NIFTY")
    spy = _OrSpy()
    res = _orp.Replay(_or_bars(days, path), spy, cfg, _OR_VIX).run()
    assert len(spy.seen) == 3 and len(res.trades) == 3
    for now, spot, last_ts, n, last_close in spy.seen:
        assert last_ts < now and last_ts == now - _ortd(minutes=1), (now, last_ts)
    # the same past with a wildly different future must give the same decisions and fills
    wild = lambda i: path(i) if (i % 375) < 65 else path(i) + (900 if (i // 375) % 2 else -900)
    spy2 = _OrSpy()
    res2 = _orp.Replay(_or_bars(days, wild), spy2, cfg, _OR_VIX).run()
    assert [s[:4] for s in spy.seen] == [s[:4] for s in spy2.seen]
    for a, b in zip(res.trades, res2.trades):
        assert [(l.opt, l.strike, l.side, l.entry_px) for l in a.legs] == \
               [(l.opt, l.strike, l.side, l.entry_px) for l in b.legs]
    assert any(a.net != b.net for a, b in zip(res.trades, res2.trades))   # ...but different outcomes
    # the trend20 / first-hour signals use only prior sessions and today's bars before 10:15
    cs = _orp.CreditSpread(direction="first_hour")
    v = _orp.MarketView("NIFTY", _ordt(2026, 10, 5, 10, 15), 24000.0,
                        [b for b in _or_bars([_OR_DAYS[0]], lambda i: i * 1.0) if b[0].time() < _ortime(10, 15)],
                        [], 0.13, _ordate(2026, 10, 6), 50, 1 / 365)
    assert cs.signal(v) == 1 and cs.decide(v)[0].opt == "PE"                   # up first hour → bull put


def t_or_costs_per_leg():
    days = _OR_DAYS[:2]
    res = _orp.Replay(_or_bars(days, lambda i: 30 * _ormath.sin(i / 40.0)), _orp.IronCondor(),
                      _orp.ReplayConfig(underlying="NIFTY"), _OR_VIX).run()
    assert res.trades, "no trade"
    for tr in res.trades:
        assert len(tr.legs) == 4 and tr.qty == _orp.LOT_SIZE["NIFTY"]
        manual = 0.0
        for l in tr.legs:
            ev, xv = l.entry_px * tr.qty, l.exit_px * tr.qty
            buy_v, sell_v = (ev, xv) if l.side > 0 else (xv, ev)
            exp = _orp.round_trip_cost(buy_v, sell_v, "option")
            assert abs(l.cost - exp) < 0.02, (l, exp)
            manual += exp
        assert abs(tr.costs - manual) < 0.05 and tr.costs >= 4 * 2 * 20 * 1.18   # ₹20 per order per leg + GST
        gross = sum((l.exit_px - l.entry_px) * l.side * tr.qty for l in tr.legs)
        assert abs(tr.gross - gross) < 0.05 and abs(tr.net - (tr.gross - tr.costs)) < 1e-6
        assert tr.max_risk > 0 and tr.entry_net > 0                                # defined-risk credit


def t_or_haircut_and_settlement():
    assert _orp.fill_price(100.0, +1, "NIFTY") == 102.0 and _orp.fill_price(100.0, -1, "NIFTY") == 98.0
    assert _orp.fill_price(10.0, +1, "NIFTY") == 11.0 and _orp.fill_price(0.3, -1, "NIFTY") == _orp.TICK
    # a long ITM leg settled at expiry pays exercise STT (0.15 % of intrinsic, 2026), which is exactly the
    # sell-side option STT round_trip_cost already charges on the settlement value: nothing extra
    plain = _orp.leg_cost(50.0, 120.0, +1, 75)
    assert _orp.EXERCISE_STT_EXTRA == 0.0
    assert abs(_orp.leg_cost(50.0, 120.0, +1, 75, settled=True) - plain) < 0.02
    assert plain >= 120 * 75 * 0.0015                                    # the 0.15 % is in there
    # hold-to-expiry condor settles at intrinsic of the 15:00-15:29 mean on expiry day
    days = [_ordate(2026, 10, 9), _ordate(2026, 10, 12), _ordate(2026, 10, 13)]
    ic = _orp.IronCondor(hold="expiry", target_frac=99, stop_mult=99)
    res = _orp.Replay(_or_bars(days, lambda i: 0.0), ic, _orp.ReplayConfig(underlying="NIFTY"), _OR_VIX).run()
    tr = res.trades[0]
    assert tr.expiry == _ordate(2026, 10, 13) and tr.reason == "expiry", (tr.expiry, tr.reason)
    assert all(l.exit_px == 0.0 for l in tr.legs) and tr.gross > 0                # flat market: all OTM
    assert len(res.trades) == 1                                                    # one position at a time


def t_or_metrics_and_loader():
    import json as _j
    T = lambda d, n: _orp.Trade("X", "X", "NIFTY", _ordt.combine(d, _ortime(10)), d, 0, 0, "", 1, [], 0,
                                exit_ts=_ordt.combine(d, _ortime(15)), gross=n, costs=0)
    res = _orp.ReplayResult([T(_OR_DAYS[0], 100), T(_OR_DAYS[1], -300), T(_OR_DAYS[6], 200)], _OR_DAYS)
    m = _orp.metrics(res)
    assert m["all"]["trades"] == 3 and m["all"]["net"] == 0 and m["all"]["max_drawdown"] == 300
    assert abs(m["all"]["profit_factor"] - 1.0) < 1e-9 and m["all"]["win_rate"] == 66.7
    assert m["first_half"]["trades"] == 2 and m["second_half"]["trades"] == 1
    bars = [[b[0].isoformat(), *b[1:], 0] for b in _or_bars(_OR_DAYS[:1])]
    with _tmp0.TemporaryDirectory() as dd:
        with open(_os.path.join(dd, "index.json"), "w") as fh:
            _j.dump({"NIFTY 50": {"bars": bars}, "NIFTY BANK": {"bars": bars}}, fh)
        u, vix = _orp.load_data(_os.path.join(dd, "index.json"))
        assert set(u) == {"NIFTY", "BANKNIFTY"} and vix is None and len(u["NIFTY"]) == 375
        d2 = _os.path.join(dd, "m"); _os.mkdir(d2)
        for nm, sym in (("_NIFTY50_min.json", "NIFTY 50"), ("_NIFTYBANK_min.json", "NIFTY BANK")):
            with open(_os.path.join(d2, nm), "w") as fh:
                _j.dump({"symbol": sym, "interval": "minute", "bars": bars}, fh)
        with open(_os.path.join(d2, "_VIX_day.json"), "w") as fh:
            _j.dump({"symbol": "INDIA VIX", "interval": "day",
                     "bars": [["2026-10-02T00:00:00", 12, 13, 11, 12.5, 0]]}, fh)
        u, vix = _orp.load_data(d2)
        assert set(u) == {"NIFTY", "BANKNIFTY"} and vix == {_ordate(2026, 10, 2): 12.5}
    rv = _orp.realised_vol([24000 * (1.01 if i % 2 else 1.0) for i in range(21)])
    assert rv is not None and 0.1 < rv < 0.3 and _orp.realised_vol([1.0] * 5) is None


run("options replay: Black-Scholes put-call parity",                       t_or_put_call_parity)
run("options replay: deltas in range, delta-picked strikes on the grid",   t_or_delta_range)
run("options replay: NIFTY weekly Tue / BANKNIFTY monthly expiry, holidays", t_or_expiry_calendar)
run("options replay: no look-ahead (decisions see only earlier bars)",     t_or_no_lookahead)
run("options replay: NSE costs charged per leg via round_trip_cost",       t_or_costs_per_leg)
run("options replay: haircut direction, expiry settlement, exercise STT",  t_or_haircut_and_settlement)
run("options replay: metrics, halves and both data layouts load",          t_or_metrics_and_loader)


# ── stratlab/options_vrp: pre-registered weekly NIFTY short strangles (VRP) ──
from stratlab import options_vrp as _vrp
from stratlab import protocol as _vrp_P

# Oct 2026 weekdays; 2026-10-20 (Tue) is a holiday, so that week's expiry moves to Mon 10-19.
_VRP_DAYS = [_ordate(2026, 10, d) for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 21, 22, 23)]
_VRP_VIX = {_ordate(2026, 9, 20) + _ortd(days=i): 13.0 for i in range(60)}


def _vrp_spec(wing=None, lots=1, min_days=3):
    return _vrp.OptSpec("T-" + str(wing), "t", {"short_delta": 0.16, "min_days_left": min_days, "wing_pts": wing},
                        lots=lots)


def t_vrp_expiry_transition_and_cycle():
    td = _orp._calendar_trading_day
    # the first Tuesday expiry: from Fri 2025-08-29 the next weekly expiry is Tue 2025-09-02 (not Thu 09-04)
    assert _orp.next_expiry("NIFTY", _ordate(2025, 8, 29), td) == _ordate(2025, 9, 2)
    assert _orp.next_expiry("NIFTY", _ordate(2025, 8, 26), td) == _ordate(2025, 8, 28)
    # entries only on the first trading day after an expiry, never re-entered later in the week
    res, strat = _vrp._run_replay(_vrp_spec(), _or_bars(_VRP_DAYS, lambda i: 20 * _ormath.sin(i / 60.0)),
                                  _VRP_VIX, 20261023)
    ent = [(t.entry_ts.date(), t.expiry, t.reason) for t in res.trades]
    assert [e[0] for e in ent] == [_ordate(2026, 10, 7), _ordate(2026, 10, 14), _ordate(2026, 10, 21)], ent
    assert ent[0][1] == _ordate(2026, 10, 13) and ent[1][1] == _ordate(2026, 10, 19)
    assert ent[0][2] == ent[1][2] == "expiry" and ent[2][2] == "data_end"
    assert all(t.entry_ts.time() == _ortime(10, 20) for t in res.trades)
    # with 10-15 and 10-16 also closed only 10-19 is left after 10-14 (< 3 days): that week is skipped, not entered later
    days = [d for d in _VRP_DAYS if d not in (_ordate(2026, 10, 15), _ordate(2026, 10, 16))]
    res, strat = _vrp._run_replay(_vrp_spec(), _or_bars(days), _VRP_VIX, 20261023)
    assert [t.entry_ts.date() for t in res.trades] == [_ordate(2026, 10, 7), _ordate(2026, 10, 21)]
    assert strat.skipped_short_weeks == [_ordate(2026, 10, 14)]
    assert _orp.trading_days_left(_ordate(2026, 10, 7), _ordate(2026, 10, 13), lambda d: d.weekday() < 5) == 4


def t_vrp_strikes_wings_lots_margin():
    res, _ = _vrp._run_replay(_vrp_spec(wing=400.0, lots=2), _or_bars(_VRP_DAYS), _VRP_VIX, 20261023)
    tr = res.trades[0]
    ce = [l for l in tr.legs if l.opt == "CE"]; pe = [l for l in tr.legs if l.opt == "PE"]
    sc, lc = [l for l in ce if l.side < 0][0], [l for l in ce if l.side > 0][0]
    sp_, lp = [l for l in pe if l.side < 0][0], [l for l in pe if l.side > 0][0]
    assert lc.strike == sc.strike + 400 and lp.strike == sp_.strike - 400 and sc.strike % 50 == 0
    T = _orp.years_between(tr.entry_ts, tr.expiry)
    for l, want in ((sc, 0.16), (sp_, 0.16)):                              # closest to 0.16 on the 50-pt grid
        d0 = abs(abs(_orp.bs_delta(tr.spot_in, l.strike, T, 0.13, l.opt)) - want)
        for k in (l.strike - 50, l.strike + 50):
            assert d0 <= abs(abs(_orp.bs_delta(tr.spot_in, k, T, 0.13, l.opt)) - want) + 1e-12
    assert tr.qty == 2 * 65 and tr.max_risk > 0                             # 2026 dates: lot 65
    sp = _vrp_spec(wing=400.0, lots=2)
    assert abs(tr.max_risk - (400 - tr.entry_net) * tr.qty) < 0.05
    assert abs(sp.margin(tr) - (tr.max_risk + 0.02 * tr.spot_in * tr.qty)) < 1e-6
    tr.max_risk = 0.0
    assert abs(_vrp_spec().margin(tr) - 0.12 * tr.spot_in * tr.qty) < 1e-6
    # the pre-registered lot size in force on the entry date: 75, then 65 from 2025-12-31
    assert _orp.lot_size_on(_vrp.LOT_SCHEDULE, _ordate(2025, 12, 30)) == 75
    assert _orp.lot_size_on(_vrp.LOT_SCHEDULE, _ordate(2025, 12, 31)) == 65
    assert _orp.lot_size_on(_vrp.LOT_SCHEDULE, _ordate(2023, 11, 10)) == 75
    res, _ = _vrp._run_replay(_vrp_spec(), _or_bars(_VRP_DAYS), _VRP_VIX, 20261023)
    assert all(t.qty == 65 for t in res.trades)                             # 2026 dates: 65


def t_vrp_no_lookahead():
    path = lambda i: 25 * _ormath.sin(i / 70.0)
    base = _or_bars(_VRP_DAYS, path)
    # 1) a wildly different future (every bar after the 10:20 entry bar) leaves that entry unchanged
    a, _ = _vrp._run_replay(_vrp_spec(400.0), base, _VRP_VIX, 20261023)
    key = lambda r: [(t.entry_ts, [(l.opt, l.strike, l.side, l.entry_px) for l in t.legs]) for t in r.trades]
    assert len(a.trades) == 3
    for k, t in enumerate(a.trades):
        wild = [(x[0], x[1] + 700, x[2] + 700, x[3] + 700, x[4] + 700) if x[0] > t.entry_ts else x for x in base]
        b, _ = _vrp._run_replay(_vrp_spec(400.0), wild, _VRP_VIX, 20261023)
        assert key(b)[k] == key(a)[k] and b.trades[k].net != t.net
    # 2) today's VIX (not yet closed at 10:20) is never used: only the previous close
    vx = dict(_VRP_VIX); vx[_ordate(2026, 10, 7)] = 80.0
    c, _ = _vrp._run_replay(_vrp_spec(), base, vx, 20261023)
    assert key(c)[0] == key(_vrp._run_replay(_vrp_spec(), base, _VRP_VIX, 20261023)[0])[0]
    assert c.trades[0].sigma == 0.13
    # 3) the runtime audit agrees, and a DISCOVERY-style run up to `hi` never reads a later bar
    assert _vrp.audit_lookahead(_vrp_spec(400.0), base, _VRP_VIX, a.trades) is None
    assert _vrp.audit_lookahead(_vrp_spec(), base, _VRP_VIX, a.trades) is not None      # wrong legs are caught
    poisoned = base + [(b[0] + _ortd(days=28), float("nan"), float("nan"), float("nan"), float("nan"))
                       for b in _or_bars(_VRP_DAYS[-5:])]
    sim1 = _vrp.make_simulator(base, _VRP_VIX)([_vrp_spec()], 20261001, 20261016)
    sim2 = _vrp.make_simulator(poisoned, _VRP_VIX)([_vrp_spec()], 20261001, 20261016)
    assert sim1 == sim2 and len(sim1["T-None"]["trades"]) == 2 and sim1["T-None"]["lookahead"] is None


def t_vrp_costs_and_haircut():
    # flat market: both shorts expire worthless; fees = 2 sell orders at the 2026 option rates, no settlement haircut
    res, _ = _vrp._run_replay(_vrp_spec(), _or_bars(_VRP_DAYS[:7]), _VRP_VIX, 20261013)
    tr = res.trades[0]
    assert tr.reason == "expiry" and all(l.exit_px == 0.0 for l in tr.legs)
    man = 0.0
    for l in tr.legs:
        prem = l.entry_px * tr.qty
        bro, stt, exch, sebi = 20.0, 0.0015 * prem, 0.0003553 * prem, prem * 10 / 1e7
        man += bro + stt + exch + sebi + 0.18 * (bro + exch + sebi)
    assert abs(tr.costs - man) < 0.05, (tr.costs, man)
    # haircut: max(1 pt, 2 %) per leg at entry only (settlement is not a fill)
    T = _orp.years_between(tr.entry_ts, tr.expiry)
    for l in tr.legs:
        m = _orp.bs(tr.spot_in, l.strike, T, 0.13, l.opt)
        assert abs(l.entry_px - max(_orp.TICK, m - max(1.0, 0.02 * m))) < 1e-9
    assert abs(tr.slippage - sum(max(1.0, 0.02 * _orp.bs(tr.spot_in, l.strike, T, 0.13, l.opt)) for l in tr.legs) * tr.qty) < 0.05
    assert abs(tr.net - (tr.entry_net * tr.qty - tr.costs)) < 0.05
    # the 2.5x sensitivity haircut costs more and changes nothing else
    s5, _ = _vrp._run_replay(_vrp_spec(), _or_bars(_VRP_DAYS[:7]), _VRP_VIX, 20261013, haircut_mult=2.5)
    assert [l.strike for l in s5.trades[0].legs] == [l.strike for l in tr.legs] and s5.trades[0].net < tr.net
    # a short leg settled in the money is bought back at intrinsic: gross loss, STT only on the sale
    up = _or_bars(_VRP_DAYS[:7], lambda i: 0.0 if i < 2 * 375 + 65 else 900.0)
    tr2 = _vrp._run_replay(_vrp_spec(), up, _VRP_VIX, 20261013)[0].trades[0]
    ce = [l for l in tr2.legs if l.opt == "CE"][0]
    assert ce.exit_px > 0 and abs(ce.exit_px - (24900.0 - ce.strike)) < 1e-6 and tr2.net < 0


def t_vrp_protocol_ledger():
    import tempfile as _t, os as _o
    days = [d for d in (_ordate(2026, 1, 1) + _ortd(days=i) for i in range(400)) if d.weekday() < 5][:150]
    vix = {days[0] - _ortd(days=5) + _ortd(days=i): 13.0 for i in range(260)}
    bars = _or_bars(days, lambda i: 30 * _ormath.sin(i / 300.0))
    cal = _vrp.calendar_of(bars)
    sp = _vrp_P.split_sessions(cal)
    with _t.TemporaryDirectory() as td:
        led = _o.path.join(td, "final_ledger.json")
        seen = []
        sim = _vrp.make_simulator(bars, vix)
        spy = lambda specs, lo, hi: (seen.append((lo, hi)), sim(specs, lo, hi))[1]
        res, _ = _vrp_P.evaluate([_vrp_spec()], cal, spy, n_symbols=1, ledger_path=led, screen_only=True,
                                 cost_model=_vrp.COST_MODEL)
        assert seen == [sp.discovery] and not _o.path.exists(led)              # screen only: FINAL never simulated
        assert res[0]["screen"]["disc"]["trades"] > 0
        assert all(sp.discovery[0] <= t[2] <= sp.discovery[1] for t in sim([_vrp_spec()], *sp.discovery)["T-None"]["trades"])
    assert _vrp.COST_MODEL == _vrp._costs.COST_MODEL_VERSION + "/o1"


run("options VRP: first Tuesday expiry 2025-09-02; entries only the day after an expiry; short weeks skipped",
    t_vrp_expiry_transition_and_cycle)
run("options VRP: 0.16-delta strikes on the grid, +/-400 wings, lot 75 then 65, margin rules", t_vrp_strikes_wings_lots_margin)
run("options VRP: no look-ahead (future bars, same-day VIX, DISCOVERY run never reads later bars)", t_vrp_no_lookahead)
run("options VRP: 2026 option fees per order, entry-only haircut, ITM settlement, 2.5x sensitivity", t_vrp_costs_and_haircut)
run("options VRP: lab protocol on DISCOVERY only with --screen-only, no ledger write", t_vrp_protocol_ledger)

# ══════════════════════════════════════════════════════════════════════════
# STRATLAB — strategy screening lab (research only)
# ══════════════════════════════════════════════════════════════════════════
section("STRATLAB: simulator, protocol, catalogue")

import tempfile as _sl_tmp, json as _sl_json
from stratlab import data as _sld, sim as _sls, protocol as _slp, indicators as _sli
from stratlab.api import Entry as _SlEntry, Swing as _SlSwing, Spec as _SlSpec, REGISTRY as _SL_REG
import stratlab.catalog  # noqa: F401  (registers the strategies)
from costs import round_trip_cost as _sl_rtc


def _sl_flat(days=(20260105,), n=375, px=100.0, symbol="SYN"):
    """Flat sessions (every bar o=h=l=c=px) to which a test adds the bars it needs."""
    date = np.repeat(np.array(days, np.int32), n)
    t = np.tile(555 + np.arange(n), len(days)).astype(np.int16)
    a = {k: np.full(len(date), px) for k in "ohlc"}
    return date, t, a, np.full(len(date), 1000.0)


def _sl_session(a, date, t, v, k=0):
    sd = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    return _sli.build_sessions(sd)[k]


def _sl_walk(n_days=6, seed=7, symbol="RW", n=375):
    rng = np.random.default_rng(seed)
    days = [20260100 + d for d in range(1, n_days + 1)]
    date = np.repeat(np.array(days, np.int32), n)
    t = np.tile(555 + np.arange(n), n_days).astype(np.int16)
    c = 500.0 * np.exp(np.cumsum(rng.normal(0, 0.0012, n * n_days)))
    o = np.concatenate(([500.0], c[:-1]))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0004, len(c))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0004, len(c))))
    v = rng.integers(500, 5000, len(c)).astype(float)
    v[rng.random(len(c)) < 0.02] *= 8
    return _sld.from_arrays(symbol, date, t, o, h, l, c, v)


def _sl_spec(fn, kind="intraday", style="vector", horizon=None, **kw):
    hz = horizon or ("positional" if kind == "swing" else "intraday")
    return _SlSpec(id=kw.pop("id", "t_" + fn.__name__), kind=kind, family="test", desc="test", fn=fn, style=style,
                   horizon=hz, **kw)


def t_sl_bar_view_cannot_see_ahead():
    date, t, a, v = _sl_flat()
    s = _sl_session(a, date, t, v)
    seen = []

    def peeker(view, p):
        i = len(view) - 1
        seen.append((i, len(view.c), len(view.vwap), len(view.ema21), int(view.t[-1])))
        try:
            view.c[i + 1]
            raise AssertionError("bar i+1 was readable")
        except IndexError:
            pass
        return None
    _sls.sim_session(_sl_spec(peeker, style="bar"), s)
    assert len(seen) > 300
    for i, nc, nv, ne, tl in seen:
        assert nc == nv == ne == i + 1 and tl == 555 + i, (i, nc, nv, ne, tl)
    v10 = s.view(10)
    assert len(v10) == 10 and len(v10.h) == 10 and v10.idx(700) == 10
    assert v10.orb(15)[2] == 10 and s.orb(15)[2] == 15          # an unfinished range is visibly unfinished
    assert len(v10.bars5().c) == 2 and len(s.view(9).bars5().c) == 1   # only completed 5-minute bars


run("stratlab: a per-bar strategy receives bars 0..i only (bar i+1 is unreadable)", t_sl_bar_view_cannot_see_ahead)


def t_sl_future_bars_do_not_change_earlier_fills():
    ids = ["orb15_break", "vwap_reclaim", "scalp_ema9_21_cross", "scalp_range20_break", "inside_bar_5m"]
    specs = [_SL_REG[i] for i in ids]
    sd = _sl_walk()
    base = _sls.run_symbol("", "RW", specs, 0, 99999999, sd=sd)
    cut_day, cut_min = int(sd.days[-1]), 12 * 60
    sd2 = _sl_walk()
    m = (sd2.date == cut_day) & (sd2.t >= cut_min)
    for arr in (sd2.o, sd2.h, sd2.l, sd2.c):
        arr[m] *= 1.07                                           # rewrite the future: everything from 12:00
    sd2.v[m] *= 3
    alt = _sls.run_symbol("", "RW", specs, 0, 99999999, sd=sd2)
    n_before = 0
    for i in ids:
        early = lambda tr: [x for x in tr if x[6] < cut_day or x[7] < "12:00"]
        a, b = early(base[i]["trades"]), early(alt[i]["trades"])
        assert a == b, f"{i}: trades closed before 12:00 changed when later bars changed"
        n_before += len(a)
        assert base[i]["error"] is None and base[i]["lookahead"] is None, (i, base[i])
    assert n_before >= 10, n_before


run("stratlab: changing future bars does not change earlier fills", t_sl_future_bars_do_not_change_earlier_fills)


def t_sl_lookahead_audit_disqualifies_cheats():
    sd = _sl_walk(n_days=3)
    s = _sli.build_sessions(sd)[1]

    def cheat(s, p):                                             # at 09:25 it already knows the day's low
        return [_SlEntry(10, 1, stop=float(s.l.min()) * 0.999)]

    def honest(s, p):
        return [_SlEntry(10, 1, stop=float(s.c[10]) * 0.99)] if len(s) > 10 else []
    assert _sls.audit_lookahead(_sl_spec(honest), s) is None
    msg = _sls.audit_lookahead(_sl_spec(cheat), s)
    assert msg and "differ" in msg, msg
    res = {"t_cheat": {"trades": [], "skipped": 0, "lookahead": msg, "error": None}}
    out, _ = _slp.evaluate([_sl_spec(cheat)], range(20260001, 20260201), lambda specs, lo, hi: res)
    assert out[0]["verdict"] == "INVALID" and "LOOK-AHEAD" in out[0]["why"] and out[0]["final"] is None


run("stratlab: look-ahead audit catches a vector strategy that reads the future", t_sl_lookahead_audit_disqualifies_cheats)


def t_sl_fill_next_open_with_slippage():
    date, t, a, v = _sl_flat()
    a["o"][11] = 101.0; a["h"][11] = 101.0; a["c"][11] = 100.5; a["l"][11] = 100.5
    s = _sl_session(a, date, t, v)
    k, fill, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=99.0, target=110.0))
    assert abs(fill - 101.0 * (1 + _sls.SLIPPAGE)) < 1e-9, fill               # next bar's open, SLIPPAGE worse
    k2, fill2, _, _ = _sls.fill_intraday(s, _SlEntry(10, -1, stop=103.0, target=90.0))
    assert abs(fill2 - 101.0 * (1 - _sls.SLIPPAGE)) < 1e-9, fill2
    assert _sls.fill_intraday(s, _SlEntry(10, 1, stop=101.5)) is None      # stop already through at the open
    assert _sls.fill_intraday(s, _SlEntry(len(s) - 1, 1, stop=99.0)) is None  # no next bar
    rr = _sls.fill_intraday(s, _SlEntry(10, 1, stop=100.0, rr=2.0))
    assert rr is not None


run("stratlab: entries fill at the next bar's open with the default (2 bps) slippage", t_sl_fill_next_open_with_slippage)


def t_sl_stop_before_target_same_bar():
    date, t, a, v = _sl_flat()
    a["h"][20] = 103.0; a["l"][20] = 97.0                        # one bar touches target and stop
    s = _sl_session(a, date, t, v)
    k, fill, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=98.0, target=102.0))
    assert (k, reason) == (20, "stop") and abs(px - 98.0 * (1 - _sls.SLIPPAGE)) < 1e-9, (k, reason, px)
    k, fill, px, reason = _sls.fill_intraday(s, _SlEntry(10, -1, stop=102.0, target=98.0))
    assert (k, reason) == (20, "stop") and abs(px - 102.0 * (1 + _sls.SLIPPAGE)) < 1e-9, (k, reason, px)
    a["l"][20] = 100.0                                           # target only
    s = _sl_session(a, date, t, v)
    k, fill, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=98.0, target=102.0))
    assert (k, reason, px) == (20, "target", 102.0), (k, reason, px)
    a["o"][30] = 95.0; a["l"][30] = 95.0; a["h"][20] = 100.0     # gap through the stop: filled at the open
    s = _sl_session(a, date, t, v)
    k, fill, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=98.0, target=102.0))
    assert (k, reason) == (30, "stop") and abs(px - 95.0 * (1 - _sls.SLIPPAGE)) < 1e-9, (k, reason, px)
    k, fill, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=98.0, trail=("pct", 1.0)))
    assert (k, reason) == (30, "stop")


run("stratlab: stop is assumed when stop and target are hit in one bar; gaps fill at the open", t_sl_stop_before_target_same_bar)


def t_sl_squareoff_and_time_stop():
    date, t, a, v = _sl_flat()
    sq = 15 * 60 + 15 - 555                                      # index of the 15:15 bar
    a["o"][sq] = 101.0
    s = _sl_session(a, date, t, v)
    k, fill, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=90.0, target=120.0))
    assert (k, reason) == (sq, "squareoff") and abs(px - 101.0 * (1 - _sls.SLIPPAGE)) < 1e-9, (k, reason, px)
    assert _sls.fill_intraday(s, _SlEntry(sq - 1, 1, stop=90.0)) is None      # no entries from 15:15
    assert _sls.fill_intraday(s, _SlEntry(sq - 2, 1, stop=90.0)) is not None
    k, fill, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=90.0, time_stop=15))
    assert (k, reason) == (26, "time"), (k, reason)              # filled at bar 11, out 15 minutes later
    s360 = _sl_session({x: y[:360] for x, y in a.items()}, date[:360], t[:360], v[:360])   # data ends 15:14
    k, fill, px, reason = _sls.fill_intraday(s360, _SlEntry(10, 1, stop=90.0))
    assert (k, reason) == (359, "squareoff")
    two = _sl_spec(lambda s, p: [_SlEntry(10, 1, stop=90.0), _SlEntry(50, 1, stop=90.0)], max_per_day=3)
    trades, _ = _sls.sim_session(two, s)
    assert len(trades) == 1, trades                              # one open position per symbol


run("stratlab: square-off at 15:15, time stop, one open position per symbol", t_sl_squareoff_and_time_stop)


def t_sl_costs_applied():
    date, t, a, v = _sl_flat()
    a["h"][20] = 102.5
    s = _sl_session(a, date, t, v)
    for kind, hz, ck in (("intraday", None, "equity_mis"), ("scalping", None, "equity_mis"),
                         ("futures", "intraday", "future")):
        sp = _sl_spec(lambda s, p: [_SlEntry(10, 1, stop=98.0, target=102.0)], kind=kind, horizon=hz)
        (tr,), _ = _sls.sim_session(sp, s)
        qty = int(100_000 // tr[5])
        assert tr[9] == qty and abs(tr[5] * qty - 100_000) < tr[5], tr       # ₹1L notional, whole shares
        gross = round((102.0 - tr[5]) * qty, 2)
        cost = _sl_rtc(tr[5] * qty, 102.0 * qty, ck)
        assert tr[10] == gross and tr[11] == cost and cost > 0, (kind, tr, cost)
        assert abs(tr[12] - (gross - cost)) < 0.011, tr


run("stratlab: costs from costs.round_trip_cost are charged (MIS for intraday/scalping, future for futures)", t_sl_costs_applied)


def t_sl_costs_applied_short():
    date, t, a, v = _sl_flat()
    a["l"][40] = 98.0
    s = _sl_session(a, date, t, v)
    sh = _sl_spec(lambda s, p: [_SlEntry(10, -1, stop=103.0, target=99.0)])
    (tr,), _ = _sls.sim_session(sh, s)
    qty = tr[9]
    assert tr[13] == "target" and tr[10] == round((tr[5] - 99.0) * qty, 2), tr
    assert tr[11] == _sl_rtc(99.0 * qty, tr[5] * qty, "equity_mis") and tr[12] < tr[10], tr


run("stratlab: costs from costs.round_trip_cost are charged on every trade (short side)", t_sl_costs_applied_short)


def t_sl_positional():
    days = tuple(20260101 + d for d in range(8))
    date, t, a, v = _sl_flat(days=days, n=300)
    def day(k):
        return slice(k * 300, (k + 1) * 300)
    a["o"][day(4)] = a["h"][day(4)] = a["l"][day(4)] = a["c"][day(4)] = 90.0     # day 4 gaps down through the stop
    sd = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    seen = []

    def buy_day1(d, p):
        seen.append(len(d))
        return _SlSwing(1, stop_pct=5.0, target_pct=10.0, max_hold=10) if len(d) == 2 else None
    sw = _sl_spec(buy_day1, kind="swing")
    (tr,), _ = _sls.sim_positional(sw, sd, 0, 99999999)
    assert tr[2] == days[2] and abs(tr[5] - 100.0 * (1 + _sls.SLIPPAGE)) < 1e-6, tr        # decided on day 1, filled next open
    assert tr[13] == "stop" and tr[6] == days[4] and abs(tr[8] - 90.0 * (1 - _sls.SLIPPAGE)) < 1e-6, tr   # gap: fill at the open
    qty = tr[9]
    assert tr[11] == _sl_rtc(tr[5] * qty, tr[8] * qty, "equity_cnc") and tr[11] > 150, tr      # delivery costs
    assert seen[:2] == [1, 2] and max(seen) <= 8
    hold = _sl_spec(lambda d, p: _SlSwing(1, stop_pct=20.0, max_hold=2) if len(d) == 1 else None, kind="swing",
                    id="t_hold")
    (tr,), _ = _sls.sim_positional(hold, sd, 0, 99999999)
    assert tr[13] == "max_hold" and tr[2] == days[1] and tr[6] == days[2], tr
    try:
        _sls.sim_positional(_sl_spec(lambda d, p: _SlSwing(-1, stop_pct=5.0), kind="swing", id="t_short"), sd, 0, 99999999)
        raise AssertionError("swing short was accepted")
    except ValueError as exc:
        assert "long-only" in str(exc)
    fut = _sl_spec(lambda d, p: _SlSwing(-1, stop_pct=5.0, target_pct=8.0, max_hold=10) if len(d) == 2 else None, kind="futures",
                   horizon="positional", id="t_fut")
    (tr,), _ = _sls.sim_positional(fut, sd, 0, 99999999)
    assert tr[3] == -1 and tr[13] == "target" and tr[6] == days[4] and tr[12] > 0 and tr[11] == _sl_rtc(tr[8] * tr[9], tr[5] * tr[9], "future"), tr
    # a decision_time strategy sees today's bar only up to that minute and is filled at that minute
    a["c"][day(6)][-1] = 150.0; a["h"][day(6)][-1] = 150.0                    # a late spike it must not see
    sd2 = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    highs = []

    def at_1100(d, p):
        highs.append(float(d.h[-1]))
        return None
    _sls.sim_positional(_sl_spec(at_1100, kind="swing", decision_time=11 * 60, id="t_dt"), sd2, 0, 99999999)
    assert max(highs) <= 100.0, highs
    assert _sls.sim_positional(sw, sd, days[5], 99999999)[0] == []           # entries only inside the window


run("stratlab: positional sim (next-open fill, gap through stop, max hold, long-only swing, decision time)", t_sl_positional)


def _sl_tr(sid, sym, date, net):
    return (sid, sym, date, 1, "10:00", 100.0, date, "11:00", 101.0, 1000, net + 80.0, 80.0, net, "target")


def _sl_book(sid, cal, per_day=2, syms=("A", "B", "C", "D", "E"), win=300.0, loss=-200.0):
    out, k = [], 0
    for d in cal:
        for j in range(per_day):
            j_, r_ = k % len(syms), k // len(syms)           # every symbol wins 3 rounds in 5
            out.append(_sl_tr(sid, syms[j_], d, win if (r_ + j_) % 5 < 3 else loss))
            k += 1
    return out


def t_sl_protocol_gates_and_final_hidden():
    cal = list(range(20260001, 20260201))                        # 200 sessions: 140 discovery (70 + 70), 60 final
    sp = _slp.split_sessions(cal)
    assert (sp.n_discovery, sp.n_final) == (140, 60) and sp.h1 == (cal[0], cal[69]) and sp.h2 == (cal[70], cal[139])
    assert sp.final == (cal[140], cal[199]) and sp.h2[1] < sp.final[0]
    good = _sl_book("good", cal)
    sc = _slp.screen(good, sp, "intraday")
    assert sc["passed"] and sc["disc"]["trades"] == 280 and sc["disc"]["pf"] >= 1.15, sc
    books = {
        "good": good,
        "h2_loses": [x if x[2] <= sp.h1[1] or x[2] > sp.h2[1] else _sl_tr("h2_loses", x[1], x[2], -abs(x[12])) for x in _sl_book("h2_loses", cal)],
        "low_pf": _sl_book("low_pf", cal, win=140.0, loss=-200.0),
        "few": _sl_book("few", cal[::3], per_day=1),
        "one_symbol": _sl_book("one_symbol", cal, syms=("A", "B")) + [_sl_tr("one_symbol", "Z", cal[5], 30000.0),
                                                                    _sl_tr("one_symbol", "Z", cal[90], 30000.0)],
        "final_loses": [x if x[2] < sp.final[0] else _sl_tr("final_loses", x[1], x[2], -150.0) for x in _sl_book("final_loses", cal)],
    }
    expect = {"h2_loses": "H2 net <= 0", "low_pf": "PF ", "few": "trades <", "one_symbol": "of net"}
    specs = [_sl_spec(lambda s, p: [], id=i) for i in books]
    calls = []

    def simulate(specs_, lo, hi):
        calls.append(([s.id for s in specs_], lo, hi))
        return {s.id: {"trades": [x for x in books[s.id] if lo <= x[2] <= hi], "skipped": 0, "lookahead": None,
                       "error": None} for s in specs_}
    with _sl_tmp.TemporaryDirectory() as td:
        ledger = os.path.join(td, "ledger.json")
        res, trades = _slp.evaluate(specs, cal, simulate, n_symbols=5, ledger_path=ledger)
        by = {r["id"]: r for r in res}
        assert by["good"]["verdict"] == "CONFIRMED" and by["good"]["final"]["trades"] == 120, by["good"]
        assert by["final_loses"]["verdict"] == "REJECTED ON FINAL" and by["final_loses"]["final"]["net"] < 0
        for i, why in expect.items():
            assert by[i]["verdict"] == "FAILED SCREEN" and why in by[i]["why"], (i, by[i]["why"])
            assert by[i]["final"] is None, i                                  # FINAL hidden...
            assert all(x[2] < sp.final[0] for x in trades[i]), i              # ...and its trades never produced
        assert len(calls) == 2 and calls[0][1:] == sp.discovery and calls[1][1:] == sp.final
        assert sorted(calls[1][0]) == ["final_loses", "good"], calls[1]       # FINAL simulated for passers only
        assert all(lo >= sp.final[0] or hi <= sp.h2[1] for _, lo, hi in calls)
        text = _slp.luck_reminder(res)
        assert "Tested 6 strategies" in text and "2 passed the screen" in text and "1 confirmed" in text, text
        assert "5%" in text and "luck" in text
        # the same strategy again: the same single FINAL result; a changed strategy: not a fresh confirmation
        res2, _ = _slp.evaluate(specs[:1], cal, simulate, n_symbols=5, ledger_path=ledger)
        assert res2[0]["verdict"] == "CONFIRMED" and "first reported" in res2[0]["final_note"]
        changed = _sl_spec(lambda s, p: None, id="good")
        res3, _ = _slp.evaluate([changed], cal, simulate, n_symbols=5, ledger_path=ledger)
        assert res3[0]["verdict"].startswith("RETEST") and "not a confirmation" in res3[0]["final_note"], res3[0]
        # --peek-final shows FINAL for failures, labelled, and burns them
        calls.clear()
        res4, _ = _slp.evaluate([s for s in specs if s.id == "low_pf"], cal, simulate, n_symbols=5,
                                ledger_path=ledger, peek_final=True)
        assert res4[0]["verdict"] == "FAILED SCREEN" and res4[0]["final"] is not None
        assert "PEEK" in res4[0]["final_note"] and len(calls) == 2
    # too few sessions: no verdict, FINAL never simulated even for a book that would pass
    calls.clear()
    short = cal[:30]
    res5, _ = _slp.evaluate(specs[:1], short, simulate)
    assert res5[0]["verdict"] == "NO VERDICT" and "too few sessions" in res5[0]["why"] and res5[0]["final"] is None
    assert len(calls) == 1
    assert (_slp.DISCOVERY_FRAC, _slp.SCREEN_MIN_PF, _slp.SCREEN_MAX_SYMBOL_SHARE, _slp.FINAL_MIN_PF) == (0.70, 1.15, 0.35, 1.05)
    assert _slp.SCREEN_MIN_TRADES == {"intraday": 150, "positional": 60}


run("stratlab: protocol gates; FINAL is hidden (never simulated) for screen failures", t_sl_protocol_gates_and_final_hidden)


def t_sl_data_layouts_and_cache():
    def bars(day, n=375, px=100.0, start=555):
        return [[f"{day}T{(start + i) // 60:02d}:{(start + i) % 60:02d}:00", px, px + 1, px - 1, px, 10 + i]
                for i in range(n)]
    with _sl_tmp.TemporaryDirectory() as td:
        src = os.path.join(td, "bars"); os.makedirs(src)
        one = bars("2026-01-05") + bars("2026-01-06", px=101.0) + bars("2026-01-07", n=55)   # + a short session
        one += [["2026-01-06T15:45:00", 1, 1, 1, 1, 1], ["2026-01-06T09:00:00", 1, 1, 1, 1, 1]]  # outside hours
        with open(os.path.join(src, "AAA.json"), "w") as fh:
            _sl_json.dump({"symbol": "AAA", "interval": "minute", "bars": one}, fh)
        with open(os.path.join(src, "_NIFTY50_min.json"), "w") as fh:
            _sl_json.dump({"symbol": "NIFTY 50", "interval": "minute", "bars": bars("2026-01-05", px=25000.0)}, fh)
        with open(os.path.join(src, "_VIX_day.json"), "w") as fh:
            _sl_json.dump({"symbol": "INDIA VIX", "interval": "day", "bars": [["2026-01-05T00:00:00", 14, 15, 13, 14.5, 0]]}, fh)
        with open(os.path.join(src, "b_0.json"), "w") as fh:
            _sl_json.dump({"BBB": {"source": "kite:1m", "bars": bars("2026-01-05", n=360)},
                           "CCC": {"bars": bars("2026-01-05", n=360, px=50.0)}, "EMPTY": {"bars": []}}, fh)
        with open(os.path.join(src, "half.json"), "w") as fh:
            fh.write('{"symbol": "HALF", "bars": [[')                          # a download in progress
        cache = os.path.join(td, "cache")
        man = _sld.prepare(src, cache)
        assert sorted(man["symbols"]) == ["AAA", "BBB", "CCC"], man["symbols"]
        assert sorted(man["aux"]) == ["_NIFTY50_min", "_VIX_day"], man["aux"]
        assert any("half.json" in x for x in man["problems"]) and any("EMPTY" in x for x in man["problems"])
        sd = _sld.load_symbol(cache, "AAA")
        assert sd.n_sessions == 2 and list(sd.days) == [20260105, 20260106] and sd.dropped == [20260107]
        assert sd.t.min() == 555 and sd.t.max() == 929 and len(sd.c) == 750
        d = sd.daily()
        assert list(d["c"]) == [100.0, 101.0] and list(d["h"]) == [101.0, 102.0] and d["v"][0] == sum(10 + i for i in range(375))
        assert _sld.load_symbol(cache, "BBB").t.max() == 914 and _sld.load_daily_aux(cache, "_VIX_day")["c"][0] == 14.5
        idx = _sld.load_symbol(cache, "_NIFTY50_min")
        ss = _sli.build_sessions(sd, idx)
        assert ss[0].nifty is not None and ss[0].nifty[0] == 25000.0 and ss[1].nifty is None
        assert ss[1].pdc == 100.0 and ss[1].pdh == 101.0 and abs(ss[1].gap - 0.01) < 1e-12 and np.isnan(ss[0].pdc)
        sig = dict(_sl_json.load(open(os.path.join(cache, "_manifest.json")))["files"])
        man2 = _sld.prepare(src, cache)                                        # unchanged files are not re-parsed
        assert dict(_sl_json.load(open(os.path.join(cache, "_manifest.json")))["files"]) == sig
        assert sorted(man2["symbols"]) == ["AAA", "BBB", "CCC"]
    for inside in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), "stratlab", "out")):
        try:
            _sld.assert_outside_repo(inside)
            raise AssertionError("a path inside the repository was accepted")
        except ValueError:
            pass


run("stratlab: both bar layouts load, cache, drop short sessions; outputs refused inside the repo", t_sl_data_layouts_and_cache)


def t_sl_indicators_causal():
    sd = _sl_walk(n_days=3, seed=3)
    full = _sli.build_sessions(sd)
    cut = int(sd.starts[2]) + 200                                # truncate the data mid-way through day 3
    sd_cut = _sld.from_arrays("RW", sd.date[:cut], sd.t[:cut], sd.o[:cut], sd.h[:cut], sd.l[:cut], sd.c[:cut], sd.v[:cut])
    part = _sli.build_sessions(sd_cut)
    for name in ("vwap", "vsd", "ema9", "ema21", "ema50", "rsi", "atr", "e9_5", "e21_5", "e50_5", "rsi5", "atr5",
                 "vr", "day_hi", "day_lo"):
        a, b = getattr(full[2], name)[:195], getattr(part[2], name)[:195]
        assert np.allclose(a, b, equal_nan=True), name           # values do not depend on later bars
    s = full[1]
    tp = (s.h + s.l + s.c) / 3
    assert abs(s.vwap[99] - (tp[:100] * s.v[:100]).sum() / s.v[:100].sum()) < 1e-9
    assert s.day_hi[50] == s.h[:51].max() and s.pdc == full[0].c[-1] and len(s.prev) == 1


run("stratlab: indicators are causal (unchanged when later bars are removed)", t_sl_indicators_causal)


def t_sl_catalogue_first_batch():
    kinds = {}
    for sp in _SL_REG.values():
        kinds.setdefault(sp.kind, []).append(sp.id)
        # <= 5 parameters is the first batch's convention; round-3..7 research strategies list theirs in their
        # pre-registration files (scratchpad r3/r4 PREREGISTERED.txt, stratlab/prereg/*.json) and were judged as written
        assert sp.desc and "\n" not in sp.desc and sp.family and (len(sp.params) <= 5 or sp.id.startswith(("r3_", "r4_", "r5_", "r6_", "r7_"))), sp.id
    assert len(kinds["intraday"]) >= 12 and len(kinds["scalping"]) >= 8, {k: len(v) for k, v in kinds.items()}
    assert len(set(_SL_REG)) == len(_SL_REG)
    sd = _sl_walk(n_days=30, seed=11)
    rng = np.random.default_rng(1)
    idx = _sld.from_arrays("_NIFTY50_min", sd.date, sd.t, sd.c * 50, sd.c * 50, sd.c * 50,
                           25000 * np.exp(np.cumsum(rng.normal(0, 0.0004, len(sd.c)))), sd.v * 0)
    specs = [sp for sp in _SL_REG.values() if sp.horizon == "intraday"]
    sessions = _sli.build_sessions(sd, idx)
    rare = {sp.id for sp in specs if sp.family == "candlestick"}   # textbook patterns may not occur in 30 days
    # round-3..7 research rules need real NSE data, the index or rare conditions: audited below, not expected to trade here
    rare |= {sp.id for sp in specs if sp.id.startswith(("r3_", "r4_", "r5_", "r6_", "r7_"))}
    traded = 0
    for sp in specs:
        n = 0
        for s in sessions:
            tr, _ = _sls.sim_session(sp, s)
            n += len(tr)
            assert len(tr) <= sp.max_per_day, sp.id
            for x in tr:
                assert x[4] < "15:15" and x[7] <= "15:15" and x[4] <= x[7], (sp.id, x)
            bad = _sls.audit_lookahead(sp, s, cuts=(20, 46, 75, 110, 170, 226, 290, 330))
            assert bad is None, f"{sp.id}: {bad}"
        traded += n > 0 and sp.id not in rare
    assert traded >= len(specs) - len(rare) - 3, traded                      # nearly every strategy trades on a random walk
    out = _sls.run_symbol("", "RW", list(_SL_REG.values()), 0, 99999999, sd=sd, index=idx)
    assert all(r["error"] is None and r["lookahead"] is None for r in out.values()), \
        {k: (r["error"], r["lookahead"]) for k, r in out.items() if r["error"] or r["lookahead"]}


run("stratlab: catalogue has 12+ intraday and 8+ scalping strategies; all run and pass the look-ahead audit", t_sl_catalogue_first_batch)


def t_sl_cli_end_to_end():
    from stratlab import run as _slr
    import contextlib, io

    def bars(sd):
        return [[f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}T{m // 60:02d}:{m % 60:02d}:00", float(o), float(h), float(l), float(c), float(v)]
                for d, m, o, h, l, c, v in zip(sd.date, sd.t, sd.o, sd.h, sd.l, sd.c, sd.v)]
    with _sl_tmp.TemporaryDirectory() as td:
        src, out = os.path.join(td, "bars"), os.path.join(td, "out"); os.makedirs(src)
        for k, sym in enumerate(("AAA", "BBB")):
            with open(os.path.join(src, sym + ".json"), "w") as fh:
                _sl_json.dump({"symbol": sym, "interval": "minute", "bars": bars(_sl_walk(n_days=12, seed=20 + k, symbol=sym))}, fh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _slr.main(["--data", src, "--kind", "scalping", "--ids", "scalp_orb5,scalp_ema9_21_cross",
                            "--out", out, "--cache", os.path.join(td, "cache"), "--jobs", "1"])
        assert rc == 0
        text = open(os.path.join(out, "summary.txt")).read()
        assert "too few sessions" in text and "NO VERDICT" in text and "REMINDER" in text and "hidden" in text, text[:400]
        rows = list(__import__("csv").DictReader(open(os.path.join(out, "results.csv"))))
        assert [r["id"] for r in rows] == ["scalp_ema9_21_cross", "scalp_orb5"] and rows[0]["final_trades"] == ""
        tr = list(__import__("csv").DictReader(open(os.path.join(out, "trades", "scalp_ema9_21_cross.csv"))))
        final_from = _slp.split_sessions(range(20260101, 20260113)).final[0]
        assert tr and all(int(x["date"]) < final_from for x in tr)             # no FINAL trades on disk
        try:
            _slr.main(["--data", src, "--kind", "scalping", "--out", os.path.join(os.path.dirname(os.path.abspath(__file__)), "x")])
            raise AssertionError("--out inside the repository was accepted")
        except ValueError:
            pass


run("stratlab: CLI end to end (results.csv, summary.txt, trade CSVs without FINAL trades)", t_sl_cli_end_to_end)


# ══════════════════════════════════════════════════════════════════════════
# STRATLAB — candlestick patterns (stratlab/candles.py, catalog/candlestick.py)
# ══════════════════════════════════════════════════════════════════════════
section("STRATLAB: candlestick patterns")

from stratlab import candles as _cdl
from stratlab.catalog import candlestick as _cdlcat

# 12 ordinary candles (alternating white / black, body 1, range 2) so avgB = 1 and avgR = 2 before a pattern
_CS_BASE = [(100.0, 101.5, 99.5, 101.0) if k % 2 == 0 else (101.0, 101.5, 99.5, 100.0) for k in range(12)]


def _cs(*rows):
    """OHLC arrays: the 12 ordinary candles followed by `rows` (each (o, h, l, c))."""
    a = np.array(_CS_BASE + list(rows), float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def _cs_hit(fn, *rows) -> bool:
    return bool(fn(*_cs(*rows))[-1])


def t_cs_single_candle_detectors():
    D = _cdl
    hammer = (100.5, 101.0, 98.5, 101.0)            # B .5, lower shadow 2.0, no upper shadow
    assert _cs_hit(D.hammer, hammer) and _cs_hit(D.hanging_man, hammer)
    assert not _cs_hit(D.hammer, (100.5, 101.6, 98.5, 101.0))      # upper shadow .6 > 10 % of range
    assert not _cs_hit(D.hammer, (100.5, 101.0, 99.6, 101.0))      # lower shadow .9 < 2 x body
    assert not _cs_hit(D.hammer, (100.95, 101.0, 98.5, 101.0))     # body .05: a doji, not a hammer
    assert not _cs_hit(D.hammer, (100.0, 100.1, 99.85, 100.1))     # range .25 < .5 x average range
    inv = (99.5, 102.0, 99.5, 100.0)                # B .5, upper shadow 2.0, no lower shadow
    assert _cs_hit(D.inverted_hammer, inv) and _cs_hit(D.shooting_star, inv)
    assert not _cs_hit(D.inverted_hammer, (99.5, 102.0, 99.0, 100.0)) and not _cs_hit(D.shooting_star, hammer)
    assert _cs_hit(D.bullish_marubozu, (99.0, 101.55, 99.0, 101.5))
    assert not _cs_hit(D.bullish_marubozu, (99.0, 102.0, 99.0, 101.5))            # upper shadow .5
    assert not _cs_hit(D.bullish_marubozu, (100.0, 100.8, 100.0, 100.8))           # body .8 not long
    assert _cs_hit(D.bearish_marubozu, (101.5, 101.5, 98.95, 99.0))
    assert not _cs_hit(D.bearish_marubozu, (99.0, 101.55, 99.0, 101.5))
    assert _cs_hit(D.doji, (100.0, 101.0, 99.0, 100.05)) and not _cs_hit(D.doji, (100.0, 101.0, 99.0, 100.5))
    assert _cs_hit(D.dragonfly_doji, (101.0, 101.0, 99.0, 101.0))
    assert not _cs_hit(D.dragonfly_doji, (101.0, 101.5, 99.0, 101.0))              # upper shadow .5 of 2.5
    assert _cs_hit(D.gravestone_doji, (99.0, 101.0, 99.0, 99.0))
    assert not _cs_hit(D.gravestone_doji, (101.0, 101.0, 99.0, 101.0))
    assert _cs_hit(D.long_legged_doji, (100.0, 101.0, 99.0, 100.0))
    assert not _cs_hit(D.long_legged_doji, (101.0, 101.0, 99.0, 101.0))            # dragonfly, not long-legged
    assert _cs_hit(D.spinning_top, (100.0, 101.5, 99.0, 100.5))
    assert not _cs_hit(D.spinning_top, hammer)
    assert "doji" in _cdl.CONTEXT_ONLY and "spinning_top" in _cdl.CONTEXT_ONLY
    assert not any(k in _cdl.PATTERNS for k in ("doji", "spinning_top"))


run("candles: single-candle detectors on textbook examples (positive and negative)", t_cs_single_candle_detectors)


def t_cs_two_candle_detectors():
    D = _cdl
    black, white = (101.0, 101.5, 99.5, 100.0), (100.0, 101.5, 99.5, 101.0)
    assert _cs_hit(D.bullish_engulfing, black, (99.9, 101.3, 99.8, 101.2))
    assert not _cs_hit(D.bullish_engulfing, black, (100.2, 101.0, 100.1, 100.8))      # inside, not engulfing
    assert not _cs_hit(D.bullish_engulfing, white, (99.9, 101.3, 99.8, 101.2))        # first candle not black
    assert _cs_hit(D.bearish_engulfing, white, (101.1, 101.2, 99.7, 99.8))
    assert not _cs_hit(D.bearish_engulfing, white, (100.8, 100.9, 100.1, 100.2))
    long_black, long_white = (102.0, 102.1, 99.9, 100.0), (100.0, 102.1, 99.9, 102.0)
    assert _cs_hit(D.piercing_line, long_black, (99.8, 101.4, 99.7, 101.3))
    assert not _cs_hit(D.piercing_line, long_black, (100.1, 101.4, 100.0, 101.3))     # did not open below the low
    assert not _cs_hit(D.piercing_line, long_black, (99.8, 100.9, 99.7, 100.8))       # closed below the midpoint
    assert not _cs_hit(D.piercing_line, long_black, (99.8, 102.3, 99.7, 102.2))       # above the open: engulfing
    assert _cs_hit(D.dark_cloud_cover, long_white, (102.2, 102.3, 100.6, 100.7))
    assert not _cs_hit(D.dark_cloud_cover, long_white, (101.9, 102.0, 100.6, 100.7))  # did not open above the high
    assert not _cs_hit(D.dark_cloud_cover, long_white, (102.2, 102.3, 99.7, 99.8))
    assert _cs_hit(D.bullish_harami, (102.0, 102.1, 98.9, 99.0), (100.0, 100.6, 99.9, 100.5))
    assert not _cs_hit(D.bullish_harami, (102.0, 102.1, 98.9, 99.0), (98.5, 99.6, 98.4, 99.5))   # body outside
    assert not _cs_hit(D.bullish_harami, (100.0, 101.5, 99.5, 100.5), (100.2, 100.4, 100.1, 100.3))  # 1st not black
    assert _cs_hit(D.bearish_harami, (99.0, 102.1, 98.9, 102.0), (101.0, 101.1, 100.4, 100.5))
    assert not _cs_hit(D.bearish_harami, (99.0, 102.1, 98.9, 102.0), (101.0, 102.6, 100.4, 102.5))
    assert _cs_hit(D.tweezer_bottom, black, (100.0, 101.5, 99.5, 101.0))
    assert not _cs_hit(D.tweezer_bottom, black, (100.0, 101.5, 99.2, 101.0))          # lows .3 apart
    assert _cs_hit(D.tweezer_top, white, (101.0, 101.5, 99.5, 100.0))
    assert not _cs_hit(D.tweezer_top, white, (101.0, 101.9, 99.5, 100.0))
    assert _cs_hit(D.inside_bar, (100.0, 102.0, 98.0, 101.0), (100.5, 101.0, 99.0, 100.0))
    assert not _cs_hit(D.inside_bar, (100.0, 102.0, 98.0, 101.0), (100.5, 102.0, 99.0, 100.0))  # equal high
    assert _cs_hit(D.outside_bar, (100.0, 101.0, 99.0, 100.5), (100.5, 101.5, 98.5, 99.0))
    assert not _cs_hit(D.outside_bar, (100.0, 101.0, 99.0, 100.5), (100.5, 101.5, 99.2, 99.5))


run("candles: two-candle detectors on textbook examples (positive and negative)", t_cs_two_candle_detectors)


def t_cs_three_candle_detectors():
    D = _cdl
    ms = [(102.0, 102.1, 99.4, 99.5), (99.2, 99.3, 98.8, 99.0)]
    assert _cs_hit(D.morning_star, *ms, (99.3, 101.3, 99.2, 101.2))
    assert not _cs_hit(D.morning_star, *ms, (99.3, 100.6, 99.2, 100.5))              # below candle 1's midpoint
    assert not _cs_hit(D.morning_star, ms[0], (99.6, 99.8, 99.3, 99.7), (99.3, 101.3, 99.2, 101.2))  # no gap
    es = [(99.5, 102.1, 99.4, 102.0), (102.3, 102.7, 102.2, 102.5)]
    assert _cs_hit(D.evening_star, *es, (102.2, 102.3, 100.4, 100.5))
    assert not _cs_hit(D.evening_star, *es, (102.2, 102.3, 100.9, 101.0))
    sold = [(100.0, 101.6, 99.9, 101.5), (101.0, 102.6, 100.9, 102.5), (102.0, 103.6, 101.9, 103.5)]
    assert _cs_hit(D.three_white_soldiers, *sold)
    assert not _cs_hit(D.three_white_soldiers, *sold[:2], (103.0, 104.6, 102.9, 104.5))   # opened above body 2
    assert not _cs_hit(D.three_white_soldiers, *sold[:2], (102.0, 104.5, 101.9, 103.5))   # far from its high
    crows = [(101.0, 101.1, 99.4, 99.5), (100.0, 100.1, 98.4, 98.5), (99.0, 99.1, 97.4, 97.5)]
    assert _cs_hit(D.three_black_crows, *crows)
    assert not _cs_hit(D.three_black_crows, *crows[:2], (98.0, 98.1, 96.4, 96.5))         # opened below body 2
    har_up = [(102.0, 102.1, 98.9, 99.0), (100.0, 100.6, 99.9, 100.5)]
    assert _cs_hit(D.three_inside_up, *har_up, (100.5, 102.6, 100.4, 102.5))
    assert not _cs_hit(D.three_inside_up, *har_up, (100.5, 101.6, 100.4, 101.5))          # not above candle 1's open
    har_dn = [(99.0, 102.1, 98.9, 102.0), (101.0, 101.1, 100.4, 100.5)]
    assert _cs_hit(D.three_inside_down, *har_dn, (100.5, 100.6, 98.4, 98.5))
    assert not _cs_hit(D.three_inside_down, *har_dn, (100.5, 100.6, 99.4, 99.5))
    eng_up = [(101.0, 101.5, 99.5, 100.0), (99.9, 101.3, 99.8, 101.2)]
    assert _cs_hit(D.three_outside_up, *eng_up, (101.2, 101.9, 101.1, 101.8))
    assert not _cs_hit(D.three_outside_up, *eng_up, (101.2, 101.3, 100.9, 101.0))         # lower close
    eng_dn = [(100.0, 101.5, 99.5, 101.0), (101.1, 101.2, 99.7, 99.8)]
    assert _cs_hit(D.three_outside_down, *eng_dn, (99.8, 99.9, 99.2, 99.3))
    assert not _cs_hit(D.three_outside_down, *eng_dn, (99.8, 100.2, 99.7, 100.1))


run("candles: three-candle detectors on textbook examples (positive and negative)", t_cs_three_candle_detectors)


def _cs_trend(direction: int, n: int = 30):
    """n candles trending by 0.5 per candle (body .4, range .8)."""
    rows = []
    for k in range(n):
        m = 100.0 + direction * 0.5 * k
        rows.append((m - direction * 0.2, m + 0.4, m - 0.4, m + direction * 0.2))
    return rows


def t_cs_trend_context_once_for_all():
    down, up = _cs_trend(-1), _cs_trend(1)
    last_dn, last_up = down[-1][3], up[-1][3]
    ham_dn = (last_dn - 0.3, last_dn + 0.2, last_dn - 2.0, last_dn + 0.2)       # hammer after the decline
    ham_up = (last_up - 0.3, last_up + 0.2, last_up - 2.0, last_up + 0.2)       # same shape after the rise
    for rows, want_long, want_short in ((down + [ham_dn], True, False), (up + [ham_up], False, True)):
        o, h, l, c = (np.array(x) for x in zip(*rows))
        sig = _cdl.signals(o, h, l, c, ["hammer", "hanging_man", "inside_bar", "outside_bar"])
        assert bool(sig["hammer"][0][-1]) is want_long, ("hammer", want_long)
        assert bool(sig["hanging_man"][1][-1]) is want_short, ("hanging man", want_short)
        assert not sig["hammer"][1].any() and not sig["hanging_man"][0].any()
    o, h, l, c = (np.array(x) for x in zip(*_cs_trend(-1)))
    dn, upm = _cdl.trend_context(c)
    assert dn[-1] and not upm[-1] and not dn[:_cdl.EMA_N].any()
    assert not _cdl.trend_context(c[:_cdl.EMA_N])[0].any()                       # too short for EMA20
    # the context is the candle BEFORE the pattern's first candle
    assert _cdl.PATTERNS["morning_star"].n == 3 and _cdl.PATTERNS["hammer"].n == 1
    # every traded pattern has a direction rule; reversals need the opposite trend
    for name, p in _cdl.PATTERNS.items():
        if p.context in ("down", "up"):
            assert (p.side > 0) == (p.context == "down"), name


run("candles: one prior-trend rule (EMA20 + 5-candle close) gates every reversal pattern", t_cs_trend_context_once_for_all)


def t_cs_resample_alignment():
    n = 375
    t = 555 + np.arange(n)
    o = 100.0 + np.arange(n) * 0.01
    c = o + 0.005
    h, l, v = c + 0.02, o - 0.02, np.ones(n)
    for tf, count, starts in ((5, 75, [555, 560, 565]), (15, 25, [555, 570, 585]), (60, 7, [555, 615, 675])):
        cd = _cdl.resample(t, o, h, l, c, v, tf)
        assert len(cd.o) == count and list(cd.start[:3]) == starts, (tf, len(cd.o), cd.start[:3])
        assert (cd.end - cd.start <= tf).all() and cd.sig[0] == tf - 1 and (cd.sig == cd.last).all()
    cd = _cdl.resample(t, o, h, l, c, v, 60)
    assert list(cd.start) == [555, 615, 675, 735, 795, 855, 915] and list(cd.end)[-2:] == [915, 930]
    assert cd.last[-1] - cd.first[-1] + 1 == 15                                  # short last candle 15:15-15:29
    cd = _cdl.resample(t, o, h, l, c, v, 15)
    k = 1                                                                         # 09:30-09:45 = bars 15..29
    assert cd.o[k] == o[15] and cd.c[k] == c[29] and cd.h[k] == h[15:30].max() and cd.l[k] == l[15:30].min()
    assert cd.v[k] == 15 and _cdl.SESSION_END == 930
    # a candle is complete only when its last minute (end-1) is known
    part = _cdl.resample(t[:29], o[:29], h[:29], l[:29], c[:29], v[:29], 15)   # bars up to 09:43
    assert len(part.o) == 1 and part.end[-1] == 570
    full = _cdl.resample(t[:30], o[:30], h[:30], l[:30], c[:30], v[:30], 15)   # bar 09:44 known
    assert len(full.o) == 2 and full.sig[-1] == 29 and t[29] == 584
    raw = _cdl.resample(t[:29], o[:29], h[:29], l[:29], c[:29], v[:29], 15, complete_only=False)
    assert len(raw.o) == 2 and raw.sig[-1] == -1
    # the 09:44 bar is missing: the candle is known complete only at the close of the 09:45 bar
    keep = t != 584
    gap = _cdl.resample(t[keep], o[keep], h[keep], l[keep], c[keep], v[keep], 15)
    assert t[keep][gap.sig[1]] == 585 and gap.last[1] == 28
    g29 = _cdl.resample(t[keep][:29], o[keep][:29], h[keep][:29], l[keep][:29], c[keep][:29], v[keep][:29], 15)
    assert len(g29.o) == 1                                                       # 09:43 known, 09:44 missing: not yet


run("candles: 5m / 15m / 60m candles aligned at 09:15, short 15:15 candle, complete only at minute end-1", t_cs_resample_alignment)


def _cs_hammer_day(shift_after: float = 1.0, drop_944: bool = False):
    """Day 1: a steady decline (120 -> 100). Day 2: the decline continues to 09:29, the 09:30-09:44 candle is a
    hammer (open 99.5, low 98.0, close 99.9 at 09:44), then flat. `shift_after` rescales every bar from 09:45."""
    n = 375
    p1 = np.linspace(120.0, 100.0, n + 1)
    o1, c1 = p1[:-1], p1[1:]
    p2 = np.concatenate((np.linspace(100.0, 99.5, 16), np.full(n - 15, 99.9)))
    o2, c2 = p2[:-1].copy(), p2[1:].copy()
    o2[15:30] = 99.5; c2[15:30] = 99.5
    o2[20], c2[20] = 99.5, 98.2                                              # the dip at 09:35
    o2[21], c2[21] = 98.2, 99.5
    o2[29], c2[28], c2[29] = 99.9, 99.9, 99.9                                # the candle closes at 99.9
    o = np.concatenate((o1, o2)); c = np.concatenate((c1, c2))
    h = np.maximum(o, c) + 0.01; l = np.minimum(o, c) - 0.01
    a2 = n
    l[a2 + 20] = 98.0; h[a2 + 29] = 99.95
    for k in range(a2 + 15, a2 + 30):                                         # keep the candle's high at 99.95
        h[k] = min(h[k], 99.95)
    date = np.repeat(np.array([20260105, 20260106], np.int32), n)
    t = np.tile(555 + np.arange(n), 2).astype(np.int16)
    after = (date == 20260106) & (t >= 585)
    for arr in (o, h, l, c):
        arr[after] *= shift_after
    keep = np.ones(len(t), bool)
    if drop_944:
        keep &= ~((date == 20260106) & (t == 584))
    return _sld.from_arrays("HAM", date[keep], t[keep], o[keep], h[keep], l[keep], c[keep], np.full(keep.sum(), 1000.0))


def t_cs_pattern_complete_before_signal():
    sp = _SL_REG["cs_hammer_15m"]
    s = _sli.build_sessions(_cs_hammer_day())[1]
    cs = _cdl.session_candles(s, 15)
    assert cs.start[1] == 570 and bool(cs.sigs["hammer"][0][1]), "the 09:30-09:45 candle should be a hammer"
    ents = sp.fn(s, sp.p)
    assert len(ents) == 1 and ents[0].side == 1 and int(s.t[ents[0].i]) == 584, [(e.i, e.side) for e in ents]
    e = ents[0]
    assert e.stop < 98.0 and abs(e.stop - min(98.0 - 0.05, 99.9 - 0.3 * cs.atr[1])) < 1e-9 and e.rr == 2.0
    trades, _ = _sls.sim_session(sp, s)
    assert trades and trades[0][4] == "09:45" and abs(trades[0][5] - float(s.o[ents[0].i + 1]) * (1 + _sls.SLIPPAGE)) < 1e-3
    # at the close of 09:43 the candle is not finished: no signal yet; at 09:44 it is
    assert sp.fn(s.view(int(np.searchsorted(s.t, 584))), sp.p) == []
    assert len(sp.fn(s.view(int(np.searchsorted(s.t, 584)) + 1), sp.p)) == 1
    # changing every bar after the pattern candle changes neither the signal nor its fill time
    s2 = _sli.build_sessions(_cs_hammer_day(shift_after=1.02))[1]
    e2 = sp.fn(s2, sp.p)
    assert [(x.i, x.side, round(x.stop, 9), x.rr) for x in e2] == [(e.i, e.side, round(e.stop, 9), e.rr)]
    tr2, _ = _sls.sim_session(sp, s2)
    assert tr2 and tr2[0][4] == "09:45"
    assert _sls.audit_lookahead(sp, s, cuts=(20, 29, 30, 31, 60, 200)) is None
    # with the 09:44 bar missing the candle is known complete at the close of 09:45: fill at 09:46
    s3 = _sli.build_sessions(_cs_hammer_day(drop_944=True))[1]
    e3 = sp.fn(s3, sp.p)
    assert len(e3) == 1 and int(s3.t[e3[0].i]) == 585
    tr3, _ = _sls.sim_session(sp, s3)
    assert tr3 and tr3[0][4] == "09:46"
    # confirmation variant: the next candle (09:45-10:00) closes above 99.9 only when the rest of the day is higher
    spc = _SL_REG["cs_hammer_15m_conf"]
    assert spc.fn(s, spc.p) == []                                            # flat at 99.9: not confirmed
    ec = spc.fn(s2, spc.p)
    cs2 = _cdl.session_candles(s2, 15)
    assert len(ec) == 1 and int(s2.t[ec[0].i]) == 599                         # after the 09:45-10:00 candle
    assert abs(ec[0].stop - min(98.0 - 0.05, float(cs2.c[2]) - 0.3 * float(cs2.atr[2]))) < 1e-9


run("candles: pattern candle must be complete (15m candle to 09:45 fills at 09:45 open); later bars change nothing", t_cs_pattern_complete_before_signal)


def t_cs_catalogue_and_windows():
    ids = [i for i, sp in _SL_REG.items() if sp.family == "candlestick"]
    assert len(ids) == _cdlcat.N_REGISTERED == 152 == len(_cdlcat.IDS) == len(set(_cdlcat.IDS))
    n_conf = sum(i.endswith("_conf") for i in ids)
    assert n_conf == 4 * len(_cdlcat.CONFIRM) == 44
    for pat in _cdl.PATTERNS:
        for tf in ("5m", "15m", "60m", "d"):
            assert f"cs_{pat}_{tf}" in _SL_REG, (pat, tf)
    for i in ids:
        sp = _SL_REG[i]
        if i.split("_")[-1] in ("d",) or i.endswith("_d_conf"):
            side = _cdl.PATTERNS[sp.params["pattern"]].side
            assert sp.kind == ("swing" if side > 0 else "futures") and sp.horizon == "positional", i
        else:
            assert sp.kind == "intraday" and sp.max_per_day == 1, i
    # windows: 5m / 15m decisions on candles ending 09:30-14:30; 60m by 14:15; stops on the losing side
    sd = _sl_walk(n_days=8, seed=5)
    sessions = _sli.build_sessions(sd)
    seen = {5: 0, 15: 0, 60: 0}
    for s in sessions[1:]:
        for pat in _cdl.PATTERNS:
            for tf in (5, 15, 60):
                for conf in (False, True):
                    for e in _cdlcat.candle_entries(s, pat, tf, conf):
                        seen[tf] += 1
                        tt = int(s.t[e.i])
                        assert (569 <= tt <= 869) if tf != 60 else (614 <= tt <= 854), (pat, tf, tt)
                        assert e.side * (float(s.c[e.i]) - e.stop) > 0, (pat, tf)
    assert seen[5] > 0 and seen[15] > 0, seen


run("candles: 152 strategies cs_<pattern>_<tf>[_conf], right kinds, entry windows and stop sides", t_cs_catalogue_and_windows)


def t_cs_daily_rr_and_no_lookahead():
    # Swing(rr=2): target = fill + 2 x (fill - stop), as for intraday entries
    days = [20260105 + k for k in range(5)]
    date, t, a, v = _sl_flat(days=days)
    m = date == days[3]
    a["h"][m] = 120.0
    sd = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    calls = []

    def fn(d, p):
        calls.append(len(d))
        return _SlSwing(1, stop=95.0, rr=2.0, max_hold=3) if len(d) == 2 else None
    tr, _ = _sls.sim_positional(_sl_spec(fn, kind="swing"), sd, 0, 99999999)
    assert tr and tr[0][-1] == "target", tr
    fill = 100.0 * (1 + _sls.SLIPPAGE)
    assert abs(tr[0][8] - (fill + 2 * (fill - 95.0))) < 1e-9 and tr[0][2] == days[2]
    # daily candlestick signal: a hammer on the decision day, unchanged by anything after it
    o, h, l, c = (np.array(x) for x in zip(*_cs_trend(-1, 40)))
    last = c[-1]
    ham = (last - 0.3, last + 0.2, last - 2.0, last + 0.2)

    cols = [np.append(x, y) for x, y in zip((o, h, l, c), ham)]
    d = _sls.DailyView(np.arange(len(cols[0])), *cols, np.ones(len(cols[0])))
    d.symbol, d.nifty, d.vix = "X", None, None
    sw = _cdlcat.daily_signal(d, "hammer", False, (1,))
    assert sw is not None and sw.side == 1 and sw.rr == 2.0 and sw.max_hold == 10 and sw.stop <= ham[2] - 0.05
    assert _cdlcat.daily_signal(d, "hanging_man", False, (-1,)) is None                  # wrong trend
    assert _cdlcat.daily_signal(d, "hammer", True, (1,)) is None                         # no confirmation day yet
    sp = _SL_REG["cs_hammer_d"]
    assert sp.kind == "swing" and isinstance(sp.fn(d, sp.p), _SlSwing)


run("candles: daily patterns: Swing rr=2 target from the fill, hammer signal on the decision day only", t_cs_daily_rr_and_no_lookahead)


def _cs_perturb(sd, mask, seed):
    """A copy of sd whose bars under `mask` are randomly moved (OHLC kept consistent, volume tripled)."""
    r = np.random.default_rng(seed)
    o, h, l, c = (np.asarray(x, float).copy() for x in (sd.o, sd.h, sd.l, sd.c))
    k = int(mask.sum())
    f = np.exp(r.normal(0, 0.004, k))
    o[mask] *= f
    c[mask] *= f * np.exp(r.normal(0, 0.004, k))
    h[mask] = np.maximum(np.maximum(o[mask], c[mask]), h[mask] * f * (1 + np.abs(r.normal(0, 0.002, k))))
    l[mask] = np.minimum(np.minimum(o[mask], c[mask]), l[mask] * f * (1 - np.abs(r.normal(0, 0.002, k))))
    return _sld.from_arrays(sd.symbol, sd.date, sd.t, o, h, l, c, np.asarray(sd.v, float) * np.where(mask, 3.0, 1.0))


_CS_PERTURB_PATTERNS = ("outside_bar", "inside_bar", "bull_engulfing", "bear_engulfing", "tweezer_bottom",
                        "tweezer_top", "bull_marubozu", "bear_marubozu")


def t_cs_future_bars_change_nothing_intraday():
    """Perturb every bar from a cut minute onward (and every later session): signals decided before the cut and
    fills before the cut must not change, on 5m, 15m and 60m; every fill is at or after its candle's end."""
    sd = _sl_walk(n_days=24, seed=11, symbol="PERT")
    sessions = _sli.build_sessions(sd)
    compared = {}
    for k in (6, 11, 17, 22):
        s = sessions[k]
        for T in (600, 690, 780, 870):                       # 10:00, 11:30, 13:00, 14:30
            sd2 = _cs_perturb(sd, (sd.date > s.date) | ((sd.date == s.date) & (sd.t >= T)), k * 1000 + T)
            s2 = _sli.build_sessions(sd2)[k]
            cut = int(np.searchsorted(s.t, T))               # bars 0..cut-1 are identical in s and s2
            assert np.array_equal(s.c[:cut], s2.c[:cut]) and not np.array_equal(s.c[cut:], s2.c[cut:])
            for tf in ("5m", "15m", "60m"):
                for pat in _CS_PERTURB_PATTERNS:
                    sp = _SL_REG[f"cs_{pat}_{tf}"]
                    canon = lambda ents: sorted((int(e.i), int(e.side), round(float(e.stop), 9), e.rr)
                                                for e in ents if e.i <= cut - 1)
                    a, b = canon(sp.fn(s, sp.p)), canon(sp.fn(s2, sp.p))
                    assert a == b, (sp.id, s.date, T, a, b)
                    compared[sp.id] = compared.get(sp.id, 0) + len(a)
                    cs = _cdl.session_candles(s, sp.p.tf)
                    for e in sp.fn(s, sp.p):                 # the fill is never before the candle has closed
                        d = int(np.flatnonzero(cs.sig == e.i)[0])
                        assert e.i + 1 >= len(s.t) or int(s.t[e.i + 1]) >= int(cs.end[d]), (sp.id, e.i)
                    ta, tb = _sls.sim_session(sp, s)[0], _sls.sim_session(sp, s2)[0]
                    before = lambda tr: [x[3:6] for x in tr if int(x[4][:2]) * 60 + int(x[4][3:]) < T]
                    assert before(ta) == before(tb), (sp.id, s.date, T, before(ta), before(tb))
    for tf in ("5m", "15m", "60m"):
        hit = [p for p in _CS_PERTURB_PATTERNS if compared.get(f"cs_{p}_{tf}", 0) > 0]
        assert len(hit) >= 3, (tf, compared)                  # at least 3 strategies per timeframe really compared
    # signal-anchored: for EVERY intraday candlestick strategy, move every bar after each signal bar i; the signal
    # (bar, side, stop) must survive. Moving every bar after the fill bar i+1 must leave the fill (time, price)
    # unchanged.
    perturbed = {}

    def moved(k, s, first_bar):
        if (k, first_bar) not in perturbed:
            T = int(s.t[first_bar]) if first_bar < len(s.t) else 930
            sd2 = _cs_perturb(sd, (sd.date > s.date) | ((sd.date == s.date) & (sd.t >= T)), k * 7919 + first_bar)
            perturbed[(k, first_bar)] = _sli.build_sessions(sd2)[k]
        return perturbed[(k, first_bar)]
    survived = {"5m": set(), "15m": set(), "60m": set()}
    for k in (6, 13, 20):
        s = sessions[k]
        for sp in (x for x in _SL_REG.values() if x.family == "candlestick" and x.horizon == "intraday"):
            for e in sorted(sp.fn(s, sp.p), key=lambda x: x.i)[:2]:
                s2 = moved(k, s, e.i + 1)
                same = lambda ents: sorted((int(x.i), int(x.side), round(float(x.stop), 9), x.rr)
                                           for x in ents if x.i <= e.i)
                assert same(sp.fn(s, sp.p)) == same(sp.fn(s2, sp.p)), (sp.id, s.date, e.i)
                ta = _sls.sim_session(sp, s)[0]
                if ta and ta[0][4] == _sls._hm(s.t[e.i + 1]):
                    tb = _sls.sim_session(sp, moved(k, s, e.i + 2))[0]
                    assert tb and tb[0][3:6] == ta[0][3:6], (sp.id, s.date, ta[:1], tb[:1])
                survived[f"{sp.p.tf}m"].add(sp.id)
    assert all(len(v) >= 3 for v in survived.values()), {k: len(v) for k, v in survived.items()}


run("candles: perturbing future bars changes no earlier 5m/15m/60m signal or fill; fills at/after candle end",
    t_cs_future_bars_change_nothing_intraday)


def t_cs_future_days_change_nothing_daily():
    """Daily patterns: perturb every day after D; decisions up to D and trades entered up to D are unchanged.
    Entries are filled at the next session's open; exits are never before the entry day."""
    sd = _sl_walk(n_days=90, seed=12, symbol="PERTD")
    dd = sd.daily()
    ids = [f"cs_{p}_d" for p in _CS_PERTURB_PATTERNS]
    seen = {i: 0 for i in ids}

    def view(dct, upto):
        v = _sls.DailyView(dct["date"][:upto + 1], *(dct[x][:upto + 1] for x in "ohlcv"))
        v.symbol, v.nifty, v.vix = "PERTD", None, None
        return v
    for j in (40, 55, 70, 82):
        D = int(sd.days[j])
        sd2 = _cs_perturb(sd, sd.date > D, j)
        dd2 = sd2.daily()
        for i in ids:
            sp = _SL_REG[i]
            for day in range(25, j + 1):
                a, b = sp.fn(view(dd, day), sp.p), sp.fn(view(dd2, day), sp.p)
                assert (a is None) == (b is None) and (a is None or (a.side, round(a.stop, 9), a.rr) ==
                                                       (b.side, round(b.stop, 9), b.rr)), (i, day)
                seen[i] += a is not None
            ta, tb = _sls.sim_positional(sp, sd, 0, 99999999)[0], _sls.sim_positional(sp, sd2, 0, 99999999)[0]
            assert [x[2:6] for x in ta if x[2] <= D] == [x[2:6] for x in tb if x[2] <= D], (i, D)
            for x in ta:
                e_day = int(np.searchsorted(dd["date"], x[2]))
                sig = sp.fn(view(dd, e_day - 1), sp.p)       # the signal came from the previous session
                assert sig is not None and sig.side == x[3], (i, x)
                assert abs(x[5] - dd["o"][e_day] * (1 + x[3] * _sls.SLIPPAGE)) < 1e-3 and x[4] == "09:15", (i, x)   # trade rows round to 4 dp
                assert x[6] >= x[2], (i, x)
    assert sum(v > 0 for v in seen.values()) >= 3, ("seen", seen)


run("candles: daily patterns: perturbing later days changes no decision or entry; fill at next session open",
    t_cs_future_days_change_nothing_daily)


def t_cs_costs_per_lab():
    """Candlestick trades pay the lab's costs: MIS for intraday, CNC for swing (daily bullish), futures for daily
    shorts / two-sided patterns; net = gross - cost."""
    kinds = {"intraday": "equity_mis", "swing": "equity_cnc", "futures": "future"}
    sd = _sl_walk(n_days=60, seed=13, symbol="COST")
    ids = ["cs_outside_bar_5m", "cs_bull_engulfing_15m", "cs_outside_bar_60m", "cs_bull_engulfing_d",
           "cs_bear_engulfing_d", "cs_outside_bar_d"]
    want = {"cs_bull_engulfing_d": "equity_cnc", "cs_bear_engulfing_d": "future", "cs_outside_bar_d": "future"}
    got = set()
    for i in ids:
        sp = _SL_REG[i]
        assert sp.cost_kind == kinds[sp.kind] == want.get(i, "equity_mis"), i
        res = _sls.run_symbol("", "COST", [sp], 0, 99999999, sd=sd, index=None, audit=False)[i]
        assert res["error"] is None, res["error"]
        for x in res["trades"]:
            side, fill, px, qty = x[3], x[5], x[8], x[9]
            bv, sv = (fill * qty, px * qty) if side > 0 else (px * qty, fill * qty)
            assert abs(x[11] - _sl_rtc(bv, sv, sp.cost_kind)) < 0.011 and abs(x[12] - (x[10] - x[11])) < 0.02, x
            got.add(sp.cost_kind)
    assert got == {"equity_mis", "equity_cnc", "future"}, got
    # magnitudes on Rs 1 lakh: delivery costs the most, then futures, then intraday
    m, d, f = (_sl_rtc(1e5, 1e5, k) for k in ("equity_mis", "equity_cnc", "future"))
    # 2026 rates: futures STT 0.05% on the sell side makes a ₹1L futures round trip (~₹104) dearer than MIS (~₹83);
    # delivery ~₹238 incl. the ₹15.34 DP charge
    assert 50 < m < 100 and 200 < d < 260 and m < f < 2 * m, (m, d, f)


run("candles: trades pay MIS (intraday), CNC (daily long) and futures (daily short) costs per the lab", t_cs_costs_per_lab)


def t_cs_float_ties_resolve_the_same_way():
    """B == avgB up to floating-point noise: neither long nor short, whatever history went into the cumsum.
    Same for the ratio tests (e.g. U == 10 % of R): a tie satisfies '<=' every time."""
    # ten candles with body 1.65 and a last candle with body 1.65, at price levels that make the sums inexact
    tail = [(2162.10 + 0.05 * k, 2164.0 + 0.05 * k, 2161.0 + 0.05 * k, 2163.75 + 0.05 * k) for k in range(11)]
    rng = np.random.default_rng(3)
    seen, raw = set(), set()
    for n_hist in (0, 1, 7, 37, 250, 1000, 3000):
        hist = [(x, x + 3.3, x - 2.7, x + rng.choice([-1, 1]) * rng.integers(1, 60) * 0.05)
                for x in 2000.0 + 0.05 * rng.integers(0, 40000, n_hist)]
        a = np.array(hist + tail, float)
        k = _cdl._K(a[:, 0], a[:, 1], a[:, 2], a[:, 3])
        assert abs(k.B[-1] - k.avgB[-1]) < 1e-8, (n_hist, k.B[-1], k.avgB[-1])
        seen.add((bool(k.long[-1]), bool(k.short[-1])))
        raw.add((bool(k.B[-1] > k.avgB[-1]), bool(k.B[-1] < k.avgB[-1])))
    assert seen == {(False, False)}, (seen, raw)          # raw comparisons may flip with the history; these do not
    # a ratio tie: upper shadow exactly 10 % of the range counts as "very little" (hammer)
    rows = _cs_trend(-1, 40)
    last = rows[-1][3]
    lo_, hi_ = last - 2.0, last + 0.2                     # R = 2.2 -> 10 % = 0.22
    ham = (hi_ - 0.22 - 0.5, hi_, lo_, hi_ - 0.22)       # U = 0.22 (a tie up to float noise), B = 0.5, L = 1.48
    o, h, l, c = (np.array(x) for x in zip(*(rows + [ham])))
    assert bool(_cdl.hammer(o, h, l, c)[-1])


run("candles: exact body / ratio ties resolve the same way whatever the cumsum history (tolerance EPS)",
    t_cs_float_ties_resolve_the_same_way)


def t_cs_confirmation_beyond_the_body():
    """_conf: the confirmation candle must close beyond the pattern candle's REAL BODY (Nison's hanging-man rule),
    not merely beyond its close: a close inside a white hanging man's body is no confirmation."""
    rows = _cs_trend(1, 40)
    last = rows[-1][3]
    hm = (last + 0.1, last + 0.42, last - 1.5, last + 0.4)       # white hanging man: body 0.1..0.4 at the top

    def view(extra):
        cols = [np.array(x, float) for x in zip(*(rows + [hm] + extra))]
        d = _sls.DailyView(np.arange(len(cols[0])), *cols, np.ones(len(cols[0])))
        d.symbol, d.nifty, d.vix = "HM", None, None
        return d
    assert _cdlcat.daily_signal(view([]), "hanging_man", False, (-1,)) is not None
    inside = (last + 0.4, last + 0.45, last + 0.2, last + 0.25)  # closes below the HM close but inside its body
    below = (last + 0.4, last + 0.45, last - 0.2, last + 0.0)    # closes below the HM body (below its open)
    assert _cdlcat.daily_signal(view([inside]), "hanging_man", True, (-1,)) is None
    sw = _cdlcat.daily_signal(view([below]), "hanging_man", True, (-1,))
    assert sw is not None and sw.side == -1
    # the same helper decides intraday (candle_entries)
    mk = lambda o, c: _cdl.SessionCandles(np.array(o), None, None, np.array(c), None, None, None, None, None, None,
                                          None)
    assert _cdlcat._confirmed(mk([1.0, 2.0], [1.3, 0.5]), -1, 0, 1)
    assert not _cdlcat._confirmed(mk([1.0, 2.0], [1.3, 1.2]), -1, 0, 1)      # 1.2 is inside the body 1.0..1.3
    assert not _cdlcat._confirmed(mk([1.3, 1.0], [1.0, 1.2]), 1, 0, 1)       # black hammer: must close above 1.3
    assert _cdlcat._confirmed(mk([1.3, 1.0], [1.0, 1.35]), 1, 0, 1)


run("candles: _conf needs a close beyond the pattern candle's real body (white hanging man: below its open)",
    t_cs_confirmation_beyond_the_body)


def t_cs_60m_history_skips_short_last_candle():
    """60m: the short 15:15-15:30 candle of earlier sessions is not in the history used for averages / EMA / ATR:
    moving those 15 minutes of earlier days changes no 60m measurement of a later day (it does on 15m)."""
    sd = _sl_walk(n_days=8, seed=21, symbol="H60")
    ho, hh, hl, hc, hs, hf = _cdl._symbol_candles(sd, 60)
    assert np.all(np.bincount(hs) == 6), np.bincount(hs)          # 09:15 ... 14:15, no 15:15 candle
    m = (sd.date < sd.days[-1]) & (sd.t >= 915)
    r = np.random.default_rng(4)
    f = np.exp(r.normal(0, 0.02, int(m.sum())))
    o, h, l, c = (np.asarray(x, float).copy() for x in (sd.o, sd.h, sd.l, sd.c))
    o[m] *= f
    c[m] *= f
    h[m] = np.maximum(np.maximum(o[m], c[m]), h[m] * f * 1.01)
    l[m] = np.minimum(np.minimum(o[m], c[m]), l[m] * f * 0.99)
    sd2 = _sld.from_arrays(sd.symbol, sd.date, sd.t, o, h, l, c, np.asarray(sd.v, float))
    s1, s2 = _sli.build_sessions(sd)[-1], _sli.build_sessions(sd2)[-1]
    a, b = _cdl.session_candles(s1, 60), _cdl.session_candles(s2, 60)
    assert np.allclose(a.atr, b.atr, equal_nan=True)
    assert all(np.array_equal(a.sigs[k][j], b.sigs[k][j]) for k in a.sigs for j in (0, 1))
    a15, b15 = _cdl.session_candles(s1, 15), _cdl.session_candles(s2, 15)
    assert not np.allclose(a15.atr, b15.atr, equal_nan=True)       # the 15m history does contain those bars


run("candles: 60m history leaves out the short 15:15-15:30 candle of earlier sessions",
    t_cs_60m_history_skips_short_last_candle)


def t_cs_daily_full_history():
    """Daily candles are measured on the whole history up to the decision day: a day's values equal those of
    the full series at that day (no moving 150-day window), and nothing after the day is read."""
    r = np.random.default_rng(8)
    n = 320
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.015, n)))
    o = np.concatenate(([100.0], c[:-1])) * np.exp(r.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + np.abs(r.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(r.normal(0, 0.006, n)))

    def view(k):
        d = _sls.DailyView(np.arange(k), o[:k], h[:k], l[:k], c[:k], np.ones(k))
        d.symbol, d.nifty, d.vix = "FH", None, None
        return d
    full = _cdl.daily_candles(view(n))
    for k in (40, 151, 200, 260, 319):
        got = _cdl.daily_candles(view(k))
        assert len(got.c) == k
        assert np.isclose(got.atr[-1], full.atr[k - 1]), k
        for name, (lm, sm) in got.sigs.items():
            assert lm[-1] == full.sigs[name][0][k - 1] and sm[-1] == full.sigs[name][1][k - 1], (name, k)
    assert len(_cdl.daily_candles(view(300), lookback=150).c) == 150      # the old window is still available


run("candles: daily patterns use the whole history up to the decision day (no moving window)", t_cs_daily_full_history)


def t_sl_positional_exit_stays_in_period():
    """A positional trade entered in a period is closed by the end of that period: DISCOVERY trades never use
    FINAL-period prices (reason period_end, at that session's close)."""
    days = tuple(20260101 + d for d in range(10))
    date, t, a, v = _sl_flat(days=days, n=300)
    for k in range(10):
        a["c"][k * 300:(k + 1) * 300] = 100.0 + k                 # each day closes higher; no stop / target hit
        a["h"][k * 300:(k + 1) * 300] = 100.0 + k
    sd = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    sp = _sl_spec(lambda d, p: _SlSwing(1, stop_pct=50.0, max_hold=6) if len(d) == 1 else None, kind="swing",
                  id="t_period")
    (tr,), _ = _sls.sim_positional(sp, sd, 0, days[3])
    assert tr[2] == days[1] and tr[6] == days[3] and tr[13] == "period_end", tr
    assert abs(tr[8] - 103.0 * (1 - _sls.SLIPPAGE)) < 1e-6, tr
    (tr,), _ = _sls.sim_positional(sp, sd, 0, 99999999)
    assert tr[6] == days[6] and tr[13] == "max_hold", tr


run("stratlab: positional exits never walk past the simulated period (DISCOVERY never uses FINAL prices)",
    t_sl_positional_exit_stays_in_period)


def t_sl_final_ledger_per_dataset():
    """The FINAL ledger is keyed by dataset: a 12-month and a 3-year run of the same strategy each get their own
    once-only FINAL (neither is a RETEST of the other), and each entry records its dataset."""
    cal12 = list(range(20260001, 20260251))
    cal3y = list(range(20240001, 20240751))
    books = {"12m": _sl_book("good", cal12), "3y": _sl_book("good", cal3y)}
    spec = _sl_spec(lambda s, p: [], id="good")
    cur = {}

    def simulate(specs_, lo, hi):
        return {s.id: {"trades": [x for x in books[cur["ds"]] if lo <= x[2] <= hi], "skipped": 0, "lookahead": None,
                       "error": None} for s in specs_}
    with _sl_tmp.TemporaryDirectory() as td:
        ledger = os.path.join(td, "final_ledger.json")
        for ds, cal in (("12m", cal12), ("3y", cal3y), ("12m", cal12)):
            cur["ds"] = ds
            res, _ = _slp.evaluate([spec], cal, simulate, n_symbols=5, ledger_path=ledger, dataset="/data/" + ds)
            assert res[0]["verdict"] == "CONFIRMED", (ds, res[0]["verdict"], res[0]["final_note"])
        with open(ledger) as fh:
            led = _sl_json.load(fh)
        assert len(led) == 2 and sorted(e["dataset"] for e in led.values()) == ["/data/12m", "/data/3y"], led
        sp12, sp3 = _slp.split_sessions(cal12), _slp.split_sessions(cal3y)
        cm = _sls.COST_MODEL
        assert _slp.ledger_key("good", sp12, 250, 5, cm) in led and _slp.ledger_key("good", sp3, 750, 5, cm) in led


run("stratlab: FINAL ledger has one once-only entry per strategy AND dataset (12m vs 3y)",
    t_sl_final_ledger_per_dataset)


# ══════════════════════════════════════════════════════════════════════════
# COSTS 2026 + STRATLAB honest fills (per-run slippage, gaps, ticks, index futures, cost-model ledger)
# ══════════════════════════════════════════════════════════════════════════
section("COSTS 2026 AND STRATLAB FILL MODEL")

import costs as _costs26


def t_costs_2026_rates():
    G = _costs26.GST
    # futures: 0.05 % STT on the sell side only; NSE 0.00183 %; brokerage min(0.03 %, ₹20) per side; stamp 0.002 % buy
    bv = sv = 1e6
    bro, exch, sebi = 40.0, 2e6 * 0.0000183, 2e6 * 1e-6
    want = bro + sv * 0.0005 + exch + sebi + bv * 0.00002 + G * (bro + exch + sebi)
    assert _rtc(bv, sv, "future") == round(want, 2), (_rtc(bv, sv, "future"), want)
    assert _rtc(bv, 0, "future") < _rtc(0, bv, "future")                       # STT only when selling
    # options: 0.15 % of the sell premium, flat ₹20 per order, NSE 0.03553 % of premium
    bro, exch, sebi = 40.0, 20000 * 0.0003553, 20000 * 1e-6
    want = bro + 10000 * 0.0015 + exch + sebi + 10000 * 0.00003 + G * (bro + exch + sebi)
    assert _rtc(10000, 10000, "option") == round(want, 2)
    assert _costs26.exercise_stt(10000) == 15.0 and _costs26.exercise_stt(-5) == 0.0
    # equity: NSE 0.00307 %; delivery pays the ₹15.34 DP charge (GST included) on the sell, once
    exch, sebi = 2e5 * 0.0000307, 2e5 * 1e-6
    cnc = 2e5 * 0.001 + exch + sebi + 1e5 * 0.00015 + G * (exch + sebi) + 15.34
    assert _rtc(1e5, 1e5, "equity_cnc") == round(cnc, 2), (_rtc(1e5, 1e5, "equity_cnc"), cnc)
    assert abs(_rtc(1e5, 0, "equity_cnc") - round(1e5 * 0.001 + 1e5 * 0.0000307 + 1e5 * 1e-6 + 1e5 * 0.00015
                                                  + G * (1e5 * 0.0000307 + 1e5 * 1e-6), 2)) < 0.011     # buy only: no DP
    mis = 40.0 + 1e5 * 0.00025 + exch + sebi + 1e5 * 0.00003 + G * (40.0 + exch + sebi)
    assert _rtc(1e5, 1e5, "equity_mis") == round(mis, 2)                      # intraday: no DP charge
    assert _costs26.DP_CHARGE == 15.34 and _costs26.COST_MODEL_VERSION == "2026-11"
    # trade_cost routes a CNC sell through the DP charge too
    assert _tc(1000.0, 1010.0, 50, "BUY", "NSE", "CNC") - _tc(1000.0, 1010.0, 50, "BUY", "NSE", "MIS") > 15.0


run("costs: 2026 F&O STT (fut 0.05% sell, opt 0.15% premium sell, exercise 0.15%), NSE 0.00307%, CNC DP ₹15.34",
    t_costs_2026_rates)


def t_sl_slippage_is_a_run_setting():
    assert _sls.DEFAULT_SLIPPAGE_BPS == 2.0 and abs(_sls.SLIPPAGE - 0.0002) < 1e-12
    assert _sls.SENSITIVITY_SLIPPAGE_BPS == 5.0 and _sls.INDEX_SLIPPAGE_BPS == 1.0
    date, t, a, v = _sl_flat()
    a["o"][11] = 101.0; a["h"][11] = 101.0; a["c"][11] = 100.5; a["l"][11] = 100.5
    s = _sl_session(a, date, t, v)
    e = _SlEntry(10, 1, stop=99.0, target=110.0)
    k, fill, px, _ = _sls.fill_intraday(s, e)
    assert abs(fill - 101.0 * 1.0002) < 1e-9 and abs(px - 100.0 * 0.9998) < 1e-9, (fill, px)    # default 2 bps
    k, fill5, px5, _ = _sls.fill_intraday(s, e, 0.0005)
    assert abs(fill5 - 101.0 * 1.0005) < 1e-9 and abs(px5 - 100.0 * 0.9995) < 1e-9
    k, fill0, px0, _ = _sls.fill_intraday(s, e, 0.0)
    assert fill0 == 101.0 and px0 == 100.0
    sp = _sl_spec(lambda s, p: [_SlEntry(10, 1, stop=99.0)], id="t_slip")
    sd = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    r2 = _sls.run_symbol("", "SYN", [sp], 0, 99999999, sd=sd, audit=False)["t_slip"]["trades"][0]
    r5 = _sls.run_symbol("", "SYN", [sp], 0, 99999999, sd=sd, audit=False, slippage=0.0005)["t_slip"]["trades"][0]
    assert abs(r2[5] - 101.0 * 1.0002) < 1e-4 and abs(r5[5] - 101.0 * 1.0005) < 1e-4 and r5[12] < r2[12], (r2, r5)
    # positional takes the run's slippage too
    days = tuple(20260101 + d for d in range(6))
    date, t, a, v = _sl_flat(days=days, n=300)
    sd = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    sw = _sl_spec(lambda d, p: _SlSwing(1, stop_pct=5.0, max_hold=2) if len(d) == 1 else None, kind="swing", id="t_sw")
    (tr,), _ = _sls.sim_positional(sw, sd, 0, 99999999, slip=0.0005)
    assert abs(tr[5] - 100.05) < 1e-6 and abs(tr[8] - 99.95) < 1e-6 and tr[13] == "max_hold", tr   # max-hold exit pays it


run("stratlab: slippage is a per-run setting (default 2 bps, 5 bps sensitivity) for intraday and positional",
    t_sl_slippage_is_a_run_setting)


def t_sl_gap_through_stop_fills_at_open():
    date, t, a, v = _sl_flat()
    a["o"][30] = 104.0; a["h"][30] = 104.5; a["l"][30] = 103.5; a["c"][30] = 104.0     # short stopped by a gap up
    s = _sl_session(a, date, t, v)
    k, fill, px, reason = _sls.fill_intraday(s, _SlEntry(10, -1, stop=102.0, target=95.0))
    assert (k, reason) == (30, "stop") and abs(px - 104.0 * (1 + _sls.SLIPPAGE)) < 1e-9, (k, reason, px)   # not 102
    # positional short: a daily gap through the stop fills at that day's open
    days = tuple(20260101 + d for d in range(6))
    date, t, a, v = _sl_flat(days=days, n=300)
    sl3 = slice(3 * 300, 4 * 300)
    a["o"][sl3] = a["h"][sl3] = a["l"][sl3] = a["c"][sl3] = 110.0
    sd = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    fut = _sl_spec(lambda d, p: _SlSwing(-1, stop_pct=3.0, max_hold=5) if len(d) == 1 else None, kind="futures",
                   horizon="positional", id="t_fgap")
    (tr,), _ = _sls.sim_positional(fut, sd, 0, 99999999)
    assert tr[13] == "stop" and tr[6] == days[3] and abs(tr[8] - 110.0 * (1 + _sls.SLIPPAGE)) < 1e-6, tr


run("stratlab: a stop gapped through fills at the bar's (or day's) open plus slippage, not at the stop",
    t_sl_gap_through_stop_fills_at_open)


def t_sl_market_exits_pay_slippage():
    date, t, a, v = _sl_flat()
    a["o"][26] = 100.4
    sq = 15 * 60 + 15 - 555
    a["o"][sq] = 99.0
    s = _sl_session(a, date, t, v)
    k, _f, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=90.0, time_stop=15), 0.0003)
    assert (k, reason) == (26, "time") and abs(px - 100.4 * 0.9997) < 1e-9, (k, reason, px)
    k, _f, px, reason = _sls.fill_intraday(s, _SlEntry(10, -1, stop=110.0), 0.0003)
    assert (k, reason) == (sq, "squareoff") and abs(px - 99.0 * 1.0003) < 1e-9, (k, reason, px)     # short buys back worse
    s360 = _sl_session({x: y[:360] for x, y in a.items()}, date[:360], t[:360], v[:360])         # data ends 15:14
    k, _f, px, reason = _sls.fill_intraday(s360, _SlEntry(10, 1, stop=90.0), 0.0003)
    assert reason == "squareoff" and abs(px - 100.0 * 0.9997) < 1e-9


run("stratlab: time, square-off and data-end exits pay slippage (both sides)", t_sl_market_exits_pay_slippage)


def t_sl_target_needs_a_tick_through():
    assert _sls.equity_tick(500.0, 20240101) == 0.05 and _sls.equity_tick(200.0, 20240101) == 0.05
    assert _sls.equity_tick(200.0, 20250616) == 0.01 and _sls.equity_tick(300.0, 20260101) == 0.05
    date, t, a, v = _sl_flat()
    a["h"][20] = 102.0                                                      # touches 102 exactly
    a["h"][40] = 102.005                                                    # less than a tick (₹0.01 under ₹250 from 16-06-2025) through
    s = _sl_session(a, date, t, v)
    k, _f, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=98.0, target=102.0))
    assert reason == "squareoff", (k, reason, px)                           # a touch is not a fill
    a["h"][60] = 102.01                                                     # one tick through
    s = _sl_session(a, date, t, v)
    k, _f, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=98.0, target=102.0))
    assert (k, reason, px) == (60, "target", 102.0), (k, reason, px)
    a["o"][50] = a["h"][50] = a["l"][50] = a["c"][50] = 103.0               # gap beyond the target: fill at the open
    s = _sl_session(a, date, t, v)
    k, _f, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=98.0, target=102.0))
    assert (k, reason, px) == (50, "target", 103.0), (k, reason, px)
    k, _f, px, reason = _sls.fill_intraday(s, _SlEntry(10, 1, stop=98.0, target=102.0, trail=("pct", 5.0)))
    assert (k, reason, px) == (50, "target", 103.0), (k, reason, px)        # trailing path: same rule
    # short side
    date, t, a, v = _sl_flat()
    a["l"][20] = 98.0
    s = _sl_session(a, date, t, v)
    assert _sls.fill_intraday(s, _SlEntry(10, -1, stop=102.0, target=98.0))[3] == "squareoff"
    a["l"][30] = 97.95
    s = _sl_session(a, date, t, v)
    assert _sls.fill_intraday(s, _SlEntry(10, -1, stop=102.0, target=98.0))[2:] == (98.0, "target")
    # positional: a daily high that only touches the target is not a fill
    days = tuple(20260101 + d for d in range(6))
    date, t, a, v = _sl_flat(days=days, n=300)
    a["h"][2 * 300 + 50] = 110.0
    sd = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    sw = _sl_spec(lambda d, p: _SlSwing(1, stop=95.0, target=110.0, max_hold=3) if len(d) == 1 else None,
                  kind="swing", id="t_ptgt")
    (tr,), _ = _sls.sim_positional(sw, sd, 0, 99999999)
    assert tr[13] == "max_hold", tr
    a["h"][2 * 300 + 50] = 110.05
    sd = _sld.from_arrays("SYN", date, t, a["o"], a["h"], a["l"], a["c"], v)
    (tr,), _ = _sls.sim_positional(sw, sd, 0, 99999999)
    assert tr[13] == "target" and tr[8] == 110.0 and tr[6] == days[2], tr


run("stratlab: a target fills only if traded through by a tick (a gap beyond fills at the open)",
    t_sl_target_needs_a_tick_through)


def _sl_index_sd(n_days=3, px=24000.0):
    days = tuple(20260105 + d for d in range(n_days))
    date, t, a, v = _sl_flat(days=days, px=px)
    return days, date, t, a, np.zeros(len(date))


def t_sl_index_futures():
    from stratlab.api import strategy as _sl_strategy
    for bad in (dict(kind="intraday", underlying="NIFTY"), dict(kind="futures", underlying="FINNIFTY")):
        try:
            _sl_strategy("t_bad_idx", bad["kind"], "x", "x", horizon="intraday", underlying=bad["underlying"])(lambda s, p: [])
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    assert _sls.LOT_SIZES == {"NIFTY": 65, "BANKNIFTY": 30}
    days, date, t, a, v = _sl_index_sd()
    a["h"][100] = 24100.0
    sd = _sld.from_arrays("_NIFTY50_min", date, t, a["o"], a["h"], a["l"], a["c"], v)
    sp = _sl_spec(lambda s, p: [_SlEntry(10, 1, stop=23900.0, target=24050.0)], kind="futures", horizon="intraday",
                  id="t_nifty_fut", underlying="NIFTY")
    stock = _sl_spec(lambda s, p: [_SlEntry(10, 1, stop=23900.0)], id="t_stock_on_index")
    out = _sls.run_symbol("", "_NIFTY50_min", [sp, stock], 0, 99999999, sd=sd, audit=False)
    trs = out["t_nifty_fut"]["trades"]
    assert len(trs) == len(days) and out["t_stock_on_index"]["trades"] == [], out
    tr = trs[0]
    assert tr[9] == 65 and abs(tr[5] - 24000.0 * 1.0001) < 1e-3 and tr[13] == "target" and tr[8] == 24050.0, tr
    assert tr[11] == _sl_rtc(tr[5] * 65, 24050.0 * 65, "future") and abs(tr[10] - (24050.0 - tr[5]) * 65) < 0.01, tr
    # index specs never run on stocks; BANKNIFTY uses its own data and lot size
    assert _sls.run_symbol("", "RELIANCE", [sp], 0, 99999999, sd=sd, audit=False)["t_nifty_fut"]["trades"] == []
    bn = _sl_spec(lambda s, p: [_SlEntry(10, -1, stop=24200.0)], kind="futures", horizon="intraday", id="t_bn",
                  underlying="BANKNIFTY", lots=2)
    assert _sls.run_symbol("", "_NIFTY50_min", [bn], 0, 99999999, sd=sd, audit=False)["t_bn"]["trades"] == []
    tb = _sls.run_symbol("", "_NIFTYBANK_min", [bn], 0, 99999999, sd=sd, audit=False)["t_bn"]["trades"][0]
    assert tb[9] == 60 and tb[13] == "squareoff" and abs(tb[8] - 24000.0 * 1.0001) < 1e-3, tb
    # run slippage override for index futures; positional holds allowed on the index
    tr5 = _sls.run_symbol("", "_NIFTY50_min", [sp], 0, 99999999, sd=sd, audit=False,
                          index_slippage=0.00025)["t_nifty_fut"]["trades"][0]
    assert abs(tr5[5] - 24000.0 * 1.00025) < 1e-3
    pos = _sl_spec(lambda d, p: _SlSwing(-1, stop_pct=2.0, max_hold=2) if len(d) == 1 else None, kind="futures",
                   horizon="positional", id="t_nifty_pos", underlying="NIFTY")
    (tp,) = _sls.run_symbol("", "_NIFTY50_min", [pos], 0, 99999999, sd=sd, audit=False)["t_nifty_pos"]["trades"]
    assert tp[3] == -1 and tp[9] == 65 and tp[13] == "max_hold" and tp[11] == _sl_rtc(tp[8] * 65, tp[5] * 65, "future")
    # the screen's one-symbol concentration gate does not apply to a single-underlying strategy
    cal = list(range(20260001, 20260201))
    book = [x[:1] + ("_NIFTY50_min",) + x[2:] for x in _sl_book("t_nifty_fut", cal)]
    split = _slp.split_sessions(cal)
    assert not _slp.screen(book, split, "intraday")["passed"]
    assert _slp.screen(book, split, "intraday", single_underlying=True)["passed"]


run("stratlab: NIFTY/BANKNIFTY futures on index bars (lots 65/30, 1 bps, futures costs, positional, own screen)",
    t_sl_index_futures)


def t_sl_ledger_keyed_by_cost_model_and_screen_only():
    cal = list(range(20260001, 20260251))
    book = _sl_book("good", cal)
    spec = _sl_spec(lambda s, p: [], id="good")
    calls = []

    def simulate(specs_, lo, hi):
        calls.append((lo, hi))
        return {s.id: {"trades": [x for x in book if lo <= x[2] <= hi], "skipped": 0, "lookahead": None,
                       "error": None} for s in specs_}
    split = _slp.split_sessions(cal)
    with _sl_tmp.TemporaryDirectory() as td:
        ledger = os.path.join(td, "final_ledger.json")
        # screen only: FINAL is never simulated and nothing is written
        res, trades = _slp.evaluate([spec], cal, simulate, n_symbols=5, ledger_path=ledger, screen_only=True)
        assert res[0]["verdict"] == "PASSED SCREEN" and res[0]["final"] is None and not os.path.exists(ledger)
        assert all(hi <= split.discovery[1] for _lo, hi in calls) and all(x[2] <= split.discovery[1] for x in trades["good"])
        # the old, unversioned key (cost model None) and a new cost model each get their own once-only FINAL
        r_old, _ = _slp.evaluate([spec], cal, simulate, n_symbols=5, ledger_path=ledger, cost_model=None)
        r_new, _ = _slp.evaluate([spec], cal, simulate, n_symbols=5, ledger_path=ledger, cost_model="2026-11/f2")
        assert r_old[0]["verdict"] == "CONFIRMED" and r_new[0]["verdict"] == "CONFIRMED", r_new[0]["final_note"]
        assert "also seen under cost model legacy" in r_new[0]["final_note"], r_new[0]["final_note"]
        led = _sl_json.load(open(ledger))
        assert set(led) == {_slp.ledger_key("good", split, 250, 5), _slp.ledger_key("good", split, 250, 5, "2026-11/f2")}
        # default cost model = the simulator's current one
        r_cur, _ = _slp.evaluate([spec], cal, simulate, n_symbols=5, ledger_path=ledger)
        assert _slp.ledger_key("good", split, 250, 5, _sls.COST_MODEL) in _sl_json.load(open(ledger))
        # a strategy whose code changed after its FINAL was seen is still a RETEST under a new cost model
        changed = _sl_spec(lambda s, p: [1], id="good")
        r_chg, _ = _slp.evaluate([changed], cal, simulate, n_symbols=5, ledger_path=ledger, cost_model="2099-01/f9")
        assert r_chg[0]["verdict"] == "RETEST (FINAL already used)", r_chg[0]
        # and a peek under one cost model is not washed out by another
        bad = _sl_spec(lambda s, p: [], id="bad")
        badbook = _sl_book("bad", cal, win=100.0, loss=-300.0)

        def sim_bad(specs_, lo, hi):
            return {s.id: {"trades": [x for x in badbook if lo <= x[2] <= hi], "skipped": 0, "lookahead": None,
                           "error": None} for s in specs_}
        _slp.evaluate([bad], cal, sim_bad, n_symbols=5, ledger_path=ledger, peek_final=True, cost_model="A")
        goodbook = _sl_book("bad", cal)                       # same id, now it would pass the screen

        def sim_good(specs_, lo, hi):
            return {s.id: {"trades": [x for x in goodbook if lo <= x[2] <= hi], "skipped": 0, "lookahead": None,
                           "error": None} for s in specs_}
        r_b, _ = _slp.evaluate([bad], cal, sim_good, n_symbols=5, ledger_path=ledger, cost_model="B")
        assert r_b[0]["verdict"] == "RETEST (FINAL already used)" and "peeked" in r_b[0]["final_note"], r_b[0]
        r_b2, _ = _slp.evaluate([bad], cal, sim_good, n_symbols=5, ledger_path=ledger, cost_model="B")
        assert r_b2[0]["verdict"] == "RETEST (FINAL already used)", r_b2[0]          # the taint is remembered
        try:
            _slp.evaluate([spec], cal, simulate, screen_only=True, peek_final=True)
            raise AssertionError("screen_only + peek_final accepted")
        except ValueError:
            pass


run("stratlab: FINAL ledger keyed by cost model (a re-costed run is not a second look; peeks/code changes still are); "
    "--screen-only never simulates FINAL", t_sl_ledger_keyed_by_cost_model_and_screen_only)


def t_sl_cli_sensitivity_and_screen_only():
    from stratlab import run as _slr
    import contextlib, io, csv as _csv

    def bars(sd):
        return [[f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}T{m // 60:02d}:{m % 60:02d}:00", float(o), float(h), float(l), float(c), float(v)]
                for d, m, o, h, l, c, v in zip(sd.date, sd.t, sd.o, sd.h, sd.l, sd.c, sd.v)]
    with _sl_tmp.TemporaryDirectory() as td:
        src, out = os.path.join(td, "bars"), os.path.join(td, "out"); os.makedirs(src)
        for k, sym in enumerate(("AAA", "BBB")):
            with open(os.path.join(src, sym + ".json"), "w") as fh:
                _sl_json.dump({"symbol": sym, "interval": "minute", "bars": bars(_sl_walk(n_days=12, seed=30 + k, symbol=sym))}, fh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _slr.main(["--data", src, "--kind", "scalping", "--ids", "scalp_orb5", "--out", out, "--screen-only",
                            "--cache", os.path.join(td, "cache"), "--jobs", "1"])
        assert rc == 0
        text = open(os.path.join(out, "summary.txt")).read()
        assert "2.0 bps slippage" in text and "net@5" in text and "SCREEN ONLY" in text and _sls.COST_MODEL in text, text[:600]
        row = list(_csv.DictReader(open(os.path.join(out, "results.csv"))))[0]
        assert row["slippage_bps"] == "2.0" and row["sens_slippage_bps"] == "5.0" and row["cost_model"] == _sls.COST_MODEL
        assert float(row["sens_disc_net"]) < float(row["disc_net"]) and int(row["sens_disc_trades"]) > 0, row
        assert not os.path.exists(os.path.join(td, "cache", "final_ledger.json"))


run("stratlab: CLI reports 2 bps by default, a 5 bps sensitivity column, the cost model, and --screen-only",
    t_sl_cli_sensitivity_and_screen_only)

def t_sl_cli_trials_ledger():
    """--trials appends one row per evaluated strategy (failures too); without it nothing is recorded."""
    from stratlab import run as _slr
    import contextlib, io, csv as _csv

    def bars(sd):
        return [[f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}T{m // 60:02d}:{m % 60:02d}:00", float(o), float(h), float(l), float(c), float(v)]
                for d, m, o, h, l, c, v in zip(sd.date, sd.t, sd.o, sd.h, sd.l, sd.c, sd.v)]
    with _sl_tmp.TemporaryDirectory() as td:
        src = os.path.join(td, "bars"); os.makedirs(src)
        for k, sym in enumerate(("AAA", "BBB")):
            with open(os.path.join(src, sym + ".json"), "w") as fh:
                _sl_json.dump({"symbol": sym, "interval": "minute", "bars": bars(_sl_walk(n_days=12, seed=30 + k, symbol=sym))}, fh)
        led = os.path.join(td, "trials.csv")
        args = ["--data", src, "--kind", "scalping", "--ids", "scalp_orb5,scalp_vwap_bounce", "--screen-only",
                "--cache", os.path.join(td, "cache"), "--jobs", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            assert _slr.main(args + ["--out", os.path.join(td, "o1"), "--trials", ""]) == 0
            assert not os.path.exists(led)
            assert _slr.main(args + ["--out", os.path.join(td, "o2"), "--trials", led]) == 0
        rows = list(_csv.DictReader(open(led)))
        assert sorted(r["strategy_id"] for r in rows) == ["scalp_orb5", "scalp_vwap_bounce"], rows
        assert all(r["stage"] == "screen" and r["dataset"] == "bars" and r["params_fp"] for r in rows)


run("stratlab: --trials records every evaluated strategy in the trials ledger; off by default", t_sl_cli_trials_ledger)

section("STRATLAB: trials ledger and Deflated Sharpe Ratio (stratlab/trials.py)")
from stratlab import tests_trials as _ttr  # noqa: E402
for _n, _f in _ttr.TESTS:
    run(_n, _f)

section("STRATLAB: long-term reversal basket (stratlab/ltr.py)")
from stratlab import tests_ltr as _tltr  # noqa: E402
for _n, _f in _tltr.TESTS:
    run(_n, _f)

section("STRATLAB: factor momentum (stratlab/factor_momentum.py, pre-registered prereg/momentum_v1.json)")
from stratlab import tests_factor_momentum as _tfm  # noqa: E402
for _n, _f in _tfm.TESTS:
    run(_n, _f)

section("STRATLAB: round-5 Kite ecosystem (stratlab/catalog/r5_kite.py, kite_*.py, prereg/kite_ecosystem_v1.json)")
from stratlab import tests_kite as _tki  # noqa: E402
for _n, _f in _tki.TESTS:
    run(_n, _f)

section("STRATLAB: round-6 GitHub strategies (stratlab/catalog/r6_github.py, r6_portfolio.py, r6_ind.py, prereg/github_v1.json)")
from stratlab import tests_r6 as _tr6  # noqa: E402
for _n, _f in _tr6.TESTS:
    run(_n, _f)

section("STRATLAB: round-7 GitHub strategies (stratlab/catalog/r7_github2.py, r7_portfolio.py, r7_tps.py, r7_ind.py, prereg/github_v2.json)")
from stratlab import tests_r7 as _tr7  # noqa: E402
for _n, _f in _tr7.TESTS:
    run(_n, _f)

section("STRATLAB: old-era minute OOS (stratlab/protocol_min_oos.py, prereg/oldera_min_v1.json)")
from stratlab import tests_min_oos as _tmo  # noqa: E402
for _n, _f in _tmo.TESTS:
    run(_n, _f)

section("STRATLAB: timeframe grid + Rs 10 lakh account (stratlab/tfgrid.py, capital.py, prereg/tfgrid_v1.json)")
from stratlab import tests_tfgrid as _ttg  # noqa: E402
for _n, _f in _ttg.TESTS:
    run(_n, _f)

section("STRATLAB: event study - promoter buying and earnings moves (stratlab/events_*.py, prereg/events_v1.json)")
from stratlab import tests_events as _tev  # noqa: E402
for _n, _f in _tev.TESTS:
    run(_n, _f)

section("STRATLAB: flow data (stratlab/flowdata.py: deals, corporate actions, index membership, FII; stratlab/optmin.py)")
from stratlab import tests_flowdata as _tfd  # noqa: E402
for _n, _f in _tfd.TESTS:
    run(_n, _f)

section("STRATLAB: flow study - bulk/block deals and NIFTY index changes (stratlab/flows_*.py, prereg/flows_v1.json)")
from stratlab import tests_flows as _tfl  # noqa: E402
for _n, _f in _tfl.TESTS:
    run(_n, _f)

section("STRATLAB: web-sourced strategies - low volatility, dividend month, earnings straddles (stratlab/web_*.py, prereg/web_v1.json)")
from stratlab import tests_web as _twb  # noqa: E402
for _n, _f in _twb.TESTS:
    run(_n, _f)

section("STRATLAB: data-driven discovery - candidates created from the lab's data, selected on old data (stratlab/datagen_*.py, prereg/datagen_v1.json)")
from stratlab import tests_datagen as _tdg  # noqa: E402
for _n, _f in _tdg.TESTS:
    run(_n, _f)

section("STRATLAB: robustness diagnostics - PBO/CSCV, White RC/Hansen SPA, Monte Carlo, walk-forward, look-ahead (stratlab/robustness.py, report-only)")
from stratlab import tests_robustness as _trb  # noqa: E402
for _n, _f in _trb.TESTS:
    run(_n, _f)

section("STRATLAB: simulator audit + corrected data (stratlab/refsim.py, corrected.py, prereg/rerun_corrected_v1.json)")
from stratlab import tests_audit as _tau  # noqa: E402
for _n, _f in _tau.TESTS:
    run(_n, _f)

section("OPTION BAR ARCHIVER (option_bar_archiver.py, OPTION_BAR_ARCHIVER_ENABLED default off)")
import test_option_bar_archiver as _toba  # noqa: E402
for _n, _f in _toba.TESTS:
    run(_n, _f)


# ── stratlab/options_vrp_vix: pre-registered VIX-regime filter on the winged weekly strangle ──
from stratlab import options_vrp_vix as _vx


def _vx_vix():
    """VIX on weekdays (no 2026-10-20 holiday) ending 2026-10-23: 13 everywhere, except 10-06 = 14 (above the
    median, +7.7 % over 5 sessions -> take the 10-07 week), 10-13 = 12 (below median -> skip 10-14) and
    10-19 = 16 (+23 % vs 10-12 -> spike, skip 10-21)."""
    days, d = [], _ordate(2026, 10, 23)
    while len(days) < 320:
        if d.weekday() < 5 and d != _ordate(2026, 10, 20):
            days.append(d)
        d -= _ortd(days=1)
    v = {x: 13.0 for x in days}
    v[_ordate(2026, 10, 6)], v[_ordate(2026, 10, 13)], v[_ordate(2026, 10, 19)] = 14.0, 12.0, 16.0
    return v


def _vx_spec(vix):
    sp = _vx.VixOptSpec("T-HV", "t", dict(_vx.HIGHVIX.params), lots=2)
    sp.vix = vix
    return sp


def t_vx_filter_rules_and_past_only():
    v = _vx_vix()
    f = _vx.vix_filter(v, _ordate(2026, 10, 7))
    assert f["take"] and f["prev"] == _ordate(2026, 10, 6) and f["median"] == 13.0 and abs(f["chg5"] - 1 / 13) < 1e-9
    assert _vx.vix_filter(v, _ordate(2026, 10, 14))["why"] == "below_median"
    f = _vx.vix_filter(v, _ordate(2026, 10, 21))
    assert f["why"] == "spike" and f["prev"] == _ordate(2026, 10, 19) and abs(f["chg5"] - 3 / 13) < 1e-9
    # strictly greater than the median: equal is not taken
    assert _vx.vix_filter(v, _ordate(2026, 10, 8))["why"] == "below_median"              # prev 10-07 = 13 = median
    # exactly +15 % over 5 sessions is a spike (>=, float-safe); just under is taken
    w = dict(v); w[_ordate(2026, 10, 19)] = 14.95                                       # 14.95 / 13 - 1 = 0.15
    assert _vx.vix_filter(w, _ordate(2026, 10, 21))["why"] == "spike"
    w[_ordate(2026, 10, 12)], w[_ordate(2026, 10, 19)] = 13.0, 13.0 * 1.1499
    assert _vx.vix_filter(w, _ordate(2026, 10, 21))["take"]
    # the median is over the 250 sessions BEFORE prev, not including it
    days = sorted(v)
    i = days.index(_ordate(2026, 10, 6))
    w = dict(v)
    for x in days[:i - 250]:
        w[x] = 99.0                                                                     # outside the window
    assert _vx.vix_filter(w, _ordate(2026, 10, 7)) == _vx.vix_filter(v, _ordate(2026, 10, 7))
    w = dict(v)
    for x in days[i - 250:i - 124]:
        w[x] = 99.0                                                                     # 126 of 250 inside: median moves
    assert _vx.vix_filter(w, _ordate(2026, 10, 7))["median"] == 99.0
    assert _vx.vix_filter(w, _ordate(2026, 10, 7))["why"] == "below_median"
    # no look-ahead: closes on/after the entry day are never read
    w = dict(v)
    for x in days:
        if x >= _ordate(2026, 10, 7):
            w[x] = 500.0
    assert _vx.vix_filter(w, _ordate(2026, 10, 7)) == _vx.vix_filter(v, _ordate(2026, 10, 7))
    # fewer than 250 closes before prev: no trade
    short = {x: 13.0 for x in days[i - 249:i + 1]}
    short[_ordate(2026, 10, 6)] = 14.0
    assert _vx.vix_filter(short, _ordate(2026, 10, 7))["why"] == "no_history"
    short[days[i - 250]] = 13.0
    assert _vx.vix_filter(short, _ordate(2026, 10, 7))["take"]


def t_vx_replay_trades_only_taken_weeks_identical_to_baseline():
    v = _vx_vix()
    bars = _or_bars(_VRP_DAYS, lambda i: 20 * _ormath.sin(i / 60.0))
    sp = _vx_spec(v)
    res, strat = _vrp._run_replay(sp, bars, v, 20261023)
    base, _ = _vrp._run_replay(_vrp.BY_ID["VRP-STRANGLE-016-WINGED"], bars, v, 20261023)
    assert [t.entry_ts.date() for t in base.trades] == [_ordate(2026, 10, 7), _ordate(2026, 10, 14), _ordate(2026, 10, 21)]
    assert [t.entry_ts.date() for t in res.trades] == [_ordate(2026, 10, 7)]
    assert [(d, f["why"]) for d, f in strat.filter_log] == [(_ordate(2026, 10, 7), "take"),
                                                           (_ordate(2026, 10, 14), "below_median"),
                                                           (_ordate(2026, 10, 21), "spike")]
    a, b = res.trades[0], base.trades[0]
    # exactly the winged structure: same legs, fills, 2 lots, costs, haircut, net; margin rule unchanged
    assert [(l.opt, l.strike, l.side, l.entry_px, l.exit_px) for l in a.legs] == \
           [(l.opt, l.strike, l.side, l.entry_px, l.exit_px) for l in b.legs]
    assert len(a.legs) == 4 and a.qty == 2 * 65 and (a.costs, a.slippage, a.net) == (b.costs, b.slippage, b.net)
    assert sp.margin(a) == _vrp.BY_ID["VRP-STRANGLE-016-WINGED"].margin(b)
    # the fingerprint covers the filter's parameters
    q = _vx_spec(v); q.params = dict(q.params, spike_pct=0.2)
    assert q.fingerprint != sp.fingerprint and sp.fingerprint != _vrp.BY_ID["VRP-STRANGLE-016-WINGED"].fingerprint


def t_vx_no_lookahead():
    v = _vx_vix()
    path = lambda i: 25 * _ormath.sin(i / 70.0)
    base = _or_bars(_VRP_DAYS, path)
    sp = _vx_spec(v)
    a, _ = _vrp._run_replay(sp, base, v, 20261023)
    # future bars and future VIX closes changed wildly: the same week, legs and entry fills
    t0 = a.trades[0].entry_ts
    wild = [(b[0], b[1] + 700, b[2] + 700, b[3] + 700, b[4] + 700) if b[0] > t0 else b for b in base]
    w = dict(v)
    for x in list(w):
        if x >= _ordate(2026, 10, 7):
            w[x] = w[x] * 3
    b, _ = _vrp._run_replay(_vx_spec(w), wild, w, 20261013)
    key = lambda r: [(t.entry_ts, [(l.opt, l.strike, l.side, l.entry_px) for l in t.legs]) for t in r.trades]
    assert key(a)[:1] == key(b) and len(key(b)) == 1 and a.trades[0].net != b.trades[0].net
    assert _vrp.audit_lookahead(sp, base, v, a.trades) is None
    # a DISCOVERY-style simulation up to `hi` never reads later bars
    poisoned = base + [(x[0] + _ortd(days=28), float("nan"), float("nan"), float("nan"), float("nan"))
                       for x in _or_bars(_VRP_DAYS[-5:])]
    s1 = _vrp.make_simulator(base, v)([sp], 20261001, 20261016)
    s2 = _vrp.make_simulator(poisoned, v)([sp], 20261001, 20261016)
    assert s1 == s2 and len(s1["T-HV"]["trades"]) == 1 and s1["T-HV"]["lookahead"] is None


def t_vx_merge_vix_and_compare():
    import tempfile as _t, json as _j, os as _o
    with _t.TemporaryDirectory() as td:
        pth = _o.path.join(td, "v.json")
        with open(pth, "w") as fh:
            _j.dump({"symbol": "INDIA VIX", "interval": "day",
                     "bars": [["2022-01-03T00:00:00", 1, 1, 1, 16.45, 0], ["2022-11-10T00:00:00", 1, 1, 1, 99.0, 0]]}, fh)
        m = _vx.merge_vix({_ordate(2022, 11, 10): 15.57}, [pth])
    assert m == {_ordate(2022, 1, 3): 16.45, _ordate(2022, 11, 10): 15.57}               # base wins on overlap
    bd = {1: {"net": 100.0, "bps": 10.0}, 2: {"net": -50.0, "bps": -5.0}, 3: {"net": 30.0, "bps": 3.0}}
    cd = {1: {"net": 100.0, "bps": 10.0}}
    dec = {1: {"take": True, "why": "take"}, 2: {"take": False, "why": "spike"}, 3: {"take": False, "why": "below_median"}}
    c = _vx._week_compare(bd, cd, dec, (1, 3))
    assert c["weeks"] == 3 and c["taken"] == 1 and c["cond_bps"] == 10.0 and abs(c["uncond_bps"] - 8 / 3) < 1e-9
    assert c["rest_bps"] == -1.0 and c["spike_bps"] == -5.0 and not c["mismatch"] and not c["missing"]


run("options VRP x VIX: strict median of the 250 prior sessions, >=15 % 5-session spike, past closes only",
    t_vx_filter_rules_and_past_only)
run("options VRP x VIX: trades only taken weeks, each identical to the unconditional winged trade",
    t_vx_replay_trades_only_taken_weeks_identical_to_baseline)
run("options VRP x VIX: no look-ahead (future bars and VIX, audit, DISCOVERY run)", t_vx_no_lookahead)
run("options VRP x VIX: older VIX merged without overwriting, per-week comparison", t_vx_merge_vix_and_compare)


# ══════════════════════════════════════════════════════════════════════════
section("STRATLAB OPENING-AUCTION FADE (stratlab/auction.py)")
# ══════════════════════════════════════════════════════════════════════════
from stratlab import auction as _sla  # noqa: E402


def _auc_sd(symbol="AUC", gaps=(0.0, 0.05, -0.03, 0.25, 0.01), n=375, seed=3):
    """One synthetic symbol: day k opens at prev close x (1 + gaps[k]) (gaps[0] ignored), flat-ish walk after."""
    rng = np.random.default_rng(seed)
    date, t, o, c = [], [], [], []
    px = 1000.0
    for k, g in enumerate(gaps):
        op = px * (1 + g) if k else px
        cl = op * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
        oo = np.concatenate(([op], cl[:-1]))
        date += [20260105 + k] * n; t += list(555 + np.arange(n)); o += list(oo); c += list(cl)
        px = float(cl[-1])
    o, c = np.array(o), np.array(c)
    return _sld.from_arrays(symbol, date, t, o, np.maximum(o, c) * 1.0002, np.minimum(o, c) * 0.9998, c,
                            np.full(len(c), 1000.0))


def t_auc_rule_and_features():
    assert _sla.auction_rule(100, 103, 0.02)[0] == -1 and _sla.auction_rule(100, 97, 0.02)[0] == 1
    assert _sla.auction_rule(100, 102, 0.02) is None            # |gap| must be > 2 %, not >=
    assert _sla.auction_rule(100, 125, 0.02) is None            # > 20 %: corporate action / circuit excluded
    sd = _auc_sd()
    f = _sla.session_features(sd)
    assert not np.isfinite(f[0][1])                             # no previous session -> no prev_close
    for k in range(1, len(f)):
        assert abs(f[k][1] - float(sd.c[sd.ends[k - 1] - 1])) < 1e-9 and abs(f[k][2] - float(sd.o[sd.starts[k]])) < 1e-9
        assert f[k][3] == 560 and abs(f[k][4] - float(sd.o[sd.starts[k] + 5])) < 1e-9
    rows = _sla.candidates_for("AUC", f, 0.02)
    assert [(r[0], r[2]) for r in rows] == [(20260106, -1), (20260107, 1)], rows   # 25 % gap and 1 % gap dropped


def t_auc_no_lookahead():
    """Changing anything after the 09:20 open of day d, or any later day, leaves day d's trade unchanged; changing
    the 09:16-09:19 bars changes nothing either (the hold has no stop)."""
    base = _sla.candidates_for("AUC", _sla.session_features(_auc_sd()), 0.02)
    sd = _auc_sd()
    a = int(sd.starts[1])
    for arr in (sd.o, sd.h, sd.l, sd.c):
        arr[a + 1:a + 5] *= 1.03                                # 09:16-09:19 of day 1
        arr[a + 6:] *= 0.9                                      # after the 09:20 open, and all later days
    pert = _sla.candidates_for("AUC", _sla.session_features(sd), 0.02)
    assert pert[0] == base[0], (pert[0], base[0])


def t_auc_ranking_cap_and_costs():
    rows = [(20260105, f"S{i:02d}", -1, 0.021 + i * 0.0006, 500.0, 560, 499.0) for i in range(14)]
    pick = _sla.select(rows, _sla.SPECS[0], 0, 99999999)
    assert len(pick) == 10 and {r[1] for r in pick} == {f"S{i:02d}" for i in range(4, 14)}   # 10 largest |gap|
    assert len(_sla.select(rows, _sla.SPECS[1], 0, 99999999)) == 0        # all gaps <= 3 %
    tr = _sla.settle("X", rows[0], 2.1e-4)
    qty = 200                                                             # floor(1e5 / 500)
    fill, ex = 500.0 * (1 - 2.1e-4), 499.0 * (1 + 2.1e-4)                 # short: sell lower, buy back higher
    assert tr[9] == qty and abs(tr[5] - fill) < 1e-3 and abs(tr[8] - ex) < 1e-3 and tr[3] == -1
    assert abs(tr[10] - (fill - ex) * qty) < 0.01
    from costs import round_trip_cost as _rtc
    assert abs(tr[11] - _rtc(ex * qty, fill * qty, "equity_mis")) < 1e-6 and abs(tr[12] - (tr[10] - tr[11])) < 0.01
    fr = (_rtc(1e5, 1e5, "equity_mis") + 2 * 2.1e-4 * 1e5) / 1e5 * 1e4  # pre-registered friction ~12.3 bps
    assert 11.8 < fr < 12.8, fr


def t_auc_protocol_final_never_for_failures():
    """Through protocol.evaluate: FINAL is never simulated for screen failures; default slippage is 2.1 bps."""
    seen = []
    rows = [(20240101 + (d % 28) + (d // 28) * 100, "A", 1, -0.03, 100.0, 560, 99.0) for d in range(200)]   # always loses
    base = _sla.make_simulator(rows)

    def sim_(specs, lo, hi):
        seen.append((lo, hi))
        return base(specs, lo, hi)
    cal = sorted({r[0] for r in rows})
    res, _tr = _slp.evaluate(_sla.SPECS, cal, sim_, n_symbols=1, ledger_path=None, cost_model=_sla.COST_MODEL)
    assert all(r["verdict"] == "FAILED SCREEN" for r in res) and len(seen) == 1, (seen, [r["why"] for r in res])
    assert _sla.DEFAULT_SLIPPAGE_BPS == 2.1 and _sla.COST_MODEL.startswith(_sls.COST_MODEL)


run("auction: signal rule, prev_close / 09:15 / 09:20 features, 20 % exclusion", t_auc_rule_and_features)
run("auction: later bars cannot change an earlier signal or fill (look-ahead)", t_auc_no_lookahead)
run("auction: top-10 |gap| per day, 3 % subset, qty, slippage and costs", t_auc_ranking_cap_and_costs)
run("auction: protocol never simulates FINAL for screen failures", t_auc_protocol_final_never_for_failures)


section("AUCTION FADE LIVE FORWARD TEST")
import forward_test as _fwd
from pathlib import Path


def _fwd_q(ltp=0.0, open_=0.0, pc=100.0):
    return {"ltp": ltp, "open": open_, "prev_close": pc}


def t_fwd_trades_variants_rules():
    """Gap > 3 % up is faded short at the auction price; the live rule uses the indicative price; 1-3 % excludes > 3 %."""
    st = {"indicative_early": {"ABB": _fwd_q(100.0, 103.5), "ADANIENT": _fwd_q(100.0, 101.0),
                               "ADANIGREEN": _fwd_q(100.0, 97.5)},
          "auction": {}, "open": {"ABB": _fwd_q(104.0, 104.0), "ADANIENT": _fwd_q(102.5, 102.5),
                                  "ADANIGREEN": _fwd_q(96.5, 96.5)},
          "exit": {"ABB": _fwd_q(103.0), "ADANIENT": _fwd_q(102.0), "ADANIGREEN": _fwd_q(97.5)}}
    tr = _fwd.build_trades(st)
    f3 = tr["AUC-FADE-3PCT"]["final"]
    assert [t["symbol"] for t in f3] == ["ABB", "ADANIGREEN"], f3
    abb = f3[0]
    assert abb["side"] == "SELL" and abb["qty"] == 961 and abb["auction_px"] == 104.0
    assert tr["AUC-FADE-3PCT"]["live"][0]["qty"] == 966                                     # sized on 103.5
    s = _fwd.SLIPPAGE_BPS / 1e4
    assert abs(abb["entry_fill"] - round(104 * (1 - s), 2)) < 1e-6 and abs(abb["exit_fill"] - round(103 * (1 + s), 2)) < 1e-6
    from costs import trade_cost as _tc
    e, x = 104 * (1 - s), 103 * (1 + s)
    assert abs(abb["gross"] - round((e - x) * 961, 2)) < 0.01 and abs(abb["cost"] - _tc(e, x, 961, "SELL")) < 1e-6
    assert f3[1]["side"] == "BUY" and f3[1]["net"] == round(f3[1]["gross"] - f3[1]["cost"], 2)
    # live rule: ABB indicative 3.5 % → in; ADANIGREEN indicative -2.5 % → not in 3 %; ADANIENT +1 % → not in 3 %
    assert [t["symbol"] for t in tr["AUC-FADE-3PCT"]["live"]] == ["ABB"]
    assert [t["symbol"] for t in tr["AUC-FADE-1TO3PCT"]["final"]] == ["ADANIENT"]          # 2.5 % in, 3+ % out
    assert sorted(t["symbol"] for t in tr["AUC-FADE-1TO3PCT"]["live"]) == ["ADANIGREEN"]    # 1.0 % is not > 1 %


def t_fwd_cap_exclusions_missing():
    """At most 10 a day by |gap|; |gap| > 20 % and symbols missing a price are skipped; summary arithmetic."""
    syms = _fwd.UNIVERSE[:14]
    st = {"open": {s: _fwd_q(100 + 4 + i, 100 + 4 + i) for i, s in enumerate(syms)},
          "exit": {s: _fwd_q(100.0) for s in syms}}
    st["open"][syms[13]] = _fwd_q(125.0, 125.0)               # 25 % gap: excluded
    st["exit"].pop(syms[12])                                   # no exit price: skipped
    tr = _fwd.build_trades(st)["AUC-FADE-3PCT"]["final"]
    assert len(tr) == 10 and syms[13] not in [t["symbol"] for t in tr] and syms[12] not in [t["symbol"] for t in tr]
    assert tr[0]["symbol"] == syms[11] and all(t["side"] == "SELL" and t["net"] > 0 for t in tr)
    sm = _fwd.summarize({"2026-11-12": _fwd.build_trades(st)})["AUC-FADE-3PCT/final"]
    assert sm["trades"] == 10 and sm["days"] == 1 and sm["win_rate"] == 1.0 and abs(sm["net"] - sum(t["net"] for t in tr)) < 0.05
    q = _fwd.indicative_quality({"indicative_early": {"ABB": _fwd_q(100.0, 103.0), "ADANIENT": _fwd_q(100.0, 0.0)},
                                 "open": {"ABB": _fwd_q(104.0, 104.0), "ADANIENT": _fwd_q(103.0, 103.0)}})["indicative_early"]
    assert q["with_indicative"] == 1 and q["missing"] == 1 and q["gaps_over_2pct"] == 2 and q["caught_by_indicative"] == 1
    # no indicative (open 0, ltp = prev close) is missing, never a 0 % gap
    assert _fwd.indicative_price({"indicative_early": {"X": _fwd_q(100.0, 0.0)}}, "X") == 0.0
    import json as _j
    _j.dumps(_fwd.summarize({"d": _fwd.build_trades(st)}), allow_nan=False)                # all winners: still valid JSON


def t_fwd_record_persists_and_scheduler():
    """record() stores each stage and builds trades at exit in its own file; the scheduler has the four jobs."""
    import tempfile, kite_client as _kc
    from config import settings as _st
    tmp = Path(tempfile.mkdtemp()) / "fwd.json"
    old_file, old_q, old_src = _fwd.FILE, _kc.kite_client.quote_kite, _st.paper_data_source
    prices = {"indicative_early": 100.0, "indicative": 100.0, "auction": 104.0, "open": 104.0, "exit": 103.0}
    stage_now = {}
    _kc.kite_client.quote_kite = lambda ins: {"NSE:ABB": {"last_price": prices[stage_now["s"]],
                                                          "ohlc": {"open": 104.0,
                                                                   "close": 100.0}}}
    try:
        _fwd.FILE = tmp
        _st.paper_data_source = "KITE"
        for s in _fwd.STAGES:
            stage_now["s"] = s
            r = _fwd.record(s)
            assert r.get("symbols") == 1, r
        st = _fwd.status()
        assert st["summary"]["AUC-FADE-3PCT/final"]["trades"] == 1 and st["summary"]["AUC-FADE-3PCT/live"]["trades"] == 1
        assert set(st["today"]["stages_recorded"]) == set(_fwd.STAGES) and tmp.exists()
    finally:
        _fwd.FILE, _kc.kite_client.quote_kite, _st.paper_data_source = old_file, old_q, old_src
    import platform_scheduler as _ps
    src = Path(_ps.__file__).read_text()
    assert 'forward_{_stage}' in src and '("indicative_early", (9, 6, 45))' in src and '("exit", (9, 20, 0))' in src
    bad = Path(tempfile.mkdtemp()) / "fwd.json"
    bad.write_text("{not json")
    _fwd.FILE = bad
    try:
        assert _fwd._load() == {"days": {}} and not bad.exists() and list(bad.parent.glob("fwd.json.corrupt-*"))
    finally:
        _fwd.FILE = old_file


run("forward test: variants, final vs live rule, fills, slippage, costs", t_fwd_trades_variants_rules)
run("forward test: 10-a-day cap, 20 % exclusion, missing prices, summary", t_fwd_cap_exclusions_missing)
run("forward test: record() persists stages, exit builds trades; scheduler jobs", t_fwd_record_persists_and_scheduler)
section("STRATLAB ROUND-2 SCALPING: SCALP-AUC-FADE-1TO3-0925 (stratlab/auction_r2.py)")
# ══════════════════════════════════════════════════════════════════════════
from stratlab import auction_r2 as _slr2  # noqa: E402


def t_r2_signal_rule_and_features():
    assert _slr2.fade_rule(100, 101)[0] == -1 and _slr2.fade_rule(100, 99)[0] == 1     # 1 % inclusive, fade side
    assert _slr2.fade_rule(100, 103)[0] == -1 and _slr2.fade_rule(100, 97)[0] == 1     # 3 % inclusive
    assert _slr2.fade_rule(100, 100.99) is None and _slr2.fade_rule(100, 103.01) is None
    assert _slr2.fade_rule(0, 100) is None
    sd = _auc_sd(gaps=(0.0, 0.05, -0.02, 0.012, 0.03, -0.005))
    f = _slr2.session_features(sd)
    assert f[0][5] is True                                       # no previous session -> skipped
    for k in range(1, len(f)):
        assert abs(f[k][1] - float(sd.c[sd.ends[k - 1] - 1])) < 1e-9 and abs(f[k][2] - float(sd.o[sd.starts[k]])) < 1e-9
        assert f[k][3] == 565 and abs(f[k][4] - float(sd.o[sd.starts[k] + 10])) < 1e-9     # 09:25 open
    rows = _slr2.candidates_for("AUC", f)
    assert [(r[0], r[2]) for r in rows] == [(20260107, 1), (20260108, -1), (20260109, -1)], rows   # 5 %, 0.5 % out; 3 % in
    # a short (dropped) previous session -> skip the next day
    sd.dropped = [20260107]
    sd2 = _auc_sd(gaps=(0.0, 0.02, 0.02))
    sd2.dropped = []
    f2 = _slr2.session_features(sd2)
    assert len(_slr2.candidates_for("B", f2)) == 2
    sd2.days = np.array([20260105, 20260107, 20260108], dtype=np.int32)
    sd2.dropped = [20260106]
    assert [r[0] for r in _slr2.candidates_for("B", _slr2.session_features(sd2))] == [20260108]


def t_r2_no_lookahead():
    """Perturbing 09:16-09:24 of day d, everything after its 09:25 open, and all later days leaves day d's trade
    (signal, entry and exit prices) unchanged."""
    g = (0.0, 0.02, -0.015, 0.025)
    base = _slr2.candidates_for("AUC", _slr2.session_features(_auc_sd(gaps=g)))
    sd = _auc_sd(gaps=g)
    a = int(sd.starts[1])
    for arr in (sd.o, sd.h, sd.l, sd.c):
        arr[a + 1:a + 10] *= 1.04
        arr[a + 11:] *= 0.9
    pert = _slr2.candidates_for("AUC", _slr2.session_features(sd))
    assert pert[0] == base[0], (pert[0], base[0])


def t_r2_ranking_and_costs():
    rows = [(20260105, f"S{i:02d}", -1, 0.010 + i * 0.0015, 500.0, 565, 499.0) for i in range(14)]
    pick = _slr2.select(rows, _slr2.SPECS[0], 0, 99999999)
    # gaps 1.0 % .. 2.95 %: all 14 qualify, the 10 largest are S04..S13
    assert len(pick) == 10 and {r[1] for r in pick} == {f"S{i:02d}" for i in range(4, 14)}
    rows2 = rows + [(20260105, "S99", -1, 0.031, 500.0, 565, 499.0)]
    assert "S99" not in {r[1] for r in _slr2.select(rows2, _slr2.SPECS[0], 0, 99999999)}
    tr = _slr2.settle("X", rows[0], 2e-4)
    fill, ex = 500.0 * (1 - 2e-4), 499.0 * (1 + 2e-4)
    assert tr[9] == 200 and abs(tr[5] - fill) < 1e-3 and abs(tr[8] - ex) < 1e-3 and tr[7] == "09:25"
    from costs import round_trip_cost as _rtc
    assert abs(tr[11] - _rtc(ex * 200, fill * 200, "equity_mis")) < 1e-6
    mis = _rtc(1e5, 1e5, "equity_mis") / 1e5 * 1e4
    assert abs(mis - 8.27) < 0.15, mis                                           # pre-registered 8.27 bps
    assert abs(mis + 2 * 2.0 - 12.27) < 0.15 and abs(mis + 2 * 5.0 - 18.27) < 0.15
    assert _slr2.DEFAULT_SLIPPAGE_BPS == 2.0 and _slr2.COST_MODEL.startswith(_sls.COST_MODEL)


def t_r2_protocol_final_never_for_failures():
    seen = []
    rows = [(20240101 + (d % 28) + (d // 28) * 100, "A", 1, -0.02, 100.0, 565, 99.0) for d in range(200)]
    base = _slr2.make_simulator(rows)

    def sim_(specs, lo, hi):
        seen.append((lo, hi))
        return base(specs, lo, hi)
    cal = sorted({r[0] for r in rows})
    res, _tr = _slp.evaluate(_slr2.SPECS, cal, sim_, n_symbols=1, ledger_path=None, cost_model=_slr2.COST_MODEL)
    assert res[0]["verdict"] == "FAILED SCREEN" and len(seen) == 1


run("auction_r2: 1-3 % inclusive band, fade side, 09:25 exit, short-previous-session skip", t_r2_signal_rule_and_features)
run("auction_r2: later bars cannot change an earlier signal or fill (look-ahead)", t_r2_no_lookahead)
run("auction_r2: top-10 |gap| per day, qty, slippage and 8.27 / 12.27 / 18.27 bps costs", t_r2_ranking_and_costs)
run("auction_r2: protocol never simulates FINAL for screen failures", t_r2_protocol_final_never_for_failures)


section("STRATLAB ATTENTION OVERNIGHT LONG (stratlab/attention.py)")
# ══════════════════════════════════════════════════════════════════════════
from stratlab import attention as _slat  # noqa: E402


def _attn_sd(symbol="ATT", n_days=25, shock_day=22, up=True, seed=5, vol_mult=4.0):
    """One synthetic symbol, 375 bars/day, flat volume 1000/bar; day `shock_day` has `vol_mult` x volume and closes
    up (or down) vs the previous close by 15:20; the next day opens 1 % higher than that day's close."""
    rng = np.random.default_rng(seed)
    date, t, o, c, v = [], [], [], [], []
    px = 1000.0
    for k in range(n_days):
        op = px * (1.01 if k == shock_day + 1 else 1.0)
        drift = (0.0001 if up else -0.0001) if k == shock_day else 0.0
        cl = op * np.exp(np.cumsum(rng.normal(drift, 0.00005, 375)))
        oo = np.concatenate(([op], cl[:-1]))
        date += [20260105 + k] * 375; t += list(555 + np.arange(375)); o += list(oo); c += list(cl)
        v += [1000.0 * (vol_mult if k == shock_day else 1.0)] * 375
        px = float(cl[-1])
    o, c = np.array(o), np.array(c)
    return _sld.from_arrays(symbol, date, t, o, np.maximum(o, c), np.minimum(o, c), c, np.array(v))


def t_attn_rule():
    pv = [100.0] * 20
    assert _slat.attention_rule(301.0, pv, 101, 100) is not None          # 3.01x and up
    assert _slat.attention_rule(300.0, pv, 101, 100) is None              # exactly 3x is not "> 3x"
    assert _slat.attention_rule(400.0, pv, 100, 100) is None              # not up
    assert _slat.attention_rule(400.0, pv[:19], 101, 100) is None         # fewer than 20 sessions of history
    assert abs(_slat.attention_rule(500.0, [50.0] * 5 + pv, 101, 100) - 5.0) < 1e-9   # only the last 20 count
    sd = _attn_sd()
    rows = _slat.session_features(sd)
    assert len(rows) == 1 and rows[0]["date"] == 20260105 + 22, rows
    r = rows[0]
    a = int(sd.starts[22])
    assert r["entry_t"] == 15 * 60 + 21 and abs(r["entry_px"] - sd.o[a + 366]) < 1e-12      # 15:21 bar open
    assert abs(r["ratio"] - 4.0 * 366 / 375) < 1e-9                       # 366 bars to 15:20 at 4x / full-day mean
    a2 = int(sd.starts[23])
    assert r["exit_date"] == 20260105 + 23 and r["exit_t"] == 555 and abs(r["exit_px"] - sd.o[a2]) < 1e-12
    assert r["exit2_t"] == 556 and abs(r["exit2_px"] - sd.o[a2 + 1]) < 1e-12
    assert _slat.session_features(_attn_sd(up=False)) == []               # volume shock on a down day: no signal
    assert _slat.session_features(_attn_sd(vol_mult=3.0)) == []           # 3x/bar to 15:20 is < 3x a full day


def t_attn_no_lookahead():
    """Perturbing everything after the 15:21 entry bar of day t (except the next 09:15/09:16 bars) and all of the
    later days changes neither the signal nor the entry; changing the next 09:15 open changes only the exit."""
    sd = _attn_sd()
    base = _slat.session_features(sd)[0]
    a, a2 = int(sd.starts[22]), int(sd.starts[23])
    cut = a + 367                                                          # first bar after the 15:21 entry bar
    sd2 = _attn_sd()
    rng = np.random.default_rng(9)
    for arr in (sd2.o, sd2.h, sd2.l, sd2.c):
        arr[cut:a2] *= 1 + rng.normal(0, 0.05, a2 - cut)
        arr[a2 + 2:] *= 1 + rng.normal(0, 0.05, len(arr) - a2 - 2)
    sd2.v[cut:] *= 50.0
    r2 = [r for r in _slat.session_features(sd2) if r["date"] == base["date"]][0]
    for key in ("ratio", "entry_t", "entry_px", "exit_px", "exit2_px"):
        assert r2[key] == base[key], key
    sd3 = _attn_sd()
    sd3.o[a2] *= 1.02
    r3 = [r for r in _slat.session_features(sd3) if r["date"] == base["date"]][0]
    assert r3["ratio"] == base["ratio"] and r3["entry_px"] == base["entry_px"] and r3["exit_px"] != base["exit_px"]


def _attn_row(d, sym, ratio, nd=None, px=500.0, xp=505.0):
    return {"date": d, "symbol": sym, "ratio": ratio, "entry_t": 921, "entry_px": px, "c1520": px,
            "prev_close": px * 0.99, "exit_date": nd or d + 1, "exit_t": 555, "exit_px": xp,
            "exit2_t": 556, "exit2_px": xp}


def t_attn_exclusions_ranking_costs():
    row = _attn_row
    dates = np.array([20260105, 20260106, 20260107, 20260108], dtype=np.int64)
    assert _slat.eligible(row(20260105, "A", 4), dates, set())
    assert not _slat.eligible(row(20260105, "A", 4), dates, {20260106})                   # special t+1
    assert not _slat.eligible(row(20260105, "A", 4), dates, {20260105})                   # special t
    assert not _slat.eligible(row(20260105, "A", 4, nd=20260107), dates, set())          # session in between
    assert not _slat.eligible(row(20250404, "SIEMENS", 4, nd=20250407),
                              np.array([20250404, 20250407]), set())                     # CA gap on t+1
    assert not _slat.eligible({**row(20260105, "A", 4), "exit_date": None}, dates, set())
    cnt = {20260105: [375, 375, 375], 20260106: [375, 0, 200], 20260107: [250, 280, 375]}
    assert _slat.special_days(cnt, list(cnt)) == {20260106, 20260107}
    rows = [row(20260105, f"S{i:02d}", 3.0 + i) for i in range(14)] + [row(20260108, "Z", 9.0, nd=20260109)]
    pick = _slat.select(rows, _slat.SPECS[0], 20260101, 20260108)
    assert len(pick) == 10 and {r["symbol"] for r in pick} == {f"S{i:02d}" for i in range(4, 14)}
    assert all(r["symbol"] != "Z" for r in pick)                          # exit after the period end: excluded
    tr = _slat.settle("X", row(20260105, "A", 4), 2e-4)
    qty = 200
    fill, ex = 500.0 * 1.0002, 505.0 * 0.9998
    assert tr[3] == 1 and tr[9] == qty and abs(tr[5] - fill) < 1e-3 and abs(tr[8] - ex) < 1e-3 and tr[6] == 20260106
    from costs import round_trip_cost as _rtc
    assert abs(tr[10] - (ex - fill) * qty) < 0.01
    assert abs(tr[11] - _rtc(fill * qty, ex * qty, "future")) < 1e-6 and abs(tr[12] - (tr[10] - tr[11])) < 0.01
    fut = _rtc(1e5, 1e5, "future") / 1e5 * 1e4
    cnc = _rtc(1e5, 1e5, "equity_cnc") / 1e5 * 1e4
    assert 8 < fut < 13 and 20 < cnc < 30, (fut, cnc)                     # futures ~10 bps, CNC ~24 bps + slippage
    tc = _slat.settle("X", row(20260105, "A", 4), 2e-4, kind="equity_cnc")
    assert tc[11] > tr[11]


def t_attn_protocol_final_never_for_failures():
    seen = []
    days = [20240101 + (d % 28) + (d // 28) * 100 for d in range(200)]
    rows = [{**_attn_row(d, "A", 4.0, nd=d, px=100.0, xp=99.0)} for d in days]   # always loses
    base = _slat.make_simulator(rows)

    def sim_(specs, lo, hi):
        seen.append((lo, hi))
        return base(specs, lo, hi)
    res, _tr = _slp.evaluate(_slat.SPECS, sorted(set(days)), sim_, n_symbols=1, ledger_path=None,
                             cost_model=_slat.COST_MODEL)
    assert all(r["verdict"] == "FAILED SCREEN" for r in res) and len(seen) == 1, (seen, [r["why"] for r in res])
    assert _slat.SPECS[0].horizon == "positional" and _slat.DEFAULT_SLIPPAGE_BPS == 2.0
    assert _slat.COST_MODEL.startswith(_sls.COST_MODEL)


run("attention: signal rule, 15:20 volume / up-day, 15:21 entry, 09:15 and 09:16 exits", t_attn_rule)
run("attention: later bars cannot change the signal or entry (look-ahead)", t_attn_no_lookahead)
run("attention: exclusions, top-10 ratio per day, period end, futures / CNC costs", t_attn_exclusions_ranking_costs)
run("attention: protocol never simulates FINAL for screen failures", t_attn_protocol_final_never_for_failures)


section("VALIDATED AGENTS — AUC-FADE opening-auction fade (agents/validated_auction.py)")
# ══════════════════════════════════════════════════════════════════════════
import contextlib as _ctx_va
import copy as _copy_va
import json as _json_va
import stratlab.auction as _A_va
import validated_data as _vd
import agents.validated_auction as _va
import bot_state as _bs_va
import event_calendar as _ec_va
import performance as _perf_mod_va
import tick_engine as _te_mod_va
from agents.validated_auction import AuctionFadeAgent, AuctionInput, plan_auction
from agents.strategy_agents import ALL_AGENTS as _ALL_va
from costs import trade_cost as _tc_va
from ist_clock import _IST as _IST_va
from performance import performance as _perf_va
from sebi_compliance import sebi_compliance as _sebi_va, KillSwitchState as _KSS_va

_D_va = date(2026, 11, 13)                       # a Friday session; the previous one is Thursday 2026-11-12
_PIN_FINGERPRINT_va = "acce4b191d400eb4a0ccb30ab805e4250f8c709f"


def _va_at(d, hh, mm, ss=0):
    return datetime.combine(d, time(hh, mm, ss)).replace(tzinfo=_IST_va)


class _VaClock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _va_q(px, bid=None, ask=None, open_=None):
    return {"ltp": px, "open": px if open_ is None else open_, "prev_close": 0.0, "bid": bid or 0.0,
            "ask": ask or 0.0, "volume": 1000}


@_ctx_va.contextmanager
def _va_env(now, prev_closes: dict, quotes, agent=None, prev_day: int = 20261112):
    """PAPER with Kite data, a fixed IST clock (sleeps advance it), a store of previous sessions, fake quotes
    (dict or callable(symbols)), an empty paper book / ledger; everything put back afterwards."""
    clock = _VaClock(now)

    async def _adv(s):
        clock.t = clock.t + timedelta(seconds=s)

    st = settings
    saved = {k: getattr(st, k) for k in ("trading_mode", "paper_data_source", "validated_only",
                                         "validated_auction_enabled", "validated_auction_variant",
                                         "validated_notional_scale")}
    saved_en = _bs_va.is_agent_enabled("auc_fade")
    saved_book = (list(_te_kite_client._paper_orders), _copy_va.deepcopy(_te_kite_client._paper_positions))
    store = {"symbols": {s: [_vd.SessionSummary(prev_day, 375, 1e6, pc, 555, pc)] for s, pc in prev_closes.items()},
             "updated": None}
    qfn = quotes if callable(quotes) else (lambda ins: {s: quotes[s] for s in ins if s in quotes})
    a = agent or AuctionFadeAgent()
    patches = [_patch.object(_va, "now_ist", clock), _patch.object(_va, "_sleep", _adv),
               _patch.object(_pb, "now_ist", clock), _patch.object(_perf_mod_va, "now_ist", clock),
               _patch.object(_vd, "quotes", lambda ins: qfn(list(ins))), _patch.object(_vd, "_cache", store),
               _patch.object(_perf_va, "trades", []), _patch.object(_te_mod_va.tick_engine, "_latest_tick", {}),
               _patch.object(_sebi_va, "_state", _KSS_va.ACTIVE),
               _patch.object(_ec_va, "_price_actions", {}), _patch.object(_ec_va, "_loaded_at", now)]
    try:
        for p in patches:
            p.start()
        st.trading_mode, st.paper_data_source, st.validated_only = "PAPER", "KITE", True
        st.validated_auction_enabled, st.validated_auction_variant, st.validated_notional_scale = True, "3PCT", 1.0
        _bs_va.set_agent_enabled("auc_fade", True)
        _te_kite_client._paper_orders.clear(); _te_kite_client._paper_positions.clear()
        a._pos.clear(); a._day = {}; a.day_log = []; a._kill = None; a._dd_since = ""; a.deviations.clear()
        a.state.running = True
        yield a, clock
    finally:
        for p in reversed(patches):
            p.stop()
        for k, v in saved.items():
            setattr(st, k, v)
        _bs_va.set_agent_enabled("auc_fade", saved_en)
        _te_kite_client._paper_orders[:] = saved_book[0]
        _te_kite_client._paper_positions[:] = saved_book[1]
        a._pos.clear(); a._day = {}; a.day_log = []; a._kill = None; a._dd_since = ""
        a.state.running = False


def _va_orders(tag):
    return [o for o in _te_kite_client._paper_orders if o.get("tag") == tag]


def _va_row(sym):
    return next((p for p in _te_kite_client._paper_positions if p["tradingsymbol"] == sym
                 and p.get("product") == "MIS"), None)


# ABB +4 % (fade: SELL), ADANIENT −5 % (BUY), AMBUJACEM −6 % (BUY), ADANIGREEN +0.33 % (no signal),
# ADANIPORTS open missing, ADANIPOWER no previous session in the store
_VA_PC = {"ABB": 100.0, "ADANIENT": 200.0, "AMBUJACEM": 50.0, "ADANIGREEN": 300.0, "ADANIPORTS": 400.0}
_VA_OPEN = {"ABB": _va_q(104.0), "ADANIENT": _va_q(190.0), "AMBUJACEM": _va_q(47.0), "ADANIGREEN": _va_q(301.0),
            "ADANIPORTS": _va_q(401.0, open_=0.0), "ADANIPOWER": _va_q(520.0)}


def _lab_pick_va(inputs, spec_id):
    """What stratlab.auction.select picks from the same gaps (the lab's own code)."""
    rows = []
    for x in inputs:
        sig = _A_va.auction_rule(x.prev_close, x.auction_px, _va.LOOSEST_GAP)
        if sig:
            rows.append((20260101, x.symbol, sig[0], sig[1], x.auction_px, 560, x.auction_px))
    spec = next(s for s in _A_va.SPECS if s.id == spec_id)
    return [(r[1], r[2]) for r in _A_va.select(rows, spec, 0, 99999999)]


def t_va_constants_fingerprint():
    assert _va.A is _A_va, "rules are imported from the lab module, not copied"
    assert (_A_va.MAX_ABS_GAP, _A_va.MAX_PER_DAY, _A_va.NOTIONAL, _A_va.FILL_MODEL) == (0.20, 10, 100_000.0, "auction1")
    assert plan_auction.__defaults__ == (_A_va.NOTIONAL, _A_va.MAX_PER_DAY)
    assert _va.LAB_FINGERPRINT == _PIN_FINGERPRINT_va, \
        f"stratlab/auction.py changed ({_va.LAB_FINGERPRINT}): re-run the protocol, bump VERSION, re-pin"
    a = AuctionFadeAgent()
    assert a.VERSION == "AUC-FADE-3PCT/auction1/v1" and a.LAB_ID == "AUC-FADE-3PCT" and a.min_gap == 0.03
    assert _perf_mod_va._version("auc_fade") == a.VERSION
    assert tuple(_vd.UNIVERSE) == tuple(_fwd.UNIVERSE) and len(_vd.UNIVERSE) == 97
    assert a.scheduled and a.product == "MIS" and "auc_fade" in _bs_va.VALIDATED_AGENTS


def t_va_boundaries_match_lab_select():
    ins = [AuctionInput("EX3", 100.0, 103.0), AuctionInput("EX301", 100.0, 103.01),
           AuctionInput("DN301", 100.0, 96.99), AuctionInput("BIG", 100.0, 120.01),
           AuctionInput("AT20", 100.0, 120.0), AuctionInput("EX2", 100.0, 102.0), AuctionInput("ZERO", 0.0, 100.0)]
    got3 = [(p.symbol, p.side) for p in plan_auction(ins, 0.03)[0]]
    assert got3 == _lab_pick_va(ins, "AUC-FADE-3PCT"), (got3, _lab_pick_va(ins, "AUC-FADE-3PCT"))
    assert ("EX301", -1) in got3 and ("DN301", 1) in got3 and ("AT20", -1) in got3
    assert not {"BIG", "EX2", "ZERO"} & {s for s, _ in got3}
    # an exact 3.00 % gap: the lab's select() compares the float gap (103/100 − 1 = 0.030000000000000027) without
    # the epsilon auction_rule applies, so the lab TRADES it — parity, not the docstring, decides
    assert ("EX3" in {s for s, _ in got3}) == ("EX3" in {s for s, _ in _lab_pick_va(ins, "AUC-FADE-3PCT")})
    got2 = [(p.symbol, p.side) for p in plan_auction(ins, 0.02)[0]]
    assert got2 == _lab_pick_va(ins, "AUC-FADE-2PCT") and "EX2" not in {s for s, _ in got2}   # exact 2 %: not > 2 %
    rng = np.random.default_rng(7)
    for _ in range(300):                         # random days: identical to the lab's selection every time
        ins = [AuctionInput(f"S{i:02d}", 100.0, float(np.round(100 * (1 + rng.uniform(-0.25, 0.25)), 2)))
               for i in range(30)]
        for sid, mg in (("AUC-FADE-3PCT", 0.03), ("AUC-FADE-2PCT", 0.02)):
            assert [(p.symbol, p.side) for p in plan_auction(ins, mg)[0]] == _lab_pick_va(ins, sid)


def t_va_top10_ties_variant_qty():
    ins = [AuctionInput(f"T{i:02d}", 100.0, 105.0) for i in range(12)] + [AuctionInput("BIGGEST", 100.0, 90.0)]
    plan, skips = plan_auction(ins, 0.03)
    assert [p.symbol for p in plan] == ["BIGGEST"] + [f"T{i:02d}" for i in range(9)]
    assert plan[0].side == 1 and all(p.side == -1 for p in plan[1:])
    assert skips["T10"] == skips["T11"] == "not_top10"
    p = plan_auction([AuctionInput("Q", 1300.0, 1234.5)], 0.03)[0][0]
    assert p.qty == int(100_000 // 1234.5) == 81 and p.entry_t == "09:15" and p.exit_t == "09:20"
    assert plan_auction([AuctionInput("Q", 1300.0, 1234.5)], 0.03, notional=200_000)[0][0].qty == 162
    assert plan_auction([AuctionInput("Q", 210_000.0, 200_000.0)], 0.03)[0][0].qty == 1           # at least 1
    capped, f = _va.apply_gross_cap(plan, 500_000.0)
    assert f is not None and [c.symbol for c in capped] == [p.symbol for p in plan], "the cap never drops a name"
    assert sum(c.qty * c.entry_ref for c in capped) <= 500_000.0 + 1e-6
    a, old = AuctionFadeAgent(), settings.validated_auction_variant
    try:
        settings.validated_auction_variant = "2PCT"
        assert a.min_gap == 0.02 and a.VERSION == "AUC-FADE-2PCT/auction1/v1"
        assert a.LAB_EXPECTATION["FINAL"]["pf"] == 1.08
    finally:
        settings.validated_auction_variant = old


def t_va_entry_fills_at_open_full_size():
    """A signal reaches the paper broker at the tested size and product, filled exactly at the official open."""
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, _VA_OPEN) as (a, clock):
        r = asyncio.run(a.enter())
        assert r == {"entered": 3, "planned": 3}, r
        ins = {o["tradingsymbol"]: o for o in _va_orders("VAL-AUC-IN")}
        assert set(ins) == {"ABB", "ADANIENT", "AMBUJACEM"}
        for sym, side, px in (("ABB", "SELL", 104.0), ("ADANIENT", "BUY", 190.0), ("AMBUJACEM", "BUY", 47.0)):
            o = ins[sym]
            assert (o["transaction_type"], o["product"], o["price"], o["average_price"], o["fill_kind"],
                    o["quantity"], o["status"]) == (side, "MIS", px, px, "AUCTION", int(100_000 // px), "COMPLETE"), o
            assert _te_kite_client.paper_fill_vs_ltp(o["order_id"]) == 0.0
            assert _va_row(sym)["quantity"] == (1 if side == "BUY" else -1) * int(100_000 // px)
        assert len(a.positions()) == 3 and not _perf_va.trades, "nothing booked before the exit"
        log = {(e["symbol"], e["reason"]) for e in a.day_log}
        assert ("ADANIPORTS", "no_data:open") in log and ("ADANIPOWER", "no_data:prev_close") in log
        assert ("ADANIGREEN", "no_signal") in log and a.state.skip_counts.get("no_signal", 0) >= 1
        assert sum(v for k, v in a.state.skip_counts.items() if k.startswith("no_data")) >= 90   # each counted
        assert clock.t == _va_at(_D_va, 9, 15, 50), "three retries for the missing open, then decided"
        assert asyncio.run(a.enter())["skipped"] == "already_entered" and len(_va_orders("VAL-AUC-IN")) == 3
        assert a.day_state()["planned"] == [["AMBUJACEM", 1, 2127], ["ADANIENT", 1, 526], ["ABB", -1, 961]]
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, _VA_OPEN) as (a, clock):
        settings.validated_notional_scale = 2.0
        asyncio.run(a.enter())
        assert {o["tradingsymbol"]: o["quantity"] for o in _va_orders("VAL-AUC-IN")}["ABB"] == int(200_000 // 104.0)
    with _va_env(_va_at(_D_va, 9, 16, 1), _VA_PC, _VA_OPEN) as (a, clock):
        assert asyncio.run(a.enter())["skipped"] == "entry_window_missed" and not _va_orders("VAL-AUC-IN")
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, {}) as (a, clock):
        assert asyncio.run(a.enter())["skipped"] == "quote_failed" and a.state.skip_counts["quote_failed"] == 1
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, _VA_OPEN) as (a, clock):
        _ec_va._price_actions = {"ABB": [(_D_va, "Face Value Split (Sub-Division) From Rs 2 To Rs 1")]}
        asyncio.run(a.enter())
        assert "ABB" not in {o["tradingsymbol"] for o in _va_orders("VAL-AUC-IN")}
        assert a.state.skip_counts.get("corporate_action") == 1
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, _VA_OPEN) as (a, clock):
        _sebi_va._state = _KSS_va.KILLED
        assert asyncio.run(a.enter())["entered"] == 0 and a.state.skip_counts.get("sebi") == 3


def t_va_exit_at_touch_retry_late_ledger():
    """09:20 exit at the quote touch; a symbol quoted late is retried every 10 s; one never quoted is exited at
    its last price at 09:29:50 (late_exit). Booked once each in the ledger at equity_mis costs, outside risk."""
    def quotes(ins):
        t = clock_ref[0].t.time()
        out = {"ABB": _va_q(103.0, 102.9, 103.1)}
        if t >= time(9, 20, 20):
            out["ADANIENT"] = _va_q(195.0, 194.9, 195.1)
        return {s: out[s] for s in ins if s in out} if t >= time(9, 20) else {s: _VA_OPEN[s] for s in ins if s in _VA_OPEN}
    clock_ref = []
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, quotes) as (a, clock):
        clock_ref.append(clock)
        asyncio.run(a.enter())
        oids = {p.symbol: p.oid for p in a.positions()}
        clock.t = _va_at(_D_va, 9, 20, 0)
        r = asyncio.run(a.exit())
        assert r == {"closed": 3, "open": 0}, r
        rows = {t["symbol"]: t for t in _perf_va.trades}
        assert set(rows) == {"ABB", "ADANIENT", "AMBUJACEM"} and len(_perf_va.trades) == 3
        abb, ade, amb = rows["ABB"], rows["ADANIENT"], rows["AMBUJACEM"]
        assert (abb["side"], abb["entry"], abb["exit"], abb["qty"], abb["oid"]) == ("SELL", 104.0, 103.1, 961, oids["ABB"])
        assert abs(abb["gross"] - round((104.0 - 103.1) * 961, 2)) < 0.01
        assert abb["cost"] == _tc_va(104.0, 103.1, 961, "SELL", "NSE", "MIS", "ABB")          # equity_mis, as the lab
        assert abb["version"] == a.VERSION and abb["note"] == "09:20 exit" and abb["exit_kind"] == "normal"
        assert (ade["exit"], ade["exit_kind"]) == (194.9, "normal"), "quoted at 09:20:20: SELL at the bid"
        assert (amb["exit"], amb["exit_kind"]) == (47.0, "late_exit"), "never quoted: last price at 09:29:50"
        assert a.deviations == {"late_exit": 1} and clock.t >= _va_at(_D_va, 9, 29, 50)
        assert all(_va_row(s)["quantity"] == 0 for s in rows), "paper rows flat"
        assert abs(a.state.pnl_today - sum(t["net"] for t in _perf_va.trades)) < 0.05
        assert a.realised_today() == round(sum(t["net"] for t in _perf_va.trades), 2)
        assert {o["tradingsymbol"]: o["transaction_type"] for o in _va_orders("VAL-AUC-OUT")} == \
            {"ABB": "BUY", "ADANIENT": "SELL", "AMBUJACEM": "SELL"}
        assert asyncio.run(a.exit()) == {"closed": 0, "open": 0} and len(_perf_va.trades) == 3


def t_va_squareoff_flattened_reconciled():
    """An auction position still open at 15:15 is flattened by the broker square-off (it is not an overnight
    holding); the agent books it once at the row's last price as a counted deviation."""
    with _va_env(_va_at(_D_va, 9, 15, 5), {"ABB": 100.0}, {"ABB": _va_q(104.0)}) as (a, clock):
        asyncio.run(a.enter())
        from agents.base_agent import overnight_hold_qty
        assert ("ABB", "MIS") not in overnight_hold_qty() and a.overnight_holdings() == {}
        _te_kite_client.squareoff_all_positions()
        assert _va_row("ABB")["quantity"] == 0
        clock.t = _va_at(_D_va, 15, 16)
        out = asyncio.run(a.catch_up())
        assert out == {} and not a.positions() and len(_perf_va.trades) == 1
        assert _perf_va.trades[0]["note"] == "flattened_by_squareoff" and a.deviations == {"flattened_by_squareoff": 1}


def t_va_live_refused_kill_switch():
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, _VA_OPEN) as (a, clock):
        settings.trading_mode = "LIVE"
        assert asyncio.run(a.enter())["skipped"] == "live_mode_refused" and not _va_orders("VAL-AUC-IN")
        try:
            _te_kite_client.paper_fill_at("ABB", "NSE", "BUY", 1, "MIS", 100.0, "X", "AUCTION")
            raise AssertionError("paper_fill_at must refuse LIVE")
        except RuntimeError:
            pass
        settings.trading_mode = "PAPER"
        settings.validated_auction_enabled = False
        assert asyncio.run(a.enter())["skipped"] == "kill_switch" and not _va_orders("VAL-AUC-IN")
        settings.validated_auction_enabled = True
        _bs_va.set_agent_enabled("auc_fade", False)
        assert asyncio.run(a.enter())["skipped"] == "disabled"
        _bs_va.set_agent_enabled("auc_fade", True)
        settings.validated_only = False
        assert asyncio.run(a.enter())["skipped"] == "validated_only_off"
        settings.validated_only = True
        a.state.running = False
        assert asyncio.run(a.enter())["skipped"] == "stopped"
        a.state.running = True
        settings.paper_data_source = "SIM"
        assert asyncio.run(a.enter())["skipped"] == "no_kite_data"
        settings.paper_data_source = "KITE"
        assert asyncio.run(a.enter())["entered"] == 3
        settings.validated_auction_enabled = False            # the kill switch blocks entries only
        _bs_va.set_agent_enabled("auc_fade", False)
        clock.t = _va_at(_D_va, 9, 20)
        _vd.quotes = lambda ins: {s: _va_q(100.0, 99.9, 100.1) for s in ins}
        assert asyncio.run(a.exit())["closed"] == 3 and not a.positions() and len(_perf_va.trades) == 3


def t_va_daily_loss_stop_and_drawdown_kill():
    def row(day, net):
        return {"t": f"{day}T09:20:00+05:30", "strategy": "auc_fade", "version": "AUC-FADE-3PCT/auction1/v1",
                "symbol": "X", "side": "BUY", "qty": 1, "entry": 100.0, "exit": 100.0, "gross": net, "cost": 0.0,
                "net": net}
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, _VA_OPEN) as (a, clock):
        _perf_va.trades[:] = [row("2026-11-13", -16_000.0)]
        assert a.entry_block_reason() == "daily_loss_stop"
        assert asyncio.run(a.enter())["skipped"] == "daily_loss_stop" and not _va_orders("VAL-AUC-IN")
        settings.validated_notional_scale = 2.0                               # limits scale with the notional
        assert a.entry_block_reason() is None
        settings.validated_notional_scale = 1.0
        _perf_va.trades[:] = [row("2026-11-11", 10_000.0), row("2026-11-12", -47_000.0)]  # dd 47k ≥ 37k
        assert a.get_status()["entry_block"] == "drawdown_kill" and _bs_va.is_agent_enabled("auc_fade"), \
            "status is read-only"
        assert asyncio.run(a.enter())["skipped"] == "drawdown_kill" and not _bs_va.is_agent_enabled("auc_fade")
        assert a._kill and a.entry_block_reason() == "drawdown_kill" and not _va_orders("VAL-AUC-IN")
        _bs_va.set_agent_enabled("auc_fade", True)                            # manual re-enable
        assert a.entry_block_reason() is None and a._kill is None and a.drawdown() == 0.0
        assert asyncio.run(a.enter())["entered"] == 3
        assert a.export_book()["dd_since"] == _va_at(_D_va, 9, 15, 5).isoformat(timespec="seconds")


def t_va_restart_same_day_and_stale():
    """paper_book round trips: a same-day restart keeps the positions without a second entry and books the
    09:20 exit once; a restart from an earlier snapshot whose ledger already has the exit does not book it again;
    an earlier session's snapshot with auction positions still open books each missed exit once."""
    a = _ALL_va["auc_fade"]
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, _VA_OPEN, agent=a) as (a, clock), _pb_isolated() as f:
        _te_kite_client._paper_orders.clear(); _te_kite_client._paper_positions.clear()
        asyncio.run(a.enter())
        oids = sorted(p.oid for p in a.positions())
        assert _pb.save(force=True)
        snap = _json_va.loads(f.read_text())
        assert sorted(p["oid"] for p in snap["validated"]["auc_fade"]["positions"]) == oids
        before_exit = f.read_text()
        # restart at 09:15:40 (still inside the entry window)
        _pb_wipe(); a._pos.clear(); a._day = {}; a.day_log = []
        clock.t = _va_at(_D_va, 9, 15, 40)
        out = _pb.restore()
        assert out["restored"] and sorted(p.oid for p in a.positions()) == oids, out
        assert a.day_state().get("entry_done") and all(_va_row(s)["quantity"] for s in ("ABB", "ADANIENT", "AMBUJACEM"))
        asyncio.run(a.catch_up(clock.t))
        assert len(_va_orders("VAL-AUC-IN")) == 3, "no second entry after the restart"
        clock.t = _va_at(_D_va, 9, 20)
        _vd.quotes = lambda ins: {s: _va_q(100.0, 99.9, 100.1) for s in ins}
        asyncio.run(a.exit())
        assert len(_perf_va.trades) == 3 and sorted(t["oid"] for t in _perf_va.trades) == oids
        # crash: the last snapshot on disk is the one from before the exit
        f.write_text(before_exit)
        _pb_wipe(); a._pos.clear(); a._day = {}
        _pb.restore()
        assert not a.positions() and len(_perf_va.trades) == 3, "already in the ledger: not restored, not re-booked"
        assert all(_va_row(s)["quantity"] == 0 for s in ("ABB", "ADANIENT", "AMBUJACEM")), "paper rows reconciled"

    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, _VA_OPEN, agent=a) as (a, clock), _pb_isolated() as f:
        _te_kite_client._paper_orders.clear(); _te_kite_client._paper_positions.clear()
        asyncio.run(a.enter())
        assert _pb.save(force=True)
        stale = f.read_text()
        # the process stayed down until Monday 09:05
        _pb_wipe(); a._pos.clear(); a._day = {}
        clock.t = _va_at(date(2026, 11, 16), 9, 5)
        out = _pb.restore()
        assert out["restored"] and out["stale"], out
        assert not a.positions() and len(_perf_va.trades) == 3
        assert all(t["note"] == "missed_exit 2026-11-13" for t in _perf_va.trades)
        assert {t["symbol"]: t["exit"] for t in _perf_va.trades} == {"ABB": 104.0, "ADANIENT": 190.0, "AMBUJACEM": 47.0}
        assert a.deviations.get("missed_exit") == 3
        assert all(_va_row(s)["quantity"] == 0 for s in ("ABB", "ADANIENT", "AMBUJACEM"))
        f.write_text(stale)                                   # restored again from the same old snapshot
        _pb_wipe(); a._pos.clear()
        _pb.restore()
        assert not a.positions() and len(_perf_va.trades) == 3, "a missed exit is booked once"
        assert all(_va_row(s)["quantity"] == 0 for s in ("ABB", "ADANIENT", "AMBUJACEM"))


def t_va_status_parity_audit_scheduler():
    with _va_env(_va_at(_D_va, 9, 15, 5), _VA_PC, _VA_OPEN) as (a, clock):
        asyncio.run(a.enter())
        st = a.get_status()
        _json_va.dumps(st, default=str)
        assert st["lab_id"] == "AUC-FADE-3PCT" and st["fingerprint"] == _PIN_FINGERPRINT_va and len(st["positions"]) == 3
        assert st["lab_expectation"]["DISC"]["net_bps"] == 46.63 and st["live_readiness"]["ready"] is False
        assert st["candidates_today"][0]["symbol"] == "AMBUJACEM"
        # 15:40: the day's bars agree with the quotes → match; a drifted bar open → counted mismatch
        for s, pc in _VA_PC.items():
            o = _VA_OPEN[s]["open"]
            _vd._cache["symbols"][s].append(_vd.SessionSummary(20261113, 375, 1e6, o or pc, 555, o or pc))
        assert a.parity_audit()["match"] and not a.deviations
        _vd._cache["symbols"]["ABB"][-1] = _vd.SessionSummary(20261113, 375, 1e6, 102.0, 555, 102.0)   # +2 %: no
        assert not a.parity_audit()["match"] and a.deviations == {"parity_mismatch": 1}
    import platform_scheduler as _ps_va
    src = Path(_ps_va.__file__).read_text()
    for job in ('"validated_open", (9, 15, 5)', '"validated_auc_exit", (9, 20, 0)', '"validated_store", (15, 40, 0)',
                '"validated_preflight", (8, 55, 0)', 'id="validated_catch_up"'):
        assert job in src, job
    assert not hasattr(_perf_va, "_evaluate_pruning"), "no automatic pruning in the rebuilt ledger"
    for p in (Path(_va.__file__), Path(_va.__file__).with_name("validated_base.py"), Path(_vd.__file__)):
        assert "send_telegram" not in p.read_text(), f"{p.name}: no Telegram from validated code"


run("AUC-FADE: lab constants imported, fingerprint pinned, VERSION",                  t_va_constants_fingerprint)
run("AUC-FADE: thresholds / 20 % cap / sides identical to stratlab.auction.select",   t_va_boundaries_match_lab_select)
run("AUC-FADE: top 10 by |gap| (ties by symbol), qty, gross cap, 3PCT/2PCT switch",   t_va_top10_ties_variant_qty)
run("AUC-FADE: entry filled at the official open, MIS, full size; skips counted",     t_va_entry_fills_at_open_full_size)
run("AUC-FADE: 09:20 exit at the touch, retry, late_exit; ledger once, equity_mis",   t_va_exit_at_touch_retry_late_ledger)
run("AUC-FADE: a position flattened at 15:15 is reconciled once",                     t_va_squareoff_flattened_reconciled)
run("AUC-FADE: LIVE refused; kill switch / disable / stop block entries, not exits", t_va_live_refused_kill_switch)
run("AUC-FADE: daily loss stop and drawdown kill (scaled, manual re-enable)",          t_va_daily_loss_stop_and_drawdown_kill)
run("AUC-FADE: paper_book restart same-day / crash / stale: booked once",              t_va_restart_same_day_and_stale)
run("AUC-FADE: status, 15:40 parity audit, scheduler jobs, no Telegram",              t_va_status_parity_audit_scheduler)


# ── review fixes (AUC-FADE): stale store, overlapping job + catch_up ──
import time as _time_rf


def t_va_refuses_stale_store():
    """Store's newest session 11-11, today Fri 11-13 (Thu 11-12 missing): the gap would be measured against a
    2-session-old close — no entry, counted as no_data:store_stale, entry marked done."""
    q = {"ABB": _va_q(104.0)}
    with _va_env(_va_at(_D_va, 9, 15, 5), {"ABB": 100.0}, q, prev_day=20261111) as (a, clock):
        assert not _vd.store_current(_D_va) and _vd.prev_close("ABB", _D_va) == 100.0
        out = asyncio.run(a.enter(q))
        assert out == {"entered": 0, "skipped": "no_data:store_stale"}, out
        assert not a.positions() and not _va_orders("VAL-AUC-IN")
        assert a.state.skip_counts.get("no_data:store_stale") == 1 and a.day_state().get("entry_done")


def t_va_job_and_catch_up_serialised():
    """The 09:15:05 job (in its missing-open retry loop, slow Kite quotes) and a restart's catch_up overlap:
    one basket, one order per name; the 09:20 exit and a catch_up exit overlap: one SELL/BUY per position."""
    pcs = {"ABB": 100.0, "ADANIENT": 200.0, "ADANIPORTS": 400.0}

    def slow(ins):
        _time_rf.sleep(0.1)
        return {s: v for s, v in {"ABB": _va_q(104.0), "ADANIENT": _va_q(190.0),
                                  "ADANIPORTS": _va_q(401.0, open_=0.0)}.items() if s in ins}
    with _va_env(_va_at(_D_va, 9, 15, 20), pcs, slow) as (a, clock):
        async def both():
            return await asyncio.gather(a.enter(), a.catch_up(clock.t))
        asyncio.run(both())
        orders = _va_orders("VAL-AUC-IN")
        assert len(orders) == 2 and len(a.positions()) == 2, [(o["tradingsymbol"], o["quantity"]) for o in orders]
        assert abs(_va_row("ABB")["quantity"]) == int(100_000 // 104.0)
        clock.t = _va_at(_D_va, 9, 20, 0)
        _vd.quotes = lambda ins: (_time_rf.sleep(0.1), {s: _va_q(100.0, 99.9, 100.1) for s in ins})[1]

        async def both_exit():
            return await asyncio.gather(a.exit(), a.catch_up(clock.t))
        asyncio.run(both_exit())
        assert len(_va_orders("VAL-AUC-OUT")) == 2 and not a.positions()
        assert _va_row("ABB")["quantity"] == 0 and _va_row("ADANIENT")["quantity"] == 0
        assert len(_perf_va.trades) == 2


run("AUC-FADE: no entry on a stale session store (no_data:store_stale)", t_va_refuses_stale_store)
run("AUC-FADE: scheduled job and restart catch_up serialised: one entry and one exit per name",
    t_va_job_and_catch_up_serialised)



# ── Lab-parity acceptance test (real 3-year bars + the lab's FINAL trade lists) ─────────────────────────────
_PARITY_BARS_va = _os.getenv("VALIDATED_PARITY_BARS",
                             "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad/bars3y")
_PARITY_LAB_va = _os.getenv("VALIDATED_PARITY_LAB",
                            "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad/dd")
_AUC_SPEC_DAYS_va = (20231204, 20231207, 20250922, 20260309, 20261026, 20261105)
_AUC_2PCT_DAYS_va = (20231204, 20260204)


def _va_hm(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _va_lab_rows(path):
    import csv as _csv
    with open(path) as fh:
        return list(_csv.DictReader(fh))


def _va_parity_load(days_needed):
    """Per symbol, from the lab's npz cache (I/O only): every sample day's session plus the previous sessions back
    to a valid one, as Kite-candle dicts through validated_data.summarize_sessions (the live store builder), and
    the day's exit bar (first bar at/after 09:20, before 09:30). Also the full-history scan of the known
    look-ahead divergences, with the lab's own session_features."""
    from stratlab import data as _D
    man = _D.prepare(_PARITY_BARS_va)
    cdir = man["cache_dir"]
    syms = sorted(man["symbols"])
    store = {d: {} for d in days_needed}
    opens = {d: {} for d in days_needed}
    exits = {d: {} for d in days_needed}
    noexit, short, cal, cands = [], [], set(), []
    for s in syms:
        with np.load(_os.path.join(cdir, _D._cache_name(s))) as z:
            raw = {k: z[k] for k in z.files}
        starts, ends = _D._bounds(raw["date"])
        rdays = raw["date"][starts]
        nbars = ends - starts
        sd = _D.load_symbol(cdir, s)
        cal.update(int(x) for x in sd.days)
        for d in days_needed:
            k = int(np.searchsorted(rdays, d))
            idx = []
            j = k - 1
            while j >= 0:                                  # previous sessions back to the last valid one
                idx.append(j)
                if nbars[j] >= _D.MIN_SESSION_BARS or len(idx) >= 10:
                    break
                j -= 1
            if k < len(rdays) and rdays[k] == d:
                idx.append(k)
                a_, b_ = int(starts[k]), int(ends[k])
                tt = raw["t"][a_:b_]
                jj = int(np.searchsorted(tt, _A_va.EXIT_T))
                if jj < len(tt) and int(tt[jj]) < _A_va.EXIT_LATEST:
                    exits[d][s] = (int(tt[jj]), float(raw["o"][a_ + jj]))
            rows = []
            for j in sorted(idx):
                a_, b_ = int(starts[j]), int(ends[j])
                dd = int(rdays[j])
                base = datetime(dd // 10000, dd // 100 % 100, dd % 100)
                for i in range(a_, b_):
                    rows.append({"date": base + timedelta(minutes=int(raw["t"][i])), "open": float(raw["o"][i]),
                                 "high": float(raw["h"][i]), "low": float(raw["l"][i]), "close": float(raw["c"][i]),
                                 "volume": float(raw["v"][i])})
            summ = _vd.summarize_sessions(rows)
            prev = [x for x in summ if x.day < d]
            if prev:
                store[d][s] = prev
            today = [x for x in summ if x.day == d]
            if today:
                opens[d][s] = today[0].first_open if today[0].first_t == _A_va.AUCTION_T else 0.0
        # look-ahead divergences over all 3 years: a signal whose exit bar is missing (lab drops it before
        # ranking) and a signal on a session the lab drops for being short (< 200 bars)
        feats = _A_va.session_features(sd)
        cands.extend(_A_va.candidates_for(s, feats, _va.LOOSEST_GAP))
        for date_, pc, apx, xt, xp in feats:
            if np.isfinite(apx) and not np.isfinite(xp):
                sig = _A_va.auction_rule(pc, apx, _va.LOOSEST_GAP)
                if sig:
                    noexit.append((date_, s, sig[0], sig[1], apx, -1, float("nan")))
        for dd in sd.dropped or []:
            k = int(np.searchsorted(rdays, dd))
            if raw["t"][int(starts[k])] != _A_va.AUCTION_T:
                continue
            prev_valid = [x for x in range(len(sd.days)) if sd.days[x] < dd]
            if not prev_valid:
                continue
            pc = float(sd.c[int(sd.ends[prev_valid[-1]]) - 1])
            sig = _A_va.auction_rule(pc, float(raw["o"][int(starts[k])]), _va.LOOSEST_GAP)
            if sig:
                short.append((dd, s, sig[0], sig[1], float(raw["o"][int(starts[k])]), -1, float("nan")))
        del raw, sd
    return syms, store, opens, exits, noexit, short, sorted(cal), cands


def _va_divergent_days(cands, extra, spec_id):
    """Days where adding the candidates the live agent cannot exclude in advance (missing exit bar / short
    session) changes the lab's own selection (stratlab.auction.select on the lab's own candidate rows)."""
    spec = next(x for x in _A_va.SPECS if x.id == spec_id)
    lab = {}
    for r in _A_va.select(cands, spec, 0, 99999999):
        lab.setdefault(r[0], []).append((r[1], r[2]))
    live = {}
    for r in _A_va.select(cands + extra, spec, 0, 99999999):
        live.setdefault(r[0], []).append((r[1], r[2]))
    return {d for d in set(lab) | set(live) if sorted(lab.get(d, [])) != sorted(live.get(d, []))}, lab


import bisect as _bisect_auc
_VA_LAB_CAL_auc: list = [[]]


def _va_run_day(d, store, opens, exits, ca, variant):
    """The live agent's own enter() / exit() on day d from Kite-shaped data. Returns [(sym, side, '09:15', exit_t,
    qty, entry_fill, exit_fill)]."""
    def quotes(ins):
        if clock_ref[0].t.time() < time(9, 20):
            return {s: _va_q(opens[s], open_=opens[s]) for s in ins if s in opens}
        return {s: _va_q(exits[s][1], exits[s][1], exits[s][1]) for s in ins if s in exits}
    clock_ref = []
    day = date(d // 10000, d // 100 % 100, d % 100)
    # the store-current guard reads the previous trading day; the lab's calendar is its data's session list
    # (ist_clock's holiday table starts in 2026)
    cal = _VA_LAB_CAL_auc[0]
    i = _bisect_auc.bisect_left(cal, d)
    lab_prev = cal[i - 1] if i > 0 else d
    with _va_env(_va_at(day, 9, 15, 5), {}, quotes) as (a, clock), \
            _patch.object(_vd, "previous_trading_day",
                          lambda _d: date(lab_prev // 10000, lab_prev // 100 % 100, lab_prev % 100)):
        clock_ref.append(clock)
        settings.validated_auction_variant = variant
        _vd._cache["symbols"] = dict(store)
        _ec_va._price_actions = ca
        asyncio.run(a.enter())
        ins = {o["tradingsymbol"]: o for o in _va_orders("VAL-AUC-IN")}
        assert all(o["product"] == "MIS" and o["fill_kind"] == "AUCTION" for o in ins.values())
        clock.t = _va_at(day, 9, 20, 0)
        asyncio.run(a.exit())
        outs = {o["tradingsymbol"]: o for o in _va_orders("VAL-AUC-OUT")}
        assert not a.positions() and len(_perf_va.trades) == len(ins), "every entry booked exactly once"
        res = []
        for s, o in sorted(ins.items()):
            side = 1 if o["transaction_type"] == "BUY" else -1
            res.append((s, side, "09:15", _va_hm(exits[s][0]), o["quantity"], o["price"], outs[s]["price"]))
        return res


def t_va_lab_parity():
    import os as _o
    f3 = _o.path.join(_PARITY_LAB_va, "auction_final", "trades_AUC-FADE-3PCT.csv")
    f2 = _o.path.join(_PARITY_LAB_va, "auction_final", "trades_AUC-FADE-2PCT.csv")
    if not (_o.path.isdir(_PARITY_BARS_va) and _o.path.exists(f3) and _o.path.exists(f2)):
        print("     SKIP (no lab data)")
        return "skip"
    from stratlab.ltr import CA_GAPS
    lab3, lab2 = _va_lab_rows(f3), _va_lab_rows(f2)
    days3 = sorted({int(r["date"]) for r in lab3})
    days2 = sorted({int(r["date"]) for r in lab2})
    # sample: the spec days + every k-th 3PCT trading day (≥ 40 days with signals) + 2PCT days + empty days
    k = max(1, len(days3) // 44)
    sample3 = sorted(set(_AUC_SPEC_DAYS_va) | set(days3[::k]))
    sample2 = sorted(set(_AUC_2PCT_DAYS_va) | set(days2[5::max(1, len(days2) // 10)]))
    # need the calendar for the empty days first: a cheap pass over one load is part of _va_parity_load
    needed = sorted(set(sample3) | set(sample2) | {20240101, 20240102, 20240103, 20240104, 20240105})
    syms, store, opens, exits, noexit, short, cal, cands = _va_parity_load(needed)
    assert set(syms) == set(_vd.UNIVERSE), "the live universe is the lab's"
    _VA_LAB_CAL_auc[0] = sorted(cal)
    empty = [d for d in cal if d >= 20240101 and d not in set(days3)][:2]
    assert set(empty) <= set(needed), empty
    ca = {}
    for (s, dd), why in CA_GAPS.items():
        ca.setdefault(s, []).append((date(dd // 10000, dd // 100 % 100, dd % 100), why))
    assert not [r for r in lab2 if (r["symbol"], int(r["date"])) in CA_GAPS], "the lab never traded a CA gap"
    div3, sel3 = _va_divergent_days(cands, noexit + short, "AUC-FADE-3PCT")
    div2, sel2 = _va_divergent_days(cands, noexit + short, "AUC-FADE-2PCT")
    # the lab's code re-run on the same bars reproduces its FINAL CSVs (so the CSVs are the reference)
    for lab, sel in ((lab3, sel3), (lab2, sel2)):
        csv_days = {}
        for r in lab:
            csv_days.setdefault(int(r["date"]), []).append((r["symbol"], int(r["side"])))
        assert {d: sorted(v) for d, v in csv_days.items()} == {d: sorted(v) for d, v in sel.items()}
    print(f"     known look-ahead divergence over 3 years: {len(noexit)} signals with no 09:20-09:29 exit bar "
          f"{[(e[0], e[1], round(e[3] * 100, 2)) for e in noexit]}, "
          f"short sessions {[(e[0], e[1], round(e[3] * 100, 2)) for e in short]}; "
          f"{len(short)} signals on short (<200 bar) sessions → 3PCT days affected {sorted(div3)}, "
          f"2PCT days affected {sorted(div2)}")
    assert len(div3) <= 0.02 * len(days3) and len(div2) <= 0.02 * len(days2), "known divergence must stay rare"
    s = _A_va.DEFAULT_SLIPPAGE_BPS / 1e4
    report = {}
    for variant, lab, sample in (("3PCT", lab3, sample3 + empty), ("2PCT", lab2, sample2)):
        matched = total = days = sig_days = 0
        for d in sample:
            if d in (div3 if variant == "3PCT" else div2):
                continue                                     # documented divergence, counted above
            want = sorted((r["symbol"], int(r["side"]), r["entry_t"], r["exit_t"], int(r["qty"]),
                           float(r["entry"]), float(r["exit"])) for r in lab if int(r["date"]) == d)
            got = _va_run_day(d, store[d], opens[d], exits[d], ca, variant)
            days += 1
            sig_days += bool(want)
            total += len(want)
            assert [g[:5] for g in got] == [w[:5] for w in want], (variant, d, got, want)
            for g, w in zip(got, want):
                side = g[1]
                assert abs(round(g[5] * (1 + side * s), 4) - w[5]) <= 1e-3, ("entry", d, g, w)
                assert abs(round(g[6] * (1 - side * s), 4) - w[6]) <= 1e-3, ("exit", d, g, w)
                matched += 1
        report[variant] = (days, sig_days, matched, total)
        print(f"     AUC-FADE-{variant}: {days} days compared ({sig_days} with signals), "
              f"trades matched {matched}/{total}")
    assert report["3PCT"][1] >= 40, report
    for d in empty:
        assert not [r for r in lab3 if int(r["date"]) == d]
    return report


def t_va_lab_parity_or_skip():
    r = t_va_lab_parity()
    if r == "skip":
        raise _VaSkip()


class _VaSkip(Exception):
    pass


try:
    t_va_lab_parity_or_skip()
    ok("AUC-FADE: lab parity on real 3-year bars (live enter/exit vs FINAL trade CSVs)")
except _VaSkip:
    print("  ⏭️   AUC-FADE: lab parity — SKIPPED (no lab data; set VALIDATED_PARITY_BARS / VALIDATED_PARITY_LAB)")
except Exception as exc:
    fail("AUC-FADE: lab parity on real 3-year bars (live enter/exit vs FINAL trade CSVs)", str(exc)[:600])


# ══════════════════════════════════════════════════════════════════════════
# VALIDATED AGENTS — SWING-ATTN-ON-3X (agents/validated_attention.py)
# ══════════════════════════════════════════════════════════════════════════
section("VALIDATED AGENTS — SWING-ATTN-ON-3X attention overnight long (agents/validated_attention.py)")
import contextlib as _ctx_va
import validated_data as _VD
import agents.validated_attention as _VA
import agents.validated_base as _VB
import bot_state as _bs_va
from stratlab import attention as _T_va
from costs import round_trip_cost as _rtc_va

_DA = date(2026, 11, 11)                      # Wednesday; previous session Mon 11-09 (11-10 holiday), next Thu 11-12
_DA_NEXT = date(2026, 11, 12)
_VA_SIG = ["NHPC", "IOC", "NTPC", "ONGC", "SBIN", "ITC", "GAIL", "BPCL", "COALINDIA", "TATASTEEL", "WIPRO", "VEDL"]


def _va_at(d, hh, mm, ss=0):
    return datetime.combine(d, time(hh, mm, ss)).replace(tzinfo=_IST_pb)


def _va_sessions(day, n=25, vol=100_000.0, close=100.0):
    """n full sessions on the trading days before `day` (previous close `close`, full-day volume `vol`)."""
    out, d = [], day
    for _ in range(n):
        d = _VD.previous_trading_day(d)
        out.append(_VD.SessionSummary(d.year * 10000 + d.month * 100 + d.day, 375, vol, close, 555))
    return out[::-1]


def _va_quote(ltp, vol, ask=None, bid=None, open_=0.0):
    return {"ltp": ltp, "open": open_, "prev_close": 100.0, "bid": bid or ltp - 0.05, "ask": ask or ltp + 0.05,
            "volume": vol}


def _va_signal_quotes(n=12):
    """n signalling symbols with distinct volume ratios (3.5x, 3.6x, ...), all up on the day."""
    return {s: _va_quote(101.0 + i, 350_000 + 10_000 * i) for i, s in enumerate(_VA_SIG[:n])}


@_ctx_va.contextmanager
def _va_env(now, quotes=None, store_syms=None, n_sessions=25):
    """A fresh attention agent at IST `now` with a synthetic session store and fake Kite quotes; PAPER+KITE;
    private ledger, paper book and snapshot file."""
    from agents.strategy_agents import ALL_AGENTS as _ALL
    clk = [now]
    qfn = quotes if callable(quotes) else (lambda instruments, _q=quotes or {}: dict(_q))
    p, _ = _fresh_perf()
    a = _VA.AttentionOvernightAgent()
    a.state.running = True
    syms = store_syms if store_syms is not None else list(_VD.UNIVERSE)
    prev_cache = _VD._cache
    prev = {k: getattr(settings, k) for k in ("trading_mode", "paper_data_source", "validated_only",
                                             "validated_attention_enabled", "validated_notional_scale")}
    prev_en = _bs_va.is_agent_enabled("attn_on")
    import tick_engine as _te_va
    async def _sleep(s):
        clk[0] = clk[0] + timedelta(seconds=s)
    with _pb_isolated():
        prev_agent = _ALL.get("attn_on")
        _ALL["attn_on"] = a
        for s in _VA_SIG:
            _te_va.tick_engine._latest_tick.pop(s, None)
        settings.trading_mode, settings.paper_data_source = "PAPER", "KITE"
        settings.validated_only, settings.validated_attention_enabled, settings.validated_notional_scale = True, True, 1.0
        _bs_va._agent_enabled["attn_on"] = True
        _VD.set_store({s: _va_sessions(clk[0].date(), n_sessions) for s in syms})
        try:
            with _patch.object(_VA, "now_ist", side_effect=lambda: clk[0]), \
                 _patch.object(_perf_mod, "now_ist", side_effect=lambda: clk[0]), \
                 _patch.object(_pb, "now_ist", side_effect=lambda: clk[0]), \
                 _patch.object(_perf_mod, "performance", p), \
                 _patch.object(_VA.VD, "quotes", side_effect=lambda ins: qfn(ins)), \
                 _patch.object(_VA.VD, "ca_calendar_loaded", return_value=True), \
                 _patch.object(_VA.VD, "price_adjusting_actions", return_value=[]), \
                 _patch.object(_VA.asyncio, "sleep", _sleep), \
                 _patch.object(_sebi_x, "pre_order_check", return_value=(True, "ALGO-ATTN-008", "ok")):
                yield a, clk, p
        finally:
            _ALL["attn_on"] = prev_agent
            for k, v in prev.items():
                setattr(settings, k, v)
            _bs_va._agent_enabled["attn_on"] = prev_en
            _VD._cache = prev_cache


def _va_rows(sym, product="CNC"):
    return [r for r in _te_kite_client._paper_positions if r["tradingsymbol"] == sym and r.get("product") == product]


def _va_orders(tag):
    return [o for o in _te_kite_client._paper_orders if o.get("tag") == tag]


# ── pure rule ──
def t_va_lab_constants_and_fingerprint():
    assert (_T_va.VOL_MULT, _T_va.LOOKBACK, _T_va.MAX_PER_DAY, _T_va.NOTIONAL) == (3.0, 20, 10, 100_000.0)
    assert (_T_va.DECISION_T, _T_va.ENTRY_LATEST, _T_va.OPEN_T, _T_va.EXIT_LATEST) == (920, 924, 555, 570)
    # pinned: a change to stratlab.attention's rule or constants must come with a VERSION bump
    assert _VA.LAB_FINGERPRINT == "7aeeaffc15b1aacb", _VA.LAB_FINGERPRINT
    a = _VA.AttentionOvernightAgent()
    assert a.VERSION == "SWING-ATTN-ON-3X/attn1/v1" and a.LAB_ID == "SWING-ATTN-ON-3X"
    assert a.product == "CNC" and a.COST_KIND == "equity_cnc" and a.scheduled and not a.strategy_rule_exits
    assert _VA.T.attention_rule is _T_va.attention_rule, "the rule is imported from the lab, not copied"


def t_va_rule_boundaries():
    p20 = (100_000.0,) * 20
    mk = lambda s, vol, c, pc=100.0, px=100.0: _VA.AttentionInput(s, vol, c, pc, p20, px)
    planned, skips = _VA.plan_attention([mk("AAA", 300_000.0, 101.0),            # ratio exactly 3.0: not traded
                                         mk("BBB", 300_001.0, 101.0),            # just above: traded
                                         mk("CCC", 500_000.0, 100.0),            # close == prev close: not traded
                                         mk("DDD", 500_000.0, 99.0)], {})
    assert [p.symbol for p in planned] == ["BBB"] and skips == {"AAA": "no_signal", "CCC": "no_signal",
                                                                "DDD": "no_signal"}, (planned, skips)
    p = planned[0]
    assert (p.side, p.qty, p.entry_t, p.exit_t) == (1, 1000, "15:21", "09:15")
    planned, _ = _VA.plan_attention([mk("Q", 400_000.0, 101.0, px=333.0)], {})
    assert planned[0].qty == int(100_000 // 333.0) == 300
    planned, _ = _VA.plan_attention([mk("Q", 400_000.0, 101.0, px=150_000.0)], {})
    assert planned[0].qty == 1, "qty is at least 1 (lab: max(1, int(NOTIONAL // px)))"
    # fewer than 20 sessions of history: no signal
    short = _VA.AttentionInput("S", 900_000.0, 101.0, 100.0, (100_000.0,) * 19, 100.0)
    assert _VA.plan_attention([short], {})[0] == []


def t_va_top10_ties_by_symbol():
    p20 = (100_000.0,) * 20
    ins = [_VA.AttentionInput(f"S{i:02d}", 400_000.0 + (i // 2) * 1000, 101.0, 100.0, p20, 100.0) for i in range(14)]
    planned, skips = _VA.plan_attention(ins, {})
    got = [p.symbol for p in planned]
    want = [x.symbol for x in sorted(ins, key=lambda x: (-x.cum_vol_1520, x.symbol))][:10]
    assert got == want and len(got) == 10, got
    assert got[:2] == ["S12", "S13"], "equal ratios rank by symbol"
    assert sum(1 for v in skips.values() if v == "rank") == 4


def t_va_exclusions():
    ca = lambda s, d: ["Split"] if (s, d) in {("AAA", _DA), ("BBB", _DA_NEXT)} else []
    assert _VA.attention_exclusions(_DA, _DA_NEXT, set(), ca, ["AAA", "BBB", "CCC"]) == \
        {"AAA": "corporate_action", "BBB": "corporate_action"}
    for special in ({_DA}, {_DA_NEXT}, {date(2026, 11, 11) + timedelta(hours=0)}):
        assert set(_VA.attention_exclusions(_DA, _DA_NEXT, special, ca, ["CCC"]).values()) == {"special_session"}
    fri, mon = date(2026, 11, 6), date(2026, 11, 9)
    assert _VA.attention_exclusions(fri, mon, {date(2026, 11, 8)}, lambda s, d: [], ["X"]) == {"X": "special_session"}, \
        "a special session (Sunday Muhurat) strictly between t and the next session excludes the night"
    assert _VA.attention_exclusions(fri, mon, {date(2026, 11, 10)}, lambda s, d: [], ["X"]) == {}
    # the lab's own CA list
    assert _VA.lab_ca("VEDL", date(2026, 4, 30)) and not _VA.lab_ca("VEDL", date(2026, 4, 29))


def t_va_scale_to_cap():
    pts = [_VB.PlannedTrade(f"S{i}", 1, 1000, 100.0, "15:21", "09:15", 4.0 - i / 10) for i in range(10)]
    same, f = _VA.scale_to_cap(pts, 1_000_000.0)
    assert same == pts and f is None
    half, f = _VA.scale_to_cap(pts, 500_000.0)
    assert f == 0.5 and [p.symbol for p in half] == [p.symbol for p in pts] and all(p.qty == 500 for p in half), \
        "a capital cap scales every name down, never drops one"


def t_va_summarize_sessions():
    def cand(d, hh, mm, c=100.0, v=10.0):
        return {"date": datetime(d.year, d.month, d.day, hh, mm, tzinfo=_IST_pb), "open": c, "high": c, "low": c,
                "close": c, "volume": v}
    d1, d2 = date(2026, 11, 5), date(2026, 11, 6)
    rows = [cand(d1, 9, 14, 1.0, 1e6), cand(d1, 15, 30, 1.0, 1e6)]                  # outside 09:15..15:29: dropped
    rows += [cand(d1, 9, 15 + i // 60 * 0, 100.0, 1.0) for i in range(0)]
    t0 = 9 * 60 + 15
    rows += [cand(d1, (t0 + i) // 60, (t0 + i) % 60, 100.0 + i / 100, 2.0) for i in range(375)]
    rows += [cand(d1, 15, 29, 777.0, 5.0)]                                           # duplicate minute: last wins
    rows += [cand(d2, (t0 + i) // 60, (t0 + i) % 60, 50.0, 1.0) for i in range(150)]  # short session (< 200 bars)
    rows += [{**cand(d2, 15, 0, 50.0), "close": 0.0}]                                 # non-positive price: dropped
    s = _VD.summarize_sessions(rows)
    assert [x.day for x in s] == [20261105, 20261106]
    assert s[0].bars == 375 and s[0].first_t == 555 and s[0].last_close == 777.0 and s[0].volume == 374 * 2.0 + 5.0, s[0]
    assert s[1].bars == 150 and _VD.valid_sessions(s) == [s[0]]
    assert _VD.prev_close_from(s, date(2026, 11, 9)) == 777.0, "the previous FULL session's last close"
    assert _VD.prev20_from(s, date(2026, 11, 9)) is None
    many = _va_sessions(_DA, 22)
    assert _VD.prev20_from(many, _DA) == (100_000.0,) * 20 and _VD.prev_close_from(many, _DA) == 100.0


# ── live path ──
def t_va_live_path_end_to_end():
    """Acceptance: a signal reaches the paper broker at the tested size and product (CNC carrier), survives the
    15:15 square-off, a same-day restart and an overnight (stale-snapshot) restart, exits at the next session's
    09:15 auction print, and is booked once in the ledger at CNC delivery costs (the lab's futures cost alongside)."""
    import platform_scheduler as _mav_va
    q = _va_signal_quotes(12)
    with _va_env(_va_at(_DA, 15, 21, 0), q) as (a, clk, p):
        out = asyncio.run(a.decide_and_enter())
        want = [s for s in _VA_SIG[::-1] if s in q][:10]          # largest ratios first
        assert out["opened"] == want, out
        ins = _va_orders("VAL-ATTN-IN")
        assert len(ins) == 10 and all(o["product"] == "CNC" and o["transaction_type"] == "BUY"
                                      and o["status"] == "COMPLETE" for o in ins)
        for s in want:
            pos = next(x for x in a.positions() if x.symbol == s)
            assert pos.qty == int(100_000 // q[s]["ask"]) and pos.entry_px == q[s]["ask"] and pos.overnight
            assert _va_rows(s)[0]["quantity"] == pos.qty
        assert a.state.skip_counts.get("rank") == 2 and a.state.skip_counts.get("no_data:quote") == 97 - 12
        # entry is once a day
        asyncio.run(a.decide_and_enter())
        assert len(_va_orders("VAL-ATTN-IN")) == 10 and a.state.skip_counts.get("already_entered") == 1
        # a square-off after the entry (dashboard button / restored square-off) leaves the overnight longs alone
        held = _ba.overnight_hold_qty()
        assert all(held.get((s, "CNC")) == _va_rows(s)[0]["quantity"] for s in want)
        with _patch.object(_mav_va, "send_telegram", _noop_async):
            _mav_va.PlatformScheduler.squareoff_book()
        assert all(_va_rows(s)[0]["quantity"] > 0 for s in want) and len(a.positions()) == 10
        # same-day restart
        clk[0] = _va_at(_DA, 15, 30)
        assert _pb.save(force=True)
        snap_day = _json_pb.loads(_pb.path().read_text())
        assert len(snap_day["validated"]["attn_on"]["positions"]) == 10
        _pb_wipe(); a._positions.clear(); a.days_done.clear()
        out = _pb.restore()
        assert out["restored"] and not out["stale"] and len(a.positions()) == 10, out
        assert a.done_today("entry"), "no second entry after a same-day restart"
        # overnight: the process restarts next morning from yesterday's snapshot
        _pb.save(force=True)
        clk[0] = _va_at(_DA_NEXT, 9, 0)
        _pb_wipe(); a._positions.clear()
        out = _pb.restore()
        assert out["restored"] and out["stale"] and len(a.positions()) == 10, out
        assert all(_va_rows(s)[0]["quantity"] == next(x.qty for x in a.positions() if x.symbol == s) for s in want)
        # 09:15:05 next session: sell at the official open (call-auction print)
        clk[0] = _va_at(_DA_NEXT, 9, 15, 5)
        q2 = {s: {**_va_quote(110.0, 1), "open": 105.0 + i} for i, s in enumerate(want)}
        closed = asyncio.run(a.exit_at_open(q2))
        assert sorted(closed) == sorted(want) and not a.positions()
        outs = _va_orders("VAL-ATTN-OUT")
        assert len(outs) == 10 and all(o["fill_kind"] == "AUCTION" and o["product"] == "CNC" for o in outs)
        assert all(o["average_price"] == q2[o["tradingsymbol"]]["open"] for o in outs)
        assert all(_va_rows(s)[0]["quantity"] == 0 for s in want)
        rows = [t for t in p.trades if t["strategy"] == "attn_on"]
        assert len(rows) == 10 and len({t["oid"] for t in rows}) == 10
        for t in rows:
            e, x, n = q[t["symbol"]]["ask"], q2[t["symbol"]]["open"], t["qty"]
            assert t["exit"] == round(x, 2) and t["gross"] == round((x - e) * n, 2)
            assert t["cost"] == _rtc_va(e * n, x * n, "equity_cnc"), "booked at CNC delivery costs (what is held)"
            assert t["cost_lab_future"] == _rtc_va(e * n, x * n, "future"), "the lab's futures cost alongside"
            assert t["cost_cnc"] == _rtc_va(e * n, x * n, "equity_cnc") and t["version"] == a.VERSION
        # nothing books twice: a second exit run, and a restart from the pre-exit snapshot
        assert asyncio.run(a.exit_at_open(q2)) == []
        a.import_book(snap_day["validated"]["attn_on"], closed={t["oid"] for t in rows}, stale=True)
        assert not a.positions(), "positions the ledger already closed are not restored"
        assert len([t for t in p.trades if t["strategy"] == "attn_on"]) == 10
        st = a.get_status()
        assert st["live"]["n"] == 10 and st["live_cnc_costs"]["n"] == 10 and st["live_readiness"]["ready"] is False


def t_va_entry_window_and_blocks():
    q = _va_signal_quotes(3)
    with _va_env(_va_at(_DA, 15, 24, 30), q) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert not _va_orders("VAL-ATTN-IN") and a.state.skip_counts.get("entry_window_missed") == 1
    with _va_env(_va_at(_DA, 15, 21), q) as (a, clk, p):
        for attr, val, why in (("validated_attention_enabled", False, "kill_switch"),
                               ("validated_only", False, "validated_only_off"),
                               ("paper_data_source", "SIM", "no_kite_data")):
            old = getattr(settings, attr)
            setattr(settings, attr, val)
            asyncio.run(a.decide_and_enter())
            setattr(settings, attr, old)
            assert a.state.skip_counts.get(why) == 1, (why, a.state.skip_counts)
        _bs_va._agent_enabled["attn_on"] = False
        asyncio.run(a.decide_and_enter())
        _bs_va._agent_enabled["attn_on"] = True
        a.stop()
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("disabled") == 1 and a.state.skip_counts.get("stopped") == 1
        assert not _va_orders("VAL-ATTN-IN") and not a.done_today("entry"), "a blocked batch can still run later"
        assert any(r["outcome"] == "skip" and r["reason"] == "kill_switch" for r in a.day_log)


def t_va_live_mode_refused():
    q = _va_signal_quotes(3)
    with _va_env(_va_at(_DA, 15, 21), q) as (a, clk, p):
        settings.trading_mode = "LIVE"
        n0 = len(_te_kite_client._paper_orders)
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("live_mode_refused") == 1 and len(_te_kite_client._paper_orders) == n0
        try:
            _te_kite_client.paper_fill_at("NHPC", "NSE", "SELL", 1, "CNC", 100.0, "x", "AUCTION")
            raise AssertionError("paper_fill_at must raise in LIVE")
        except RuntimeError:
            pass
        a._positions["X"] = _VB.ValidatedPosition("X", "attn_on", "NHPC", "NSE", "CNC", "BUY", 10, 100.0, 100.0,
                                                  _va_at(_DA - timedelta(days=2), 15, 21).isoformat(), True)
        clk[0] = _va_at(_DA, 9, 15, 5)
        assert asyncio.run(a.exit_at_open({"NHPC": _va_quote(100, 1, open_=101.0)})) == []
        assert len(_te_kite_client._paper_orders) == n0 and a.positions(), "LIVE: nothing is sent"


def t_va_kill_switch_still_exits():
    with _va_env(_va_at(_DA, 15, 21), _va_signal_quotes(2)) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert len(a.positions()) == 2
        settings.validated_attention_enabled = False
        _bs_va._agent_enabled["attn_on"] = False
        a.stop()
        clk[0] = _va_at(_DA_NEXT, 9, 15, 5)
        assert len(asyncio.run(a.exit_at_open({s: _va_quote(100, 1, open_=102.0) for s in _VA_SIG[:2]}))) == 2
        assert not a.positions() and len([t for t in p.trades if t["strategy"] == "attn_on"]) == 2


def t_va_history_and_store_gaps():
    q = _va_signal_quotes(3)
    with _va_env(_va_at(_DA, 15, 21), q, n_sessions=19) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert not _va_orders("VAL-ATTN-IN") and a.state.skip_counts.get("no_data:history") == 3
    with _va_env(_va_at(_DA, 15, 21), q, store_syms=["NHPC", "IOC"]) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert sorted(x.symbol for x in a.positions()) == ["IOC", "NHPC"]
        assert a.state.skip_counts.get("no_data:history") == 1, "a symbol missing from the store is skipped alone"
    with _va_env(_va_at(_DA, 15, 21), q) as (a, clk, p):
        _VD.set_store({s: _va_sessions(date(2026, 11, 5), 25) for s in _VD.UNIVERSE})   # stops before 11-09
        asyncio.run(a.decide_and_enter())
        assert not _va_orders("VAL-ATTN-IN") and a.state.skip_counts.get("no_data:store_stale") == 1


def t_va_quote_retries():
    calls = []
    def flaky(ins, ok_after=2):
        calls.append(1)
        return _va_signal_quotes(2) if len(calls) > ok_after else {}
    with _va_env(_va_at(_DA, 15, 21), flaky) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert len(calls) == 3 and len(a.positions()) == 2 and clk[0] == _va_at(_DA, 15, 21, 20)
    calls.clear()
    with _va_env(_va_at(_DA, 15, 21), lambda ins: (calls.append(1), {})[1]) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert len(calls) == 3 and a.state.skip_counts.get("quote_failed") == 1 and not a.positions()


def t_va_exclusions_in_live_flow():
    q = _va_signal_quotes(4)
    with _va_env(_va_at(_DA, 15, 21), q) as (a, clk, p):
        with _patch.object(_VA.VD, "price_adjusting_actions",
                           side_effect=lambda s, d: ["Demerger"] if (s, d) == ("NTPC", _DA_NEXT) else []):
            asyncio.run(a.decide_and_enter())
        assert sorted(x.symbol for x in a.positions()) == ["IOC", "NHPC", "ONGC"]
        assert a.state.skip_counts.get("corporate_action") == 1
    with _va_env(_va_at(_DA, 15, 21), q) as (a, clk, p):
        settings.validated_special_sessions = "2026-11-12"
        try:
            asyncio.run(a.decide_and_enter())
        finally:
            settings.validated_special_sessions = ""
        assert not a.positions() and a.state.skip_counts.get("special_session") == 4
    with _va_env(_va_at(_DA, 15, 21), q) as (a, clk, p):
        with _patch.object(_VA.VD, "ca_calendar_loaded", return_value=False):
            asyncio.run(a.decide_and_enter())
        assert len(a.positions()) == 4 and a.flags.get("ca_calendar_unavailable") == _DA.isoformat()


def t_va_exit_retry_and_late():
    with _va_env(_va_at(_DA, 15, 21), _va_signal_quotes(2)) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        clk[0] = _va_at(_DA_NEXT, 9, 15, 5)
        seq = [{"NHPC": _va_quote(100, 1, open_=0.0), "IOC": _va_quote(100, 1, open_=0.0)},
               {"NHPC": _va_quote(100, 1, open_=103.0), "IOC": _va_quote(100, 1, open_=0.0)}]
        later = lambda ins: seq.pop(0) if seq else {"IOC": _va_quote(99.0, 1, open_=0.0)}
        with _patch.object(_VA.VD, "quotes", side_effect=later):
            closed = asyncio.run(a.exit_at_open(seq.pop(0)))
        assert closed == ["NHPC", "IOC"], closed
        rows = {t["symbol"]: t for t in p.trades if t["strategy"] == "attn_on"}
        assert rows["NHPC"]["exit"] == 103.0 and "deviation" not in rows["NHPC"]
        assert rows["IOC"]["exit"] == 99.0 and rows["IOC"]["deviation"] == "late_exit"
        assert a.deviations.get("late_exit") == 1 and clk[0].time() >= _VA.EXIT_DEADLINE
        outs = {o["tradingsymbol"]: o for o in _va_orders("VAL-ATTN-OUT")}
        assert outs["NHPC"]["fill_kind"] == "AUCTION" and outs["IOC"]["fill_kind"] == "LTP"


def t_va_catch_up_after_restart():
    with _va_env(_va_at(_DA, 15, 21), _va_signal_quotes(2)) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        clk[0] = _va_at(_DA_NEXT, 10, 0)              # down over the 09:15 exit
        with _patch.object(_VA.VD, "quotes", return_value={s: _va_quote(104.0, 1, open_=102.0) for s in _VA_SIG[:2]}):
            out = asyncio.run(a.catch_up())
        assert sorted(out["exited"]) == ["IOC", "NHPC"] and not a.positions()
        rows = [t for t in p.trades if t["strategy"] == "attn_on"]
        assert all(t["exit"] == 104.0 and t["deviation"] == "late_exit" for t in rows), \
            "after 09:30 the LTP, never the (hindsight) official open"
    with _va_env(_va_at(_DA, 15, 22, 10), _va_signal_quotes(2)) as (a, clk, p):
        out = asyncio.run(a.catch_up())
        assert out["entry"]["opened"] and a.done_today("entry")
        assert asyncio.run(a.catch_up()) == {}


def t_va_daily_loss_stop_and_drawdown_kill():
    with _va_env(_va_at(_DA, 15, 21), _va_signal_quotes(2)) as (a, clk, p):
        p.record_trade("attn_on", "NHPC", "BUY", 100, 100.0, -10_500.0, "NSE", "CNC", cost_kind="future",
                       charge_risk=False)
        assert a.realised_today() < -10_000
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("daily_loss_stop") == 1 and not a.positions()
        clk[0] = _va_at(_DA_NEXT, 15, 21)              # next day: the daily stop no longer applies
        p.record_trade("attn_on", "NHPC", "BUY", 100, 100.0, -10_000.0, "NSE", "CNC", cost_kind="future",
                       charge_risk=False)
        _VD.set_store({s: _va_sessions(_DA_NEXT, 25) for s in _VD.UNIVERSE})
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("drawdown_kill") == 1 and not _bs_va.is_agent_enabled("attn_on")
        assert a.drawdown() >= 20_000
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("drawdown_kill") == 2, "stays killed until re-enabled by hand"
        _bs_va._agent_enabled["attn_on"] = True        # manual re-enable: drawdown measured afresh
        clk[0] = _va_at(date(2026, 11, 13), 15, 21)
        _VD.set_store({s: _va_sessions(clk[0].date(), 25) for s in _VD.UNIVERSE})
        asyncio.run(a.decide_and_enter())
        assert len(a.positions()) == 2 and a.drawdown() == 0.0


def t_va_parity_audit():
    def bars(d, vol_per_bar, close_1520, open_1521):
        rows, t0 = [], 9 * 60 + 15
        for i in range(375):
            t = t0 + i
            c = close_1520 if t == 920 else 100.0
            o = open_1521 if t == 921 else 100.0
            rows.append({"date": datetime(d.year, d.month, d.day, t // 60, t % 60), "open": o, "high": max(o, c),
                         "low": min(o, c), "close": c, "volume": vol_per_bar})
        return rows
    with _va_env(_va_at(_DA, 15, 21), _va_signal_quotes(2)) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        clk[0] = _va_at(_DA, 15, 40)
        rows = {"NHPC": bars(_DA, 350_000 / 366, 101.0, 101.05), "IOC": bars(_DA, 360_000 / 366, 102.0, 102.05),
                "NTPC": bars(_DA, 100.0, 101.0, 101.0)}
        assert _VA.bar_features(rows["NHPC"])["entry_t"] == 921
        res = a.parity_audit(rows)
        assert res["audited"] and res["bars_only"] == [] and res["quotes_only"] == [], res
        rows["NTPC"] = bars(_DA, 500_000 / 366, 101.0, 101.0)
        res = a.parity_audit(rows)
        assert res["bars_only"] == ["NTPC"] and a.deviations.get("parity_mismatch") == 1
        det = res["detail"]["NTPC"]                         # a mismatch carries the bars' own numbers
        assert det["bar_ratio"] and det["bar_ratio"] > 3 and det["bar_cum_vol"] > 0 and det["prev20_mean"] > 0
        assert det["close_1520"] > det["prev_close"] and det["excluded"] is None and det["quote_ratio"] is None


def t_va_pruning_exempt_and_no_telegram():
    p, _ = _fresh_perf()
    for _ in range(40):
        p.record_trade("attn_on", "NHPC", "BUY", 100, 100.0, -500.0, "NSE", "CNC", cost_kind="future",
                       charge_risk=False)
    assert "attn_on" not in p.pruned and not hasattr(p, "is_allowed"), "no automatic pruning in the rebuilt ledger"
    async def _boom(*a, **k):
        raise AssertionError("validated jobs must not send Telegram")
    with _va_env(_va_at(_DA, 15, 21), _va_signal_quotes(2)) as (a, clk, p), \
            _patch.object(_ba, "send_telegram", _boom):
        import platform_scheduler as _ps_va
        asyncio.run(_ps_va.PlatformScheduler()._validated_call("attn_on", "decide_and_enter"))
        clk[0] = _va_at(_DA_NEXT, 9, 15, 5)
        with _patch.object(_ps_va, "_now_ist", side_effect=lambda: clk[0]), \
             _patch.object(_VA.VD, "quotes", return_value={s: _va_quote(100, 1, open_=101.0) for s in _VA_SIG[:2]}):
            asyncio.run(_ps_va.PlatformScheduler()._validated_open())
        assert not a.positions() and len([t for t in p.trades if t["strategy"] == "attn_on"]) == 2
        assert not a.state.errors, a.state.errors


def t_va_lab_parity():
    """Acceptance gate: every day of the lab's 3-year data set through the live decision code. Inputs come from the
    lab's minute bars mapped to the live quotes (volume = Σ 09:15..15:20 bar volumes, ltp = 15:20 close, entry =
    15:21 bar open, next-day open = first bar before 09:30); the session store is built by
    validated_data.summarize_sessions from Kite-shaped candles. Compared with r2_swing_out's FINAL trade list."""
    import csv as _csv
    import json as _json
    from collections import defaultdict as _dd
    import validated_parity as _VP
    from stratlab import data as _SDm, sim as _sim
    rel = ("r2_swing_out/trades_SWING-ATTN-ON-3X.csv", "r2_swing_out/report.json")
    if not _VP.available(*rel):
        print("      SKIP (no lab data)")
        return
    man, cdir, syms, summ, cal = _VP.load("r2_swing_out")
    store = _VP.Store(summ)
    rep = _json.load(open(_os.path.join(_VP.PARITY_LAB, rel[1])))
    special = {20240302, 20240518, 20251021, 20261108}
    assert set(rep["special_days"]) == special
    special_d = {_VP.to_date(d) for d in special}
    period_ends = (rep["split"]["h2"][1], rep["split"]["final"][1])        # DISC and FINAL are simulated apart
    feat, raw = {}, _dd(dict)
    for s in syms:
        sd = _SDm.load_symbol(cdir, s)
        f = {}
        for k in range(sd.n_sessions):
            a0, b0 = int(sd.starts[k]), int(sd.ends[k])
            tt = sd.t[a0:b0]
            rec = {"first_open": float(sd.o[a0]), "first_t": int(tt[0])}
            j = int(np.searchsorted(tt, _T_va.DECISION_T, side="right")) - 1
            if j >= 0 and int(tt[j]) >= _T_va.DECISION_EARLIEST and j + 1 < len(tt) \
                    and int(tt[j + 1]) <= _T_va.ENTRY_LATEST:
                rec.update(volume=float(sd.v[a0:a0 + j + 1].sum()), ltp=float(sd.c[a0 + j]),
                           entry=float(sd.o[a0 + j + 1]), entry_t=int(tt[j + 1]))
            f[int(sd.days[k])] = rec
        feat[s] = f
        for r in _T_va.session_features(sd):           # the lab's own candidates (for explaining differences)
            raw[r["date"]][s] = r
    lab = _dd(dict)
    with open(_os.path.join(_VP.PARITY_LAB, rel[0])) as fh:
        for r in _csv.DictReader(fh):
            lab[int(r["date"])][r["symbol"]] = r
    matched = total = sig_days = 0
    bad, explained = [], _dd(list)
    plans, excls = {}, {}
    for i, d in enumerate(cal):
        nd = cal[i + 1] if i + 1 < len(cal) else None
        inputs = []
        for s in syms:
            f = feat[s].get(d)
            if not f or "volume" not in f:
                continue                                   # no quote (no full session / no 15:20 bar)
            st = store.before(s, d)
            pc, p20 = _VD.prev_close_from(st, d), _VD.prev20_from(st, d)
            if pc is None or p20 is None:
                continue
            inputs.append(_VA.AttentionInput(s, f["volume"], f["ltp"], pc, p20, f["entry"]))
        excl = _VA.attention_exclusions(_VP.to_date(d), _VP.to_date(nd), special_d, _VA.lab_ca,
                                        [x.symbol for x in inputs]) if nd else {}
        planned, _ = _VA.plan_attention(inputs, excl)
        live = {p_.symbol: p_ for p_ in planned}
        plans[d] = live
        excls[d] = excl
        labd = lab.get(d, {})
        sig_days += bool(live or labd)
        total += len(labd)
        for s, r in labd.items():
            p_ = live.get(s)
            if p_ is None:
                continue
            f, fn = feat[s][d], feat[s].get(nd, {})
            got = (s, 1, _sim._hm(f["entry_t"]), str(nd), _sim._hm(fn.get("first_t", -1)), p_.qty)
            want = (r["symbol"], int(r["side"]), r["entry_t"], r["exit_date"], r["exit_t"], int(r["qty"]))
            fills = (abs(round(p_.entry_ref * (1 + 0.0002), 4) - float(r["entry"])) < 1e-3
                     and abs(round(fn.get("first_open", 0) * (1 - 0.0002), 4) - float(r["exit"])) < 1e-3)
            if got == want and fills:
                matched += 1
            else:
                bad.append((d, s, got, want, fills))
        live_only, lab_only = set(live) - set(labd), set(labd) - set(live)
        if live_only or lab_only:
            why = {}
            for s in live_only:                            # dropped by the lab for what it learned afterwards
                rr = raw[d].get(s)
                if rr is None:
                    why[s] = None
                elif d in period_ends or rr["exit_date"] is None:
                    why[s] = "period_end"
                elif not np.isfinite(rr["exit_px"]) or rr["exit_date"] != nd:
                    why[s] = "lookahead_exit"
                else:
                    why[s] = None
            floor = min((live[s].key for s in live_only), default=None)
            for s in lab_only:                             # bumped out of the top 10 by such a candidate
                why[s] = "bumped" if (len(live) == _T_va.MAX_PER_DAY and floor is not None
                                      and raw[d][s]["ratio"] < floor) else None
            if all(why.values()):
                explained[",".join(sorted(set(why.values())))].append((d, sorted(live_only), sorted(lab_only)))
            else:
                bad.append((d, "set", why))
    # the spec's sample days, explicitly
    for d, n in ((20240603, 7), (20241230, 7), (20250627, 7), (20251020, 0), (20261105, 1)):
        assert len(plans[d]) == n == len(lab.get(d, {})), (d, sorted(plans[d]), sorted(lab.get(d, {})))
    assert {r["exit_date"] for r in lab[20251017].values()} == {"20251020"}
    assert {r["exit_date"] for r in lab[20251231].values()} == {"20260101"}
    assert excls[20260429].get("VEDL") == "corporate_action" and "VEDL" not in plans[20260429], \
        "VEDL is excluded on the eve of its CA gap (2026-04-30)"
    assert set(excls[20240517].values()) == {"special_session"}, \
        "the night into the 2024-05-18 special session is excluded (the lab's only excluded signal: NAUKRI)"
    # the same decisions from the LIVE sources: validated_data.special_sessions() and ist_clock.next_trading_day
    # (2026 onward: ist_clock's holiday table starts in 2026; full weekend sessions via EXTRA_TRADING_DAYS)
    from ist_clock import next_trading_day as _ntd_live
    live_sp = _VD.special_sessions()
    assert special_d <= live_sp, "every special day of the lab is in the live list"
    cal_diff = {}
    for i, d in enumerate(cal):
        nd = cal[i + 1] if i + 1 < len(cal) else None
        if d < 20260101 or nd is None or d in special:
            continue
        day = _VP.to_date(d)
        nl = _ntd_live(day)
        inp = [x for x in (_VA.AttentionInput(s, feat[s][d]["volume"], feat[s][d]["ltp"],
                                              _VD.prev_close_from(store.before(s, d), d),
                                              _VD.prev20_from(store.before(s, d), d), feat[s][d]["entry"])
                           for s in syms if "volume" in feat[s].get(d, {}))
               if x.prev_close is not None and x.prev20_full_vols is not None]
        ex = _VA.attention_exclusions(day, nl, live_sp, _VA.lab_ca, [x.symbol for x in inp])
        pl, _ = _VA.plan_attention(inp, ex)
        if {p_.symbol for p_ in pl} != set(plans[d]) or nl != _VP.to_date(nd):
            cal_diff[d] = (str(_VP.to_date(nd)), str(nl), sorted(set(plans[d]) ^ {p_.symbol for p_ in pl}))
    print(f"      live sources (special_sessions + next_trading_day) vs the lab's calendar, 2026: {cal_diff}")
    # the decisions are identical on every 2026 day; the next session differs only where the lab's bars have no
    # session on a date ist_clock trades (2026-03-12, 2026-05-12, 2026-09-11: data gaps, exit one day earlier live)
    # and around the 2026-11-08 Muhurat (excluded by both)
    assert all(not v[2] for v in cal_diff.values()), cal_diff
    assert set(cal_diff) <= {20260311, 20260511, 20260910, 20261106}, cal_diff
    n_div = sum(len(v) for v in explained.values())
    print(f"      parity: {len(cal)} days, {sig_days} with a signal (live or lab); trades matched {matched}/{total}; "
          f"days the lab's selection used later information: {n_div} {dict(explained)}")
    assert not bad, bad[:5]
    assert matched == total and total == 774, (matched, total)
    assert n_div <= 3, explained


run("attention: lab constants imported, LAB_FINGERPRINT pinned, CNC carrier at CNC costs", t_va_lab_constants_and_fingerprint)
run("attention: ratio exactly 3.0 / close == prev close not traded; qty floor(1e5/px) ≥ 1; < 20 sessions", t_va_rule_boundaries)
run("attention: top 10 by volume ratio, ties by symbol", t_va_top10_ties_by_symbol)
run("attention: special-session (t, t+1, in between) and corporate-action exclusions", t_va_exclusions)
run("attention: a capital cap scales every quantity, never drops a name", t_va_scale_to_cap)
run("attention: summarize_sessions keeps 09:15-15:29, drops < 200-bar sessions, previous FULL session", t_va_summarize_sessions)
run("attention LIVE PATH: CNC entry at size → square-off → restarts → 09:15 auction exit → ledger once", t_va_live_path_end_to_end)
run("attention: entry_window_missed after 15:24; kill switch / VALIDATED_ONLY / data / disabled / stopped", t_va_entry_window_and_blocks)
run("attention: LIVE mode refused (no orders; paper_fill_at raises)", t_va_live_mode_refused)
run("attention: kill switch blocks entries, the overnight exit still runs", t_va_kill_switch_still_exits)
run("attention: < 20 sessions, symbol missing from the store, stale store", t_va_history_and_store_gaps)
run("attention: whole-batch quote retried at +10 s / +20 s, then quote_failed", t_va_quote_retries)
run("attention: CA / special-session exclusions and the CA-calendar flag in the live flow", t_va_exclusions_in_live_flow)
run("attention: missing open retried until 09:29:50, then LTP late_exit", t_va_exit_retry_and_late)
run("attention: catch-up after a restart (late exit at LTP, in-window entry once)", t_va_catch_up_after_restart)
run("attention: daily loss stop and drawdown kill (until re-enabled by hand)", t_va_daily_loss_stop_and_drawdown_kill)
run("attention: 15:40 parity audit from the day's minute bars", t_va_parity_audit)
run("attention: exempt from pruning; scheduler jobs never send Telegram", t_va_pruning_exempt_and_no_telegram)
run("attention LAB PARITY: 3 years of lab days through the live decision code", t_va_lab_parity)


section("VALIDATED AGENTS — VRP-STRANGLE-016-WINGED NIFTY weekly winged strangle (agents/validated_strangle.py)")
import agents.validated_strangle as _VS
import options_replay as _OR_vs
import options_resolver as _ORS_vs
from stratlab import options_vrp as _V_vs

_VS_ENTRY = date(2026, 11, 11)      # Wed: previous session Mon 11-09 was the (holiday-moved) expiry → cycle day
_VS_EXP = date(2026, 11, 17)        # Tue weekly expiry
_VS_SPOT, _VS_VIX = 25_000.0, 14.0


def _vs_at(d, hh, mm, ss=0):
    return datetime.combine(d, time(hh, mm, ss)).replace(tzinfo=_IST_pb)


def _vs_wk(d):                       # a calendar with weekends only
    return d.weekday() < 5


def _vs_plan(lots=2, lot=None):
    p, why = _VS.plan_strangle(datetime.combine(_VS_ENTRY, time(10, 20)), _VS_SPOT, _VS_VIX, _is_td_vs, lots, lot)
    assert p is not None, why
    return p


from ist_clock import is_trading_day as _is_td_vs


def _vs_tsym(opt, k, exp=_VS_EXP):
    return f"NIFTY{exp.strftime('%y%m%d')}{int(k)}{opt}"


def _vs_instruments(exp=_VS_EXP, lot=65, extra_exp=()):
    """A fake NFO instrument master around the plan's strikes (every 50 points)."""
    rows = []
    for e in (exp,) + tuple(extra_exp):
        for k in range(22_500, 27_550, 50):
            for opt in ("CE", "PE"):
                rows.append({"name": "NIFTY", "instrument_type": opt, "segment": "NFO-OPT", "expiry": e,
                             "strike": float(k), "tradingsymbol": _vs_tsym(opt, k, e), "lot_size": lot})
    return rows


def _vs_quotes_fn(spot=_VS_SPOT, vix=_VS_VIX, legs=None, missing=()):
    """Fake Kite quotes: index spot / VIX previous close, and every option leg two-sided around its model price."""
    plan = _vs_plan()
    model = {_vs_tsym(o, k): m for o, k, _, m in plan.legs}

    def fn(instruments):
        out = {}
        for ins in instruments:
            s = ins.split(":", 1)[1]
            if s == "NIFTY 50" and spot:
                out[s] = {"ltp": spot, "open": spot, "prev_close": spot, "bid": 0, "ask": 0, "volume": 0}
            elif s == "INDIA VIX" and vix:
                out[s] = {"ltp": vix + 0.5, "open": vix, "prev_close": vix, "bid": 0, "ask": 0, "volume": 0}
            elif s in model and s not in missing:
                m = (legs or {}).get(s, model[s])
                out[s] = {"ltp": round(m, 2), "open": 0, "prev_close": 0, "bid": round(max(0.05, m - 0.5), 2),
                          "ask": round(m + 0.5, 2), "volume": 1000}
        return out
    return fn


@_ctx_va.contextmanager
def _vs_env(now, quotes=None, instruments=None, closes=None):
    """A fresh strangle agent at IST `now`; PAPER+KITE; fake quotes, NFO instrument master and expiry candles;
    private ledger, paper book and snapshot file."""
    from agents.strategy_agents import ALL_AGENTS as _ALL
    clk = [now]
    qfn = quotes or _vs_quotes_fn()
    inst = instruments if instruments is not None else _vs_instruments()
    cl = closes if closes is not None else {}
    p, _ = _fresh_perf()
    a = _VS.WingedStrangleAgent()
    a.state.running = True
    prev = {k: getattr(settings, k) for k in ("trading_mode", "paper_data_source", "validated_only",
                                             "validated_strangle_enabled", "validated_vrp_lots")}
    prev_en = _bs_va.is_agent_enabled("vrp_winged")

    async def _sleep(s):
        clk[0] = clk[0] + timedelta(seconds=s)
    with _pb_isolated():
        prev_agent = _ALL.get("vrp_winged")
        _ALL["vrp_winged"] = a
        settings.trading_mode, settings.paper_data_source = "PAPER", "KITE"
        settings.validated_only, settings.validated_strangle_enabled, settings.validated_vrp_lots = True, True, 2
        _bs_va._agent_enabled["vrp_winged"] = True
        try:
            with _patch.object(_VS, "now_ist", side_effect=lambda: clk[0]), \
                 _patch.object(_perf_mod, "now_ist", side_effect=lambda: clk[0]), \
                 _patch.object(_pb, "now_ist", side_effect=lambda: clk[0]), \
                 _patch.object(_perf_mod, "performance", p), \
                 _patch.object(_VS.VD, "quotes", side_effect=lambda ins: qfn(ins)), \
                 _patch.object(_ORS_vs, "_instruments", side_effect=lambda: inst), \
                 _patch.object(_VS, "expiry_closes", side_effect=lambda d: list(cl.get(d, []))), \
                 _patch.object(_VS, "vix_prev_close_hist", return_value=None), \
                 _patch.object(_VS.asyncio, "sleep", _sleep), \
                 _patch.object(_sebi_x, "pre_order_check", return_value=(True, "ALGO-VRP-009", "ok")):
                yield a, clk, p
        finally:
            _ALL["vrp_winged"] = prev_agent
            for k, v in prev.items():
                setattr(settings, k, v)
            _bs_va._agent_enabled["vrp_winged"] = prev_en


def _vs_rows(sym):
    return [r for r in _te_kite_client._paper_positions if r["tradingsymbol"] == sym and r.get("product") == "NRML"]


def _vs_orders(*tags):
    return [o for o in _te_kite_client._paper_orders if o.get("tag") in tags]


# ── pure rule ──
def t_vs_lab_spec_and_fingerprint():
    sp = _VS.SPEC
    assert sp.id == "VRP-STRANGLE-016-WINGED" and sp.lots == 2 and sp.underlying == "NIFTY"
    assert sp.params == {"short_delta": 0.16, "min_days_left": 3, "wing_pts": 400.0}
    st = sp.strategy()
    assert (st.decision_times, st.hold, st.target_frac, st.stop_mult, st.weekly_cycle, st.exits_early) == \
        ([time(10, 20)], "expiry", None, None, True, False), "hold to expiry: no stop, no target, no adjustment"
    # pinned: a change to the lab engine / spec must come with a VERSION bump
    assert _VS.LAB_FINGERPRINT == "80768f8470a56ca9", _VS.LAB_FINGERPRINT
    a = _VS.WingedStrangleAgent()
    assert a.VERSION == "VRP-STRANGLE-016-WINGED/o1/v1" and a.LAB_ID == "VRP-STRANGLE-016-WINGED"
    assert a.product == "NRML" and a.COST_KIND is None and a.scheduled and not a.strategy_rule_exits
    assert _VS.OR.ShortStrangle is _OR_vs.ShortStrangle and _VS.V is _V_vs, "the lab engine is imported, not copied"
    from agents.strategy_agents import ALL_AGENTS as _ALL
    from sebi_compliance import APPROVED_ALGO_IDS as _ids
    assert isinstance(_ALL["vrp_winged"], _VS.WingedStrangleAgent) and _ids["vrp_winged"] == "ALGO-VRP-009"
    assert "vrp_winged" in _bs_va.VALIDATED_AGENTS and _bs_va._DEFAULTS["vrp_winged"] is True


def t_vs_cycle_days():
    sched = lambda d: datetime.combine(d, time(10, 20))
    p, why = _VS.plan_strangle(sched(date(2026, 11, 12)), _VS_SPOT, _VS_VIX, _is_td_vs, 2)
    assert p is None and why == "not_cycle_day"
    assert _VS.cycle_status(sched(date(2026, 11, 12)), _is_td_vs) == (None, "not_cycle_day")
    p, why = _VS.plan_strangle(sched(date(2026, 11, 14)), _VS_SPOT, _VS_VIX, _is_td_vs, 2)
    assert p is None and why == "weekend"
    # short week: entry Wed 11-18 after the Tue 11-17 expiry; with 11-20 and 11-23 holidays only 11-19 and 11-24
    # remain (< 3): the week is skipped, and not entered later
    hol = {date(2026, 11, 20), date(2026, 11, 23)}
    is_td = lambda d: d.weekday() < 5 and d not in hol
    p, why = _VS.plan_strangle(sched(date(2026, 11, 18)), _VS_SPOT, _VS_VIX, is_td, 2)
    assert p is None and why == "short_week"
    assert _VS.cycle_status(sched(date(2026, 11, 18)), is_td) == (None, "short_week")
    assert _VS.cycle_status(sched(_VS_ENTRY), _is_td_vs) == (_VS_EXP, "")
    p, why = _VS.plan_strangle(sched(_VS_ENTRY), 0.0, _VS_VIX, _is_td_vs, 2)
    assert p is None and why == "no_data:spot"
    p, why = _VS.plan_strangle(sched(_VS_ENTRY), _VS_SPOT, None, _is_td_vs, 2)
    assert p is None and why == "no_data:vix"


def t_vs_legs_delta_and_wings():
    p = _vs_plan()
    assert p.expiry == _VS_EXP and p.qty == 130 and p.lot == 65 and abs(p.sigma - 0.14) < 1e-12
    (o1, kc, s1, _), (o2, kp, s2, _), (o3, kcw, s3, _), (o4, kpw, s4, _) = p.legs
    assert (o1, s1, o2, s2, o3, s3, o4, s4) == ("CE", -1, "PE", -1, "CE", 1, "PE", 1)
    assert kcw == kc + 400 and kpw == kp - 400 and all(k % 50 == 0 for k in (kc, kp)) and kp < _VS_SPOT < kc
    T = _OR_vs.years_between(p.entry, p.expiry)
    for opt, k in (("CE", kc), ("PE", kp)):
        dd = lambda x: abs(abs(_OR_vs.bs_delta(_VS_SPOT, x, T, 0.14, opt)) - 0.16)
        assert dd(k) <= dd(k - 50) and dd(k) <= dd(k + 50), (opt, k)
    assert abs(p.model_credit - sum(-s * m for _, _, s, m in p.legs)) < 1e-3 and p.model_credit > p.lab_credit > 0
    assert p.max_risk == round((400 - p.lab_credit) * 130, 2)
    assert abs(p.margin_est - (p.max_risk + 0.02 * _VS_SPOT * 130)) < 0.01


def t_vs_lot_schedule_and_expiry_calendar():
    sched = lambda d: datetime.combine(d, time(10, 20))
    p, _ = _VS.plan_strangle(sched(date(2025, 12, 24)), 26_000.0, 10.0, _vs_wk, 2)
    assert p.lot == 75 and p.qty == 150, "lot 75 before 2025-12-31"
    p, _ = _VS.plan_strangle(sched(date(2025, 12, 31)), 26_000.0, 10.0, _vs_wk, 2)
    assert p.lot == 65 and p.qty == 130 and p.expiry == date(2026, 1, 6), "lot 65 from 2025-12-31, Tuesday expiry"
    p, why = _VS.plan_strangle(sched(date(2025, 8, 29)), 24_500.0, 12.0, _vs_wk, 2)
    assert p is None and why == "short_week", "Fri 08-29 → first Tuesday expiry 09-02: 2 days left"
    p, _ = _VS.plan_strangle(sched(date(2025, 9, 3)), 24_500.0, 12.0, _vs_wk, 2)
    assert p.expiry == date(2025, 9, 9) and p.expiry.weekday() == 1
    p, _ = _VS.plan_strangle(sched(date(2025, 8, 22)), 24_500.0, 12.0, _vs_wk, 2)
    assert p.expiry == date(2025, 8, 28) and p.expiry.weekday() == 3, "Thursday before 2025-09-01"
    p, _ = _VS.plan_strangle(sched(date(2026, 11, 4)), 25_000.0, 14.0, _is_td_vs, 2)
    assert p.expiry == date(2026, 11, 9), "Tue 2026-11-10 is a holiday: the expiry moves to Mon 11-09"


def t_vs_settle_intrinsic():
    legs = [("CE", 25_300.0), ("PE", 24_600.0), ("CE", 25_700.0), ("PE", 24_200.0)]
    S, iv = _VS.settle_strangle(legs, [25_390.0, 25_410.0])
    assert S == 25_400.0 and iv == [100.0, 0.0, 0.0, 0.0]
    S, iv = _VS.settle_strangle(legs, [24_000.0])
    assert iv == [0.0, 600.0, 0.0, 200.0]


# ── live path ──
def t_vs_live_path_end_to_end():
    """Acceptance: a 10:20 cycle-day signal reaches the paper broker as 4 NRML legs at the tested size (2 lots),
    wings first; survives the 15:15 square-off, a same-day restart and overnight restarts; a restart that missed the
    expiry settles it at intrinsic from the expiry day's 15:00-15:29 closes; each leg is booked once in the ledger at
    intrinsic with the lab's option round-trip cost."""
    import platform_scheduler as _mav_vs
    plan = _vs_plan()
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20, 0)) as (a, clk, p):
        out = asyncio.run(a.decide_and_enter())
        assert out["gid"] == "VRP-20261111" and len(out["opened"]) == 4, out
        orders = _vs_orders("VAL-VRP-W", "VAL-VRP-S")
        assert [o["tag"] for o in orders] == ["VAL-VRP-W", "VAL-VRP-W", "VAL-VRP-S", "VAL-VRP-S"], "wings first"
        assert all(o["product"] == "NRML" and o["exchange"] == "NFO" and o["quantity"] == 130
                   and o["status"] == "COMPLETE" for o in orders)
        q = _vs_quotes_fn()([f"NFO:{_vs_tsym(o, k)}" for o, k, _, _ in plan.legs])
        for (opt, k, side, m) in plan.legs:
            sym = _vs_tsym(opt, k)
            pos = next(x for x in a.positions() if x.symbol == sym)
            assert pos.side == ("BUY" if side > 0 else "SELL") and pos.qty == 130 and pos.overnight
            assert pos.entry_px == (q[sym]["ask"] if side > 0 else q[sym]["bid"]) and pos.group == out["gid"]
            assert _vs_rows(sym)[0]["quantity"] == 130 * side
        st = a.structures[out["gid"]]
        assert st["status"] == "open" and abs(st["model_credit"] - plan.model_credit) < 1e-9
        assert abs(st["real_credit"] - (plan.model_credit - 4 * 0.5)) < 0.05, "real credit vs model: recorded"
        asyncio.run(a.decide_and_enter())
        assert len(_vs_orders("VAL-VRP-W", "VAL-VRP-S")) == 4 and a.state.skip_counts.get("already_entered") == 1
        # the 15:15 square-off leaves the NRML legs alone
        held = _ba.overnight_hold_qty()
        assert all(held.get((x.symbol, "NRML")) == _vs_rows(x.symbol)[0]["quantity"] for x in a.positions())
        clk[0] = _vs_at(_VS_ENTRY, 15, 15)
        with _patch.object(_mav_vs, "send_telegram", _noop_async):
            _mav_vs.PlatformScheduler.squareoff_book()
        assert len(a.positions()) == 4 and all(_vs_rows(x.symbol)[0]["quantity"] != 0 for x in a.positions())
        # same-day restart
        clk[0] = _vs_at(_VS_ENTRY, 15, 30)
        assert _pb.save(force=True)
        snap_day = _json_pb.loads(_pb.path().read_text())
        assert len(snap_day["validated"]["vrp_winged"]["positions"]) == 4
        _pb_wipe(); a._positions.clear(); a.days_done.clear(); a.structures.clear()
        out2 = _pb.restore()
        assert out2["restored"] and not out2["stale"] and len(a.positions()) == 4, out2
        assert a.done_today("entry") and a.structures["VRP-20261111"]["status"] == "open"
        # overnight: next morning from yesterday's snapshot; nothing settles before the expiry
        _pb.save(force=True)
        clk[0] = _vs_at(date(2026, 11, 12), 9, 0)
        _pb_wipe(); a._positions.clear()
        out2 = _pb.restore()
        assert out2["restored"] and out2["stale"] and len(a.positions()) == 4, out2
        assert all(_vs_rows(x.symbol)[0]["quantity"] == x.qty * (1 if x.side == "BUY" else -1) for x in a.positions())
        res = asyncio.run(a.catch_up(_vs_at(date(2026, 11, 12), 10, 20, 30)))
        assert res["settled"] == [] and "entry" not in res and len(a.positions()) == 4
        clk[0] = _vs_at(date(2026, 11, 16), 15, 40)
        assert _pb.save(force=True)
        pre_settle = _json_pb.loads(_pb.path().read_text())
        # the process was down through the expiry: restarts Wednesday morning
        clk[0] = _vs_at(date(2026, 11, 18), 9, 5)
        _pb_wipe(); a._positions.clear()
        out2 = _pb.restore()
        assert out2["stale"] and len(a.positions()) == 4
        kc = plan.legs[0][1]
        closes = [kc + 90.0] * 15 + [kc + 110.0] * 15          # S = short CE + 100
        with _patch.object(_VS, "expiry_closes", side_effect=lambda d: closes if d == _VS_EXP else []):
            res = asyncio.run(a.catch_up(clk[0]))
        assert res["settled"] == ["VRP-20261111"] and not a.positions(), res
        sett = _vs_orders("VAL-VRP-SETTLE")
        assert len(sett) == 4 and all(o["fill_kind"] == "SETTLEMENT" and o["product"] == "NRML" for o in sett)
        S, iv = _VS.settle_strangle(plan.legs, closes)
        assert S == kc + 100.0 and iv[0] == 100.0 and iv[1:] == [0.0, 0.0, 0.0]
        assert all(_vs_rows(_vs_tsym(o, k))[0]["quantity"] == 0 for o, k, _, _ in plan.legs), "paper rows flat"
        rows = [t for t in p.trades if t["strategy"] == "vrp_winged"]
        assert len(rows) == 4 and len({t["oid"] for t in rows}) == 4
        by_sym = {t["symbol"]: t for t in rows}
        for (opt, k, side, m), x in zip(plan.legs, iv):
            t = by_sym[_vs_tsym(opt, k)]
            e, n = t["entry"], t["qty"]
            assert t["exit"] == round(x, 2) and t["gross"] == round((x - e) * n * side, 2)
            assert t["cost"] == _OR_vs.leg_cost(e, x, side, n, settled=True), "the lab's option round-trip leg cost"
            assert t["version"] == a.VERSION and t["group"] == "VRP-20261111" and t["product"] == "NRML"
        assert a.structures["VRP-20261111"]["status"] == "settled"
        # nothing books twice: another settle run, and an import of the pre-settlement snapshot
        assert asyncio.run(a.settle_due(clk[0])) == []
        a.import_book(pre_settle["validated"]["vrp_winged"], closed={t["oid"] for t in rows}, stale=True)
        assert not a.positions(), "legs the ledger already closed are not restored"
        assert len([t for t in p.trades if t["strategy"] == "vrp_winged"]) == 4
        st = a.get_status()
        assert st["live"]["n"] == 1 and st["live"]["net"] == round(sum(t["net"] for t in rows), 2)
        assert st["live_readiness"]["ready"] is False and st["live_readiness"]["checks"]["cycles>=26"] is False


def t_vs_settle_on_expiry_day_and_pending():
    plan = _vs_plan()
    kp = plan.legs[1][1]
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20)) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        clk[0] = _vs_at(_VS_EXP, 15, 21)
        assert asyncio.run(a.settle_due()) == [], "not before 15:31 on the expiry day"
        clk[0] = _vs_at(_VS_EXP, 15, 31)
        assert asyncio.run(a.settle_due()) == [] and len(a.positions()) == 4, "no candles: never a guessed price"
        assert a.state.skip_counts.get("settlement_pending") == 1 and "settlement_pending" not in a.flags
        clk[0] = _vs_at(date(2026, 11, 19), 16, 0)                 # 2 trading days later: status alert
        asyncio.run(a.settle_due())
        assert a.flags.get("settlement_pending", "").startswith("VRP-20261111")
        with _patch.object(_VS, "expiry_closes", return_value=[kp - 300.0] * 30):
            assert asyncio.run(a.settle_due()) == ["VRP-20261111"] and not a.positions()
        rows = {t["symbol"]: t for t in p.trades if t["strategy"] == "vrp_winged"}
        assert rows[_vs_tsym("PE", kp)]["exit"] == 300.0 and rows[_vs_tsym("PE", kp - 400)]["exit"] == 0.0
        assert "settlement_pending" not in a.flags


def t_vs_no_partial_structure():
    plan = _vs_plan()
    wing = _vs_tsym(plan.legs[2][0], plan.legs[2][1])
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20), quotes=_vs_quotes_fn(missing={wing})) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert not _vs_orders("VAL-VRP-W", "VAL-VRP-S") and a.state.skip_counts.get("no_data:leg_quote") == 1
        assert a.done_today("entry") and not a.positions()
    # a short leg fails after the wings filled: the wings are closed at once, CRITICAL partial_structure
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20)) as (a, clk, p):
        real = _VS.paper_marketable
        def flaky(sym, exch, side, qty, product, quote, tag):
            if tag == "VAL-VRP-S":
                raise RuntimeError("order rejected")
            return real(sym, exch, side, qty, product, quote, tag)
        with _patch.object(_VS, "paper_marketable", side_effect=flaky):
            out = asyncio.run(a.decide_and_enter())
        assert out["reason"] == "partial_structure" and not a.positions()
        assert [o["tag"] for o in _te_kite_client._paper_orders] == ["VAL-VRP-W", "VAL-VRP-W", "VAL-VRP-UNWIND",
                                                                     "VAL-VRP-UNWIND"]
        assert all(r["quantity"] == 0 for r in _te_kite_client._paper_positions)
        rows = [t for t in p.trades if t["strategy"] == "vrp_winged"]
        assert len(rows) == 2 and all(t.get("deviation") == "partial_structure" for t in rows)
        assert a.deviations.get("partial_structure") == 2


def t_vs_contract_checks():
    plan = _vs_plan()
    # Kite lists an earlier weekly expiry than the lab's calendar: skip the week
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20), instruments=_vs_instruments(extra_exp=(date(2026, 11, 16),))) \
            as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("expiry_mismatch") == 1 and not _te_kite_client._paper_orders
    # a strike not listed: no_contract
    inst = [r for r in _vs_instruments() if not (r["strike"] == plan.legs[3][1] and r["instrument_type"] == "PE")]
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20), instruments=inst) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("no_contract") == 1 and not _te_kite_client._paper_orders
    # Kite's lot differs from the lab schedule: exchange truth, flagged
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20), instruments=_vs_instruments(lot=75)) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert a.flags.get("lot_mismatch") and all(x.qty == 150 for x in a.positions()) and len(a.positions()) == 4
    # risk cap
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20)) as (a, clk, p):
        old = settings.validated_vrp_max_loss
        settings.validated_vrp_max_loss = 10_000.0
        try:
            asyncio.run(a.decide_and_enter())
        finally:
            settings.validated_vrp_max_loss = old
        assert a.state.skip_counts.get("risk_cap") == 1 and not _te_kite_client._paper_orders
    # VIX previous close missing from the quote: Kite day-candle fallback; both missing: the week is skipped
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20), quotes=_vs_quotes_fn(vix=0)) as (a, clk, p):
        with _patch.object(_VS, "vix_prev_close_hist", return_value=_VS_VIX):
            asyncio.run(a.decide_and_enter())
        assert len(a.positions()) == 4
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20), quotes=_vs_quotes_fn(vix=0)) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("no_data:vix") == 1 and not a.positions()


def t_vs_entry_window_and_blocks():
    with _vs_env(_vs_at(date(2026, 11, 12), 10, 20)) as (a, clk, p):
        out = asyncio.run(a.decide_and_enter())
        assert out["reason"] == "not_cycle_day" and not a.state.skip_counts, "info only, not a skip"
        assert any(r["reason"] == "not_cycle_day" for r in a.day_log)
    with _vs_env(_vs_at(_VS_ENTRY, 10, 21, 0)) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("entry_window_missed") == 1 and not _te_kite_client._paper_orders
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20, 40)) as (a, clk, p):
        out = asyncio.run(a.catch_up())
        assert len(out["entry"]["opened"]) == 4 and a.deviations.get("late_entry") == 1
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20)) as (a, clk, p):
        for attr, val, why in (("validated_strangle_enabled", False, "kill_switch"),
                               ("validated_only", False, "validated_only_off"),
                               ("paper_data_source", "SIM", "no_kite_data")):
            old = getattr(settings, attr)
            setattr(settings, attr, val)
            asyncio.run(a.decide_and_enter())
            setattr(settings, attr, old)
            assert a.state.skip_counts.get(why) == 1, (why, a.state.skip_counts)
        _bs_va._agent_enabled["vrp_winged"] = False
        asyncio.run(a.decide_and_enter())
        _bs_va._agent_enabled["vrp_winged"] = True
        assert a.state.skip_counts.get("disabled") == 1 and not _te_kite_client._paper_orders
        asyncio.run(a.decide_and_enter())
        assert len(a.positions()) == 4, "a blocked batch can still run inside the window"
        # one structure at a time: next cycle day with the previous structure still open (not yet expired)
        a.days_done.clear()
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("position_open") == 1 and len(a.positions()) == 4


def t_vs_live_mode_refused_and_kill_switch_still_settles():
    plan = _vs_plan()
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20)) as (a, clk, p):
        settings.trading_mode = "LIVE"
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("live_mode_refused") == 1 and not _te_kite_client._paper_orders
        settings.trading_mode = "PAPER"
        asyncio.run(a.decide_and_enter())
        assert len(a.positions()) == 4
        settings.trading_mode = "LIVE"
        clk[0] = _vs_at(_VS_EXP, 15, 31)
        with _patch.object(_VS, "expiry_closes", return_value=[_VS_SPOT] * 30):
            assert asyncio.run(a.settle_due()) == [] and len(a.positions()) == 4, "LIVE: nothing is sent"
        settings.trading_mode = "PAPER"
        settings.validated_strangle_enabled = False             # the kill switch blocks entries only
        _bs_va._agent_enabled["vrp_winged"] = False
        with _patch.object(_VS, "expiry_closes", return_value=[_VS_SPOT] * 30):
            assert asyncio.run(a.settle_due()) == ["VRP-20261111"] and not a.positions()
        assert len([t for t in p.trades if t["strategy"] == "vrp_winged"]) == 4


def t_vs_drawdown_kill_and_no_telegram():
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20)) as (a, clk, p):
        assert a.dd_limit() == 195_000.0 and a.daily_loss_limit() == 0.0
        settings.validated_vrp_lots = 4
        assert a.dd_limit() == 390_000.0, "limits scale with lots / 2"
        settings.validated_vrp_lots = 2
        p.record_trade("vrp_winged", "X", "SELL", 130, 100.0, -200_000.0, "NFO", "NRML", charge_risk=False)
        asyncio.run(a.decide_and_enter())
        assert a.state.skip_counts.get("drawdown_kill") == 1 and not _bs_va.is_agent_enabled("vrp_winged")
        assert not _te_kite_client._paper_orders
    async def _boom(*a_, **k):
        raise AssertionError("validated jobs must not send Telegram")
    import platform_scheduler as _ps_vs
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20)) as (a, clk, p), _patch.object(_ba, "send_telegram", _boom):
        ps = _ps_vs.PlatformScheduler()
        with _patch.object(_ps_vs, "_now_ist", side_effect=lambda: clk[0]), \
             _patch("ist_clock.now_ist", side_effect=lambda: clk[0]):
            asyncio.run(ps._validated_vrp_entry())
            asyncio.run(ps._validated_vrp_mark())
            clk[0] = _vs_at(_VS_EXP, 15, 31)
            with _patch.object(_VS, "expiry_closes", return_value=[_VS_SPOT] * 30):
                asyncio.run(ps._validated_vrp_settle())
        assert not a.positions() and len([t for t in p.trades if t["strategy"] == "vrp_winged"]) == 4
        assert not a.state.errors, a.state.errors
        assert "VRP-20261111" in a.marks or a.structures["VRP-20261111"]["status"] == "settled"


from stratlab import options_r3 as _R3_vs


def t_vic_lab_parity():
    """r3_options_ic_25_10 as the paper agent opt_ic2510 (added 25 Nov 2026): every session of the 3-year NIFTY data through the
    live decision code with the lab spec, every lab trade's legs, fills, credit, max risk, margin and settlement
    compared with r3/options/full/trades/r3_options_ic_25_10.csv."""
    import csv as _csv
    import validated_parity as _VP
    rel = "../r3/options/full/trades/r3_options_ic_25_10.csv"
    if not (_os.path.exists(_os.path.join(_VP.PARITY_BARS, "_NIFTY50_min.json"))
            and _os.path.exists(_os.path.join(_VP.PARITY_LAB, rel))):
        print("      SKIP (no lab data)")
        return
    bars, vix = _V_vs.load(_VP.PARITY_BARS)
    lab = {int(r["date"]): r for r in _csv.DictReader(open(_os.path.join(_VP.PARITY_LAB, rel)))}
    by_day = {}
    for b in bars:
        by_day.setdefault(b[0].date(), []).append(b)
    sessions = sorted(by_day)
    is_td = _OR_vs.make_trading_day_fn(sessions)
    tz = _IST_pb
    kite_min = lambda d: [{"date": b[0].replace(tzinfo=tz), "open": b[1], "high": b[2], "low": b[3], "close": b[4],
                           "volume": 0} for b in by_day.get(d, [])]
    kite_vix = [{"date": datetime.combine(d, time(0, 0)).replace(tzinfo=tz), "open": c, "high": c, "low": c,
                 "close": c, "volume": 0} for d, c in sorted(vix.items())]

    def hist(token, frm, to, interval="day", **k):
        if interval == "minute":
            return kite_min(frm.date())
        return [r for r in kite_vix if frm.date() <= r["date"].date() <= to.date()]
    plans, reasons, off_1020 = {}, {}, []
    with _patch.object(_te_kite_client, "historical_data", side_effect=hist), \
         _patch.object(_VS, "_token", side_effect=lambda s, f: f):
        for d in sessions:
            bar = next((b for b in by_day[d] if b[0].time() >= time(10, 20)), None)
            if bar is None:
                reasons[d] = "no_bar"
                continue
            vx = _VS.vix_prev_close_hist(d)                        # live fallback path: last day candle < d
            assert vx == (vix[max(k for k in vix if k < d)] if any(k < d for k in vix) else None)
            p, why = _VS.plan_strangle(bar[0], bar[1], vx, is_td, 1, spec=_R3_vs.BY_ID["r3_options_ic_25_10"])
            live_p, _ = _VS.plan_strangle(datetime.combine(d, time(10, 20)), bar[1], vx, is_td, 1, spec=_R3_vs.BY_ID["r3_options_ic_25_10"])
            k = int(d.strftime("%Y%m%d"))
            if p is not None:
                plans[k] = p
                if bar[0].time() != time(10, 20):
                    off_1020.append(k)
                elif [x[:3] for x in live_p.legs] != [x[:3] for x in p.legs]:
                    raise AssertionError(f"{k}: the scheduled-instant plan differs from the bar-time plan")
            else:
                reasons[k] = why
            if not why.startswith("no_data"):
                cs = _VS.cycle_status(bar[0], is_td, spec=_R3_vs.BY_ID["r3_options_ic_25_10"])
                assert (cs[0] is not None) == (p is not None) and (p is not None or cs[1] == why), (k, why, cs)
        matched, mism = 0, []
        for k, r in sorted(lab.items()):
            p = plans.get(k)
            if p is None:
                mism.append((k, "no live plan", reasons.get(k)))
                continue
            legs = [x.split() for x in r["legs"].split(" | ")]
            want = [(l[1][-2:], float(l[1][:-2]), 1 if l[0] == "B" else -1) for l in legs]
            errs = []
            if [(o, kk, s) for o, kk, s, _ in p.legs] != want:
                errs.append("legs")
            if p.expiry.isoformat() != r["expiry"] or p.qty != int(r["qty"]) or p.lot != int(r["lot"]):
                errs.append("expiry/qty")
            for (o, kk, s, m), l in zip(p.legs, legs):
                e = float(l[2].split("->")[0])
                if abs(_OR_vs.fill_price(m, s, "NIFTY") - e) > 0.006:        # model price before the haircut
                    errs.append(f"fill {o}{kk}")
            if abs(p.lab_credit - float(r["credit_pts"])) > 0.006 or abs(p.max_risk - float(r["max_risk"])) > 0.02 \
                    or abs(p.margin_est - float(r["margin"])) > 0.02 or abs(p.sigma * 100 - float(r["vix"])) > 0.006:
                errs.append("credit/risk/margin/vix")
            exp = date.fromisoformat(r["expiry"])
            S, iv = _VS.settle_strangle(p.legs, _VS.expiry_closes(exp))
            if abs(S - float(r["spot_out"])) > 0.006 or any(abs(v - float(l[2].split("->")[1])) > 0.006
                                                            for v, l in zip(iv, legs)):
                errs.append("settlement")
            if errs:
                mism.append((k, errs, None))
            else:
                matched += 1
    extra = sorted(set(plans) - set(lab))
    from collections import Counter as _Cn
    rc = _Cn(reasons.values())
    first_short = min((k for k, v in reasons.items() if v == "short_week"), default=None)
    print(f"      r3_options_ic_25_10 parity: {len(sessions)} sessions, {len(plans)} live plans, lab trades matched {matched}/{len(lab)}; "
          f"live-only plans {extra}; no-plan reasons {dict(rc)}; first short week {first_short}; "
          f"cycle days whose 10:20 bar is missing (lab decides on the next bar) {off_1020}")
    assert not mism, mism[:5]
    assert matched == len(lab) == 152 and not extra
    from agents.strategy_agents import ALL_AGENTS as _AA
    a = _AA["opt_ic2510"]
    assert a.SPEC is _R3_vs.BY_ID["r3_options_ic_25_10"] and a.LAB_ID == "r3_options_ic_25_10" and a.ENTRY_T == time(10, 20)
    assert a.enabled_by_setting() is False, "off by default"


def t_vwf_lab_parity():
    """r3_options_wc_fri as the paper agent opt_wcfri (added 25 Nov 2026): every session of the 3-year NIFTY data through the
    live decision code with the lab spec, every lab trade's legs, fills, credit, max risk, margin and settlement
    compared with r3/options/full/trades/r3_options_wc_fri.csv."""
    import csv as _csv
    import validated_parity as _VP
    rel = "../r3/options/full/trades/r3_options_wc_fri.csv"
    if not (_os.path.exists(_os.path.join(_VP.PARITY_BARS, "_NIFTY50_min.json"))
            and _os.path.exists(_os.path.join(_VP.PARITY_LAB, rel))):
        print("      SKIP (no lab data)")
        return
    bars, vix = _V_vs.load(_VP.PARITY_BARS)
    lab = {int(r["date"]): r for r in _csv.DictReader(open(_os.path.join(_VP.PARITY_LAB, rel)))}
    by_day = {}
    for b in bars:
        by_day.setdefault(b[0].date(), []).append(b)
    sessions = sorted(by_day)
    is_td = _OR_vs.make_trading_day_fn(sessions)
    tz = _IST_pb
    kite_min = lambda d: [{"date": b[0].replace(tzinfo=tz), "open": b[1], "high": b[2], "low": b[3], "close": b[4],
                           "volume": 0} for b in by_day.get(d, [])]
    kite_vix = [{"date": datetime.combine(d, time(0, 0)).replace(tzinfo=tz), "open": c, "high": c, "low": c,
                 "close": c, "volume": 0} for d, c in sorted(vix.items())]

    def hist(token, frm, to, interval="day", **k):
        if interval == "minute":
            return kite_min(frm.date())
        return [r for r in kite_vix if frm.date() <= r["date"].date() <= to.date()]
    plans, reasons, off_1020 = {}, {}, []
    with _patch.object(_te_kite_client, "historical_data", side_effect=hist), \
         _patch.object(_VS, "_token", side_effect=lambda s, f: f):
        for d in sessions:
            bar = next((b for b in by_day[d] if b[0].time() >= time(15, 0)), None)
            if bar is None:
                reasons[d] = "no_bar"
                continue
            vx = _VS.vix_prev_close_hist(d)                        # live fallback path: last day candle < d
            assert vx == (vix[max(k for k in vix if k < d)] if any(k < d for k in vix) else None)
            p, why = _VS.plan_strangle(bar[0], bar[1], vx, is_td, 1, spec=_R3_vs.BY_ID["r3_options_wc_fri"])
            live_p, _ = _VS.plan_strangle(datetime.combine(d, time(15, 0)), bar[1], vx, is_td, 1, spec=_R3_vs.BY_ID["r3_options_wc_fri"])
            k = int(d.strftime("%Y%m%d"))
            if p is not None:
                plans[k] = p
                if bar[0].time() != time(15, 0):
                    off_1020.append(k)
                elif [x[:3] for x in live_p.legs] != [x[:3] for x in p.legs]:
                    raise AssertionError(f"{k}: the scheduled-instant plan differs from the bar-time plan")
            else:
                reasons[k] = why
            if not why.startswith("no_data"):
                cs = _VS.cycle_status(bar[0], is_td, spec=_R3_vs.BY_ID["r3_options_wc_fri"])
                assert (cs[0] is not None) == (p is not None) and (p is not None or cs[1] == why), (k, why, cs)
        matched, mism = 0, []
        for k, r in sorted(lab.items()):
            p = plans.get(k)
            if p is None:
                mism.append((k, "no live plan", reasons.get(k)))
                continue
            legs = [x.split() for x in r["legs"].split(" | ")]
            want = [(l[1][-2:], float(l[1][:-2]), 1 if l[0] == "B" else -1) for l in legs]
            errs = []
            if [(o, kk, s) for o, kk, s, _ in p.legs] != want:
                errs.append("legs")
            if p.expiry.isoformat() != r["expiry"] or p.qty != int(r["qty"]) or p.lot != int(r["lot"]):
                errs.append("expiry/qty")
            for (o, kk, s, m), l in zip(p.legs, legs):
                e = float(l[2].split("->")[0])
                if abs(_OR_vs.fill_price(m, s, "NIFTY") - e) > 0.006:        # model price before the haircut
                    errs.append(f"fill {o}{kk}")
            if abs(p.lab_credit - float(r["credit_pts"])) > 0.006 or abs(p.max_risk - float(r["max_risk"])) > 0.02 \
                    or abs(p.margin_est - float(r["margin"])) > 0.02 or abs(p.sigma * 100 - float(r["vix"])) > 0.006:
                errs.append("credit/risk/margin/vix")
            exp = date.fromisoformat(r["expiry"])
            S, iv = _VS.settle_strangle(p.legs, _VS.expiry_closes(exp))
            if abs(S - float(r["spot_out"])) > 0.006 or any(abs(v - float(l[2].split("->")[1])) > 0.006
                                                            for v, l in zip(iv, legs)):
                errs.append("settlement")
            if errs:
                mism.append((k, errs, None))
            else:
                matched += 1
    extra = sorted(set(plans) - set(lab))
    from collections import Counter as _Cn
    rc = _Cn(reasons.values())
    first_short = min((k for k, v in reasons.items() if v == "short_week"), default=None)
    print(f"      r3_options_wc_fri parity: {len(sessions)} sessions, {len(plans)} live plans, lab trades matched {matched}/{len(lab)}; "
          f"live-only plans {extra}; no-plan reasons {dict(rc)}; first short week {first_short}; "
          f"cycle days whose 10:20 bar is missing (lab decides on the next bar) {off_1020}")
    assert not mism, mism[:5]
    assert matched == len(lab) == 145
    # live-only plans: the last session of the week was the expiry day of the lab's still-open structure (Friday a
    # holiday or missing); the lab holds one structure at a time, and so does the agent (decide_and_enter skips
    # "position_open" while a structure's expiry >= today), so these never become orders
    lab_exps = {int(r["expiry"].replace("-", "")) for r in lab.values()}
    assert len(extra) == 9 and all(k in lab_exps for k in extra), extra
    from agents.strategy_agents import ALL_AGENTS as _AA
    a = _AA["opt_wcfri"]
    assert a.SPEC is _R3_vs.BY_ID["r3_options_wc_fri"] and a.LAB_ID == "r3_options_wc_fri" and a.ENTRY_T == time(15, 0)
    assert a.enabled_by_setting() is False, "off by default"


def t_vn_lab_parity():
    """NAKED twin of t_vs_lab_parity (added 25 Nov 2026): the same live decision code with the lab's NAKED spec,
    1 lot, against dd/vrp/final_run/trades/VRP-STRANGLE-016-NAKED.csv. Acceptance gate: every session of the lab's 3-year NIFTY data through the live decision code
    (plan_strangle at the 10:20 bar's open with the previous VIX close; contract-free), and every lab trade's legs,
    fills, max risk, margin and settlement compared with dd/vrp/final_run/trades/VRP-STRANGLE-016-WINGED.csv.
    The settlement closes go through the live candle filter (expiry_closes) from Kite-shaped candles; the VIX
    previous close through the live day-candle fallback (vix_prev_close_hist)."""
    import csv as _csv
    import validated_parity as _VP
    rel = "vrp/final_run/trades/VRP-STRANGLE-016-NAKED.csv"
    if not (_os.path.exists(_os.path.join(_VP.PARITY_BARS, "_NIFTY50_min.json"))
            and _os.path.exists(_os.path.join(_VP.PARITY_LAB, rel))):
        print("      SKIP (no lab data)")
        return
    bars, vix = _V_vs.load(_VP.PARITY_BARS)
    lab = {int(r["date"]): r for r in _csv.DictReader(open(_os.path.join(_VP.PARITY_LAB, rel)))}
    by_day = {}
    for b in bars:
        by_day.setdefault(b[0].date(), []).append(b)
    sessions = sorted(by_day)
    is_td = _OR_vs.make_trading_day_fn(sessions)
    tz = _IST_pb
    kite_min = lambda d: [{"date": b[0].replace(tzinfo=tz), "open": b[1], "high": b[2], "low": b[3], "close": b[4],
                           "volume": 0} for b in by_day.get(d, [])]
    kite_vix = [{"date": datetime.combine(d, time(0, 0)).replace(tzinfo=tz), "open": c, "high": c, "low": c,
                 "close": c, "volume": 0} for d, c in sorted(vix.items())]

    def hist(token, frm, to, interval="day", **k):
        if interval == "minute":
            return kite_min(frm.date())
        return [r for r in kite_vix if frm.date() <= r["date"].date() <= to.date()]
    plans, reasons, off_1020 = {}, {}, []
    with _patch.object(_te_kite_client, "historical_data", side_effect=hist), \
         _patch.object(_VS, "_token", side_effect=lambda s, f: f):
        for d in sessions:
            bar = next((b for b in by_day[d] if b[0].time() >= time(10, 20)), None)
            if bar is None:
                reasons[d] = "no_bar"
                continue
            vx = _VS.vix_prev_close_hist(d)                        # live fallback path: last day candle < d
            assert vx == (vix[max(k for k in vix if k < d)] if any(k < d for k in vix) else None)
            p, why = _VS.plan_strangle(bar[0], bar[1], vx, is_td, 1, spec=_VS.NAKED_SPEC)
            live_p, _ = _VS.plan_strangle(datetime.combine(d, time(10, 20)), bar[1], vx, is_td, 1, spec=_VS.NAKED_SPEC)
            k = int(d.strftime("%Y%m%d"))
            if p is not None:
                plans[k] = p
                if bar[0].time() != time(10, 20):
                    off_1020.append(k)
                elif [x[:3] for x in live_p.legs] != [x[:3] for x in p.legs]:
                    raise AssertionError(f"{k}: the scheduled-instant plan differs from the bar-time plan")
            else:
                reasons[k] = why
            if not why.startswith("no_data"):
                cs = _VS.cycle_status(bar[0], is_td, spec=_VS.NAKED_SPEC)
                assert (cs[0] is not None) == (p is not None) and (p is not None or cs[1] == why), (k, why, cs)
        matched, mism = 0, []
        for k, r in sorted(lab.items()):
            p = plans.get(k)
            if p is None:
                mism.append((k, "no live plan", reasons.get(k)))
                continue
            legs = [x.split() for x in r["legs"].split(" | ")]
            want = [(l[1][-2:], float(l[1][:-2]), 1 if l[0] == "B" else -1) for l in legs]
            errs = []
            if [(o, kk, s) for o, kk, s, _ in p.legs] != want:
                errs.append("legs")
            if p.expiry.isoformat() != r["expiry"] or p.qty != int(r["qty"]) or p.lot != int(r["lot"]):
                errs.append("expiry/qty")
            for (o, kk, s, m), l in zip(p.legs, legs):
                e = float(l[2].split("->")[0])
                if abs(_OR_vs.fill_price(m, s, "NIFTY") - e) > 0.006:        # model price before the haircut
                    errs.append(f"fill {o}{kk}")
            if abs(p.lab_credit - float(r["credit_pts"])) > 0.006 or abs(p.max_risk - float(r["max_risk"])) > 0.02 \
                    or abs(p.margin_est - float(r["margin"])) > 0.02 or abs(p.sigma * 100 - float(r["vix"])) > 0.006:
                errs.append("credit/risk/margin/vix")
            exp = date.fromisoformat(r["expiry"])
            S, iv = _VS.settle_strangle(p.legs, _VS.expiry_closes(exp))
            if abs(S - float(r["spot_out"])) > 0.006 or any(abs(v - float(l[2].split("->")[1])) > 0.006
                                                            for v, l in zip(iv, legs)):
                errs.append("settlement")
            if errs:
                mism.append((k, errs, None))
            else:
                matched += 1
    extra = sorted(set(plans) - set(lab))
    from collections import Counter as _Cn
    rc = _Cn(reasons.values())
    first_short = min((k for k, v in reasons.items() if v == "short_week"), default=None)
    print(f"      naked parity: {len(sessions)} sessions, {len(plans)} live plans, lab trades matched {matched}/{len(lab)}; "
          f"live-only plans {extra}; no-plan reasons {dict(rc)}; first short week {first_short}; "
          f"cycle days whose 10:20 bar is missing (lab decides on the next bar) {off_1020}")
    assert not mism, mism[:5]
    assert matched == len(lab) == 152 and not extra
    assert all(p.max_risk == 0.0 and len(p.legs) == 2 and all(x[2] == -1 for x in p.legs) for p in plans.values())
    from agents.strategy_agents import ALL_AGENTS as _AA
    a = _AA["vrp_naked"]
    assert a.SPEC is _VS.NAKED_SPEC and a.LAB_ID == "VRP-STRANGLE-016-NAKED" and a.TAG == "VAL-VRN" and a.SPEC.lots == 1
    assert a.enabled_by_setting() is False, "off by default (VALIDATED_STRANGLE_NAKED_ENABLED)"


def t_vs_lab_parity():
    """Acceptance gate: every session of the lab's 3-year NIFTY data through the live decision code
    (plan_strangle at the 10:20 bar's open with the previous VIX close; contract-free), and every lab trade's legs,
    fills, max risk, margin and settlement compared with dd/vrp/final_run/trades/VRP-STRANGLE-016-WINGED.csv.
    The settlement closes go through the live candle filter (expiry_closes) from Kite-shaped candles; the VIX
    previous close through the live day-candle fallback (vix_prev_close_hist)."""
    import csv as _csv
    import validated_parity as _VP
    rel = "vrp/final_run/trades/VRP-STRANGLE-016-WINGED.csv"
    if not (_os.path.exists(_os.path.join(_VP.PARITY_BARS, "_NIFTY50_min.json"))
            and _os.path.exists(_os.path.join(_VP.PARITY_LAB, rel))):
        print("      SKIP (no lab data)")
        return
    bars, vix = _V_vs.load(_VP.PARITY_BARS)
    lab = {int(r["date"]): r for r in _csv.DictReader(open(_os.path.join(_VP.PARITY_LAB, rel)))}
    by_day = {}
    for b in bars:
        by_day.setdefault(b[0].date(), []).append(b)
    sessions = sorted(by_day)
    is_td = _OR_vs.make_trading_day_fn(sessions)
    tz = _IST_pb
    kite_min = lambda d: [{"date": b[0].replace(tzinfo=tz), "open": b[1], "high": b[2], "low": b[3], "close": b[4],
                           "volume": 0} for b in by_day.get(d, [])]
    kite_vix = [{"date": datetime.combine(d, time(0, 0)).replace(tzinfo=tz), "open": c, "high": c, "low": c,
                 "close": c, "volume": 0} for d, c in sorted(vix.items())]

    def hist(token, frm, to, interval="day", **k):
        if interval == "minute":
            return kite_min(frm.date())
        return [r for r in kite_vix if frm.date() <= r["date"].date() <= to.date()]
    plans, reasons, off_1020 = {}, {}, []
    with _patch.object(_te_kite_client, "historical_data", side_effect=hist), \
         _patch.object(_VS, "_token", side_effect=lambda s, f: f):
        for d in sessions:
            bar = next((b for b in by_day[d] if b[0].time() >= time(10, 20)), None)
            if bar is None:
                reasons[d] = "no_bar"
                continue
            vx = _VS.vix_prev_close_hist(d)                        # live fallback path: last day candle < d
            assert vx == (vix[max(k for k in vix if k < d)] if any(k < d for k in vix) else None)
            p, why = _VS.plan_strangle(bar[0], bar[1], vx, is_td, 2)
            live_p, _ = _VS.plan_strangle(datetime.combine(d, time(10, 20)), bar[1], vx, is_td, 2)
            k = int(d.strftime("%Y%m%d"))
            if p is not None:
                plans[k] = p
                if bar[0].time() != time(10, 20):
                    off_1020.append(k)
                elif [x[:3] for x in live_p.legs] != [x[:3] for x in p.legs]:
                    raise AssertionError(f"{k}: the scheduled-instant plan differs from the bar-time plan")
            else:
                reasons[k] = why
            if not why.startswith("no_data"):
                cs = _VS.cycle_status(bar[0], is_td)
                assert (cs[0] is not None) == (p is not None) and (p is not None or cs[1] == why), (k, why, cs)
        matched, mism = 0, []
        for k, r in sorted(lab.items()):
            p = plans.get(k)
            if p is None:
                mism.append((k, "no live plan", reasons.get(k)))
                continue
            legs = [x.split() for x in r["legs"].split(" | ")]
            want = [(l[1][-2:], float(l[1][:-2]), 1 if l[0] == "B" else -1) for l in legs]
            errs = []
            if [(o, kk, s) for o, kk, s, _ in p.legs] != want:
                errs.append("legs")
            if p.expiry.isoformat() != r["expiry"] or p.qty != int(r["qty"]) or p.lot != int(r["lot"]):
                errs.append("expiry/qty")
            for (o, kk, s, m), l in zip(p.legs, legs):
                e = float(l[2].split("->")[0])
                if abs(_OR_vs.fill_price(m, s, "NIFTY") - e) > 0.006:        # model price before the haircut
                    errs.append(f"fill {o}{kk}")
            if abs(p.lab_credit - float(r["credit_pts"])) > 0.006 or abs(p.max_risk - float(r["max_risk"])) > 0.02 \
                    or abs(p.margin_est - float(r["margin"])) > 0.02 or abs(p.sigma * 100 - float(r["vix"])) > 0.006:
                errs.append("credit/risk/margin/vix")
            exp = date.fromisoformat(r["expiry"])
            S, iv = _VS.settle_strangle(p.legs, _VS.expiry_closes(exp))
            if abs(S - float(r["spot_out"])) > 0.006 or any(abs(v - float(l[2].split("->")[1])) > 0.006
                                                            for v, l in zip(iv, legs)):
                errs.append("settlement")
            if errs:
                mism.append((k, errs, None))
            else:
                matched += 1
    extra = sorted(set(plans) - set(lab))
    from collections import Counter as _Cn
    rc = _Cn(reasons.values())
    first_short = min((k for k, v in reasons.items() if v == "short_week"), default=None)
    print(f"      parity: {len(sessions)} sessions, {len(plans)} live plans, lab trades matched {matched}/{len(lab)}; "
          f"live-only plans {extra}; no-plan reasons {dict(rc)}; first short week {first_short}; "
          f"cycle days whose 10:20 bar is missing (lab decides on the next bar) {off_1020}")
    assert not mism, mism[:5]
    assert matched == len(lab) == 152 and not extra
    assert first_short is not None and rc["short_week"] == 5
    for k in (20231110, 20240927, 20250317, 20250903, 20251231, 20260923, 20261028, 20261104):
        assert k in plans and k in lab, k
    assert plans[20261104].expiry == date(2026, 11, 9) and plans[20251231].lot == 65
    # the live calendar (ist_clock.is_trading_day: weekdays minus NSE_HOLIDAYS, plus EXTRA_TRADING_DAYS) against the
    # lab's (its data's session list), on 2026 — the year ist_clock's holiday table covers
    from ist_clock import is_trading_day as _itd_live
    cal_diff = {}
    for k, p in sorted(plans.items()):
        if k < 20260101:
            continue
        d = date(k // 10000, k // 100 % 100, k % 100)
        lp, lwhy = _VS.plan_strangle(p.entry, p.spot, p.sigma * 100, _itd_live, 2)
        if lp is None or lp.expiry != p.expiry or [x[:3] for x in lp.legs] != [x[:3] for x in p.legs]:
            cal_diff[k] = (str(p.expiry), str(lp.expiry) if lp else lwhy)
    for k, why in sorted(reasons.items()):
        if isinstance(k, int) and k >= 20260101 and why == "short_week":
            bar = next(b for b in by_day[date(k // 10000, k // 100 % 100, k % 100)] if b[0].time() >= time(10, 20))
            lp, _ = _VS.plan_strangle(bar[0], bar[1], 15.0, _itd_live, 2)     # the cycle does not depend on vol
            if lp is not None:
                cal_diff[k] = (why, str(lp.expiry))
    print(f"      live-calendar (ist_clock) vs lab-calendar plans on 2026 cycle days: {cal_diff}")
    # 2026-05-12 and 2026-09-11 have no session in the lab's bars (0 of 97 symbols) but are trading days for
    # ist_clock: the only 2026 weeks the live calendar plans differently (documented, not hidden)
    assert set(cal_diff) <= {20260506, 20260909}, cal_diff


run("strangle: lab spec imported (WINGED, 2 lots, hold to expiry), LAB_FINGERPRINT pinned, registered", t_vs_lab_spec_and_fingerprint)
run("strangle: cycle days only (not_cycle_day / weekend / short_week / missing spot or VIX)", t_vs_cycle_days)
run("strangle: 0.16-delta shorts on the 50 grid, ±400 wings, lab credit / max risk / margin", t_vs_legs_delta_and_wings)
run("strangle: lot 75 → 65 on 2025-12-31; Thursday → Tuesday expiry; holiday-moved expiry", t_vs_lot_schedule_and_expiry_calendar)
run("strangle: settlement at intrinsic against the mean 15:00-15:29 close", t_vs_settle_intrinsic)
run("strangle LIVE PATH: 4 NRML legs at size, wings first → 15:15 → restarts → missed expiry settled once", t_vs_live_path_end_to_end)
run("strangle: settles from 15:31 on the expiry day; no candles → pending (alert after 2 trading days)", t_vs_settle_on_expiry_day_and_pending)
run("strangle: never a partial structure (missing leg quote; short rejected after wings → unwound)", t_vs_no_partial_structure)
run("strangle: expiry_mismatch, no_contract, lot_mismatch (Kite lot), risk_cap, VIX fallback", t_vs_contract_checks)
run("strangle: entry window / late catch-up entry / blocks / one structure at a time", t_vs_entry_window_and_blocks)
run("strangle: LIVE refused (no orders, no settlement fills); kill switch still settles", t_vs_live_mode_refused_and_kill_switch_still_settles)
run("strangle: drawdown kill (scales with lots); scheduler jobs never send Telegram", t_vs_drawdown_kill_and_no_telegram)
run("strangle LAB PARITY: 3 years of sessions through the live decision + settlement code", t_vs_lab_parity)
run("strangle NAKED: lab parity on the 3-year NIFTY data (live plan vs every lab trade, 1 lot, no wings)", t_vn_lab_parity)
run("options_r3 ic_25_10: lab parity on the 3-year NIFTY data (live plan vs every lab trade)", t_vic_lab_parity)
run("options_r3 wc_fri: lab parity on the 3-year NIFTY data (15:00 plan vs every lab trade)", t_vwf_lab_parity)


# ══════════════════════════════════════════════════════════════════════════
section("VALIDATED AGENTS — integration: VALIDATED_ONLY switch, scheduler, status")
# ══════════════════════════════════════════════════════════════════════════
import contextlib as _ctx_vo
import datetime as _ai_dt
import re as _re_vo
from unittest.mock import AsyncMock as _AsyncMock_vo, MagicMock as _MagicMock_vo, patch as _patch_vo













# ══════════════════════════════════════════════════════════════════════════
# Validated agents: review fixes (concurrency, restart, calendars, settlement)
# ══════════════════════════════════════════════════════════════════════════
import time as _time_rv
import bot_state as _bs_rv


def t_rv_attention_job_and_catch_up_serialised():
    """15:21 job + restart catch_up: one basket (10 names, 1x qty); 09:15:05 job + catch_up: one SELL per long,
    no naked CNC short."""
    qd = _va_signal_quotes(12)

    def slow(ins):
        _time_rv.sleep(0.2)
        return dict(qd)
    with _va_env(_va_at(_DA, 15, 21, 0), slow) as (a, clk, p):
        async def both():
            return await asyncio.gather(a.decide_and_enter(), a.catch_up(_va_at(_DA, 15, 21, 10)))
        asyncio.run(both())
        assert len(a.positions()) == 10 and len(_va_orders("VAL-ATTN-IN")) == 10
        syms = [x.symbol for x in a.positions()]
        assert all(_va_rows(s)[0]["quantity"] == x.qty for s, x in zip(syms, a.positions()))
        clk[0] = _va_at(_DA_NEXT, 9, 15, 6)
        q2 = {s: {**_va_quote(110.0, 1), "open": 105.0} for s in syms}

        def slow2(ins):
            _time_rv.sleep(0.2)
            return dict(q2)
        _VA.VD.quotes.side_effect = slow2

        async def job():                       # platform_scheduler._validated_open
            qq = await asyncio.to_thread(_VA.VD.quotes, ["x"])
            return await a.exit_at_open(qq)

        async def both_exit():
            return await asyncio.gather(job(), a.catch_up(clk[0]))
        asyncio.run(both_exit())
        assert len(_va_orders("VAL-ATTN-OUT")) == 10 and not a.positions()
        assert all(_va_rows(s)[0]["quantity"] == 0 for s in syms), "never a second SELL (naked CNC short)"
        assert len([t for t in p.trades if t["strategy"] == "attn_on"]) == 10


def t_rv_strangle_job_and_catch_up_serialised():
    """10:20 job + catch_up: one structure of 4 legs at the tested 2 lots; 15:31 settle + catch_up: 4 settlement
    fills, every NRML row flat, 4 ledger rows."""
    plan = _vs_plan()
    kc = plan.legs[0][1]
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20, 0)) as (a, clk, p):
        async def both():
            return await asyncio.gather(a.decide_and_enter(), a.catch_up(_vs_at(_VS_ENTRY, 10, 20, 0)),
                                        a.decide_and_enter(late=True))
        asyncio.run(both())
        assert len(a.positions()) == 4 and len(_vs_orders("VAL-VRP-W", "VAL-VRP-S")) == 4
        assert all(abs(r_["quantity"]) == plan.qty for r_ in _te_kite_client._paper_positions)
        clk[0] = _vs_at(_VS_EXP, 15, 31, 0)

        def ec(d):
            _time_rv.sleep(0.2)
            return [kc + 100.0] * 30
        with _patch.object(_VS, "expiry_closes", side_effect=ec):
            async def both_settle():
                return await asyncio.gather(a.settle_due(), a.catch_up(clk[0]))
            asyncio.run(both_settle())
        assert len(_vs_orders("VAL-VRP-SETTLE")) == 4 and not a.positions()
        assert all(r_["quantity"] == 0 for r_ in _te_kite_client._paper_positions)
        assert len([t for t in p.trades if t["strategy"] == "vrp_winged"]) == 4


def t_rv_strangle_settles_on_full_window_only():
    """Expiry day 15:31 with 5 of the 30 closes 15:00..15:29: pending (no fill, no ledger row). The next day the
    window is final as Kite has it: settled, flagged partial_settle_window."""
    plan = _vs_plan()
    kc = plan.legs[0][1]
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20, 0)) as (a, clk, p):
        asyncio.run(a.decide_and_enter())
        clk[0] = _vs_at(_VS_EXP, 15, 31, 0)
        with _patch.object(_VS, "expiry_closes", side_effect=lambda d: [kc - 50.0] * 5):
            assert asyncio.run(a.settle_due()) == []
            assert len(a.positions()) == 4 and not _vs_orders("VAL-VRP-SETTLE")
            assert a.state.skip_counts.get("settlement_pending") == 1
            clk[0] = _vs_at(_VS_EXP + timedelta(days=1), 15, 31, 0)
            assert asyncio.run(a.settle_due()) == ["VRP-20261111"]
        assert a.deviations.get("partial_settle_window") == 1 and not a.positions()
        assert a.structures["VRP-20261111"]["n_settle_closes"] == 5


def t_rv_strangle_crash_mid_structure_unwound():
    """The process dies after the wings filled (each leg saved as it fills): after the restart the structure in
    status 'opening' is unwound (deviation partial_structure), never held as wings only."""
    class _Die(BaseException):
        pass
    with _vs_env(_vs_at(_VS_ENTRY, 10, 20, 0)) as (a, clk, p):
        real = _VS.paper_marketable

        def dying(sym, exch, side, qty, product, quote, tag):
            if tag == "VAL-VRP-S":
                raise _Die()
            return real(sym, exch, side, qty, product, quote, tag)
        with _patch.object(_VS, "paper_marketable", side_effect=dying):
            try:
                asyncio.run(a.decide_and_enter())
            except _Die:
                pass
        snap = _json_pb.loads(_pb.path().read_text())
        legs = snap["validated"]["vrp_winged"]["positions"]
        assert len(legs) == 2 and all(x["group"] == "VRP-20261111" for x in legs), legs
        _pb_wipe(); a._positions.clear(); a.days_done.clear(); a.structures.clear()
        _pb.restore()
        assert a.structures["VRP-20261111"]["status"] == "opening" and len(a.positions()) == 2
        clk[0] = _vs_at(_VS_ENTRY, 10, 20, 40)
        r = asyncio.run(a.catch_up(clk[0]))
        assert r["unwound"] == ["VRP-20261111"] and not a.positions() and "entry" not in r, r
        assert a.done_today("entry"), "no second structure the same day (as an in-flight unwind)"
        assert a.deviations.get("partial_structure") == 2
        assert a.structures["VRP-20261111"]["status"] == "partial_structure"
        assert all(r_["quantity"] == 0 for r_ in _te_kite_client._paper_positions)


def t_rv_operator_stop_persists_and_restart_starts_agents():
    """/bot/stop is persisted: the next 08:55 preflight and a restart's catch_up keep every validated agent stopped.
    Without an operator Stop a restart's catch_up sets attention / the strangle running (they used to stay 'stopped'
    and lose the 15:21 entry / the weekly cycle)."""
    import platform_scheduler as _ps
    import event_calendar as _ec_rv
    from agents.strategy_agents import ALL_AGENTS as _ALL_rv
    names = ("auc_fade", "attn_on", "vrp_winged")
    saved_run = {n: _ALL_rv[n].state.running for n in names}
    try:
        with _patch.object(_ps, "send_telegram", _AsyncMock_vo()):
            asyncio.run(_ps.platform_scheduler.bot_stop())
        assert _bs_rv.operator_stopped() and _json_pb.loads(_bs_rv._STOP_FILE.read_text())["stopped"] is True
        assert not any(_ALL_rv[n].state.running for n in names)

        async def _noop():
            return None
        with _patch.object(_ps, "_now_ist", return_value=_va_at(_DA, 8, 55)), \
             _patch("ist_clock.is_trading_day", return_value=True), \
             _patch("validated_data.store_fresh", return_value=True), _patch.object(_ec_rv, "refresh_calendar", _noop):
            asyncio.run(_ps.PlatformScheduler()._validated_preflight())
        assert not any(_ALL_rv[n].state.running for n in names), "an operator Stop never silently expires"
        with _va_env(_va_at(_DA, 15, 21, 30), _va_signal_quotes(3)) as (a, clk, p):
            a.state.running = False
            asyncio.run(a.catch_up(clk[0]))
            assert not a.positions() and a.state.skip_counts.get("stopped") == 1
            _bs_rv.set_operator_stopped(False)              # what /bot/start does (platform_scheduler.bot_start)
            a.day_state(); a.days_done.clear()
            asyncio.run(a.catch_up(clk[0]))
            assert a.state.running and len(a.positions()) == 3
        with _vs_env(_vs_at(_VS_ENTRY, 10, 20, 30)) as (a, clk, p):
            a.state.running = False
            r = asyncio.run(a.catch_up(clk[0]))
            assert a.state.running and len(a.positions()) == 4, r
        assert "set_operator_stopped(False)" in _inspect_rv.getsource(_ps.PlatformScheduler.bot_start)
    finally:
        _bs_rv.set_operator_stopped(False)
        for n in names:
            _ALL_rv[n].state.running = saved_run[n]


def t_rv_catch_up_loads_store_and_ca_calendar():
    """A restart after 09:10 has no corporate-action calendar in memory and may have a stale store: the platform
    catch-up refreshes both before the agents' catch_up; the 15:21 decision also retries the calendar itself."""
    import platform_scheduler as _ps
    import validated_data as _vd_rv
    import event_calendar as _ec_rv
    calls = []

    async def _refresh_cal():
        calls.append("cal")
        _ec_rv._loaded_at = _va_at(_DA, 9, 30)
    with _patch.object(_ps, "_now_ist", return_value=_va_at(_DA, 9, 30)), \
         _patch("ist_clock.is_trading_day", return_value=True), \
         _patch.object(_vd_rv, "store_fresh", return_value=False), \
         _patch.object(_vd_rv, "refresh_store", side_effect=lambda: calls.append("store") or {"ok": True}), \
         _patch.object(_ec_rv, "_loaded_at", None), _patch.object(_ec_rv, "refresh_calendar", _refresh_cal), \
         _patch.object(_ps.PlatformScheduler, "_validated_agents", staticmethod(lambda: {})):
        asyncio.run(_ps.PlatformScheduler()._validated_catch_up())
        assert calls == ["store", "cal"], calls
    with _va_env(_va_at(_DA, 15, 21, 5), _va_signal_quotes(3)) as (a, clk, p):
        loaded = [False]

        async def _ens(timeout=30.0):
            loaded[0] = True
            return True
        with _patch.object(_VA.VD, "ca_calendar_loaded", side_effect=lambda: loaded[0]), \
             _patch.object(_VA.VD, "ensure_ca_calendar", side_effect=_ens):
            asyncio.run(a.decide_and_enter())
        assert loaded[0] and "ca_calendar_unavailable" not in a.flags and len(a.positions()) == 3


def t_rv_live_calendars_match_lab():
    """Special sessions come from a maintained list (the lab's special days + announced ones), not only an empty
    setting; full weekend sessions (Budget days) are trading days; the live sources exclude the night into the
    2024-05-18 DR drill and the 2026-11-08 Muhurat as the lab did."""
    import validated_data as _vd_rv
    from ist_clock import is_trading_day as _itd, next_trading_day as _ntd
    sp = _vd_rv.special_sessions()
    assert {date(2024, 3, 2), date(2024, 5, 18), date(2025, 10, 21), date(2026, 11, 8)} <= sp
    assert all(_itd(d) for d in (date(2024, 1, 20), date(2025, 2, 1), date(2026, 2, 1)))
    assert not _itd(date(2026, 1, 31)) and _ntd(date(2026, 1, 30)) == date(2026, 2, 1)
    no_ca = (lambda s, d: [])
    for day in (date(2024, 5, 17), date(2026, 11, 6)):
        ex = _VA.attention_exclusions(day, _ntd(day), sp, no_ca, ["NAUKRI"])
        assert ex == {"NAUKRI": "special_session"}, (day, ex)
    assert _VA.attention_exclusions(date(2026, 11, 12), _ntd(date(2026, 11, 12)), sp, no_ca, ["X"]) == {}


import inspect as _inspect_rv
run("attention: 15:21 job + catch_up → one basket; 09:15 exit + catch_up → one SELL each (stage lock)",
    t_rv_attention_job_and_catch_up_serialised)
run("strangle: 10:20 job + catch_up → one structure; settlement + catch_up → 4 fills, rows flat (stage lock)",
    t_rv_strangle_job_and_catch_up_serialised)
run("strangle: settles on the full 15:00-15:29 window on expiry day; a later day settles flagged",
    t_rv_strangle_settles_on_full_window_only)
run("strangle: a crash mid-entry leaves no partial structure (unwound at restart)",
    t_rv_strangle_crash_mid_structure_unwound)
run("validated: an operator Stop persists through preflight / catch-up; a restart otherwise starts the agents",
    t_rv_operator_stop_persists_and_restart_starts_agents)
run("validated: restart catch-up refreshes the session store and the corporate-action calendar",
    t_rv_catch_up_loads_store_and_ca_calendar)
run("validated: live special sessions / weekend sessions match the lab's calendar",
    t_rv_live_calendars_match_lab)


# ══════════════════════════════════════════════════════════════════════════
# CANDIDATE AGENT — EV-PEAD-LONG-H60 post-earnings drift (agents/validated_pead.py)
# ══════════════════════════════════════════════════════════════════════════
section("CANDIDATE AGENT — EV-PEAD-LONG-H60 post-earnings drift, PAPER only (agents/validated_pead.py)")
import test_validated_pead as _TVP


def _pead_case(fn):
    def go():
        try:
            fn()
        except _TVP.Skip as exc:                      # the lab's scratch data set is absent: fixture tests still run
            print(f"      (skipped: {exc})")
    return go


for _n, _f in _TVP.TESTS:
    run("PEAD: " + _n[len("test_"):].replace("_", " "), _pead_case(_f))


# ══════════════════════════════════════════════════════════════════════════
# CORE REBUILD — old state loads unchanged; engine upgrades (test_rebuild.py)
# ══════════════════════════════════════════════════════════════════════════
section("CORE REBUILD — persisted state of the old build, trading states, approvals, risk budget, drift, v2 API")
import test_rebuild as _TRB

for _n, _f in _TRB.TESTS:
    run("rebuild: " + _n[len("test_"):].replace("_", " "), _f)


# ══════════════════════════════════════════════════════════════════════════
# FORWARD LAB — every lab strategy after the close, pre-registered rule (test_forward_lab.py)
# ══════════════════════════════════════════════════════════════════════════
section("FORWARD LAB — store, engine (separate process), parity with the lab, ledgers, frozen rule, v2 API")
import test_forward_lab as _TFL


def _flab_case(fn):
    def go():
        try:
            fn()
        except _TFL.Skip as exc:                      # the lab's scratch data set is absent: fixture tests still run
            print(f"      (skipped: {exc})")
    return go


for _n, _f in _TFL.TESTS:
    run("forward lab: " + _n[len("test_"):].replace("_", " "), _flab_case(_f))


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY-BUILDER DATA LAYER — deep minute store, NSE archives, forward-lab NSE ids (test_builder_data.py)
# ══════════════════════════════════════════════════════════════════════════
section("BUILDER DATA — Kite minute backfill, NSE archives in nsedata's layout, guards, disk cap, forward-lab NSE ids")
import test_builder_data as _TBD


def _bdata_case(fn):
    def go():
        try:
            fn()
        except _TBD.Skip as exc:                      # the lab's scratch data set is absent: fixture tests still run
            print(f"      (skipped: {exc})")
    return go


for _n, _f in _TBD.TESTS:
    run("builder data: " + _n[len("test_"):].replace("_", " "), _bdata_case(_f))


# ══════════════════════════════════════════════════════════════════════════
# AUTOMATIC STRATEGY BUILDER — grammar, freeze-before-evaluate, splits, N, promotion, builder/v1 (test_builder.py)
# ══════════════════════════════════════════════════════════════════════════
section("STRATEGY BUILDER — grammar, frozen rule, registry, evaluation process, N, promotion, builder/v1 family, API")
import test_builder as _TBL

for _n, _f in _TBL.TESTS:
    run("builder: " + _n[len("test_"):].replace("_", " "), _f)


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
failed = summary()
sys.exit(1 if failed else 0)
