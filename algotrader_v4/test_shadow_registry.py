"""
test_shadow_registry.py — tests of the frozen shadow-incubation registry (shadow_registry.py).

Run: cd algotrader_v4 && python test_shadow_registry.py      (also collectable by pytest: test_* functions)
Never writes into the repo: every registry lives in a temporary directory.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("PERFORMANCE_FILE", os.path.join(tempfile.mkdtemp(), "performance.json"))
os.environ["SHADOW_TRIALS_CSV"] = os.path.join(tempfile.mkdtemp(), "absent_trials.csv")   # never the lab's file

import shadow_registry as SR

REG_DAY = date(2026, 11, 14)


def _reg() -> SR.ShadowRegistry:
    return SR.ShadowRegistry(tempfile.mkdtemp())


def _spec(i="t1", **cfg) -> SR.ShadowSpec:
    return SR.ShadowSpec(id=i, family="fam", strategy_ref="x:y", source="manual",
                         config={"kind": "manual", "max_hold_days": 0, **cfg})


def _trade(day: date, k: int, net: float, notional: float = 100000.0) -> dict:
    key = f"{day.isoformat()}:S{k}"
    return {"key": key, "trade_key": key, "day": day.isoformat(), "net": net, "notional": notional}


# ── hash chain ────────────────────────────────────────────────────────────────────────────────────────────────
def _filled_ledger():
    r = _reg()
    r.register(_spec(), REG_DAY)
    for i in range(5):
        r.record_trade("t1", _trade(REG_DAY + timedelta(days=1 + i), 0, 100.0 + i))
    return r, r.ledger("t1").path


def test_chain_verifies_clean():
    r, p = _filled_ledger()
    res = SR.verify_chain(p)
    assert res["ok"] and res["records"] == 6, res          # genesis + 5 trades
    assert r.verify_all()["ok"]


def test_chain_detects_edit():
    r, p = _filled_ledger()
    lines = p.read_text().splitlines()
    rec = json.loads(lines[3])
    rec["data"]["net"] = 999999.0                          # doctor one P&L, keep its stored hash
    lines[3] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    res = SR.verify_chain(p)
    assert not res["ok"] and res["bad_seq"] == 3 and "edited" in res["error"], res


def test_chain_detects_rehashed_edit():
    """Re-hashing the edited record does not help: the next record's prev_hash no longer matches."""
    r, p = _filled_ledger()
    lines = p.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["data"]["net"] = -5.0
    rec["hash"] = SR._record_hash(rec)
    lines[2] = SR._canon(rec)
    p.write_text("\n".join(lines) + "\n")
    res = SR.verify_chain(p)
    assert not res["ok"] and res["bad_seq"] == 3 and "prev_hash" in res["error"], res


def test_chain_detects_delete_and_reorder():
    r, p = _filled_ledger()
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:2] + lines[3:]) + "\n")
    assert not SR.verify_chain(p)["ok"]
    p.write_text("\n".join([lines[0], lines[2], lines[1]] + lines[3:]) + "\n")
    assert not SR.verify_chain(p)["ok"]


def test_chain_detects_truncation_via_checkpoint():
    r, p = _filled_ledger()
    r.checkpoint("t1")
    assert r.verify_all()["ok"]
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-2]) + "\n")             # drop the last two trades: chain itself still valid
    assert SR.verify_chain(p)["ok"]
    v = r.verify_all()
    assert not v["ok"] and "truncated" in v["t1"]["error"], v
    ev = r.evaluate("t1")
    assert not ev["chain_ok"] and "error" in ev           # a broken ledger is never evaluated


def test_registry_chain_tamper_detected():
    r, _ = _filled_ledger()
    p = r.index.path
    txt = p.read_text().replace('"registered_on":"2026-11-14"', '"registered_on":"2026-01-01"')
    p.write_text(txt)
    assert not r.verify_all()["registry"]["ok"]


# ── frozen config, forward only ───────────────────────────────────────────────────────────────────────────────
def test_frozen_config_and_idempotent_register():
    r = _reg()
    a = r.register(_spec(param=1), REG_DAY)
    assert r.register(_spec(param=1), REG_DAY)["config_hash"] == a["config_hash"]
    try:
        r.register(_spec(param=2), REG_DAY)
        raise AssertionError("changed config accepted")
    except SR.FrozenConfigError:
        pass
    assert len([x for x in r.index.records() if x["kind"] == "register"]) == 1
    assert a["config_hash"] == SR.config_hash({"kind": "manual", "max_hold_days": 0, "param": 1})
    assert a["rule_hash"] == SR.RULE_HASH


def test_no_backfill_and_dedup():
    r = _reg()
    r.register(_spec(max_hold_days=5), REG_DAY)
    e = r.entries()["t1"]
    assert e["forward_from"] == (REG_DAY + timedelta(days=6)).isoformat()
    assert r.record_trade("t1", _trade(REG_DAY, 0, 1.0)) is None                       # before registration
    assert r.record_trade("t1", _trade(REG_DAY + timedelta(days=5), 0, 1.0)) is None   # overnight straddling it
    assert r.record_trade("t1", _trade(REG_DAY + timedelta(days=6), 0, 1.0)) is not None
    assert r.record_trade("t1", _trade(REG_DAY + timedelta(days=6), 0, 1.0)) is None   # same key twice
    assert len(r.trades("t1")) == 1


def test_paper_fill_model():
    assert SR.paper_fill("BUY", {"bid": 99.9, "ask": 100.1}) == 100.1
    assert SR.paper_fill("SELL", {"bid": 99.9, "ask": 100.1}) == 99.9
    assert abs(SR.paper_fill("BUY", {"ltp": 100.0}) - 100.021) < 1e-9                  # +2.1 bps half-spread
    assert abs(SR.paper_fill("SELL", None, ref_px=100.0) - 99.979) < 1e-9
    import forward_test
    assert SR.HALF_SPREAD_BPS == forward_test.SLIPPAGE_BPS


# ── promotion math ────────────────────────────────────────────────────────────────────────────────────────────
def test_bonferroni_threshold():
    assert SR.bonferroni_threshold(1) == 0.05 / 3
    assert abs(SR.bonferroni_threshold(9) - 0.05 / 27) < 1e-15
    assert SR.bonferroni_threshold(0) == 0.05 / 3                                       # floor at one shadow
    assert SR.bonferroni_threshold(4, alpha=0.1, n_looks=2) == 0.1 / 8


def test_bootstrap_pvalue_behaviour():
    import numpy as np
    rng = np.random.default_rng(1)
    strong = list(rng.normal(500, 1000, 120))
    null = list(rng.normal(0, 1000, 120))
    neg = list(rng.normal(-500, 1000, 120))
    p_s, p_0, p_n = (SR.bootstrap_pvalue(x, 4000, 7) for x in (strong, null, neg))
    assert p_s < 0.001 and p_0 > 0.01 and p_n > 0.9, (p_s, p_0, p_n)
    assert SR.bootstrap_pvalue(strong, 4000, 7) == p_s                                  # deterministic seed
    assert SR.bootstrap_pvalue([5.0], 100, 0) == 1.0
    # p has the bootstrap floor 1/(B+1)
    assert SR.bootstrap_pvalue([10.0] * 50 + [11.0] * 50, 999, 3) == 1 / 1000
    # agrees with a t-test order of magnitude for a moderate effect
    x = list(rng.normal(150, 1000, 200))
    from math import erf, sqrt
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    z = m / (s / sqrt(len(x)))
    p_norm = 0.5 * (1 - erf(z / sqrt(2)))
    p_b = SR.bootstrap_pvalue(x, 20000, 11)
    assert abs(p_b - p_norm) < 0.02 + 0.5 * p_norm, (p_b, p_norm)


def test_look_sample_rules():
    d0 = REG_DAY + timedelta(days=1)
    # 5 trades a day: 50 trades reached on day 10, but min 20 days → sample = first 20 days (100 trades)
    trades = [_trade(d0 + timedelta(days=i), k, 1.0) for i in range(30) for k in range(5)]
    s = SR.look_sample(trades, 50, 20)
    assert s is not None and len({t["day"] for t in s}) == 20 and len(s) == 100
    assert SR.look_sample(trades[:40], 50, 1) is None
    # strangle legs share a trade_key: 4 legs = 1 trade
    legs = [{**_trade(d0 + timedelta(days=i), k, 1.0), "trade_key": f"g{i}"} for i in range(10) for k in range(4)]
    assert SR.look_sample(legs, 10, 1) is not None and SR.look_sample(legs, 11, 1) is None


def test_promotion_uses_bonferroni_over_active_shadows():
    """The same forward record promotes when it is the only shadow and does not when 40 shadows are active."""
    d0 = REG_DAY + timedelta(days=1)
    # 50 trades over 50 days, an edge whose bootstrap p sits between 0.05/(40·3) and 0.05/3
    nets = [380.0 + (1000.0 if i % 2 else -1000.0) * (1 + 0.01 * i) for i in range(50)]   # z ≈ 2.6
    sample = [_trade(d0 + timedelta(days=i), 0, v) for i, v in enumerate(nets)]
    p = SR.judge_look("solo", 1, sample, 1)["p_value"]
    assert SR.bonferroni_threshold(40) < p <= SR.bonferroni_threshold(1), p

    r = _reg()
    r.register(_spec("solo"), REG_DAY)
    for t in sample:
        r.record_trade("solo", t)
    with mock.patch.object(SR, "look_seed", return_value=SR.look_seed("solo", 1)):
        ev = r.evaluate("solo")
    assert ev["status"] == "promotion_eligible" and ev["looks"]["1"]["n_active"] == 1, ev

    r2 = _reg()
    for i in range(40):
        r2.register(_spec(f"s{i:02d}"), REG_DAY)
    for t in sample:
        r2.record_trade("s00", t)
    with mock.patch.object(SR, "look_seed", return_value=SR.look_seed("solo", 1)):
        ev2 = r2.evaluate("s00")
    lk = ev2["looks"]["1"]
    assert ev2["status"] == "incubating" and lk["n_active"] == 40 and not lk["promote"], ev2
    assert abs(lk["p_value"] - p) < 1e-9 and abs(lk["threshold"] - 0.05 / 120) < 1e-12
    # one look, evaluated once: re-evaluating does not re-run (or re-write) look 1
    n_looks = sum(1 for x in r2.index.records() if x["kind"] == "look")
    ev3 = r2.evaluate("s00")
    assert ev3["new_looks"] == [] and sum(1 for x in r2.index.records() if x["kind"] == "look") == n_looks


def test_min_trades_blocks_promotion():
    r = _reg()
    r.register(_spec(), REG_DAY)
    for i in range(49):                                        # a huge edge, one trade short of the first look
        r.record_trade("t1", _trade(REG_DAY + timedelta(days=1 + i), 0, 10000.0 + i))
    ev = r.evaluate("t1")
    assert ev["status"] == "incubating" and ev["looks"] == {} and ev["new_looks"] == []
    r.record_trade("t1", _trade(REG_DAY + timedelta(days=60), 0, 10000.0))
    assert r.evaluate("t1")["status"] == "promotion_eligible"


def test_demotion_on_drawdown():
    r = _reg()
    r.register(_spec(backtest_max_dd=10000.0, per_trade_notional=100000), REG_DAY)
    d = REG_DAY + timedelta(days=1)
    for i, v in enumerate([3000, -6000, -5000, 2000]):         # DD 11,000 ≤ 15,000
        r.record_trade("t1", _trade(d + timedelta(days=i), 0, v))
    assert r.evaluate("t1")["status"] == "incubating"
    # notional 200,000 → normalised to lab size the −9,000 is −4,500 → DD 13,500, still ≤ 15,000
    r.record_trade("t1", _trade(d + timedelta(days=5), 0, -9000, notional=200000))
    ev = r.evaluate("t1")
    assert ev["status"] == "incubating" and ev["forward_max_dd"] == 13500.0, ev
    r.record_trade("t1", _trade(d + timedelta(days=6), 0, -2000))
    ev = r.evaluate("t1")
    assert ev["status"] == "demoted" and r.entries()["t1"]["status"] == "demoted", ev
    assert r.record_trade("t1", _trade(d + timedelta(days=7), 0, 1.0)) is None          # no more trades
    assert "t1" not in r.active_ids()
    # without a backtest reference the demotion cannot fire, and says so
    r2 = _reg()
    r2.register(_spec(), REG_DAY)
    r2.record_trade("t1", _trade(d, 0, -1e9))
    ev = r2.evaluate("t1")
    assert ev["status"] == "incubating" and "demotion_note" in ev


def test_rule_is_preregistered_constants():
    R = SR.PROMOTION_RULE
    assert R["looks_trades"][0] == R["min_forward_trades"] and R["family_alpha"] == 0.05
    assert R["demotion_dd_multiple"] == 1.5 and "bootstrap" in R["test"]
    doc = (Path(__file__).resolve().parent / "SHADOW_REGISTRY.md").read_text()
    assert R["version"] in doc and "Bonferroni" in doc


# ── trials ledger hook ────────────────────────────────────────────────────────────────────────────────────────
def test_trials_hook_optional():
    d = tempfile.mkdtemp()
    p = Path(d) / "trials.csv"
    assert SR.append_trial(_spec(), "2026-11-14", p) is False and not p.exists()      # never creates the file
    p.write_text("id,family,protocol,extra_col\nold,f,final,x\n")
    assert SR.append_trial(_spec(), "2026-11-14", p) is True
    assert SR.append_trial(_spec(), "2026-11-14", p) is False                          # once per id
    rows = p.read_text().splitlines()
    assert rows[-1] == "t1,fam,shadow," and len(rows) == 3, rows


# ── enrolment of the forward test and the validated agents ────────────────────────────────────────────────────
def _fake_agents():
    mk = lambda n, lab, ver, prod: SimpleNamespace(name=n, LAB_ID=lab, VERSION=ver, LAB_FINGERPRINT="fp" + n,
                                                    product=prod, scheduled=True)
    return {"auc_fade": mk("auc_fade", "AUC-FADE-3PCT", "AUC-FADE-3PCT/f/v1", "MIS"),
            "attn_on": mk("attn_on", "SWING-ATTN-ON-3X", "SWING-ATTN-ON-3X/f/v1", "CNC"),
            "vrp_winged": mk("vrp_winged", "VRP-STRANGLE-016-WINGED", "VRP/f/v1", "NRML")}


def _fwd_state(day: str) -> dict:
    t = {"symbol": "INFY", "side": "SELL", "qty": 60, "auction_px": 1650.0, "entry_fill": 1649.65,
         "exit_fill": 1640.35, "gross": 558.0, "cost": 40.0, "net": 518.0, "signal_gap_pct": 3.4}
    import forward_test as F
    return {"days": {day: {"trades": {v: {r: [dict(t)] for r in F.RULES} for v in F.VARIANTS}}}}


def test_enrol_and_sync_without_changing_trading():
    r = _reg()
    agents = _fake_agents()
    with mock.patch.object(SR, "_today", return_value=REG_DAY):
        out = SR.ensure_enrolled(r, agents)
    assert len(out) == 9 and set(out.values()) == {"registered"}, out
    assert "forward:AUC-FADE-3PCT/live" in out and "attn_on@SWING-ATTN-ON-3X/f/v1" in out
    e = r.entries()
    assert e["forward:AUC-FADE-3PCT/final"]["config"]["backtest_max_dd"] == 24605.28
    assert e["attn_on@SWING-ATTN-ON-3X/f/v1"]["forward_from"] == "2026-11-20"
    # forward test: a day before forward_from is ignored, a later day copied once
    assert SR.sync_forward_test(r, _fwd_state("2026-11-14")) == 0
    assert SR.sync_forward_test(r, _fwd_state("2026-11-16")) == 6
    assert SR.sync_forward_test(r, _fwd_state("2026-11-16")) == 0
    # validated agents: rows of the frozen VERSION only; strangle legs grouped
    rows = [{"t": "2026-11-16T09:20:05", "strategy": "auc_fade", "version": "AUC-FADE-3PCT/f/v1", "symbol": "TCS",
             "side": "SELL", "qty": 25, "entry": 4000.0, "exit": 3990.0, "gross": 250, "cost": 30, "net": 220,
             "oid": "P1"},
            {"t": "2026-11-16T09:20:05", "strategy": "auc_fade", "version": "OLD/v0", "symbol": "TCS",
             "side": "SELL", "qty": 25, "entry": 4000.0, "exit": 3990.0, "gross": 250, "cost": 30, "net": 220,
             "oid": "P0"}] + [
            {"t": "2026-11-26T15:31:00", "strategy": "vrp_winged", "version": "VRP/f/v1", "symbol": f"NIFTY{k}",
             "side": "SELL", "qty": 130, "entry": 50.0, "exit": 0.0, "gross": 6500, "cost": 60, "net": 6440,
             "oid": f"L{k}", "group": "G1"} for k in range(4)]
    assert SR.sync_validated_agents(r, agents, rows) == 5
    assert SR.sync_validated_agents(r, agents, rows) == 0
    vt = r.trades("vrp_winged@VRP/f/v1")
    assert len(vt) == 4 and {t["trade_key"] for t in vt} == {"G1"}
    # a lab change without a VERSION bump: config drift, nothing copied any more, no re-registration
    agents["auc_fade"].LAB_FINGERPRINT = "changed"
    out2 = SR.ensure_enrolled(r, agents)
    assert out2["auc_fade@AUC-FADE-3PCT/f/v1"] == "config_drift"
    rows[0] = {**rows[0], "oid": "P2", "t": "2026-11-17T09:20:05"}
    assert SR.sync_validated_agents(r, agents, rows[:1]) == 0
    assert r.verify_all()["ok"]
    r.evaluate_all()
    assert r.verify_all()["ok"]


def test_daily_job_gated_by_flag_and_paper():
    from config import settings
    with mock.patch.object(settings, "shadow_registry_enabled", False):
        assert "skipped" in SR.daily_job(_reg(), _fake_agents())
    with mock.patch.object(settings, "shadow_registry_enabled", True), \
         mock.patch.object(settings, "trading_mode", "LIVE"):
        assert "skipped" in SR.daily_job(_reg(), _fake_agents())
    assert settings.shadow_registry_enabled is False                                    # default off


# ── nothing places an order ───────────────────────────────────────────────────────────────────────────────────
ORDER_NAMES = {"place_order", "modify_order", "cancel_order", "paper_fill_at", "paper_marketable", "place_gtt",
               "kite_client", "atomic_bracket", "place_bracket", "exit_order", "square_off"}


def test_source_never_references_order_placement():
    src = (Path(__file__).resolve().parent / "shadow_registry.py").read_text()
    tree = ast.parse(src)
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            used.update(a.name.split(".")[0] for a in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                used.add(node.module.split(".")[0])
    assert not (used & ORDER_NAMES), used & ORDER_NAMES


def test_runtime_daily_job_places_no_order():
    """The whole daily job (enrol, sync, evaluate) with every order entry point booby-trapped."""
    from config import settings
    from kite_client import kite_client
    boom = mock.Mock(side_effect=AssertionError("order placed"))
    r = _reg()
    with mock.patch.object(kite_client, "place_order", boom), \
         mock.patch.object(kite_client, "paper_fill_at", boom), \
         mock.patch.object(settings, "shadow_registry_enabled", True), \
         mock.patch.object(settings, "trading_mode", "PAPER"), \
         mock.patch("forward_test._load", return_value=_fwd_state("2026-12-01")), \
         mock.patch.object(SR, "_today", return_value=REG_DAY):
        out = SR.daily_job(r, _fake_agents())
    assert boom.call_count == 0 and out["verify_ok"] and out["forward_trades_added"] == 6, out


def test_scheduler_job_only_with_flag():
    import platform_scheduler as ps
    from config import settings

    class _S:
        def __init__(self):
            self.ids = []

        def add_job(self, *a, **k):
            self.ids.append(k.get("id"))

        def start(self):
            pass
    for flag, mode, expect in ((False, "PAPER", False), (True, "LIVE", False), (True, "PAPER", True)):
        s = ps.PlatformScheduler()
        s._sched = _S()
        with mock.patch.object(settings, "shadow_registry_enabled", flag), \
             mock.patch.object(settings, "trading_mode", mode):
            s.start()
        assert ("shadow_registry" in s._sched.ids) is expect, (flag, mode, s._sched.ids)
    # the job itself re-checks the flag (off → skipped, nothing written)
    with mock.patch.object(settings, "shadow_registry_enabled", False), \
         mock.patch.object(SR, "daily_job", wraps=SR.daily_job) as dj:
        asyncio.run(ps.PlatformScheduler()._shadow_registry())
        assert dj.call_count == 1


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  ok    {n}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
