"""
api_v2.py — the versioned dashboard API (/api/v2/...; documented with examples in API_V2.md).

Read endpoints (GET) and the few write endpoints the dashboard needs (trading states, approvals, per-strategy
enable, brake reset). Auth: with API_KEY set every /api/v2 path except /api/v2/health needs X-API-Key or a Bearer
JWT from POST /auth/login (main.py's gate). Responses are plain JSON, never NaN; times are IST ISO-8601.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

import bot_state
from config import settings
from ist_clock import holiday_name, is_market_open, is_trading_day, next_trading_day, now_ist
from tick_engine import json_safe

API_VERSION = "2.0"
router = APIRouter(prefix="/v2", tags=["v2"])

LABELS = {"auc_fade": "Opening-auction fade", "attn_on": "Attention overnight long",
          "vrp_winged": "NIFTY winged strangle", "pead_h60": "Post-earnings drift (60 sessions)"}
READINESS_TARGET = {"auc_fade": 150, "attn_on": 100, "vrp_winged": 26, "pead_h60": None}
JOB_PREFIX = {"auc_fade": ("validated_open", "validated_auc_exit", "validated_store", "forward_"),
              "attn_on": ("validated_attn_entry", "validated_open", "validated_store"),
              "vrp_winged": ("validated_vrp_",), "pead_h60": ("validated_pead_",)}


def _agents() -> dict:
    from agents.strategy_agents import ALL_AGENTS
    return ALL_AGENTS


def _who(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        from auth import decode_token
        u = decode_token(auth[7:])
        if u:
            return f"user:{u}"
    if request.headers.get("X-API-Key"):
        return "api-key"
    return "anonymous"


def _today_rows(name: Optional[str] = None) -> list[dict]:
    from performance import performance
    d = now_ist().date().isoformat()
    return [t for t in list(performance.trades) if str(t.get("t", ""))[:10] == d
            and (name is None or t.get("strategy") == name)]


def _jobs() -> list[dict]:
    try:
        from platform_scheduler import platform_scheduler
        return platform_scheduler.jobs()
    except Exception:
        return []


def _kite() -> dict:
    try:
        import main
        return main.kite_connection()
    except Exception as exc:
        return {"connected": False, "error": str(exc)[:120]}


def _card(name: str, a, jobs: list[dict], drift: Optional[dict] = None) -> dict:
    import approvals
    import risk_budget
    import trading_state
    from agents.strategy_agents import PROVEN
    st = a.get_status()
    lab = risk_budget.lab_stats(name)
    live = st.get("live") or {}
    n = int(live.get("n") or live.get("trades") or 0)
    target = READINESS_TARGET.get(name)
    if name == "pead_h60":
        try:
            from agents.validated_pead import READINESS
            target = READINESS.get("min_trades")
        except Exception:
            target = None
    rd = st.get("live_readiness") or {}
    rows_today = _today_rows(name)
    f = 1.0
    try:
        import drift_monitor
        f = drift_monitor._size_factor(a)
    except Exception:
        pass
    unreal = [p.get("unrealised") for p in st.get("positions") or [] if p.get("unrealised") is not None]
    tk = approvals.ticket_for(name, now_ist().date().isoformat())
    return {
        "name": name, "label": LABELS.get(name, name), "lab_id": st.get("lab_id"), "version": st.get("version"),
        "status": "proven" if PROVEN.get(name) else "candidate", "proven": bool(PROVEN.get(name)),
        "enabled": {"setting": bool(st.get("enabled")), "toggle": bot_state.is_agent_enabled(name)},
        "running": bool(st.get("running")), "entry_block": st.get("entry_block"),
        "trading_state": {"own": trading_state.strategy_state(name), "effective": trading_state.effective(name)},
        "allocator_scale": risk_budget.scale(name),
        "today": {"plan": st.get("day") or {}, "positions": st.get("positions") or [],
                  "trades": rows_today, "realised": st.get("realised_today"),
                  "unrealised": round(sum(unreal), 2) if unreal else 0.0,
                  "approval": tk},
        "performance": {"live": live, "drawdown": st.get("drawdown"),
                        "lab_expectation": st.get("lab_expectation"),
                        "lab_reference": {"net_mean_per_trade": round(float(lab.get("net_mean", 0)) * f, 2) if lab else None,
                                          "win_pct": lab.get("win_pct"), "trades_per_session": lab.get("trades_per_session"),
                                          "observed_max_dd": round(float(lab.get("observed_max_dd", 0)) * f, 2) if lab else None},
                        "live_mean_per_trade": round(float(live.get("net") or 0) / n, 2) if n else None},
        "readiness": {"ready": bool(rd.get("ready")), "trades": n, "target_trades": target,
                      "progress": round(min(1.0, n / target), 3) if target else None,
                      "criterion": rd.get("criteria") or rd.get("criterion"), "detail": rd},
        "drift": drift if drift is not None else {},
        "limits": st.get("limits"), "deviations": st.get("deviations"), "skip_counts": st.get("skip_counts"),
        "flags": st.get("flags"), "errors": st.get("errors"),
        "next_jobs": [j for j in jobs if j["id"].startswith(JOB_PREFIX.get(name, ("~",)))][:6],
    }


# ── read endpoints ────────────────────────────────────────────────────────────
@router.get("/health")
def v2_health():
    return {"status": "ok", "api_version": API_VERSION, "mode": settings.trading_mode,
            "time": now_ist().isoformat(timespec="seconds")}


@router.get("/overview")
def v2_overview():
    import approvals
    import risk_budget
    import trading_state
    from platform_scheduler import platform_scheduler
    agents = _agents()
    today = _today_rows()
    kept = set(bot_state.VALIDATED_AGENTS)
    unreal = 0.0
    n_pos = 0
    for a in agents.values():
        for p in a.positions():
            n_pos += 1
            try:
                u = a._mark(p)
                unreal += u or 0.0
            except Exception:
                pass
    rb = risk_budget.status()
    try:
        import drift_monitor
        dr = drift_monitor.status()["strategies"]
        warn = sum(len(d.get("warnings") or []) + len(d.get("alerts") or []) for d in dr.values())
    except Exception:
        warn = None
    now = now_ist()
    return json_safe({
        "api_version": API_VERSION, "time": now.isoformat(timespec="seconds"), "mode": settings.trading_mode,
        "paper_data_source": settings.paper_data_source, "kite": _kite(),
        "market": {"open": is_market_open(), "trading_day": is_trading_day(now.date()),
                   "holiday": holiday_name(now.date()), "next_session": next_trading_day(now.date()).isoformat()},
        "bot_running": platform_scheduler.bot_running(),
        "trading_state": {"global": trading_state.global_state(), "brake": trading_state.brake_state()},
        "approval_mode": approvals.mode_on(),
        "portfolio": {"realised_today": round(sum(float(t.get("net", 0)) for t in today if t.get("strategy") in kept), 2),
                      "unrealised": round(unreal, 2), "open_positions": n_pos,
                      "realised_since_brake_reset": rb["brake"]["net"], "drawdown": rb["brake"]["drawdown"],
                      "drawdown_pct": rb["brake"]["drawdown_pct"], "brake_level": rb["brake"]["state"],
                      "risk_capital": rb["brake"]["capital"]},
        "strategies": {n: {"lab_id": getattr(a, "LAB_ID", ""), "running": a.state.running,
                           "entry_block": a.entry_block_reason_readonly(),
                           "state": trading_state.effective(n), "positions": len(a.positions())}
                       for n, a in agents.items()},
        "drift_warnings": warn,
    })


@router.get("/strategies")
def v2_strategies():
    jobs = _jobs()
    try:
        import drift_monitor
        dr = drift_monitor.status()["strategies"]
    except Exception:
        dr = {}
    return json_safe({"strategies": [_card(n, a, jobs, dr.get(n)) for n, a in _agents().items()]})


@router.get("/strategies/{name}")
def v2_strategy(name: str):
    a = _agents().get(name)
    if a is None:
        raise HTTPException(404, f"unknown strategy {name!r}")
    try:
        import drift_monitor
        dr = drift_monitor.strategy_drift(name, a)
    except Exception as exc:
        dr = {"error": str(exc)[:200]}
    card = _card(name, a, _jobs(), dr)
    card["day_log"] = a.get_status().get("day_log", [])
    return json_safe(card)


@router.get("/positions")
def v2_positions():
    from kite_client import kite_client
    import paper_book
    out = []
    for n, a in _agents().items():
        for p in a.positions():
            out.append({"strategy": n, "symbol": p.symbol, "exchange": p.exchange, "product": p.product,
                        "side": p.side, "qty": p.qty, "entry_px": p.entry_px, "entry_ref": p.entry_ref,
                        "opened": p.opened, "overnight": p.overnight, "group": p.group, "unrealised": a._mark(p)})
    try:
        rows = kite_client.positions().get("net", [])
    except Exception:
        rows = []
    return json_safe({"positions": out, "broker_rows": rows,
                      "orphan_rows": paper_book.status().get("orphan_rows", [])})


@router.get("/trades")
def v2_trades(strategy: Optional[str] = None, since: Optional[str] = None, n: int = 200):
    from performance import performance
    rows = [t for t in list(performance.trades)
            if (strategy is None or t.get("strategy") == strategy) and (since is None or str(t.get("t", "")) >= since)]
    n = max(1, min(int(n), 2000))
    return json_safe({"count": len(rows), "trades": rows[-n:][::-1],
                      "stats": {s: v for s, v in performance.strategy_stats().items()
                                if strategy is None or s == strategy}})


@router.get("/shadow-registry")
def v2_shadow_registry():
    import shadow_registry
    try:
        st = shadow_registry.ShadowRegistry().status()
    except Exception as exc:
        return {"enabled": shadow_registry.enabled(), "error": str(exc)[:200]}
    return json_safe({"enabled": shadow_registry.enabled(), "rule_hash": st.get("rule_hash"),
                      "n_active": st.get("n_active"), "n_family": st.get("n_family"),
                      "chain_ok": all(bool(v.get("ok", True)) if isinstance(v, dict) else bool(v)
                                      for v in (st.get("verify") or {}).values()) if isinstance(st.get("verify"), dict)
                      else st.get("verify"),
                      "shadows": st.get("shadows", {})})


@router.get("/forward-lab")
def v2_forward_lab():
    import forward_lab
    return json_safe(forward_lab.api_status())


@router.get("/builder/data")
def v2_builder_data():
    import builder_data
    return json_safe(builder_data.api_status())


@router.get("/forward-lab/{sid}")
def v2_forward_lab_strategy(sid: str):
    import forward_lab
    out = forward_lab.api_strategy(sid)
    if out is None:
        raise HTTPException(404, f"unknown forward-lab strategy {sid!r}")
    return json_safe(out)


@router.get("/archiver")
def v2_archiver():
    import option_bar_archiver
    return json_safe(option_bar_archiver.status())


@router.get("/risk-budget")
def v2_risk_budget():
    import risk_budget
    return json_safe(risk_budget.status())


@router.get("/drift")
def v2_drift():
    import drift_monitor
    return json_safe(drift_monitor.status())


@router.get("/fills")
def v2_fills(strategy: Optional[str] = None, n: int = 200):
    import drift_monitor
    return json_safe({"fills": drift_monitor.recent_fills(strategy, max(1, min(int(n), 2000)))[::-1]})


@router.get("/trading-state")
def v2_trading_state():
    import trading_state
    return trading_state.snapshot()


@router.get("/approvals")
def v2_approvals():
    import approvals
    return approvals.status()


@router.get("/readiness")
def v2_readiness():
    import live_readiness
    return live_readiness.check(kite_connected=_kite().get("connected"))


@router.get("/system")
def v2_system():
    import paper_book
    from tick_engine import tick_engine
    from sebi_compliance import sebi_compliance
    import live_readiness
    sebi = {}
    try:
        s = sebi_compliance.status()
        sebi = {k: s.get(k) for k in ("state", "kill_switch_state", "kill_reason", "pause_reason") if k in s} or s
    except Exception as exc:
        sebi = {"error": str(exc)[:120]}
    return json_safe({"version": __import__("main").VERSION, "api_version": API_VERSION,
                      "mode": settings.trading_mode, "time": now_ist().isoformat(timespec="seconds"),
                      "feed": tick_engine.status(), "kite": _kite(), "jobs": _jobs()[:40],
                      "paper_book": paper_book.status(), "sebi": sebi,
                      "live_readiness": live_readiness.check()})


# ── write endpoints (auth-protected by main.py's gate: POST needs X-API-Key or a Bearer JWT) ──────────────────
class TradingStateRequest(BaseModel):
    scope: Literal["global", "strategy"] = "global"
    strategy: Optional[str] = None
    state: Literal["ACTIVE", "REDUCE_ONLY", "HALTED"]
    reason: str = Field(default="", max_length=200)


@router.post("/trading-state")
def v2_set_trading_state(req: TradingStateRequest, request: Request):
    import trading_state
    by = _who(request)
    if req.scope == "global":
        ev = trading_state.set_global(req.state, req.reason, by)
    else:
        if req.strategy not in _agents():
            raise HTTPException(422, f"unknown strategy {req.strategy!r}")
        ev = trading_state.set_strategy(req.strategy, req.state, req.reason, by)
    return {"changed": ev, **trading_state.snapshot()}


class ApprovalRequest(BaseModel):
    strategy: str
    day: Optional[str] = None          # YYYY-MM-DD, default today
    deadline: Optional[str] = None     # ISO datetime (IST), default the strategy's entry time that day
    note: str = Field(default="", max_length=200)


@router.post("/approvals")
def v2_open_approval(req: ApprovalRequest, request: Request):
    import approvals
    if req.strategy not in _agents():
        raise HTTPException(422, f"unknown strategy {req.strategy!r}")
    try:
        if req.day:
            datetime.fromisoformat(req.day)
        t = approvals.request(req.strategy, req.day, req.deadline, req.note, by=_who(request))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"ticket": t, **approvals.status()}


class DecisionRequest(BaseModel):
    note: str = Field(default="", max_length=200)


def _decide(ticket_id: str, approve: bool, req: DecisionRequest, request: Request) -> dict:
    import approvals
    try:
        t = approvals.decide(ticket_id, approve, by=_who(request), note=req.note)
    except KeyError:
        raise HTTPException(404, "unknown ticket")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return {"ticket": t, **approvals.status()}


@router.post("/approvals/{ticket_id}/approve")
def v2_approve(ticket_id: str, request: Request, req: DecisionRequest = DecisionRequest()):
    return _decide(ticket_id, True, req, request)


@router.post("/approvals/{ticket_id}/reject")
def v2_reject(ticket_id: str, request: Request, req: DecisionRequest = DecisionRequest()):
    return _decide(ticket_id, False, req, request)


class ModeRequest(BaseModel):
    on: bool


@router.post("/approvals/mode")
def v2_approval_mode(req: ModeRequest, request: Request):
    import approvals
    return approvals.set_mode(req.on, by=_who(request))


class EnableRequest(BaseModel):
    enabled: bool


@router.post("/strategies/{name}/enable")
def v2_enable(name: str, req: EnableRequest, request: Request):
    a = _agents().get(name)
    if a is None:
        raise HTTPException(404, f"unknown strategy {name!r}")
    bot_state.set_agent_enabled(name, req.enabled)
    if not req.enabled and a.state.running:
        a.stop()
    elif req.enabled and not a.state.running:
        a.auto_start()          # trading day and no operator-wide Stop -> running again; otherwise 08:55 preflight
    from loguru import logger
    logger.warning("[v2] {} {} by {}", name, "enabled" if req.enabled else "disabled", _who(request))
    return {"name": name, "enabled": bot_state.is_agent_enabled(name), "setting": a.enabled_by_setting(),
            "running": a.state.running}


class BrakeResetRequest(BaseModel):
    reason: str = Field(default="manual reset", max_length=200)


@router.post("/risk/brake/reset")
def v2_brake_reset(request: Request, req: BrakeResetRequest = BrakeResetRequest()):
    import risk_budget
    return json_safe(risk_budget.reset_brake(by=_who(request), reason=req.reason))
