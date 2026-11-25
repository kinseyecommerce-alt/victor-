"""
risk_budget.py — the risk-budget allocator (replaces the removed LLM allocator) and the portfolio drawdown brake.

Per-strategy size factor (scale):
    lab_std_i   = the strategy's lab daily net P&L standard deviation at lab size (lab_risk.json, built by
                  lab_risk_build.py from the 2023-26 lab trade files)
    budget      = RISK_BUDGET_DAILY_VOL (₹) split equally over the enabled kept strategies (equal risk)
    scale_i     = min(cap, budget_i / lab_std_i)   with cap = 1.0, or RISK_BUDGET_MAX_SCALE when
                  RISK_BUDGET_ALLOW_UPSCALE is true
    RISK_BUDGET_DAILY_VOL = 0 (default) → scale 1.0 for every strategy: the lab's own sizing, so paper results stay
    comparable with the lab. The allocator can only scale DOWN unless upscaling is explicitly allowed. The agents'
    own hard caps (VALIDATED_* gross / margin / loss limits) remain the upper bounds in every case.

Portfolio drawdown brake (latching):
    realised net P&L of the kept strategies in the ledger (since the last brake reset), drawdown from its running
    peak (start 0), in % of RISK_CAPITAL:
        ≥ RISK_BRAKE_REDUCE_PCT → brake REDUCE_ONLY (every strategy: no new entries, exits by rule)
        ≥ RISK_BRAKE_HALT_PCT   → brake HALTED      (risk exits only — see trading_state.py)
    Checked every minute by platform_scheduler (and on status reads, read-only). The brake only escalates by
    itself; POST /api/v2/risk/brake/reset sets it back to ACTIVE and restarts the measurement from that moment.

Read-only status: GET /api/v2/risk-budget.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

LAB_RISK_FILE = Path(__file__).resolve().parent / "lab_risk.json"


@lru_cache(maxsize=1)
def lab_risk() -> dict:
    try:
        return json.loads(LAB_RISK_FILE.read_text())
    except Exception as exc:
        logger.error("[risk_budget] {} unreadable: {}", LAB_RISK_FILE, exc)
        return {"strategies": {}}


def _agents() -> dict:
    from agents.strategy_agents import ALL_AGENTS
    return ALL_AGENTS


def lab_stats(name: str) -> dict:
    a = _agents().get(name)
    lab_id = getattr(a, "LAB_ID", "") if a is not None else ""
    return dict((lab_risk().get("strategies") or {}).get(lab_id) or {})


def _active_names() -> list[str]:
    import bot_state
    out = []
    for n, a in _agents().items():
        try:
            on = a.enabled_by_setting() and bot_state.is_agent_enabled(n)
        except Exception:
            on = False
        if on:
            out.append(n)
    return out


def cap() -> float:
    if getattr(settings, "risk_budget_allow_upscale", False):
        return max(0.0, float(getattr(settings, "risk_budget_max_scale", 1.0) or 1.0))
    return 1.0


def scale(name: str) -> float:
    """The strategy's size factor (1.0 = lab sizing)."""
    budget = float(getattr(settings, "risk_budget_daily_vol", 0.0) or 0.0)
    if budget <= 0:
        return min(1.0, cap())
    active = _active_names() or [name]
    std = float(lab_stats(name).get("daily_std") or 0.0)
    if std <= 0:
        return min(1.0, cap())               # no lab volatility on file: never upsize an unknown
    return round(max(0.0, min(cap(), budget / len(active) / std)), 4)


# ── drawdown brake ────────────────────────────────────────────────────────────

def _kept_rows(since: str = "") -> list[dict]:
    import bot_state
    from performance import performance
    return [t for t in list(performance.trades) if t.get("strategy") in bot_state.VALIDATED_AGENTS
            and str(t.get("t", "")) >= since]


def drawdown(since: str = "") -> dict:
    cum = peak = 0.0
    for t in sorted(_kept_rows(since), key=lambda r: str(r.get("t", ""))):
        cum += float(t.get("net", 0) or 0)
        peak = max(peak, cum)
    dd = peak - cum
    capital = float(getattr(settings, "risk_capital", 0.0) or 0.0)
    return {"net": round(cum, 2), "peak": round(peak, 2), "drawdown": round(dd, 2),
            "drawdown_pct": round(100 * dd / capital, 3) if capital > 0 else None, "capital": capital}


def brake_level(dd_pct: Optional[float]) -> str:
    if dd_pct is None:
        return "ACTIVE"
    if dd_pct >= float(settings.risk_brake_halt_pct):
        return "HALTED"
    if dd_pct >= float(settings.risk_brake_reduce_pct):
        return "REDUCE_ONLY"
    return "ACTIVE"


def _since() -> str:
    import trading_state
    return str((trading_state.snapshot()["brake"] or {}).get("since") or "")


def apply_brake() -> Optional[dict]:
    """Escalate the brake when the drawdown crossed a threshold (never de-escalates by itself)."""
    import trading_state
    cur = trading_state.brake_state()
    since = _since()
    dd = drawdown(since)
    lvl = brake_level(dd["drawdown_pct"])
    rank = {s: i for i, s in enumerate(trading_state.STATES)}
    if rank[lvl] <= rank[cur]:
        return None
    reason = (f"portfolio drawdown ₹{dd['drawdown']:,.0f} = {dd['drawdown_pct']:.2f} % of ₹{dd['capital']:,.0f} "
              f"(reduce {settings.risk_brake_reduce_pct} % / halt {settings.risk_brake_halt_pct} %)")
    ev = trading_state.set_brake(lvl, reason, by="risk_budget", extra={"since": since, "drawdown": dd})
    try:
        import asyncio
        from agents.base_agent import send_telegram
        asyncio.get_running_loop().create_task(send_telegram(f"⛔ <b>Drawdown brake: {lvl}</b>\n{reason}"))
    except Exception:
        pass
    return ev


def reset_brake(by: str = "operator", reason: str = "manual reset") -> dict:
    """Brake back to ACTIVE; the drawdown is measured afresh from now."""
    import trading_state
    since = now_ist().isoformat(timespec="seconds")
    trading_state.set_brake("ACTIVE", reason, by=by, extra={"since": since})
    return status()


def status() -> dict:
    import trading_state
    since = _since()
    dd = drawdown(since)
    strategies = {}
    active = _active_names()
    for n, a in _agents().items():
        ls = lab_stats(n)
        strategies[n] = {"lab_id": getattr(a, "LAB_ID", ""), "active": n in active, "scale": scale(n),
                         "lab_daily_std": ls.get("daily_std"), "lab_max_dd": ls.get("observed_max_dd"),
                         "lab_block_dd_p99": (ls.get("block_dd") or {}).get("p99"),
                         "hard_caps": _hard_caps(a)}
    return {"mode": "lab sizing (scale 1.0)" if float(settings.risk_budget_daily_vol or 0) <= 0
            else f"risk budget ₹{float(settings.risk_budget_daily_vol):,.0f} daily vol, equal risk",
            "daily_vol_budget": float(settings.risk_budget_daily_vol or 0), "cap": cap(),
            "upscale_allowed": bool(settings.risk_budget_allow_upscale), "strategies": strategies,
            "brake": {"state": trading_state.brake_state(), "since": since or None, **dd,
                      "reduce_pct": float(settings.risk_brake_reduce_pct),
                      "halt_pct": float(settings.risk_brake_halt_pct),
                      "would_be": brake_level(dd["drawdown_pct"])},
            "lab_combined": (lab_risk().get("combined_default_book") or {}),
            "lab_built": lab_risk().get("built")}


def _hard_caps(a) -> dict:
    try:
        st = a.get_status().get("limits") or {}
    except Exception:
        st = {}
    n = getattr(a, "name", "")
    extra = {"auc_fade": {"max_gross": settings.validated_auction_max_gross},
             "attn_on": {"max_gross": settings.validated_attention_max_gross},
             "vrp_winged": {"lots": settings.validated_vrp_lots, "max_margin": settings.validated_vrp_max_margin,
                            "max_loss": settings.validated_vrp_max_loss},
             "vrp_naked": {"lots": settings.validated_vrp_naked_lots,
                           "max_margin": settings.validated_vrp_naked_max_margin},
             "pead_h60": {"max_gross": settings.validated_pead_max_gross,
                          "max_positions": settings.validated_pead_max_positions}}.get(n, {})
    return {**st, **extra}
