"""
drift_monitor.py — live-vs-lab drift and fill-quality monitor. WARNINGS ONLY: it never places, blocks or resizes an
order, never changes a trading state and never touches the shadow registry's pre-registered promotion / demotion
rule. Its numbers are for the operator (v2 API, optional Telegram).

Per strategy (current VERSION's ledger rows; lab reference = lab_risk.json at lab size, money figures scaled by the
strategy's current size factor):
  frequency     live trades per trading session since its first live trade vs the lab's trades per session
                (warn outside 0.5×-2× once ≥ 20 sessions are live)
  fill quality  entry slippage vs the price the rule saw (ledger slip_bps: the auction fade's fill IS the official
                open, so its slippage is 0 by construction; attention pays the 15:21 touch) and the paper broker's
                quote-vs-fill log (fill_log.jsonl), against the lab fill model's slippage (warn when the mean entry
                slippage exceeds the lab's by > 5 bps over ≥ 20 trades)
  expectancy    mean net of the last 20 trades vs the lab's bootstrap band of 20-trade means (warn below p5)
  drawdown      live drawdown vs the lab's stationary block-bootstrap drawdown percentiles (warn ≥ p95, alert ≥ p99)
Telegram (DRIFT_TELEGRAM=true, default off): at most one message per strategy per day, from the 17:15 job.
"""
from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

MIN_SESSIONS = 20
MIN_TRADES = 20
FREQ_BAND = (0.5, 2.0)
SLIP_TOLERANCE_BPS = 5.0
_fills: deque = deque(maxlen=2000)
_lock = threading.Lock()


def _dir() -> Path:
    return Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json").parent


def fill_log_path() -> Path:
    return Path(os.getenv("FILL_LOG_FILE") or _dir() / "fill_log.jsonl")


def state_path() -> Path:
    return Path(os.getenv("DRIFT_STATE_FILE") or _dir() / "drift_state.json")


# ── fill log (quote vs fill; written by the paper order helpers and the LIVE protected-order path) ────────────

def record_fill(agent: Optional[str], symbol: str, side: str, qty: int, quote_px: float, fill: float,
                kind: str) -> dict:
    adverse = (fill - quote_px) if side == "BUY" else (quote_px - fill)
    row = {"t": now_ist().isoformat(timespec="seconds"), "agent": agent or "", "symbol": symbol, "side": side,
           "qty": int(qty), "quote": round(float(quote_px), 4), "fill": round(float(fill), 4), "kind": kind,
           "slip_bps": round(adverse / quote_px * 1e4, 2) if quote_px else None, "mode": settings.trading_mode}
    with _lock:
        _fills.append(row)
        try:
            p = fill_log_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception as exc:
            logger.debug("[drift] fill log not written: {}", exc)
    return row


def recent_fills(agent: Optional[str] = None, n: int = 200) -> list[dict]:
    rows = list(_fills)
    if not rows:
        try:
            with open(fill_log_path()) as fh:
                rows = [json.loads(x) for x in deque(fh, maxlen=2000) if x.strip()]
        except Exception:
            rows = []
    if agent:
        rows = [r for r in rows if r.get("agent") == agent]
    return rows[-n:]


# ── per-strategy drift ────────────────────────────────────────────────────────

def _size_factor(a) -> float:
    try:
        base = a._lot_scale() if hasattr(a, "_lot_scale") else a._scale()
        return float(base) * float(a.alloc_scale())
    except Exception:
        return 1.0


def _sessions_between(d0: date, d1: date) -> int:
    from ist_clock import is_trading_day
    n, d = 0, d0
    while d <= d1:
        n += 1 if is_trading_day(d) else 0
        d += timedelta(days=1)
    return n


def strategy_drift(name: str, a=None) -> dict:
    import risk_budget
    if a is None:
        from agents.strategy_agents import ALL_AGENTS
        a = ALL_AGENTS[name]
    lab = risk_budget.lab_stats(name)
    rows = sorted(a.ledger_rows(), key=lambda r: str(r.get("t", "")))
    f = _size_factor(a)
    out = {"lab_id": getattr(a, "LAB_ID", ""), "trades": len(rows), "size_factor": round(f, 4), "warnings": [],
           "alerts": [], "lab_available": bool(lab)}
    if not lab:
        out["warnings"].append("no lab reference in lab_risk.json")
        return out
    today = now_ist().date()
    # frequency
    if rows:
        first = date.fromisoformat(str(rows[0]["t"])[:10])
        sess = max(1, _sessions_between(first, today))
        live_tps = len(rows) / sess
        ratio = live_tps / lab["trades_per_session"] if lab.get("trades_per_session") else None
        out["frequency"] = {"live_sessions": sess, "live_per_session": round(live_tps, 4),
                            "lab_per_session": lab.get("trades_per_session"),
                            "ratio": round(ratio, 3) if ratio is not None else None}
        if ratio is not None and sess >= MIN_SESSIONS and not (FREQ_BAND[0] <= ratio <= FREQ_BAND[1]):
            out["warnings"].append(f"trade frequency {ratio:.2f}× the lab's ({live_tps:.3f} vs "
                                   f"{lab['trades_per_session']:.3f} per session over {sess} sessions)")
    else:
        out["frequency"] = {"live_sessions": 0, "live_per_session": 0.0, "lab_per_session": lab.get("trades_per_session"),
                            "ratio": None}
    # fill quality
    slips = [float(r["slip_bps"]) for r in rows if r.get("slip_bps") is not None]
    fills = [x["slip_bps"] for x in recent_fills(name) if x.get("slip_bps") is not None]
    ref = lab.get("ref_slippage_bps")
    out["fill_quality"] = {"lab_fill_model": lab.get("fill_model"), "lab_slippage_bps": ref,
                           "entry_slip_bps_mean": round(sum(slips) / len(slips), 2) if slips else None,
                           "entry_slip_n": len(slips),
                           "quote_vs_fill_bps_mean": round(sum(fills) / len(fills), 2) if fills else None,
                           "quote_vs_fill_n": len(fills)}
    if ref is not None and len(slips) >= MIN_TRADES and sum(slips) / len(slips) > ref + SLIP_TOLERANCE_BPS:
        out["warnings"].append(f"mean entry slippage {sum(slips) / len(slips):.1f} bps vs lab {ref} bps")
    # rolling expectancy vs the lab bootstrap band
    band = {k: (v * f if v is not None else None) for k, v in (lab.get("mean20_band") or {}).items()}
    last = [float(r.get("net", 0)) for r in rows[-20:]]
    m20 = sum(last) / len(last) if last else None
    out["expectancy"] = {"last20_mean": round(m20, 2) if m20 is not None else None, "n": len(last),
                         "lab_band": {k: round(v, 2) for k, v in band.items() if v is not None},
                         "lab_mean": round(float(lab.get("net_mean", 0)) * f, 2)}
    if m20 is not None and len(last) >= MIN_TRADES and band.get("p5") is not None and m20 < band["p5"]:
        out["warnings"].append(f"last-20-trade mean ₹{m20:,.0f} below the lab band p5 ₹{band['p5']:,.0f}")
    # drawdown vs Monte Carlo bands
    dd = float(a.drawdown())
    bdd = {k: (v * f if v is not None else None) for k, v in (lab.get("block_dd") or {}).items()}
    out["drawdown"] = {"live": round(dd, 2), "lab_block_dd": {k: round(v, 2) for k, v in bdd.items() if v is not None},
                       "lab_observed_max_dd": round(float(lab.get("observed_max_dd", 0)) * f, 2)}
    if bdd.get("p99") and dd >= bdd["p99"]:
        out["alerts"].append(f"live drawdown ₹{dd:,.0f} ≥ lab block-bootstrap p99 ₹{bdd['p99']:,.0f}")
    elif bdd.get("p95") and dd >= bdd["p95"]:
        out["warnings"].append(f"live drawdown ₹{dd:,.0f} ≥ lab block-bootstrap p95 ₹{bdd['p95']:,.0f}")
    out["status"] = "alert" if out["alerts"] else ("warn" if out["warnings"] else "ok")
    return out


def status() -> dict:
    from agents.strategy_agents import ALL_AGENTS
    out = {}
    for n, a in ALL_AGENTS.items():
        try:
            out[n] = strategy_drift(n, a)
        except Exception as exc:
            out[n] = {"error": str(exc)[:200], "warnings": [], "alerts": [], "status": "error"}
    st = _load_state()
    return {"strategies": out, "telegram": bool(getattr(settings, "drift_telegram", False)),
            "last_sent": st.get("last_sent", {}), "note": "warnings only — never trades, never changes a state"}


def _load_state() -> dict:
    try:
        return json.loads(state_path().read_text())
    except Exception:
        return {}


def _save_state(d: dict) -> None:
    try:
        p = state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, p)
    except Exception as exc:
        logger.debug("[drift] state not saved: {}", exc)


async def daily_job() -> dict:
    """17:15: evaluate; with DRIFT_TELEGRAM on, one message per strategy per day when it has warnings / alerts."""
    st = status()
    today = now_ist().date().isoformat()
    state = _load_state()
    sent = state.setdefault("last_sent", {})
    msgs = []
    for n, d in st["strategies"].items():
        issues = (d.get("alerts") or []) + (d.get("warnings") or [])
        if not issues or d.get("trades", 0) == 0:
            continue
        logger.warning("[drift] {}: {}", n, "; ".join(issues))
        if getattr(settings, "drift_telegram", False) and sent.get(n) != today:
            sent[n] = today
            msgs.append(f"⚠️ <b>Drift {n}</b> ({d.get('lab_id')})\n" + "\n".join(f"• {x}" for x in issues[:5]))
    state["last_run"] = today
    _save_state(state)
    if msgs:
        from agents.base_agent import send_telegram
        for m in msgs:
            await send_telegram(m)
    return {"evaluated": len(st["strategies"]), "telegram_sent": len(msgs)}
