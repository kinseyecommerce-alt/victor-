"""
live_readiness.py — read-only report of the LIVE prerequisites. It never switches anything to LIVE.

Since 1 April 2026 Kite Connect accepts API orders only from a static outbound IP registered in the developer
console. Railway's default egress IP is not static, so LIVE needs either Railway's static egress IP add-on or a host
with a fixed IP (AWS Mumbai + Elastic IP, deploy/AWS.md); the IP must be put in KITE_STATIC_IP and registered with
Kite. This module checks that and the other prerequisites; infrastructure is not set up here.

LIVE stays impossible for the kept strategies regardless: every validated agent refuses LIVE
(entry_block_reason "live_mode_refused", paper helpers raise, kite_client.paper_fill_at raises in LIVE).
"""
from __future__ import annotations

import ipaddress

from config import settings


def _ip_ok(v: str) -> bool:
    try:
        ip = ipaddress.ip_address((v or "").strip())
        return not (ip.is_private or ip.is_loopback or ip.is_unspecified or ip.is_link_local)
    except ValueError:
        return False


def check(kite_connected: bool | None = None) -> dict:
    """Every LIVE prerequisite with ok / detail; ready_for_live is False while any blocker remains."""
    from sebi_compliance import sebi_compliance
    items = []

    def add(key: str, ok: bool, detail: str, blocker: bool = True):
        items.append({"check": key, "ok": bool(ok), "blocker": blocker, "detail": detail})

    add("static_ip", _ip_ok(settings.kite_static_ip),
        f"KITE_STATIC_IP={settings.kite_static_ip or '(unset)'}: Kite requires a whitelisted static outbound IP "
        "for API orders (from 1 Apr 2026); register it in the Kite developer console")
    add("api_key", bool(settings.api_key), "API_KEY set (protects every write endpoint)")
    add("jwt_secret", bool(settings.jwt_secret_key), "JWT_SECRET_KEY set (dashboard login)")
    add("kill_switch_reset_secret", bool(settings.kill_switch_reset_secret),
        "KILL_SWITCH_RESET_SECRET set (resetting the SEBI kill switch needs it)")
    add("kite_keys", bool(settings.kite_api_key and settings.kite_api_secret), "Kite API key / secret configured")
    add("kite_auto_login", bool(settings.kite_user_id and settings.kite_password and settings.kite_totp_secret),
        "daily automatic Kite login configured (user id, password, TOTP seed)", blocker=False)
    if kite_connected is not None:
        add("kite_session", kite_connected, "a working Kite session today", blocker=False)
    try:
        wl = list(getattr(sebi_compliance, "_whitelisted_ips", []) or [])
    except Exception:
        wl = []
    add("admin_ip_whitelist", bool(wl), "SEBI admin IP whitelist configured (kill switch / resume endpoints)",
        blocker=False)
    add("telegram", bool(settings.telegram_bot_token and settings.telegram_chat_id),
        "Telegram alerts configured", blocker=False)
    add("approval_mode", bool(_approval_mode()), "semi-auto approval mode ON for the first LIVE weeks", blocker=False)
    strategies = {}
    try:
        from agents.strategy_agents import ALL_AGENTS
        for n, a in ALL_AGENTS.items():
            strategies[n] = {"live_capable": False,
                             "why": "PAPER-only validated agent: refuses LIVE (live_mode_refused); lab rule not "
                                    "executable as tested / not proven for real money"}
    except Exception:
        pass
    add("strategies_live_capable", False, "no kept strategy may trade LIVE (every validated agent refuses LIVE)")
    blockers = [i["check"] for i in items if i["blocker"] and not i["ok"]]
    return {"mode": settings.trading_mode, "ready_for_live": not blockers, "blockers": blockers, "checks": items,
            "strategies": strategies,
            "note": "read-only; nothing here switches to LIVE. LIVE stays impossible for the kept strategies."}


def _approval_mode() -> bool:
    try:
        import approvals
        return approvals.mode_on()
    except Exception:
        return False
