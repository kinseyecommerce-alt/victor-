"""
alerts.py — the owner hears about problems without reading logs (PAPER and LIVE alike; never places orders).

1. Error forwarding: every ERROR (and CRITICAL) log line is sent to Telegram, rate limited (the same message at most
   once per ALERT_REPEAT_S, at most ALERT_MAX_PER_DAY a day, then one "muted" note). Off unless ALERTS_ENABLED.
2. Health checks: `checks()` lists what must hold right now (scheduler running; on a trading day from 08:58 a Kite
   session; in market hours a fresh price tick). `/health` reports them; `/health/strict` answers HTTP 503 when one
   fails, for an outside uptime monitor (one that is not on this server) to call every few minutes.
3. Daily all-clear: a morning message (08:58, trading days) and an evening message (15:50) — so a missing message is
   itself the alarm.

Telegram needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID; without them nothing is sent (checks still work).
"""
from __future__ import annotations

import queue
import re
import threading
import time
from datetime import datetime, time as dtime
from typing import Optional

import httpx
from loguru import logger

from config import settings

ALERT_REPEAT_S = 1800          # the same message (digits ignored) at most once per 30 minutes
ALERT_MAX_PER_DAY = 30
TICK_STALE_S = 300             # in market hours a price tick older than this fails the check

_q: "queue.Queue[str]" = queue.Queue(maxsize=200)
_seen: dict[str, float] = {}
_day = {"date": "", "n": 0, "muted": False}
_lock = threading.Lock()
_installed = {"sink": None, "thread": None}


def enabled() -> bool:
    return bool(getattr(settings, "alerts_enabled", False))


def telegram_ready() -> bool:
    return bool(settings.telegram_bot_token and settings.telegram_chat_id)


def _key(msg: str) -> str:
    return re.sub(r"\d+", "#", msg)[:120]


def should_send(msg: str, now: Optional[float] = None, today: Optional[str] = None) -> Optional[str]:
    """The text to send for one error line, or None when rate limited."""
    now = time.time() if now is None else now
    today = today or datetime.now().date().isoformat()
    with _lock:
        if _day["date"] != today:
            _day.update({"date": today, "n": 0, "muted": False})
        k = _key(msg)
        if now - _seen.get(k, 0.0) < ALERT_REPEAT_S:
            return None
        if _day["n"] >= ALERT_MAX_PER_DAY:
            if _day["muted"]:
                return None
            _day["muted"] = True
            return f"⚠️ {ALERT_MAX_PER_DAY} alerts today: further errors are muted until tomorrow (see the logs)"
        _seen[k] = now
        _day["n"] += 1
        return "⚠️ " + msg[:900]


def _sink(message) -> None:
    rec = message.record
    if rec["level"].no < 40 or rec["name"] == __name__:
        return
    text = should_send(f"{rec['name']}: {rec['message']}")
    if text:
        try:
            _q.put_nowait(text)
        except queue.Full:
            pass


def send_now(text: str) -> bool:
    """Send one plain-text Telegram message (synchronous). False when not configured or it failed."""
    if not telegram_ready():
        return False
    try:
        r = httpx.post(f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                       json={"chat_id": settings.telegram_chat_id, "text": text[:4000],
                             "disable_web_page_preview": True}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _worker() -> None:
    while True:
        text = _q.get()
        send_now(text)


def install() -> bool:
    """Register the error sink and the sender thread once (only when ALERTS_ENABLED)."""
    if not enabled() or _installed["sink"] is not None:
        return False
    _installed["sink"] = logger.add(_sink, level="ERROR", enqueue=False)
    t = threading.Thread(target=_worker, name="alerts", daemon=True)
    t.start()
    _installed["thread"] = t
    return True


# ── health checks ────────────────────────────────────────────────────────────
def checks(now: Optional[datetime] = None) -> list[dict]:
    from ist_clock import now_ist, is_trading_day
    now = now or now_ist()
    out = []

    def add(name: str, ok: bool, detail: str) -> None:
        out.append({"check": name, "ok": bool(ok), "detail": detail})
    try:
        from platform_scheduler import platform_scheduler
        running = bool(platform_scheduler._sched.running)
    except Exception:
        running = False
    add("scheduler", running, "job scheduler running")
    trading = is_trading_day(now.date())
    t = now.time()
    if trading and dtime(8, 58) <= t <= dtime(15, 35):
        try:
            from kite_client import kite_client
            ok = bool(kite_client._kite is not None)
        except Exception:
            ok = False
        add("kite_session", ok, "a Kite session exists (daily login done)")
    if trading and dtime(9, 16) <= t <= dtime(15, 29):
        try:
            from tick_engine import tick_engine
            age = tick_engine.status().get("last_tick_age_s")
        except Exception:
            age = None
        add("price_feed", age is not None and age <= TICK_STALE_S,
            f"newest price tick {age if age is not None else '—'} s old (limit {TICK_STALE_S} s)")
    return out


def health_summary(now: Optional[datetime] = None) -> dict:
    cs = checks(now)
    bad = [c["check"] for c in cs if not c["ok"]]
    return {"healthy": not bad, "failing": bad, "checks": cs}


# ── daily all-clear ──────────────────────────────────────────────────────────
def _agents_line() -> str:
    from agents.strategy_agents import ALL_AGENTS
    parts = []
    for n, a in ALL_AGENTS.items():
        try:
            block = a.entry_block_reason_readonly()
        except Exception as exc:
            block = f"error {exc}"
        parts.append(f"{n}: {'ready' if block is None else block}, open {len(a.positions())}")
    return "\n".join(parts)


def morning_text(now: Optional[datetime] = None) -> str:
    h = health_summary(now)
    head = "✅ Morning check: all clear" if h["healthy"] else f"❌ Morning check FAILED: {', '.join(h['failing'])}"
    return f"{head} (PAPER={settings.trading_mode == 'PAPER'})\n{_agents_line()}"


def evening_text(now: Optional[datetime] = None) -> str:
    from agents.strategy_agents import ALL_AGENTS
    lines, total = [], 0.0
    for n, a in ALL_AGENTS.items():
        try:
            r = float(a.realised_today())
        except Exception:
            r = 0.0
        total += r
        errs = len(getattr(a.state, "errors", []) or [])
        devs = sum((getattr(a, "deviations", {}) or {}).values()) if isinstance(getattr(a, "deviations", None),
                                                                                  dict) else 0
        lines.append(f"{n}: realised ₹{r:,.0f}, open {len(a.positions())}, errors today {errs}, deviations (all time) {devs}")
    h = health_summary(now)
    head = "✅ Evening report" if h["healthy"] else f"❌ Evening report — failing: {', '.join(h['failing'])}"
    return f"{head}\nRealised today ₹{total:,.0f}\n" + "\n".join(lines)
