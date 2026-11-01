"""n8n webhook bridge — fire-and-forget outbound events to n8n.

Usage at call sites (always use create_task — never await in hot paths):
    from n8n_bridge import notify as _n8n
    asyncio.create_task(_n8n("trade_entry", {...}))
"""
import asyncio
import hashlib
import hmac
import json

import httpx
from loguru import logger

from config import settings
from ist_clock import now_ist

# Shared client reuses TCP connections across many notify() calls
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    return _client


async def notify(event_type: str, data: dict) -> None:
    """POST a structured event to the configured n8n webhook URL.

    Never raises. Never blocks the caller. Silently no-ops when
    N8N_WEBHOOK_URL is not configured.
    """
    url = settings.n8n_webhook_url
    if not url:
        return

    payload = {
        "event":        event_type,
        "timestamp":    now_ist().isoformat(),
        "trading_mode": settings.trading_mode,
        "data":         data,
    }
    body = json.dumps(payload, default=str).encode()
    headers = {"Content-Type": "application/json"}

    if settings.n8n_webhook_secret:
        sig = hmac.new(
            settings.n8n_webhook_secret.encode(), body, hashlib.sha256
        ).hexdigest()
        headers["X-AlgoTrader-Signature"] = f"sha256={sig}"

    client = _get_client()
    for attempt in range(2):
        try:
            r = await client.post(url, content=body, headers=headers)
            if r.status_code < 300:
                return
            # 4xx/5xx — n8n responded; don't retry (bad payload, not network)
            logger.warning(
                "[n8n] HTTP {} for event={} (attempt {})",
                r.status_code, event_type, attempt + 1,
            )
            break
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            if attempt == 0:
                await asyncio.sleep(2)
            else:
                logger.debug("[n8n] delivery failed ({}): {}", event_type, exc)
        except Exception as exc:
            logger.debug("[n8n] unexpected error ({}): {}", event_type, exc)
            break
