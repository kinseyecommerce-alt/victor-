"""
nse_http.py — the NSE public-site JSON client (cookie handshake + 8 req/s throttle, NSE circular 54/2024).

The only piece of the removed market_data.py the kept system needs: event_calendar.py reads NSE's corporate-action
and event-calendar feeds through it (the validated agents' corporate-action guard).
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import httpx
from loguru import logger

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}
NSE_HOME = "https://www.nseindia.com/"


class NSEClient:
    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._session_ok = False
        self._last_session = 0.0
        self._last_req_ts = 0.0
        self._min_interval = 0.125

    async def _ensure_session(self) -> None:
        now = time.time()
        if self._session_ok and (now - self._last_session) < 300:
            return
        if self._client:
            await self._client.aclose()
        self._client = httpx.AsyncClient(headers=NSE_HEADERS, timeout=10, follow_redirects=True)
        try:
            await self._client.get(NSE_HOME)
            self._session_ok, self._last_session = True, now
        except Exception as exc:
            logger.warning("NSE session init failed: {}", exc)
            self._session_ok = False

    async def get(self, url: str) -> Optional[dict]:
        await self._ensure_session()
        wait = self._min_interval - (time.monotonic() - self._last_req_ts)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_req_ts = time.monotonic()
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                self._session_ok = False
            return None
        except Exception:
            return None

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()


nse_client = NSEClient()
