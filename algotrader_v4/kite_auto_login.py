"""
kite_auto_login.py — the daily Zerodha Kite Connect login, done without a browser.

Zerodha issues a new access token every trading day and only after the user's ID, password
and TOTP are entered on kite.zerodha.com. This module walks Zerodha's own login endpoints the
way the login page does (user ID + password, then TOTP, then the Kite Connect redirect) and
exchanges the resulting request_token for an access token with the API secret.

Credentials come from the encrypted vault (Settings → API Keys): KITE_USER_ID, KITE_PASSWORD and
KITE_TOTP_SECRET (the base32 seed shown when enabling TOTP on Zerodha, not a 6-digit code).
Nothing is typed in here — the scheduler calls it at 08:45 IST every day (weekends too, for research data), again at 08:55 and
09:05 if it failed, and once at startup when there is no valid token for the day.

Note: your Zerodha password and TOTP seed stored on the server let anyone with the server AND the
vault master key log in to your account. Keep SECRETS_MASTER_KEY out of the repo and the image.
"""
from __future__ import annotations

import asyncio
import re
import threading
from datetime import datetime
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

import httpx
import pyotp
from loguru import logger

from config import settings
from ist_clock import now_ist

KITE = "https://kite.zerodha.com"
REQUIRED = ("kite_api_key", "kite_api_secret", "kite_user_id", "kite_password", "kite_totp_secret")
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "X-Kite-Version": "3.0",
    "Accept": "application/json, text/plain, */*",
}

_lock = threading.Lock()
_state: dict = {"last_attempt": None, "last_success": None, "last_error": None, "running": False, "attempts": 0}


class KiteLoginError(RuntimeError):
    pass


def credentials_ready() -> bool:
    return all(getattr(settings, k, "") for k in REQUIRED)


def missing_credentials() -> list[str]:
    return [k.upper() for k in REQUIRED if not getattr(settings, k, "")]


def status() -> dict:
    return {**_state, "enabled": credentials_ready(), "missing": missing_credentials(),
            "schedule_ist": ["08:45", "08:55", "09:05"]}


def _err(r: httpx.Response, what: str) -> str:
    try:
        j = r.json()
        return f"{what}: {j.get('message') or j.get('error_type') or r.text[:200]}"
    except Exception:
        return f"{what}: HTTP {r.status_code} {r.text[:200]}"


def fetch_request_token(client: Optional[httpx.Client] = None, totp_now: Optional[Callable[[], str]] = None) -> str:
    """Log in to Zerodha with user ID, password and TOTP and return the Kite Connect request_token."""
    missing = missing_credentials()
    if missing:
        raise KiteLoginError("missing " + ", ".join(missing))
    totp_now = totp_now or pyotp.TOTP(settings.kite_totp_secret.replace(" ", "")).now
    own = client is None
    client = client or httpx.Client(headers=_HEADERS, timeout=20, follow_redirects=True)
    try:
        # 1. Kite Connect login page: gives us the OAuth session id for this api_key
        r = client.get(f"{KITE}/connect/login", params={"v": "3", "api_key": settings.kite_api_key})
        sess_id = parse_qs(urlparse(str(r.url)).query).get("sess_id", [""])[0]
        if not sess_id:
            raise KiteLoginError(f"no sess_id from Kite Connect login page (HTTP {r.status_code}); is the API key right?")
        # 2. user id + password
        r = client.post(f"{KITE}/api/login", data={"user_id": settings.kite_user_id, "password": settings.kite_password})
        if r.status_code != 200 or r.json().get("status") != "success":
            raise KiteLoginError(_err(r, "password step failed"))
        request_id = r.json()["data"]["request_id"]
        twofa_type = r.json()["data"].get("twofa_type", "totp")
        # 3. TOTP
        r = client.post(f"{KITE}/api/twofa", data={"user_id": settings.kite_user_id, "request_id": request_id,
                                                  "twofa_value": totp_now(), "twofa_type": twofa_type,
                                                  "skip_session": "true"})
        if r.status_code != 200 or r.json().get("status") != "success":
            raise KiteLoginError(_err(r, "TOTP step failed (check KITE_TOTP_SECRET and the server clock)"))
        # 4. finish the OAuth flow: Zerodha redirects to our callback URL carrying request_token.
        #    Follow redirects by hand so the callback itself (our own server) is never hit from here.
        url = f"{KITE}/connect/finish?v=3&api_key={settings.kite_api_key}&sess_id={sess_id}"
        for _ in range(6):
            r = client.get(url, follow_redirects=False)
            token = parse_qs(urlparse(str(r.url)).query).get("request_token", [""])[0]
            loc = r.headers.get("location", "")
            tok_in_loc = parse_qs(urlparse(loc).query).get("request_token", [""])[0]
            if tok_in_loc:
                return tok_in_loc
            if token:
                return token
            if r.status_code in (301, 302, 303, 307, 308) and loc:
                url = loc if loc.startswith("http") else KITE + loc
                continue
            raise KiteLoginError(f"Kite did not redirect with a request_token (HTTP {r.status_code}): {r.text[:200]}")
        raise KiteLoginError("too many redirects finishing the Kite Connect login")
    finally:
        if own:
            client.close()


def run_kite_auto_login(remember: bool = True) -> str:
    """Full daily login: request_token → access token (via API secret), applied to the Kite client
    and saved in the vault for the day. Returns the access token. Raises KiteLoginError."""
    from kite_client import kite_client
    with _lock:
        _state["running"] = True
        _state["attempts"] += 1
        _state["last_attempt"] = now_ist().isoformat(timespec="seconds")
    try:
        request_token = fetch_request_token()
        token = kite_client.set_access_token(request_token=request_token)
        settings.kite_access_token = token
        if remember:
            try:
                from secret_store import secret_store
                secret_store.save({"kite_access_token": token})
            except Exception as exc:
                logger.warning("[kite_login] token not saved to vault: {}", exc)
        _state.update(last_success=now_ist().isoformat(timespec="seconds"), last_error=None)
        logger.info("[kite_login] Kite access token refreshed automatically ({}…)", token[:6])
        try:
            from tick_engine import tick_engine
            tick_engine.on_kite_login()           # subscribe the watchlist / restart the WebSocket
        except Exception as exc:
            logger.warning("[kite_login] tick engine refresh failed: {}", exc)
        return token
    except Exception as exc:
        _state["last_error"] = str(exc)
        logger.error("[kite_login] automatic Kite login failed: {}", exc)
        raise KiteLoginError(str(exc)) from exc
    finally:
        _state["running"] = False


async def refresh_kite_token_async() -> str:
    return await asyncio.to_thread(run_kite_auto_login)


def token_valid_today() -> bool:
    """Is there an access token that Kite accepts right now?"""
    from kite_client import kite_client
    if not settings.kite_access_token:
        return False
    try:
        kite_client.profile()
        return True
    except Exception:
        return False


def needed_now() -> bool:
    """Startup / catch-up rule: credentials present, no working token, and either a weekend (any time,
    for research data) or a weekday before 15:30 IST."""
    now = now_ist()
    in_window = now.weekday() >= 5 or (now.hour, now.minute) < (15, 30)
    return credentials_ready() and in_window and not token_valid_today()
