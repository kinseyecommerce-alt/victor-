"""
kite_auto_login.py
Headless Playwright automation for the Zerodha Kite Connect OAuth flow.
Called by the platform scheduler at 08:50 AM IST every trading day.

Requires in .env:
  KITE_USER_ID       — Zerodha client ID (e.g. ZA1234)
  KITE_PASSWORD      — Zerodha login password
  KITE_TOTP_SECRET   — TOTP secret from Zerodha 2FA setup (Google Authenticator seed)
  KITE_REDIRECT_URL  — full callback URL registered in your Kite app
                       e.g. https://yourdomain.com/auth/kite/callback
"""
from __future__ import annotations

import re
import asyncio
from loguru import logger

import pyotp
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

from config import settings
from kite_client import kite_client


_CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def _extract_request_token(url: str) -> str | None:
    m = re.search(r"[?&]request_token=([^&]+)", url)
    return m.group(1) if m else None


def run_kite_auto_login() -> str:
    """
    Drives the full Zerodha Kite Connect OAuth flow in a headless browser.

    Returns the new access token string on success.
    Raises RuntimeError on any failure (caller sends Telegram alert).
    """
    missing = [k for k in ("kite_user_id", "kite_password", "kite_totp_secret",
                            "kite_api_key", "kite_redirect_url")
               if not getattr(settings, k, "")]
    if missing:
        raise RuntimeError(f"Missing config: {', '.join(k.upper() for k in missing)}")

    totp = pyotp.TOTP(settings.kite_totp_secret)
    login_url = kite_client.login_url()
    redirect_prefix = settings.kite_redirect_url.split("?")[0]

    import os
    chromium = _CHROMIUM if os.path.exists(_CHROMIUM) else None  # falls back to installed

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=chromium,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        ctx  = browser.new_context()
        page = ctx.new_page()
        page.set_default_timeout(30_000)

        try:
            # ── 1. Open Kite login ────────────────────────────────────────
            logger.info("[kite_login] navigating to Kite OAuth…")
            page.goto(login_url, wait_until="domcontentloaded")

            # ── 2. User ID + password ─────────────────────────────────────
            page.fill('input[id="userid"], input[placeholder*="User ID"]', settings.kite_user_id)
            page.fill('input[id="password"], input[type="password"]',      settings.kite_password)
            page.click('button[type="submit"]')

            # ── 3. TOTP 2FA ───────────────────────────────────────────────
            page.wait_for_selector(
                'input[id="pin"], input[placeholder*="TOTP"], input[placeholder*="code"]',
                timeout=15_000,
            )
            page.fill(
                'input[id="pin"], input[placeholder*="TOTP"], input[placeholder*="code"]',
                totp.now(),
            )
            page.click('button[type="submit"]')

            # ── 4. Wait for redirect to our callback URL ──────────────────
            try:
                page.wait_for_url(f"{redirect_prefix}**", timeout=20_000)
            except PwTimeout:
                raise RuntimeError("Kite did not redirect to callback URL in time")

            request_token = _extract_request_token(page.url)
            if not request_token:
                raise RuntimeError(f"No request_token in redirect URL: {page.url}")

        finally:
            browser.close()

    # ── 5. Exchange request_token for access_token ────────────────────────
    token = kite_client.set_access_token(request_token=request_token)
    logger.info("[kite_login] access token refreshed: {}…", token[:8])
    return token


async def refresh_kite_token_async() -> str:
    """Async wrapper — runs blocking Playwright in a thread pool."""
    return await asyncio.to_thread(run_kite_auto_login)
