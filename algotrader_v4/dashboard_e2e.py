"""
dashboard_e2e.py — browser test for the animated Overview dashboard.

Needs the backend and the dashboard running (run_demo.sh starts both in demo mode):
    ./run_demo.sh                    # terminal 1
    python3 dashboard_e2e.py         # terminal 2

Environment:
    DASH_URL      dashboard URL              (default http://localhost:5000)
    API_URL       backend URL                (default http://localhost:8000)
    API_KEY       sent as X-API-Key if the backend requires it
    CHROMIUM_PATH browser binary             (default: Playwright's own Chromium)
    TRADE_WAIT_S  max seconds to wait for a trade (default 180)

Screenshots are written to ./e2e_screenshots/.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

DASH = os.getenv("DASH_URL", "http://localhost:5000")
API = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
CHROMIUM = os.getenv("CHROMIUM_PATH") or None
TRADE_WAIT_S = int(os.getenv("TRADE_WAIT_S", "180"))
SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_screenshots")
os.makedirs(SHOTS, exist_ok=True)

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'}  {name}{'' if ok else ' — ' + detail}")


def summary() -> dict:
    req = urllib.request.Request(f"{API}/dashboard/summary",
                                 headers={"X-API-Key": API_KEY} if API_KEY else {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def main() -> int:
    try:
        s = summary()
    except Exception as exc:
        print(f"Backend not reachable at {API}: {exc}\nStart it first (./run_demo.sh).")
        return 2
    print(f"Backend: mode={s['mode']} demo={s['demo']} market_open={s['market_open']}\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        if API_KEY:
            page.add_init_script(f"localStorage.setItem('api_key', {json.dumps(API_KEY)})")
        page.add_init_script("localStorage.setItem('view', 'overview')")

        page.goto(DASH, wait_until="networkidle")
        page.wait_for_selector('[data-testid="overview"]', timeout=15000)
        check("Overview page renders", True)

        kpis = page.locator('[data-testid="kpis"] > div')
        page.wait_for_timeout(1500)
        check("Six KPI cards", kpis.count() == 6, f"found {kpis.count()}")

        t1 = page.locator('[data-testid="clock"]').inner_text()
        page.wait_for_timeout(2500)
        t2 = page.locator('[data-testid="clock"]').inner_text()
        check("Clock is live (data refreshing)", t1 != t2 and t2 != "--:--:--", f"{t1} → {t2}")

        if not s["bot_running"]:
            start = page.locator('[data-testid="start-bot"]')
            if start.count() and start.is_enabled():
                start.click()
                print("     (clicked Start bot)")
        page.wait_for_function(
            "() => [...document.querySelectorAll('[data-testid^=agent-]')]"
            ".some(e => e.innerText.includes('running'))", timeout=60000)
        check("At least one agent running", True)

        page.wait_for_selector('[data-testid="ticker"]', timeout=60000)
        check("Live ticker tape shows prices", page.locator('[data-testid="ticker"] span').count() > 0)

        # Wait for real trading activity: a position card or a closed trade
        deadline = time.time() + TRADE_WAIT_S
        traded = False
        while time.time() < deadline:
            d = summary()
            if d["positions"] or d["kpis"]["trades_closed"]:
                traded = True
                break
            page.wait_for_timeout(3000)
        check(f"A trade happened within {TRADE_WAIT_S}s", traded,
              "no trades — outside market hours? Use DEMO_MODE (run_demo.sh)")

        if traded:
            page.wait_for_timeout(3000)
            d = summary()
            if d["positions"]:
                page.wait_for_selector('[data-testid="position-card"]', timeout=10000)
                card = page.locator('[data-testid="position-card"]').first.inner_text()
                check("Position card shows stop, target and LTP",
                      "SL" in card and "TGT" in card and "LTP" in card, card[:120])
            check("Activity feed lists orders",
                  page.locator('[data-testid="activity"] li').count() > 0)

        page.wait_for_timeout(4000)
        check("P&L chart drawn", page.locator('[data-testid="pnl-chart"] path').count() > 0)
        check("Risk gauge drawn", page.locator('[data-testid="loss-gauge"] path').count() >= 2)
        page.screenshot(path=os.path.join(SHOTS, "overview_desktop.png"), full_page=True)

        page.click('[data-testid="open-trading"]')
        page.wait_for_selector('[data-testid="open-overview"]', timeout=10000)
        check("Switch to Trading view", True)
        page.screenshot(path=os.path.join(SHOTS, "trading_view.png"))
        page.click('[data-testid="open-overview"]')
        page.wait_for_selector('[data-testid="overview"]', timeout=10000)
        check("Switch back to Overview", True)

        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(1500)
        overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
        check("Phone width: no horizontal scroll", overflow <= 1, f"{overflow}px overflow")
        page.screenshot(path=os.path.join(SHOTS, "overview_mobile.png"))

        real_errors = [e for e in errors if "WebSocket" not in e]
        check("No JavaScript errors", not real_errors, "; ".join(real_errors)[:300])
        browser.close()

    failed = [r for r in results if not r[1]]
    print(f"\n  RESULTS: {len(results)} checks — {len(results) - len(failed)} passed, {len(failed)} failed")
    print(f"  Screenshots: {SHOTS}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
