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
import requests
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


def api_get(path: str) -> dict:
    r = requests.get(f"{API}{path}", headers={"X-API-Key": API_KEY} if API_KEY else {}, timeout=10)
    r.raise_for_status()
    return r.json()


def api_get_key() -> str:
    return API_KEY


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

        # ── Scoreboard vs index + strategy health (after-cost ledger) ──────────
        sb = page.locator('[data-testid="scoreboard"]')
        check("Scoreboard panel rendered", sb.count() == 1)
        sb_text = sb.inner_text() if sb.count() else ""
        check("Scoreboard compares with NIFTY 50 for every period",
              all(k in sb_text for k in ("Today", "Week", "Month", "Year", "Since start", "NIFTY")), sb_text[:160])
        check("Scoreboard shows a verdict and the trend filter",
              ("judge" in sb_text or "NIFTY 50 by" in sb_text) and "Trend filter" in sb_text, sb_text[:200])
        d = summary()
        strat = page.locator('[data-testid="strategy-table"]')
        if d["kpis"]["trades_closed"]:
            check("Strategy health table lists strategies with closed trades",
                  strat.count() == 1 and strat.locator("tbody tr").count() >= 1)
            row = strat.locator("tbody tr").first.inner_text() if strat.count() else ""
            check("Strategy row shows trades, win rate, net P&L and 'to judge' countdown",
                  "%" in row and "₹" in row and ("judge" in row or "Active" in row), row[:120])
            check("Ledger has costs deducted (net ≠ gross)",
                  any(t["cost"] > 0 and t["net"] != t["gross"]
                      for t in api_get("/performance/trades?n=20")["trades"]))
        else:
            check("Strategy health shows the 'judged after 30 trades' note", "30" in page.inner_text("body"))

        page.click('[data-testid="open-trading"]')
        page.wait_for_selector('[data-testid="open-overview"]', timeout=10000)
        check("Switch to Trading view", True)
        page.screenshot(path=os.path.join(SHOTS, "trading_view.png"))

        # ── Settings → API Keys: encrypted vault UI ──────────────────────────
        page.locator("header button:has(svg.lucide-settings)").first.click()
        page.get_by_role("button", name="API Keys").click()
        page.wait_for_timeout(1200)
        kite_box = page.locator('[data-testid="kite-session"]')
        check("API Keys tab: Kite login box + 'Log in with Zerodha'",
              kite_box.count() == 1 and "Zerodha" in kite_box.inner_text())
        check("API Keys tab: 'Remember on this server (encrypted)' toggle",
              page.get_by_text("Remember on this server").count() == 1)
        inputs = page.locator("input[placeholder='Leave blank to keep current']")
        inputs.nth(2).fill("sk-ant-e2e-test-key-ZZ99")
        page.get_by_role("button", name="Save Credentials").click()
        page.wait_for_timeout(1500)
        check("Saved key shows only its last 4 chars, never the value",
              page.get_by_text("Saved …ZZ99").count() == 1 and "sk-ant-e2e" not in page.inner_text("body"))
        st = api_get("/settings/credentials/status")
        check("Vault reports the key as stored, value absent from the API",
              st["vault"]["fields"]["anthropic_api_key"]["stored"] and "sk-ant-e2e" not in json.dumps(st))
        page.get_by_role("button", name="Remove").first.click()
        page.wait_for_timeout(1000)
        check("Remove deletes it from the vault",
              not api_get("/settings/credentials/status")["vault"]["fields"]["anthropic_api_key"]["stored"])
        page.screenshot(path=os.path.join(SHOTS, "settings_api_keys.png"))
        page.get_by_role("button", name="Cancel").first.click()
        page.wait_for_timeout(500)

        # ── AI Trader tab ────────────────────────────────────────────────────
        page.get_by_role("button", name="AI Trader").first.click()
        page.wait_for_timeout(1500)
        body = page.inner_text("body")
        check("AI Trader tab renders status and chat box",
              ("Message the AI trader" in body or "Anthropic key" in body), body[-200:])
        page.screenshot(path=os.path.join(SHOTS, "ai_tab.png"))

        # ── Single-process serving: the backend serves the built dashboard too ─
        r = requests.get(f"{API}/dashboard", timeout=10)
        check("Backend serves the built React dashboard at /dashboard (no Vite needed)",
              r.status_code == 200 and 'id="root"' in r.text and "/assets/" in r.text)
        r = requests.get(f"{API}/api/dashboard/summary", timeout=10)
        check("/api prefix works and is protected like the real path",
              r.status_code == (200 if not api_get_key() else 200) and r.headers.get("content-type", "").startswith("application/json")
              if not api_get_key() else r.status_code == 401)

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
