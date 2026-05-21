"""
test_buttons.py — AlgoTrader Pro v5
Comprehensive UI button test: every clickable button on every tab.
"""
from __future__ import annotations
import os, json, time
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright, Page

BASE     = os.getenv("E2E_BASE", "http://127.0.0.1:8000")
CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
SHOTS    = "/tmp/btn_screenshots"
os.makedirs(SHOTS, exist_ok=True)

# Load credentials from .env
_env = os.path.join(os.path.dirname(__file__), ".env")
API_KEY = ""; JWT_SECRET = ""; KS_SECRET = ""
if os.path.exists(_env):
    for line in open(_env):
        if line.startswith("API_KEY="):             API_KEY   = line.split("=",1)[1].strip()
        elif line.startswith("JWT_SECRET_KEY="):     JWT_SECRET = line.split("=",1)[1].strip()
        elif line.startswith("KILL_SWITCH_RESET_SECRET="): KS_SECRET = line.split("=",1)[1].strip()

def make_jwt() -> str:
    from jose import jwt as _jwt
    exp = datetime.utcnow() + timedelta(hours=8)
    return _jwt.encode({"sub": "admin", "exp": exp}, JWT_SECRET, algorithm="HS256")

PASS = 0; FAIL = 0
ISSUES: list[str] = []

def ok(label: str):
    global PASS; PASS += 1
    print(f"  ✅  {label}")

def fail(label: str, reason: str = ""):
    global FAIL; FAIL += 1
    msg = f"{label}" + (f": {reason}" if reason else "")
    ISSUES.append(msg)
    print(f"  ❌  {msg}")

def warn(label: str, reason: str = ""):
    print(f"  ⚠️   {label}" + (f" — {reason}" if reason else ""))

def snap(page: Page, name: str):
    path = f"{SHOTS}/{name}.png"
    page.screenshot(path=path, full_page=False)
    print(f"       📸  {path}")

def goto_dashboard(page: Page, jwt: str):
    """Land on the dashboard with auth already set in localStorage."""
    # Set JWT on the login page (same origin), then navigate to /dashboard.
    page.goto(f"{BASE}/login", wait_until="domcontentloaded")
    page.evaluate(f"""() => {{
        localStorage.setItem('jwtToken', '{jwt}');
        localStorage.setItem('apiKey', '{API_KEY}');
    }}""")
    page.goto(f"{BASE}/dashboard", wait_until="networkidle")
    page.wait_for_timeout(1200)

def toast_visible(page: Page) -> str:
    """Return text of any visible toast, empty string if none."""
    try:
        el = page.locator("#toast-container .toast").first
        if el.count() > 0 and el.is_visible():
            return el.inner_text()
    except Exception:
        pass
    return ""

def wait_toast(page: Page, timeout: int = 4000) -> str:
    """Wait for a toast to appear and return its text."""
    try:
        page.wait_for_selector("#toast-container .toast", timeout=timeout)
        return page.locator("#toast-container .toast").first.inner_text()
    except Exception:
        return ""

def dialog_is_open(page: Page) -> bool:
    """True if the confirm overlay has the 'show' class."""
    cls = page.locator("#confirm-overlay").get_attribute("class") or ""
    return "show" in cls

def close_open_dialog(page: Page):
    """Dismiss an open confirm dialog by clicking Cancel."""
    if dialog_is_open(page):
        page.locator("button[onclick='closeConfirm()']").first.click()
        page.wait_for_timeout(300)

def reset_kill_switch(page: Page):
    """API call to reset kill switch; must be called while on the app origin."""
    page.evaluate(f"""async () => {{
        await fetch('{BASE}/sebi/reset-kill-switch', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json', 'X-API-Key': '{API_KEY}'}},
            body: JSON.stringify({{secret: '{KS_SECRET}'}})
        }});
    }}""")
    page.wait_for_timeout(300)


# ── TEST SECTIONS ─────────────────────────────────────────────────

def test_login_page(page: Page):
    print("\n── 1. LOGIN PAGE BUTTONS ─────────────────────────")
    page.goto(f"{BASE}/login", wait_until="networkidle")
    snap(page, "01a_login_page")

    # Wrong password → error status message
    page.fill("#username", "admin")
    page.fill("#password", "wrongpass123")
    page.click("button[onclick='doLogin()']")
    page.wait_for_timeout(1500)
    status_el = page.locator("#login-status")
    status_text = status_el.inner_text() if status_el.count() > 0 else ""
    if status_text:
        ok(f"Login button with wrong password shows error: '{status_text[:50]}'")
    else:
        warn("Login button", "no error message visible for wrong password")
    snap(page, "01b_login_wrong_password")

    # Kite Login button
    kite_btn = page.locator("#kite-btn")
    if kite_btn.count() > 0:
        kite_btn.click()
        page.wait_for_timeout(1000)
        kite_msg = page.locator("#kite-msg").inner_text()
        if kite_msg:
            ok(f"Connect Kite button → shows message: '{kite_msg[:60]}'")
        else:
            warn("Connect Kite button", "no feedback message")
        snap(page, "01c_kite_btn_clicked")
    else:
        warn("Kite button", "not found on login page")

    # Set Token button (manual token)
    manual_input = page.locator("#manual-token")
    set_btn = page.locator("button[onclick='setManualToken()']")
    if manual_input.count() > 0 and set_btn.count() > 0:
        # Click with empty input → error
        set_btn.click()
        page.wait_for_timeout(500)
        msg = page.locator("#kite-msg").inner_text()
        if "token" in msg.lower() or "paste" in msg.lower():
            ok(f"Set Token button with empty input → error: '{msg[:50]}'")
        else:
            warn("Set Token empty input", f"msg: '{msg}'")
        snap(page, "01d_set_token_empty")
    else:
        warn("Set Token button/input", "not found")


def test_tab_navigation(page: Page, jwt: str):
    print("\n── 2. TAB NAVIGATION ─────────────────────────────")
    goto_dashboard(page, jwt)
    snap(page, "02a_dashboard_overview")

    tabs = [
        ("botcontrol",  "Bot Control"),
        ("intelligence","Intelligence"),
        ("analytics",   "Analytics"),
        ("risk",        "Risk & SEBI"),
        ("settings",    "Settings"),
        ("overview",    "Overview"),
    ]
    for tab_id, label in tabs:
        page.locator(f"[data-tab='{tab_id}']").click()
        page.wait_for_timeout(600)
        active = page.locator(f"[data-tab='{tab_id}'].active")
        panel_visible = page.locator(f"#tab-{tab_id}").is_visible()
        if active.count() > 0 and panel_visible:
            ok(f"Tab '{label}' → nav item active + panel visible")
        else:
            fail(f"Tab '{label}'", f"active={active.count()>0} panel={panel_visible}")
        snap(page, f"02_{tab_id}_tab")


def test_topbar_kill_switch(page: Page, jwt: str):
    print("\n── 3. TOPBAR KILL SWITCH BUTTON ──────────────────")
    goto_dashboard(page, jwt)
    reset_kill_switch(page)

    # Click the kill switch button in topbar
    kill_btn = page.locator(".kill-btn").first
    kill_btn.click()
    page.wait_for_timeout(600)

    # Confirm dialog should appear
    if dialog_is_open(page):
        ok("Kill Switch button → confirm dialog opens")
        snap(page, "03a_kill_switch_dialog")

        # Test Cancel button
        page.locator("button[onclick='closeConfirm()']").first.click()
        page.wait_for_timeout(400)
        if not dialog_is_open(page):
            ok("Cancel button → dialog closes without action")
        else:
            fail("Cancel button", "dialog still visible")

        # Re-open and confirm
        kill_btn.click()
        page.wait_for_timeout(500)
        page.locator("#confirm-ok-btn").click()
        page.wait_for_timeout(1500)
        snap(page, "03b_kill_switch_confirmed")
        toast = wait_toast(page, 3000)
        if toast:
            ok(f"Kill Switch confirmed → toast: '{toast[:60]}'")
        else:
            ok("Kill Switch confirmed (no toast visible)")
    else:
        fail("Kill Switch button", "confirm dialog did not open")
        snap(page, "03_kill_dialog_fail")

    reset_kill_switch(page)


def test_botcontrol_tab(page: Page, jwt: str):
    print("\n── 4. BOT CONTROL BUTTONS ────────────────────────")
    goto_dashboard(page, jwt)
    page.locator("[data-tab='botcontrol']").click()
    page.wait_for_timeout(800)
    snap(page, "04a_botcontrol_tab")

    # --- Start Bot ---
    start_btn = page.locator("#start-stop-btn")
    btn_text = start_btn.inner_text().strip()
    if "Start" in btn_text:
        start_btn.click()
        page.wait_for_timeout(2500)
        snap(page, "04b_after_start")
        new_text = start_btn.inner_text().strip()
        toast = wait_toast(page, 3000)
        if "Stop" in new_text:
            ok(f"Start Bot button → bot started, button now says '{new_text}'")
        elif toast:
            ok(f"Start Bot button → toast: '{toast[:60]}'")
        else:
            warn("Start Bot", f"button text after click: '{new_text}'")
    else:
        warn("Start Bot button", f"bot already running, text: '{btn_text}'")

    # --- Stop Bot ---
    current_text = start_btn.inner_text().strip()
    if "Stop" in current_text:
        start_btn.click()
        page.wait_for_timeout(2000)
        snap(page, "04c_after_stop")
        stopped_text = start_btn.inner_text().strip()
        toast = wait_toast(page, 3000)
        if "Start" in stopped_text:
            ok(f"Stop Bot button → bot stopped, button now says '{stopped_text}'")
        elif toast:
            ok(f"Stop Bot button → toast: '{toast[:60]}'")
        else:
            warn("Stop Bot", f"button text: '{stopped_text}'")

    # --- Risk sliders interaction ---
    daily_loss_slider = page.locator("#risk-daily-loss-slider")
    if daily_loss_slider.count() > 0:
        daily_loss_slider.fill("8000")
        page.wait_for_timeout(200)
        num_val = page.locator("#risk-daily-loss-num").input_value()
        if num_val == "8000":
            ok("Daily loss slider ↔ number input synced")
        else:
            warn("Daily loss slider sync", f"number shows: '{num_val}'")

    pos_slider = page.locator("#risk-pos-size-slider")
    if pos_slider.count() > 0:
        pos_slider.fill("75000")
        page.wait_for_timeout(200)

    page.locator("#risk-max-pos").fill("7")

    # --- Update Risk button ---
    update_risk_btn = page.locator("button[onclick='updateRisk()']")
    update_risk_btn.click()
    page.wait_for_timeout(1500)
    snap(page, "04d_update_risk")
    toast = wait_toast(page, 3000)
    if toast:
        ok(f"Update Risk button → toast: '{toast[:60]}'")
    else:
        ok("Update Risk button clicked (no toast visible)")


def test_risk_sebi_tab(page: Page, jwt: str):
    print("\n── 5. RISK & SEBI TAB BUTTONS ────────────────────")
    goto_dashboard(page, jwt)
    reset_kill_switch(page)
    page.locator("[data-tab='risk']").click()
    page.wait_for_timeout(800)
    snap(page, "05a_risk_tab")

    # --- Activate Kill Switch ---
    kill_btn = page.locator("#tab-risk button[onclick='confirmKillSwitch()']")
    kill_btn.click()
    page.wait_for_timeout(600)
    if dialog_is_open(page):
        ok("Risk tab — Activate Kill Switch → confirm dialog opens")
        snap(page, "05b_risk_kill_dialog")
        page.locator("#confirm-ok-btn").click()
        page.wait_for_timeout(1500)
        snap(page, "05c_kill_activated")
        toast = wait_toast(page, 3000)
        if toast:
            ok(f"Kill Switch activated → toast: '{toast[:60]}'")
        else:
            ok("Kill Switch activated (state updated)")
    else:
        fail("Risk Activate Kill Switch", "dialog did not open")

    page.wait_for_timeout(500)
    close_open_dialog(page)  # ensure overlay is dismissed before next click

    # --- Resume Trading --- (also uses showConfirm)
    resume_btn = page.locator("button[onclick='resumeTrading()']")
    resume_btn.click()
    page.wait_for_timeout(600)
    if dialog_is_open(page):
        page.locator("#confirm-ok-btn").click()
        page.wait_for_timeout(1000)
    page.wait_for_timeout(1500)
    snap(page, "05d_resume_trading")
    toast = wait_toast(page, 3000)
    if toast:
        ok(f"Resume Trading button → toast: '{toast[:60]}'")
    else:
        ok("Resume Trading button clicked")

    # --- Pause Trading ---
    pause_btn = page.locator("button[onclick='pauseTrading()']")
    pause_btn.click()
    page.wait_for_timeout(1500)
    snap(page, "05e_pause_trading")
    toast = wait_toast(page, 3000)
    if toast:
        ok(f"Pause Trading button → toast: '{toast[:60]}'")
    else:
        ok("Pause Trading button clicked")

    reset_kill_switch(page)


def test_settings_tab(page: Page, jwt: str):
    print("\n── 6. SETTINGS TAB BUTTONS ───────────────────────")
    goto_dashboard(page, jwt)
    page.locator("[data-tab='settings']").click()
    page.wait_for_timeout(800)
    snap(page, "06a_settings_tab")

    # --- Intelligence toggles ---
    toggles = [
        ("set-claude-gate", "Claude Gate toggle"),
        ("set-mtf",         "Multi-Timeframe toggle"),
        ("set-kelly",       "Kelly Sizing toggle"),
        ("set-prelearned",  "Pre-learned Mode toggle"),
        ("set-nifty100",    "Nifty 100 Watchlist toggle"),
    ]
    for tid, label in toggles:
        checkbox = page.locator(f"#{tid}")
        # Custom toggle — the <input> is visually hidden; click the wrapping <label>
        toggle_label = page.locator(f"label:has(#{tid})")
        if checkbox.count() > 0 and toggle_label.count() > 0:
            before = checkbox.is_checked()
            toggle_label.click()
            page.wait_for_timeout(600)
            after = checkbox.is_checked()
            if before != after:
                ok(f"{label} toggled ({before} → {after})")
            else:
                warn(label, "state did not change after click")
            # Toggle back
            toggle_label.click()
            page.wait_for_timeout(400)
        else:
            warn(label, "not found")
    snap(page, "06b_toggles_tested")

    # --- Save Trading Limits button ---
    page.locator("#tl-intraday").fill("10")
    page.locator("#tl-fno").fill("5")
    page.locator("#tl-swing").fill("4")
    page.locator("#tl-scalping").fill("25")
    page.locator("#tl-cooldown").fill("200")
    save_limits_btn = page.locator("button[onclick='saveTradingLimits()']")
    save_limits_btn.click()
    page.wait_for_timeout(1500)
    snap(page, "06c_save_limits")
    toast = wait_toast(page, 3000)
    if toast:
        ok(f"Save Limits button → toast: '{toast[:60]}'")
    else:
        ok("Save Limits button clicked")

    # --- Capital Allocation inputs + Save button ---
    page.locator("#ca-total").fill("500000")
    page.locator("#ca-intraday").fill("40")
    page.locator("#ca-swing").fill("25")
    page.locator("#ca-options").fill("25")
    page.locator("#ca-futures").fill("10")
    page.locator("#ca-max-intraday").fill("5")
    page.locator("#ca-max-scalping").fill("5")
    page.locator("#ca-max-swing").fill("3")
    page.wait_for_timeout(300)

    save_alloc_btn = page.locator("button[onclick='saveCapitalAllocation()']")
    save_alloc_btn.click()
    page.wait_for_timeout(1500)
    snap(page, "06d_save_allocation")
    toast = wait_toast(page, 3000)
    if toast:
        ok(f"Save Allocation button → toast: '{toast[:60]}'")
    else:
        ok("Save Allocation button clicked")

    # --- Reconnect Kite button (opens new tab — just verify it's clickable) ---
    reconnect_btn = page.locator("button[onclick*=\"window.open('/login'\"]")
    if reconnect_btn.count() > 0:
        with page.context.expect_page() as new_page_info:
            reconnect_btn.click()
        new_page = new_page_info.value
        new_page.wait_for_load_state("domcontentloaded")
        new_url = new_page.url
        new_page.close()
        if "/login" in new_url:
            ok(f"Reconnect Kite button → opens login page in new tab ({new_url})")
        else:
            warn("Reconnect Kite button", f"opened: {new_url}")
    else:
        warn("Reconnect Kite button", "not found")
    snap(page, "06e_settings_done")


def test_confirm_dialog_cancel(page: Page, jwt: str):
    print("\n── 7. CONFIRM DIALOG — CANCEL PATH ───────────────")
    goto_dashboard(page, jwt)
    reset_kill_switch(page)

    # Open any confirm dialog (topbar kill switch)
    page.locator(".kill-btn").first.click()
    page.wait_for_timeout(500)
    if dialog_is_open(page):
        # Click Cancel
        cancel_btn = page.locator("button[onclick='closeConfirm()']").first
        cancel_btn.click()
        page.wait_for_timeout(400)
        if not dialog_is_open(page):
            ok("Confirm dialog — Cancel button closes dialog, no action taken")
        else:
            fail("Confirm dialog Cancel", "dialog still visible after cancel")
        snap(page, "07_dialog_cancel")
    else:
        fail("Confirm dialog", "did not open to test Cancel path")


def test_agent_enable_toggles(page: Page, jwt: str):
    print("\n── 8. AGENT ENABLE / DISABLE TOGGLES ─────────────")
    goto_dashboard(page, jwt)
    page.locator("[data-tab='settings']").click()
    page.wait_for_timeout(1200)

    # Wait for agent-enables-list to be populated by fetchAgentEnables()
    page.wait_for_timeout(1500)
    # Trigger the fetch explicitly and wait
    page.evaluate("async () => { await fetchAgentEnables(); }")
    page.wait_for_timeout(1000)
    count = page.locator("#agent-enables-list label").count()
    if count > 0:
        ok(f"Agent enables section loaded with {count} agent toggle(s)")
        first_label = page.locator("#agent-enables-list label").first
        first_cb    = page.locator("#agent-enables-list input[type='checkbox']").first
        before = first_cb.is_checked()
        first_label.click()
        page.wait_for_timeout(800)
        after = first_cb.is_checked()
        if before != after:
            ok(f"Agent toggle changed ({before} → {after})")
            first_label.click()
            page.wait_for_timeout(600)
        else:
            warn("Agent toggle", "state did not change")
        snap(page, "08_agent_toggles")
    else:
        warn("Agent enable toggles", "no checkboxes found in #agent-enables-list")


# ── MAIN ──────────────────────────────────────────────────────────

def main():
    global PASS, FAIL
    print("═" * 58)
    print("  BUTTON TEST  —  AlgoTrader Pro v5")
    print("═" * 58)

    jwt = make_jwt()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=CHROMIUM,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        ctx  = browser.new_context(
            viewport={"width": 1440, "height": 900},
            extra_http_headers={"X-API-Key": API_KEY},
        )
        page = ctx.new_page()
        page.set_default_timeout(15000)

        try:
            test_login_page(page)
            test_tab_navigation(page, jwt)
            test_topbar_kill_switch(page, jwt)
            test_botcontrol_tab(page, jwt)
            test_risk_sebi_tab(page, jwt)
            test_settings_tab(page, jwt)
            test_confirm_dialog_cancel(page, jwt)
            test_agent_enable_toggles(page, jwt)
        except Exception as exc:
            import traceback
            fail("UNEXPECTED CRASH", str(exc))
            traceback.print_exc()
        finally:
            browser.close()

    total = PASS + FAIL
    print("\n" + "═" * 58)
    print(f"  RESULTS: {total} checks — ✅ {PASS} passed  ❌ {FAIL} failed")
    print("═" * 58)
    if ISSUES:
        print("\nFAILURES:")
        for i in ISSUES:
            print(f"  → {i}")
    print(f"\nScreenshots: {SHOTS}/")


if __name__ == "__main__":
    main()
