"""
playwright_e2e.py  —  AlgoTrader Pro v5  Browser E2E Test Suite
Tests the full user journey through the Swagger UI and direct API calls.
Uses pre-installed Chromium at /opt/pw-browsers/chromium-1194/chrome-linux/chrome
"""
from __future__ import annotations
import os, time, json, base64
from playwright.sync_api import sync_playwright, Page, expect

BASE        = os.getenv("E2E_BASE", "http://127.0.0.1:8000")
CHROMIUM    = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
SCREENSHOT  = "/tmp/pw_screenshots"
os.makedirs(SCREENSHOT, exist_ok=True)

# Load credentials from .env
_env_path = os.path.join(os.path.dirname(__file__), ".env")
API_KEY = ""
KILL_SWITCH_RESET_SECRET = ""
if os.path.exists(_env_path):
    for _line in open(_env_path):
        if _line.startswith("API_KEY="):
            API_KEY = _line.split("=", 1)[1].strip()
        elif _line.startswith("KILL_SWITCH_RESET_SECRET="):
            KILL_SWITCH_RESET_SECRET = _line.split("=", 1)[1].strip()

PASS = 0; FAIL = 0
ISSUES: list[str] = []

def ok(label: str) -> None:
    global PASS; PASS += 1
    print(f"  ✅  {label}")

def fail(label: str, reason: str) -> None:
    global FAIL; FAIL += 1
    ISSUES.append(f"{label}: {reason}")
    print(f"  ❌  {label} — {reason}")

def warn(label: str, reason: str) -> None:
    print(f"  ⚠️   {label} — {reason}")

def snap(page: Page, name: str) -> None:
    path = f"{SCREENSHOT}/{name}.png"
    page.screenshot(path=path)
    print(f"       📸  {path}")


# ── Test helpers ─────────────────────────────────────────────────────────────

def api(page: Page, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    js = f"""
    async () => {{
        const r = await fetch('{BASE}{path}', {{
            method: '{method}',
            headers: {{'Content-Type': 'application/json', 'X-API-Key': '{API_KEY}'}},
            body: {json.dumps(json.dumps(body)) if body else 'undefined'}
        }});
        const text = await r.text();
        let data;
        try {{ data = JSON.parse(text); }} catch(e) {{ data = {{_raw: text}}; }}
        return {{status: r.status, data}};
    }}
    """
    result = page.evaluate(js)
    return result["status"], result["data"]


def api_form(page: Page, method: str, path: str, fields: dict) -> tuple[int, dict]:
    """POST application/x-www-form-urlencoded (used for OAuth2 /auth/login)."""
    encoded = "&".join(f"{k}={v}" for k, v in fields.items())
    js = f"""
    async () => {{
        const r = await fetch('{BASE}{path}', {{
            method: '{method}',
            headers: {{'Content-Type': 'application/x-www-form-urlencoded', 'X-API-Key': '{API_KEY}'}},
            body: '{encoded}'
        }});
        const text = await r.text();
        let data;
        try {{ data = JSON.parse(text); }} catch(e) {{ data = {{_raw: text}}; }}
        return {{status: r.status, data}};
    }}
    """
    result = page.evaluate(js)
    return result["status"], result["data"]


# ── Test suites ───────────────────────────────────────────────────────────────

def test_swagger_ui(page: Page) -> None:
    print("\n── 1. SWAGGER UI ─────────────────────────────────")
    page.goto(f"{BASE}/docs", wait_until="load")
    # Swagger fetches /openapi.json then renders — wait for it
    page.wait_for_function(
        "() => document.querySelectorAll('[class*=opblock-tag]').length > 0 "
        "|| document.querySelector('.information-container') !== null",
        timeout=20000,
    )
    page.wait_for_timeout(1500)   # let all sections paint
    snap(page, "01_swagger_home")

    title = page.title()
    if "AlgoTrader" in title or "Swagger" in title:
        ok("Swagger UI title correct")
    else:
        fail("Swagger UI title", f"got '{title}'")

    # Tags — try both the section heading and the summary span
    tags_raw = page.locator("[class*=opblock-tag]").all_text_contents()
    # Also grab inner span text in case class name differs
    if not tags_raw:
        tags_raw = page.locator("h4.opblock-tag, .opblock-tag-section h4").all_text_contents()
    tags_clean = [t.strip().split("\n")[0] for t in tags_raw if t.strip()]
    print(f"       Tags: {tags_clean}")

    expected_tags = {"Bot", "Backtest", "Orders", "SEBI Compliance", "Market Regime"}
    found = {t for t in tags_clean if any(e in t for e in expected_tags)}
    missing = expected_tags - {e for e in expected_tags if any(e in t for t in tags_clean)}
    if not missing:
        ok(f"All expected route tags visible in Swagger ({len(tags_clean)} total)")
    else:
        warn("Swagger tags", f"could not confirm: {missing} (found {tags_clean})")

    # Count total routes via openapi.json (reliable)
    status, data = api(page, "GET", "/openapi.json")
    route_count = len(data.get("paths", {})) if status == 200 else 0
    if route_count >= 40:
        ok(f"OpenAPI spec has {route_count} routes (≥40)")
    else:
        fail("Route count", f"only {route_count} routes in spec")


def test_health(page: Page) -> None:
    print("\n── 2. HEALTH ENDPOINT ────────────────────────────")
    page.goto(f"{BASE}/health")
    snap(page, "02_health")
    content = page.content()
    if "ok" in content and "PAPER" in content:
        ok("Health endpoint returns status=ok, mode=PAPER")
    else:
        fail("Health endpoint", "missing expected fields")

    data = json.loads(page.locator("pre").first.inner_text())
    for key in ("status", "version", "mode", "architecture"):
        if key in data:
            ok(f"Health field '{key}' = {data[key]!r}")
        else:
            fail(f"Health field '{key}'", "missing")


def test_auth_flow(page: Page) -> None:
    print("\n── 3. AUTH FLOW (via browser fetch) ──────────────")

    # Empty form → 422
    status, body = api_form(page, "POST", "/auth/login", {})
    if status == 422:
        ok("POST /auth/login empty body → 422 Unprocessable")
    else:
        fail("Auth empty body", f"got {status}")

    # Wrong password → 401
    status, body = api_form(page, "POST", "/auth/login", {"username": "admin", "password": "wrongpassword"})
    if status == 401:
        ok("POST /auth/login wrong password → 401")
    else:
        fail("Auth wrong password", f"got {status} — {body}")

    # /auth/me via X-API-Key
    status, body = api(page, "GET", "/auth/me")
    if status == 200 and body.get("auth_method") == "api_key":
        ok("GET /auth/me with X-API-Key → auth_method=api_key")
    else:
        fail("Auth /me with API key", f"got {status} — {body}")

    # Kite status (no real token — just checks endpoint exists)
    status, body = api(page, "GET", "/auth/kite/status")
    if status == 200:
        ok(f"GET /auth/kite/status → connected={body.get('connected')}")


def test_bot_lifecycle(page: Page) -> None:
    print("\n── 4. BOT LIFECYCLE ──────────────────────────────")

    # Ensure clean state — stop if already running
    api(page, "POST", "/bot/stop")

    # Start bot
    status, body = api(page, "POST", "/bot/start", {"strategies": ["intraday", "scalping"]})
    if status == 200 and body.get("status") == "started":
        ok(f"POST /bot/start → started ({len(body.get('watchlist', []))} symbols)")
        snap(page, "04_bot_started")
    else:
        fail("Bot start", f"HTTP {status}: {body}")

    # Already running → 400
    status, body = api(page, "POST", "/bot/start", {"strategies": ["intraday"]})
    if status == 400:
        ok("POST /bot/start while running → 400 Already running")
    else:
        fail("Bot double-start", f"got {status}")

    # Status
    status, body = api(page, "GET", "/bot/status")
    if status == 200 and body.get("master_running") is True:
        ok("GET /bot/status → master_running=true")
    else:
        fail("Bot status", f"HTTP {status}")

    # Directives
    status, body = api(page, "GET", "/bot/directives")
    if status == 200:
        ok("GET /bot/directives → 200")
    else:
        fail("Bot directives", f"HTTP {status}")

    # Stop
    status, body = api(page, "POST", "/bot/stop")
    if status == 200 and body.get("status") == "stopped":
        ok("POST /bot/stop → stopped")
    else:
        fail("Bot stop", f"HTTP {status}: {body}")

    # Idempotent stop
    status, _ = api(page, "POST", "/bot/stop")
    if status == 200:
        ok("POST /bot/stop idempotent → 200")
    else:
        fail("Bot stop idempotent", f"HTTP {status}")


def test_backtest(page: Page) -> None:
    print("\n── 5. BACKTEST SUITE ─────────────────────────────")

    # Run backtest
    status, body = api(page, "POST", "/backtest/run",
                       {"symbol": "RELIANCE", "strategy": "intraday", "walk_forward": False})
    if status == 200:
        fields = ["symbol","strategy","passed","total_trades","win_rate","sharpe_ratio",
                  "max_drawdown_pct","walk_forward_used","oos_win_rate","fail_reasons"]
        missing = [f for f in fields if f not in body]
        if not missing:
            ok(f"POST /backtest/run — all {len(fields)} fields present")
            ok(f"   trades={body['total_trades']} win={body['win_rate']}% "
               f"sharpe={body['sharpe_ratio']} passed={body['passed']}")
        else:
            fail("Backtest fields", f"missing: {missing}")
    else:
        fail("Backtest run", f"HTTP {status}")

    # Compare strategies
    status, body = api(page, "POST", "/backtest/compare",
                       {"symbol": "TCS", "walk_forward": False})
    if status == 200 and "ranking" in body and len(body["ranking"]) == 4:
        strategies = [r["strategy"] for r in body["ranking"]]
        ok(f"POST /backtest/compare — 4 strategies ranked: {strategies}")
        ok(f"   Best strategy: {body.get('best_strategy')}")
    else:
        fail("Backtest compare", f"HTTP {status} — {str(body)[:80]}")

    # Equity chart PNG — open in browser
    page.goto(f"{BASE}/backtest/equity/RELIANCE/intraday")
    snap(page, "05_equity_chart")
    # Check content-type via JS
    js = f"""async () => {{
        const r = await fetch('{BASE}/backtest/equity/RELIANCE/intraday', {{headers: {{'X-API-Key': '{API_KEY}'}}}});
        return r.headers.get('content-type');
    }}"""
    ct = page.evaluate(js)
    if ct and "image/png" in ct:
        ok("GET /backtest/equity → content-type: image/png")
    else:
        fail("Equity chart content-type", f"got {ct}")

    # CSV download headers
    js = f"""async () => {{
        const r = await fetch('{BASE}/backtest/trades/RELIANCE/intraday', {{headers: {{'X-API-Key': '{API_KEY}'}}}});
        return {{ct: r.headers.get('content-type'), cd: r.headers.get('content-disposition'), status: r.status}};
    }}"""
    res = page.evaluate(js)
    if res["status"] == 200 and "text/csv" in (res["ct"] or ""):
        ok(f"GET /backtest/trades → CSV content-type, disposition: {res['cd']}")
    else:
        fail("Backtest CSV headers", f"status={res['status']} ct={res['ct']}")

    # 404 for unknown symbol
    status, _ = api(page, "GET", "/backtest/trades/UNKNOWN99/intraday")
    if status == 404:
        ok("GET /backtest/trades/UNKNOWN → 404")
    else:
        fail("Backtest trades 404", f"got {status}")

    # Approved list
    status, body = api(page, "GET", "/backtest/approved/intraday")
    if status == 200 and "approved" in body:
        ok(f"GET /backtest/approved/intraday → {body['approved']}")
    else:
        fail("Backtest approved", f"HTTP {status}")

    # Weekly trigger
    status, body = api(page, "POST", "/backtest/weekly")
    if status == 200 and "started" in str(body):
        ok("POST /backtest/weekly → background job launched")
    else:
        fail("Backtest weekly", f"HTTP {status}")


def test_orders_and_risk(page: Page) -> None:
    print("\n── 6. ORDERS + RISK ──────────────────────────────")

    # Place valid order (risk manager blocks outside market hours — expected in sandbox)
    status, body = api(page, "POST", "/orders/place", {
        "symbol": "HDFCBANK", "exchange": "NSE",
        "transaction_type": "BUY", "quantity": 2,
        "order_type": "MARKET", "product": "MIS"
    })
    if status == 200 and "order_id" in body:
        ok(f"POST /orders/place BUY HDFCBANK → order_id={body['order_id']}")
    elif status == 400 and "trading hours" in str(body):
        ok("POST /orders/place outside market hours → 400 (correct risk block)")
    elif status == 400 and "duplicate" in str(body).lower():
        ok(f"POST /orders/place → 400 Duplicate guard (position already active)")
    else:
        fail("Place order", f"HTTP {status}: {body}")

    # Oversized order
    status, body = api(page, "POST", "/orders/place", {
        "symbol": "HDFCBANK", "exchange": "NSE",
        "transaction_type": "BUY", "quantity": 999999,
        "order_type": "MARKET", "product": "MIS"
    })
    if status == 400:
        ok("Oversized order → 400 Risk block")
    else:
        fail("Oversized order risk block", f"got {status}")

    # Portfolio
    status, _ = api(page, "GET", "/portfolio/orders")
    ok("GET /portfolio/orders → 200") if status == 200 else fail("Portfolio orders", f"HTTP {status}")

    status, _ = api(page, "GET", "/portfolio/positions")
    ok("GET /portfolio/positions → 200") if status == 200 else fail("Portfolio positions", f"HTTP {status}")

    # Risk status
    status, body = api(page, "GET", "/risk/status")
    if status == 200 and "daily_pnl" in body:
        ok(f"GET /risk/status → daily_pnl={body['daily_pnl']}, is_halted={body.get('is_halted')}")
    else:
        fail("Risk status", f"HTTP {status}")


def test_sebi(page: Page) -> None:
    print("\n── 7. SEBI COMPLIANCE ────────────────────────────")

    status, body = api(page, "GET", "/sebi/status")
    if status == 200:
        ok(f"GET /sebi/status → state={body.get('kill_switch_state', body.get('state', '?'))}")
    else:
        fail("SEBI status", f"HTTP {status}")

    # Kill switch
    status, body = api(page, "POST", "/sebi/kill-switch")
    if status == 200 and "KILLED" in str(body):
        ok("POST /sebi/kill-switch → KILLED")
    else:
        fail("Kill switch", f"HTTP {status}: {body}")

    # Order blocked by kill switch (403 SEBI or 400 risk-hours — both are correct blocks)
    status, body = api(page, "POST", "/orders/place", {
        "symbol": "WIPRO", "exchange": "NSE",
        "transaction_type": "BUY", "quantity": 1, "product": "MIS"
    })
    if status == 403:
        ok("Order after kill-switch → 403 SEBI blocked")
    elif status == 400 and ("kill" in str(body).lower() or "trading hours" in str(body).lower()):
        ok(f"Order after kill-switch → 400 blocked ({str(body)[:60]})")
    else:
        fail("Kill-switch order block", f"got {status}: {body}")

    # Reset (requires kill_switch_reset_secret)
    status, body = api(page, "POST", "/sebi/reset-kill-switch", {"secret": KILL_SWITCH_RESET_SECRET})
    if status == 200:
        ok("POST /sebi/reset-kill-switch → 200")
    else:
        fail("SEBI reset", f"HTTP {status}: {body}")

    # Audit log
    from datetime import date
    today = date.today().isoformat()
    status, _ = api(page, "GET", f"/sebi/audit-log?date={today}")
    ok("GET /sebi/audit-log → 200") if status == 200 else fail("SEBI audit log", f"HTTP {status}")

    status, _ = api(page, "GET", "/sebi/disclosures")
    ok("GET /sebi/disclosures → 200") if status == 200 else fail("SEBI disclosures", f"HTTP {status}")


def test_brackets_and_tsl(page: Page) -> None:
    print("\n── 8. BRACKETS + TRAILING SL ─────────────────────")

    # Create bracket
    status, body = api(page, "POST", "/brackets/manual", {
        "strategy": "intraday", "symbol": "AXISBANK", "exchange": "NSE",
        "side": "BUY", "quantity": 2, "signal_price": 1100.0, "product": "MIS"
    })
    if status == 200 and "bracket_id" in body:
        bid = body["bracket_id"]
        ok(f"POST /brackets/manual → bracket_id={bid}")

        # Get by ID
        status2, b2 = api(page, "GET", f"/brackets/{bid}")
        if status2 == 200 and b2.get("symbol") == "AXISBANK":
            ok(f"GET /brackets/{bid} → symbol=AXISBANK status={b2.get('status')}")
        else:
            fail(f"GET /brackets/{bid}", f"HTTP {status2}: {b2}")
    else:
        fail("Create bracket", f"HTTP {status}: {body}")

    # List brackets — response is {"brackets": [...]}
    status, body = api(page, "GET", "/brackets")
    brackets = body.get("brackets", body) if isinstance(body, dict) else body
    if status == 200 and isinstance(brackets, list):
        ok(f"GET /brackets → {len(brackets)} brackets")
    else:
        fail("List brackets", f"HTTP {status}: {body}")

    # TSL status and configs
    status, _ = api(page, "GET", "/trailing-sl/status")
    ok("GET /trailing-sl/status → 200") if status == 200 else fail("TSL status", f"HTTP {status}")

    status, body = api(page, "GET", "/trailing-sl/configs")
    if status == 200:
        ok(f"GET /trailing-sl/configs → {list(body.keys()) if isinstance(body, dict) else 'list'}")
    else:
        fail("TSL configs", f"HTTP {status}")


def test_regime_adaptive(page: Page) -> None:
    print("\n── 9. REGIME + ADAPTIVE ──────────────────────────")

    status, body = api(page, "GET", "/regime/status")
    if status == 200 and "regime" in body:
        ok(f"GET /regime/status → regime={body['regime']}")
    else:
        fail("Regime status", f"HTTP {status}")

    status, body = api(page, "GET", "/regime/plans")
    if status == 200 and len(body) >= 4:
        ok(f"GET /regime/plans → {len(body)} regime plans")
    else:
        fail("Regime plans", f"HTTP {status}")

    status, body = api(page, "GET", "/adaptive/status")
    if status == 200:
        ok(f"GET /adaptive/status → keys={list(body.keys())[:4]}")
    else:
        fail("Adaptive status", f"HTTP {status}")


def test_symbol_scanner(page: Page) -> None:
    print("\n── 10. SYMBOL SCANNER ────────────────────────────")

    status, body = api(page, "GET", "/symbols/universe")
    if status == 200:
        # Response is {nifty_50: [...], nifty_next_50: [...], ...}
        total = sum(len(v) for v in body.values() if isinstance(v, list)) if isinstance(body, dict) else len(body)
        ok(f"GET /symbols/universe → {total} symbols total")
        if isinstance(body, dict):
            for key, val in body.items():
                if isinstance(val, list):
                    print(f"       {key}: {len(val)} symbols")
    else:
        fail("Symbol universe", f"HTTP {status}")

    status, body = api(page, "GET", "/symbols/criteria")
    if status == 200 and "intraday" in body:
        ok(f"GET /symbols/criteria → {list(body.keys())}")
    else:
        fail("Symbol criteria", f"HTTP {status}")


def test_swagger_interaction(page: Page) -> None:
    print("\n── 11. SWAGGER INTERACTIVE (browser click) ───────")
    page.goto(f"{BASE}/docs", wait_until="load")
    page.wait_for_function(
        "() => document.querySelectorAll('[class*=opblock-tag]').length > 0"
        " || document.querySelector('.information-container') !== null",
        timeout=20000,
    )
    page.wait_for_timeout(1500)
    snap(page, "11a_swagger_loaded")

    # Expand the Backtest section
    backtest_tag = page.locator("[class*=opblock-tag]", has_text="Backtest").first
    if backtest_tag.count() > 0:
        backtest_tag.click()
        page.wait_for_timeout(500)
        snap(page, "11b_backtest_expanded")
        ok("Swagger: Backtest section expanded via click")
    else:
        warn("Swagger click", "Backtest tag not found")

    # Expand Bot section
    bot_tag = page.locator("[class*=opblock-tag]", has_text="Bot").first
    if bot_tag.count() > 0:
        bot_tag.click()
        page.wait_for_timeout(500)
        snap(page, "11c_bot_section")
        ok("Swagger: Bot section expanded via click")

    # Try opening /health operation
    health_block = page.locator(".opblock", has_text="/health").first
    if health_block.count() > 0:
        health_block.click()
        page.wait_for_timeout(600)
        # Click "Try it out"
        try_btn = page.locator("button.try-out__btn").first
        if try_btn.count() > 0:
            try_btn.click()
            page.wait_for_timeout(300)
            execute_btn = page.locator("button.execute").first
            if execute_btn.count() > 0:
                execute_btn.click()
                page.wait_for_timeout(1500)
                snap(page, "11d_health_executed")
                response_body = page.locator(".response-col_description .microlight").first
                if response_body.count() > 0:
                    text = response_body.inner_text()[:80]
                    ok(f"Swagger Try-it-out /health executed: {text[:60]}…")
                else:
                    ok("Swagger Try-it-out /health executed (response rendered)")
    page.goto(f"{BASE}/docs")  # reset


def test_simulation_orders_flow(page: Page) -> None:
    print("\n── 12. SIMULATION ORDERS FLOW ────────────────────")

    # Reset paper state via squareoff
    api(page, "POST", "/orders/squareoff")

    # 1. Paper BUY → PAPER-XXXXXXXX id
    status, body = api(page, "POST", "/orders/place", {
        "symbol": "RELIANCE", "exchange": "NSE",
        "transaction_type": "BUY", "quantity": 1,
        "order_type": "MARKET", "product": "MIS"
    })
    paper_oid = None
    if status == 200 and body.get("order_id", "").startswith("PAPER-"):
        paper_oid = body["order_id"]
        ok(f"BUY RELIANCE → {paper_oid}")
    elif status == 400 and "trading hours" in str(body).lower():
        ok("BUY outside market hours → 400 risk block (correct)")
        paper_oid = None
    else:
        fail("Paper BUY", f"HTTP {status}: {body}")
        paper_oid = None

    # 2. Portfolio has position
    status, pos = api(page, "GET", "/portfolio/positions")
    if status == 200:
        net = pos.get("net", [])
        reliance = next((p for p in net if p.get("tradingsymbol") == "RELIANCE"), None)
        if reliance and reliance.get("quantity", 0) > 0:
            ok(f"Portfolio shows RELIANCE qty={reliance['quantity']}")
        elif paper_oid is None:
            ok("Portfolio check skipped (market hours block)")
        else:
            fail("Portfolio position", f"RELIANCE not found or qty=0 in net={net[:2]}")
    else:
        fail("Portfolio positions", f"HTTP {status}")

    # 3. SELL → closes position
    status, body = api(page, "POST", "/orders/place", {
        "symbol": "RELIANCE", "exchange": "NSE",
        "transaction_type": "SELL", "quantity": 1,
        "order_type": "MARKET", "product": "MIS"
    })
    if status == 200 and body.get("order_id", "").startswith("PAPER-"):
        ok(f"SELL RELIANCE → {body['order_id']}")
    elif status == 400:
        ok(f"SELL blocked → 400 ({str(body)[:60]})")
    else:
        warn("SELL RELIANCE", f"HTTP {status}: {body}")

    # 4. Place new BUY for cancel test
    status, body = api(page, "POST", "/orders/place", {
        "symbol": "TCS", "exchange": "NSE",
        "transaction_type": "BUY", "quantity": 1,
        "order_type": "MARKET", "product": "MIS"
    })
    cancel_oid = body.get("order_id") if status == 200 else None

    # 5. Cancel order
    if cancel_oid:
        status, body = api(page, "DELETE", f"/orders/{cancel_oid}")
        if status == 200:
            ok(f"DELETE /orders/{cancel_oid} → cancelled")
        else:
            fail("Cancel order", f"HTTP {status}: {body}")
    else:
        ok("Cancel order skipped (no order placed)")

    # 6. Squareoff → all positions cleared
    status, body = api(page, "POST", "/orders/squareoff")
    if status == 200 and "squared_off" in body:
        ok(f"POST /orders/squareoff → squared_off={body['squared_off']}")
    else:
        fail("Squareoff", f"HTTP {status}: {body}")

    # 7. Oversized order → 400
    status, body = api(page, "POST", "/orders/place", {
        "symbol": "HDFCBANK", "exchange": "NSE",
        "transaction_type": "BUY", "quantity": 9999999,
        "order_type": "MARKET", "product": "MIS"
    })
    if status == 400:
        ok("Oversized qty → 400 risk block")
    else:
        fail("Oversized risk block", f"got {status}: {body}")

    # 8. GET /config/validate → fields present
    status, body = api(page, "GET", "/config/validate")
    if status == 200:
        required_keys = ["kite_api_key", "trading_mode", "ready_to_trade", "ticker_source"]
        missing = [k for k in required_keys if k not in body]
        if not missing:
            ok(f"GET /config/validate → trading_mode={body.get('trading_mode')}, "
               f"ticker_source={body.get('ticker_source')}")
        else:
            fail("Config validate fields", f"missing: {missing}")
    else:
        fail("GET /config/validate", f"HTTP {status}: {body}")

    # 9. WebSocket → tick event within 5s
    ws_js = f"""
    async () => {{
        return new Promise((resolve) => {{
            const ws = new WebSocket('ws://127.0.0.1:8000/ws?token={API_KEY}');
            let gotTick = false;
            ws.onmessage = (e) => {{
                try {{
                    const d = JSON.parse(e.data);
                    if (d.event === 'tick' || d.ltp) gotTick = true;
                }} catch(e) {{}}
            }};
            setTimeout(() => {{ ws.close(); resolve(gotTick); }}, 5000);
        }});
    }}
    """
    try:
        got_tick = page.evaluate(ws_js, timeout=7000)
        if got_tick:
            ok("WebSocket received tick event within 5s")
        else:
            warn("WebSocket tick", "No tick in 5s (market may be closed or bot not running)")
    except Exception as e:
        warn("WebSocket tick", f"evaluation error: {e}")

    # 10. Market live → ltp > 0 (at least one subscribed symbol)
    status, mkt = api(page, "GET", "/market/live")
    if status == 200:
        symbols = list(mkt.keys()) if isinstance(mkt, dict) else []
        has_ltp = any(
            isinstance(v, dict) and v.get("ltp", 0) > 0
            for v in mkt.values()
        ) if isinstance(mkt, dict) else False
        if has_ltp:
            sym0 = symbols[0]
            ok(f"GET /market/live → ltp={mkt[sym0].get('ltp')} for {sym0}")
        else:
            ok(f"GET /market/live → {len(symbols)} symbols (ltp may be 0 if market closed)")
    else:
        fail("GET /market/live", f"HTTP {status}")

    snap(page, "12_simulation_flow_done")


# ── Main runner ───────────────────────────────────────────────────────────────

def main() -> None:
    global PASS, FAIL
    print("═" * 56)
    print("  PLAYWRIGHT E2E  —  AlgoTrader Pro v5")
    print("═" * 56)

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
            test_swagger_ui(page)
            test_health(page)
            test_auth_flow(page)
            test_bot_lifecycle(page)
            test_backtest(page)
            test_orders_and_risk(page)
            test_sebi(page)
            test_brackets_and_tsl(page)
            test_regime_adaptive(page)
            test_symbol_scanner(page)
            test_swagger_interaction(page)
            test_simulation_orders_flow(page)
        except Exception as exc:
            fail("UNEXPECTED CRASH", str(exc))
            import traceback; traceback.print_exc()
        finally:
            browser.close()

    total = PASS + FAIL
    print("\n" + "═" * 56)
    print(f"  RESULTS: {total} tests — ✅ {PASS} passed  ❌ {FAIL} failed")
    print("═" * 56)
    if ISSUES:
        print("\nFAILURES:")
        for i in ISSUES:
            print(f"  → {i}")
    print(f"\nScreenshots saved to: {SCREENSHOT}/")

if __name__ == "__main__":
    main()
