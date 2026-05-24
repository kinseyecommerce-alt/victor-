"""
main.py — AlgoTrader Pro v4 (tick-driven)
Real-time tick streaming via /ws WebSocket.
Kite used for order placement AND live market data (WebSocket streaming + REST quote fallback).
"""
from __future__ import annotations
import asyncio
import hashlib
import hmac
import re
import time
from collections import defaultdict
from datetime import datetime
from ist_clock import now_ist
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Query, Depends, Body
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field, model_validator
from pydantic.networks import IPvAnyAddress
from loguru import logger

from config import settings
from auth import authenticate, create_token, decode_token, hash_password
from market_data import nse_client, yf_client, is_market_open
from kite_client import kite_client
from risk_manager import risk_manager
from order_guard import order_guard
from backtest_engine import backtest_engine
from tick_engine import tick_engine
from signal_engine import signal_engine
from agents.master_agent import master_agent
from agents.strategy_agents import ALL_AGENTS
from trailing_sl_engine import trailing_sl_engine, TRAIL_CONFIGS
from symbol_scanner import symbol_scanner, CRITERIA, FULL_UNIVERSE
from market_regime import regime_detector, REGIME_PLANS
from adaptive_engine import adaptive_engine
from sebi_compliance import sebi_compliance, KillSwitchState, APPROVED_ALGO_IDS
from atomic_bracket import atomic_bracket_engine
import bot_state

from fastapi.security import OAuth2PasswordRequestForm
import swagger_ui_bundle

app = FastAPI(
    title="AlgoTrader Pro v4", version="4.0.0",
    description="Tick-driven · KiteConnect WebSocket + REST quote · orders + market data",
    docs_url=None, redoc_url=None,
)
app.mount("/swagger-static", StaticFiles(directory=swagger_ui_bundle.swagger_ui_path), name="swagger-static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# MED-1: restrict CORS to explicit methods and headers (no wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# Swagger UI 4.x only supports OpenAPI ≤3.0.x; override to 3.0.3
from fastapi.openapi.utils import get_openapi
def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    app.openapi_schema = get_openapi(
        title=app.title, version=app.version,
        openapi_version="3.0.3", description=app.description,
        routes=app.routes,
    )
    return app.openapi_schema
app.openapi = _custom_openapi


# ── CRIT-1: API key gate (all mutating routes + sensitive GETs) ───────────────────
_EXEMPT_PATHS = frozenset({"/health", "/openapi.json", "/auth/login-url", "/config/validate"})
_EXEMPT_PREFIXES = ("/swagger-static",)
_SENSITIVE_GETS = frozenset({
    "/portfolio/positions", "/portfolio/orders", "/sebi/audit-log",
    "/docs", "/redoc",
})

@app.middleware("http")
async def _api_key_gate(request: Request, call_next):
    mutates = request.method in ("POST", "PUT", "PATCH", "DELETE")
    is_sensitive_get = request.url.path in _SENSITIVE_GETS
    needs_auth = mutates or is_sensitive_get
    is_exempt = (
        request.url.path in _EXEMPT_PATHS
        or request.url.path in ("/auth/login", "/login")
        or any(request.url.path.startswith(p) for p in _EXEMPT_PREFIXES)
    )
    if needs_auth and not is_exempt and settings.api_key:
        # Accept X-API-Key (programmatic) OR JWT Bearer (browser/UI)
        api_key = request.headers.get("X-API-Key", "")
        auth_hdr = request.headers.get("Authorization", "")
        has_key = api_key == settings.api_key
        has_jwt = False
        if auth_hdr.startswith("Bearer ") and settings.jwt_secret_key:
            has_jwt = decode_token(auth_hdr[7:]) is not None
        if not has_key and not has_jwt:
            return JSONResponse({"detail": "Unauthorized: provide X-API-Key or Bearer token"}, status_code=401)
    return await call_next(request)


# ── HIGH-5: IP whitelist enforcement for orders and SEBI admin ──────────────────
_IP_GUARDED_PREFIXES = ("/orders/", "/sebi/kill-switch", "/sebi/resume",
                         "/sebi/reset-kill-switch", "/sebi/pause")

@app.middleware("http")
async def _ip_whitelist_gate(request: Request, call_next):
    if any(request.url.path.startswith(p) for p in _IP_GUARDED_PREFIXES):
        client_ip = request.client.host if request.client else "0.0.0.0"
        if not sebi_compliance.is_ip_allowed(client_ip):
            return JSONResponse({"detail": f"IP {client_ip} not whitelisted"}, status_code=403)
    return await call_next(request)


# ── MED-2: In-memory rate limiter for orders and AI signals ──────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_WINDOW = 60.0
_RATE_LIMITS = {"/orders/place": 30, "/signals/generate": 10}

@app.middleware("http")
async def _rate_limiter(request: Request, call_next):
    for path, limit in _RATE_LIMITS.items():
        if request.url.path == path and request.method == "POST":
            client_ip = request.client.host if request.client else "unknown"
            key = f"{client_ip}:{path}"
            now = time.monotonic()
            calls = _rate_store[key]
            calls[:] = [t for t in calls if now - t < _RATE_WINDOW]
            if len(calls) >= limit:
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
            calls.append(now)
    return await call_next(request)


# ── HIGH-2: Input validation helpers (prompt injection / path traversal) ────────
_SYMBOL_RE = re.compile(r"^[A-Z0-9\-&]{1,20}$")
_VALID_STRATEGIES = frozenset({"intraday", "fno", "swing", "scalping"})

def _clean_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if not _SYMBOL_RE.match(s):
        raise HTTPException(422, f"Invalid symbol: {sym!r}")
    return s

def _clean_strategy(strategy: str) -> str:
    s = strategy.strip().lower()
    if s not in _VALID_STRATEGIES:
        raise HTTPException(422, f"Unknown strategy: {strategy!r}. Valid: {sorted(_VALID_STRATEGIES)}")
    return s


# ── WebSocket connection pool ────────────────────────────────────────────────
_MAX_WS_CONNECTIONS = 50
ws_clients: list[WebSocket] = []


# ── LOW-4: fixed broadcast — no bare except, explicit dead-client removal ────────
async def broadcast(data: dict) -> None:
    dead: list[WebSocket] = []
    for ws in ws_clients[:]:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in ws_clients:
            ws_clients.remove(ws)

tick_engine.ws_broadcast = broadcast


# ── Pydantic models ────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    request_token: str | None = None
    access_token:  str | None = None

    @model_validator(mode="after")
    def at_least_one_token(self) -> "TokenRequest":
        if not self.request_token and not self.access_token:
            raise ValueError("Provide request_token or access_token")
        return self

class BacktestRequest(BaseModel):
    symbol: str; exchange: str = "NSE"; strategy: str = "intraday"
    lookback_days: int | None = None; walk_forward: bool = True

class BatchBacktestRequest(BaseModel):
    symbols: list[dict]; strategy: str = "intraday"; walk_forward: bool = True

class CompareRequest(BaseModel):
    symbol: str; exchange: str = "NSE"
    lookback_days: int | None = None; walk_forward: bool = True

# HIGH-3: Literal types on all enum-like fields to prevent injection via order type
class OrderRequest(BaseModel):
    symbol: str
    exchange: Literal["NSE", "BSE", "NFO", "BFO", "CDS", "MCX"] = "NSE"
    transaction_type: Literal["BUY", "SELL"]
    quantity: int = Field(gt=0)
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = "MARKET"
    product: Literal["MIS", "CNC", "NRML"] = "MIS"
    price: float = 0.0
    trigger_price: float = 0.0

class SignalRequest(BaseModel):
    symbol: str; exchange: str = "NSE"; strategy: str = "intraday"

# CRIT-3: Field(gt=0) bounds prevent negative/zero risk parameters
class RiskUpdateRequest(BaseModel):
    max_daily_loss:    float | None = Field(default=None, gt=0)
    max_position_size: float | None = Field(default=None, gt=0)
    stop_loss_pct:     float | None = Field(default=None, gt=0, le=20.0)
    target_pct:        float | None = Field(default=None, gt=0, le=50.0)

class BotStartRequest(BaseModel):
    strategies: list[str]
    watchlist:  list[dict] | None = None
    force_scan: bool = False

class ManualBracketRequest(BaseModel):
    strategy: str; symbol: str; exchange: str = "NSE"; side: str
    quantity: int; signal_price: float; product: str = "MIS"
    stop_loss: float | None = None; target_1: float | None = None
    target_2: float | None = None

class TSLUpdateRequest(BaseModel):
    strategy: str
    initial_sl_pct:  float | None = None
    trail_pct:       float | None = None
    activation_pct:  float | None = None
    target1_pct:     float | None = None

# MED-3: IPvAnyAddress validates both IPv4 and IPv6
class WhitelistIPRequest(BaseModel):
    ip: IPvAnyAddress

# CRIT-2: kill-switch reset requires a separate secret
class KillSwitchResetRequest(BaseModel):
    secret: str

class TradingLimitsRequest(BaseModel):
    max_trades_intraday:  int | None = Field(default=None, ge=1, le=100)
    max_trades_fno:       int | None = Field(default=None, ge=1, le=50)
    max_trades_swing:     int | None = Field(default=None, ge=1, le=30)
    max_trades_scalping:  int | None = Field(default=None, ge=1, le=200)
    cooldown_after_loss_sec: int | None = Field(default=None, ge=0, le=3600)

class AgentEnablesRequest(BaseModel):
    intraday: bool | None = None
    fno:      bool | None = None
    swing:    bool | None = None
    scalping: bool | None = None

class CapitalAllocationRequest(BaseModel):
    total_capital:           float | None = Field(None, ge=10000, le=100_000_000)
    intraday_capital_pct:    float | None = Field(None, ge=0, le=100)
    swing_capital_pct:       float | None = Field(None, ge=0, le=100)
    options_capital_pct:     float | None = Field(None, ge=0, le=100)
    futures_capital_pct:     float | None = Field(None, ge=0, le=100)
    max_intraday_positions:  int | None   = Field(None, ge=1, le=20)
    max_scalping_positions:  int | None   = Field(None, ge=1, le=20)
    max_swing_positions:     int | None   = Field(None, ge=1, le=10)

class CredentialsUpdateRequest(BaseModel):
    kite_api_key:      str | None = Field(default=None, min_length=1)
    kite_api_secret:   str | None = Field(default=None, min_length=1)
    anthropic_api_key: str | None = Field(default=None, min_length=1)
    truedata_username: str | None = Field(default=None, min_length=1)
    truedata_password: str | None = Field(default=None, min_length=1)

class AppPasswordRequest(BaseModel):
    username:     str | None = Field(default=None, min_length=1, max_length=50)
    new_password: str        = Field(min_length=8, max_length=128)


# ── UI pages ───────────────────────────────────────────────────────────────
@app.get("/login", include_in_schema=False)
def login_page():
    """Serve the browser login UI (app + Kite OAuth)."""
    with open("static/login.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    """Serve the main trading dashboard."""
    p = Path("static/dashboard.html")
    if not p.exists():
        return HTMLResponse("<h2>Dashboard not found — run deploy to build static assets.</h2>", status_code=404)
    return HTMLResponse(p.read_text())

@app.get("/", include_in_schema=False)
def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")

@app.get("/gate/log", tags=["Intelligence"])
def gate_log(n: int = 50):
    """Last N Claude trade gate decisions (newest first)."""
    from claude_trade_gate import get_gate_log
    return {"decisions": get_gate_log(n), "total": n}


# ── App auth (JWT) ──────────────────────────────────────────────────────────
@app.post("/auth/login", tags=["Auth"])
def app_login(form: OAuth2PasswordRequestForm = Depends()):
    """Exchange username + password for a JWT access token."""
    if not authenticate(form.username, form.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token, expires_in = create_token(form.username)
    return {"access_token": token, "token_type": "bearer", "expires_in": expires_in}

@app.get("/auth/me", tags=["Auth"])
def me(request: Request):
    """Return currently authenticated user (JWT or API key)."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        user = decode_token(auth[7:])
        if user:
            return {"user": user, "auth_method": "jwt"}
    if request.headers.get("X-API-Key") == settings.api_key and settings.api_key:
        return {"user": "api-key-user", "auth_method": "api_key"}
    raise HTTPException(401, "Not authenticated")

@app.get("/auth/kite/status", tags=["Auth"])
def kite_status():
    """Check whether a valid Kite access token is loaded."""
    try:
        profile = kite_client.profile()
        return {"connected": True, "account_id": profile.get("user_id", ""),
                "name": profile.get("user_name", ""), "email": profile.get("email", "")}
    except Exception:
        return {"connected": False, "message": "No valid Kite session. Use Connect Kite Account."}

@app.get("/auth/kite/callback", tags=["Auth"], include_in_schema=False)
def kite_callback(request_token: str = "", action: str = "", status: str = ""):
    """Zerodha redirects here after OAuth. Auto-captures the request_token."""
    if status != "success" or not request_token:
        return HTMLResponse(
            "<h2 style='font-family:sans-serif;color:#f85149'>Kite login failed or cancelled.</h2>"
            "<p><a href='/login'>← Back to login</a></p>",
            status_code=400,
        )
    try:
        token = kite_client.set_access_token(request_token=request_token)
        return HTMLResponse(f"""
        <html><head><title>Kite Connected</title></head>
        <body style="font-family:sans-serif;background:#0d1117;color:#e6edf3;
                     display:flex;align-items:center;justify-content:center;height:100vh">
          <div style="text-align:center">
            <div style="font-size:3rem">✅</div>
            <h2 style="color:#3fb950">Kite Connected!</h2>
            <p style="color:#8b949e">Token: {token[:8]}…</p>
            <p style="margin-top:16px"><a href="/login" style="color:#58a6ff">Back to dashboard</a></p>
          </div>
        </body></html>
        """)
    except Exception as e:
        logger.error("Kite callback error: {}", e)
        return HTMLResponse(
            "<h2 style='font-family:sans-serif;color:#f85149'>Token exchange failed.</h2>"
            "<p><a href='/login'>← Try again</a></p>",
            status_code=500,
        )


# ── Docs (LOW-3: protected by _api_key_gate middleware above) ─────────────────
@app.get("/docs", include_in_schema=False)
def swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="AlgoTrader Pro v4 - Swagger UI",
        swagger_js_url="/swagger-static/swagger-ui-bundle.js",
        swagger_css_url="/swagger-static/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
def redoc_ui() -> HTMLResponse:
    return get_redoc_html(openapi_url="/openapi.json", title="AlgoTrader Pro v4 - ReDoc")


# ── Auth ────────────────────────────────────────────────────────────────────
@app.get("/auth/login-url", tags=["Auth"])
def login_url(): return {"login_url": kite_client.login_url()}

@app.post("/auth/token", tags=["Auth"])
def set_token(req: TokenRequest):
    t = kite_client.set_access_token(req.request_token, req.access_token)
    return {"status": "ok", "access_token": t[:6] + "…"}


# ── Bot control ──────────────────────────────────────────────────────────────
@app.post("/bot/start", tags=["Bot"])
async def start_bot(req: BotStartRequest):
    if master_agent.running:
        raise HTTPException(400, "Already running")
    strategies = [s for s in req.strategies if bot_state.is_agent_enabled(s)]
    if not strategies:
        raise HTTPException(400, "All requested strategies are disabled")
    watchlist = req.watchlist
    if not watchlist:
        selected = await symbol_scanner.run(strategies=strategies, force=req.force_scan)
        watchlist = symbol_scanner.all_selected_flat()
        if not watchlist:
            from symbol_scanner import NIFTY_50
            watchlist = [{"symbol": s, "exchange": "NSE"} for s in NIFTY_50[:20]]
            logger.warning("[bot/start] Symbol scanner returned no results — using Nifty 50 fallback ({} symbols)", len(watchlist))
    report = master_agent.start(strategies, watchlist)
    return {"status": "started", "architecture": "tick-driven 1s",
            "symbol_selection": "auto-scanned" if not req.watchlist else "manual",
            "watchlist": [w["symbol"] for w in watchlist], "report": report}

@app.post("/bot/stop", tags=["Bot"])
async def stop_bot():
    await master_agent.stop()
    return {"status": "stopped"}

@app.get("/bot/status", tags=["Bot"])
def bot_status(): return master_agent.get_status()

@app.get("/bot/directives", tags=["Bot"])
def directives(): return master_agent.last_directives


# ── Market data ──────────────────────────────────────────────────────────────
@app.get("/market/live", tags=["Market"])
def live_market(): return tick_engine.all_latest()

@app.get("/market/live/{symbol}", tags=["Market"])
def live_symbol(symbol: str):
    symbol = _clean_symbol(symbol)
    tick, ind = tick_engine.latest(symbol)
    if not tick: raise HTTPException(404, f"{symbol} not subscribed")
    return {"symbol": symbol, "ltp": tick.ltp, "bid": tick.bid, "ask": tick.ask,
            "spread": tick.ask - tick.bid, "volume": tick.volume,
            "indicators": {"ema9": round(ind.ema9,2), "ema21": round(ind.ema21,2),
                "ema50": round(ind.ema50,2), "vwap": round(ind.vwap,2),
                "rsi_14": round(ind.rsi_14,1), "rsi_7": round(ind.rsi_7,1),
                "macd": round(ind.macd,4), "macd_signal": round(ind.macd_signal,4),
                "macd_hist": round(ind.macd_hist,4), "bb_upper": round(ind.bb_upper,2),
                "bb_lower": round(ind.bb_lower,2), "atr_14": round(ind.atr_14,2),
                "volume_ratio": round(ind.volume_ratio,2),
                "trend": ind.trend, "momentum": ind.momentum, "volatility": ind.volatility},
            "ts": tick.timestamp.isoformat()}

@app.get("/market/status", tags=["Market"])
async def market_status():
    status = await tick_engine.get_market_status()
    status["market_open"] = is_market_open()
    status["data_source"] = "NSE India API (not Kite)"
    return status

@app.get("/market/option-chain/{symbol}", tags=["Market"])
async def option_chain(symbol: str):
    symbol = _clean_symbol(symbol)
    data = await tick_engine.get_option_chain(symbol)
    if not data: raise HTTPException(404, f"Option chain not available for {symbol}")
    return data


# ── Agents ────────────────────────────────────────────────────────────────────
@app.get("/agents", tags=["Agents"])
def agents(): return {n: a.get_status() for n, a in ALL_AGENTS.items()}

@app.post("/agents/{name}/pause", tags=["Agents"])
def pause_agent(name: str):
    a = ALL_AGENTS.get(name)
    if not a: raise HTTPException(404, "Not found")
    a.stop(); return {"status": "paused"}

@app.post("/agents/{name}/resume", tags=["Agents"])
def resume_agent(name: str):
    a = ALL_AGENTS.get(name)
    if not a: raise HTTPException(404, "Not found")
    if not bot_state.is_agent_enabled(name):
        raise HTTPException(400, f"Agent '{name}' is disabled")
    wl = master_agent._agent_watchlists.get(name, [])
    if wl:
        q = tick_engine.add_subscriber(f"agent_{name}")
        a.start(q)
    return {"status": "resumed", "symbols": [w["symbol"] for w in wl]}


# ── Backtest ────────────────────────────────────────────────────────────────────
@app.post("/backtest/run", tags=["Backtest"])
def run_bt(req: BacktestRequest):
    sym = _clean_symbol(req.symbol)
    strat = _clean_strategy(req.strategy)
    return backtest_engine.run(sym, req.exchange, strat, req.lookback_days,
                               force=True, walk_forward=req.walk_forward).to_dict()

@app.post("/backtest/batch", tags=["Backtest"])
def batch_bt(req: BatchBacktestRequest):
    strat = _clean_strategy(req.strategy)
    results = backtest_engine.run_batch(req.symbols, strat, walk_forward=req.walk_forward)
    return {"strategy": strat,
            "passed":  [s for s, r in results.items() if r.passed],
            "failed":  [s for s, r in results.items() if not r.passed],
            "details": {s: r.to_dict() for s, r in results.items()}}

@app.get("/backtest/approved/{strategy}", tags=["Backtest"])
def approved(strategy: str):
    strategy = _clean_strategy(strategy)
    return {"strategy": strategy, "approved": backtest_engine.get_approved_symbols(strategy)}

@app.post("/backtest/compare", tags=["Backtest"])
def compare_strategies(req: CompareRequest):
    """Run all 4 strategies on a symbol and rank by Sharpe ratio."""
    sym = _clean_symbol(req.symbol)
    return backtest_engine.compare_strategies(sym, req.exchange, req.lookback_days, req.walk_forward)

@app.get("/backtest/trades/{symbol}/{strategy}", tags=["Backtest"])
def download_trades(symbol: str, strategy: str):
    """Download the full trade log for a symbol/strategy as CSV."""
    symbol = _clean_symbol(symbol)
    strategy = _clean_strategy(strategy)
    key = (symbol, strategy)
    result = backtest_engine._cache.get(key)
    if result is None:
        raise HTTPException(404, f"No backtest result cached for {symbol}/{strategy}. Run /backtest/run first.")
    csv_data = result.to_csv()
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={symbol}_{strategy}_trades.csv"},
    )

@app.get("/backtest/equity/{symbol}/{strategy}", tags=["Backtest"])
def equity_chart(symbol: str, strategy: str):
    """Return equity curve chart as PNG image."""
    symbol = _clean_symbol(symbol)
    strategy = _clean_strategy(strategy)
    key = (symbol, strategy)
    result = backtest_engine._cache.get(key)
    if result is None:
        raise HTTPException(404, f"No backtest result cached for {symbol}/{strategy}. Run /backtest/run first.")
    png_bytes = result.equity_chart_png()
    return StreamingResponse(
        iter([png_bytes]),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename={symbol}_{strategy}_equity.png"},
    )

@app.post("/backtest/weekly", tags=["Backtest"])
async def trigger_weekly_backtest():
    """Manually trigger the weekly auto-backtest across full universe."""
    asyncio.create_task(asyncio.to_thread(backtest_engine.weekly_auto_backtest))
    return {"status": "weekly backtest started", "note": "runs in background, check logs"}


# ── Orders ────────────────────────────────────────────────────────────────────
@app.post("/orders/place", tags=["Orders"])
async def place_order(req: OrderRequest):
    ok, reason = order_guard.can_place(req.symbol, "manual", req.transaction_type)
    if not ok: raise HTTPException(400, f"Guard: {reason}")
    ok, reason = risk_manager.check_before_order(req.symbol, req.quantity, req.price or 1, req.transaction_type)
    if not ok: raise HTTPException(400, f"Risk: {reason}")
    sebi_ok, algo_id, sebi_reason = sebi_compliance.pre_order_check(
        strategy="manual", symbol=req.symbol, exchange=req.exchange,
        transaction_type=req.transaction_type, quantity=req.quantity,
        order_type=req.order_type, price_at_signal=req.price or 0,
        signal_source="manual_api", regime=regime_detector.current_regime.value)
    if not sebi_ok: raise HTTPException(403, f"SEBI compliance: {sebi_reason}")
    oid = kite_client.place_order(
        tradingsymbol=req.symbol, exchange=req.exchange,
        transaction_type=req.transaction_type, quantity=req.quantity,
        order_type=req.order_type, product=req.product,
        price=req.price, trigger_price=req.trigger_price, tag=algo_id)
    sebi_compliance.record_order_id("manual", req.symbol, oid)
    order_guard.register_order(req.symbol, "manual", req.transaction_type, oid)
    await broadcast({"event": "order_placed", "order_id": oid, "symbol": req.symbol})
    return {"status": "ok", "order_id": oid}

@app.delete("/orders/{order_id}", tags=["Orders"])
async def cancel(order_id: str):
    kite_client.cancel_order(order_id); return {"status": "ok"}

@app.post("/orders/squareoff", tags=["Orders"])
async def squareoff():
    ids = kite_client.squareoff_all_positions()
    return {"status": "ok", "squared_off": len(ids)}

# HIGH-6: generic error messages, raw exceptions logged server-side only
@app.get("/portfolio/positions", tags=["Portfolio"])
def positions():
    try: return kite_client.positions()
    except Exception as e:
        logger.error("Portfolio positions error: {}", e)
        raise HTTPException(500, "Unable to fetch positions")

@app.get("/portfolio/orders", tags=["Portfolio"])
def orders():
    try: return kite_client.orders()
    except Exception as e:
        logger.error("Portfolio orders error: {}", e)
        raise HTTPException(500, "Unable to fetch orders")


# ── Claude Gate Log (dashboard) ──────────────────────────────────────────────
@app.get("/gate/log", tags=["AI Signal"])
def gate_log(n: int = 50):
    """Return last n Claude trade-gate decisions (newest first). Used by dashboard."""
    from claude_trade_gate import get_gate_log
    decisions = get_gate_log(n)
    # Normalise: add `enter` bool from decision string if needed
    for d in decisions:
        if "enter" not in d:
            d["enter"] = d.get("decision", "").upper() == "ENTER"
    return {"decisions": decisions, "count": len(decisions)}


# ── Signals / Risk ──────────────────────────────────────────────────────────
@app.post("/signals/generate", tags=["AI Signal"])
async def gen_signal(req: SignalRequest):
    # HIGH-2: sanitise inputs before they reach Claude prompt
    sym   = _clean_symbol(req.symbol)
    strat = _clean_strategy(req.strategy)
    try:
        sig = signal_engine.generate(sym, req.exchange, strat)
        await broadcast({"event": "signal", "symbol": sym, "signal": sig})
        from n8n_bridge import notify as _n8n
        asyncio.create_task(_n8n("signal", {
            "symbol":     sym,
            "strategy":   strat,
            "action":     sig.get("action", ""),
            "price":      sig.get("price", 0),
            "confidence": sig.get("confidence", 0),
            "pattern":    sig.get("trigger", ""),
        }))
        return sig
    except Exception as e:
        logger.error("Signal generation error for {}: {}", sym, e)
        raise HTTPException(500, "Signal generation failed")

@app.get("/risk/status", tags=["Risk"])
def risk_st(): return risk_manager.status()

@app.patch("/risk/update", tags=["Risk"])
def risk_update(req: RiskUpdateRequest):
    if req.max_daily_loss    is not None: settings.max_daily_loss    = req.max_daily_loss
    if req.max_position_size is not None: settings.max_position_size = req.max_position_size
    if req.stop_loss_pct     is not None: settings.stop_loss_pct     = req.stop_loss_pct
    if req.target_pct        is not None: settings.target_pct        = req.target_pct
    return risk_manager.status()

@app.get("/settings/trading-limits", tags=["Settings"])
def get_trading_limits():
    return {
        "max_trades_intraday":     settings.max_trades_intraday,
        "max_trades_fno":          settings.max_trades_fno,
        "max_trades_swing":        settings.max_trades_swing,
        "max_trades_scalping":     settings.max_trades_scalping,
        "cooldown_after_loss_sec": settings.cooldown_after_loss_sec,
    }

@app.patch("/settings/trading-limits", tags=["Settings"])
def patch_trading_limits(req: TradingLimitsRequest):
    if req.max_trades_intraday     is not None: settings.max_trades_intraday     = req.max_trades_intraday
    if req.max_trades_fno          is not None: settings.max_trades_fno          = req.max_trades_fno
    if req.max_trades_swing        is not None: settings.max_trades_swing        = req.max_trades_swing
    if req.max_trades_scalping     is not None: settings.max_trades_scalping     = req.max_trades_scalping
    if req.cooldown_after_loss_sec is not None: settings.cooldown_after_loss_sec = req.cooldown_after_loss_sec
    return get_trading_limits()


# ── Agent Enable/Disable ───────────────────────────────────────────────────
@app.get("/settings/agent-enables", tags=["Settings"])
def get_agent_enables():
    return dict(bot_state._agent_enabled)

@app.post("/settings/agent-enables", tags=["Settings"])
def set_agent_enables(req: AgentEnablesRequest):
    updates = req.model_dump(exclude_none=True)
    for name, val in updates.items():
        bot_state.set_agent_enabled(name, val)
        if not val:
            a = ALL_AGENTS.get(name)
            if a and a.state.running:
                a.stop()
    return dict(bot_state._agent_enabled)


# ── Capital Allocation ──────────────────────────────────────────────────────────
@app.get("/settings/capital-allocation", tags=["Settings"])
def get_capital_allocation():
    intraday_bucket = round(settings.total_capital * settings.intraday_capital_pct / 100)
    swing_bucket    = round(settings.total_capital * settings.swing_capital_pct    / 100)
    options_bucket  = round(settings.total_capital * settings.options_capital_pct  / 100)
    futures_bucket  = round(settings.total_capital * settings.futures_capital_pct  / 100)
    return {
        "total_capital":        settings.total_capital,
        "intraday_capital_pct": settings.intraday_capital_pct,
        "swing_capital_pct":    settings.swing_capital_pct,
        "options_capital_pct":  settings.options_capital_pct,
        "futures_capital_pct":  settings.futures_capital_pct,
        "max_positions": {
            "intraday": settings.max_intraday_positions,
            "scalping":  settings.max_scalping_positions,
            "swing":     settings.max_swing_positions,
        },
        "per_type_rupees": {
            "intraday": intraday_bucket,
            "swing":    swing_bucket,
            "options":  options_bucket,
            "futures":  futures_bucket,
        },
        "per_trade_rupees": {
            "intraday": round(intraday_bucket / max(settings.max_intraday_positions, 1)),
            "scalping":  round(intraday_bucket / max(settings.max_scalping_positions, 1)),
            "swing":     round(swing_bucket    / max(settings.max_swing_positions,    1)),
            "options":   options_bucket,
        },
        "agent_buckets": {
            "intraday": "intraday", "scalping": "intraday",
            "swing": "swing",       "fno": "options",
        }
    }

@app.patch("/settings/capital-allocation", tags=["Settings"])
def patch_capital_allocation(req: CapitalAllocationRequest):
    pcts = [req.intraday_capital_pct, req.swing_capital_pct,
            req.options_capital_pct,  req.futures_capital_pct]
    provided = [p for p in pcts if p is not None]
    if len(provided) == 4 and round(sum(provided), 2) > 100:
        raise HTTPException(400, "Capital percentages exceed 100%")
    if req.total_capital           is not None: settings.total_capital           = req.total_capital
    if req.intraday_capital_pct    is not None: settings.intraday_capital_pct    = req.intraday_capital_pct
    if req.swing_capital_pct       is not None: settings.swing_capital_pct       = req.swing_capital_pct
    if req.options_capital_pct     is not None: settings.options_capital_pct     = req.options_capital_pct
    if req.futures_capital_pct     is not None: settings.futures_capital_pct     = req.futures_capital_pct
    if req.max_intraday_positions  is not None: settings.max_intraday_positions  = req.max_intraday_positions
    if req.max_scalping_positions  is not None: settings.max_scalping_positions  = req.max_scalping_positions
    if req.max_swing_positions     is not None: settings.max_swing_positions     = req.max_swing_positions
    return get_capital_allocation()


@app.post("/settings/credentials", tags=["Settings"])
def update_credentials(req: CredentialsUpdateRequest):
    """Update API credentials in-memory. Restart reverts to env values."""
    if req.kite_api_key      is not None: settings.kite_api_key      = req.kite_api_key
    if req.kite_api_secret   is not None: settings.kite_api_secret   = req.kite_api_secret
    if req.anthropic_api_key is not None: settings.anthropic_api_key = req.anthropic_api_key
    if req.truedata_username is not None: settings.truedata_username = req.truedata_username
    if req.truedata_password is not None: settings.truedata_password = req.truedata_password
    creds = kite_client.validate_credentials()
    creds["truedata_username"] = bool(settings.truedata_username)
    creds["truedata_password"] = bool(settings.truedata_password)
    return {"status": "updated", "credentials": creds}


@app.post("/settings/app-password", tags=["Settings"])
def update_app_password(req: AppPasswordRequest):
    """Update admin login credentials in-memory. Restart reverts to env values."""
    if req.username is not None:
        settings.admin_username = req.username
    settings.admin_password_hash = hash_password(req.new_password)
    return {"status": "updated", "admin_username": settings.admin_username}


# ── WebSocket ────────────────────────────────────────────────────────────────────
# HIGH-1: token auth via ?token= query param + max connection cap
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    token = ws.query_params.get("token", "")
    if settings.api_key and token != settings.api_key:
        await ws.close(code=4001, reason="Unauthorized")
        return
    if len(ws_clients) >= _MAX_WS_CONNECTIONS:
        await ws.close(code=4002, reason="Too many connections")
        return
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"event": "pong", "ts": datetime.now().isoformat()})
    except WebSocketDisconnect:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ── Regime ────────────────────────────────────────────────────────────────────
@app.get("/regime/status", tags=["Market Regime"])
def regime_status(): return regime_detector.status()

@app.post("/regime/refresh", tags=["Market Regime"])
async def regime_refresh():
    regime, plan = await regime_detector.update()
    return {"regime": regime.value, "label": regime_detector._regime_label(),
            "active": plan.active, "paused": plan.paused,
            "allocation": plan.allocation, "size_factor": plan.size_factor,
            "reasoning": plan.reasoning}

@app.get("/regime/history", tags=["Market Regime"])
def regime_history(): return {"history": regime_detector.history}

@app.get("/regime/plans", tags=["Market Regime"])
def regime_plans():
    return {r.value: {"active": p.active, "paused": p.paused, "allocation": p.allocation,
            "size_factor": p.size_factor, "reasoning": p.reasoning}
            for r, p in REGIME_PLANS.items()}


# ── Adaptive engine ────────────────────────────────────────────────────────────
@app.get("/adaptive/status", tags=["Adaptive Engine"])
def adaptive_status():
    return adaptive_engine.summary()

# MED-5: bounded vix parameter
@app.post("/adaptive/review", tags=["Adaptive Engine"])
async def adaptive_review(
    vix: float = Query(default=14.0, ge=0.0, le=200.0, description="VIX value"),
    regime_changed: bool = False,
):
    report = await adaptive_engine.nightly_review(vix, regime_changed)
    return report


# ── Symbol scanner ────────────────────────────────────────────────────────────
@app.post("/symbols/scan", tags=["Symbol Scanner"])
async def run_scan(strategies: list[str] | None = None):
    result = await symbol_scanner.run(strategies=strategies, force=True)
    return {"status": "complete",
            "selected": {s: [x["symbol"] for x in syms] for s, syms in result.items()},
            "total_symbols": sum(len(v) for v in result.values())}

@app.get("/symbols/selected", tags=["Symbol Scanner"])
def get_selected(strategy: str | None = None): return symbol_scanner.get_selected(strategy)

@app.get("/symbols/scores/{strategy}", tags=["Symbol Scanner"])
def get_scores(strategy: str):
    scores = symbol_scanner.get_scores(strategy)
    if not scores: raise HTTPException(404, "No scan results yet.")
    return {"strategy": strategy,
            "passed":   sorted([s for s in scores if s["selected"]], key=lambda x: x["total_score"], reverse=True),
            "rejected": sorted([s for s in scores if not s["selected"]], key=lambda x: x["total_score"], reverse=True)}

@app.get("/symbols/criteria", tags=["Symbol Scanner"])
def get_criteria():
    return {name: {"description": c.description, "universe_size": len(c.universe),
                   "top_n": c.top_n, "score_weights": c.score_weights,
                   "filters": {"min_avg_volume": c.min_avg_volume,
                               "atr_range": f"{c.min_atr_pct}–{c.max_atr_pct}%",
                               "rsi_range": f"{c.rsi_min}–{c.rsi_max}",
                               "min_adx": c.min_adx, "fo_only": c.fo_eligible_only,
                               "require_trend": c.require_trend}}
            for name, c in CRITERIA.items()}

@app.get("/symbols/universe", tags=["Symbol Scanner"])
def get_universe():
    from symbol_scanner import NIFTY_50, NIFTY_NEXT_50, NIFTY_BANK
    return {"nifty_50": NIFTY_50, "nifty_next_50": NIFTY_NEXT_50,
            "nifty_bank": NIFTY_BANK, "total": len(FULL_UNIVERSE)}


# ── Trailing SL ────────────────────────────────────────────────────────────────
@app.get("/trailing-sl/status", tags=["Trailing SL"])
def tsl_status(): return trailing_sl_engine.status_summary()

@app.get("/trailing-sl/configs", tags=["Trailing SL"])
def tsl_configs():
    return {name: {"initial_sl_pct": cfg.initial_sl_pct, "trail_pct": cfg.trail_pct,
                   "breakeven_pct": cfg.breakeven_pct, "activation_pct": cfg.activation_pct,
                   "target1_pct": cfg.target1_pct, "target2_pct": cfg.target2_pct,
                   "mode": cfg.mode.value, "atr_multiplier": cfg.atr_multiplier}
            for name, cfg in TRAIL_CONFIGS.items()}

@app.patch("/trailing-sl/config", tags=["Trailing SL"])
def update_tsl_config(req: TSLUpdateRequest):
    cfg = TRAIL_CONFIGS.get(req.strategy)
    if not cfg: raise HTTPException(404, f"Strategy '{req.strategy}' not found")
    if req.initial_sl_pct  is not None: cfg.initial_sl_pct  = req.initial_sl_pct
    if req.trail_pct       is not None: cfg.trail_pct        = req.trail_pct
    if req.activation_pct  is not None: cfg.activation_pct  = req.activation_pct
    if req.target1_pct     is not None: cfg.target1_pct      = req.target1_pct
    return {"status": "updated", "strategy": req.strategy}


# ── Brackets ────────────────────────────────────────────────────────────────────
@app.get("/brackets", tags=["Brackets"])
def get_all_brackets(active_only: bool = False):
    return {"brackets": atomic_bracket_engine.all_brackets(active_only),
            "summary": atomic_bracket_engine.summary()}

@app.get("/brackets/{bracket_id}", tags=["Brackets"])
def get_bracket(bracket_id: str):
    b = atomic_bracket_engine.get_bracket(bracket_id)
    if not b: raise HTTPException(404, "Bracket not found")
    return b

@app.post("/brackets/manual", tags=["Brackets"])
async def manual_bracket(req: ManualBracketRequest):
    bracket = await atomic_bracket_engine.execute(
        strategy=req.strategy, symbol=req.symbol, exchange=req.exchange,
        side=req.side, quantity=req.quantity, signal_price=req.signal_price,
        product=req.product, stop_loss=req.stop_loss,
        target_1=req.target_1, target_2=req.target_2,
        sub_strategy="manual", trigger="manual_api")
    if not bracket: raise HTTPException(500, "Bracket execution failed")
    return bracket.to_dict()


# ── Paper simulation helper ─────────────────────────────────────────────────────

class SimTickRequest(BaseModel):
    symbol: str
    ltp: float = Field(gt=0)
    atr_14: float = 0.0

@app.post("/simulate/price-tick", tags=["Simulate"])
async def simulate_price_tick(req: SimTickRequest):
    """PAPER mode only — inject a price tick directly into the TSL engine.
    Bypasses candle-history guard so profit booking can be tested immediately."""
    if settings.trading_mode != "PAPER":
        raise HTTPException(403, "Only available in PAPER mode")
    sym = _clean_symbol(req.symbol)
    before = {b["bracket_id"]: b["status"]
              for b in atomic_bracket_engine.all_brackets(active_only=False)}
    await trailing_sl_engine.on_tick(sym, req.ltp, req.atr_14)
    after  = {b["bracket_id"]: b["status"]
              for b in atomic_bracket_engine.all_brackets(active_only=False)}
    changes = {bid: {"before": before.get(bid), "after": after[bid]}
               for bid in after if after[bid] != before.get(bid)}
    brackets_now = [b for b in atomic_bracket_engine.all_brackets()
                    if b["symbol"] == sym]
    return {"symbol": sym, "ltp": req.ltp, "status_changes": changes,
            "brackets": brackets_now}


@app.get("/sebi/status", tags=["SEBI Compliance"])
def sebi_status(): return sebi_compliance.status()

@app.get("/sebi/disclosures", tags=["SEBI Compliance"])
def sebi_disclosures(): return sebi_compliance.get_disclosure_document()

@app.post("/sebi/kill-switch", tags=["SEBI Compliance"])
async def trigger_kill_switch(reason: str = "Manual kill switch"):
    sebi_compliance.trigger_kill_switch(reason)
    from n8n_bridge import notify as _n8n
    asyncio.create_task(_n8n("system", {"type": "kill_switch", "reason": reason}))
    return {"status": "KILLED", "reason": reason}

@app.post("/sebi/resume", tags=["SEBI Compliance"])
def resume_trading():
    ok, msg = sebi_compliance.resume_trading()
    if not ok:
        raise HTTPException(409, f"SEBI: {msg}")
    return {"status": "ACTIVE"}

@app.post("/sebi/pause", tags=["SEBI Compliance"])
def pause_trading(reason: str = "Manual pause"):
    sebi_compliance.pause_trading(reason)
    return {"status": "PAUSED", "reason": reason}

@app.get("/sebi/audit-log", tags=["SEBI Compliance"])
def query_audit_log(date: str, strategy: str = None, symbol: str = None, decision: str = None):
    records = sebi_compliance.query_audit_log(date, strategy, symbol, decision)
    return {"date": date, "count": len(records), "records": records}

@app.get("/sebi/algo-ids", tags=["SEBI Compliance"])
def get_algo_ids(): return {"algo_ids": APPROVED_ALGO_IDS, "total": len(APPROVED_ALGO_IDS)}

@app.get("/sebi/strategy-disclosure/{strategy}", tags=["SEBI Compliance"])
def strategy_disclosure(strategy: str): return sebi_compliance.get_strategy_logic_disclosure(strategy)

# CRIT-2: reset kill switch requires a separate secret (not just API key)
@app.post("/sebi/reset-kill-switch", tags=["SEBI Compliance"])
def reset_kill_switch(req: KillSwitchResetRequest):
    ok, msg = sebi_compliance.reset_kill_switch(req.secret)
    if not ok:
        raise HTTPException(403, f"SEBI: {msg}")
    return {"status": "ACTIVE", "note": "Kill switch reset. Trading re-enabled."}

@app.post("/sebi/whitelist-ip", tags=["SEBI Compliance"])
def whitelist_ip(req: WhitelistIPRequest):
    sebi_compliance.add_whitelisted_ip(str(req.ip))
    return {"status": "added", "ip": str(req.ip)}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "4.0.0", "mode": settings.trading_mode,
            "architecture": "tick-driven 1s",
            "market_data_source": "KiteConnect (WebSocket + REST quote; orders + market data)",
            "market_open": is_market_open(),
            "master": "running" if master_agent.running else "stopped",
            "tick_engine": "running" if tick_engine._running else "stopped",
            "agents": {n: a.state.running for n, a in ALL_AGENTS.items()},
            "agent_enabled": dict(bot_state._agent_enabled),
            "subscribed_symbols": tick_engine.symbols(),
            "time": now_ist().strftime("%H:%M:%S IST")}


# ── Config validate ────────────────────────────────────────────────────────────
@app.get("/config/validate", tags=["System"])
def config_validate():
    """Validate all required credentials and show current tick data source."""
    creds = kite_client.validate_credentials()
    use_ws = getattr(tick_engine, "_use_ws", False)
    if settings.trading_mode == "LIVE":
        ticker_source = "KITE_WS" if use_ws else "KITE_REST"
    else:
        ticker_source = "PAPER"
    creds["ticker_source"] = ticker_source
    creds["truedata_username"] = bool(settings.truedata_username)
    creds["truedata_password"] = bool(settings.truedata_password)
    creds["admin_username"] = settings.admin_username
    creds["ready_to_trade"] = bool(
        creds.get("kite_api_key")
        and creds.get("kite_access_token")
        and creds.get("kite_initialised")
    )
    return creds


# ── n8n Inbound Webhook ───────────────────────────────────────────────────────

class N8NWebhookRequest(BaseModel):
    action:  str        # "place_order" | "start_bot" | "stop_bot" | "squareoff" | "get_status"
    payload: dict = {}


async def _verify_n8n_sig(request: Request, body: bytes) -> bool:
    """Return True if HMAC signature is valid, or if no secret is configured."""
    secret = settings.n8n_webhook_secret
    if not secret:
        return True
    sig_header = request.headers.get("X-AlgoTrader-Signature", "")
    if not sig_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig_header[7:], expected)


@app.post("/webhooks/n8n", tags=["Integration"])
async def n8n_inbound(request: Request):
    """Inbound webhook from n8n — routes to internal AlgoTrader actions.

    Requires X-API-Key header (standard auth).
    Optional HMAC-SHA256 body signature via X-AlgoTrader-Signature header.

    Supported actions:
      get_status   — returns bot status (read-only)
      stop_bot     — stops the trading bot
      squareoff    — squares off all open positions
      start_bot    — payload: {strategies:[...], watchlist:[...]}
      place_order  — payload matches OrderRequest fields
    """
    body = await request.body()
    if not await _verify_n8n_sig(request, body):
        raise HTTPException(401, "Invalid webhook signature")
    try:
        data = N8NWebhookRequest.model_validate_json(body)
    except Exception:
        raise HTTPException(422, "Invalid JSON body — expected {action, payload}")

    action = data.action
    p = data.payload

    if action == "get_status":
        return master_agent.get_status()

    elif action == "stop_bot":
        await master_agent.stop()
        return {"status": "stopped"}

    elif action == "squareoff":
        ids = kite_client.squareoff_all_positions()
        return {"status": "ok", "squared_off": len(ids)}

    elif action == "start_bot":
        if master_agent.running:
            raise HTTPException(400, "Bot already running")
        strategies = p.get("strategies", [])
        if not strategies:
            raise HTTPException(422, "strategies required in payload")
        watchlist = p.get("watchlist", [])
        report = master_agent.start(strategies, watchlist)
        return {"status": "started", "report": report}

    elif action == "place_order":
        try:
            req = OrderRequest(**p)
        except Exception as exc:
            raise HTTPException(422, f"Invalid order payload: {exc}")
        ok, reason = order_guard.can_place(req.symbol, "n8n", req.transaction_type)
        if not ok:
            raise HTTPException(400, f"Guard blocked: {reason}")
        ok, reason = risk_manager.check_before_order(
            req.symbol, req.quantity, req.price or 1.0, req.transaction_type
        )
        if not ok:
            raise HTTPException(400, f"Risk blocked: {reason}")
        sebi_ok, algo_id, sebi_reason = sebi_compliance.pre_order_check(
            strategy="n8n", symbol=req.symbol, exchange=req.exchange,
            transaction_type=req.transaction_type, quantity=req.quantity,
            order_type=req.order_type, price_at_signal=req.price or 0.0,
            signal_source="n8n_webhook",
            regime=regime_detector.current_regime.value if regime_detector.current_regime else "unknown",
        )
        if not sebi_ok:
            raise HTTPException(403, f"SEBI blocked: {sebi_reason}")
        oid = kite_client.place_order(
            tradingsymbol=req.symbol, exchange=req.exchange,
            transaction_type=req.transaction_type, quantity=req.quantity,
            order_type=req.order_type, product=req.product,
            price=req.price, trigger_price=req.trigger_price, tag=algo_id,
        )
        sebi_compliance.record_order_id("n8n", req.symbol, oid)
        order_guard.register_order(req.symbol, "n8n", req.transaction_type, oid)
        await broadcast({"event": "order_placed", "order_id": oid, "symbol": req.symbol})
        return {"status": "ok", "order_id": oid}

    else:
        raise HTTPException(422, f"Unknown action: {action!r}")


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    tick_engine.start_loop()
    atomic_bracket_engine.ws_broadcast = broadcast
    logger.info("FastAPI startup: tick engine + atomic bracket engine launched")
    asyncio.create_task(symbol_scanner.run())
    from platform_scheduler import platform_scheduler
    platform_scheduler.start()


# HIGH-7: reload=False in production — auto-reload bypasses security middleware
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
