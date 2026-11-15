"""
main.py — AlgoTrader Pro (core rebuild): FastAPI app.

The kept system: four lab-validated PAPER agents (agents/strategy_agents.py) driven by platform_scheduler, the
Kite data / login layer, the paper broker and its persisted book, the SEBI pre-order checks and kill switch, the
performance ledger, the forward test, the shadow registry and the option-bar archiver. The legacy endpoints of the
removed engine (AI trader, trade gates, backtests, regime, scanner, brackets, orders, n8n …) are listed in
REBUILD.md. The dashboard's read API is the versioned /api/v2 (api_v2.py, documented in API_V2.md).

Auth (unchanged): with API_KEY set, every mutating request and every sensitive GET (the /v2 API included) needs
X-API-Key or a Bearer JWT from /auth/login. The built dashboard calls /api/<path>; the prefix is stripped before
the gate looks at the path.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from pydantic.networks import IPvAnyAddress
from typing import Literal

import bot_state
import ist_clock as _ic
from auth import authenticate, create_token, decode_token, hash_password
from config import settings
from ist_clock import is_market_open, now_ist
from kite_client import kite_client
from sebi_compliance import sebi_compliance, APPROVED_ALGO_IDS
from tick_engine import tick_engine
from agents.strategy_agents import ALL_AGENTS

VERSION = "5.0.0"

app = FastAPI(title="AlgoTrader Pro", version=VERSION,
              description="Lab-validated PAPER strategies · Kite data · paper broker · v2 dashboard API",
              docs_url=None, redoc_url=None)
try:
    import swagger_ui_bundle
    app.mount("/swagger-static", StaticFiles(directory=swagger_ui_bundle.swagger_ui_path), name="swagger-static")
except Exception:                                    # optional (docs page only)
    swagger_ui_bundle = None
_STATIC = Path(__file__).resolve().parent / "static"
if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# The React dashboard (frontend/dist, built by `npm run build`) is served by this process.
_FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
if (_FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ── API key / JWT gate (all mutating routes + sensitive GETs) ─────────────────
_EXEMPT_PATHS = frozenset({"/health", "/openapi.json", "/auth/login-url", "/config/validate", "/v2/health"})
_EXEMPT_PREFIXES = ("/swagger-static",)
_SENSITIVE_GETS = frozenset({"/portfolio/positions", "/portfolio/orders", "/sebi/audit-log",
                             "/settings/credentials/status", "/docs", "/redoc"})
_SENSITIVE_PREFIXES = ("/v2/",)


def _authorised(request: Request) -> bool:
    api_key = request.headers.get("X-API-Key", "")
    auth_hdr = request.headers.get("Authorization", "")
    if api_key and api_key == settings.api_key:
        return True
    return bool(auth_hdr.startswith("Bearer ") and settings.jwt_secret_key and decode_token(auth_hdr[7:]))


@app.middleware("http")
async def _api_key_gate(request: Request, call_next):
    path = request.url.path
    mutates = request.method in ("POST", "PUT", "PATCH", "DELETE")
    sensitive = path in _SENSITIVE_GETS or any(path.startswith(p) for p in _SENSITIVE_PREFIXES)
    exempt = (path in _EXEMPT_PATHS or path in ("/auth/login", "/login")
              or any(path.startswith(p) for p in _EXEMPT_PREFIXES))
    if (mutates or sensitive) and not exempt and settings.api_key and not _authorised(request):
        return JSONResponse({"detail": "Unauthorized: provide X-API-Key or Bearer token"}, status_code=401)
    return await call_next(request)


# ── IP whitelist for SEBI admin actions ───────────────────────────────────────
_IP_GUARDED_PREFIXES = ("/orders/", "/sebi/kill-switch", "/sebi/resume", "/sebi/reset-kill-switch", "/sebi/pause")


@app.middleware("http")
async def _ip_whitelist_gate(request: Request, call_next):
    if any(request.url.path.startswith(p) for p in _IP_GUARDED_PREFIXES):
        client_ip = request.client.host if request.client else "0.0.0.0"
        if not sebi_compliance.is_ip_allowed(client_ip):
            return JSONResponse({"detail": f"IP {client_ip} not whitelisted"}, status_code=403)
    return await call_next(request)


# ── rate limiter for the write endpoints that act on trading ──────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_WINDOW = 60.0
_RATE_LIMITS = {"/orders/squareoff": 6, "/v2/trading-state": 30, "/v2/approvals": 60}


@app.middleware("http")
async def _rate_limiter(request: Request, call_next):
    if request.method == "POST":
        for path, limit in _RATE_LIMITS.items():
            if request.url.path.startswith(path):
                key = f"{request.client.host if request.client else 'unknown'}:{path}"
                now = time.monotonic()
                calls = _rate_store[key]
                calls[:] = [t for t in calls if now - t < _RATE_WINDOW]
                if len(calls) >= limit:
                    return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
                calls.append(now)
    return await call_next(request)


# ── the versioned dashboard API (api_v2.py, API_V2.md) ────────────────────────
from api_v2 import router as _v2_router  # noqa: E402
app.include_router(_v2_router)


# ── models ────────────────────────────────────────────────────────────────────
class TokenRequest(BaseModel):
    request_token: str | None = None
    access_token: str | None = None

    @model_validator(mode="after")
    def at_least_one_token(self) -> "TokenRequest":
        if not self.request_token and not self.access_token:
            raise ValueError("Provide request_token or access_token")
        return self


class WhitelistIPRequest(BaseModel):
    ip: IPvAnyAddress


class KillSwitchResetRequest(BaseModel):
    secret: str


class AgentEnablesRequest(BaseModel):
    auc_fade: bool | None = None
    attn_on: bool | None = None
    vrp_winged: bool | None = None
    pead_h60: bool | None = None      # VALIDATED_PEAD_ENABLED stays the master flag


class CredentialsUpdateRequest(BaseModel):
    kite_api_key: str | None = Field(default=None, min_length=1)
    kite_api_secret: str | None = Field(default=None, min_length=1)
    kite_user_id: str | None = Field(default=None, min_length=1)
    kite_password: str | None = Field(default=None, min_length=1)
    kite_totp_secret: str | None = Field(default=None, min_length=1)
    persist: bool = True


class CredentialsDeleteRequest(BaseModel):
    fields: list[Literal["kite_api_key", "kite_api_secret", "kite_access_token",
                         "kite_user_id", "kite_password", "kite_totp_secret"]] = Field(min_length=1)


class AppPasswordRequest(BaseModel):
    username: str | None = Field(default=None, min_length=1, max_length=50)
    new_password: str = Field(min_length=8, max_length=128)


# ── UI pages ──────────────────────────────────────────────────────────────────
@app.get("/login", include_in_schema=False)
def login_page():
    p = _STATIC / "login.html"
    if not p.exists():
        return HTMLResponse("<h2>Login page not found</h2>", status_code=404)
    return HTMLResponse(p.read_text())


@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    built = _FRONTEND_DIST / "index.html"
    if built.exists():
        return HTMLResponse(built.read_text())
    return HTMLResponse("<h2>Dashboard not built — run `npm run build` in frontend/. The API is at /api/v2.</h2>",
                        status_code=404)


@app.get("/", include_in_schema=False)
def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


# ── App auth (JWT) ────────────────────────────────────────────────────────────
@app.post("/auth/login", tags=["Auth"])
def app_login(form: OAuth2PasswordRequestForm = Depends()):
    if not authenticate(form.username, form.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token, expires_in = create_token(form.username)
    return {"access_token": token, "token_type": "bearer", "expires_in": expires_in}


@app.get("/auth/me", tags=["Auth"])
def me(request: Request):
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        user = decode_token(auth[7:])
        if user:
            return {"user": user, "auth_method": "jwt"}
    if request.headers.get("X-API-Key") == settings.api_key and settings.api_key:
        return {"user": "api-key-user", "auth_method": "api_key"}
    raise HTTPException(401, "Not authenticated")


# ── Kite login ────────────────────────────────────────────────────────────────
def _remember_kite_token(token: str) -> None:
    """Keep today's Kite access token so a restart doesn't need a fresh login (expires next morning)."""
    from secret_store import secret_store
    settings.kite_access_token = token
    try:
        secret_store.save({"kite_access_token": token})
    except Exception as e:
        logger.warning("Kite token not saved to vault: {}", e)
    tick_engine.on_kite_login()


@app.get("/auth/kite/status", tags=["Auth"])
def kite_status():
    try:
        profile = kite_client.profile()
        return {"connected": True, "account_id": profile.get("user_id", ""), "name": profile.get("user_name", "")}
    except Exception:
        return {"connected": False, "message": "No valid Kite session. Use Connect Kite Account."}


@app.get("/auth/kite/callback", tags=["Auth"], include_in_schema=False)
def kite_callback(request_token: str = "", action: str = "", status: str = ""):
    if status != "success" or not request_token:
        return HTMLResponse("<h2 style='font-family:sans-serif'>Kite login failed or cancelled.</h2>"
                            "<p><a href='/dashboard'>Back to the dashboard</a></p>", status_code=400)
    try:
        token = kite_client.set_access_token(request_token=request_token)
        _remember_kite_token(token)
        return HTMLResponse("<html><body style='font-family:sans-serif;text-align:center;padding-top:20vh'>"
                            "<h2>Kite connected</h2><p>You can close this tab.</p>"
                            "<p><a href='/dashboard'>Back to the dashboard</a></p></body></html>")
    except Exception as e:
        logger.error("Kite callback error: {}", e)
        return HTMLResponse("<h2 style='font-family:sans-serif'>Token exchange failed.</h2>"
                            "<p><a href='/dashboard'>Back to the dashboard and try again</a></p>", status_code=500)


@app.get("/auth/login-url", tags=["Auth"])
def login_url():
    return {"login_url": kite_client.login_url()}


@app.post("/auth/kite/auto-login", tags=["Auth"])
async def kite_auto_login_now():
    import kite_auto_login
    if not kite_auto_login.credentials_ready():
        raise HTTPException(422, "Automatic login needs " + ", ".join(kite_auto_login.missing_credentials()))
    try:
        token = await kite_auto_login.refresh_kite_token_async()
    except kite_auto_login.KiteLoginError as exc:
        raise HTTPException(502, str(exc))
    return {"status": "connected", "token_hint": token[:6] + "…", "auto_login": kite_auto_login.status()}


@app.get("/auth/kite/auto-login/status", tags=["Auth"])
def kite_auto_login_status():
    import kite_auto_login
    return kite_auto_login.status()


@app.post("/auth/token", tags=["Auth"])
def set_token(req: TokenRequest):
    t = kite_client.set_access_token(req.request_token, req.access_token)
    _remember_kite_token(t)
    return {"status": "ok", "access_token": t[:6] + "…"}


# ── Docs ──────────────────────────────────────────────────────────────────────
@app.get("/docs", include_in_schema=False)
def swagger_ui() -> HTMLResponse:
    if swagger_ui_bundle is None:
        return get_swagger_ui_html(openapi_url="/openapi.json", title="AlgoTrader Pro - Swagger UI")
    return get_swagger_ui_html(openapi_url="/openapi.json", title="AlgoTrader Pro - Swagger UI",
                               swagger_js_url="/swagger-static/swagger-ui-bundle.js",
                               swagger_css_url="/swagger-static/swagger-ui.css")


@app.get("/redoc", include_in_schema=False)
def redoc_ui() -> HTMLResponse:
    return get_redoc_html(openapi_url="/openapi.json", title="AlgoTrader Pro - ReDoc")


# ── Status ────────────────────────────────────────────────────────────────────
def kite_connection() -> dict:
    import kite_auto_login
    kite: dict = {"connected": False, "keys": bool(settings.kite_api_key and settings.kite_api_secret),
                  "auto_login": kite_auto_login.credentials_ready()}
    if settings.uses_kite_market_data:
        try:
            prof = kite_client.profile()
            kite.update(connected=True, account=prof.get("user_id", ""))
        except Exception as exc:
            kite["detail"] = "No Kite session for today" if settings.kite_api_key else "Kite API key not set"
            kite["error"] = str(exc)[:120]
    else:
        kite["detail"] = "PAPER_DATA_SOURCE=SIM — Kite not used (validated agents take no entries)"
    return kite


@app.get("/status/connections", tags=["Status"])
def connection_status():
    data = {**tick_engine.status(), "market_open": is_market_open(), "trading_day": _ic.is_trading_day(),
            "holiday": _ic.holiday_name(), "next_session": _ic.next_trading_day().isoformat(),
            "holiday_calendar_known": _ic.holiday_calendar_known()}
    from platform_scheduler import platform_scheduler
    return {"kite": kite_connection(), "data": data, "bot_running": platform_scheduler.bot_running(),
            "mode": settings.trading_mode, "time_ist": now_ist().strftime("%H:%M:%S")}


@app.get("/health", tags=["System"])
def health():
    from platform_scheduler import platform_scheduler
    return {"status": "ok", "version": VERSION, "mode": settings.trading_mode,
            "market_data_source": tick_engine.data_source(), "market_open": is_market_open(),
            "bot": "running" if platform_scheduler.bot_running() else "stopped",
            "agents": {n: a.state.running for n, a in ALL_AGENTS.items()},
            "agent_enabled": dict(bot_state._agent_enabled),
            "time": now_ist().strftime("%H:%M:%S IST")}


@app.get("/config/validate", tags=["System"])
def config_validate():
    creds = kite_client.validate_credentials()
    creds["paper_data_source"] = settings.paper_data_source
    creds["ticker_source"] = ("KITE_WS" if getattr(tick_engine, "_use_ws", False) else "KITE_REST") \
        if settings.uses_kite_market_data else "NONE"
    creds["admin_username"] = settings.admin_username
    creds["ready_to_trade"] = bool(creds.get("kite_api_key") and creds.get("kite_access_token"))
    return creds


# ── Bot control ───────────────────────────────────────────────────────────────
@app.post("/bot/start", tags=["Bot"])
async def start_bot(_body: dict | None = Body(default=None)):
    """Clear an operator Stop and set every enabled validated agent running (the 08:55 preflight does the same
    on every trading day unless the bot was stopped). Body ignored (older clients sent strategies / watchlist)."""
    from platform_scheduler import platform_scheduler
    return {"status": "started", "agents": platform_scheduler.bot_start()}


@app.post("/bot/stop", tags=["Bot"])
async def stop_bot():
    from platform_scheduler import platform_scheduler
    return {"status": "stopped", "agents": await platform_scheduler.bot_stop()}


@app.get("/bot/status", tags=["Bot"])
def bot_status():
    from platform_scheduler import platform_scheduler
    return {"running": platform_scheduler.bot_running(), "mode": settings.trading_mode,
            "agents": {n: a.get_status() for n, a in ALL_AGENTS.items()}}


# ── Agents ────────────────────────────────────────────────────────────────────
@app.get("/agents", tags=["Agents"])
def agents():
    return {n: a.get_status() for n, a in ALL_AGENTS.items()}


@app.post("/agents/{name}/pause", tags=["Agents"])
def pause_agent(name: str):
    a = ALL_AGENTS.get(name)
    if not a:
        raise HTTPException(404, "Not found")
    a.stop()
    return {"status": "paused"}


@app.post("/agents/{name}/resume", tags=["Agents"])
def resume_agent(name: str):
    a = ALL_AGENTS.get(name)
    if not a:
        raise HTTPException(404, "Not found")
    if not bot_state.is_agent_enabled(name):
        raise HTTPException(400, f"Agent '{name}' is disabled")
    a.start()
    return {"status": "resumed", "scheduled": True}


@app.get("/validated", tags=["Agents"])
def validated_status():
    """The lab-validated PAPER agents: full status, plus forward_test's auction comparison."""
    agents_ = {}
    for n, a in ALL_AGENTS.items():
        if getattr(a, "scheduled", False):
            try:
                agents_[n] = a.get_status()
            except Exception as exc:
                agents_[n] = {"error": str(exc)[:200]}
    try:
        import forward_test
        fwd = forward_test.status().get("summary")
    except Exception as exc:
        fwd = {"error": str(exc)[:200]}
    return {"validated_only": settings.validated_only, "scale": settings.validated_notional_scale,
            "agents": agents_, "forward_test": fwd}


@app.get("/settings/agent-enables", tags=["Settings"])
def get_agent_enables():
    return dict(bot_state._agent_enabled)


@app.post("/settings/agent-enables", tags=["Settings"])
def set_agent_enables(req: AgentEnablesRequest):
    for name, val in req.model_dump(exclude_none=True).items():
        bot_state.set_agent_enabled(name, val)
        if not val:
            a = ALL_AGENTS.get(name)
            if a and a.state.running:
                a.stop()
    return dict(bot_state._agent_enabled)


# ── Portfolio (paper book) ────────────────────────────────────────────────────
@app.get("/portfolio/positions", tags=["Portfolio"])
def positions():
    try:
        return kite_client.positions()
    except Exception as e:
        logger.error("Portfolio positions error: {}", e)
        raise HTTPException(500, "Unable to fetch positions")


@app.get("/portfolio/orders", tags=["Portfolio"])
def orders():
    try:
        return kite_client.orders()
    except Exception as e:
        logger.error("Portfolio orders error: {}", e)
        raise HTTPException(500, "Unable to fetch orders")


@app.post("/orders/squareoff", tags=["Orders"])
async def squareoff():
    """Flatten leftover MIS / NRML paper rows now (validated overnight holdings stay)."""
    from platform_scheduler import platform_scheduler
    ids = platform_scheduler.squareoff_book()
    return {"status": "ok", "squared_off": len(ids)}


@app.get("/paper-book/status", tags=["Orders"])
async def paper_book_status():
    import paper_book
    return paper_book.status()


@app.get("/market/live", tags=["Market"])
def live_market():
    return tick_engine.all_latest()


# ── Credentials vault ─────────────────────────────────────────────────────────
_CRED_FIELDS = ("kite_api_key", "kite_api_secret", "kite_user_id", "kite_password", "kite_totp_secret")


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost")


def _credential_view() -> dict:
    import kite_auto_login
    creds = kite_client.validate_credentials()
    for f in ("kite_user_id", "kite_password", "kite_totp_secret"):
        creds[f] = bool(getattr(settings, f))
    creds["auto_login"] = kite_auto_login.status()
    return creds


@app.post("/settings/credentials", tags=["Settings"])
def update_credentials(req: CredentialsUpdateRequest, request: Request):
    """Set Kite credentials (persist=true: encrypted vault, reloaded at startup). Values are never returned."""
    from secret_store import secret_store
    values = {f: v.strip() for f in _CRED_FIELDS if (v := getattr(req, f)) is not None and v.strip()}
    if not values:
        raise HTTPException(422, "No credentials given")
    if "kite_totp_secret" in values:
        seed = values["kite_totp_secret"].replace(" ", "").upper()
        if seed.isdigit() or len(seed) < 16:
            raise HTTPException(422, "KITE_TOTP_SECRET must be the base32 seed shown when enabling TOTP on Zerodha "
                                     "(16+ letters/digits), not the 6-digit code")
        try:
            import pyotp
            pyotp.TOTP(seed).now()
        except Exception:
            raise HTTPException(422, "KITE_TOTP_SECRET is not a valid base32 TOTP seed")
        values["kite_totp_secret"] = seed
    if req.persist and not settings.api_key and not _is_loopback(request):
        raise HTTPException(403, "Set API_KEY in .env before saving keys from another machine")
    if req.persist:
        try:
            secret_store.save(values)
        except Exception as e:
            logger.error("Credential vault write failed: {}", e)
            raise HTTPException(500, "Could not save to the encrypted vault — see server log")
    for f, v in values.items():
        setattr(settings, f, v)
    if set(values) & {"kite_api_key", "kite_api_secret"}:
        kite_client._kite = None
    logger.info("Credentials updated: {} (persisted={})", sorted(values), req.persist)
    return {"status": "saved" if req.persist else "updated", "persisted": req.persist,
            "credentials": _credential_view(), "vault": secret_store.status()}


@app.get("/settings/credentials/status", tags=["Settings"])
def credentials_status():
    from secret_store import secret_store
    return {"credentials": _credential_view(), "vault": secret_store.status()}


@app.delete("/settings/credentials", tags=["Settings"])
def delete_credentials(req: CredentialsDeleteRequest):
    from secret_store import secret_store
    secret_store.delete(req.fields)
    for f in req.fields:
        setattr(settings, f, "")
    if set(req.fields) & {"kite_access_token", "kite_api_key", "kite_api_secret"}:
        kite_client._kite = None
    logger.info("Credentials deleted: {}", sorted(req.fields))
    return {"status": "deleted", "credentials": _credential_view(), "vault": secret_store.status()}


@app.post("/settings/app-password", tags=["Settings"])
def update_app_password(req: AppPasswordRequest):
    from secret_store import secret_store
    if req.username is not None:
        settings.admin_username = req.username
    settings.admin_password_hash = hash_password(req.new_password)
    try:
        secret_store.save({"admin_username": settings.admin_username,
                           "admin_password_hash": settings.admin_password_hash})
        persisted = True
    except Exception as e:
        logger.error("Admin password not saved to vault: {}", e)
        persisted = False
    return {"status": "updated", "admin_username": settings.admin_username, "persisted": persisted}


# ── SEBI compliance ───────────────────────────────────────────────────────────
@app.get("/sebi/status", tags=["SEBI Compliance"])
def sebi_status():
    return sebi_compliance.status()


@app.get("/sebi/disclosures", tags=["SEBI Compliance"])
def sebi_disclosures():
    return sebi_compliance.get_disclosure_document()


@app.post("/sebi/kill-switch", tags=["SEBI Compliance"])
async def trigger_kill_switch(reason: str = "Manual kill switch"):
    sebi_compliance.trigger_kill_switch(reason)
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
def get_algo_ids():
    return {"algo_ids": APPROVED_ALGO_IDS, "total": len(APPROVED_ALGO_IDS)}


@app.get("/sebi/strategy-disclosure/{strategy}", tags=["SEBI Compliance"])
def strategy_disclosure(strategy: str):
    return sebi_compliance.get_strategy_logic_disclosure(strategy)


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


# ── Performance ledger ────────────────────────────────────────────────────────
def _day_pnl() -> float:
    return round(sum(float(getattr(a.state, "pnl_today", 0.0) or 0.0) for a in ALL_AGENTS.values()), 2)


@app.get("/performance/scoreboard", tags=["Performance"])
def performance_scoreboard():
    from performance import performance
    return {**performance.scoreboard(), "equity_curve": performance.equity_curve()}


@app.get("/performance/strategies", tags=["Performance"])
def performance_strategies():
    from performance import performance
    return {"strategies": list(performance.strategy_stats().values())}


@app.get("/performance/execution", tags=["Performance"])
def performance_execution():
    from performance import performance
    return performance.execution_stats()


@app.get("/performance/trades", tags=["Performance"])
def performance_trades(n: int = 100, strategy: str | None = None):
    from performance import performance
    rows = [t for t in performance.trades if not strategy or t["strategy"] == strategy]
    return {"trades": rows[-max(1, min(n, 1000)):]}


@app.post("/performance/snapshot", tags=["Performance"])
def performance_snapshot():
    from performance import performance
    return performance.snapshot_today()


# ── Research (read-only) ──────────────────────────────────────────────────────
@app.get("/shadow-registry", tags=["Research"])
def shadow_registry_status():
    import shadow_registry
    return {"enabled": shadow_registry.enabled(), **shadow_registry.ShadowRegistry().status()}


@app.get("/forward-test", tags=["Research"])
def forward_test_status(quotes: bool = False):
    import forward_test
    return forward_test.status(include_quotes=quotes)


@app.get("/backtest/history-bars", tags=["Research"])
async def backtest_history_bars(symbol: str, start: str, end: str, interval: str = "minute", exchange: str = "NSE"):
    """Kite history for one symbol over any date range (research data fetch scripts use it): minute bars are
    fetched in 60-day chunks. Indices use their Kite names ("NIFTY 50", "INDIA VIX").
    → {symbol, interval, bars: [[ts, o, h, l, c, v], ...]}"""
    from datetime import datetime as _dt
    if interval not in ("minute", "3minute", "5minute", "15minute", "60minute", "day"):
        raise HTTPException(400, "interval must be minute, 3/5/15/60minute or day")
    try:
        d0, d1 = _dt.fromisoformat(start), _dt.fromisoformat(end)
    except ValueError:
        raise HTTPException(400, "start/end must be YYYY-MM-DD")
    if d1 <= d0 or (d1 - d0).days > 800:
        raise HTTPException(400, "end must be after start, at most 800 days apart")
    sym = symbol.strip().upper()

    def _load():
        tok = kite_client.get_instrument_tokens([sym], exchange).get(sym)
        if not tok:
            return None
        return kite_client.historical_data(tok, d0, d1.replace(hour=23, minute=59), interval)
    rows = await asyncio.to_thread(_load)
    if rows is None:
        raise HTTPException(404, f"no instrument token for {exchange}:{sym}")

    def _ts(d):
        return (d.replace(tzinfo=None) if getattr(d, "tzinfo", None) else d).isoformat()
    return {"symbol": sym, "interval": interval,
            "bars": [[_ts(r["date"]), r["open"], r["high"], r["low"], r["close"], r.get("volume") or 0] for r in rows]}


# ── Startup / shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    import paper_book
    tick_engine.start_loop()
    paper_book.restore()                    # PAPER: the book from before a restart (never in LIVE; never raises)
    from platform_scheduler import platform_scheduler
    platform_scheduler.start()
    paper_book.start()                      # periodic snapshots
    try:
        tick_engine.warm_watchlist("startup")
    except Exception as exc:
        logger.warning("Warm watchlist failed: {}", exc)
    try:
        import live_readiness
        rd = live_readiness.check()
        (logger.warning if settings.trading_mode == "LIVE" else logger.info)(
            "[readiness] LIVE prerequisites: {} (blockers: {})", "met" if rd["ready_for_live"] else "NOT met",
            ", ".join(rd["blockers"]) or "none")
    except Exception as exc:
        logger.warning("LIVE readiness check failed: {}", exc)
    logger.info("FastAPI startup complete ({} mode, data: {})", settings.trading_mode, tick_engine.data_source())


@app.on_event("shutdown")
async def on_shutdown():
    try:
        import paper_book
        paper_book.save(force=True)         # last snapshot before a redeploy / restart
    except Exception as exc:
        logger.error("Paper book save at shutdown failed: {}", exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)


# Outermost middleware (added last): strips the /api prefix the built dashboard uses BEFORE the gate looks at the
# path, so /api/v2/overview is protected like /v2/overview.
@app.middleware("http")
async def _strip_frontend_prefixes(request: Request, call_next):
    path = request.scope.get("path", "")
    if path.startswith("/api/"):
        request.scope["path"] = path[4:]
        request.scope["raw_path"] = request.scope["path"].encode()
    return await call_next(request)
