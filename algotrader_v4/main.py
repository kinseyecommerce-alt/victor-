"""
main.py — AlgoTrader Pro v4 (tick-driven)
Real-time tick streaming via /ws WebSocket.
Kite used ONLY for order placement. Market data from NSE India API + yfinance.
"""
from __future__ import annotations
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from loguru import logger

from config import settings
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
from sebi_compliance import sebi_compliance, KillSwitchState, APPROVED_ALGO_IDS
from atomic_bracket import atomic_bracket_engine

app = FastAPI(title="AlgoTrader Pro v4", version="4.0.0",
              description="Tick-driven · NSE India API · yfinance · Kite for orders only")
app.add_middleware(CORSMiddleware, allow_origins=settings.origins_list,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

ws_clients: list[WebSocket] = []

async def broadcast(data: dict) -> None:
    for ws in ws_clients[:]:
        try:    await ws.send_json(data)
        except: ws_clients.remove(ws)

tick_engine.ws_broadcast = broadcast


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

class OrderRequest(BaseModel):
    symbol: str; exchange: str = "NSE"; transaction_type: str
    quantity: int; order_type: str = "MARKET"; product: str = "MIS"
    price: float = 0.0; trigger_price: float = 0.0

class SignalRequest(BaseModel):
    symbol: str; exchange: str = "NSE"; strategy: str = "intraday"

class RiskUpdateRequest(BaseModel):
    max_daily_loss: float | None = None
    max_position_size: float | None = None
    stop_loss_pct: float | None = None
    target_pct: float | None = None

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

class WhitelistIPRequest(BaseModel):
    ip: str


# ── Auth ─────────────────────────────────────────────────────────────────────────────
@app.get("/auth/login-url", tags=["Auth"])
def login_url(): return {"login_url": kite_client.login_url()}

@app.post("/auth/token", tags=["Auth"])
def set_token(req: TokenRequest):
    t = kite_client.set_access_token(req.request_token, req.access_token)
    return {"status": "ok", "access_token": t[:6] + "…"}


# ── Bot control ───────────────────────────────────────────────────────────────────
@app.post("/bot/start", tags=["Bot"])
async def start_bot(req: BotStartRequest):
    if master_agent.running:
        raise HTTPException(400, "Already running")
    watchlist = req.watchlist
    if not watchlist:
        selected = await symbol_scanner.run(strategies=req.strategies, force=req.force_scan)
        watchlist = symbol_scanner.all_selected_flat()
        if not watchlist:
            raise HTTPException(400, "Symbol scanner returned no symbols.")
    report = master_agent.start(req.strategies, watchlist)
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


# ── Market data ───────────────────────────────────────────────────────────────────
@app.get("/market/live", tags=["Market"])
def live_market(): return tick_engine.all_latest()

@app.get("/market/live/{symbol}", tags=["Market"])
def live_symbol(symbol: str):
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
    data = await tick_engine.get_option_chain(symbol)
    if not data: raise HTTPException(404, f"Option chain not available for {symbol}")
    return data


# ── Agents ───────────────────────────────────────────────────────────────────────────
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
    wl = master_agent._agent_watchlists.get(name, [])
    if wl:
        q = tick_engine.add_subscriber(f"agent_{name}")
        a.start(q)
    return {"status": "resumed", "symbols": [w["symbol"] for w in wl]}


# ── Backtest ───────────────────────────────────────────────────────────────────────
@app.post("/backtest/run", tags=["Backtest"])
def run_bt(req: BacktestRequest):
    return backtest_engine.run(
        req.symbol, req.exchange, req.strategy,
        req.lookback_days, force=True, walk_forward=req.walk_forward,
    ).to_dict()

@app.post("/backtest/batch", tags=["Backtest"])
def batch_bt(req: BatchBacktestRequest):
    results = backtest_engine.run_batch(req.symbols, req.strategy, walk_forward=req.walk_forward)
    return {"strategy": req.strategy,
            "passed":  [s for s, r in results.items() if r.passed],
            "failed":  [s for s, r in results.items() if not r.passed],
            "details": {s: r.to_dict() for s, r in results.items()}}

@app.get("/backtest/approved/{strategy}", tags=["Backtest"])
def approved(strategy: str):
    return {"strategy": strategy, "approved": backtest_engine.get_approved_symbols(strategy)}

@app.post("/backtest/compare", tags=["Backtest"])
def compare_strategies(req: CompareRequest):
    """Run all 4 strategies on a symbol and rank by Sharpe ratio."""
    return backtest_engine.compare_strategies(
        req.symbol, req.exchange, req.lookback_days, req.walk_forward,
    )

@app.get("/backtest/trades/{symbol}/{strategy}", tags=["Backtest"])
def download_trades(symbol: str, strategy: str):
    """Download the full trade log for a symbol/strategy as CSV."""
    key = (symbol.upper(), strategy)
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
    key = (symbol.upper(), strategy)
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
    import asyncio
    asyncio.create_task(asyncio.to_thread(backtest_engine.weekly_auto_backtest))
    return {"status": "weekly backtest started", "note": "runs in background, check logs"}


# ── Orders ───────────────────────────────────────────────────────────────────────────
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

@app.get("/portfolio/positions", tags=["Portfolio"])
def positions():
    try: return kite_client.positions()
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/portfolio/orders", tags=["Portfolio"])
def orders():
    try: return kite_client.orders()
    except Exception as e: raise HTTPException(500, str(e))


# ── Signals / Risk ──────────────────────────────────────────────────────────────────
@app.post("/signals/generate", tags=["AI Signal"])
async def gen_signal(req: SignalRequest):
    try:
        sig = signal_engine.generate(req.symbol, req.exchange, req.strategy)
        await broadcast({"event": "signal", "symbol": req.symbol, "signal": sig})
        return sig
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/risk/status", tags=["Risk"])
def risk_st(): return risk_manager.status()

@app.patch("/risk/update", tags=["Risk"])
def risk_update(req: RiskUpdateRequest):
    if req.max_daily_loss    is not None: settings.max_daily_loss    = req.max_daily_loss
    if req.max_position_size is not None: settings.max_position_size = req.max_position_size
    if req.stop_loss_pct     is not None: settings.stop_loss_pct     = req.stop_loss_pct
    if req.target_pct        is not None: settings.target_pct        = req.target_pct
    return risk_manager.status()


# ── WebSocket ─────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"event": "pong", "ts": datetime.now().isoformat()})
    except WebSocketDisconnect:
        ws_clients.remove(ws)


# ── Regime ───────────────────────────────────────────────────────────────────────
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


# ── Symbol scanner ──────────────────────────────────────────────────────────────────
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


# ── Trailing SL ───────────────────────────────────────────────────────────────────
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


# ── Brackets ───────────────────────────────────────────────────────────────────────
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


# ── SEBI ──────────────────────────────────────────────────────────────────────────────
@app.get("/sebi/status", tags=["SEBI Compliance"])
def sebi_status(): return sebi_compliance.status()

@app.get("/sebi/disclosures", tags=["SEBI Compliance"])
def sebi_disclosures(): return sebi_compliance.get_disclosure_document()

@app.post("/sebi/kill-switch", tags=["SEBI Compliance"])
def trigger_kill_switch(reason: str = "Manual kill switch"):
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
def get_algo_ids(): return {"algo_ids": APPROVED_ALGO_IDS, "total": len(APPROVED_ALGO_IDS)}

@app.get("/sebi/strategy-disclosure/{strategy}", tags=["SEBI Compliance"])
def strategy_disclosure(strategy: str): return sebi_compliance.get_strategy_logic_disclosure(strategy)

@app.post("/sebi/reset-kill-switch", tags=["SEBI Compliance"])
def reset_kill_switch():
    sebi_compliance.reset_kill_switch()
    return {"status": "ACTIVE", "note": "Kill switch reset. Trading re-enabled."}

@app.post("/sebi/whitelist-ip", tags=["SEBI Compliance"])
def whitelist_ip(req: WhitelistIPRequest):
    sebi_compliance.add_whitelisted_ip(req.ip)
    return {"status": "added", "ip": req.ip}


# ── Health ────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "4.0.0", "mode": settings.trading_mode,
            "architecture": "tick-driven 1s",
            "market_data_source": "NSE India API + yfinance (Kite = orders only)",
            "market_open": is_market_open(),
            "master": "running" if master_agent.running else "stopped",
            "tick_engine": "running" if tick_engine._running else "stopped",
            "agents": {n: a.state.running for n, a in ALL_AGENTS.items()},
            "subscribed_symbols": tick_engine.symbols(),
            "time": datetime.now().strftime("%H:%M:%S IST")}


# ── Startup ─────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    tick_engine.start_loop()
    atomic_bracket_engine.ws_broadcast = broadcast
    logger.info("FastAPI startup: tick engine + atomic bracket engine launched")
    asyncio.create_task(symbol_scanner.run())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
