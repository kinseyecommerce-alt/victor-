"""
ai_trader.py — Autonomous Claude trading agent for AlgoTrader Pro.

Claude runs an agentic tool-use loop over the bot's own state: it reads quotes,
indicators, regime and its own positions, and can open or close intraday
(MIS) equity positions. It answers a dashboard chat with the same tools.

Real time: code watches every tick (_detect, microseconds) — the intraday
pattern engine, momentum bursts, and AI positions nearing their stop or
flipping trend — and the moment one fires, Claude is called with that live
data already in the prompt (_handle_event), so it can usually decide and
place or close the order in a single step. A slower full review
(AI_AGENT_INTERVAL_MIN) runs as a safety net.

Safety model — the model decides, the code enforces:
  * every trade goes through _place_trade(), which applies the AI_AGENT_* hard
    limits (value cap, trades/day, open positions, SL distance, trading window,
    fresh tick) and then the same order_guard → risk_manager → SEBI → SL-M path
    the strategy agents use; a rejected check is returned to Claude as an error
  * LIVE orders only when AI_AGENT_ALLOW_LIVE=true (PAPER otherwise)
  * autonomous cycles stop once today's estimated API spend hits the cap
  * at 15:00 IST every AI position is closed by code, not by the model
  * the TSL engine and AI targets are driven from the tick feed, so stops work
    between decisions
  * real-time alerts: per-symbol cooldown, min gap between calls, stale alerts
    dropped, and an entry is rejected if price drifted since the alert
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from datetime import datetime, time
from typing import Any, Optional

import anthropic
from loguru import logger

from config import settings
from ist_clock import is_market_open, now_ist
from kite_client import kite_client
from order_guard import order_guard
from risk_manager import risk_manager
from tick_engine import tick_engine
from trailing_sl_engine import trailing_sl_engine

AGENT = "ai"

ENTRY_START = time(9, 20)    # no entries in the first 5 minutes
ENTRY_END   = time(14, 45)   # no new entries after this
FORCE_EXIT  = time(15, 0)    # code closes every AI position here
TICK_MAX_AGE_S = 120         # quotes older than this are not tradable

# Per-MTok prices for spend tracking (Anthropic first-party rates, 2026-10).
# Unknown models are costed at Opus 5.5 rates.
_PRICES = {
    "claude-opus-5-5":   {"in": 4.00,  "out": 20.0, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"in": 2.00,  "out": 10.0, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-5-5":  {"in": 0.10,  "out": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "claude-fable-5-1":  {"in": 10.00, "out": 50.0, "cache_read": 0.25, "cache_write": 12.50},
}
# Models that accept the server-side refusal fallback (`fallbacks: "default"`)
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}

SYSTEM_PROMPT = """You are the autonomous intraday trader inside AlgoTrader Pro, an algorithmic \
trading system for NSE (India) equities. You trade with real or paper money on behalf of the \
account owner, so capital preservation comes first: a skipped trade costs nothing, a bad one costs money.

How you work
- Use your tools to look before you act: get_market_overview first, then get_watchlist_quotes, \
then get_symbol_detail for anything you are considering, and get_ai_positions for what you hold.
- Only trade a clear setup: trend and momentum agreeing across several indicators (VWAP side, \
EMA9/21/50 stack, ADX with the matching DI line, MACD histogram, volume ratio), or a clean \
mean-reversion with a defined level. Prefer no trade over a marginal one. Most cycles should end \
with no new trade.
- Every trade needs a stop-loss on the correct side of price and a target giving at least 1.5x \
the risk. Base the stop on structure or ATR (roughly 1-1.5x ATR), inside the allowed range \
reported by get_market_overview.
- Size so that quantity x price stays within the per-trade value limit. Smaller is fine.
- Close a position when its reason to exist is gone (trend broken, momentum reversed), not on \
noise. Stops and targets are already enforced automatically between your cycles.
- Respect the market regime and the risk status. If trading is halted, paused, or limits are \
reached, do not try to work around it.
- The tools enforce hard limits. If a tool rejects an action, accept it and do not retry the \
same action with tweaked numbers to slip past a limit.

Real-time alerts
- A REAL-TIME ALERT message means code just detected something on a live tick and already gives \
you the quote, indicators, recent candles, your positions and limits. Decide fast: act with \
place_trade or close_position, or pass. Fetch more data only if something essential is missing. \
An alert is a reason to look, not a reason to trade — most alerts should be passed.
- If you pass, reply with one line starting "PASS:" and the reason.

When you finish, reply with a short plain summary: what you looked at, what you did and why, or \
why you did nothing. In chat, answer the owner's question directly and use tools for facts — \
never invent prices or positions."""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


# Fixed order and content — part of the cached prompt prefix.
TOOLS: list[dict] = [
    _tool("get_market_overview",
          "Time, trading mode, market status, regime, account risk status, the AI's hard limits "
          "and how much of each is left today, and the AI's recent actions. Call this first.",
          {}, []),
    _tool("get_watchlist_quotes",
          "Compact live quote + key indicators for every streamed symbol (LTP, change %, trend, "
          "RSI, VWAP, EMA9/21, ADX, +DI/-DI, MACD hist, volume ratio, Supertrend, Keltner, CCI, MFI).",
          {}, []),
    _tool("get_symbol_detail",
          "All live indicators for one symbol plus its last 30 one-minute candles.",
          {"symbol": {"type": "string", "description": "NSE trading symbol, e.g. RELIANCE"}},
          ["symbol"]),
    _tool("get_ai_positions",
          "Open positions this AI holds: entry, live price, unrealised P&L, stop and target.",
          {}, []),
    _tool("place_trade",
          "Open an intraday (MIS) NSE equity position at market with a protective stop-loss. "
          "Rejected unless every hard limit and risk/SEBI check passes.",
          {
              "symbol":    {"type": "string", "description": "NSE trading symbol from the watchlist"},
              "side":      {"type": "string", "enum": ["BUY", "SELL"]},
              "quantity":  {"type": "integer", "description": "Shares; quantity x price within the per-trade limit"},
              "stop_loss": {"type": "number", "description": "Stop price: below price for BUY, above for SELL"},
              "target":    {"type": "number", "description": "Target price: at least 1.5x the risk away"},
              "reasoning": {"type": "string", "description": "One or two sentences: the setup and why now"},
          },
          ["symbol", "side", "quantity", "stop_loss", "target", "reasoning"]),
    _tool("close_position",
          "Close an open AI position at market (cancels its stop-loss order).",
          {
              "symbol":    {"type": "string"},
              "reasoning": {"type": "string", "description": "Why the position should go now"},
          },
          ["symbol", "reasoning"]),
]


class AITrader:
    def __init__(self) -> None:
        self.enabled: bool = False                     # autonomous cycles on/off
        self._loop_task:  Optional[asyncio.Task] = None
        self._tick_task:  Optional[asyncio.Task] = None
        self._queue:      Optional[asyncio.Queue] = None
        self._client:     Optional[anthropic.AsyncAnthropic] = None
        self._action_lock = asyncio.Lock()             # one trade action at a time
        self._cycle_lock  = asyncio.Lock()
        self._chats: dict[str, list] = {}              # conversation_id → messages
        self._targets: dict[str, float] = {}           # entry order id → target price
        self._open: dict[str, str] = {}                # entry order id → symbol (detects stop exits)
        self.log: deque = deque(maxlen=300)            # decisions, actions, errors
        self._day = now_ist().date()
        self.trades_today = 0
        self.cycles_today = 0
        self.cost_today_usd = 0.0
        self.tokens_today = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self.last_cycle_at: Optional[str] = None
        self.last_summary: str = ""
        # Real-time alerts
        self._events: dict[str, dict] = {}             # symbol → latest pending alert (coalesced)
        self._event_signal = asyncio.Event()
        self._event_task: Optional[asyncio.Task] = None
        self._trigger_ts: dict[tuple, float] = {}      # (symbol, kind) → monotonic time of last alert
        self._prev_st: dict[str, str] = {}             # symbol → Supertrend dir at last tick
        self._event_ltp: dict[str, float] = {}         # symbol → price at the alert being decided
        self._last_call = 0.0
        self._scout = None                             # IntradayAgent used only as a pattern detector
        self.events_today = 0
        self.decisions_today = 0
        self._latencies: deque = deque(maxlen=50)
        self.last_event: Optional[dict] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, watchlist: Optional[list[dict]] = None) -> dict:
        """Enable autonomous cycles. Subscribes the watchlist so quotes stream."""
        if watchlist:
            tick_engine.subscribe(watchlist)
        if not tick_engine.symbols():
            from symbol_scanner import NIFTY_50
            tick_engine.subscribe([{"symbol": s, "exchange": "NSE"} for s in NIFTY_50[:20]])
        self._ensure_tick_monitor()
        self.enabled = True
        if settings.ai_agent_interval_min > 0 and (not self._loop_task or self._loop_task.done()):
            self._loop_task = asyncio.create_task(self._cycle_loop())
        if settings.ai_agent_realtime and (not self._event_task or self._event_task.done()):
            self._event_task = asyncio.create_task(self._event_worker())
        rt = "real-time alerts" if settings.ai_agent_realtime else "no real-time alerts"
        review = (f"review every {settings.ai_agent_interval_min} min"
                  if settings.ai_agent_interval_min > 0 else "no periodic review")
        self._record("system", f"autonomous trading ON ({self._mode_label()}, {rt}, {review})")
        return self.status()

    def stop(self) -> dict:
        """Disable autonomous cycles. Open AI positions keep their stops and are still
        force-closed at 15:00."""
        self.enabled = False
        for task in (self._loop_task, self._event_task):
            if task and not task.done():
                task.cancel()
        self._events.clear()
        self._record("system", "autonomous trading OFF")
        return self.status()

    def _ensure_tick_monitor(self) -> None:
        if not self._tick_task or self._tick_task.done():
            self._queue = tick_engine.add_subscriber("ai_trader")
            self._tick_task = asyncio.create_task(self._tick_monitor())

    def status(self) -> dict:
        self._roll_day()
        return {
            "enabled":           self.enabled,
            "mode":              self._mode_label(),
            "can_trade":         self._trading_allowed()[0],
            "blocked_reason":    self._trading_allowed()[1],
            "model":             settings.ai_agent_model,
            "effort":            settings.ai_agent_effort,
            "interval_min":      settings.ai_agent_interval_min,
            "trades_today":      self.trades_today,
            "cycles_today":      self.cycles_today,
            "open_positions":    len(self._ai_entries()),
            "cost_today_usd":    round(self.cost_today_usd, 4),
            "tokens_today":      dict(self.tokens_today),
            "last_cycle_at":     self.last_cycle_at,
            "last_summary":      self.last_summary,
            "realtime":          settings.ai_agent_realtime,
            "realtime_effort":   settings.ai_agent_realtime_effort,
            "alerts_today":      self.events_today,
            "decisions_today":   self.decisions_today,
            "pending_alerts":    len(self._events),
            "avg_decision_ms":   int(sum(self._latencies) / len(self._latencies)) if self._latencies else None,
            "last_alert":        self.last_event,
            "limits":            self._limits(),
            "api_key_set":       bool(settings.anthropic_api_key),
        }

    # ── Autonomous cycle ──────────────────────────────────────────────────

    async def _cycle_loop(self) -> None:
        while self.enabled:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._record("error", f"cycle failed: {exc}")
                logger.exception("[ai] cycle failed")
            await asyncio.sleep(max(1, settings.ai_agent_interval_min) * 60)

    async def run_cycle(self) -> Optional[str]:
        """One autonomous decision cycle. Returns Claude's summary, or None if skipped."""
        self._roll_day()
        t = now_ist().time()
        if t >= FORCE_EXIT:
            await self.close_all("end of AI trading window (15:00)")
            return None
        if not is_market_open():
            return None
        if self.cost_today_usd >= settings.ai_agent_max_daily_cost_usd:
            self._record("system", f"daily API cost cap ${settings.ai_agent_max_daily_cost_usd:.2f} "
                                   "reached — no more cycles today")
            return None
        if not settings.anthropic_api_key:
            self._record("error", "ANTHROPIC_API_KEY not set — cycle skipped")
            return None
        if self._cycle_lock.locked():
            return None
        async with self._cycle_lock:
            self.cycles_today += 1
            self.last_cycle_at = now_ist().strftime("%H:%M:%S")
            messages = [{"role": "user", "content":
                         f"Decision cycle at {self.last_cycle_at} IST. Review the market and your "
                         "positions with your tools and decide whether to open, close or do nothing."}]
            summary = await self._run_agent(messages, purpose="cycle")
            self.last_summary = summary
            self._record("cycle", summary)
            return summary

    # ── Chat ──────────────────────────────────────────────────────────────

    async def chat(self, message: str, conversation_id: Optional[str] = None) -> dict:
        if not settings.anthropic_api_key:
            return {"conversation_id": conversation_id, "reply": "ANTHROPIC_API_KEY is not set on the server.",
                    "actions": []}
        if self.cost_today_usd >= settings.ai_agent_max_daily_cost_usd:
            return {"conversation_id": conversation_id,
                    "reply": f"Today's AI API cost cap (${settings.ai_agent_max_daily_cost_usd:.2f}) is reached.",
                    "actions": []}
        cid = conversation_id if conversation_id in self._chats else uuid.uuid4().hex[:12]
        messages = self._chats.setdefault(cid, [])
        if len(messages) > 60:                      # keep long chats bounded: start fresh
            messages.clear()
        messages.append({"role": "user", "content": message})
        before = len(self.log)
        reply = await self._run_agent(messages, purpose="chat")
        actions = [e for e in list(self.log)[before:] if e["kind"] in ("trade", "close", "rejected")]
        self._record("chat", f"Q: {message[:120]} → {reply[:200]}")
        return {"conversation_id": cid, "reply": reply, "actions": actions}

    # ── Agentic loop ──────────────────────────────────────────────────────

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    async def _create(self, messages: list, effort: Optional[str] = None):
        kwargs: dict[str, Any] = dict(
            model=settings.ai_agent_model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": effort or settings.ai_agent_effort},
            cache_control={"type": "ephemeral"},     # caches tools + system + history prefix
        )
        client = self._get_client()
        if settings.ai_agent_model in _FALLBACK_MODELS:
            # Server-side refusal fallback: a declined request is re-run on a fallback model
            return await client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return await client.messages.create(**kwargs)

    async def _run_agent(self, messages: list, purpose: str, effort: Optional[str] = None) -> str:
        """Manual tool-use loop, capped at ai_agent_max_steps model calls."""
        for _ in range(max(1, settings.ai_agent_max_steps)):
            try:
                response = await self._create(messages, effort)
            except anthropic.RateLimitError:
                return "Claude API rate limit hit — try again shortly."
            except anthropic.AuthenticationError:
                return "Claude API key was rejected."
            except anthropic.APIStatusError as exc:
                logger.error("[ai] API error {}: {}", exc.status_code, exc.message)
                return f"Claude API error ({exc.status_code})."
            except anthropic.APIConnectionError:
                return "Could not reach the Claude API."
            self._track_usage(response)

            # Keep every block (thinking, fallback, tool_use) — required for valid history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                category = getattr(getattr(response, "stop_details", None), "category", None)
                self._record("error", f"model declined ({category})")
                return "The model declined this request."
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += " [truncated]"
                return text or "(no reply)"

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    out = await self._execute(block.name, block.input or {})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(out, default=str)})
                except ToolRejected as exc:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": f"REJECTED: {exc}", "is_error": True})
                except Exception as exc:
                    logger.exception("[ai] tool {} failed", block.name)
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": f"Error: {exc}", "is_error": True})
            messages.append({"role": "user", "content": results})

        self._record("error", f"{purpose}: step limit ({settings.ai_agent_max_steps}) reached")
        return "Stopped: reached the tool-step limit for this turn."

    def _track_usage(self, response) -> None:
        u = response.usage
        p = _PRICES.get(settings.ai_agent_model, _PRICES["claude-opus-5-5"])
        inp   = getattr(u, "input_tokens", 0) or 0
        out   = getattr(u, "output_tokens", 0) or 0
        cr    = getattr(u, "cache_read_input_tokens", 0) or 0
        cw    = getattr(u, "cache_creation_input_tokens", 0) or 0
        self.tokens_today["input"] += inp
        self.tokens_today["output"] += out
        self.tokens_today["cache_read"] += cr
        self.tokens_today["cache_write"] += cw
        self.cost_today_usd += (inp * p["in"] + out * p["out"]
                                + cr * p["cache_read"] + cw * p["cache_write"]) / 1_000_000

    # ── Tools ─────────────────────────────────────────────────────────────

    async def _execute(self, name: str, args: dict) -> Any:
        if name == "get_market_overview":
            return self._overview()
        if name == "get_watchlist_quotes":
            return self._quotes()
        if name == "get_symbol_detail":
            return self._detail(str(args.get("symbol", "")).upper())
        if name == "get_ai_positions":
            return self._positions()
        if name == "place_trade":
            async with self._action_lock:
                return await self._place_trade(args)
        if name == "close_position":
            async with self._action_lock:
                return await self._close_position(str(args.get("symbol", "")).upper(),
                                                  str(args.get("reasoning", "")))
        raise ToolRejected(f"unknown tool {name}")

    def _overview(self) -> dict:
        from market_regime import regime_detector
        from sebi_compliance import sebi_compliance
        allowed, why = self._trading_allowed()
        recent = [e for e in self.log if e["kind"] in ("trade", "close", "rejected", "cycle")][-6:]
        return {
            "time_ist":       now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            "mode":           self._mode_label(),
            "market_open":    is_market_open(),
            "entry_window":   f"{ENTRY_START:%H:%M}-{ENTRY_END:%H:%M} IST; AI positions force-closed at {FORCE_EXIT:%H:%M}",
            "can_open_trades": allowed,
            "blocked_reason": why,
            "regime":         regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN",
            "risk":           risk_manager.status(),
            "sebi_state":     sebi_compliance.status().get("state"),
            "limits":         self._limits(),
            "recent_ai_actions": [{"t": e["t"], "kind": e["kind"], "text": e["text"][:200]} for e in recent],
        }

    def _quotes(self) -> dict:
        rows = {}
        for sym, d in tick_engine.all_latest().items():
            rows[sym] = {k: d.get(k) for k in (
                "ltp", "change_pct", "trend", "momentum", "volatility", "rsi_14", "vwap", "ema9",
                "ema21", "adx_14", "plus_di", "minus_di", "macd_hist", "vol_ratio",
                "supertrend_dir", "squeeze_on", "keltner_u", "keltner_l", "cci_20", "mfi_14")}
            rows[sym]["age_s"] = self._tick_age(sym)
        return {"count": len(rows), "quotes": rows}

    def _detail(self, symbol: str) -> dict:
        tick, ind = tick_engine.latest(symbol)
        if not tick or not ind:
            raise ToolRejected(f"{symbol} is not streaming — pick a symbol from get_watchlist_quotes")
        indicators = {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in vars(ind).items() if k not in ("symbol", "computed_at")}
        buf = tick_engine._bufs_1min.get(symbol)
        candles = []
        if buf:
            for c in buf.candles()[-30:]:
                candles.append([c.ts.strftime("%H:%M"), round(c.open, 2), round(c.high, 2),
                                round(c.low, 2), round(c.close, 2), c.volume])
        return {"symbol": symbol, "ltp": tick.ltp, "bid": tick.bid, "ask": tick.ask,
                "age_s": self._tick_age(symbol), "indicators": indicators,
                "candles_1m": {"columns": ["time", "open", "high", "low", "close", "volume"],
                               "rows": candles}}

    def _positions(self) -> dict:
        out = []
        for oid, e in self._ai_entries().items():
            pos = trailing_sl_engine._positions.get(oid)
            tick, _ = tick_engine.latest(e["symbol"])
            ltp = tick.ltp if tick else None
            entry = pos.entry_price if pos else None
            qty = pos.quantity if pos else None
            pnl = None
            if ltp and entry and qty:
                pnl = round((ltp - entry) * qty * (1 if e["side"] == "BUY" else -1), 2)
            out.append({"symbol": e["symbol"], "side": e["side"], "quantity": qty,
                        "entry": entry, "ltp": ltp, "unrealised_pnl": pnl,
                        "stop_loss": pos.current_sl if pos else None,
                        "target": self._targets.get(oid)})
        return {"open": out}

    # ── Actions (hard limits enforced here) ───────────────────────────────

    async def _place_trade(self, a: dict) -> dict:
        from agents.base_agent import (
            place_sl_with_retry, flatten_unprotected, _setup_tsl_callbacks, _tsl_sl_orders,
        )
        from sebi_compliance import sebi_compliance
        from market_regime import regime_detector

        sym  = str(a.get("symbol", "")).upper().strip()
        side = str(a.get("side", "")).upper()
        qty  = int(a.get("quantity") or 0)
        sl   = float(a.get("stop_loss") or 0)
        tgt  = float(a.get("target") or 0)
        why  = str(a.get("reasoning", ""))[:300]

        def reject(msg: str):
            self._record("rejected", f"{side} {qty} {sym}: {msg}")
            raise ToolRejected(msg)

        allowed, blocked = self._trading_allowed()
        if not allowed:
            reject(blocked)
        if side not in ("BUY", "SELL"):
            reject("side must be BUY or SELL")
        tick, _ = tick_engine.latest(sym)
        if not tick or tick.ltp <= 0:
            reject(f"{sym} is not streaming")
        if self._tick_age(sym) > TICK_MAX_AGE_S:
            reject(f"{sym} quote is stale ({self._tick_age(sym)}s old)")
        ltp = float(tick.ltp)
        ref = self._event_ltp.get(sym)
        if ref and abs(ltp - ref) / ref * 100 > settings.ai_agent_max_drift_pct:
            reject(f"price moved {abs(ltp - ref) / ref * 100:.2f}% since the alert "
                   f"(₹{ref:.2f} → ₹{ltp:.2f}); the setup is stale")
        if qty <= 0:
            reject("quantity must be positive")
        if qty * ltp > settings.ai_agent_max_position_value:
            reject(f"value ₹{qty * ltp:,.0f} exceeds per-trade limit "
                   f"₹{settings.ai_agent_max_position_value:,.0f} (max qty {int(settings.ai_agent_max_position_value // ltp)})")
        risk_dist = (ltp - sl) if side == "BUY" else (sl - ltp)
        if risk_dist <= 0:
            reject(f"stop_loss {sl} is on the wrong side of price {ltp}")
        sl_pct = risk_dist / ltp * 100
        if not (settings.ai_agent_min_sl_pct <= sl_pct <= settings.ai_agent_max_sl_pct):
            reject(f"stop is {sl_pct:.2f}% away; allowed {settings.ai_agent_min_sl_pct}-"
                   f"{settings.ai_agent_max_sl_pct}%")
        reward = (tgt - ltp) if side == "BUY" else (ltp - tgt)
        if reward < 1.5 * risk_dist:
            reject(f"target {tgt} gives {reward / risk_dist:.2f}x risk; need >= 1.5x")
        if self.trades_today >= settings.ai_agent_max_trades_per_day:
            reject(f"daily AI trade limit ({settings.ai_agent_max_trades_per_day}) reached")
        if len(self._ai_entries()) >= settings.ai_agent_max_open_positions:
            reject(f"max open AI positions ({settings.ai_agent_max_open_positions}) reached")
        ok, msg = order_guard.can_place(sym, AGENT, side)
        if not ok:
            reject(msg)
        if order_guard.is_symbol_active_anywhere(sym):
            reject(f"{sym} already has an open position from another strategy")
        ok, msg = risk_manager.check_before_order(sym, qty, ltp, side)
        if not ok:
            reject(msg)
        ok, _algo, msg = sebi_compliance.pre_order_check(
            strategy=AGENT, symbol=sym, exchange="NSE", transaction_type=side, quantity=qty,
            order_type="MARKET", price_at_signal=ltp, signal_source="ai_trader",
            regime=regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN",
        )
        if not ok:
            reject(msg)

        order_id = kite_client.place_order(
            tradingsymbol=sym, exchange="NSE", transaction_type=side, quantity=qty,
            order_type="MARKET", product="MIS", tag="Agent-ai",
        )
        sebi_compliance.record_order_id(AGENT, sym, order_id)

        sl = round(sl, 2)
        sl_oid = await place_sl_with_retry(AGENT, sym, "NSE", side, qty, "MIS", sl)
        if sl_oid is None and await flatten_unprotected(AGENT, sym, "NSE", side, qty, "MIS"):
            reject("stop-loss order was rejected by the broker; entry was closed at market")

        order_guard.register_order(sym, AGENT, side, order_id)
        risk_manager.position_opened()
        _setup_tsl_callbacks()
        _tsl_sl_orders[order_id] = {
            "sl_order_id": sl_oid, "product": "MIS", "exchange": "NSE",
            "symbol": sym, "guard_symbol": sym, "agent": AGENT, "side": side,
        }
        pos = trailing_sl_engine.register(
            symbol=sym, strategy=AGENT, side=side, entry_price=ltp,
            quantity=qty, order_id=order_id,
        )
        pos.current_sl = sl                     # software stop = the AI's stop, not the config default
        self._targets[order_id] = round(tgt, 2)
        self._open[order_id] = sym
        self.trades_today += 1
        self._ensure_tick_monitor()

        text = f"{side} {qty} {sym} @ ~₹{ltp:.2f} SL ₹{sl:.2f} TGT ₹{tgt:.2f} — {why}"
        self._record("trade", text)
        await self._notify(f"🧠 <b>[AI]</b> {text}")
        return {"status": "placed", "order_id": order_id, "symbol": sym, "side": side,
                "quantity": qty, "approx_price": ltp, "stop_loss": sl, "target": round(tgt, 2),
                "mode": self._mode_label()}

    async def _close_position(self, symbol: str, reason: str, by_code: bool = False) -> dict:
        from agents.base_agent import _close_tracked
        found = [(oid, e) for oid, e in self._ai_entries().items() if e["symbol"] == symbol]
        if not found:
            raise ToolRejected(f"no open AI position in {symbol}")
        closed = []
        for oid, e in found:
            pos = trailing_sl_engine._positions.get(oid)
            if not pos:
                continue
            tick, _ = tick_engine.latest(symbol)
            ltp = tick.ltp if tick else pos.entry_price
            pnl = (ltp - pos.entry_price) * pos.quantity * (1 if pos.side == "BUY" else -1)
            _close_tracked(oid, symbol, pos.side, pos.quantity, pnl, "Agent-ai-EXIT", AGENT)
            self._targets.pop(oid, None)
            self._open.pop(oid, None)
            closed.append({"symbol": symbol, "quantity": pos.quantity, "exit_price": ltp,
                           "pnl": round(pnl, 2)})
            text = f"CLOSE {symbol} @ ~₹{ltp:.2f} P&L ₹{pnl:,.0f} — {reason[:200]}"
            self._record("close", text if not by_code else f"[auto] {text}")
            await self._notify(f"🧠 <b>[AI]</b> {text}")
        return {"status": "closed", "closed": closed}

    async def close_all(self, reason: str) -> list:
        out = []
        for sym in sorted({e["symbol"] for e in self._ai_entries().values()}):
            async with self._action_lock:
                try:
                    out.append(await self._close_position(sym, reason, by_code=True))
                except ToolRejected:
                    pass
        return out

    # ── Tick monitor: stops and targets between cycles ────────────────────

    async def _tick_monitor(self) -> None:
        while True:
            try:
                snap = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self._on_snap(snap)
            except Exception as exc:
                logger.error("[ai] tick monitor error: {}", exc)

    async def _on_snap(self, snap) -> None:
        held = {oid: e for oid, e in self._ai_entries().items() if e["symbol"] == snap.symbol}
        if held:
            await trailing_sl_engine.on_tick(snap.symbol, snap.tick.ltp, snap.indicators.atr_14)
            ltp = snap.tick.ltp
            for oid, e in held.items():
                tgt = self._targets.get(oid)
                if not tgt or oid not in self._ai_entries():     # closed by the TSL just now
                    continue
                if (e["side"] == "BUY" and ltp >= tgt) or (e["side"] == "SELL" and ltp <= tgt):
                    async with self._action_lock:
                        await self._close_position(snap.symbol, f"target ₹{tgt:.2f} hit", by_code=True)
        self._note_stop_exits()
        if held and now_ist().time() >= FORCE_EXIT:
            await self.close_all("end of AI trading window (15:00)")
        if self.enabled and settings.ai_agent_realtime:
            self._detect(snap)

    # ── Real-time: trigger detection (every tick, no API calls) ───────────

    def _detect(self, snap) -> None:
        sym, ind, ltp = snap.symbol, snap.indicators, snap.tick.ltp
        st_prev = self._prev_st.get(sym)
        self._prev_st[sym] = ind.supertrend_dir

        held = [(o, e) for o, e in self._ai_entries().items() if e["symbol"] == sym]
        if held:
            oid, e = held[0]
            pos = trailing_sl_engine._positions.get(oid)
            if not pos:
                return
            long = e["side"] == "BUY"
            risk = (pos.entry_price - pos.current_sl) if long else (pos.current_sl - pos.entry_price)
            lost = (pos.entry_price - ltp) if long else (ltp - pos.entry_price)
            if risk > 0 and lost / risk >= 0.6:
                self._raise(snap, "position_risk",
                            f"{e['side']} position has used {lost / risk:.0%} of its stop distance")
            elif st_prev and st_prev != ind.supertrend_dir and ind.supertrend_dir == ("DOWN" if long else "UP"):
                self._raise(snap, "trend_flip",
                            f"Supertrend flipped {ind.supertrend_dir} against the {e['side']} position")
            return

        if not self._can_consider_entry(sym):
            return
        scout = self._get_scout()
        if len(snap.candles_1min) >= scout.min_candles_1min:
            try:
                action, sig = scout.evaluate_tick(snap)
            except Exception:
                action, sig = "HOLD", None
            if action in ("BUY", "SELL") and sig:
                self._raise(snap, "setup", sig.get("trigger", action),
                            {"side": action, "suggested_stop": sig.get("stop_loss"),
                             "suggested_target": sig.get("target")})
                return
        if snap.candles_1min:
            c = snap.candles_1min[-1]
            move = (ltp - c.open) / c.open * 100 if c.open else 0.0
            if abs(move) >= settings.ai_agent_burst_pct and ind.volume_ratio >= 2.0:
                self._raise(snap, "momentum_burst",
                            f"{move:+.2f}% within the current minute on {ind.volume_ratio:.1f}x volume")

    def _can_consider_entry(self, sym: str) -> bool:
        if not self._trading_allowed()[0] or not settings.anthropic_api_key:
            return False
        if self.cost_today_usd >= settings.ai_agent_max_daily_cost_usd:
            return False
        if self.trades_today >= settings.ai_agent_max_trades_per_day:
            return False
        if len(self._ai_entries()) >= settings.ai_agent_max_open_positions:
            return False
        return not order_guard.is_symbol_active_anywhere(sym)

    def _get_scout(self):
        if self._scout is None:
            from agents.strategy_agents import IntradayAgent
            self._scout = IntradayAgent()
        return self._scout

    def _raise(self, snap, kind: str, reason: str, extra: Optional[dict] = None) -> None:
        import time as _t
        key = (snap.symbol, kind)
        now = _t.monotonic()
        if now - self._trigger_ts.get(key, -1e9) < settings.ai_agent_trigger_cooldown_s:
            return
        self._trigger_ts[key] = now
        ev = {"symbol": snap.symbol, "kind": kind, "reason": reason, "ltp": snap.tick.ltp,
              "at": now_ist().strftime("%H:%M:%S"), "mono": now, **(extra or {})}
        prev = self._events.get(snap.symbol)
        if prev and prev["kind"] in ("position_risk", "trend_flip") and kind not in ("position_risk", "trend_flip"):
            return                                   # never let an entry alert replace an exit alert
        self._events[snap.symbol] = ev
        self.events_today += 1
        self.last_event = {k: ev[k] for k in ("symbol", "kind", "reason", "ltp", "at")}
        self._record("alert", f"{kind} {snap.symbol} @ ₹{snap.tick.ltp:.2f}: {reason}")
        self._event_signal.set()

    # ── Real-time: alert worker (one Claude decision at a time) ───────────

    async def _event_worker(self) -> None:
        import time as _t
        while self.enabled:
            try:
                await self._event_signal.wait()
                self._event_signal.clear()
                while self._events and self.enabled:
                    # exits first, then oldest
                    sym, ev = min(self._events.items(), key=lambda kv: (
                        kv[1]["kind"] not in ("position_risk", "trend_flip"), kv[1]["mono"]))
                    del self._events[sym]
                    wait = settings.ai_agent_min_call_gap_s - (_t.monotonic() - self._last_call)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    await self._handle_event(ev)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._record("error", f"alert worker: {exc}")
                logger.exception("[ai] alert worker failed")

    async def _handle_event(self, ev: dict) -> Optional[str]:
        """Ask Claude to decide on one alert. Returns its reply, or None if dropped."""
        import time as _t
        sym = ev["symbol"]
        age = _t.monotonic() - ev["mono"]
        if age > settings.ai_agent_event_max_age_s:
            self._record("system", f"dropped stale alert {ev['kind']} {sym} ({age:.0f}s old)")
            return None
        if not settings.anthropic_api_key or self.cost_today_usd >= settings.ai_agent_max_daily_cost_usd:
            return None
        is_exit = ev["kind"] in ("position_risk", "trend_flip")
        if not is_exit and not self._can_consider_entry(sym):
            return None

        async with self._cycle_lock:
            self._last_call = _t.monotonic()
            self._event_ltp[sym] = ev["ltp"]
            t0 = _t.monotonic()
            try:
                messages = [{"role": "user", "content": self._alert_prompt(ev)}]
                reply = await self._run_agent(messages, purpose="alert",
                                              effort=settings.ai_agent_realtime_effort)
            finally:
                self._event_ltp.pop(sym, None)
            ms = int((_t.monotonic() - t0) * 1000)
            self._latencies.append(ms)
            self.decisions_today += 1
            self._record("decision", f"{ev['kind']} {sym} → {reply[:240]} ({ms / 1000:.1f}s)")
            return reply

    def _alert_prompt(self, ev: dict) -> str:
        sym = ev["symbol"]
        quote = tick_engine.all_latest().get(sym, {})
        buf = tick_engine._bufs_1min.get(sym)
        candles = [[c.ts.strftime("%H:%M"), round(c.open, 2), round(c.high, 2), round(c.low, 2),
                    round(c.close, 2), c.volume] for c in (buf.candles()[-10:] if buf else [])]
        allowed, blocked = self._trading_allowed()
        data = {
            "alert": {k: ev[k] for k in ev if k != "mono"},
            "quote": quote,
            "candles_1m_last10": {"columns": ["time", "open", "high", "low", "close", "volume"],
                                  "rows": candles},
            "your_positions": self._positions()["open"],
            "can_open_trades": allowed,
            "blocked_reason": blocked,
            "limits": self._limits(),
        }
        action = ("Decide now whether to close this position early or keep it (its stop is "
                  "already working)." if ev["kind"] in ("position_risk", "trend_flip")
                  else "Decide now: place_trade if the setup is clearly good, otherwise PASS.")
        return (f"REAL-TIME ALERT {ev['at']} IST — {ev['kind']} on {sym}: {ev['reason']}\n"
                f"{action}\n{json.dumps(data, default=str)}")

    def _note_stop_exits(self) -> None:
        """Log AI positions that the stop-loss / trailing-stop path closed on its own."""
        live = self._ai_entries()
        for oid in [o for o in self._open if o not in live]:
            sym = self._open.pop(oid)
            self._targets.pop(oid, None)
            self._record("close", f"[auto] {sym} exited by stop-loss / trailing stop")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ai_entries(self) -> dict:
        from agents.base_agent import _tsl_sl_orders
        return {oid: e for oid, e in _tsl_sl_orders.items() if e.get("agent") == AGENT}

    def _tick_age(self, symbol: str) -> int:
        tick, _ = tick_engine.latest(symbol)
        if not tick:
            return 10**6
        ts = tick.timestamp
        now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
        return max(0, int((now - ts).total_seconds()))

    def _mode_label(self) -> str:
        if settings.trading_mode == "LIVE":
            return "LIVE" if settings.ai_agent_allow_live else "LIVE-BLOCKED"
        return "PAPER-KITE" if settings.paper_data_source == "KITE" else "PAPER-SIM"

    def _trading_allowed(self) -> tuple[bool, str]:
        if settings.trading_mode == "LIVE" and not settings.ai_agent_allow_live:
            return False, "server is LIVE and AI_AGENT_ALLOW_LIVE is false — the AI may not place live orders"
        t = now_ist().time()
        if not is_market_open():
            return False, "market is closed"
        if not (ENTRY_START <= t <= ENTRY_END):
            return False, f"outside the AI entry window {ENTRY_START:%H:%M}-{ENTRY_END:%H:%M} IST"
        if risk_manager.is_trading_halted:
            return False, "trading halted for the day (daily loss limit)"
        return True, ""

    def _limits(self) -> dict:
        return {
            "max_trades_per_day":   settings.ai_agent_max_trades_per_day,
            "trades_left_today":    max(0, settings.ai_agent_max_trades_per_day - self.trades_today),
            "max_open_positions":   settings.ai_agent_max_open_positions,
            "max_position_value":   settings.ai_agent_max_position_value,
            "stop_loss_pct_range":  [settings.ai_agent_min_sl_pct, settings.ai_agent_max_sl_pct],
            "min_reward_to_risk":   1.5,
            "daily_cost_cap_usd":   settings.ai_agent_max_daily_cost_usd,
            "product":              "MIS (NSE equity intraday)",
        }

    def _roll_day(self) -> None:
        today = now_ist().date()
        if today != self._day:
            self._day = today
            self.trades_today = 0
            self.cycles_today = 0
            self.cost_today_usd = 0.0
            self.tokens_today = {k: 0 for k in self.tokens_today}

    def _record(self, kind: str, text: str) -> None:
        self.log.append({"t": now_ist().strftime("%H:%M:%S"), "kind": kind, "text": text})
        logger.info("[ai] {}: {}", kind, text[:300])

    async def _notify(self, text: str) -> None:
        try:
            from agents.base_agent import send_telegram
            await send_telegram(text)
        except Exception:
            pass


class ToolRejected(Exception):
    """A tool refused the action; the message is returned to Claude as an error result."""


ai_trader = AITrader()
