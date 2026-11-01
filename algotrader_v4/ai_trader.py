"""
ai_trader.py — Autonomous Claude trading agent for AlgoTrader Pro.

Claude runs an agentic tool-use loop over the bot's own state: it reads quotes,
indicators, regime and its own positions, and can trade
  * intraday NSE equity (MIS, long or short)
  * index options (buy only, intraday)
  * index futures (NFO, intraday, long or short — off unless AI_AGENT_FUTURES_ENABLED)
  * swing NSE equity (CNC delivery, long only, held for days, protected by a GTT stop at
    Zerodha and saved to data/ai_swing_positions.json so it survives restarts)
It answers a dashboard chat with the same tools.

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
  * at 15:00 IST every intraday AI position is closed by code, not by the model;
    swing positions are exited by code when their maximum holding period ends
  * the TSL engine and AI targets are driven from the tick feed, so stops work
    between decisions
  * real-time alerts: per-symbol cooldown, min gap between calls, stale alerts
    dropped, and an entry is rejected if price drifted since the alert
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from collections import deque
from datetime import date, datetime, time, timedelta
from pathlib import Path
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
from performance import performance
import market_filter
from options_resolver import (
    INDEX_QUOTE, OptionContract, estimate_premium, normalize_underlying, resolve, resolve_future,
    round_tick,
)

AGENT = "ai"

ENTRY_START = time(9, 20)    # no entries in the first 5 minutes
ENTRY_END   = time(14, 45)   # no new entries after this
FORCE_EXIT  = time(15, 0)    # code closes every AI position here
TICK_MAX_AGE_S = 120         # quotes older than this are not tradable
# Alert priority: protect positions first, then manage trails, then look for entries
_ALERT_RANK = {"position_risk": 0, "trend_flip": 0, "trail_review": 1}
R_MILESTONES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)   # profit in multiples of initial risk
OPT_STRATEGY = "ai_opt"      # order_guard / TSL key for AI option buys (premium-based trail)
OPT_ENTRY_START = time(9, 30)
OPT_ENTRY_END   = time(14, 30)
FUT_STRATEGY   = "ai_fut"      # AI index futures (intraday)
SWING_STRATEGY = "ai_swing"    # AI swing / delivery
SWING_OPT_STRATEGY = "ai_swing_opt"  # AI bearish swing: long put held for days (NRML)
SPREAD_STRATEGY = "ai_spread"  # AI credit spreads (option selling, defined risk, intraday)
MOMENTUM_STRATEGY = "momentum" # monthly momentum rotation (CNC), executed by code
MOMENTUM_HOLD_SESSIONS = 45    # time exit well past the next monthly rebalance
SWING_ENTRY_END = time(15, 15) # swing buys may use the last part of the session (daily close)
SWING_TIME_EXIT = time(15, 10) # code exits swing positions whose holding period ends today
SWING_REVIEW_S  = 1800         # swing trail reviews at most every 30 min (daily timeframe)
GTT_TRIGGER_OFFSET_PCT = 0.25  # GTT sits just beyond the software stop: the software stop exits
                               # first during market hours; the GTT covers gaps and downtime
SWING_FILE = Path(os.getenv("AI_SWING_FILE") or Path(__file__).resolve().parent / "data" / "ai_swing_positions.json")

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

SYSTEM_PROMPT = """You are the autonomous trader inside AlgoTrader Pro, an algorithmic \
trading system for NSE (India): intraday equities, index options, index futures and swing \
(delivery) equities. You trade with real or paper money on behalf of the \
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

Index options (when enabled)
- You may BUY index options (CE for a bullish view, PE for a bearish view) with \
place_option_trade. You never sell or write options. The most a trade can lose is the premium paid.
- Call get_option_chain first: it shows spot, the expiry that will be used, lot size, live \
premiums and spreads for strikes around the money, and IV rank.
- Prefer ATM or one strike OTM. Options lose value with time: only buy on a clear directional \
move in the index that you expect to continue soon, not on a slow drift. Stop and target are \
percentages of the premium.

Index futures (when enabled)
- place_future_trade opens an intraday NIFTY/BANKNIFTY future, long (BUY) or short (SELL). \
Futures are leveraged: a 1% index move is ~1% of the whole contract value (lakhs of rupees), \
so stops are tight (a fraction of a percent) and the code caps the rupee loss at the stop and \
the margin per trade. Use them for a clear, strong intraday index trend; prefer options when \
you want the loss limited to the premium. Closed by code at 15:00 like every intraday trade.

Swing trades (when enabled)
- place_swing_trade buys an NSE stock for delivery (CNC) and holds it for days (long only). \
Decide on the DAILY chart: call get_daily_chart first (EMA20/50/200, RSI, ATR, 20-day and \
52-week range). Buy uptrends at a breakout above the 20-day high on good volume, or on a \
pullback to the 20-day EMA that is holding — not extended moves far above the EMAs.
- Stop below daily structure (the pullback low / below EMA20 / ~1.5-2x daily ATR), target at \
least 2x the risk, and a maximum holding period in days. The stop is placed at Zerodha as a \
GTT so it also works overnight. Swing positions are NOT closed at 15:00; code exits them at \
the end of their holding period.
- A DAILY SWING SCAN message near the close lists shortlisted stocks and your swing positions. \
Most days you should buy nothing. Review held positions on the daily chart: trail the stop \
under new daily higher lows with adjust_stop, or close if the trend has broken.

Bearish swing (when enabled)
- Delivery shares cannot be shorted overnight, so a bearish swing is place_swing_put: BUY a put \
on an F&O stock or an index (NRML), with an expiry well after your planned exit, held for days. \
The most it can lose is the premium. Use it for a stock in a clear daily DOWNtrend breaking \
down or failing at the 20-day EMA from below (check get_daily_chart). Stop and target are % of \
the premium; the stop is a GTT at Zerodha. Puts lose value every day, so only on a clear trend.

Option selling (when enabled) — credit spreads only
- place_option_spread sells an option and buys a further-out-of-the-money one of the same type \
and expiry (BULLISH = bull put spread: sell a put below the index; BEARISH = bear call spread: \
sell a call above it). You earn the credit if the index stays on your side; the worst case is \
fixed at entry: (strike width − credit) × quantity. Naked option selling is not possible.
- Sell when you expect the index to stay above (bull put) or below (bear call) a level, not \
necessarily to move far; prefer short strikes 1-2 steps OTM. Intraday: closed by code at 15:00, \
at your take-profit (% of the credit kept) or at your stop (% of the credit lost).

Trailing stops (you manage them)
- A trail_review alert means a position is in profit (it lists the R multiple: profit divided by \
the initial risk). Decide whether to tighten the stop with adjust_stop to lock in profit, move the \
target, close, or leave it. Stops can only tighten, never loosen.
- Trail to structure, not to noise: below the recent swing low / VWAP / EMA9 for longs (mirror for \
shorts), roughly 1-1.5x ATR behind price. Around +1R moving to breakeven is usually sensible; give \
strong trends room, tighten quickly when momentum fades.

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
          "Open positions this AI holds (intraday, option, future, swing): entry, live price, "
          "unrealised P&L, stop, target, and for swing the days held and exit date.",
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
    _tool("get_daily_chart",
          "Daily (swing) view of an NSE stock: EMA20/50/200, RSI14, ATR14, 20-day and 52-week "
          "high/low, volume vs average, trend label and the last 30 daily candles.",
          {"symbol": {"type": "string", "description": "NSE trading symbol, e.g. RELIANCE"}},
          ["symbol"]),
    _tool("place_swing_trade",
          "BUY an NSE stock for delivery (CNC) to hold for days, with a GTT stop-loss at the broker "
          "that also works overnight. Long only. Rejected unless every hard limit and risk/SEBI check passes.",
          {
              "symbol":        {"type": "string", "description": "NSE trading symbol from the watchlist"},
              "quantity":      {"type": "integer", "description": "Shares; quantity x price within the swing per-trade limit"},
              "stop_loss":     {"type": "number", "description": "Stop price below the current price (daily structure)"},
              "target":        {"type": "number", "description": "Target price, at least the swing minimum reward:risk"},
              "max_hold_days": {"type": "integer", "description": "Trading days to hold at most; code exits after that"},
              "reasoning":     {"type": "string", "description": "One or two sentences: the daily setup and why now"},
          },
          ["symbol", "quantity", "stop_loss", "target", "max_hold_days", "reasoning"]),
    _tool("place_swing_put",
          "Bearish swing: BUY a put on an F&O stock or an index (NRML), expiring well after the planned "
          "exit, held for days with a GTT stop at the broker. Loss limited to the premium. Rejected "
          "unless every hard limit and risk/SEBI check passes.",
          {
              "underlying":    {"type": "string", "description": "F&O stock symbol (e.g. RELIANCE) or NIFTY / BANKNIFTY"},
              "strike_offset": {"type": "integer", "enum": [-1, 0, 1, 2],
                                "description": "0 = ATM, 1 = one strike out of the money (lower), -1 = one in the money"},
              "lots":          {"type": "integer", "description": "Number of lots"},
              "stop_loss_pct": {"type": "number", "description": "Stop as % below the premium paid"},
              "target_pct":    {"type": "number", "description": "Target as % above the premium; >= 1.5x the stop %"},
              "max_hold_days": {"type": "integer", "description": "Trading days to hold at most; code exits after that"},
              "reasoning":     {"type": "string", "description": "One or two sentences: the daily downtrend setup"},
          },
          ["underlying", "strike_offset", "lots", "stop_loss_pct", "target_pct", "max_hold_days", "reasoning"]),
    _tool("place_option_spread",
          "SELL a defined-risk credit spread on an index (intraday): BULLISH = sell a put and buy a "
          "lower put; BEARISH = sell a call and buy a higher call. Worst case = (width − credit) × qty. "
          "Rejected unless every hard limit and risk/SEBI check passes.",
          {
              "underlying":          {"type": "string", "enum": ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]},
              "view":                {"type": "string", "enum": ["BULLISH", "BEARISH"]},
              "short_strike_offset": {"type": "integer", "enum": [0, 1, 2, 3, 4],
                                      "description": "Strikes out of the money for the SOLD option (0 = ATM)"},
              "width_strikes":       {"type": "integer", "enum": [1, 2, 3, 4],
                                      "description": "How many strikes further out the protective BOUGHT option is"},
              "lots":                {"type": "integer", "description": "Number of lots"},
              "stop_loss_pct":       {"type": "number", "description": "Exit when the loss reaches this % of the credit (50-200)"},
              "target_pct":          {"type": "number", "description": "Take profit when this % of the credit is kept (30-90)"},
              "reasoning":           {"type": "string", "description": "One or two sentences: the level and why it should hold"},
          },
          ["underlying", "view", "short_strike_offset", "width_strikes", "lots", "stop_loss_pct",
           "target_pct", "reasoning"]),
    _tool("get_future_quote",
          "Nearest-month index future: contract, expiry, lot size, live price and spread, index spot "
          "and basis, estimated margin per lot, and the stop distance the rupee risk limit allows.",
          {"underlying": {"type": "string", "enum": ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]}},
          ["underlying"]),
    _tool("place_future_trade",
          "Open an intraday index FUTURE (nearest month), long (BUY) or short (SELL), with a "
          "stop-loss order. Leveraged. Rejected unless every hard limit and risk/SEBI check passes.",
          {
              "underlying": {"type": "string", "enum": ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]},
              "side":       {"type": "string", "enum": ["BUY", "SELL"]},
              "lots":       {"type": "integer", "description": "Number of lots"},
              "stop_loss":  {"type": "number", "description": "Stop as a FUTURE price (below for BUY, above for SELL)"},
              "target":     {"type": "number", "description": "Target as a FUTURE price, >= 1.5x the risk away"},
              "reasoning":  {"type": "string", "description": "One or two sentences: the index setup and why now"},
          },
          ["underlying", "side", "lots", "stop_loss", "target", "reasoning"]),
    _tool("get_option_chain",
          "Index option chain around the money for the nearest tradable expiry (same-day expiry "
          "excluded): spot, expiry, lot size, CE/PE premiums with bid/ask spread, IV rank.",
          {"underlying": {"type": "string", "enum": ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]}},
          ["underlying"]),
    _tool("place_option_trade",
          "BUY an index option (CE = bullish, PE = bearish) with a protective stop-loss order. "
          "Intraday; never sells options. Rejected unless every hard limit and risk/SEBI check passes.",
          {
              "underlying":    {"type": "string", "enum": ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]},
              "option_type":   {"type": "string", "enum": ["CE", "PE"]},
              "strike_offset": {"type": "integer", "enum": [-2, -1, 0, 1, 2],
                                "description": "0 = ATM, 1 = one strike out of the money, -1 = one in the money"},
              "lots":          {"type": "integer", "description": "Number of lots"},
              "stop_loss_pct": {"type": "number", "description": "Stop as % below the premium paid"},
              "target_pct":    {"type": "number", "description": "Target as % above the premium; >= 1.5x the stop %"},
              "reasoning":     {"type": "string", "description": "One or two sentences: the index setup and why now"},
          },
          ["underlying", "option_type", "strike_offset", "lots", "stop_loss_pct", "target_pct", "reasoning"]),
    _tool("adjust_stop",
          "Tighten the stop-loss (and optionally move the target) of an open AI position. The stop "
          "can only move in the position's favour and must stay a minimum distance from price; the "
          "exchange stop order is updated.",
          {
              "symbol":     {"type": "string", "description": "The position's tradingsymbol"},
              "new_stop":   {"type": "number"},
              "new_target": {"type": "number", "description": "New target price, or 0 to keep the current one"},
              "reasoning":  {"type": "string"},
          },
          ["symbol", "new_stop", "new_target", "reasoning"]),
    _tool("close_position",
          "Close an open AI position at market (cancels its stop-loss order). For a credit spread "
          "pass its spread_id or the index name.",
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
        # AI-managed trailing: entry order id → {risk, milestone, last_review, best_at_review}
        self._trail: dict[str, dict] = {}
        self._ai_errors = 0                            # consecutive Claude API failures
        # Swing (delivery) positions: entry order id → {opened, hold_until, reason, initial_risk}
        self._swing_meta: dict[str, dict] = {}
        self._swing_scan_day: Optional[date] = None
        self._swing_scan_task: Optional[asyncio.Task] = None
        self._gtt_checked: dict[str, float] = {}       # entry order id → monotonic time of last GTT check
        # Credit spreads (two legs — tracked here, not in the single-symbol trailing engine)
        self._spreads: dict[str, dict] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, watchlist: Optional[list[dict]] = None) -> dict:
        """Enable autonomous cycles. Subscribes the watchlist so quotes stream."""
        if watchlist:
            tick_engine.subscribe(watchlist)
        if not tick_engine.symbols():
            from symbol_scanner import NIFTY_50
            tick_engine.subscribe([{"symbol": s, "exchange": "NSE"} for s in NIFTY_50[:20]])
        if settings.ai_agent_options_enabled:
            tick_engine.subscribe([{"symbol": INDEX_QUOTE[u], "exchange": "NSE"}
                                   for u in self._option_underlyings()])
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
                self._ai_errors += 1
                return "Claude API rate limit hit — try again shortly."
            except anthropic.AuthenticationError:
                self._ai_errors += 1
                return "Claude API key was rejected."
            except anthropic.APIStatusError as exc:
                self._ai_errors += 1
                logger.error("[ai] API error {}: {}", exc.status_code, exc.message)
                return f"Claude API error ({exc.status_code})."
            except anthropic.APIConnectionError:
                self._ai_errors += 1
                return "Could not reach the Claude API."
            self._ai_errors = 0
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
        if name == "get_daily_chart":
            return await self._daily_chart(str(args.get("symbol", "")).upper().strip())
        if name == "place_swing_trade":
            async with self._action_lock:
                return await self._place_swing_trade(args)
        if name == "place_swing_put":
            async with self._action_lock:
                return await self._place_swing_put(args)
        if name == "place_option_spread":
            async with self._action_lock:
                return await self._place_option_spread(args)
        if name == "get_future_quote":
            return self._future_quote(str(args.get("underlying", "")))
        if name == "place_future_trade":
            async with self._action_lock:
                return await self._place_future_trade(args)
        if name == "get_option_chain":
            return self._option_chain(str(args.get("underlying", "")))
        if name == "place_option_trade":
            async with self._action_lock:
                return await self._place_option_trade(args)
        if name == "adjust_stop":
            async with self._action_lock:
                return await self._adjust_stop(args)
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
            "entry_window":   (f"{ENTRY_START:%H:%M}-{ENTRY_END:%H:%M} IST (swing buys until {SWING_ENTRY_END:%H:%M}); "
                               f"intraday AI positions force-closed at {FORCE_EXIT:%H:%M}, swing positions held"),
            "can_open_trades": allowed,
            "blocked_reason": why,
            "regime":         regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN",
            "risk":           risk_manager.status(),
            "sebi_state":     sebi_compliance.status().get("state"),
            "limits":         self._limits(),
            "trend_filter":   market_filter.state(),
            "pruned_strategies": {k: v["reason"] for k, v in performance.pruned.items()},
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
            st = self._trail.get(oid)
            r = None
            if st and st["risk"] and ltp and entry:
                r = round(((ltp - entry) if e["side"] == "BUY" else (entry - ltp)) / st["risk"], 2)
            row = {"symbol": e["symbol"], "kind": self._kind(e), "product": e.get("product", "MIS"),
                   "side": e["side"], "quantity": qty,
                   "entry": entry, "ltp": ltp, "unrealised_pnl": pnl,
                   "stop_loss": pos.current_sl if pos else None,
                   "target": self._targets.get(oid),
                   "best_price": pos.best_price if pos else None,
                   "r_multiple": r,
                   "trailing": ("AI" if pos and pos.external_trail else "standard")}
            meta = self._swing_meta.get(oid)
            if meta:
                row.update(opened=meta["opened"], hold_until=meta["hold_until"],
                           days_held=self._sessions_between(date.fromisoformat(meta["opened"]), now_ist().date()))
            out.append(row)
        for sid, sp in self._spreads.items():
            out.append({"symbol": sid, "kind": "spread", "product": "MIS", "side": "SELL",
                        "structure": "bull put spread" if sp["view"] == "BULLISH" else "bear call spread",
                        "short": sp["short"]["symbol"], "long": sp["long"]["symbol"], "quantity": sp["qty"],
                        "credit": sp["credit"], "current_cost_to_close": sp["value"],
                        "unrealised_pnl": round((sp["credit"] - sp["value"]) * sp["qty"], 2),
                        "stop_when_cost": sp["stop_value"], "take_profit_when_cost": sp["target_value"],
                        "max_loss": sp["max_loss"]})
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
        ok, why_pruned = performance.is_allowed(AGENT)
        if not ok:
            reject(why_pruned)
        if sym in INDEX_QUOTE.values() or tick_engine._exchange.get(sym, "NSE") != "NSE":
            reject(f"{sym} is not an NSE equity — use place_option_trade for index options")
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
        max_value = settings.ai_agent_max_position_value * market_filter.size_factor()
        if qty * ltp > max_value:
            reject(f"value ₹{qty * ltp:,.0f} exceeds per-trade limit ₹{max_value:,.0f}"
                   f"{' (halved: NIFTY below its 200-day EMA)' if max_value < settings.ai_agent_max_position_value else ''}"
                   f" (max qty {int(max_value // ltp)})")
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
        if self._intraday_count() >= settings.ai_agent_max_open_positions:
            reject(f"max open intraday AI positions ({settings.ai_agent_max_open_positions}) reached")
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
            "symbol": sym, "guard_symbol": sym, "agent": AGENT, "side": side, "kind": "equity",
        }
        pos = trailing_sl_engine.register(
            symbol=sym, strategy=AGENT, side=side, entry_price=ltp,
            quantity=qty, order_id=order_id,
        )
        pos.current_sl = sl                     # software stop = the AI's stop, not the config default
        self._start_trail(order_id, pos)
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

    # ── Index options (buy only) ──────────────────────────────────────────

    def _index_spot(self, und: str) -> tuple[str, float]:
        idx = INDEX_QUOTE[und]
        tick, _ = tick_engine.latest(idx)
        if not tick or tick.ltp <= 0:
            raise ToolRejected(f"{idx} is not streaming — start the AI trader with options enabled")
        if self._tick_age(idx) > TICK_MAX_AGE_S:
            raise ToolRejected(f"{idx} quote is stale ({self._tick_age(idx)}s old)")
        return idx, float(tick.ltp)

    def _option_quotes(self, contracts: list[OptionContract], spot: float) -> dict[str, dict]:
        """Live bid/ask/LTP from Kite, or a rough estimate for the paper simulator."""
        out: dict[str, dict] = {}
        live = [c for c in contracts if c.live]
        if live and settings.uses_kite_market_data:
            raw = kite_client.quote_kite([f"NFO:{c.tradingsymbol}" for c in live])
            for c in live:
                q = raw.get(f"NFO:{c.tradingsymbol}") or {}
                depth = q.get("depth") or {}
                buy, sell = depth.get("buy") or [{}], depth.get("sell") or [{}]
                out[c.tradingsymbol] = {"ltp": float(q.get("last_price") or 0),
                                        "bid": float(buy[0].get("price") or 0),
                                        "ask": float(sell[0].get("price") or 0), "estimated": False}
        for c in contracts:
            if c.tradingsymbol not in out:
                days = max((c.expiry - now_ist().date()).days, 1) if c.expiry else 5.0
                px = estimate_premium(spot, c.strike, c.option_type, days=days)
                out[c.tradingsymbol] = {"ltp": px, "bid": px, "ask": px, "estimated": True}
        return out

    def _iv_rank(self, und: str) -> Optional[float]:
        try:
            from options_intelligence import get_cached
            for key in (und, INDEX_QUOTE[und]):
                ctx = get_cached(key)
                if ctx and ctx.get("iv_rank") is not None:
                    return float(ctx["iv_rank"])
        except Exception:
            pass
        return None

    def _option_chain(self, underlying: str) -> dict:
        und = normalize_underlying(underlying)
        if not settings.ai_agent_options_enabled:
            raise ToolRejected("option trading is disabled (AI_AGENT_OPTIONS_ENABLED=false)")
        if und not in self._option_underlyings():
            raise ToolRejected(f"allowed underlyings: {self._option_underlyings()}")
        _, spot = self._index_spot(und)
        today = now_ist().date()
        rows, contracts = [], []
        for off in (-2, -1, 0, 1, 2):
            pair = {}
            for ot in ("CE", "PE"):
                c = resolve(und, spot, ot, off, today)
                if c:
                    pair[ot] = c
                    contracts.append(c)
            if pair:
                rows.append((off, pair))
        if not contracts:
            raise ToolRejected(f"no tradable {und} option contracts found")
        quotes = self._option_quotes(contracts, spot)

        def view(c: OptionContract) -> dict:
            q = quotes[c.tradingsymbol]
            spread = (q["ask"] - q["bid"]) / q["ltp"] * 100 if q["ltp"] and q["ask"] else None
            return {"symbol": c.tradingsymbol, "strike": c.strike, "ltp": q["ltp"], "bid": q["bid"],
                    "ask": q["ask"], "spread_pct": round(spread, 2) if spread is not None else None}
        first = contracts[0]
        return {
            "underlying": und, "spot": spot,
            "expiry": first.expiry.isoformat() if first.expiry else "simulated",
            "lot_size": first.lot_size, "iv_rank": self._iv_rank(und),
            "premiums_estimated": any(q["estimated"] for q in quotes.values()),
            "strikes": [{"offset": off, **{ot: view(c) for ot, c in pair.items()}} for off, pair in rows],
            "note": "offset is per option type: +1 = one strike OTM (higher for CE, lower for PE)",
        }

    async def _place_option_trade(self, a: dict) -> dict:
        from agents.base_agent import (
            place_sl_with_retry, flatten_unprotected, _setup_tsl_callbacks, _tsl_sl_orders,
        )
        from sebi_compliance import sebi_compliance
        from market_regime import regime_detector

        und  = normalize_underlying(str(a.get("underlying", ""))) or str(a.get("underlying", ""))
        opt  = str(a.get("option_type", "")).upper()
        off  = int(a.get("strike_offset") or 0)
        lots = int(a.get("lots") or 0)
        slp  = float(a.get("stop_loss_pct") or 0)
        tgp  = float(a.get("target_pct") or 0)
        why  = str(a.get("reasoning", ""))[:300]

        def reject(msg: str):
            self._record("rejected", f"BUY {lots} lot {und} {opt} {off:+d}: {msg}")
            raise ToolRejected(msg)

        if not settings.ai_agent_options_enabled:
            reject("option trading is disabled (AI_AGENT_OPTIONS_ENABLED=false)")
        allowed, blocked = self._trading_allowed()
        if not allowed:
            reject(blocked)
        ok, why_pruned = performance.is_allowed(OPT_STRATEGY)
        if not ok:
            reject(why_pruned)
        if not (OPT_ENTRY_START <= now_ist().time() <= OPT_ENTRY_END):
            reject(f"outside the option entry window {OPT_ENTRY_START:%H:%M}-{OPT_ENTRY_END:%H:%M} IST")
        if und not in self._option_underlyings():
            reject(f"underlying must be one of {self._option_underlyings()}")
        if opt not in ("CE", "PE"):
            reject("option_type must be CE or PE")
        if off not in (-2, -1, 0, 1, 2):
            reject("strike_offset must be between -2 and 2")
        if not 1 <= lots <= settings.ai_agent_max_option_lots:
            reject(f"lots must be 1-{settings.ai_agent_max_option_lots}")
        try:
            idx, spot = self._index_spot(und)
        except ToolRejected as e:
            reject(str(e))
        ref = self._event_ltp.get(idx)
        if ref and abs(spot - ref) / ref * 100 > settings.ai_agent_max_drift_pct:
            reject(f"{idx} moved {abs(spot - ref) / ref * 100:.2f}% since the alert; the setup is stale")
        ivr = self._iv_rank(und)
        if ivr is not None and ivr > settings.ai_agent_option_max_iv_rank:
            reject(f"IV rank {ivr:.0f} is above {settings.ai_agent_option_max_iv_rank:.0f} — premiums too expensive")
        if not (settings.ai_agent_option_min_sl_pct <= slp <= settings.ai_agent_option_max_sl_pct):
            reject(f"stop must be {settings.ai_agent_option_min_sl_pct}-{settings.ai_agent_option_max_sl_pct}% "
                   "of the premium")
        if tgp < 1.5 * slp:
            reject(f"target {tgp}% is under 1.5x the {slp}% stop")
        if self.trades_today >= settings.ai_agent_max_trades_per_day:
            reject(f"daily AI trade limit ({settings.ai_agent_max_trades_per_day}) reached")
        if self._intraday_count() >= settings.ai_agent_max_open_positions:
            reject(f"max open intraday AI positions ({settings.ai_agent_max_open_positions}) reached")
        if order_guard.is_symbol_active_anywhere(und):
            reject(f"{und} already has an open option or future position")

        c = resolve(und, spot, opt, off, now_ist().date())
        if not c or c.lot_size <= 0:
            reject(f"no tradable {und} {opt} contract (expiry/strike not found in the instrument list)")
        q = self._option_quotes([c], spot)[c.tradingsymbol]
        ltp, bid, ask = q["ltp"], q["bid"], q["ask"]
        if ltp < settings.ai_agent_option_min_premium:
            reject(f"premium ₹{ltp:.2f} is below ₹{settings.ai_agent_option_min_premium:.0f}")
        if not q["estimated"]:
            if ask <= 0 or bid <= 0:
                reject(f"{c.tradingsymbol} has no two-sided quote")
            spread = (ask - bid) / ltp * 100
            if spread > settings.ai_agent_option_max_spread_pct:
                reject(f"{c.tradingsymbol} spread {spread:.1f}% exceeds {settings.ai_agent_option_max_spread_pct}%")
        entry_px = round_tick((ask or ltp) * 1.01)          # marketable limit
        qty = lots * c.lot_size
        if entry_px * qty > settings.ai_agent_max_option_premium:
            per_lot = entry_px * c.lot_size
            reject(f"premium ₹{entry_px * qty:,.0f} exceeds ₹{settings.ai_agent_max_option_premium:,.0f} "
                   f"(₹{per_lot:,.0f}/lot → max {int(settings.ai_agent_max_option_premium // per_lot)} lots)")
        ok, msg = order_guard.can_place(c.tradingsymbol, OPT_STRATEGY, "BUY")
        if not ok:
            reject(msg)
        ok, msg = risk_manager.check_before_order(c.tradingsymbol, qty, entry_px, "BUY")
        if not ok:
            reject(msg)
        ok, _algo, msg = sebi_compliance.pre_order_check(
            strategy=AGENT, symbol=c.tradingsymbol, exchange="NFO", transaction_type="BUY", quantity=qty,
            order_type="LIMIT", price_at_signal=entry_px, signal_source="ai_trader",
            regime=regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN",
        )
        if not ok:
            reject(msg)

        tick_engine.subscribe([{"symbol": c.tradingsymbol, "exchange": "NFO", "seed_price": ltp}])
        order_id = kite_client.place_order(
            tradingsymbol=c.tradingsymbol, exchange="NFO", transaction_type="BUY", quantity=qty,
            order_type="LIMIT", price=entry_px, product="MIS", tag="Agent-ai-opt",
        )
        sebi_compliance.record_order_id(AGENT, c.tradingsymbol, order_id)

        sl_px  = round_tick(entry_px * (1 - slp / 100))
        tgt_px = round_tick(entry_px * (1 + tgp / 100))
        sl_oid = await place_sl_with_retry(AGENT, c.tradingsymbol, "NFO", "BUY", qty, "MIS", sl_px,
                                           order_type="SL", limit_buffer_pct=5.0)
        if sl_oid is None and await flatten_unprotected(
                AGENT, c.tradingsymbol, "NFO", "BUY", qty, "MIS",
                exit_limit_price=round_tick(max(bid or ltp, 0.1) * 0.97)):
            reject("stop-loss order was rejected by the broker; entry was closed")

        order_guard.register_order(und, OPT_STRATEGY, "BUY", order_id)
        risk_manager.position_opened()
        _setup_tsl_callbacks()
        _tsl_sl_orders[order_id] = {
            "sl_order_id": sl_oid, "product": "MIS", "exchange": "NFO",
            "symbol": c.tradingsymbol, "guard_symbol": und, "agent": AGENT, "side": "BUY",
            "sl_type": "SL", "sl_buffer_pct": 5.0, "exit_limit_buffer_pct": 3.0, "kind": "option",
        }
        pos = trailing_sl_engine.register(
            symbol=c.tradingsymbol, strategy=OPT_STRATEGY, side="BUY", entry_price=entry_px,
            quantity=qty, order_id=order_id,
        )
        pos.current_sl = sl_px
        self._start_trail(order_id, pos)
        self._targets[order_id] = tgt_px
        self._open[order_id] = c.tradingsymbol
        self.trades_today += 1
        self._ensure_tick_monitor()

        exp = c.expiry.isoformat() if c.expiry else "sim"
        text = (f"BUY {lots} lot ({qty}) {c.tradingsymbol} [{exp}] @ ₹{entry_px:.2f} "
                f"SL ₹{sl_px:.2f} TGT ₹{tgt_px:.2f} (cost ₹{entry_px * qty:,.0f}) — {why}")
        self._record("trade", text)
        await self._notify(f"🧠 <b>[AI]</b> {text}")
        return {"status": "placed", "order_id": order_id, "symbol": c.tradingsymbol,
                "expiry": exp, "quantity": qty, "limit_price": entry_px, "stop_loss": sl_px,
                "target": tgt_px, "max_loss": round((entry_px - sl_px) * qty, 2),
                "premium_paid": round(entry_px * qty, 2), "mode": self._mode_label()}

    # ── Swing (delivery, long only) ───────────────────────────────────────

    async def _daily_chart(self, symbol: str) -> dict:
        from daily_data import daily_view
        if normalize_underlying(symbol):
            symbol = INDEX_QUOTE[normalize_underlying(symbol)]
        if not symbol or tick_engine._exchange.get(symbol, "NSE") != "NSE":
            raise ToolRejected(f"{symbol} is not an NSE stock or index")
        v = await asyncio.to_thread(daily_view, symbol)
        if not v:
            raise ToolRejected(f"no daily history for {symbol} (is it streaming?)")
        return v

    async def _place_swing_trade(self, a: dict, _strategy: str = SWING_STRATEGY) -> dict:
        """_strategy=MOMENTUM_STRATEGY: same CNC + GTT plumbing for the monthly rotation (its own
        slot limit, hold period and stop width; counted as a momentum trade, not a swing one)."""
        from agents.base_agent import flatten_unprotected, _setup_tsl_callbacks, _tsl_sl_orders, sl_limit_price
        from sebi_compliance import sebi_compliance
        from market_regime import regime_detector

        sym  = str(a.get("symbol", "")).upper().strip()
        qty  = int(a.get("quantity") or 0)
        sl   = float(a.get("stop_loss") or 0)
        tgt  = float(a.get("target") or 0)
        hold = int(a.get("max_hold_days") or 0)
        why  = str(a.get("reasoning", ""))[:300]

        def reject(msg: str):
            self._record("rejected", f"SWING BUY {qty} {sym}: {msg}")
            raise ToolRejected(msg)

        momentum = _strategy == MOMENTUM_STRATEGY
        if not momentum and not settings.ai_agent_swing_enabled:
            reject("swing trading is disabled (AI_AGENT_SWING_ENABLED=false)")
        if momentum and not settings.momentum_enabled:
            reject("momentum rotation is disabled (MOMENTUM_ENABLED=false)")
        allowed, blocked = self._trading_allowed(swing=True)
        if not allowed:
            reject(blocked)
        ok, why_pruned = performance.is_allowed(_strategy)
        if not ok:
            reject(why_pruned)
        if sym in INDEX_QUOTE.values() or tick_engine._exchange.get(sym, "NSE") != "NSE":
            reject(f"{sym} is not an NSE equity")
        tick, _ = tick_engine.latest(sym)
        if not tick or tick.ltp <= 0:
            reject(f"{sym} is not streaming")
        if self._tick_age(sym) > TICK_MAX_AGE_S:
            reject(f"{sym} quote is stale ({self._tick_age(sym)}s old)")
        ltp = float(tick.ltp)
        if qty <= 0:
            reject("quantity must be positive")
        if market_filter.blocks_new_longs():
            reject("NIFTY 50 is below its 200-day EMA — no new swing buys while the trend filter is RISK_OFF")
        max_sl_pct = max(settings.ai_agent_swing_max_sl_pct, settings.momentum_stop_pct) if momentum \
            else settings.ai_agent_swing_max_sl_pct
        max_hold = MOMENTUM_HOLD_SESSIONS if momentum else settings.ai_agent_swing_max_hold_days
        if qty * ltp > settings.ai_agent_max_swing_value:
            reject(f"value ₹{qty * ltp:,.0f} exceeds the swing per-trade limit "
                   f"₹{settings.ai_agent_max_swing_value:,.0f} (max qty {int(settings.ai_agent_max_swing_value // ltp)})")
        risk_dist = ltp - sl
        if risk_dist <= 0:
            reject(f"stop_loss {sl} must be below price {ltp}")
        sl_pct = risk_dist / ltp * 100
        if not (settings.ai_agent_swing_min_sl_pct <= sl_pct <= max_sl_pct):
            reject(f"stop is {sl_pct:.2f}% away; allowed {settings.ai_agent_swing_min_sl_pct}-{max_sl_pct}%")
        if tgt - ltp < settings.ai_agent_swing_min_rr * risk_dist:
            reject(f"target {tgt} gives {(tgt - ltp) / risk_dist:.2f}x risk; swing needs >= "
                   f"{settings.ai_agent_swing_min_rr}x")
        if not 1 <= hold <= max_hold:
            reject(f"max_hold_days must be 1-{max_hold}")
        if momentum:
            if len(self._momentum_entries()) >= settings.momentum_top_n:
                reject(f"momentum rotation already holds {settings.momentum_top_n} names")
        else:
            if self.trades_today >= settings.ai_agent_max_trades_per_day:
                reject(f"daily AI trade limit ({settings.ai_agent_max_trades_per_day}) reached")
            if len(self._swing_entries()) >= settings.ai_agent_max_swing_positions:
                reject(f"max swing positions ({settings.ai_agent_max_swing_positions}) reached")
        ok, msg = order_guard.can_place(sym, _strategy, "BUY")
        if not ok:
            reject(msg)
        if order_guard.is_symbol_active_anywhere(sym):
            reject(f"{sym} already has an open position")
        ok, msg = risk_manager.check_before_order(sym, qty, ltp, "BUY")
        if not ok:
            reject(msg)
        ok, _algo, msg = sebi_compliance.pre_order_check(
            strategy=AGENT, symbol=sym, exchange="NSE", transaction_type="BUY", quantity=qty,
            order_type="MARKET", price_at_signal=ltp, signal_source="ai_trader",
            regime=regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN",
        )
        if not ok:
            reject(msg)

        order_id = kite_client.place_order(
            tradingsymbol=sym, exchange="NSE", transaction_type="BUY", quantity=qty,
            order_type="MARKET", product="CNC", tag="Agent-ai-mom" if momentum else "Agent-ai-swing",
        )
        sebi_compliance.record_order_id(AGENT, sym, order_id)

        sl = round_tick(sl)
        buf = settings.ai_agent_swing_gtt_buffer_pct
        gtt_trigger = round_tick(sl * (1 - GTT_TRIGGER_OFFSET_PCT / 100))
        gtt_id = None
        for attempt in range(3):
            try:
                gtt_id = kite_client.place_gtt_stop(sym, "NSE", qty, gtt_trigger,
                                                    sl_limit_price(gtt_trigger, "BUY", buf), ltp)
                break
            except Exception as exc:
                logger.error("[ai] GTT for {} failed (attempt {}/3): {}", sym, attempt + 1, exc)
                await asyncio.sleep(0.2)
        if gtt_id is None and await flatten_unprotected(AGENT, sym, "NSE", "BUY", qty, "CNC"):
            reject("the GTT stop-loss was rejected by the broker; the purchase was sold again")

        today = now_ist().date()
        order_guard.register_order(sym, _strategy, "BUY", order_id)
        risk_manager.position_opened()
        _setup_tsl_callbacks()
        _tsl_sl_orders[order_id] = {
            "sl_order_id": None, "gtt_id": gtt_id, "gtt_buffer_pct": buf,
            "gtt_trigger_offset_pct": GTT_TRIGGER_OFFSET_PCT, "product": "CNC", "exchange": "NSE",
            "symbol": sym, "guard_symbol": sym, "agent": AGENT, "side": "BUY", "kind": "swing",
        }
        pos = trailing_sl_engine.register(symbol=sym, strategy=_strategy, side="BUY",
                                          entry_price=ltp, quantity=qty, order_id=order_id)
        pos.current_sl = sl
        self._start_trail(order_id, pos)
        if momentum:
            pos.external_trail = False          # mechanical wide trail; Claude is not involved
        self._targets[order_id] = round_tick(tgt)
        self._open[order_id] = sym
        hold_until = self._add_sessions(today, hold)
        self._swing_meta[order_id] = {"opened": today.isoformat(), "hold_until": hold_until.isoformat(),
                                      "reason": why, "initial_risk": risk_dist}
        self._save_swing()
        if not momentum:
            self.trades_today += 1
        self._ensure_tick_monitor()

        text = (f"{'MOMENTUM' if momentum else 'SWING'} BUY {qty} {sym} @ ~₹{ltp:.2f} SL ₹{sl:.2f} (GTT) TGT ₹{tgt:.2f}, "
                f"hold until {hold_until:%d %b} — {why}")
        self._record("trade", text)
        await self._notify(f"🧠 <b>[AI]</b> {text}")
        return {"status": "placed", "order_id": order_id, "symbol": sym, "product": "CNC",
                "quantity": qty, "approx_price": ltp, "stop_loss": sl, "gtt_id": gtt_id,
                "target": round_tick(tgt), "hold_until": hold_until.isoformat(), "mode": self._mode_label()}

    async def _swing_housekeeping(self, held: dict, now_t: time) -> None:
        """Time exit at the end of the holding period; in LIVE, notice a GTT that already sold."""
        import time as _t
        today = now_ist().date()
        for oid, e in held.items():
            meta = self._swing_meta.get(oid)
            if (meta and now_t >= SWING_TIME_EXIT and today >= date.fromisoformat(meta["hold_until"])
                    and oid in self._ai_entries()):
                async with self._action_lock:
                    try:
                        await self._close_position(e["symbol"], "maximum holding period reached", by_code=True)
                    except ToolRejected:
                        pass
                continue
            if settings.trading_mode == "LIVE" and e.get("gtt_id"):
                last = self._gtt_checked.get(oid, 0.0)
                if _t.monotonic() - last >= 300:
                    self._gtt_checked[oid] = _t.monotonic()
                    status = await asyncio.to_thread(kite_client.gtt_status, e["gtt_id"])
                    if status == "triggered":
                        self._forget_position(oid, "GTT stop triggered at Zerodha")

    def _forget_position(self, oid: str, note: str) -> None:
        """Drop a position that the broker already closed (no exit order is sent)."""
        from agents.base_agent import _tsl_sl_orders
        e = _tsl_sl_orders.pop(oid, None)
        pos = trailing_sl_engine._positions.get(oid)
        if not e:
            return
        tick, _ = tick_engine.latest(e["symbol"])
        px = (pos.current_sl if pos else 0.0) or (tick.ltp if tick else 0.0)
        pnl = (px - pos.entry_price) * pos.quantity if pos else 0.0
        order_guard.release_order(e.get("guard_symbol", e["symbol"]), pos.strategy if pos else SWING_STRATEGY,
                                  e["side"], pnl)
        risk_manager.record_trade(pnl)
        risk_manager.position_closed()
        trailing_sl_engine.deregister(oid)
        if pos:
            from performance import performance
            performance.record_trade(pos.strategy, e["symbol"], "BUY", pos.quantity, pos.entry_price, pnl,
                                     e.get("exchange", "NSE"), e.get("product", "CNC"), note=note)
        self._open.pop(oid, None)
        self._targets.pop(oid, None)
        self._trail.pop(oid, None)
        if self._swing_meta.pop(oid, None) is not None:
            self._save_swing()
        self._record("close", f"[auto] {e['symbol']}: {note} (≈₹{pnl:,.0f})")

    def _maybe_swing_scan(self, now_t: time) -> None:
        if not (self.enabled and settings.ai_agent_swing_enabled and settings.anthropic_api_key):
            return
        today = now_ist().date()
        try:
            h, m = (int(x) for x in settings.ai_agent_swing_scan_time.split(":"))
        except ValueError:
            return
        if self._swing_scan_day == today or not (time(h, m) <= now_t <= SWING_ENTRY_END):
            return
        if self._swing_scan_task and not self._swing_scan_task.done():
            return
        self._swing_scan_day = today
        self._swing_scan_task = asyncio.create_task(self.run_swing_scan())

    async def run_swing_scan(self) -> Optional[str]:
        """Once a day near the close: shortlist swing setups in code, then let Claude decide,
        and review held swing positions on the daily chart."""
        from daily_data import swing_candidates
        if self.cost_today_usd >= settings.ai_agent_max_daily_cost_usd:
            return None
        held = {e["symbol"] for e in self._ai_entries().values()}
        universe = [s for s in tick_engine.symbols()
                    if tick_engine._exchange.get(s, "NSE") == "NSE" and s not in INDEX_QUOTE.values()
                    and s not in held]
        try:
            cands = await asyncio.to_thread(swing_candidates, universe)
        except Exception as exc:
            self._record("error", f"swing scan: {exc}")
            cands = []
        swing_pos = [p for p in self._positions()["open"] if p["kind"] == "swing"]
        if not cands and not swing_pos:
            self._record("cycle", "swing scan: no candidates, no swing positions")
            return None
        allowed, blocked = self._trading_allowed(swing=True)
        data = {"candidates": cands, "your_swing_positions": swing_pos,
                "can_open_trades": allowed, "blocked_reason": blocked, "limits": self._limits()["swing"]}
        prompt = (f"DAILY SWING SCAN {now_ist():%H:%M} IST. Code shortlisted these stocks: BULLISH ones "
                  "(uptrend, breakout or pullback to EMA20) can be bought with place_swing_trade; BEARISH "
                  "ones (downtrend, breakdown or bounce into EMA20) can be traded with place_swing_put if "
                  "bearish swing is enabled. For any you consider, call get_daily_chart first and trade "
                  "only if the daily setup is clearly good. Review each held swing "
                  "position on its daily chart: trail with adjust_stop, close, or hold. Buying nothing "
                  f"is normal.\n{json.dumps(data, default=str)}")
        async with self._cycle_lock:
            reply = await self._run_agent([{"role": "user", "content": prompt}], purpose="swing_scan")
        self._record("cycle", f"swing scan → {reply[:240]}")
        return reply

    def _save_swing(self) -> None:
        """Persist open swing and momentum positions (atomic write) so a restart can pick them up."""
        rows = []
        for oid, e in self._overnight_entries().items():
            pos = trailing_sl_engine._positions.get(oid)
            meta = self._swing_meta.get(oid)
            if not pos or not meta:
                continue
            rows.append({"order_id": oid, "symbol": e["symbol"], "quantity": pos.quantity,
                         "entry": pos.entry_price, "stop": pos.current_sl, "best": pos.best_price,
                         "target": self._targets.get(oid), "gtt_id": e.get("gtt_id"),
                         "gtt_buffer_pct": e.get("gtt_buffer_pct", settings.ai_agent_swing_gtt_buffer_pct),
                         "exchange": e.get("exchange", "NSE"), "product": e.get("product", "CNC"),
                         "guard_symbol": e.get("guard_symbol", e["symbol"]), "strategy": pos.strategy,
                         "exit_limit_buffer_pct": e.get("exit_limit_buffer_pct"),
                         **meta})
        try:
            SWING_FILE.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=SWING_FILE.parent, prefix=".swing.")
            with os.fdopen(fd, "w") as f:
                json.dump({"saved_at": now_ist().isoformat(timespec="seconds"), "positions": rows}, f, indent=1)
            os.replace(tmp, SWING_FILE)
        except Exception as exc:
            logger.error("[ai] could not save swing positions: {}", exc)

    def restore_swing(self) -> list[str]:
        """Re-track swing positions saved by an earlier run (startup). Positions whose GTT
        already triggered at Zerodha are dropped instead."""
        from agents.base_agent import _setup_tsl_callbacks, _tsl_sl_orders
        try:
            rows = json.loads(SWING_FILE.read_text()).get("positions", [])
        except FileNotFoundError:
            return []
        except Exception as exc:
            logger.error("[ai] swing file unreadable: {}", exc)
            return []
        restored = []
        for r in rows:
            oid = r["order_id"]
            if oid in _tsl_sl_orders:
                continue
            if settings.trading_mode == "LIVE" and r.get("gtt_id") and kite_client.has_session():
                if kite_client.gtt_status(r["gtt_id"]) == "triggered":
                    self._record("close", f"[auto] {r['symbol']}: GTT stop triggered while the server was down")
                    continue
            _setup_tsl_callbacks()
            exch, guard = r.get("exchange", "NSE"), r.get("guard_symbol", r["symbol"])
            strategy = r.get("strategy", SWING_STRATEGY)
            _tsl_sl_orders[oid] = {
                "sl_order_id": None, "gtt_id": r.get("gtt_id"), "gtt_buffer_pct": r.get("gtt_buffer_pct", 1.0),
                "gtt_trigger_offset_pct": GTT_TRIGGER_OFFSET_PCT, "product": r.get("product", "CNC"),
                "exchange": exch, "symbol": r["symbol"], "guard_symbol": guard, "agent": AGENT,
                "side": "BUY", "kind": "swing",
            }
            if r.get("exit_limit_buffer_pct"):
                _tsl_sl_orders[oid]["exit_limit_buffer_pct"] = r["exit_limit_buffer_pct"]
            pos = trailing_sl_engine.register(symbol=r["symbol"], strategy=strategy, side="BUY",
                                              entry_price=r["entry"], quantity=r["quantity"], order_id=oid)
            pos.current_sl = r["stop"]
            pos.best_price = r.get("best") or r["entry"]
            self._start_trail(oid, pos)
            self._trail[oid]["risk"] = r.get("initial_risk") or self._trail[oid]["risk"]
            if r.get("target"):
                self._targets[oid] = r["target"]
            self._open[oid] = r["symbol"]
            self._swing_meta[oid] = {k: r[k] for k in ("opened", "hold_until", "reason", "initial_risk",
                                                       "underlying", "expiry") if k in r}
            order_guard.restore_active(guard, strategy, "BUY", oid)
            risk_manager.position_opened()
            tick_engine.subscribe([{"symbol": r["symbol"], "exchange": exch, "seed_price": r.get("best") or r["entry"]}])
            restored.append(r["symbol"])
        if restored:
            try:
                asyncio.get_running_loop()
                self._ensure_tick_monitor()     # stops/targets/time exits run from the tick feed
            except RuntimeError:
                pass                            # no loop yet: started with the AI or the next trade
            self._record("system", f"restored swing positions: {', '.join(restored)}")
        self._save_swing()
        return restored

    def reapply_overnight(self) -> None:
        """After the daily counter reset: positions held from earlier days still exist."""
        for oid, e in self._overnight_entries().items():
            pos = trailing_sl_engine._positions.get(oid)
            order_guard.restore_active(e["guard_symbol"], pos.strategy if pos else SWING_STRATEGY, "BUY", oid)
            risk_manager.position_opened()

    # ── Index futures (intraday, long or short) ───────────────────────────

    def _future_price(self, c, spot: float) -> dict:
        if c.live and settings.uses_kite_market_data:
            q = kite_client.quote_kite([f"NFO:{c.tradingsymbol}"]).get(f"NFO:{c.tradingsymbol}") or {}
            depth = q.get("depth") or {}
            buy, sell = depth.get("buy") or [{}], depth.get("sell") or [{}]
            return {"ltp": float(q.get("last_price") or 0), "bid": float(buy[0].get("price") or 0),
                    "ask": float(sell[0].get("price") or 0), "estimated": False}
        tick, _ = tick_engine.latest(c.tradingsymbol)
        px = float(tick.ltp) if tick and tick.ltp > 0 else spot
        return {"ltp": px, "bid": px, "ask": px, "estimated": True}

    def _future_margin(self, c, side: str, qty: int, price: float) -> float:
        m = kite_client.order_margin(c.tradingsymbol, "NFO", side, qty, "MIS", "LIMIT", price)
        return m if m else price * qty * settings.ai_agent_future_margin_pct / 100

    def _future_quote(self, underlying: str) -> dict:
        und = normalize_underlying(underlying)
        if not settings.ai_agent_futures_enabled:
            raise ToolRejected("futures trading is disabled (AI_AGENT_FUTURES_ENABLED=false)")
        if und not in self._future_underlyings():
            raise ToolRejected(f"allowed underlyings: {self._future_underlyings()}")
        _, spot = self._index_spot(und)
        c = resolve_future(und, now_ist().date())
        if not c or c.lot_size <= 0:
            raise ToolRejected(f"no tradable {und} future found in the instrument list")
        q = self._future_price(c, spot)
        ltp = q["ltp"] or spot
        margin = self._future_margin(c, "BUY", c.lot_size, ltp)
        max_stop = settings.ai_agent_max_future_risk / c.lot_size
        return {"underlying": und, "contract": c.tradingsymbol,
                "expiry": c.expiry.isoformat() if c.expiry else "simulated", "lot_size": c.lot_size,
                "ltp": ltp, "bid": q["bid"], "ask": q["ask"], "price_estimated": q["estimated"],
                "spot": spot, "basis": round(ltp - spot, 2),
                "notional_per_lot": round(ltp * c.lot_size), "margin_per_lot": round(margin),
                "max_stop_distance_points_for_1_lot": round(max_stop, 2),
                "max_stop_distance_pct_for_1_lot": round(max_stop / ltp * 100, 3),
                "note": "stop_loss and target are future prices; the rupee loss at the stop is capped"}

    async def _place_future_trade(self, a: dict) -> dict:
        from agents.base_agent import (
            place_sl_with_retry, flatten_unprotected, _setup_tsl_callbacks, _tsl_sl_orders,
        )
        from sebi_compliance import sebi_compliance
        from market_regime import regime_detector

        und  = normalize_underlying(str(a.get("underlying", ""))) or str(a.get("underlying", ""))
        side = str(a.get("side", "")).upper()
        lots = int(a.get("lots") or 0)
        sl   = float(a.get("stop_loss") or 0)
        tgt  = float(a.get("target") or 0)
        why  = str(a.get("reasoning", ""))[:300]

        def reject(msg: str):
            self._record("rejected", f"{side} {lots} lot {und} FUT: {msg}")
            raise ToolRejected(msg)

        if not settings.ai_agent_futures_enabled:
            reject("futures trading is disabled (AI_AGENT_FUTURES_ENABLED=false)")
        allowed, blocked = self._trading_allowed()
        if not allowed:
            reject(blocked)
        ok, why_pruned = performance.is_allowed(FUT_STRATEGY)
        if not ok:
            reject(why_pruned)
        if not (OPT_ENTRY_START <= now_ist().time() <= OPT_ENTRY_END):
            reject(f"outside the futures entry window {OPT_ENTRY_START:%H:%M}-{OPT_ENTRY_END:%H:%M} IST")
        if und not in self._future_underlyings():
            reject(f"underlying must be one of {self._future_underlyings()}")
        if side not in ("BUY", "SELL"):
            reject("side must be BUY or SELL")
        if not 1 <= lots <= settings.ai_agent_max_future_lots:
            reject(f"lots must be 1-{settings.ai_agent_max_future_lots}")
        try:
            idx, spot = self._index_spot(und)
        except ToolRejected as e:
            reject(str(e))
        ref = self._event_ltp.get(idx)
        if ref and abs(spot - ref) / ref * 100 > settings.ai_agent_max_drift_pct:
            reject(f"{idx} moved {abs(spot - ref) / ref * 100:.2f}% since the alert; the setup is stale")
        c = resolve_future(und, now_ist().date())
        if not c or c.lot_size <= 0:
            reject(f"no tradable {und} future found in the instrument list")
        q = self._future_price(c, spot)
        ltp = q["ltp"]
        if ltp <= 0:
            reject(f"{c.tradingsymbol} has no price")
        if not q["estimated"] and (q["bid"] <= 0 or q["ask"] <= 0):
            reject(f"{c.tradingsymbol} has no two-sided quote")
        entry_px = round_tick((q["ask"] or ltp) * 1.0005 if side == "BUY" else (q["bid"] or ltp) * 0.9995)
        qty = lots * c.lot_size
        risk_dist = (entry_px - sl) if side == "BUY" else (sl - entry_px)
        if risk_dist <= 0:
            reject(f"stop_loss {sl} is on the wrong side of the future price {entry_px}")
        sl_pct = risk_dist / entry_px * 100
        if not (settings.ai_agent_future_min_sl_pct <= sl_pct <= settings.ai_agent_future_max_sl_pct):
            reject(f"stop is {sl_pct:.2f}% away; futures allow {settings.ai_agent_future_min_sl_pct}-"
                   f"{settings.ai_agent_future_max_sl_pct}%")
        if risk_dist * qty > settings.ai_agent_max_future_risk:
            reject(f"loss at the stop would be ₹{risk_dist * qty:,.0f}; the limit is "
                   f"₹{settings.ai_agent_max_future_risk:,.0f} (max {settings.ai_agent_max_future_risk / qty:.1f} points)")
        reward = (tgt - entry_px) if side == "BUY" else (entry_px - tgt)
        if reward < 1.5 * risk_dist:
            reject(f"target {tgt} gives {reward / risk_dist:.2f}x risk; need >= 1.5x")
        margin = self._future_margin(c, side, qty, entry_px)
        if margin > settings.ai_agent_max_future_margin:
            reject(f"margin ₹{margin:,.0f} exceeds the per-trade limit ₹{settings.ai_agent_max_future_margin:,.0f}")
        if self.trades_today >= settings.ai_agent_max_trades_per_day:
            reject(f"daily AI trade limit ({settings.ai_agent_max_trades_per_day}) reached")
        if self._intraday_count() >= settings.ai_agent_max_open_positions:
            reject(f"max open intraday AI positions ({settings.ai_agent_max_open_positions}) reached")
        if order_guard.is_symbol_active_anywhere(und):
            reject(f"{und} already has an open option or future position")
        ok, msg = order_guard.can_place(und, FUT_STRATEGY, side)
        if not ok:
            reject(msg)
        ok, msg = risk_manager.check_before_order(c.tradingsymbol, qty, entry_px, side, check_size=False)
        if not ok:
            reject(msg)
        ok, _algo, msg = sebi_compliance.pre_order_check(
            strategy=AGENT, symbol=c.tradingsymbol, exchange="NFO", transaction_type=side, quantity=qty,
            order_type="LIMIT", price_at_signal=entry_px, signal_source="ai_trader",
            regime=regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN",
            is_future=True,
        )
        if not ok:
            reject(msg)

        tick_engine.subscribe([{"symbol": c.tradingsymbol, "exchange": "NFO", "seed_price": entry_px}])
        order_id = kite_client.place_order(
            tradingsymbol=c.tradingsymbol, exchange="NFO", transaction_type=side, quantity=qty,
            order_type="LIMIT", price=entry_px, product="MIS", tag="Agent-ai-fut",
        )
        sebi_compliance.record_order_id(AGENT, c.tradingsymbol, order_id)

        sl, tgt = round_tick(sl), round_tick(tgt)
        sl_oid = await place_sl_with_retry(AGENT, c.tradingsymbol, "NFO", side, qty, "MIS", sl,
                                           order_type="SL", limit_buffer_pct=0.5)
        exit_px = round_tick(entry_px * (0.997 if side == "BUY" else 1.003))
        if sl_oid is None and await flatten_unprotected(AGENT, c.tradingsymbol, "NFO", side, qty, "MIS",
                                                        exit_limit_price=exit_px):
            reject("stop-loss order was rejected by the broker; entry was closed")

        order_guard.register_order(und, FUT_STRATEGY, side, order_id)
        risk_manager.position_opened()
        _setup_tsl_callbacks()
        _tsl_sl_orders[order_id] = {
            "sl_order_id": sl_oid, "product": "MIS", "exchange": "NFO",
            "symbol": c.tradingsymbol, "guard_symbol": und, "agent": AGENT, "side": side,
            "sl_type": "SL", "sl_buffer_pct": 0.5, "exit_limit_buffer_pct": 0.3, "kind": "future",
        }
        pos = trailing_sl_engine.register(symbol=c.tradingsymbol, strategy=FUT_STRATEGY, side=side,
                                          entry_price=entry_px, quantity=qty, order_id=order_id)
        pos.current_sl = sl
        self._start_trail(order_id, pos)
        self._targets[order_id] = tgt
        self._open[order_id] = c.tradingsymbol
        self.trades_today += 1
        self._ensure_tick_monitor()

        exp = c.expiry.isoformat() if c.expiry else "sim"
        text = (f"{side} {lots} lot ({qty}) {c.tradingsymbol} [{exp}] @ ₹{entry_px:.2f} SL ₹{sl:.2f} "
                f"TGT ₹{tgt:.2f} (risk ₹{risk_dist * qty:,.0f}, margin ≈₹{margin:,.0f}) — {why}")
        self._record("trade", text)
        await self._notify(f"🧠 <b>[AI]</b> {text}")
        return {"status": "placed", "order_id": order_id, "symbol": c.tradingsymbol, "expiry": exp,
                "side": side, "quantity": qty, "limit_price": entry_px, "stop_loss": sl, "target": tgt,
                "max_loss_at_stop": round(risk_dist * qty, 2), "margin": round(margin),
                "mode": self._mode_label()}

    # ── Bearish swing: long put held for days (NRML, loss limited to the premium) ──

    def _underlying_spot(self, name: str) -> tuple[str, str, float]:
        """(underlying name for NFO, streamed spot symbol, spot) for an index or an F&O stock."""
        und = normalize_underlying(name)
        if und:
            idx, spot = self._index_spot(und)
            return und, idx, spot
        sym = name.strip().upper()
        tick, _ = tick_engine.latest(sym)
        if not tick or tick.ltp <= 0 or tick_engine._exchange.get(sym, "NSE") != "NSE":
            raise ToolRejected(f"{sym} is not streaming — pick a stock from the watchlist")
        if self._tick_age(sym) > TICK_MAX_AGE_S:
            raise ToolRejected(f"{sym} quote is stale ({self._tick_age(sym)}s old)")
        return sym, sym, float(tick.ltp)

    async def _place_swing_put(self, a: dict) -> dict:
        from agents.base_agent import flatten_unprotected, _setup_tsl_callbacks, _tsl_sl_orders, sl_limit_price
        from sebi_compliance import sebi_compliance
        from market_regime import regime_detector

        name = str(a.get("underlying", "")).strip().upper()
        off  = int(a.get("strike_offset") or 0)
        lots = int(a.get("lots") or 0)
        slp  = float(a.get("stop_loss_pct") or 0)
        tgp  = float(a.get("target_pct") or 0)
        hold = int(a.get("max_hold_days") or 0)
        why  = str(a.get("reasoning", ""))[:300]

        def reject(msg: str):
            self._record("rejected", f"SWING PUT {lots} lot {name} {off:+d}: {msg}")
            raise ToolRejected(msg)

        if not (settings.ai_agent_swing_enabled and settings.ai_agent_swing_puts_enabled):
            reject("bearish swing (put buying) is disabled (AI_AGENT_SWING_PUTS_ENABLED=false)")
        allowed, blocked = self._trading_allowed(swing=True)
        if not allowed:
            reject(blocked)
        ok, why_pruned = performance.is_allowed(SWING_OPT_STRATEGY)
        if not ok:
            reject(why_pruned)
        if off not in (-1, 0, 1, 2):
            reject("strike_offset must be between -1 and 2")
        if not 1 <= lots <= settings.ai_agent_max_option_lots:
            reject(f"lots must be 1-{settings.ai_agent_max_option_lots}")
        if not (settings.ai_agent_swing_put_min_sl_pct <= slp <= settings.ai_agent_swing_put_max_sl_pct):
            reject(f"stop must be {settings.ai_agent_swing_put_min_sl_pct}-{settings.ai_agent_swing_put_max_sl_pct}% "
                   "of the premium")
        if tgp < 1.5 * slp:
            reject(f"target {tgp}% is under 1.5x the {slp}% stop")
        if not 1 <= hold <= settings.ai_agent_swing_max_hold_days:
            reject(f"max_hold_days must be 1-{settings.ai_agent_swing_max_hold_days}")
        if self.trades_today >= settings.ai_agent_max_trades_per_day:
            reject(f"daily AI trade limit ({settings.ai_agent_max_trades_per_day}) reached")
        if len(self._swing_entries()) >= settings.ai_agent_max_swing_positions:
            reject(f"max swing positions ({settings.ai_agent_max_swing_positions}) reached")
        try:
            und, spot_sym, spot = self._underlying_spot(name)
        except ToolRejected as e:
            reject(str(e))
        if order_guard.is_symbol_active_anywhere(und) or order_guard.is_symbol_active_anywhere(spot_sym):
            reject(f"{und} already has an open position")

        today = now_ist().date()
        hold_until = self._add_sessions(today, hold)
        min_exp = hold_until + timedelta(days=settings.ai_agent_swing_put_expiry_buffer_days)
        c = resolve(und, spot, "PE", off, today, min_expiry=min_exp)
        if not c or c.lot_size <= 0:
            reject(f"no {und} put expiring after {min_exp:%d %b} in the instrument list (is it an F&O stock?)")
        q = self._option_quotes([c], spot)[c.tradingsymbol]
        ltp, bid, ask = q["ltp"], q["bid"], q["ask"]
        if ltp < settings.ai_agent_option_min_premium:
            reject(f"premium ₹{ltp:.2f} is below ₹{settings.ai_agent_option_min_premium:.0f}")
        if not q["estimated"]:
            if ask <= 0 or bid <= 0:
                reject(f"{c.tradingsymbol} has no two-sided quote")
            if (ask - bid) / ltp * 100 > settings.ai_agent_option_max_spread_pct:
                reject(f"{c.tradingsymbol} spread {(ask - bid) / ltp * 100:.1f}% exceeds "
                       f"{settings.ai_agent_option_max_spread_pct}%")
        entry_px = round_tick((ask or ltp) * 1.01)
        qty = lots * c.lot_size
        if entry_px * qty > settings.ai_agent_max_swing_put_premium:
            per_lot = entry_px * c.lot_size
            reject(f"premium ₹{entry_px * qty:,.0f} exceeds ₹{settings.ai_agent_max_swing_put_premium:,.0f} "
                   f"(₹{per_lot:,.0f}/lot)")
        ok, msg = order_guard.can_place(und, SWING_OPT_STRATEGY, "BUY")
        if not ok:
            reject(msg)
        ok, msg = risk_manager.check_before_order(c.tradingsymbol, qty, entry_px, "BUY")
        if not ok:
            reject(msg)
        ok, _algo, msg = sebi_compliance.pre_order_check(
            strategy=AGENT, symbol=c.tradingsymbol, exchange="NFO", transaction_type="BUY", quantity=qty,
            order_type="LIMIT", price_at_signal=entry_px, signal_source="ai_trader",
            regime=regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN",
        )
        if not ok:
            reject(msg)

        tick_engine.subscribe([{"symbol": c.tradingsymbol, "exchange": "NFO", "seed_price": ltp}])
        order_id = kite_client.place_order(
            tradingsymbol=c.tradingsymbol, exchange="NFO", transaction_type="BUY", quantity=qty,
            order_type="LIMIT", price=entry_px, product="NRML", tag="Agent-ai-swput",
        )
        sebi_compliance.record_order_id(AGENT, c.tradingsymbol, order_id)

        sl_px, tgt_px = round_tick(entry_px * (1 - slp / 100)), round_tick(entry_px * (1 + tgp / 100))
        buf = 5.0
        gtt_trigger = round_tick(sl_px * (1 - GTT_TRIGGER_OFFSET_PCT / 100))
        gtt_id = None
        for attempt in range(3):
            try:
                gtt_id = kite_client.place_gtt_stop(c.tradingsymbol, "NFO", qty, gtt_trigger,
                                                    sl_limit_price(gtt_trigger, "BUY", buf), entry_px,
                                                    product="NRML")
                break
            except Exception as exc:
                logger.error("[ai] GTT for {} failed (attempt {}/3): {}", c.tradingsymbol, attempt + 1, exc)
                await asyncio.sleep(0.2)
        if gtt_id is None and await flatten_unprotected(
                AGENT, c.tradingsymbol, "NFO", "BUY", qty, "NRML",
                exit_limit_price=round_tick(max(bid or ltp, 0.1) * 0.97)):
            reject("the GTT stop-loss was rejected by the broker; the put was sold again")

        order_guard.register_order(und, SWING_OPT_STRATEGY, "BUY", order_id)
        risk_manager.position_opened()
        _setup_tsl_callbacks()
        _tsl_sl_orders[order_id] = {
            "sl_order_id": None, "gtt_id": gtt_id, "gtt_buffer_pct": buf,
            "gtt_trigger_offset_pct": GTT_TRIGGER_OFFSET_PCT, "product": "NRML", "exchange": "NFO",
            "symbol": c.tradingsymbol, "guard_symbol": und, "agent": AGENT, "side": "BUY", "kind": "swing",
            "exit_limit_buffer_pct": 3.0,
        }
        pos = trailing_sl_engine.register(symbol=c.tradingsymbol, strategy=SWING_OPT_STRATEGY, side="BUY",
                                          entry_price=entry_px, quantity=qty, order_id=order_id)
        pos.current_sl = sl_px
        self._start_trail(order_id, pos)
        self._targets[order_id] = tgt_px
        self._open[order_id] = c.tradingsymbol
        exp = c.expiry.isoformat() if c.expiry else None
        self._swing_meta[order_id] = {"opened": today.isoformat(), "hold_until": hold_until.isoformat(),
                                      "reason": why, "initial_risk": entry_px - sl_px,
                                      "underlying": und, "expiry": exp}
        self._save_swing()
        self.trades_today += 1
        self._ensure_tick_monitor()

        text = (f"SWING PUT BUY {lots} lot ({qty}) {c.tradingsymbol} [{exp or 'sim'}] @ ₹{entry_px:.2f} "
                f"SL ₹{sl_px:.2f} (GTT) TGT ₹{tgt_px:.2f}, hold until {hold_until:%d %b}, "
                f"cost ₹{entry_px * qty:,.0f} — {why}")
        self._record("trade", text)
        await self._notify(f"🧠 <b>[AI]</b> {text}")
        return {"status": "placed", "order_id": order_id, "symbol": c.tradingsymbol, "underlying": und,
                "expiry": exp or "simulated", "product": "NRML", "quantity": qty, "limit_price": entry_px,
                "stop_loss": sl_px, "target": tgt_px, "gtt_id": gtt_id, "hold_until": hold_until.isoformat(),
                "max_loss": round((entry_px - sl_px) * qty, 2), "premium_paid": round(entry_px * qty, 2),
                "mode": self._mode_label()}

    # ── Option selling: defined-risk credit spreads (intraday) ────────────
    # Sell one option and buy a further-OTM one of the same type and expiry. The worst case
    # is fixed at entry: (strike width − credit) × quantity. Naked selling is never allowed.

    async def _wait_filled(self, order_id: str, timeout_s: float = 6.0) -> bool:
        """True once the broker reports the order COMPLETE. A spread's second leg is only sent
        after the first one filled, so a short option is never left without its hedge."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            try:
                hist = await asyncio.to_thread(kite_client.order_history, order_id)
                status = str((hist[-1] if hist else {}).get("status", "")).upper()
            except Exception:
                status = ""
            if status == "COMPLETE":
                return True
            if status in ("REJECTED", "CANCELLED") or loop.time() >= deadline:
                return False
            await asyncio.sleep(0.5)

    def _spread_legs(self, und: str, spot: float, view: str, off: int, width: int):
        otype = "PE" if view == "BULLISH" else "CE"
        today = now_ist().date()
        short = resolve(und, spot, otype, off, today)
        long_ = resolve(und, spot, otype, off + width, today)
        return otype, short, long_

    def _spread_value(self, sp: dict, spot: Optional[float] = None) -> float:
        """What it costs now to close the spread (per unit): buy back short − sell long."""
        if sp["live"] and settings.uses_kite_market_data:
            ts, tl = tick_engine.latest(sp["short"]["symbol"])[0], tick_engine.latest(sp["long"]["symbol"])[0]
            if ts and tl and ts.ltp > 0 and tl.ltp > 0:
                return max((ts.ask or ts.ltp) - (tl.bid or tl.ltp), 0.0)
        if spot is None:
            tick, _ = tick_engine.latest(INDEX_QUOTE[sp["underlying"]])
            spot = float(tick.ltp) if tick else 0.0
        if spot <= 0:
            return sp["value"]
        days = max((date.fromisoformat(sp["expiry"]) - now_ist().date()).days, 1) if sp["expiry"] else 5.0
        s = estimate_premium(spot, sp["short"]["strike"], sp["option_type"], days=days)
        l = estimate_premium(spot, sp["long"]["strike"], sp["option_type"], days=days)
        return max(s - l, 0.0)

    async def _place_option_spread(self, a: dict) -> dict:
        from sebi_compliance import sebi_compliance
        from market_regime import regime_detector

        und   = normalize_underlying(str(a.get("underlying", ""))) or str(a.get("underlying", ""))
        view  = str(a.get("view", "")).upper()
        off   = int(a.get("short_strike_offset") or 0)
        width = int(a.get("width_strikes") or 0)
        lots  = int(a.get("lots") or 0)
        slp   = float(a.get("stop_loss_pct") or 0)
        tgp   = float(a.get("target_pct") or 0)
        why   = str(a.get("reasoning", ""))[:300]

        def reject(msg: str):
            self._record("rejected", f"SPREAD {view} {und} {off}/{width} x{lots}: {msg}")
            raise ToolRejected(msg)

        if not settings.ai_agent_option_selling_enabled:
            reject("option selling is disabled (AI_AGENT_OPTION_SELLING_ENABLED=false)")
        allowed, blocked = self._trading_allowed()
        if not allowed:
            reject(blocked)
        ok, why_pruned = performance.is_allowed(SPREAD_STRATEGY)
        if not ok:
            reject(why_pruned)
        if not (OPT_ENTRY_START <= now_ist().time() <= OPT_ENTRY_END):
            reject(f"outside the option entry window {OPT_ENTRY_START:%H:%M}-{OPT_ENTRY_END:%H:%M} IST")
        if und not in self._option_underlyings():
            reject(f"underlying must be one of {self._option_underlyings()}")
        if view not in ("BULLISH", "BEARISH"):
            reject("view must be BULLISH (sell a put spread) or BEARISH (sell a call spread)")
        if off not in (0, 1, 2, 3, 4):
            reject("short_strike_offset must be 0-4 (0 = ATM, higher = further out of the money)")
        if width not in (1, 2, 3, 4):
            reject("width_strikes must be 1-4")
        if not 1 <= lots <= settings.ai_agent_max_spread_lots:
            reject(f"lots must be 1-{settings.ai_agent_max_spread_lots}")
        if not 50 <= slp <= 200:
            reject("stop_loss_pct must be 50-200 (% of the credit you can lose before exiting)")
        if not 30 <= tgp <= 90:
            reject("target_pct must be 30-90 (% of the credit to keep before taking profit)")
        if self.trades_today >= settings.ai_agent_max_trades_per_day:
            reject(f"daily AI trade limit ({settings.ai_agent_max_trades_per_day}) reached")
        if self._intraday_count() >= settings.ai_agent_max_open_positions:
            reject(f"max open intraday AI positions ({settings.ai_agent_max_open_positions}) reached")
        if order_guard.is_symbol_active_anywhere(und):
            reject(f"{und} already has an open option or future position")
        try:
            idx, spot = self._index_spot(und)
        except ToolRejected as e:
            reject(str(e))
        ref = self._event_ltp.get(idx)
        if ref and abs(spot - ref) / ref * 100 > settings.ai_agent_max_drift_pct:
            reject(f"{idx} moved {abs(spot - ref) / ref * 100:.2f}% since the alert; the setup is stale")

        otype, short, long_ = self._spread_legs(und, spot, view, off, width)
        if not short or not long_ or short.lot_size <= 0 or short.expiry != long_.expiry:
            reject(f"no tradable {und} {otype} strikes {off} and {off + width} out of the money")
        q = self._option_quotes([short, long_], spot)
        qs, ql = q[short.tradingsymbol], q[long_.tradingsymbol]
        if not qs["estimated"]:
            for leg, qq in ((short, qs), (long_, ql)):
                if qq["bid"] <= 0 or qq["ask"] <= 0:
                    reject(f"{leg.tradingsymbol} has no two-sided quote")
                if (qq["ask"] - qq["bid"]) / qq["ltp"] * 100 > settings.ai_agent_option_max_spread_pct * 2:
                    reject(f"{leg.tradingsymbol} bid/ask spread too wide")
        sell_px = round_tick((qs["bid"] or qs["ltp"]) * 0.99)
        buy_px  = round_tick((ql["ask"] or ql["ltp"]) * 1.01)
        credit  = round(sell_px - buy_px, 2)
        width_pts = abs(short.strike - long_.strike)
        qty = lots * short.lot_size
        if credit <= 0:
            reject("the spread pays no credit at these prices")
        if credit < settings.ai_agent_spread_min_credit_pct / 100 * width_pts:
            reject(f"credit ₹{credit:.2f} is under {settings.ai_agent_spread_min_credit_pct:.0f}% of the "
                   f"{width_pts:.0f}-point width — too little reward for the risk; sell closer or narrower")
        max_loss = (width_pts - credit) * qty
        if max_loss > settings.ai_agent_max_spread_loss:
            reject(f"worst-case loss ₹{max_loss:,.0f} exceeds ₹{settings.ai_agent_max_spread_loss:,.0f}; "
                   "use a narrower width")
        stop_value = round(min(credit * (1 + slp / 100), width_pts * 0.95), 2)
        target_value = round(credit * (1 - tgp / 100), 2)
        margin = None
        if short.live and settings.trading_mode != "PAPER" and kite_client.has_session():
            try:
                res = kite_client.ensure_session().basket_order_margins([
                    {"exchange": "NFO", "tradingsymbol": long_.tradingsymbol, "transaction_type": "BUY",
                     "variety": "regular", "product": "MIS", "order_type": "LIMIT", "quantity": qty, "price": buy_px},
                    {"exchange": "NFO", "tradingsymbol": short.tradingsymbol, "transaction_type": "SELL",
                     "variety": "regular", "product": "MIS", "order_type": "LIMIT", "quantity": qty, "price": sell_px}])
                margin = float(res["final"]["total"])
            except Exception as exc:
                logger.warning("[ai] basket margin failed: {}", exc)
        margin = margin or max_loss * 1.25 + buy_px * qty
        if margin > settings.ai_agent_max_spread_margin:
            reject(f"margin ₹{margin:,.0f} exceeds ₹{settings.ai_agent_max_spread_margin:,.0f}")
        ok, msg = order_guard.can_place(und, SPREAD_STRATEGY, "SELL")
        if not ok:
            reject(msg)
        ok, msg = risk_manager.check_before_order(short.tradingsymbol, qty, credit, "SELL", check_size=False)
        if not ok:
            reject(msg)
        regime = regime_detector.current_regime.value if regime_detector.current_regime else "UNKNOWN"
        for leg, side, px in ((long_, "BUY", buy_px), (short, "SELL", sell_px)):
            ok, _algo, msg = sebi_compliance.pre_order_check(
                strategy=AGENT, symbol=leg.tradingsymbol, exchange="NFO", transaction_type=side, quantity=qty,
                order_type="LIMIT", price_at_signal=px, signal_source="ai_trader", regime=regime)
            if not ok:
                reject(msg)

        tick_engine.subscribe([{"symbol": long_.tradingsymbol, "exchange": "NFO", "seed_price": ql["ltp"]},
                               {"symbol": short.tradingsymbol, "exchange": "NFO", "seed_price": qs["ltp"]}])
        # Protective leg first: the short leg must never exist without it
        long_oid = kite_client.place_order(
            tradingsymbol=long_.tradingsymbol, exchange="NFO", transaction_type="BUY", quantity=qty,
            order_type="LIMIT", price=buy_px, product="MIS", tag="Agent-ai-spr-L")
        sebi_compliance.record_order_id(AGENT, long_.tradingsymbol, long_oid)
        if not await self._wait_filled(long_oid):
            try:
                kite_client.cancel_order(long_oid)
            except Exception:
                pass
            reject(f"the protective {long_.tradingsymbol} order did not fill; nothing was sold "
                   "(check Kite in case it filled late)")
        try:
            short_oid = kite_client.place_order(
                tradingsymbol=short.tradingsymbol, exchange="NFO", transaction_type="SELL", quantity=qty,
                order_type="LIMIT", price=sell_px, product="MIS", tag="Agent-ai-spr-S")
        except Exception as exc:
            try:
                kite_client.place_order(
                    tradingsymbol=long_.tradingsymbol, exchange="NFO", transaction_type="SELL", quantity=qty,
                    order_type="LIMIT", price=round_tick(max(ql["bid"] or ql["ltp"], 0.1) * 0.97),
                    product="MIS", tag="Agent-ai-spr-UNDO")
            except Exception:
                logger.critical("[ai] spread: short leg failed AND unwinding {} failed", long_.tradingsymbol)
            reject(f"short leg order failed ({exc}); the protective leg was sold again")
        sebi_compliance.record_order_id(AGENT, short.tradingsymbol, short_oid)

        order_guard.register_order(und, SPREAD_STRATEGY, "SELL", short_oid)
        risk_manager.position_opened()
        sid = f"{und}-{'BPS' if view == 'BULLISH' else 'BCS'}-{short_oid}"
        self._spreads[sid] = {
            "id": sid, "underlying": und, "view": view, "option_type": otype, "lots": lots, "qty": qty,
            "short": {"symbol": short.tradingsymbol, "strike": short.strike, "order_id": short_oid},
            "long":  {"symbol": long_.tradingsymbol, "strike": long_.strike, "order_id": long_oid},
            "credit": credit, "width": width_pts, "max_loss": round(max_loss, 2), "margin": round(margin),
            "stop_value": stop_value, "target_value": target_value, "value": credit,
            "expiry": short.expiry.isoformat() if short.expiry else None, "live": short.live,
            "opened": now_ist().strftime("%H:%M:%S"), "reason": why,
        }
        self.trades_today += 1
        self._ensure_tick_monitor()

        name = "bull put spread" if view == "BULLISH" else "bear call spread"
        text = (f"SELL {name} {und} {short.strike:g}/{long_.strike:g} {otype} x{lots} lot ({qty}) "
                f"credit ₹{credit:.2f} (₹{credit * qty:,.0f}), max loss ₹{max_loss:,.0f}, "
                f"stop at ₹{stop_value:.2f}, take profit at ₹{target_value:.2f} — {why}")
        self._record("trade", text)
        await self._notify(f"🧠 <b>[AI]</b> {text}")
        return {"status": "placed", "spread_id": sid, "structure": name, "short": short.tradingsymbol,
                "long": long_.tradingsymbol, "quantity": qty, "credit_per_unit": credit,
                "credit_total": round(credit * qty, 2), "max_loss": round(max_loss, 2),
                "stop_when_spread_costs": stop_value, "take_profit_when_spread_costs": target_value,
                "margin": round(margin), "expiry": self._spreads[sid]["expiry"] or "simulated",
                "mode": self._mode_label()}

    async def _close_spread(self, sid: str, reason: str, by_code: bool = False) -> dict:
        sp = self._spreads.pop(sid, None)
        if not sp:
            raise ToolRejected(f"no open spread {sid}")
        value = self._spread_value(sp)
        days = max((date.fromisoformat(sp["expiry"]) - now_ist().date()).days, 1) if sp["expiry"] else 5.0
        tick, _ = tick_engine.latest(INDEX_QUOTE[sp["underlying"]])
        spot = float(tick.ltp) if tick else 0.0

        def px(leg: str, buy: bool) -> float:
            t, _ = tick_engine.latest(sp[leg]["symbol"])
            if sp["live"] and t and t.ltp > 0:
                ref = (t.ask or t.ltp) if buy else (t.bid or t.ltp)
            else:
                ref = estimate_premium(spot, sp[leg]["strike"], sp["option_type"], days=days) if spot else 0.05
            return round_tick(max(ref * (1.03 if buy else 0.97), 0.05))

        # Short leg first (removes the risk); the protective leg is sold only once the buy-back
        # has filled — otherwise the short option would be left naked
        bought_back = False
        try:
            buy_oid = kite_client.place_order(
                tradingsymbol=sp["short"]["symbol"], exchange="NFO", transaction_type="BUY",
                quantity=sp["qty"], order_type="LIMIT", price=px("short", True),
                product="MIS", tag="Agent-ai-spr-EXIT")
            bought_back = await self._wait_filled(buy_oid)
        except Exception as exc:
            logger.critical("[ai] spread exit BUY {} FAILED: {}", sp["short"]["symbol"], exc)
        if bought_back:
            try:
                kite_client.place_order(
                    tradingsymbol=sp["long"]["symbol"], exchange="NFO", transaction_type="SELL",
                    quantity=sp["qty"], order_type="LIMIT", price=px("long", False),
                    product="MIS", tag="Agent-ai-spr-EXIT")
            except Exception as exc:
                logger.critical("[ai] spread exit SELL {} FAILED: {}", sp["long"]["symbol"], exc)
        else:
            msg = (f"⚠️ spread {sid}: buy-back of {sp['short']['symbol']} did not fill — the protective "
                   f"{sp['long']['symbol']} was KEPT so the short is still hedged. Close both in Kite.")
            self._record("error", msg)
            await self._notify(f"🧠 <b>[AI]</b> {msg}")
        pnl = round((sp["credit"] - value) * sp["qty"], 2)
        order_guard.release_order(sp["underlying"], SPREAD_STRATEGY, "SELL", pnl)
        risk_manager.record_trade(pnl)
        risk_manager.position_closed()
        from performance import performance
        # two legs ≈ twice the option round-trip costs: record as one trade with double quantity
        performance.record_trade(SPREAD_STRATEGY, sid, "SELL", sp["qty"] * 2, sp["credit"], pnl, "NFO", "MIS",
                                 exit_price=value, note=reason)
        text = (f"CLOSE spread {sp['underlying']} {sp['short']['strike']:g}/{sp['long']['strike']:g} "
                f"{sp['option_type']} at ₹{value:.2f} (sold ₹{sp['credit']:.2f}) P&L ₹{pnl:,.0f} — {reason[:200]}")
        self._record("close", f"[auto] {text}" if by_code else text)
        await self._notify(f"🧠 <b>[AI]</b> {text}")
        return {"status": "closed", "closed": [{"symbol": sid, "quantity": sp["qty"], "exit_value": round(value, 2),
                                                "pnl": pnl}]}

    async def _watch_spreads(self, snap) -> None:
        """Stop, take-profit and risk alerts for open spreads, on index and leg ticks."""
        for sid, sp in list(self._spreads.items()):
            legs = (sp["short"]["symbol"], sp["long"]["symbol"])
            if snap.symbol not in legs and snap.symbol != INDEX_QUOTE[sp["underlying"]]:
                continue
            spot = snap.tick.ltp if snap.symbol == INDEX_QUOTE[sp["underlying"]] else None
            v = self._spread_value(sp, spot)
            sp["value"] = round(v, 2)
            if v >= sp["stop_value"]:
                async with self._action_lock:
                    if sid in self._spreads:
                        await self._close_spread(sid, f"stop: spread cost ₹{v:.2f} ≥ ₹{sp['stop_value']:.2f}", by_code=True)
            elif v <= sp["target_value"]:
                async with self._action_lock:
                    if sid in self._spreads:
                        await self._close_spread(sid, f"take profit: spread cost ₹{v:.2f} ≤ ₹{sp['target_value']:.2f}",
                                                 by_code=True)
            elif self.enabled and settings.ai_agent_realtime and spot is not None:
                used = (v - sp["credit"]) / max(sp["stop_value"] - sp["credit"], 0.01)
                if used >= 0.6:
                    self._raise(snap, "position_risk",
                                f"credit spread {sid} has used {used:.0%} of its stop (cost ₹{v:.2f}, "
                                f"sold ₹{sp['credit']:.2f})", {"spread_id": sid})

    async def _close_position(self, symbol: str, reason: str, by_code: bool = False) -> dict:
        from agents.base_agent import _close_tracked
        spread_ids = [sid for sid, sp in self._spreads.items()
                      if symbol in (sid, sp["underlying"], sp["short"]["symbol"], sp["long"]["symbol"])
                      or normalize_underlying(symbol) == sp["underlying"]]
        if spread_ids:
            return await self._close_spread(spread_ids[0], reason, by_code)
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
            _close_tracked(oid, symbol, pos.side, pos.quantity, pnl, "Agent-ai-EXIT", pos.strategy)
            self._targets.pop(oid, None)
            self._open.pop(oid, None)
            self._trail.pop(oid, None)
            if self._swing_meta.pop(oid, None) is not None:
                self._save_swing()
            closed.append({"symbol": symbol, "quantity": pos.quantity, "exit_price": ltp,
                           "pnl": round(pnl, 2)})
            text = f"CLOSE {symbol} @ ~₹{ltp:.2f} P&L ₹{pnl:,.0f} — {reason[:200]}"
            self._record("close", text if not by_code else f"[auto] {text}")
            await self._notify(f"🧠 <b>[AI]</b> {text}")
        return {"status": "closed", "closed": closed}

    async def close_all(self, reason: str, include_swing: bool = True) -> list:
        """Close AI positions. The 15:00 force exit passes include_swing=False."""
        out = []
        entries = self._ai_entries() if include_swing else self._intraday_entries()
        for sid in list(self._spreads):
            async with self._action_lock:
                if sid in self._spreads:
                    out.append(await self._close_spread(sid, reason, by_code=True))
        for sym in sorted({e["symbol"] for e in entries.values()}):
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
            self._handover_if_ai_unavailable(held)
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
        now_t = now_ist().time()
        if self._spreads:
            if now_t >= FORCE_EXIT:
                for sid in list(self._spreads):
                    async with self._action_lock:
                        if sid in self._spreads:
                            await self._close_spread(sid, "end of AI trading window (15:00)", by_code=True)
            else:
                await self._watch_spreads(snap)
        if now_t >= FORCE_EXIT and any(self._kind(e) != "swing" for e in held.values()):
            await self.close_all("end of AI trading window (15:00)", include_swing=False)
        swing_held = {o: e for o, e in held.items() if self._kind(e) == "swing" and o in self._ai_entries()}
        if swing_held:
            await self._swing_housekeeping(swing_held, now_t)
        self._maybe_swing_scan(now_t)
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
            elif (self._kind(e) != "swing" and st_prev and st_prev != ind.supertrend_dir
                  and ind.supertrend_dir == ("DOWN" if long else "UP")):
                self._raise(snap, "trend_flip",
                            f"Supertrend flipped {ind.supertrend_dir} against the {e['side']} position")
            else:
                self._check_trail(snap, oid, pos)
            return

        if not self._can_consider_entry(sym):
            return
        und = normalize_underlying(sym)
        if und:
            self._detect_index(snap, und)
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

    # ── AI-managed trailing stops ─────────────────────────────────────────

    def _start_trail(self, order_id: str, pos) -> None:
        import time as _t
        risk = abs(pos.entry_price - pos.current_sl)
        self._trail[order_id] = {"risk": risk, "milestone": 0.0, "last_review": _t.monotonic(),
                                 "best_at_review": pos.entry_price}
        pos.external_trail = bool(settings.ai_agent_ai_trailing and risk > 0)

    def _ai_trailing_available(self) -> bool:
        return (self.enabled and settings.ai_agent_realtime and bool(settings.anthropic_api_key)
                and self.cost_today_usd < settings.ai_agent_max_daily_cost_usd and self._ai_errors < 3)

    def _handover_if_ai_unavailable(self, held: dict) -> None:
        """Runs on every tick of a held position, even when the AI is stopped: if Claude
        can't manage the trail, give the position back to the mechanical trailing stop."""
        if self._ai_trailing_available():
            return
        for oid in held:
            pos = trailing_sl_engine._positions.get(oid)
            if pos and pos.external_trail:
                pos.external_trail = False
                self._record("system", f"{pos.symbol}: AI unavailable — standard trailing stop took over")

    def _check_trail(self, snap, oid: str, pos) -> None:
        """Raise a trail_review at R milestones and while price keeps making new highs;
        hand the position back to the mechanical trail if the AI can't manage it."""
        import time as _t
        st = self._trail.get(oid)
        if not st or not pos.external_trail:
            return
        long = pos.side == "BUY"
        ltp = snap.tick.ltp
        r = ((ltp - pos.entry_price) if long else (pos.entry_price - ltp)) / st["risk"]
        reached = [m for m in R_MILESTONES if m <= r and m > st["milestone"]]
        now = _t.monotonic()
        detail = f"+{r:.2f}R, stop ₹{pos.current_sl:.2f}, best ₹{pos.best_price:.2f}"
        if reached:
            st["milestone"] = max(reached)
            st["last_review"], st["best_at_review"] = now, pos.best_price
            self._raise(snap, "trail_review", f"reached {max(reached):g}R milestone ({detail})",
                        {"r_multiple": round(r, 2)}, cooldown=0)
            return
        better = pos.best_price > st["best_at_review"] if long else pos.best_price < st["best_at_review"]
        gap_s = (SWING_REVIEW_S if pos.strategy in (SWING_STRATEGY, SWING_OPT_STRATEGY)
                 else settings.ai_agent_trail_review_s)
        if r >= 0.5 and better and now - st["last_review"] >= gap_s:
            st["last_review"], st["best_at_review"] = now, pos.best_price
            self._raise(snap, "trail_review", f"new {'high' if long else 'low'} since last review ({detail})",
                        {"r_multiple": round(r, 2)}, cooldown=gap_s)

    async def _adjust_stop(self, a: dict) -> dict:
        from options_resolver import round_tick as _rt
        sym = str(a.get("symbol", "")).upper().strip()
        new_stop = float(a.get("new_stop") or 0)
        new_tgt = float(a.get("new_target") or 0)
        why = str(a.get("reasoning", ""))[:300]

        def reject(msg: str):
            self._record("rejected", f"adjust_stop {sym} → {new_stop}: {msg}")
            raise ToolRejected(msg)

        found = [(o, e) for o, e in self._ai_entries().items() if e["symbol"] == sym]
        if not found:
            reject(f"no open AI position in {sym}")
        oid, e = found[0]
        pos = trailing_sl_engine._positions.get(oid)
        if not pos:
            reject(f"{sym} is no longer tracked")
        tick, _ = tick_engine.latest(sym)
        if not tick or tick.ltp <= 0:
            reject(f"{sym} has no live price")
        ltp, long = float(tick.ltp), pos.side == "BUY"
        new_stop = _rt(new_stop)
        is_opt = self._kind(e) == "option" or (self._kind(e) == "swing" and e.get("exchange") == "NFO")
        gap = (settings.ai_agent_option_trail_min_gap_pct if is_opt else settings.ai_agent_trail_min_gap_pct)
        if long:
            if new_stop <= pos.current_sl:
                reject(f"stops only tighten: {new_stop} is not above the current stop {pos.current_sl}")
            if new_stop > ltp * (1 - gap / 100):
                reject(f"stop must stay at least {gap}% below price ₹{ltp:.2f} (max {_rt(ltp * (1 - gap / 100))})")
        else:
            if new_stop >= pos.current_sl:
                reject(f"stops only tighten: {new_stop} is not below the current stop {pos.current_sl}")
            if new_stop < ltp * (1 + gap / 100):
                reject(f"stop must stay at least {gap}% above price ₹{ltp:.2f} (min {_rt(ltp * (1 + gap / 100))})")
        if new_tgt:
            if (long and new_tgt <= ltp) or (not long and new_tgt >= ltp):
                reject(f"target {new_tgt} must be beyond the current price ₹{ltp:.2f}")
            self._targets[oid] = _rt(new_tgt)

        old = pos.current_sl
        pos.current_sl = new_stop
        pos.sl_moves += 1
        pos.locked_profit = (new_stop - pos.entry_price) * pos.quantity * (1 if long else -1)
        if trailing_sl_engine.on_sl_moved:
            await trailing_sl_engine.on_sl_moved(pos, old, "AI_TRAIL")   # updates the exchange stop order
        text = (f"TRAIL {sym} stop ₹{old:.2f} → ₹{new_stop:.2f}"
                + (f", target → ₹{self._targets[oid]:.2f}" if new_tgt else "")
                + f" (locks ₹{pos.locked_profit:,.0f}) — {why}")
        self._record("trail", text)
        if oid in self._swing_meta:
            self._save_swing()
        return {"status": "adjusted", "symbol": sym, "old_stop": old, "new_stop": new_stop,
                "target": self._targets.get(oid), "locked_profit": round(pos.locked_profit, 2)}

    def _detect_index(self, snap, und: str) -> None:
        """Index tick: the F&O pattern engine decides whether it is a directional index setup
        (traded with an option buy or a future)."""
        sell_ok = settings.ai_agent_option_selling_enabled and und in self._option_underlyings()
        opt_ok = settings.ai_agent_options_enabled and und in self._option_underlyings()
        fut_ok = settings.ai_agent_futures_enabled and und in self._future_underlyings()
        if not (opt_ok or fut_ok or sell_ok):
            return
        if not (OPT_ENTRY_START <= now_ist().time() <= OPT_ENTRY_END):
            return
        if order_guard.is_symbol_active_anywhere(und):
            return
        scout = self._get_opt_scout()
        if len(snap.candles_1min) < scout.min_candles_1min:
            return
        try:
            action, sig = scout.evaluate_tick(snap)
        except Exception:
            return
        if action == "BUY" and sig:
            self._raise(snap, "option_setup",
                        f"{sig.get('pattern', '')} → {sig.get('option_type')} (score {sig.get('score')})",
                        {"underlying": und, "suggested_option_type": sig.get("option_type"),
                         "direction": "BULLISH" if sig.get("option_type") == "CE" else "BEARISH",
                         "options_allowed": opt_ok, "futures_allowed": fut_ok, "selling_allowed": sell_ok})

    def _get_opt_scout(self):
        if getattr(self, "_opt_scout", None) is None:
            from agents.strategy_agents import FnOAgent
            self._opt_scout = FnOAgent()
        return self._opt_scout

    def _future_underlyings(self) -> list[str]:
        return [u for u in (normalize_underlying(x) for x in settings.ai_agent_future_underlyings.split(","))
                if u]

    def _option_underlyings(self) -> list[str]:
        return [u for u in (normalize_underlying(x) for x in settings.ai_agent_option_underlyings.split(","))
                if u]

    def _can_consider_entry(self, sym: str) -> bool:
        if not self._trading_allowed()[0] or not settings.anthropic_api_key:
            return False
        if self.cost_today_usd >= settings.ai_agent_max_daily_cost_usd:
            return False
        if self.trades_today >= settings.ai_agent_max_trades_per_day:
            return False
        if self._intraday_count() >= settings.ai_agent_max_open_positions:
            return False
        return not order_guard.is_symbol_active_anywhere(sym)

    def _get_scout(self):
        if self._scout is None:
            from agents.strategy_agents import IntradayAgent
            self._scout = IntradayAgent()
        return self._scout

    def _raise(self, snap, kind: str, reason: str, extra: Optional[dict] = None,
               cooldown: Optional[int] = None) -> None:
        import time as _t
        key = (snap.symbol, kind)
        now = _t.monotonic()
        wait = settings.ai_agent_trigger_cooldown_s if cooldown is None else cooldown
        if now - self._trigger_ts.get(key, -1e9) < wait:
            return
        self._trigger_ts[key] = now
        ev = {"symbol": snap.symbol, "kind": kind, "reason": reason, "ltp": snap.tick.ltp,
              "at": now_ist().strftime("%H:%M:%S"), "mono": now, **(extra or {})}
        prev = self._events.get(snap.symbol)
        if prev and _ALERT_RANK.get(prev["kind"], 2) < _ALERT_RANK.get(kind, 2):
            return                                   # never let a lower-priority alert replace a higher one
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
                        _ALERT_RANK.get(kv[1]["kind"], 2), kv[1]["mono"]))
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
        is_exit = ev["kind"] in _ALERT_RANK            # about an open position, not a new entry
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
        if ev["kind"] in ("position_risk", "trend_flip"):
            action = ("Decide now whether to close this position early or keep it (its stop is "
                      "already working).")
        elif ev["kind"] == "trail_review":
            action = ("Trail review: tighten the stop with adjust_stop (optionally move the target), "
                      "close, or leave it. Stops can only tighten.")
        elif ev["kind"] == "option_setup":
            ways = []
            if ev.get("options_allowed"):
                ways.append("buy an option (get_option_chain, then place_option_trade)")
            if ev.get("futures_allowed"):
                ways.append("trade the future (get_future_quote, then place_future_trade)")
            if ev.get("selling_allowed"):
                ways.append("sell a credit spread on the same side (get_option_chain, then place_option_spread)")
            action = (f"Index {ev.get('direction', '').lower()} setup. If the move is clear, "
                      f"{' or '.join(ways) or 'PASS'}; otherwise PASS.")
        else:
            action = "Decide now: place_trade if the setup is clearly good, otherwise PASS."
        return (f"REAL-TIME ALERT {ev['at']} IST — {ev['kind']} on {sym}: {ev['reason']}\n"
                f"{action}\n{json.dumps(data, default=str)}")

    def _note_stop_exits(self) -> None:
        """Log AI positions that the stop-loss / trailing-stop path closed on its own."""
        live = self._ai_entries()
        for oid in [o for o in self._open if o not in live]:
            sym = self._open.pop(oid)
            self._targets.pop(oid, None)
            self._trail.pop(oid, None)
            if self._swing_meta.pop(oid, None) is not None:
                self._save_swing()
            self._record("close", f"[auto] {sym} exited by stop-loss / trailing stop")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ai_entries(self) -> dict:
        from agents.base_agent import _tsl_sl_orders
        return {oid: e for oid, e in _tsl_sl_orders.items() if e.get("agent") == AGENT}

    @staticmethod
    def _kind(e: dict) -> str:
        """equity | option | future | swing"""
        return e.get("kind") or ("option" if e.get("exchange") == "NFO" else "equity")

    def _intraday_entries(self) -> dict:
        return {o: e for o, e in self._ai_entries().items() if self._kind(e) != "swing"}

    def _intraday_count(self) -> int:
        return len(self._intraday_entries()) + len(self._spreads)

    def _swing_entries(self) -> dict:
        """Swing positions the AI manages (momentum rotation positions are tracked separately)."""
        return {o: e for o, e in self._ai_entries().items()
                if self._kind(e) == "swing" and not self._is_momentum(o)}

    def _overnight_entries(self) -> dict:
        """Everything held past the close: swing buys, swing puts and momentum names."""
        return {o: e for o, e in self._ai_entries().items() if self._kind(e) == "swing"}

    def _momentum_entries(self) -> dict:
        return {o: e for o, e in self._ai_entries().items() if self._is_momentum(o)}

    @staticmethod
    def _is_momentum(oid: str) -> bool:
        pos = trailing_sl_engine._positions.get(oid)
        return bool(pos and pos.strategy == MOMENTUM_STRATEGY)

    async def run_momentum_rebalance(self, execute: bool = True) -> dict:
        """Monthly: sell momentum names that dropped out of the top N, buy the new ones (CNC,
        GTT stop). Pure code — no Claude call. Returns the plan and what was done."""
        import momentum_rotation
        held = {}
        for oid, e in self._momentum_entries().items():
            pos = trailing_sl_engine._positions.get(oid)
            held[e["symbol"]] = pos.entry_price if pos else 0.0
        plan = await asyncio.to_thread(momentum_rotation.rebalance_plan, held)
        done = {"sold": [], "bought": [], "kept": plan["keep"], "rejected": {}}
        if not execute or not settings.momentum_enabled:
            return {"plan": plan, "executed": False, "done": done}
        ok, why = performance.is_allowed(MOMENTUM_STRATEGY)
        if not ok:
            return {"plan": plan, "executed": False, "done": done, "blocked": why}
        for sym in plan["sell"]:
            async with self._action_lock:
                try:
                    await self._close_position(sym, "momentum rebalance: dropped out of the top list", by_code=True)
                    done["sold"].append(sym)
                except ToolRejected as exc:
                    done["rejected"][sym] = str(exc)
        for oid, e in self._momentum_entries().items():              # kept names: extend the hold
            meta = self._swing_meta.get(oid)
            if meta and e["symbol"] in plan["keep"]:
                meta["hold_until"] = self._add_sessions(now_ist().date(), MOMENTUM_HOLD_SESSIONS).isoformat()
        for r in plan["buy"]:
            sym, px = r["symbol"], r["last"]
            tick, _ = tick_engine.latest(sym)
            if not tick or tick.ltp <= 0:
                tick_engine.subscribe([{"symbol": sym, "exchange": "NSE", "seed_price": px}])
                tick, _ = tick_engine.latest(sym)
            ltp = float(tick.ltp) if tick and tick.ltp > 0 else px
            qty = int(plan["capital_per_name"] // ltp)
            if qty < 1:
                done["rejected"][sym] = f"₹{plan['capital_per_name']:,.0f} buys less than one share at ₹{ltp:.2f}"
                continue
            args = {"symbol": sym, "quantity": qty, "stop_loss": round(ltp * (1 - settings.momentum_stop_pct / 100), 2),
                    "target": round(ltp * 1.5, 2), "max_hold_days": MOMENTUM_HOLD_SESSIONS,
                    "reasoning": f"momentum rank {r['rank']}: 12-1 return {r['ret_12_1']:+.1f}%, 6-1 {r['ret_6_1']:+.1f}%"}
            async with self._action_lock:
                try:
                    await self._place_swing_trade(args, _strategy=MOMENTUM_STRATEGY)
                    done["bought"].append(sym)
                except ToolRejected as exc:
                    done["rejected"][sym] = str(exc)
        self._save_swing()
        self._record("cycle", f"momentum rebalance: sold {done['sold'] or '-'}, bought {done['bought'] or '-'}, "
                              f"kept {done['kept'] or '-'}" + (f", rejected {done['rejected']}" if done["rejected"] else ""))
        return {"plan": plan, "executed": True, "done": done}

    @staticmethod
    def _sessions_between(a: date, b: date) -> int:
        """Weekdays after `a` up to and including `b` (exchange holidays not excluded)."""
        n, d = 0, a
        while d < b:
            d += timedelta(days=1)
            n += d.weekday() < 5
        return n

    @staticmethod
    def _add_sessions(a: date, n: int) -> date:
        d = a
        while n > 0:
            d += timedelta(days=1)
            n -= d.weekday() < 5
        return d

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

    def _trading_allowed(self, swing: bool = False) -> tuple[bool, str]:
        if settings.trading_mode == "LIVE" and not settings.ai_agent_allow_live:
            return False, "server is LIVE and AI_AGENT_ALLOW_LIVE is false — the AI may not place live orders"
        t = now_ist().time()
        if not is_market_open():
            return False, "market is closed"
        end = SWING_ENTRY_END if swing else ENTRY_END
        if not (ENTRY_START <= t <= end):
            return False, f"outside the AI entry window {ENTRY_START:%H:%M}-{end:%H:%M} IST"
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
            "product":              "MIS (NSE equity intraday); intraday limits above exclude swing",
            "swing": {
                "enabled":           settings.ai_agent_swing_enabled,
                "product":           "CNC delivery, long only, GTT stop at the broker",
                "max_positions":     settings.ai_agent_max_swing_positions,
                "open_positions":    len(self._swing_entries()),
                "max_value_per_trade": settings.ai_agent_max_swing_value,
                "stop_loss_pct_range": [settings.ai_agent_swing_min_sl_pct, settings.ai_agent_swing_max_sl_pct],
                "min_reward_to_risk": settings.ai_agent_swing_min_rr,
                "max_hold_days":     settings.ai_agent_swing_max_hold_days,
                "entry_window":      f"{ENTRY_START:%H:%M}-{SWING_ENTRY_END:%H:%M}",
                "daily_scan_at":     settings.ai_agent_swing_scan_time,
            },
            "swing_puts": {
                "enabled":           settings.ai_agent_swing_enabled and settings.ai_agent_swing_puts_enabled,
                "product":           "NRML long put, held for days, GTT stop; counts toward swing positions",
                "max_premium_per_trade": settings.ai_agent_max_swing_put_premium,
                "stop_pct_of_premium": [settings.ai_agent_swing_put_min_sl_pct, settings.ai_agent_swing_put_max_sl_pct],
                "expiry_after_exit_days": settings.ai_agent_swing_put_expiry_buffer_days,
            },
            "option_selling": {
                "enabled":           settings.ai_agent_option_selling_enabled,
                "structure":         "credit spreads only (sell 1 + buy 1 further OTM), intraday",
                "max_lots":          settings.ai_agent_max_spread_lots,
                "max_worst_case_loss": settings.ai_agent_max_spread_loss,
                "max_margin":        settings.ai_agent_max_spread_margin,
                "min_credit_pct_of_width": settings.ai_agent_spread_min_credit_pct,
                "open_spreads":      len(self._spreads),
            },
            "futures": {
                "enabled":           settings.ai_agent_futures_enabled,
                "underlyings":       self._future_underlyings(),
                "intraday_only":     True,
                "max_lots":          settings.ai_agent_max_future_lots,
                "max_loss_at_stop":  settings.ai_agent_max_future_risk,
                "max_margin_per_trade": settings.ai_agent_max_future_margin,
                "stop_loss_pct_range": [settings.ai_agent_future_min_sl_pct, settings.ai_agent_future_max_sl_pct],
                "min_reward_to_risk": 1.5,
                "entry_window":      f"{OPT_ENTRY_START:%H:%M}-{OPT_ENTRY_END:%H:%M}",
            },
            "options": {
                "enabled":           settings.ai_agent_options_enabled,
                "underlyings":       self._option_underlyings(),
                "buy_only":          True,
                "max_premium_per_trade": settings.ai_agent_max_option_premium,
                "max_lots":          settings.ai_agent_max_option_lots,
                "stop_pct_of_premium": [settings.ai_agent_option_min_sl_pct, settings.ai_agent_option_max_sl_pct],
                "entry_window":      f"{OPT_ENTRY_START:%H:%M}-{OPT_ENTRY_END:%H:%M}",
                "max_iv_rank":       settings.ai_agent_option_max_iv_rank,
            },
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
