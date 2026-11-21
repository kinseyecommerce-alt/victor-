"""
tick_engine.py — the live price feed (Kite only).

Keeps the newest quote (LTP, bid, ask, day OHLC) of every subscribed instrument:
  * Kite WebSocket (kite_ticker.py) for NSE / BSE equities and indices when a Kite session exists;
  * one Kite REST quote batch per second for everything the socket does not cover (and every 60 s while the
    market is closed, so the dashboard shows the last traded prices).

Used by:
  * the paper broker (kite_client._paper_ltp / _paper_touch): a marketable paper order fills at the newest tick's
    touch when the feed has one for the symbol, exactly as before the rebuild;
  * the paper book's marks (kite_client._mark_paper_positions) and the dashboard's price strip.

No candles, no indicators, no strategy queues: the legacy tick agents that consumed them were removed. The validated
agents never read this feed for their decisions (they read validated_data's Kite quotes and stored minute sessions).
PAPER with PAPER_DATA_SOURCE=SIM has no feed at all.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import is_market_open
from kite_client import kite_client


@dataclass
class Tick:
    symbol: str
    ltp: float
    bid: float
    ask: float
    volume: int
    change: float
    change_pct: float
    high: float
    low: float
    open: float
    timestamp: datetime


def tick_from_kite_quote(symbol: str, data: dict) -> Tick:
    ohlc = data.get("ohlc", {}) or {}
    depth = data.get("depth", {}) or {}
    buys = depth.get("buy") or [{}]
    sells = depth.get("sell") or [{}]
    ltp = float(data.get("last_price", 0.0) or 0.0)
    return Tick(symbol=symbol, ltp=ltp,
                bid=float(buys[0].get("price", ltp) if buys else ltp),
                ask=float(sells[0].get("price", ltp) if sells else ltp),
                volume=int(data.get("volume_traded", 0) or 0),
                change=float(data.get("change", 0.0) or 0.0), change_pct=float(data.get("change", 0.0) or 0.0),
                high=float(ohlc.get("high", ltp)), low=float(ohlc.get("low", ltp)), open=float(ohlc.get("open", ltp)),
                timestamp=datetime.now())


# Names people (and old lists) use that are not the NSE trading symbol Kite expects
SYMBOL_ALIASES = {
    "INFOSYS": "INFY", "ADANITRANS": "ADANIENSOL", "ZOMATO": "ETERNAL", "MCDOWELL-N": "UNITDSPR",
    "NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK",
}

# The warm watchlist when AUTO_START_WATCHLIST is empty: the first 20 NIFTY 50 names (as before the rebuild) and
# the two index quotes
NIFTY_50_HEAD = ["RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK", "INFY", "SBIN", "HINDUNILVR", "ITC",
                 "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "ASIANPAINT", "HCLTECH", "WIPRO", "NESTLEIND",
                 "ADANIPORTS", "ULTRACEMCO"]


def canonical_symbol(symbol: str) -> str:
    """Kite's NSE trading symbol for a possibly stale or informal name."""
    s = (symbol or "").strip().upper()
    return SYMBOL_ALIASES.get(s, s)


def json_safe(obj):
    """Replace NaN / ±inf floats with None, recursively (JSON-safe API payloads)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


POSITION_MARK_S = 30.0          # seconds between two quote calls for the marks of unsubscribed open positions
POSITION_MARK_RETRY_S = 900.0    # after a FAILED marks call (expired Kite token at night / on a weekend): next try


class TickEngine:
    def __init__(self) -> None:
        self._running = False
        self._symbols: list[str] = []
        self._exchange: dict[str, str] = {}
        self._latest_tick: dict[str, Tick] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._kite_ticker = None
        self._use_ws = False
        self._ws_received: set[str] = set()
        self._last_quote_warn = 0.0
        self._no_quote: set[str] = set()
        # marks of open paper positions the watchlist does not cover (PEAD small caps, option legs): a price for the
        # paper book's unrealised P&L only. They are NOT ticks: latest() never returns them, so a paper fill still
        # uses the quote the agent fetched at that instant (the tested fill model), never a mark up to 30 s old.
        self._marks: dict[str, float] = {}
        self._marks_at = 0.0
        self._marks_failed = False     # the last marks call failed: retry every POSITION_MARK_RETRY_S, warn once

    # ── subscriptions ──
    def subscribe(self, watchlist: list[dict]) -> None:
        added = []
        for item in watchlist:
            exch = item.get("exchange", "NSE")
            sym = canonical_symbol(item["symbol"]) if exch in ("NSE", "BSE") else item["symbol"]
            if sym in self._exchange:
                continue
            self._symbols.append(sym)
            self._exchange[sym] = exch
            added.append(sym)
        if added:
            logger.info("TickEngine: subscribed {} symbols via {}", len(self._symbols), self.data_source())
        if self._running and any(self._exchange[s] in ("NSE", "BSE") for s in added):
            self._start_ws()

    def data_source(self) -> str:
        if settings.trading_mode == "LIVE":
            return "KiteConnect REST+WS"
        if settings.paper_data_source == "KITE":
            return "KiteConnect REST+WS (paper orders)"
        return "none (PAPER_DATA_SOURCE=SIM)"

    def default_watchlist(self) -> list[dict]:
        if settings.auto_start_watchlist:
            syms = [x.strip() for x in settings.auto_start_watchlist.split(",") if x.strip()]
        else:
            syms = NIFTY_50_HEAD
        return [{"symbol": x, "exchange": "NSE"} for x in syms] + \
               [{"symbol": "NIFTY 50", "exchange": "NSE"}, {"symbol": "NIFTY BANK", "exchange": "NSE"}]

    def warm_watchlist(self, reason: str = "") -> int:
        """With Kite data and a Kite session, subscribe the default watchlist when nothing is subscribed yet."""
        if not settings.uses_kite_market_data or self._symbols or not kite_client.has_session():
            return 0
        wl = self.default_watchlist()
        self.subscribe(wl)
        logger.info("TickEngine: warm watchlist of {} symbols ({})", len(wl), reason or "startup")
        return len(wl)

    # ── lifecycle ──
    def start_loop(self) -> None:
        if self._running:
            return
        self._running = True
        self._use_ws = False
        self._ws_received.clear()
        self._loop = asyncio.get_event_loop()
        self._task = asyncio.create_task(self._poll_loop())
        self._start_ws()
        logger.info("TickEngine poll loop started (ws={})", self._use_ws)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        if self._kite_ticker:
            self._kite_ticker.stop()
            self._kite_ticker = None
        self._use_ws = False

    def _start_ws(self) -> None:
        if not settings.uses_kite_market_data or not self._symbols or not self._loop:
            return
        if self._kite_ticker:
            self._kite_ticker.stop()
            self._kite_ticker = None
        self._use_ws = False
        self._ws_received.clear()
        if settings.use_kite_websocket and kite_client.has_session():
            try:
                from kite_ticker import KiteTicker
                self._kite_ticker = KiteTicker()
                eq = [s for s in self._symbols if self._exchange.get(s) in ("NSE", "BSE")]
                self._use_ws = self._kite_ticker.start(eq, self._ingest_kite_tick, self._loop)
            except Exception as exc:
                logger.error("TickEngine: KiteConnect WebSocket failed to start: {} — REST polling", exc)
                self._use_ws = False

    def on_kite_login(self) -> None:
        """After a Kite login (possibly from another thread): subscribe the watchlist, restart the socket."""
        def _apply() -> None:
            try:
                if not self._running:
                    self.start_loop()
                self.warm_watchlist("kite login")
                if self._symbols:
                    self._start_ws()
                    asyncio.ensure_future(self._fetch_kite_batch(idle=not is_market_open()))
            except Exception as exc:
                logger.warning("TickEngine: post-login refresh failed: {}", exc)
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(_apply)
        else:
            _apply()

    # ── ingest ──
    async def _ingest_kite_tick(self, symbol: str, tick: Tick) -> None:
        self._ws_received.add(symbol)
        self._latest_tick[symbol] = tick

    async def _poll_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            if not settings.uses_kite_market_data:
                await asyncio.sleep(60)
                continue
            await self._refresh_position_marks()
            if not is_market_open():
                await self._fetch_kite_batch(idle=True)
                await asyncio.sleep(60)
                continue
            await self._fetch_kite_batch()
            await asyncio.sleep(max(0.0, 1.0 - (time.monotonic() - t0)))

    async def _refresh_position_marks(self, force: bool = False) -> int:
        """One Kite quote() call, at most every POSITION_MARK_S, for the open paper positions that have no
        subscription. Never raises; returns the number of marks stored."""
        wait = POSITION_MARK_RETRY_S if self._marks_failed else POSITION_MARK_S
        if not force and time.monotonic() - self._marks_at < wait:
            return 0
        self._marks_at = time.monotonic()
        try:
            return await self._fetch_position_marks()
        finally:
            try:                       # write the newest prices into the paper book's rows (last_price / pnl only):
                kite_client._mark_paper_positions()        # nothing else did this between two dashboard requests
            except Exception as exc:
                logger.debug("[tick] paper book marking failed: {}", exc)

    async def _fetch_position_marks(self) -> int:
        try:
            rows = [p for p in kite_client._paper_positions
                    if p.get("quantity") and p.get("tradingsymbol") not in self._exchange]
            keep = {p["tradingsymbol"] for p in rows}
            for gone in [k for k in self._marks if k not in keep]:
                self._marks.pop(gone, None)
            if not rows or not kite_client.has_session():
                return 0
            ins = sorted({f"{p.get('exchange') or 'NSE'}:{p['tradingsymbol']}" for p in rows})
            raw = await asyncio.get_event_loop().run_in_executor(None, lambda: kite_client.quote_kite(ins))
            n = 0
            for p in rows:
                q = (raw or {}).get(f"{p.get('exchange') or 'NSE'}:{p['tradingsymbol']}") or {}
                px = float(q.get("last_price") or 0)
                if px > 0:
                    self._marks[p["tradingsymbol"]] = px
                    n += 1
            if self._marks_failed:
                self._marks_failed = False
                logger.info("[tick] position marks are back ({} marks)", n)
            return n
        except Exception as exc:
            if not self._marks_failed:                 # one line per outage, then a retry every 15 minutes
                self._marks_failed = True
                logger.warning("[tick] position marks failed (next try in {:.0f} min): {}",
                               POSITION_MARK_RETRY_S / 60.0, exc)
            return 0

    def mark(self, symbol: str) -> float:
        """The newest mark of an open paper position without a subscription (0.0 when there is none)."""
        return float(self._marks.get(symbol) or 0.0)

    async def _fetch_kite_batch(self, idle: bool = False) -> None:
        """One Kite quote() call for every subscribed symbol the WebSocket is not covering."""
        ws_live = self._use_ws and getattr(self._kite_ticker, "is_connected", True)
        pending = [s for s in self._symbols if idle or not (ws_live and s in self._ws_received)]
        if not pending or not kite_client.has_session():
            return
        instruments = [f"{self._exchange.get(s, 'NSE')}:{s}" for s in pending]
        try:
            raw = await asyncio.get_event_loop().run_in_executor(None, lambda: kite_client.quote_kite(instruments))
        except Exception as exc:
            if time.monotonic() - self._last_quote_warn > 60:
                self._last_quote_warn = time.monotonic()
                logger.warning("[tick] Kite batch quote failed: {}", exc)
            return
        for sym in pending:
            data = raw.get(f"{self._exchange.get(sym, 'NSE')}:{sym}")
            if not data:
                if sym not in self._no_quote:
                    self._no_quote.add(sym)
                    logger.warning("[tick] Kite has no quote for {}", sym)
                continue
            self._no_quote.discard(sym)
            self._latest_tick[sym] = tick_from_kite_quote(sym, data)

    # ── queries ──
    def latest(self, symbol: str) -> tuple[Optional[Tick], None]:
        """(newest tick or None, None). The second element is kept for callers written for the old engine."""
        return self._latest_tick.get(symbol), None

    def seed_quote(self, symbol: str, ltp: float, bid: float = 0.0, ask: float = 0.0) -> None:
        """Remember a quote just fetched for a symbol that has no tick yet (a real tick replaces it)."""
        if ltp <= 0 or symbol in self._latest_tick:
            return
        self._latest_tick[symbol] = Tick(symbol=symbol, ltp=float(ltp), bid=float(bid or 0), ask=float(ask or 0),
                                         volume=0, change=0.0, change_pct=0.0, high=float(ltp), low=float(ltp),
                                         open=float(ltp), timestamp=datetime.now())

    def all_latest(self) -> dict[str, dict]:
        return json_safe({s: {"ltp": t.ltp, "bid": t.bid, "ask": t.ask, "change_pct": round(t.change_pct, 2),
                              "open": t.open, "day_high": t.high, "day_low": t.low, "volume": t.volume,
                              "ts": t.timestamp.isoformat()} for s, t in list(self._latest_tick.items())})

    def symbols(self) -> list[str]:
        return list(self._symbols)

    def status(self) -> dict:
        newest = max((t.timestamp for t in list(self._latest_tick.values())), default=None)
        age = (datetime.now() - newest).total_seconds() if newest else None
        return {"running": self._running, "source": self.data_source(), "symbols": len(self._symbols),
                "quoted": len(self._latest_tick), "websocket": bool(self._use_ws),
                "last_tick_age_s": round(age, 1) if age is not None else None,
                "unknown_symbols": sorted(self._no_quote)}


tick_engine = TickEngine()
