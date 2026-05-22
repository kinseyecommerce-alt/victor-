"""
market_data.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Market data engine — NO Kite dependency for quotes or candles.

Data sources:
  Live quotes   → NSE India official website API (free, real-time, no auth)
  OHLCV history → yfinance (Yahoo Finance, .NS / .BO suffix, free)
  Option chain  → NSE India option-chain API (free)
  Market status → NSE India market-status API (free)

Kite is used ONLY for order placement — never for market data.

Public NSE endpoints used:
  /api/quote-equity?symbol={SYMBOL}        — equity quote
  /api/quote-derivative?symbol={SYMBOL}    — F&O quote
  /api/allIndices                           — all indices live
  /api/option-chain-indices?symbol=NIFTY   — option chain
  /api/market-status                        — market open/closed

Poll cadence: every 1 second during market hours via asyncio loop.
"""
from __future__ import annotations

import asyncio
import time
import random
import math
from datetime import datetime, timedelta
from typing import Optional

import httpx
import yfinance as yf
import pandas as pd
from loguru import logger

from config import settings


# ── NSE session headers (required to avoid 401 from NSE) ────────────────────
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}

NSE_BASE     = "https://www.nseindia.com"
NSE_HOME     = NSE_BASE + "/"                               # cookie handshake
QUOTE_EQ     = NSE_BASE + "/api/quote-equity?symbol={}"
QUOTE_DERIV  = NSE_BASE + "/api/quote-derivative?symbol={}"
ALL_INDICES  = NSE_BASE + "/api/allIndices"
OPT_CHAIN_I  = NSE_BASE + "/api/option-chain-indices?symbol={}"
OPT_CHAIN_EQ = NSE_BASE + "/api/option-chain-equities?symbol={}"
MKT_STATUS   = NSE_BASE + "/api/market-status"


# ── Live quote dataclass ───────────────────────────────────────────────────
class Quote:
    __slots__ = ("symbol", "ltp", "open", "high", "low", "prev_close",
                 "change", "change_pct", "volume", "bid", "ask",
                 "total_buy_qty", "total_sell_qty", "ts")

    def __init__(self, symbol: str, ltp: float, open_: float, high: float,
                 low: float, prev_close: float, change: float, change_pct: float,
                 volume: int, bid: float, ask: float,
                 total_buy_qty: int = 0, total_sell_qty: int = 0):
        self.symbol        = symbol
        self.ltp           = ltp
        self.open          = open_
        self.high          = high
        self.low           = low
        self.prev_close    = prev_close
        self.change        = change
        self.change_pct    = change_pct
        self.volume        = volume
        self.bid           = bid
        self.ask           = ask
        self.total_buy_qty = total_buy_qty
        self.total_sell_qty= total_sell_qty
        self.ts            = datetime.now()

    def to_dict(self) -> dict:
        return {
            "symbol":        self.symbol,
            "ltp":           self.ltp,
            "open":          self.open,
            "high":          self.high,
            "low":           self.low,
            "prev_close":    self.prev_close,
            "change":        round(self.change, 2),
            "change_pct":    round(self.change_pct, 2),
            "volume":        self.volume,
            "bid":           self.bid,
            "ask":           self.ask,
            "spread":        round(self.ask - self.bid, 2),
            "ts":            self.ts.isoformat(),
        }


# ── NSE India client ──────────────────────────────────────────────────
class NSEClient:
    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._session_ok = False
        self._last_session = 0.0
        # Rate-limit: cap at 8 req/s per NSE Circular 54/2024 (10 OPS limit)
        self._last_req_ts: float = 0.0
        self._MIN_INTERVAL: float = 0.125

    async def _ensure_session(self) -> None:
        now = time.time()
        if self._session_ok and (now - self._last_session) < 300:
            return
        if self._client:
            await self._client.aclose()
        self._client = httpx.AsyncClient(
            headers=NSE_HEADERS, timeout=10, follow_redirects=True,
        )
        try:
            await self._client.get(NSE_HOME)
            self._session_ok  = True
            self._last_session = now
        except Exception as exc:
            logger.warning("NSE session init failed: {}", exc)
            self._session_ok = False

    async def get(self, url: str) -> Optional[dict]:
        await self._ensure_session()
        # Throttle to 8 req/s
        import asyncio as _aio
        wait = self._MIN_INTERVAL - (time.monotonic() - self._last_req_ts)
        if wait > 0:
            await _aio.sleep(wait)
        self._last_req_ts = time.monotonic()
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                self._session_ok = False
            return None
        except Exception:
            return None

    async def quote_equity(self, symbol: str) -> Optional[Quote]:
        data = await self.get(QUOTE_EQ.format(symbol.upper()))
        if not data:
            return None
        try:
            pd_  = data.get("priceInfo", {})
            depth = data.get("marketDeptOrderBook", {})
            bids  = depth.get("bid", [{}])
            asks  = depth.get("ask", [{}])
            ltp   = float(pd_.get("lastPrice", 0))
            bid   = float(bids[0].get("price", ltp)) if bids else ltp
            ask   = float(asks[0].get("price", ltp)) if asks else ltp
            return Quote(
                symbol=symbol.upper(), ltp=ltp,
                open_=float(pd_.get("open", ltp)),
                high=float(pd_.get("intraDayHighLow", {}).get("max", ltp)),
                low=float(pd_.get("intraDayHighLow", {}).get("min", ltp)),
                prev_close=float(pd_.get("previousClose", ltp)),
                change=float(pd_.get("change", 0)),
                change_pct=float(pd_.get("pChange", 0)),
                volume=int(data.get("securityWiseDP", {}).get("quantityTraded", 0)),
                bid=bid, ask=ask,
                total_buy_qty=int(depth.get("totalBuyQuantity", 0)),
                total_sell_qty=int(depth.get("totalSellQuantity", 0)),
            )
        except Exception as exc:
            logger.debug("NSE quote parse error {}: {}", symbol, exc)
            return None

    async def index_quote(self, index_name: str) -> Optional[Quote]:
        data = await self.get(ALL_INDICES)
        if not data:
            return None
        for item in data.get("data", []):
            if item.get("indexSymbol", "").upper() == index_name.upper():
                ltp = float(item.get("last", 0))
                return Quote(
                    symbol=index_name, ltp=ltp,
                    open_=float(item.get("open", ltp)),
                    high=float(item.get("dayHigh", ltp)),
                    low=float(item.get("dayLow",  ltp)),
                    prev_close=float(item.get("previousClose", ltp)),
                    change=float(item.get("change", 0)),
                    change_pct=float(item.get("percentChange", 0)),
                    volume=0, bid=ltp, ask=ltp,
                )
        return None

    async def option_chain(self, symbol: str) -> Optional[dict]:
        url = OPT_CHAIN_I.format(symbol.upper())
        return await self.get(url)

    async def market_status(self) -> dict:
        data = await self.get(MKT_STATUS)
        if not data:
            return {"market_state": "unknown"}
        for mkt in data.get("marketState", []):
            if mkt.get("market") == "Capital Market":
                return {
                    "market_state": mkt.get("marketStatus", "unknown"),
                    "trade_date":   mkt.get("tradeDate", ""),
                    "index":        mkt.get("index", ""),
                }
        return data

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()


# ── yfinance historical data ────────────────────────────────────────────
class YFinanceClient:
    @staticmethod
    def _ticker(symbol: str, exchange: str = "NSE") -> str:
        suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
        return symbol + suffix

    def historical(self, symbol, exchange="NSE", interval="1m", period="5d") -> pd.DataFrame:
        from config import settings as _s
        if _s.use_truedata_historical:
            try:
                from truedata_client import truedata_historical
                lookback = {"5d": 5, "15d": 15, "30d": 30, "60d": 60, "90d": 90}.get(period, 5)
                df = truedata_historical.historical(symbol, exchange, interval, lookback)
                if not df.empty:
                    return df
            except Exception as exc:
                logger.debug("TrueData historical fallback to yfinance for {}: {}", symbol, exc)

        ticker = self._ticker(symbol, exchange)
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                    "Close": "close", "Volume": "volume"})
            df.index.name = "date"
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
            return df.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            logger.warning("yfinance error {}: {}", ticker, exc)
            return pd.DataFrame()

    def current_price(self, symbol: str, exchange: str = "NSE") -> float:
        from config import settings as _s
        if _s.use_truedata_historical:
            try:
                from truedata_client import truedata_historical
                price = truedata_historical.current_price(symbol, exchange)
                if price > 0:
                    return price
            except Exception:
                pass

        ticker = self._ticker(symbol, exchange)
        try:
            info = yf.Ticker(ticker).fast_info
            return float(info.get("last_price") or info.get("regularMarketPrice") or 0)
        except Exception:
            return 0.0


# ── Market hours helper ──────────────────────────────────────────────────
from ist_clock import is_market_open  # noqa: E402  (IST-aware; replaces datetime.now())


# ── Paper-mode tick simulator ─────────────────────────────────────────────
class PaperTickSimulator:
    def __init__(self) -> None:
        self._prices: dict[str, float] = {}
        self._yf = YFinanceClient()

    def seed(self, symbols: list[str], exchanges: dict[str, str]) -> None:
        for sym in symbols:
            exch = exchanges.get(sym, "NSE")
            price = self._yf.current_price(sym, exch)
            self._prices[sym] = price if price > 0 else 1000.0
            logger.info("Paper seed {} @ ₹{:.2f}", sym, self._prices[sym])

    def next_tick(self, symbol: str) -> Quote:
        price = self._prices.get(symbol, 1000.0)
        shock = random.gauss(0, 0.00008)
        price = max(price * math.exp(shock), 1.0)
        self._prices[symbol] = price
        spread   = round(price * 0.0002, 2)
        vol_tick = random.randint(100, 2000)
        pct      = round((price / self._prices.get(symbol + "__base__", price) - 1) * 100, 2)
        return Quote(
            symbol=symbol, ltp=round(price, 2),
            open_=round(price * 0.998, 2), high=round(price * 1.002, 2),
            low=round(price * 0.997, 2),   prev_close=round(price * 0.999, 2),
            change=round(price * shock, 2), change_pct=pct,
            volume=vol_tick,
            bid=round(price - spread / 2, 2),
            ask=round(price + spread / 2, 2),
        )


# ── Singletons ─────────────────────────────────────────────────────────────────
nse_client  = NSEClient()
yf_client   = YFinanceClient()
paper_sim   = PaperTickSimulator()