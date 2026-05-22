"""
kite_client.py  (v5 — Kite API limit-aware)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zerodha Kite used EXCLUSIVELY for orders + portfolio.
Market data comes from NSE India API + yfinance (market_data.py).

Kite Connect API limits enforced here
──────────────────────────────────────
  Rate limits
    • REST API (general)  : 10 req/sec  → token bucket
    • Order placement     : 10 req/sec  → shared bucket
    • Historical data     : 3 req/sec   → separate bucket
  Retry / resilience
    • NetworkException    : up to 4 retries, exponential backoff (1s→2s→4s→8s)
    • 429 / rate-limit    : wait until bucket refills, then retry
    • TokenException      : no retry — surface immediately (token must be refreshed)
  Order constraints
    • Tag                 : max 20 chars (Kite hard limit) — auto-truncated
    • Quantity            : must be > 0
    • F&O lot-size        : quantity must be multiple of instrument lot size
    • Margin check        : pre-flight check in LIVE mode before placing entry
  Historical data
    • Minute candles      : max 60-day window per request — auto-chunked
    • Day candles         : max 2 000-day window per request
"""
from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional

from kiteconnect import KiteConnect
from kiteconnect.exceptions import (
    NetworkException, TokenException, InputException,
    OrderException, DataException, GeneralException,
)
from loguru import logger

from config import settings


# ── Kite-documented limits ─────────────────────────────────────────────────
_KITE_ORDER_TAG_MAX   = 20      # chars
_KITE_REST_RPS        = 10      # requests/second (general + orders)
_KITE_HIST_RPS        = 3       # requests/second (historical data)
_KITE_HIST_MIN_DAYS   = 60      # max days per minute-candle request
_KITE_HIST_DAY_DAYS   = 2_000   # max days per day-candle request
_RETRY_MAX            = 4
_RETRY_BASE_SEC       = 1.0     # doubles each attempt: 1 2 4 8


# ── F&O lot sizes (NSE — as of 2025, update after each expiry revision) ────
_FON_LOT_SIZES: dict[str, int] = {
    "NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 65, "MIDCPNIFTY": 120,
    "RELIANCE": 250, "TCS": 150, "INFY": 300, "HDFCBANK": 550,
    "ICICIBANK": 700, "SBIN": 1500, "AXISBANK": 1200, "KOTAKBANK": 400,
    "LT": 300, "WIPRO": 1500, "BAJFINANCE": 125, "MARUTI": 100,
    "TATAMOTORS": 1425, "HINDUNILVR": 300, "SUNPHARMA": 700,
    "DRREDDY": 125, "CIPLA": 650, "ONGC": 1925, "TATASTEEL": 5500,
    "JSWSTEEL": 1350, "ADANIPORTS": 625, "TITAN": 375,
}


# ── Token-bucket rate limiter ──────────────────────────────────────────────

class _TokenBucket:
    """Thread-safe token bucket for rate limiting."""

    def __init__(self, rate: float) -> None:
        self._rate     = rate          # tokens per second
        self._tokens   = rate          # start full
        self._last     = time.monotonic()
        self._lock     = Lock()

    def acquire(self) -> None:
        with self._lock:
            now   = time.monotonic()
            delta = now - self._last
            self._tokens = min(self._rate, self._tokens + delta * self._rate)
            self._last   = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rate
                time.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


_rest_bucket = _TokenBucket(_KITE_REST_RPS)
_hist_bucket = _TokenBucket(_KITE_HIST_RPS)


# ── Retry helper ───────────────────────────────────────────────────────────

def _with_retry(fn, bucket: _TokenBucket = _rest_bucket, label: str = ""):
    """
    Call fn() with rate-limiting and exponential-backoff retry.
    TokenException is re-raised immediately (token must be refreshed externally).
    """
    delay = _RETRY_BASE_SEC
    for attempt in range(_RETRY_MAX + 1):
        bucket.acquire()
        try:
            return fn()
        except TokenException:
            raise
        except (NetworkException, DataException, GeneralException) as exc:
            if attempt == _RETRY_MAX:
                raise
            logger.warning("[kite{}] {} — retry {}/{} in {:.0f}s",
                           f"/{label}" if label else "", exc, attempt + 1, _RETRY_MAX, delay)
            time.sleep(delay)
            delay *= 2
        except InputException as exc:
            raise   # bad input — don't retry


# ── Main client ────────────────────────────────────────────────────────────

class KiteClient:
    """
    Thin wrapper around KiteConnect — orders and portfolio only.
    PAPER mode: every mutating call is simulated in memory.
    LIVE mode : all Kite API limits are enforced.
    """

    def __init__(self) -> None:
        self._kite: Optional[KiteConnect] = None
        self._paper_orders:    list[dict] = []
        self._paper_positions: list[dict] = []
        self._instruments_cache: dict[str, list[dict]] = {}

    # ── Auth ───────────────────────────────────────────────────────────────

    def login_url(self) -> str:
        return KiteConnect(api_key=settings.kite_api_key).login_url()

    def set_access_token(
        self,
        request_token: Optional[str] = None,
        access_token:  Optional[str] = None,
    ) -> str:
        kite = KiteConnect(api_key=settings.kite_api_key)
        if request_token:
            data  = kite.generate_session(request_token,
                                          api_secret=settings.kite_api_secret)
            token = data["access_token"]
        else:
            token = access_token or settings.kite_access_token
        kite.set_access_token(token)
        self._kite = kite
        logger.info("Kite auth OK (orders-only, rate-limited mode)")
        return token

    @property
    def kite(self) -> KiteConnect:
        if self._kite is None:
            raise RuntimeError(
                "KiteClient not initialised — call set_access_token() first."
            )
        return self._kite

    # ── Instrument lookup ──────────────────────────────────────────────────

    def get_instruments(self, exchange: str = "NSE") -> list[dict]:
        """Fetch and cache all instruments for the given exchange."""
        if exchange in self._instruments_cache:
            return self._instruments_cache[exchange]
        if settings.trading_mode == "PAPER":
            return []
        try:
            instruments = _with_retry(lambda: self.kite.instruments(exchange),
                                      label="instruments")
            self._instruments_cache[exchange] = instruments
            logger.info("[kite] Loaded {} instruments for {}", len(instruments), exchange)
            return instruments
        except Exception as exc:
            logger.warning("[kite] instruments fetch failed: {}", exc)
            return []

    def get_instrument_tokens(self, symbols: list[str], exchange: str = "NSE") -> dict[str, int]:
        """Return {symbol: instrument_token} for the given symbols."""
        instruments = self.get_instruments(exchange)
        token_map: dict[str, int] = {}
        for inst in instruments:
            sym = inst.get("tradingsymbol", "")
            if sym in symbols:
                token_map[sym] = int(inst["instrument_token"])
        missing = set(symbols) - set(token_map)
        if missing:
            logger.warning("[kite] No instrument tokens found for: {}", missing)
        return token_map

    def setup_ticker(
        self,
        token_to_symbol: dict[int, str],
        on_tick_cb,
        on_connect_cb=None,
        on_error_cb=None,
        on_close_cb=None,
    ):
        """Create and return a KiteTicker wired to the given callbacks.
        Caller is responsible for starting it in a daemon thread."""
        from kiteconnect import KiteTicker
        tokens = list(token_to_symbol.keys())
        access_token = (
            settings.kite_access_token
            or (getattr(self._kite, "access_token", "") if self._kite else "")
        )
        ticker = KiteTicker(api_key=settings.kite_api_key, access_token=access_token)

        def _on_connect(ws, response):
            logger.info("[KiteTicker] Connected — subscribing {} tokens", len(tokens))
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            if on_connect_cb:
                on_connect_cb(ws, response)

        def _on_error(ws, code, reason):
            logger.warning("[KiteTicker] Error {}: {}", code, reason)
            if on_error_cb:
                on_error_cb(ws, code, reason)

        def _on_close(ws, code, reason):
            logger.info("[KiteTicker] Closed {}: {}", code, reason)
            if on_close_cb:
                on_close_cb(ws, code, reason)

        ticker.on_ticks = on_tick_cb
        ticker.on_connect = _on_connect
        ticker.on_error = _on_error
        ticker.on_close = _on_close
        return ticker

    def validate_credentials(self) -> dict:
        """Return status of all required credentials and Kite initialisation."""
        return {
            "kite_api_key":       bool(settings.kite_api_key),
            "kite_api_secret":    bool(settings.kite_api_secret),
            "kite_access_token":  bool(settings.kite_access_token),
            "kite_initialised":   self._kite is not None,
            "anthropic_api_key":  bool(settings.anthropic_api_key),
            "telegram_bot_token": bool(settings.telegram_bot_token),
            "api_key_set":        bool(settings.api_key),
            "trading_mode":       settings.trading_mode,
        }

    # ── Portfolio (read-only) ──────────────────────────────────────────────

    def positions(self) -> dict:
        if settings.trading_mode == "PAPER":
            return {"net": self._paper_positions, "day": self._paper_positions}
        return _with_retry(self.kite.positions, label="positions")

    def holdings(self) -> list[dict]:
        if settings.trading_mode == "PAPER":
            return []
        return _with_retry(self.kite.holdings, label="holdings")

    def orders(self) -> list[dict]:
        if settings.trading_mode == "PAPER":
            return self._paper_orders
        return _with_retry(self.kite.orders, label="orders")

    def order_history(self, order_id: str) -> list[dict]:
        if settings.trading_mode == "PAPER":
            return [o for o in self._paper_orders if o["order_id"] == order_id]
        return _with_retry(
            lambda: self.kite.order_history(order_id), label="order_history"
        )

    def margins(self) -> dict:
        """Live margin snapshot — used for pre-flight order check."""
        if settings.trading_mode == "PAPER":
            return {"equity": {"available": {"live_balance": settings.max_position_size * 5}}}
        return _with_retry(self.kite.margins, label="margins")

    def quote_kite(self, instruments: list[str]) -> dict[str, dict]:
        """Batch live quotes from Kite. instruments = ['NSE:RELIANCE', 'NFO:NIFTY...'].
        Returns Kite's quote dict keyed by 'EXCHANGE:SYMBOL'. Empty dict in PAPER mode."""
        if settings.trading_mode == "PAPER" or not instruments:
            return {}
        return _with_retry(lambda: self.kite.quote(instruments), label="quote")

    # ── Order placement ────────────────────────────────────────────────────

    def place_order(
        self,
        tradingsymbol:    str,
        exchange:         str,
        transaction_type: str,
        quantity:         int,
        order_type:       str   = "MARKET",
        product:          str   = "MIS",
        price:            float = 0.0,
        trigger_price:    float = 0.0,
        validity:         str   = "DAY",
        tag:              str   = "AlgoTraderPro",
    ) -> str:
        tag      = tag[:_KITE_ORDER_TAG_MAX]          # enforce 20-char limit
        quantity = self._validated_quantity(tradingsymbol, exchange, product, quantity)

        if settings.trading_mode == "PAPER":
            return self._paper_place(tradingsymbol, exchange, transaction_type,
                                     quantity, order_type, product, price,
                                     trigger_price, tag)
        # Pre-flight margin check for BUY entries
        if transaction_type == "BUY" and order_type in ("MARKET", "LIMIT"):
            self._check_margin(tradingsymbol, quantity, price)

        def _place():
            return self.kite.place_order(
                variety=KiteConnect.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type,
                price=price or None,
                trigger_price=trigger_price or None,
                validity=validity,
                tag=tag,
            )

        order_id = _with_retry(_place, label="place_order")
        logger.info("LIVE order | {} {} {} qty={} @ {} | id={}",
                    transaction_type, tradingsymbol, order_type,
                    quantity, price, order_id)
        return order_id

    def modify_order(
        self, order_id: str, price: float = 0.0,
        quantity: int = 0, trigger_price: float = 0.0,
    ) -> str:
        if settings.trading_mode == "PAPER":
            for o in self._paper_orders:
                if o["order_id"] == order_id:
                    if price:         o["price"]         = price
                    if quantity:      o["quantity"]       = quantity
                    if trigger_price: o["trigger_price"]  = trigger_price
            return order_id
        return _with_retry(
            lambda: self.kite.modify_order(
                variety=KiteConnect.VARIETY_REGULAR, order_id=order_id,
                price=price or None, quantity=quantity or None,
                trigger_price=trigger_price or None,
            ),
            label="modify_order",
        )

    def cancel_order(self, order_id: str) -> str:
        if settings.trading_mode == "PAPER":
            for o in self._paper_orders:
                if o["order_id"] == order_id:
                    o["status"] = "CANCELLED"
            return order_id
        return _with_retry(
            lambda: self.kite.cancel_order(
                variety=KiteConnect.VARIETY_REGULAR, order_id=order_id
            ),
            label="cancel_order",
        )

    def squareoff_all_positions(self) -> list[str]:
        order_ids: list[str] = []
        for pos in self.positions().get("net", []):
            if pos.get("quantity", 0) == 0:
                continue
            side = "SELL" if pos["quantity"] > 0 else "BUY"
            qty  = abs(pos["quantity"])
            oid  = self.place_order(
                tradingsymbol=pos["tradingsymbol"],
                exchange=pos.get("exchange", "NSE"),
                transaction_type=side,
                quantity=qty,
                order_type="MARKET",
                product=pos.get("product", "MIS"),
                tag="SquareOff",
            )
            order_ids.append(oid)
            logger.info("Square-off {} {} qty={}", side, pos["tradingsymbol"], qty)
        return order_ids

    # ── Historical data (rate-limited + auto-chunked) ──────────────────────

    def historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day",          # "minute" | "day" | "5minute" etc.
        continuous: bool = False,
        oi: bool = False,
    ) -> list[dict]:
        """
        Fetch OHLCV data respecting Kite's window limits:
          minute data → 60-day chunks
          day data    → 2 000-day chunks
        Chunks are merged and returned as a single list.
        """
        if settings.trading_mode == "PAPER":
            return []

        is_minute = "minute" in interval
        max_days  = _KITE_HIST_MIN_DAYS if is_minute else _KITE_HIST_DAY_DAYS
        window    = timedelta(days=max_days)
        records: list[dict] = []
        chunk_start = from_date

        while chunk_start < to_date:
            chunk_end = min(chunk_start + window, to_date)

            def _fetch(cs=chunk_start, ce=chunk_end):
                return self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=cs, to_date=ce,
                    interval=interval, continuous=continuous, oi=oi,
                )

            try:
                chunk = _with_retry(_fetch, bucket=_hist_bucket,
                                    label="historical_data")
                records.extend(chunk)
            except Exception as exc:
                logger.warning("[kite/historical] chunk {}-{} failed: {}",
                               chunk_start.date(), chunk_end.date(), exc)

            chunk_start = chunk_end + timedelta(days=1)

        logger.debug("[kite/historical] {} bars fetched for token={} ({} → {})",
                     len(records), instrument_token,
                     from_date.date(), to_date.date())
        return records

    # ── Internal helpers ───────────────────────────────────────────────────

    def _validated_quantity(
        self, symbol: str, exchange: str, product: str, quantity: int
    ) -> int:
        """
        Enforce quantity > 0.
        For F&O products (NRML) snap to the nearest lot-size multiple.
        """
        if quantity <= 0:
            raise InputException(f"Invalid quantity {quantity} for {symbol}")

        if product == "NRML" or exchange in ("NFO", "BFO", "CDS", "MCX"):
            lot = _FON_LOT_SIZES.get(symbol)
            if lot and quantity % lot != 0:
                snapped = max(lot, math.ceil(quantity / lot) * lot)
                logger.warning(
                    "[kite] {} qty={} not a multiple of lot {}  → snapped to {}",
                    symbol, quantity, lot, snapped,
                )
                return snapped
        return quantity

    def _check_margin(self, symbol: str, quantity: int, price: float) -> None:
        """
        Warn (not block) if estimated order value exceeds available live balance.
        Only called in LIVE mode for BUY orders.
        """
        try:
            m         = self.margins()
            available = m.get("equity", {}).get("available", {}).get("live_balance", 0)
            order_val = quantity * max(price, 1)
            if order_val > available:
                logger.warning(
                    "[kite] Margin warning: order ₹{:,.0f} > available ₹{:,.0f} for {}",
                    order_val, available, symbol,
                )
        except Exception as exc:
            logger.warning("[kite] Margin check failed (non-blocking): {}", exc)

    # ── Paper trading ──────────────────────────────────────────────────────

    def _paper_place(
        self, tradingsymbol, exchange, transaction_type,
        quantity, order_type, product, price, trigger_price, tag,
    ) -> str:
        order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        record = {
            "order_id":         order_id,
            "tradingsymbol":    tradingsymbol,
            "exchange":         exchange,
            "transaction_type": transaction_type,
            "quantity":         quantity,
            "order_type":       order_type,
            "product":          product,
            "price":            price,
            "trigger_price":    trigger_price,
            "status":           "COMPLETE",
            "tag":              tag,
            "placed_at":        datetime.now().isoformat(),
        }
        self._paper_orders.append(record)
        self._update_paper_position(record)
        logger.info("[PAPER] {} {} {} qty={} @ ₹{} | id={}",
                    transaction_type, tradingsymbol, order_type,
                    quantity, price, order_id)
        return order_id

    def _update_paper_position(self, order: dict) -> None:
        sym       = order["tradingsymbol"]
        qty_delta = (order["quantity"] if order["transaction_type"] == "BUY"
                     else -order["quantity"])
        for pos in self._paper_positions:
            if pos["tradingsymbol"] == sym:
                pos["quantity"] += qty_delta
                return
        self._paper_positions.append({
            "tradingsymbol": sym,
            "exchange":      order["exchange"],
            "product":       order["product"],
            "quantity":      qty_delta,
            "average_price": order["price"],
            "last_price":    order["price"],
            "pnl":           0.0,
        })


kite_client = KiteClient()
