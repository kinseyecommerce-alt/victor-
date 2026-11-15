"""
kite_client.py  (v5 — Kite API limit-aware)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zerodha Kite: orders, portfolio, quotes and historical candles. PAPER mode simulates every order in memory
(_paper_orders / _paper_positions); the LIVE order path is never used by the validated agents (they refuse LIVE).

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
from ist_clock import now_ist


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
        self._paper_gtts:      dict[int, dict] = {}
        self._instruments_cache: dict[str, list[dict]] = {}
        self._instruments_day: dict[str, object] = {}

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

    def has_session(self) -> bool:
        """True if a Kite session exists or can be opened from KITE_ACCESS_TOKEN."""
        return self._kite is not None or bool(settings.kite_access_token)

    def ensure_session(self) -> KiteConnect:
        """Return the Kite session, opening one from KITE_ACCESS_TOKEN if needed."""
        if self._kite is None:
            if not settings.kite_access_token:
                raise RuntimeError("No Kite session — connect Kite or set KITE_ACCESS_TOKEN")
            self.set_access_token(access_token=settings.kite_access_token)
        return self._kite

    @property
    def kite(self) -> KiteConnect:
        if self._kite is None:
            raise RuntimeError(
                "KiteClient not initialised — call set_access_token() first."
            )
        return self._kite

    def profile(self) -> dict:
        """The logged-in Zerodha user; raises if there is no working session for today."""
        return _with_retry(lambda: self.ensure_session().profile(), label="profile")

    # ── Instrument lookup ──────────────────────────────────────────────────

    def get_instruments(self, exchange: str = "NSE") -> list[dict]:
        """Fetch and cache all instruments for the given exchange (refreshed daily —
        new F&O expiries are listed every week)."""
        today = datetime.now().date()
        cached = self._instruments_cache.get(exchange)
        if cached and self._instruments_day.get(exchange) == today:
            return cached
        if not settings.uses_kite_market_data or not self.has_session():
            return cached or []
        try:
            kite = self.ensure_session()
            instruments = _with_retry(lambda: kite.instruments(exchange),
                                      label="instruments")
            self._instruments_cache[exchange] = instruments
            self._instruments_day[exchange] = today
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

    def validate_credentials(self) -> dict:
        """Return status of all required credentials and Kite initialisation."""
        return {
            "kite_api_key":       bool(settings.kite_api_key),
            "kite_api_secret":    bool(settings.kite_api_secret),
            "kite_access_token":  bool(settings.kite_access_token),
            "kite_initialised":   self._kite is not None,
            "telegram_bot_token": bool(settings.telegram_bot_token),
            "api_key_set":        bool(settings.api_key),
            "trading_mode":       settings.trading_mode,
        }

    # ── Portfolio (read-only) ──────────────────────────────────────────────

    def positions(self) -> dict:
        if settings.trading_mode == "PAPER":
            self._mark_paper_positions()
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
        Returns Kite's quote dict keyed by 'EXCHANGE:SYMBOL'. Empty dict in PAPER mode
        unless PAPER_DATA_SOURCE=KITE (read-only market data; orders stay simulated)."""
        if not settings.uses_kite_market_data or not instruments:
            return {}
        kite = self.ensure_session()
        return _with_retry(lambda: kite.quote(instruments), label="quote")

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
        # LIVE price protection: a MARKET order goes out as a marketable LIMIT inside ORDER_PROTECTION_BAND_PCT of the
        # touch (never a raw market order into a thin book); the quote is logged next to the fill
        quote_px = 0.0
        if order_type == "MARKET":
            order_type, price, quote_px = self.protect_market_order(tradingsymbol, exchange, transaction_type)
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
        logger.info("LIVE order | {} {} {} qty={} @ {} (quote {}) | id={}",
                    transaction_type, tradingsymbol, order_type,
                    quantity, price, quote_px or "-", order_id)
        if quote_px:
            self._log_live_fill(order_id, tradingsymbol, transaction_type, quantity, quote_px, tag)
        return order_id

    @staticmethod
    def protection_limit(side: str, bid: float, ask: float, ltp: float, band_pct: float) -> tuple[float, float]:
        """(limit price, reference touch) of a marketable LIMIT: BUY at ask × (1 + band), SELL at bid × (1 − band),
        on the exchange tick; the LTP when the touch is missing. (0, 0) without any price."""
        from options_resolver import round_tick
        ref = (ask or ltp) if side == "BUY" else (bid or ltp)
        if not ref or ref <= 0:
            return 0.0, 0.0
        b = max(0.0, float(band_pct)) / 100.0
        px = ref * (1 + b) if side == "BUY" else ref * (1 - b)
        return max(0.05, round_tick(px)), float(ref)

    def protect_market_order(self, symbol: str, exchange: str, side: str) -> tuple[str, float, float]:
        """LIVE: (order_type, price, quote) for a MARKET request — a protected marketable LIMIT when Kite has a
        quote, else MARKET (an exit must still go out; logged)."""
        try:
            q = self.quote_kite([f"{exchange}:{symbol}"]).get(f"{exchange}:{symbol}") or {}
            depth = q.get("depth") or {}
            bid = float(((depth.get("buy") or [{}])[0]).get("price") or 0)
            ask = float(((depth.get("sell") or [{}])[0]).get("price") or 0)
            ltp = float(q.get("last_price") or 0)
        except Exception as exc:
            logger.warning("[kite] no quote for price protection of {} ({}) — MARKET", symbol, exc)
            return "MARKET", 0.0, 0.0
        px, ref = self.protection_limit(side, bid, ask, ltp, settings.order_protection_band_pct)
        if px <= 0:
            logger.warning("[kite] no price for protection of {} — MARKET", symbol)
            return "MARKET", 0.0, 0.0
        return "LIMIT", px, ref

    def _log_live_fill(self, order_id: str, symbol: str, side: str, qty: int, quote_px: float, tag: str) -> None:
        try:
            fill = self.fill_price(order_id)
            if fill:
                import drift_monitor
                from agents.validated_base import _agent_of
                drift_monitor.record_fill(_agent_of(tag), symbol, side, qty, quote_px, fill, "LIVE_PROTECTED")
        except Exception as exc:
            logger.debug("[kite] live fill not logged: {}", exc)

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

    def squareoff_all_positions(self, exclude: frozenset = frozenset(),
                                hold: Optional[dict] = None) -> list[str]:
        """exclude: symbols whose exit order was just sent elsewhere (a broker position can lag the fill,
        and squaring it off again would open a reverse position).
        hold: {(symbol, product): signed qty} of tracked overnight positions (base_agent.overnight_hold_qty) —
        only the quantity beyond them is flattened, row by row (symbol, product)."""
        order_ids: list[str] = []
        hold = hold or {}
        paper = settings.trading_mode == "PAPER"
        for pos in self.positions().get("net", []):
            if pos.get("quantity", 0) == 0:
                continue
            if pos.get("tradingsymbol") in exclude:
                continue
            if pos.get("product") == "CNC":       # delivery (swing) positions are held overnight
                continue
            sym = pos.get("tradingsymbol")
            held = hold.get((sym, pos.get("product", "MIS")), 0)
            if paper:
                # the paper book keeps one row per (symbol, product); a snapshot from before that kept one row
                # per symbol, so a hold whose product has no open row of its own counts against this row
                open_products = {p.get("product", "MIS") for p in self._paper_positions
                                 if p.get("tradingsymbol") == sym and p.get("quantity")}
                held += sum(q for (s, p), q in hold.items() if s == sym and p not in open_products)
            net = int(pos["quantity"]) - held
            if net == 0 or (net > 0) != (pos["quantity"] > 0):
                if held:
                    logger.info("Square-off: {} held overnight (qty {}) — left open", sym, held)
                continue
            side = "SELL" if net > 0 else "BUY"
            qty  = abs(net)
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
        if not settings.uses_kite_market_data:      # PAPER+SIM has no Kite session
            return []
        kite = self.ensure_session()

        is_minute = "minute" in interval
        max_days  = _KITE_HIST_MIN_DAYS if is_minute else _KITE_HIST_DAY_DAYS
        window    = timedelta(days=max_days)
        records: list[dict] = []
        chunk_start = from_date

        while chunk_start < to_date:
            chunk_end = min(chunk_start + window, to_date)

            def _fetch(cs=chunk_start, ce=chunk_end):
                return kite.historical_data(
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
        # Stop orders rest until cancelled — exits in paper mode are driven by the
        # software TSL / strategy rules. Filling them instantly (as before) flattened
        # every position the moment its SL-M was placed.
        resting = order_type in ("SL", "SL-M")
        ltp = self._paper_ltp(tradingsymbol)
        # the touch a real order meets: the ask for a BUY, the bid for a SELL (the LTP when no sane quote)
        touch = self._paper_touch(tradingsymbol, transaction_type, ltp)
        if resting:
            fill = price
        elif order_type == "LIMIT" and price > 0 and touch > 0:
            # a marketable limit fills at the touch, not at its limit; a passive one is assumed filled at its limit
            marketable = (price >= touch) if transaction_type == "BUY" else (price <= touch)
            fill = touch if marketable else price
        else:
            fill = price if price > 0 else touch
        if not resting and fill <= 0:
            # no tick for the symbol in this process yet (a restored position right after a restart): the paper
            # book's own last price, never ₹0 — a ₹0 fill booked the whole notional as a loss at the broker
            fill = next((float(p.get("last_price") or 0) for p in self._paper_positions
                         if p.get("tradingsymbol") == tradingsymbol), 0.0)
            if fill <= 0:
                logger.warning("[PAPER] {} {} {}: no price for the symbol yet — filled at ₹0",
                               order_type, transaction_type, tradingsymbol)
        record = {
            "order_id":         order_id,
            "tradingsymbol":    tradingsymbol,
            "exchange":         exchange,
            "transaction_type": transaction_type,
            "quantity":         quantity,
            "order_type":       order_type,
            "product":          product,
            "price":            fill,
            "average_price":    0.0 if resting else fill,
            "ref_ltp":          0.0 if resting else ltp,   # the LTP at the fill: callers price P&L at it
            "trigger_price":    trigger_price,
            "status":           "TRIGGER PENDING" if resting else "COMPLETE",
            "tag":              tag,
            "placed_at":        now_ist().isoformat(),      # IST, like every other timestamp
        }
        self._paper_orders.append(record)
        if not resting:
            self._update_paper_position(record)
        logger.info("[PAPER] {} {} {} qty={} @ ₹{} | id={}",
                    transaction_type, tradingsymbol, order_type,
                    quantity, fill or trigger_price, order_id)
        return order_id

    def paper_fill_at(self, tradingsymbol: str, exchange: str, transaction_type: str, quantity: int,
                      product: str, price: float, tag: str, fill_kind: str) -> str:
        """PAPER only (raises RuntimeError in LIVE): a COMPLETE order at exactly `price` — a pre-open order
        filled at the call-auction price (fill_kind="AUCTION"), an expiry settlement at intrinsic value
        (fill_kind="SETTLEMENT") or a book reconciliation (fill_kind="RECONCILE"). order_type "MARKET",
        average_price = ref_ltp = price (paper_fill_vs_ltp → 0), extra field fill_kind; updates the
        (symbol, product) row. A price of 0 is valid (an option expiring worthless)."""
        if settings.trading_mode != "PAPER":
            raise RuntimeError("paper_fill_at is a PAPER-mode primitive; refused in LIVE")
        if price is None or price < 0:
            raise InputException(f"Invalid fill price {price} for {tradingsymbol}")
        quantity = self._validated_quantity(tradingsymbol, exchange, product, quantity)
        order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        px = float(price)
        record = {
            "order_id": order_id, "tradingsymbol": tradingsymbol, "exchange": exchange,
            "transaction_type": transaction_type, "quantity": quantity, "order_type": "MARKET",
            "product": product, "price": px, "average_price": px, "ref_ltp": px, "trigger_price": 0.0,
            "status": "COMPLETE", "tag": (tag or "")[:_KITE_ORDER_TAG_MAX], "fill_kind": fill_kind,
            "placed_at": now_ist().isoformat(),
        }
        self._paper_orders.append(record)
        self._update_paper_position(record)
        logger.info("[PAPER] {} {} {} qty={} @ ₹{} ({}) | id={}", transaction_type, tradingsymbol, product,
                    quantity, px, fill_kind, order_id)
        return order_id

    def paper_fill_vs_ltp(self, order_id: str) -> float:
        """PAPER: fill price minus the LTP at the moment of the fill (the half-spread a marketable order paid,
        signed), 0.0 when unknown or not a paper order. Lets the ledger book what the paper broker booked."""
        for r in reversed(self._paper_orders):
            if r.get("order_id") == order_id:
                fill, ref = float(r.get("average_price") or 0), float(r.get("ref_ltp") or 0)
                return fill - ref if fill > 0 and ref > 0 else 0.0
        return 0.0

    def fill_price(self, order_id: str) -> Optional[float]:
        """Average fill price of an order, or None if not (yet) filled / unknown."""
        try:
            hist = self.order_history(order_id)
            last = hist[-1] if hist else {}
            if str(last.get("status", "")).upper() == "COMPLETE" and float(last.get("average_price") or 0) > 0:
                return float(last["average_price"])
        except Exception as exc:
            logger.debug("[kite] fill_price {}: {}", order_id, exc)
        return None

    @staticmethod
    def _paper_ltp(symbol: str) -> float:
        """Last traded price the tick engine has seen — the paper fill price for market orders."""
        from tick_engine import tick_engine   # local import: tick_engine imports this module
        tick, _ = tick_engine.latest(symbol)
        return float(tick.ltp) if tick else 0.0

    @staticmethod
    def _paper_touch(symbol: str, side: str, ltp: float) -> float:
        """Paper fill price for a marketable order: the tick's ask (BUY) or bid (SELL), so paper P&L pays the
        spread as a real fill does. Falls back to the LTP when the quote is missing, crossed or implausibly
        far (> 1 %) from the LTP."""
        try:
            from tick_engine import tick_engine
            tick, _ = tick_engine.latest(symbol)
            bid, ask = float(getattr(tick, "bid", 0) or 0), float(getattr(tick, "ask", 0) or 0)
        except Exception:
            return ltp
        if ltp <= 0 or bid <= 0 or ask <= 0 or bid > ask:
            return ltp
        px = ask if side == "BUY" else bid
        return px if abs(px / ltp - 1) <= 0.01 else ltp

    def _mark_paper_positions(self) -> None:
        """Mark the paper book to market the way Kite reports it: last_price from the tick engine and
        pnl = sell value − buy value + quantity × last price (realised + unrealised). Before this the
        paper book kept the fill price as last_price and pnl at 0 for the whole session."""
        for pos in self._paper_positions:
            px = self._paper_ltp(pos["tradingsymbol"])
            if px > 0:
                pos["last_price"] = px
            if "buy_value" in pos:
                pos["pnl"] = round(pos["sell_value"] - pos["buy_value"] + pos["quantity"] * pos["last_price"], 2)

    def _update_paper_position(self, order: dict) -> None:
        """One net row per (tradingsymbol, product), as Kite keeps them. A single row per symbol kept the
        product of the day's first order: a swing CNC buy after an intraday MIS round trip on the same symbol
        landed in the 'MIS' row, and the 15:15 square-off sold the delivery holding."""
        sym       = order["tradingsymbol"]
        product   = order.get("product") or "MIS"
        px        = order["price"]
        qty_delta = (order["quantity"] if order["transaction_type"] == "BUY"
                     else -order["quantity"])
        value     = px * order["quantity"]
        for pos in self._paper_positions:
            if pos["tradingsymbol"] == sym and (pos.get("product") or "MIS") == product:
                pos.setdefault("buy_value", 0.0); pos.setdefault("sell_value", 0.0)
                pos["buy_value" if qty_delta > 0 else "sell_value"] += value
                old_q = pos["quantity"]
                new_q = old_q + qty_delta
                if old_q == 0 or (new_q != 0 and (old_q > 0) != (new_q > 0)):
                    pos["average_price"] = px                  # fresh or flipped position
                    if old_q == 0:
                        # a fresh position carries its own product (an earlier MIS round trip in the symbol
                        # left the row MIS, and the 15:15 square-off then sold a CNC swing bought later)
                        pos["product"] = order.get("product", pos.get("product", "MIS"))
                elif (old_q > 0) == (qty_delta > 0):           # adding to the position
                    pos["average_price"] = round(
                        (pos["average_price"] * abs(old_q) + px * abs(qty_delta)) / abs(new_q), 2)
                pos["quantity"]   = new_q
                pos["last_price"] = px
                return
        self._paper_positions.append({
            "tradingsymbol": sym,
            "exchange":      order["exchange"],
            "product":       product,
            "quantity":      qty_delta,
            "average_price": px,
            "last_price":    px,
            "pnl":           0.0,
            "buy_value":     value if qty_delta > 0 else 0.0,
            "sell_value":    value if qty_delta < 0 else 0.0,
        })


kite_client = KiteClient()
