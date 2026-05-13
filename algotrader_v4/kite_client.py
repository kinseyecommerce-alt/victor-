"""
kite_client.py  (v4 — ORDERS ONLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zerodha Kite is used EXCLUSIVELY for:
  • Authentication (login URL, access token)
  • Order placement (place, modify, cancel)
  • Position & holding queries (portfolio state)
  • Square-off (emergency close all)

Market data (quotes, OHLCV, ticks) comes from:
  → NSEClient   (live quotes — market_data.py)
  → YFinanceClient (historical OHLCV — market_data.py)

REMOVED from this file vs v1/v2/v3:
  ✗ quote()          — now via NSEClient
  ✗ ltp()            — now via NSEClient
  ✗ ohlc()           — now via NSEClient
  ✗ historical_data()— now via YFinanceClient
  ✗ instruments()    — now via YFinanceClient
  ✗ start_ticker()   — now via TickEngine (NSE API)
  ✗ stop_ticker()    — same
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from kiteconnect import KiteConnect
from loguru import logger

from config import settings


class KiteClient:
    """
    Thin wrapper around KiteConnect.
    Only order and portfolio methods are exposed.
    In PAPER mode every mutating call is simulated in memory.
    """

    def __init__(self) -> None:
        self._kite: Optional[KiteConnect] = None
        self._paper_orders:    list[dict] = []
        self._paper_positions: list[dict] = []

    # ── Auth ───────────────────────────────────────────────────────────

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
        logger.info("Kite auth OK (orders-only mode)")
        return token

    @property
    def kite(self) -> KiteConnect:
        if self._kite is None:
            raise RuntimeError(
                "KiteClient not initialised — call set_access_token() first."
            )
        return self._kite

    # ── Portfolio (read-only, needed for P&L + exit decisions) ─────────

    def positions(self) -> dict:
        if settings.trading_mode == "PAPER":
            return {"net": self._paper_positions, "day": self._paper_positions}
        return self.kite.positions()

    def holdings(self) -> list[dict]:
        if settings.trading_mode == "PAPER":
            return []
        return self.kite.holdings()

    def orders(self) -> list[dict]:
        if settings.trading_mode == "PAPER":
            return self._paper_orders
        return self.kite.orders()

    def order_history(self, order_id: str) -> list[dict]:
        if settings.trading_mode == "PAPER":
            return [o for o in self._paper_orders if o["order_id"] == order_id]
        return self.kite.order_history(order_id)

    # ── Order placement ─────────────────────────────────────────────────

    def place_order(
        self,
        tradingsymbol:    str,
        exchange:         str,
        transaction_type: str,          # BUY | SELL
        quantity:         int,
        order_type:       str = "MARKET",
        product:          str = "MIS",  # MIS | CNC | NRML
        price:            float = 0.0,
        trigger_price:    float = 0.0,
        validity:         str = "DAY",
        tag:              str = "AlgoTraderPro",
    ) -> str:
        if settings.trading_mode == "PAPER":
            return self._paper_place(
                tradingsymbol, exchange, transaction_type,
                quantity, order_type, product, price, trigger_price, tag,
            )

        order_id = self.kite.place_order(
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
        logger.info("LIVE order placed | {} {} {} qty={} @ {} | id={}",
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
        return self.kite.modify_order(
            variety=KiteConnect.VARIETY_REGULAR, order_id=order_id,
            price=price or None, quantity=quantity or None,
            trigger_price=trigger_price or None,
        )

    def cancel_order(self, order_id: str) -> str:
        if settings.trading_mode == "PAPER":
            for o in self._paper_orders:
                if o["order_id"] == order_id:
                    o["status"] = "CANCELLED"
            return order_id
        return self.kite.cancel_order(
            variety=KiteConnect.VARIETY_REGULAR, order_id=order_id
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

    # ── Paper trading helpers ───────────────────────────────────────────

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