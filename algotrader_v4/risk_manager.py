"""
risk_manager.py
All risk checks run BEFORE an order is placed.
The bot_engine calls `risk.check_before_order()` — if it returns False,
the trade is skipped and a reason is logged / alerted.
"""
from __future__ import annotations

from datetime import time
from loguru import logger

from config import settings
from ist_clock import ist_time


# Capital bucket mapping: agent name → trading-type bucket
_AGENT_TO_BUCKET: dict[str, str] = {
    "intraday": "intraday",
    "scalping": "intraday",   # scalping shares the equity intraday pool
    "swing":    "swing",
    "fno":      "options",
}

_BUCKET_PCT_ATTR: dict[str, str] = {
    "intraday": "intraday_capital_pct",
    "swing":    "swing_capital_pct",
    "options":  "options_capital_pct",
    "futures":  "futures_capital_pct",
}

# Max-positions config attr per agent; None = lot-based (fno), no per-symbol split
_AGENT_MAX_POS: dict[str, str | None] = {
    "intraday": "max_intraday_positions",
    "scalping": "max_scalping_positions",
    "swing":    "max_swing_positions",
    "fno":      None,
}


class RiskManager:

    def __init__(self) -> None:
        self.daily_realised_pnl: float = 0.0
        self.open_position_count: int  = 0
        self.trades_today: int         = 0
        self.is_trading_halted: bool   = False

    def check_before_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        transaction_type: str,
    ) -> tuple[bool, str]:
        if self.is_trading_halted:
            return False, "Trading halted for the day (daily loss limit hit)"

        ok, msg = self._check_market_hours()
        if not ok:
            return False, msg

        ok, msg = self._check_daily_loss()
        if not ok:
            self.is_trading_halted = True
            return False, msg

        if transaction_type == "BUY":
            ok, msg = self._check_position_count()
            if not ok:
                return False, msg

        ok, msg = self._check_position_size(quantity, price, symbol)
        if not ok:
            return False, msg

        return True, "OK"

    def _check_market_hours(self) -> tuple[bool, str]:
        from config import settings
        if settings.trading_mode == "PAPER":
            return True, "OK"
        now_t = ist_time()
        oh, om = [int(x) for x in settings.mcx_open_time.split(":")]
        ch, cm = [int(x) for x in settings.mcx_close_time.split(":")]
        open_t, close_t = time(oh, om), time(ch, cm)
        if not (open_t <= now_t <= close_t):
            return False, f"Outside MCX hours (market {open_t}–{close_t})"
        return True, "OK"

    def _check_daily_loss(self) -> tuple[bool, str]:
        if self.daily_realised_pnl < -settings.max_daily_loss:
            return False, (
                f"Daily loss limit ₹{settings.max_daily_loss:.0f} breached "
                f"(current P&L ₹{self.daily_realised_pnl:.0f})"
            )
        return True, "OK"

    def _check_position_count(self) -> tuple[bool, str]:
        if self.open_position_count >= settings.max_open_positions:
            return False, f"Max open positions reached ({settings.max_open_positions})"
        return True, "OK"

    def _check_position_size(self, quantity: int, price: float, symbol: str = "") -> tuple[bool, str]:
        # MCX: commodity futures carry large notional but trade on margin — cap on
        # the margin blocked (lots × margin_per_lot), not the full contract value.
        if symbol:
            import mcx_universe
            c = mcx_universe.contract(symbol)
            if c is not None:
                lots   = max(1, quantity // c.lot_size)
                margin = lots * c.margin_per_lot
                limit  = getattr(settings, "mcx_max_margin_per_position", 300000.0)
                if margin > limit:
                    return False, (
                        f"Position margin ₹{margin:.0f} exceeds MCX limit ₹{limit:.0f}"
                    )
                return True, "OK"

        value = quantity * price
        if value > settings.max_position_size:
            return False, (
                f"Position size ₹{value:.0f} exceeds limit ₹{settings.max_position_size:.0f}"
            )
        return True, "OK"

    def max_capital_for_agent(self, agent_name: str) -> float:
        """Return per-symbol capital (₹) for an agent: bucket total ÷ max concurrent positions."""
        bucket      = _AGENT_TO_BUCKET.get(agent_name, "intraday")
        pct_attr    = _BUCKET_PCT_ATTR.get(bucket, "intraday_capital_pct")
        bucket_total = settings.total_capital * getattr(settings, pct_attr, 25.0) / 100

        max_pos_attr = _AGENT_MAX_POS.get(agent_name)
        if max_pos_attr:
            max_pos = getattr(settings, max_pos_attr, settings.max_open_positions)
        else:
            max_pos = 1   # lot-based agents (fno): return full bucket
        return bucket_total / max(max_pos, 1)

    def calculate_quantity(
        self,
        price: float,
        agent: str = "",
        symbol: str = "",
        capital: float | None = None,
        risk_pct: float | None = None,
    ) -> int:
        # Resolve the capital available to this agent/order
        if capital is not None:
            cap = capital
        elif agent:
            cap = self.max_capital_for_agent(agent)
        else:
            cap = settings.max_position_size

        # ── MCX lot-based sizing ──────────────────────────────────────────
        # Commodity futures trade in whole lots; size by how many lots the
        # allocated capital can margin, then return lots × lot_size.
        if symbol:
            import mcx_universe
            c = mcx_universe.contract(symbol)
            if c is not None:
                lots = int(cap // c.margin_per_lot)
                lots = max(lots, 1)
                return lots * c.lot_size

        # ── Legacy notional sizing (equity / non-MCX) ─────────────────────
        if risk_pct and price > 0:
            sl_amount = price * (settings.stop_loss_pct / 100)
            cap = min(cap, (cap * risk_pct / 100) / sl_amount * price)
        qty = int(cap // price)
        return max(qty, 1)

    def sl_price(self, entry: float, side: str) -> float:
        pct = settings.stop_loss_pct / 100
        return round(entry * (1 - pct) if side == "BUY" else entry * (1 + pct), 2)

    def target_price(self, entry: float, side: str) -> float:
        pct = settings.target_pct / 100
        return round(entry * (1 + pct) if side == "BUY" else entry * (1 - pct), 2)

    def record_trade(self, pnl: float) -> None:
        self.daily_realised_pnl += pnl
        self.trades_today += 1
        logger.info("Trade P&L ₹{:.0f} | Day total ₹{:.0f}", pnl, self.daily_realised_pnl)

    def position_opened(self) -> None:
        self.open_position_count += 1

    def position_closed(self) -> None:
        self.open_position_count = max(0, self.open_position_count - 1)

    def reset_daily(self) -> None:
        self.daily_realised_pnl  = 0.0
        self.open_position_count = 0
        self.trades_today        = 0
        self.is_trading_halted   = False
        logger.info("Risk manager reset for new trading day")

    def status(self) -> dict:
        return {
            "daily_pnl":          self.daily_realised_pnl,
            "open_positions":     self.open_position_count,
            "trades_today":       self.trades_today,
            "is_halted":          self.is_trading_halted,
            "max_daily_loss":     settings.max_daily_loss,
            "max_position_size":  settings.max_position_size,
            "max_open_positions": settings.max_open_positions,
            "stop_loss_pct":      settings.stop_loss_pct,
            "target_pct":         settings.target_pct,
        }


risk_manager = RiskManager()
