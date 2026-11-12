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
        self.wins_today: int           = 0
        self.losses_today: int         = 0
        self.gross_profit: float       = 0.0
        self.gross_loss: float         = 0.0
        self.costs_today: float        = 0.0   # transaction costs of today's closed trades (performance ledger)
        self.is_trading_halted: bool   = False

    def net_realised_pnl(self) -> float:
        """Today's realised P&L after transaction costs — what the daily loss limit is measured on."""
        return self.daily_realised_pnl - self.costs_today

    def check_before_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        transaction_type: str,
        check_size: bool = True,
    ) -> tuple[bool, str]:
        """check_size=False for leveraged instruments (futures), whose notional value is not
        the money at risk; their callers cap margin and loss-at-stop instead."""
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

        if check_size:
            ok, msg = self._check_position_size(quantity, price)
            if not ok:
                return False, msg

        return True, "OK"

    def _check_market_hours(self) -> tuple[bool, str]:
        from config import settings
        if settings.trading_mode == "PAPER":
            return True, "OK"
        now_t = ist_time()
        open_t  = time(9, 15)
        sq_h, sq_m = [int(x) for x in settings.squareoff_time.split(":")]
        close_t = time(sq_h, sq_m)
        if not (open_t <= now_t <= close_t):
            return False, f"Outside trading hours (market {open_t}–{close_t})"
        return True, "OK"

    def _check_daily_loss(self) -> tuple[bool, str]:
        # net of costs: record_trade gets the gross P&L, and the costs of hundreds of round trips a day let the
        # account lose more than MAX_DAILY_LOSS while still opening positions
        net = self.net_realised_pnl()
        if net < -settings.max_daily_loss:
            return False, (
                f"Daily loss limit ₹{settings.max_daily_loss:.0f} breached "
                f"(current P&L ₹{net:.0f} after ₹{self.costs_today:.0f} costs)"
            )
        return True, "OK"

    def _check_position_count(self) -> tuple[bool, str]:
        if self.open_position_count >= settings.max_open_positions:
            return False, f"Max open positions reached ({settings.max_open_positions})"
        return True, "OK"

    def _check_position_size(self, quantity: int, price: float) -> tuple[bool, str]:
        value = quantity * price
        if value > settings.max_position_size:
            return False, (
                f"Position size ₹{value:.0f} exceeds limit ₹{settings.max_position_size:.0f}"
            )
        return True, "OK"

    def max_capital_for_agent(self, agent_name: str) -> float:
        """Return per-symbol capital (₹) for an agent: bucket total ÷ max concurrent positions."""
        bucket_total = self.bucket_total(agent_name)

        max_pos_attr = _AGENT_MAX_POS.get(agent_name)
        if max_pos_attr:
            max_pos = getattr(settings, max_pos_attr, settings.max_open_positions)
        else:
            max_pos = 1   # lot-based agents (fno): return full bucket
        return bucket_total / max(max_pos, 1)

    def open_positions_for(self, agent_name: str) -> int:
        """Positions the trailing engine currently watches for this agent (strategy tag)."""
        from trailing_sl_engine import trailing_sl_engine, SLStatus
        return sum(1 for pos in list(trailing_sl_engine._positions.values())
                   if getattr(pos, "strategy", None) == agent_name and pos.status == SLStatus.ACTIVE)

    def open_cap_reason(self, agent_name: str) -> str | None:
        """Per-agent open-position cap (currently F&O only): the skip reason, or None."""
        if agent_name == "fno" and settings.fno_max_open_positions > 0:
            n = self.open_positions_for(agent_name)
            if n >= settings.fno_max_open_positions:
                return f"max open F&O positions ({settings.fno_max_open_positions}) reached ({n} open)"
        return None

    def bucket_exposure(self, agent_name: str) -> float:
        """₹ value of positions the trailing engine currently tracks for every agent that shares
        this agent's capital bucket (intraday + scalping share the MIS bucket)."""
        from trailing_sl_engine import trailing_sl_engine
        bucket = _AGENT_TO_BUCKET.get(agent_name, "intraday")
        peers = {a for a, b in _AGENT_TO_BUCKET.items() if b == bucket}
        total = 0.0
        for pos in list(trailing_sl_engine._positions.values()):
            if getattr(pos, "strategy", None) in peers:
                total += float(pos.entry_price) * abs(int(pos.quantity))
        return total

    def bucket_total(self, agent_name: str) -> float:
        """₹ exposure the bucket may carry. The MIS (intraday) bucket is its capital × intraday_leverage:
        the broker's intraday margin lets ₹1 of capital carry several rupees of position. Leverage
        multiplies profit and loss alike; the daily loss limit and risk-per-trade still apply."""
        bucket = _AGENT_TO_BUCKET.get(agent_name, "intraday")
        pct_attr = _BUCKET_PCT_ATTR.get(bucket, "intraday_capital_pct")
        total = settings.total_capital * getattr(settings, pct_attr, 25.0) / 100
        if bucket == "intraday":
            total *= max(1.0, min(5.0, settings.intraday_leverage))
        return total

    def bucket_room(self, agent_name: str) -> float:
        """₹ still deployable in this agent's bucket. Fewer, larger positions (max 2 per agent) can
        exceed the bucket when two agents share it, so every entry is capped by what is left."""
        return max(0.0, self.bucket_total(agent_name) - self.bucket_exposure(agent_name))

    def calculate_quantity(
        self,
        price: float,
        agent: str = "",
        capital: float | None = None,
        risk_pct: float | None = None,
    ) -> int:
        if capital is not None:
            cap = capital
        elif agent:
            cap = self.max_capital_for_agent(agent)
        else:
            cap = settings.max_position_size
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
        if pnl > 0:
            self.wins_today += 1
            self.gross_profit += pnl
        elif pnl < 0:
            self.losses_today += 1
            self.gross_loss += -pnl
        logger.info("Trade P&L ₹{:.0f} | Day total ₹{:.0f}", pnl, self.daily_realised_pnl)

    def record_cost(self, cost: float) -> None:
        """Transaction costs of a closed trade (the ledger computes them); counted against the daily loss limit."""
        self.costs_today += max(0.0, float(cost or 0.0))

    def position_opened(self) -> None:
        self.open_position_count += 1

    def position_closed(self) -> None:
        self.open_position_count = max(0, self.open_position_count - 1)

    def reset_daily(self) -> None:
        self.daily_realised_pnl  = 0.0
        self.open_position_count = 0
        self.trades_today        = 0
        self.wins_today          = 0
        self.losses_today        = 0
        self.gross_profit        = 0.0
        self.gross_loss          = 0.0
        self.costs_today         = 0.0
        self.is_trading_halted   = False
        logger.info("Risk manager reset for new trading day")

    def status(self) -> dict:
        return {
            "daily_pnl":          self.daily_realised_pnl,
            "costs_today":        round(self.costs_today, 2),
            "net_daily_pnl":      round(self.net_realised_pnl(), 2),
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
