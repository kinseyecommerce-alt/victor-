"""
strategies/ — Positional trend-following strategies (MCX + NSE, daily bars).

Three low-parameter templates per the strategy spec:
  1. DonchianBreakoutStrategy — Turtle-style S1 (20/10) + S2 (55/20),
     2N stop, 0.5N pyramiding to 4 units, S1 whipsaw-skip filter.
  2. MACrossoverStrategy      — 20/50 EMA cross with 200-SMA trend filter,
     2.5–3×ATR(14) trailing stop.
  3. TSMOMStrategy            — 12-month time-series momentum, monthly
     rebalance, volatility-targeted sizing (Moskowitz/Ooi/Pedersen 2012).

Supporting modules:
  position_sizing — fixed-fractional & Turtle-unit lot sizing; portfolio
                    caps (4 units/market, 6/cluster, 10/direction, 6% heat)
  contracts       — MCX/NSE contract specs (mini contracts for small capital)
  cost_model      — MCX CTT / NSE STT / brokerage / GST / slippage model
  kill_criteria   — pre-committed numeric kill rules (DD×1.5 auto-halt etc.)
  positional_engine — EOD orchestrator: signals → sized, capped OrderPlans
"""
from strategies.base import Action, OpenPosition, PositionalStrategy, Signal
from strategies.contracts import CONTRACTS, ContractSpec, get_contract
from strategies.cost_model import CostBreakdown, order_cost, round_trip_cost, rollover_cost
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.indicators import DailyBar, atr, donchian, ema, ema_series, sma
from strategies.kill_criteria import KillCriteria, Verdict
from strategies.ma_crossover import MACrossoverStrategy
from strategies.position_sizing import (
    PortfolioRiskBook,
    fixed_fractional_lots,
    turtle_unit_lots,
    vol_target_lots,
)
from strategies.positional_engine import OrderPlan, PositionalEngine
from strategies.tsmom import TSMOMStrategy

__all__ = [
    "Action", "Signal", "OpenPosition", "PositionalStrategy",
    "DailyBar", "atr", "donchian", "ema", "ema_series", "sma",
    "ContractSpec", "CONTRACTS", "get_contract",
    "DonchianBreakoutStrategy", "MACrossoverStrategy", "TSMOMStrategy",
    "PortfolioRiskBook", "fixed_fractional_lots", "turtle_unit_lots",
    "vol_target_lots",
    "CostBreakdown", "order_cost", "round_trip_cost", "rollover_cost",
    "KillCriteria", "Verdict",
    "OrderPlan", "PositionalEngine",
]
