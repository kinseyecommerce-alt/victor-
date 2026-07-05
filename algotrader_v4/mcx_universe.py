"""
mcx_universe.py
MCX (Multi Commodity Exchange of India) contract universe.

Replaces the NSE/BSE equity universe (nifty100.py) for commodity futures
trading. Defines every tradable contract with its lot size, tick size, an
approximate reference price (used to seed the PAPER simulator), an approximate
per-lot SPAN+exposure margin, its commodity group and its trading session.

MCX sessions (IST):
  • Normal (bullion / energy / base metals): 09:00 – 23:30
  • Agri commodities:                         09:00 – 21:00

All commodity futures are lot-based: order quantity must be a whole multiple of
the contract's lot size. Sizing is therefore margin-driven (how many lots the
allocated capital can fund) rather than the notional ÷ price used for equity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime


# ── Commodity groups & sessions ────────────────────────────────────────────────

GROUP_BULLION    = "bullion"
GROUP_ENERGY     = "energy"
GROUP_BASE_METAL = "base_metal"
GROUP_AGRI       = "agri"

# Session close times per group (open is 09:00 for all)
SESSION_OPEN        = dtime(9, 0)
SESSION_CLOSE_NORMAL = dtime(23, 30)   # bullion, energy, base metals
SESSION_CLOSE_AGRI   = dtime(21, 0)    # agri commodities

_AGRI_GROUPS = {GROUP_AGRI}


@dataclass(frozen=True)
class Contract:
    symbol:         str
    lot_size:       int      # exchange lot (order qty must be a multiple of this)
    tick_size:      float    # minimum price increment
    base_price:     float    # reference price (₹) — seeds PAPER simulator
    margin_per_lot: float    # approx SPAN + exposure margin (₹) to hold 1 lot
    group:          str
    is_mini:        bool = False   # mini / micro contract (smaller lot)

    @property
    def session_close(self) -> dtime:
        return SESSION_CLOSE_AGRI if self.group in _AGRI_GROUPS else SESSION_CLOSE_NORMAL


# ── Contract table ──────────────────────────────────────────────────────────────
# Values are representative (not live) — good enough for PAPER sizing, the
# simulator seed and lot-multiple validation. Update margins from the broker's
# SPAN file before live trading.

CONTRACTS: dict[str, Contract] = {c.symbol: c for c in [
    # Bullion
    Contract("GOLDM",     10,   1.0,  72000.0, 180000.0, GROUP_BULLION, is_mini=True),
    Contract("SILVERM",    5,   1.0,  90000.0,  95000.0, GROUP_BULLION, is_mini=True),
    Contract("SILVERMIC",  1,   1.0,  90000.0,  19000.0, GROUP_BULLION, is_mini=True),
    # Energy
    Contract("CRUDEOIL",  100,  1.0,   6500.0, 210000.0, GROUP_ENERGY),
    Contract("CRUDEOILM",  10,  1.0,   6500.0,  21000.0, GROUP_ENERGY, is_mini=True),
    Contract("NATURALGAS", 1250, 0.1,   250.0, 140000.0, GROUP_ENERGY),
    Contract("NATGASMINI", 125,  0.1,   250.0,  14000.0, GROUP_ENERGY, is_mini=True),
    # Base metals
    Contract("COPPER",    2500, 0.05,   810.0, 210000.0, GROUP_BASE_METAL),
    Contract("ZINC",      5000, 0.05,   255.0, 130000.0, GROUP_BASE_METAL),
    Contract("ALUMINIUM", 5000, 0.05,   240.0, 110000.0, GROUP_BASE_METAL),
    Contract("LEAD",      5000, 0.05,   185.0,  95000.0, GROUP_BASE_METAL),
    Contract("LEADMINI",  1000, 0.05,   185.0,  19000.0, GROUP_BASE_METAL, is_mini=True),
    Contract("NICKEL",    1500, 0.10,  1350.0, 190000.0, GROUP_BASE_METAL),
    # Agri (shorter session)
    Contract("COTTON",      25, 10.0,  61000.0,  95000.0, GROUP_AGRI),
]}

MCX_SYMBOLS: list[str] = list(CONTRACTS.keys())


# ── Per-strategy universes ──────────────────────────────────────────────────────
# Each MCX trading-type agent trades the contracts best suited to its style.

# Intraday (MIS): the full liquid set — square off before session close.
INTRADAY_UNIVERSE = [
    "GOLDM", "SILVERM", "CRUDEOIL", "CRUDEOILM", "NATURALGAS",
    "COPPER", "ZINC", "ALUMINIUM", "LEAD", "NICKEL",
]

# Scalping (MIS): fastest, tightest, most liquid contracts.
SCALPING_UNIVERSE = [
    "CRUDEOIL", "CRUDEOILM", "NATURALGAS", "NATGASMINI",
    "SILVERM", "SILVERMIC",
]

# Positional / Overnight (NRML): strong-trend carriers held across sessions.
POSITIONAL_UNIVERSE = [
    "GOLDM", "SILVERM", "CRUDEOIL", "COPPER", "NICKEL", "NATURALGAS",
]

# Options / Spread (NRML): options underlyings + spread legs.
OPTIONS_UNIVERSE = [
    "GOLDM", "SILVERM", "CRUDEOIL", "NATURALGAS",
]

# Inter-commodity spread pairs the Options/Spread agent watches
# (long leg, short leg) — traded on the ratio between two correlated contracts.
SPREAD_PAIRS: list[tuple[str, str]] = [
    ("GOLDM", "SILVERM"),        # gold/silver ratio
    ("CRUDEOIL", "NATURALGAS"),  # energy complex
]

# Correlated groups — the coordinator treats a group as one exposure bucket so
# two agents can't stack the same directional bet across correlated contracts.
CORRELATION_GROUPS: dict[str, str] = {
    sym: c.group for sym, c in CONTRACTS.items()
}

_STRATEGY_UNIVERSE: dict[str, list[str]] = {
    "intraday": INTRADAY_UNIVERSE,   # MCX Intraday (MIS)
    "scalping": SCALPING_UNIVERSE,   # MCX Scalping (MIS)
    "swing":    POSITIONAL_UNIVERSE, # MCX Positional / Overnight (NRML)
    "fno":      OPTIONS_UNIVERSE,    # MCX Options / Spread (NRML)
}


# ── Helpers ─────────────────────────────────────────────────────────────────────

def contract(symbol: str) -> Contract | None:
    """Return the Contract for a symbol (mini/regular), or None if not an MCX contract."""
    return CONTRACTS.get(symbol)


def is_mcx_symbol(symbol: str) -> bool:
    return symbol in CONTRACTS


def lot_size(symbol: str) -> int:
    c = CONTRACTS.get(symbol)
    return c.lot_size if c else 1


def group_of(symbol: str) -> str:
    c = CONTRACTS.get(symbol)
    return c.group if c else "unknown"


def round_to_tick(symbol: str, price: float) -> float:
    """Snap a price to the contract's tick grid."""
    c = CONTRACTS.get(symbol)
    if not c or c.tick_size <= 0:
        return round(price, 2)
    return round(round(price / c.tick_size) * c.tick_size, 4)


def base_prices() -> dict[str, float]:
    """Symbol → reference price, for seeding the PAPER tick simulator."""
    return {sym: c.base_price for sym, c in CONTRACTS.items()}


def as_watchlist(symbols: list[str], exchange: str = "MCX") -> list[dict]:
    return [{"symbol": s, "exchange": exchange} for s in symbols]


def get_strategy_watchlist(strategy: str) -> list[dict]:
    """Watchlist (list of {symbol, exchange}) for an agent, keyed by its registry name."""
    return as_watchlist(_STRATEGY_UNIVERSE.get(strategy, INTRADAY_UNIVERSE))
