"""
strategies/contracts.py
Contract specifications for the positional trend-following universe
(MCX energy/metals + NSE index futures).

point_value = ₹ P&L per 1-point price move per *unit* (usually 1.0);
lot_size    = units per lot, so
    rupees per point per lot = point_value × lot_size
e.g. Crude Oil Mini: quoted ₹/barrel, 10 barrels/lot → ₹10 per point per lot.

Lot sizes and tick sizes change via exchange circulars — verify against the
live instrument dump (kite.instruments()) before trading and override here.
"""
from __future__ import annotations

from dataclasses import dataclass


# Correlation clusters (Turtle-style): instruments in the same cluster count
# against the shared 6-unit "closely correlated" cap.
CLUSTER_ENERGY = "energy"
CLUSTER_METALS = "metals"
CLUSTER_INDEX  = "index"


@dataclass(frozen=True)
class ContractSpec:
    symbol:      str      # root symbol (without expiry), e.g. "CRUDEOILM"
    exchange:    str      # "MCX" or "NFO"
    lot_size:    int      # units per lot
    point_value: float    # ₹ per 1-point move per unit
    tick_size:   float    # minimum price increment (₹)
    cluster:     str      # correlation cluster
    mini:        bool = False   # mini contract (preferred for small capital)

    @property
    def rupees_per_point(self) -> float:
        """₹ P&L per 1-point price move for one lot."""
        return self.point_value * self.lot_size


# ── Universe from the strategy spec ────────────────────────────────────────
# MCX: CRUDEOIL(M), GOLD(M), SILVER(M), NATURALGAS, COPPER
# NSE: NIFTY / BANKNIFTY futures
CONTRACTS: dict[str, ContractSpec] = {
    # MCX energy (quoted ₹/barrel and ₹/mmBtu)
    "CRUDEOIL":   ContractSpec("CRUDEOIL",   "MCX", 100,  1.0, 1.00, CLUSTER_ENERGY),
    "CRUDEOILM":  ContractSpec("CRUDEOILM",  "MCX", 10,   1.0, 1.00, CLUSTER_ENERGY, mini=True),
    "NATURALGAS": ContractSpec("NATURALGAS", "MCX", 1250, 1.0, 0.10, CLUSTER_ENERGY),
    # MCX metals (GOLD quoted ₹/10g; SILVER ₹/kg; COPPER ₹/kg)
    "GOLD":       ContractSpec("GOLD",       "MCX", 100,  1.0, 1.00, CLUSTER_METALS),
    "GOLDM":      ContractSpec("GOLDM",      "MCX", 10,   1.0, 1.00, CLUSTER_METALS, mini=True),
    "SILVER":     ContractSpec("SILVER",     "MCX", 30,   1.0, 1.00, CLUSTER_METALS),
    "SILVERM":    ContractSpec("SILVERM",    "MCX", 5,    1.0, 1.00, CLUSTER_METALS, mini=True),
    "COPPER":     ContractSpec("COPPER",     "MCX", 2500, 1.0, 0.05, CLUSTER_METALS),
    # NSE index futures (lot sizes as of 2025 revisions — verify before live)
    "NIFTY":      ContractSpec("NIFTY",      "NFO", 75,   1.0, 0.05, CLUSTER_INDEX),
    "BANKNIFTY":  ContractSpec("BANKNIFTY",  "NFO", 35,   1.0, 0.05, CLUSTER_INDEX),
}


def get_contract(symbol: str) -> ContractSpec:
    try:
        return CONTRACTS[symbol.upper()]
    except KeyError:
        raise KeyError(f"Unknown contract '{symbol}' — add it to strategies/contracts.py")
