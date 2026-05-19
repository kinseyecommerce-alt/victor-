"""
greeks_engine.py
Black-Scholes option pricing, Greeks, and Newton-Raphson IV solver for NSE options.
All functions are synchronous and pure-math. No I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

RISK_FREE_RATE = 0.065          # RBI repo rate proxy
_SQRT2PI = math.sqrt(2 * math.pi)


@dataclass
class Greeks:
    delta:      float   # ∂V/∂S
    gamma:      float   # ∂²V/∂S²
    theta:      float   # daily theta (negative = time decay)
    vega:       float   # per 1% IV move
    rho:        float   # per 1% rate move
    iv:         float   # implied vol (annualised, 0–1)
    intrinsic:  float
    time_value: float
    moneyness:  str     # "ITM" | "ATM" | "OTM"
    bs_price:   float


def _N(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _n(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI

def _d1d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             opt: Literal["CE", "PE"]) -> float:
    if T <= 0:
        return max(0.0, (S - K) if opt == "CE" else (K - S))
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if opt == "CE":
        return S * _N(d1) - K * math.exp(-r * T) * _N(d2)
    return K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)

def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float,
    opt: Literal["CE", "PE"], tol: float = 1e-5, max_iter: int = 100,
) -> float:
    """Newton-Raphson IV solver. Returns NaN on failure (e.g. intrinsic-only)."""
    if T <= 0 or market_price <= 0:
        return float("nan")
    intrinsic = max(0.0, (S - K) if opt == "CE" else (K - S))
    if market_price <= intrinsic + 1e-8:
        return float("nan")
    sigma = 0.30
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, opt)
        d1, _ = _d1d2(S, K, T, r, sigma)
        vega_val = S * _n(d1) * math.sqrt(T)
        if abs(vega_val) < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega_val
        sigma = max(0.001, min(sigma, 20.0))
    return sigma


def calculate_greeks(
    spot: float,
    strike: float,
    expiry: date,
    option_type: Literal["CE", "PE"],
    market_price: float,
    r: float = RISK_FREE_RATE,
) -> Greeks:
    """Compute full Greeks for a single NSE option."""
    today = datetime.now().date()
    cal_days = (expiry - today).days
    T = max(cal_days / 365.0, 0.5 / 365.0)   # min 12h

    iv = implied_volatility(market_price, spot, strike, T, r, option_type)
    if math.isnan(iv) or iv <= 0:
        iv = 0.25

    d1, d2 = _d1d2(spot, strike, T, r, iv)
    sqrt_T  = math.sqrt(T)
    e_rT    = math.exp(-r * T)
    nd1     = _n(d1)

    gamma   = nd1 / (spot * iv * sqrt_T)
    vega    = spot * nd1 * sqrt_T / 100.0   # per 1% IV

    if option_type == "CE":
        delta     = _N(d1)
        theta     = (-(spot * nd1 * iv / (2 * sqrt_T)) - r * strike * e_rT * _N(d2))  / 365.0
        intrinsic = max(0.0, spot - strike)
        rho       =  strike * T * e_rT * _N(d2)  / 100.0
    else:
        delta     = _N(d1) - 1.0
        theta     = (-(spot * nd1 * iv / (2 * sqrt_T)) + r * strike * e_rT * _N(-d2)) / 365.0
        intrinsic = max(0.0, strike - spot)
        rho       = -strike * T * e_rT * _N(-d2) / 100.0

    time_value = max(0.0, market_price - intrinsic)
    atm_band   = spot * 0.005
    if abs(spot - strike) <= atm_band:
        moneyness = "ATM"
    elif (option_type == "CE" and spot > strike) or (option_type == "PE" and spot < strike):
        moneyness = "ITM"
    else:
        moneyness = "OTM"

    price_calc = bs_price(spot, strike, T, r, iv, option_type)

    return Greeks(
        delta=round(delta, 4), gamma=round(gamma, 6),
        theta=round(theta, 4), vega=round(vega, 4), rho=round(rho, 4),
        iv=round(iv, 4), intrinsic=round(intrinsic, 2),
        time_value=round(time_value, 2), moneyness=moneyness,
        bs_price=round(price_calc, 4),
    )


def atm_strike(spot: float, step: int = 50) -> int:
    """Round spot to nearest strike step."""
    return int(round(spot / step) * step)


def select_strike_by_delta(
    spot: float,
    strikes: list[int],
    option_type: Literal["CE", "PE"],
    target_delta: float = 0.40,
    days_to_expiry: int = 7,
    r: float = RISK_FREE_RATE,
    iv_guess: float = 0.25,
) -> int:
    """Return the strike whose absolute delta is closest to target_delta."""
    if not strikes:
        return atm_strike(spot)
    T = max(days_to_expiry / 365.0, 0.5 / 365.0)
    best_strike, best_diff = strikes[0], float("inf")
    for k in strikes:
        d1, _ = _d1d2(spot, k, T, r, iv_guess)
        delta = _N(d1) if option_type == "CE" else (_N(d1) - 1.0)
        diff  = abs(abs(delta) - target_delta)
        if diff < best_diff:
            best_diff, best_strike = diff, k
    return best_strike


def days_to_next_expiry(option_type_day: str = "thursday") -> int:
    """Return calendar days to next weekly NSE expiry (Thursday=NIFTY)."""
    today = datetime.now().date()
    day_map = {"monday": 0, "tuesday": 1, "wednesday": 2,
               "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    target = day_map.get(option_type_day.lower(), 3)
    days_ahead = (target - today.weekday()) % 7
    return max(days_ahead, 1)
