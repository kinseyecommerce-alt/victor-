"""
correlation_guard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Computes rolling 20-day correlation matrix for watchlist symbols.
Prevents over-correlated position entries that create hidden concentration risk.

Decision thresholds:
  r > 0.85  → BLOCK  (size_factor 0.0)
  r > 0.70  → ALLOW with halved size (size_factor 0.5)
  unknown   → ALLOW with reduced size (size_factor 0.75)
  otherwise → ALLOW at full size     (size_factor 1.0)
"""
from __future__ import annotations

import asyncio
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


# ── Module-level correlation matrix cache ─────────────────────────────────────
# Rows and columns are NSE symbol names (without .NS suffix).
_corr_matrix: pd.DataFrame = pd.DataFrame()

# Tracks whether the matrix has ever been successfully populated
_matrix_ready: bool = False


# ── Internal helpers ───────────────────────────────────────────────────────────

def _to_yf_ticker(symbol: str) -> str:
    """Append NSE suffix expected by yfinance."""
    sym = symbol.upper()
    return sym if sym.endswith(".NS") else sym + ".NS"


def _strip_ns(symbol: str) -> str:
    """Normalise a symbol to bare NSE name (no suffix, uppercase)."""
    return symbol.upper().removesuffix(".NS")


def _download_prices(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    """
    Blocking yfinance download — intended to be called inside asyncio.to_thread.
    Returns a DataFrame of adjusted close prices indexed by date.
    """
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when multiple tickers are requested:
    #   (field, ticker) — we want the "Close" level.
    # When only one ticker is requested it returns a flat DataFrame.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"]
        else:
            # Fallback — take the last field level (Close is always last)
            closes = raw.xs(raw.columns.get_level_values(0)[-1], axis=1, level=0)
    else:
        # Single ticker
        if "Close" in raw.columns:
            closes = raw[["Close"]]
        else:
            closes = raw.iloc[:, [0]]

    # Rename columns to bare NSE symbols (strip .NS)
    closes.columns = [_strip_ns(str(c)) for c in closes.columns]
    return closes


# ── Public: matrix refresh ─────────────────────────────────────────────────────

async def refresh_matrix(symbols: list[str]) -> None:
    """
    Download ~30 calendar days of daily prices for `symbols` (to get a solid
    20 trading-day window), then compute the pairwise return correlation matrix
    and store it in `_corr_matrix`.

    Uses asyncio.to_thread so the blocking yfinance I/O doesn't stall the event
    loop.  Handles all download / parse errors gracefully — the existing matrix
    is preserved if the refresh fails.
    """
    global _corr_matrix, _matrix_ready

    if not symbols:
        logger.warning("correlation_guard: refresh_matrix called with empty symbol list")
        return

    yf_tickers = [_to_yf_ticker(s) for s in symbols]
    logger.info(
        "correlation_guard: downloading 30d prices for {} symbols", len(yf_tickers)
    )

    try:
        closes: pd.DataFrame = await asyncio.to_thread(
            _download_prices,
            yf_tickers,
            "30d",   # period
            "1d",    # interval
        )
    except Exception as exc:
        logger.warning("correlation_guard: yfinance download failed: {}", exc)
        return

    if closes is None or closes.empty:
        logger.warning(
            "correlation_guard: no price data returned for {} — matrix unchanged",
            symbols,
        )
        return

    # Drop columns that are entirely NaN (symbols with no data)
    closes = closes.dropna(axis=1, how="all")

    if closes.shape[0] < 5:
        logger.warning(
            "correlation_guard: too few rows ({}) to compute correlation", closes.shape[0]
        )
        return

    try:
        returns = closes.pct_change().dropna(how="all")
        matrix  = returns.corr()
        _corr_matrix = matrix
        _matrix_ready = True
        logger.info(
            "correlation_guard: matrix refreshed — {}×{} (symbols: {})",
            matrix.shape[0],
            matrix.shape[1],
            list(matrix.columns),
        )
    except Exception as exc:
        logger.warning("correlation_guard: correlation computation failed: {}", exc)


# ── Public: per-pair lookup ────────────────────────────────────────────────────

def get_correlation(sym_a: str, sym_b: str) -> float:
    """
    Return the Pearson correlation coefficient between sym_a and sym_b.
    Returns 0.0 if either symbol is absent from the matrix or the matrix
    has not yet been populated.
    """
    a = _strip_ns(sym_a)
    b = _strip_ns(sym_b)

    if _corr_matrix.empty:
        return 0.0
    if a not in _corr_matrix.columns or b not in _corr_matrix.columns:
        return 0.0

    try:
        val = _corr_matrix.loc[a, b]
        # numpy scalar → plain float; guard against NaN
        result = float(val)
        return 0.0 if np.isnan(result) else result
    except Exception:
        return 0.0


# ── Public: entry guard ────────────────────────────────────────────────────────

def check(new_symbol: str, open_positions: list[str]) -> dict:
    """
    Decide whether adding `new_symbol` to the portfolio is safe given the
    currently open positions.

    Returns a dict:
        {
          "allowed":     bool,
          "reason":      str,
          "size_factor": float,   # 0.0, 0.5, 0.75, or 1.0
        }

    Never raises.
    """
    if not open_positions:
        return {
            "allowed":     True,
            "reason":      "no open positions",
            "size_factor": 1.0,
        }

    new_sym = _strip_ns(new_symbol)
    matrix_empty = _corr_matrix.empty

    max_r:          float           = -2.0   # below any real correlation
    most_correlated: Optional[str]  = None
    any_unknown:    bool            = False

    for pos in open_positions:
        pos_sym = _strip_ns(pos)
        if pos_sym == new_sym:
            continue

        if matrix_empty or new_sym not in _corr_matrix.columns or pos_sym not in _corr_matrix.columns:
            any_unknown = True
            continue

        r = get_correlation(new_sym, pos_sym)
        if r > max_r:
            max_r = r
            most_correlated = pos_sym

    # All pairs had missing data
    if most_correlated is None:
        if any_unknown:
            return {
                "allowed":     True,
                "reason":      "correlation unknown",
                "size_factor": 0.75,
            }
        # open_positions contained only the symbol itself — treat as no positions
        return {
            "allowed":     True,
            "reason":      "no open positions",
            "size_factor": 1.0,
        }

    if max_r > 0.85:
        return {
            "allowed":     False,
            "reason":      f"highly correlated with {most_correlated} (r={max_r:.2f})",
            "size_factor": 0.0,
        }

    if max_r > 0.70:
        return {
            "allowed":     True,
            "reason":      f"moderate correlation with {most_correlated} (r={max_r:.2f})",
            "size_factor": 0.5,
        }

    return {
        "allowed":     True,
        "reason":      "low correlation",
        "size_factor": 1.0,
    }


# ── Public: portfolio heat ─────────────────────────────────────────────────────

def portfolio_heat(open_positions: list[str]) -> float:
    """
    Return a 0.0-1.0 score representing the average pairwise correlation across
    all open positions.  0.0 = maximally diversified (or fewer than 2 positions),
    1.0 = all positions perfectly correlated.

    Pairs where correlation data is unavailable are excluded from the average
    (not counted as 0, which would unfairly lower the heat score).  If no pairs
    have data, returns 0.0.

    Never raises.
    """
    if len(open_positions) < 2:
        return 0.0

    syms = [_strip_ns(p) for p in open_positions]
    pairs = list(combinations(syms, 2))

    total_r = 0.0
    counted = 0

    for a, b in pairs:
        if _corr_matrix.empty:
            continue
        if a not in _corr_matrix.columns or b not in _corr_matrix.columns:
            continue
        r = get_correlation(a, b)
        if np.isnan(r):
            continue
        total_r += r
        counted += 1

    if counted == 0:
        return 0.0

    avg = total_r / counted
    # Clamp to [0, 1] — correlations can technically be negative; we treat
    # negative correlation as beneficial (heat = 0, not below 0)
    return float(max(0.0, min(1.0, avg)))
