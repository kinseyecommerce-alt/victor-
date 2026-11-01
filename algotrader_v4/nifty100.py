"""
nifty100.py
Complete Nifty 100 constituent list (Nifty 50 + Nifty Next 50).
Used as the default watchlist when USE_NIFTY100_WATCHLIST=true.
"""
from __future__ import annotations

# ── Nifty 50 ──────────────────────────────────────────────────────────────────
NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK",
    "INFY", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM",
    "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ZOMATO",
]

# ── Nifty Next 50 ─────────────────────────────────────────────────────────────
NIFTY_NEXT_50 = [
    "ABB", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "AUROPHARMA",
    "BANKBARODA", "BERGEPAINT", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "DABUR", "DMART", "DIVISLAB", "DLF",
    "GAIL", "GODREJCP", "HAL", "HAVELLS", "HINDPETRO",
    "ICICIGI", "ICICIPRULI", "INDHOTEL", "IOC", "IRCTC",
    "JIOFIN", "JUBLFOOD", "LODHA", "LTIM", "LUPIN",
    "MCDOWELL-N", "MOTHERSON", "MPHASIS", "NAUKRI", "NHPC",
    "NMDC", "OFSS", "PAGEIND", "PAYTM", "PETRONET",
    "PIDILITIND", "PIIND", "RECLTD", "SIEMENS", "TATAPOWER",
    "TORNTPHARM", "TVSMOTOR", "VBL", "VEDL", "ZYDUSLIFE",
]

# ── Combined ──────────────────────────────────────────────────────────────────
NIFTY_100 = NIFTY_50 + NIFTY_NEXT_50

# Strategy-suitability hints (used to pre-filter per-strategy watchlists)
# High-liquidity large-caps best for scalping; mid-caps better for swing
SCALPING_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "AXISBANK",
    "KOTAKBANK", "SBIN", "BHARTIARTL", "LT", "WIPRO", "HCLTECH",
    "ITC", "BAJFINANCE", "TATAMOTORS", "MARUTI", "ONGC", "NTPC",
    "POWERGRID", "HINDALCO", "TATASTEEL", "JSWSTEEL", "COALINDIA",
    "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TECHM", "M&M",
    "INDUSINDBK", "ADANIPORTS",
]

FNO_UNIVERSE = NIFTY_50  # Only Nifty 50 stocks have liquid F&O

SWING_UNIVERSE = NIFTY_100  # All 100 for swing (hold overnight)

INTRADAY_UNIVERSE = NIFTY_100  # All 100 for intraday MIS

# Watchlist dict format expected by agents
def as_watchlist(symbols: list[str], exchange: str = "NSE") -> list[dict]:
    return [{"symbol": s, "exchange": exchange} for s in symbols]

def get_strategy_watchlist(strategy: str) -> list[dict]:
    mapping = {
        "intraday":  INTRADAY_UNIVERSE,
        "scalping":  SCALPING_UNIVERSE,
        "fno":       FNO_UNIVERSE,
        "swing":     SWING_UNIVERSE,
    }
    return as_watchlist(mapping.get(strategy, NIFTY_100))
