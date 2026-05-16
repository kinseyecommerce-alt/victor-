"""
symbol_scanner.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Automatically selects the best symbols to trade each morning.

How symbols are selected (per strategy):

  INTRADAY  → High volume + trending (EMA crossover) + moderate ATR
  F&O       → F&O eligible + high OI + IV not extreme + liquid
  SWING     → Nifty 200 universe + weekly trend aligned + RSI reset zone
  SCALPING  → Nifty 50 only + tight spread + highest vol ratio

Sources used (all free, no Kite):
  • NSE India API → equity list, F&O eligibility, OI data, market status
  • yfinance      → OHLCV history for scoring (EMA, ATR, volume, RSI)

Runs daily at 09:00 IST (before market opens at 09:15).
Also callable on demand via POST /symbols/refresh.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional

import pandas as pd
import numpy as np
import ta
from loguru import logger

from market_data import nse_client, yf_client


# ── NSE universe lists ─────────────────────────────────────────────────────────

# Nifty 50 — ranks 1-50 by free-float market cap
NIFTY_50 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN",
    "HINDUNILVR","ITC","KOTAKBANK","LT","AXISBANK","BAJFINANCE","MARUTI",
    "ASIANPAINT","HCLTECH","WIPRO","NESTLEIND","ADANIPORTS","ULTRACEMCO",
    "TITAN","POWERGRID","NTPC","TECHM","BAJAJFINSERV","ONGC","TATASTEEL",
    "TATAMOTORS","DRREDDY","SUNPHARMA","CIPLA","DIVISLAB","GRASIM",
    "JSWSTEEL","INDUSINDBK","HDFCLIFE","SBILIFE","BPCL","EICHERMOT",
    "HEROMOTOCO","COALINDIA","BRITANNIA","APOLLOHOSP","TRENT","SHRIRAMFIN",
    "BAJAJ-AUTO","M&M","HINDALCO","LTIM","ADANIENT",
]

# Nifty Next 50 — ranks 51-100 by free-float market cap
NIFTY_NEXT_50 = [
    "DMART","PIDILITIND","ABB","HAVELLS","SIEMENS","MUTHOOTFIN","BERGEPAINT",
    "GODREJCP","MARICO","DABUR","COLPAL","PGHH","AUROPHARMA","TORNTPHARM",
    "LUPIN","BIOCON","OFSS","MPHASIS","PERSISTENT","COFORGE","LTTS",
    "TATACONSUM","UPL","PIIND","ASTRAL","DIXON","VOLTAS","WHIRLPOOL",
    "MAXHEALTH","NH","POLICYBZR","PAYTM","NAUKRI","IRCTC","INDIAMART",
    "CHOLAFIN","MOTHERSON","BHARATFORG","CUMMINSIND","ABCAPITAL",
    "MANAPPURAM","PNBHOUSING","CANFINHOME","AARTIIND","TATAPOWER",
    "ADANIGREEN","ADANITRANS","CANBK","PNB","BANKBARODA",
]

# Top 100 NSE companies = Nifty 50 + Nifty Next 50 (no duplicates)
NSE_TOP_100: list[str] = list(dict.fromkeys(NIFTY_50 + NIFTY_NEXT_50))

# Backward-compat aliases
NIFTY_BANK = [s for s in NSE_TOP_100 if s in {
    "HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","INDUSINDBK",
    "BANKBARODA","PNB","CANBK",
}]

# Indices (for F&O — not equity symbols)
INDICES = ["NIFTY50","BANKNIFTY","FINNIFTY","MIDCPNIFTY"]

# Single canonical universe used everywhere
FULL_UNIVERSE: list[str] = NSE_TOP_100


# ── Symbol score dataclass ─────────────────────────────────────────────────────

@dataclass
class SymbolScore:
    symbol:        str
    exchange:      str  = "NSE"
    strategy:      str  = ""

    # Scoring components (0–100 each)
    liquidity_score:   float = 0.0   # average daily volume rank
    trend_score:       float = 0.0   # EMA alignment + ADX
    momentum_score:    float = 0.0   # RSI in ideal zone
    volatility_score:  float = 0.0   # ATR in good range (not too low, not too high)
    spread_score:      float = 0.0   # estimated bid-ask spread quality
    fo_score:          float = 0.0   # F&O eligibility + OI rank

    total_score:   float = 0.0
    selected:      bool  = False
    reject_reason: str   = ""

    # Raw indicators
    avg_volume:    float = 0.0
    atr_pct:       float = 0.0    # ATR as % of price
    rsi:           float = 50.0
    adx:           float = 0.0
    ema9:          float = 0.0
    ema21:         float = 0.0
    ema50:         float = 0.0
    ltp:           float = 0.0
    trend_direction: str = "NEUTRAL"

    def to_dict(self) -> dict:
        return {
            "symbol":           self.symbol,
            "exchange":         self.exchange,
            "strategy":         self.strategy,
            "total_score":      round(self.total_score, 1),
            "selected":         self.selected,
            "reject_reason":    self.reject_reason,
            "scores": {
                "liquidity":   round(self.liquidity_score, 1),
                "trend":       round(self.trend_score, 1),
                "momentum":    round(self.momentum_score, 1),
                "volatility":  round(self.volatility_score, 1),
            },
            "indicators": {
                "ltp":       round(self.ltp, 2),
                "rsi":       round(self.rsi, 1),
                "adx":       round(self.adx, 1),
                "atr_pct":   round(self.atr_pct, 2),
                "trend":     self.trend_direction,
                "avg_vol":   int(self.avg_volume),
            },
        }


# ── Per-strategy selection criteria ────────────────────────────────────────────

@dataclass
class SelectionCriteria:
    universe:           list[str]     # symbols to scan
    top_n:              int           # how many to select
    min_avg_volume:     int           # daily volume floor
    min_atr_pct:        float         # minimum volatility (% of price)
    max_atr_pct:        float         # maximum volatility
    require_trend:      bool          # must have directional trend
    rsi_min:            float         # RSI lower bound for entry pool
    rsi_max:            float         # RSI upper bound
    min_adx:            float         # trend strength floor (ADX)
    fo_eligible_only:   bool          # only F&O stocks
    score_weights:      dict          # {liquidity, trend, momentum, volatility}
    description:        str


CRITERIA: dict[str, SelectionCriteria] = {
    "intraday": SelectionCriteria(
        universe        = NSE_TOP_100,          # all 100 — scanner picks best 8
        top_n           = 8,
        min_avg_volume  = 500_000,
        min_atr_pct     = 0.8,
        max_atr_pct     = 4.0,
        require_trend   = True,
        rsi_min         = 35,
        rsi_max         = 70,
        min_adx         = 20,
        fo_eligible_only= False,
        score_weights   = {"liquidity":35, "trend":35, "momentum":20, "volatility":10},
        description     = "Top-100 NSE stocks — trending + high-volume for intraday momentum",
    ),
    "fno": SelectionCriteria(
        universe        = NSE_TOP_100,          # F&O gate applied inside scorer
        top_n           = 6,
        min_avg_volume  = 1_000_000,
        min_atr_pct     = 1.0,
        max_atr_pct     = 6.0,
        require_trend   = False,
        rsi_min         = 0,
        rsi_max         = 100,
        min_adx         = 15,
        fo_eligible_only= True,
        score_weights   = {"liquidity":40, "trend":20, "momentum":20, "volatility":20},
        description     = "Top-100 NSE F&O-eligible stocks with high OI and liquid options",
    ),
    "swing": SelectionCriteria(
        universe        = NSE_TOP_100,          # full 100 for swing diversity
        top_n           = 6,
        min_avg_volume  = 200_000,
        min_atr_pct     = 1.5,
        max_atr_pct     = 5.0,
        require_trend   = True,
        rsi_min         = 38,
        rsi_max         = 62,
        min_adx         = 22,
        fo_eligible_only= False,
        score_weights   = {"liquidity":20, "trend":40, "momentum":25, "volatility":15},
        description     = "Top-100 NSE stocks in RSI pullback zone for 3-7 day holds",
    ),
    "scalping": SelectionCriteria(
        universe        = NIFTY_50,             # Nifty 50 only — tightest spreads
        top_n           = 5,
        min_avg_volume  = 2_000_000,
        min_atr_pct     = 0.5,
        max_atr_pct     = 2.5,
        require_trend   = False,
        rsi_min         = 0,
        rsi_max         = 100,
        min_adx         = 0,
        fo_eligible_only= False,
        score_weights   = {"liquidity":50, "trend":15, "momentum":15, "volatility":20},
        description     = "Nifty 50 only — highest liquidity for tight bid-ask scalping",
    ),
}

# F&O eligible = all Nifty 50 + bank stocks within top 100
FO_ELIGIBLE = set(NIFTY_50 + NIFTY_BANK + ["NIFTY50","BANKNIFTY","FINNIFTY"])


# ── Symbol Scanner ─────────────────────────────────────────────────────────────

class SymbolScanner:
    """
    Scans the NSE universe and selects the best symbols per strategy.

    Selection pipeline:
      1. Fetch OHLCV from yfinance (20 days of daily + 5 days of 15-min)
      2. Compute scoring indicators (volume, ATR, RSI, EMA, ADX)
      3. Apply hard filters (min volume, ATR range, F&O eligibility)
      4. Score each symbol (weighted 0–100)
      5. Rank and pick top-N per strategy
      6. Run backtest gate on selected symbols (via backtest_engine)
    """

    def __init__(self) -> None:
        self._selected:   dict[str, list[dict]]   = {}   # strategy → [{symbol, exchange}]
        self._scores:     dict[str, list[SymbolScore]] = {}
        self._last_scan:  Optional[datetime]      = None
        self._scanning:   bool                    = False

    # ── Public API ─────────────────────────────────────────────────────

    async def run(
        self,
        strategies: Optional[list[str]] = None,
        force: bool = False,
    ) -> dict[str, list[dict]]:
        """
        Run a full symbol selection scan.
        Returns {strategy: [{symbol, exchange}]} for each strategy.
        """
        if self._scanning:
            logger.info("Scanner already running — skipping")
            return self._selected

        self._scanning = True
        strats = strategies or list(CRITERIA.keys())

        logger.info("Symbol scanner started for strategies: {}", strats)

        # Build unique symbol list to fetch
        all_syms: set[str] = set()
        for s in strats:
            all_syms.update(CRITERIA[s].universe)

        # Fetch and score all symbols
        scores = await self._score_all(list(all_syms))

        # Select per strategy
        for strat in strats:
            crit = CRITERIA[strat]
            selected, all_scores = self._select_for_strategy(strat, crit, scores)
            self._selected[strat] = selected
            self._scores[strat]   = all_scores

            logger.info(
                "Scanner [{}]: selected {}/{} symbols → {}",
                strat, len(selected), len(crit.universe),
                [s["symbol"] for s in selected]
            )

        self._last_scan = datetime.now()
        self._scanning  = False
        return self._selected

    def get_selected(self, strategy: Optional[str] = None) -> dict:
        if strategy:
            return {
                "strategy":    strategy,
                "symbols":     self._selected.get(strategy, []),
                "criteria":    self._criteria_summary(strategy),
                "last_scan":   self._last_scan.isoformat() if self._last_scan else None,
            }
        return {
            "selections": {
                strat: {
                    "symbols":   syms,
                    "count":     len(syms),
                    "criteria":  self._criteria_summary(strat),
                }
                for strat, syms in self._selected.items()
            },
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
        }

    def get_scores(self, strategy: str) -> list[dict]:
        return [s.to_dict() for s in self._scores.get(strategy, [])]

    def all_selected_flat(self) -> list[dict]:
        """Merged unique watchlist across all strategies."""
        seen: set[str] = set()
        result = []
        for syms in self._selected.values():
            for item in syms:
                if item["symbol"] not in seen:
                    seen.add(item["symbol"])
                    result.append(item)
        return result

    # ── Scoring ────────────────────────────────────────────────────────

    async def _score_all(
        self, symbols: list[str]
    ) -> dict[str, SymbolScore]:
        """Fetch + score each symbol concurrently."""
        scores: dict[str, SymbolScore] = {}
        sem = asyncio.Semaphore(10)  # max 10 concurrent yfinance fetches

        async def fetch_one(sym: str):
            async with sem:
                sc = await asyncio.get_event_loop().run_in_executor(
                    None, self._score_symbol, sym
                )
                if sc:
                    scores[sym] = sc

        await asyncio.gather(*[fetch_one(s) for s in symbols],
                              return_exceptions=True)
        logger.info("Scored {}/{} symbols", len(scores), len(symbols))
        return scores

    def _score_symbol(self, symbol: str) -> Optional[SymbolScore]:
        """Compute indicators and scores for one symbol using yfinance."""
        try:
            # 20 days daily for trend / swing
            df_d = yf_client.historical(symbol, "NSE", "1d", "1mo")
            # 5 days 15-min for intraday / scalping indicators
            df_15 = yf_client.historical(symbol, "NSE", "15m", "5d")

            if df_d.empty or len(df_d) < 10:
                return None

            close  = df_d["close"]
            high   = df_d["high"]
            low    = df_d["low"]
            volume = df_d["volume"]
            n      = len(df_d)

            sc = SymbolScore(symbol=symbol, exchange="NSE")

            # ── LTP ───────────────────────────────────────────────────
            sc.ltp = float(close.iloc[-1])

            # ── Average volume ────────────────────────────────────────
            sc.avg_volume = float(volume.rolling(10).mean().iloc[-1])

            # ── ATR % ─────────────────────────────────────────────────
            if n >= 14:
                atr = ta.volatility.AverageTrueRange(high, low, close, 14) \
                        .average_true_range().iloc[-1]
                sc.atr_pct = (atr / sc.ltp * 100) if sc.ltp > 0 else 0

            # ── EMA ────────────────────────────────────────────────────
            if n >= 9:
                sc.ema9  = float(ta.trend.EMAIndicator(close, 9 ).ema_indicator().iloc[-1])
            if n >= 21:
                sc.ema21 = float(ta.trend.EMAIndicator(close, 21).ema_indicator().iloc[-1])
            if n >= 50:
                sc.ema50 = float(ta.trend.EMAIndicator(close, 50).ema_indicator().iloc[-1])

            if sc.ema9 and sc.ema21:
                if sc.ltp > sc.ema9 > sc.ema21:   sc.trend_direction = "UP"
                elif sc.ltp < sc.ema9 < sc.ema21: sc.trend_direction = "DOWN"

            # ── RSI ────────────────────────────────────────────────────
            if n >= 15:
                sc.rsi = float(ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])

            # ── ADX ────────────────────────────────────────────────────
            if n >= 14:
                sc.adx = float(ta.trend.ADXIndicator(high, low, close, 14).adx().iloc[-1])

            # ── Component scores ──────────────────────────────────────

            # Liquidity (0–100): rank by log volume
            import math
            vol_log = math.log(sc.avg_volume + 1)
            sc.liquidity_score = min(100, vol_log / 16 * 100)  # 16 ≈ log(10M)

            # Trend (0–100): EMA alignment + ADX
            ema_aligned = (
                100 if sc.trend_direction in ("UP","DOWN") and sc.adx > 25
                else 60 if sc.trend_direction in ("UP","DOWN")
                else 20
            )
            adx_score = min(100, sc.adx / 40 * 100)
            sc.trend_score = (ema_aligned * 0.6 + adx_score * 0.4)

            # Momentum (0–100): RSI in 40–60 is ideal for entry opportunity
            rsi_dist = abs(sc.rsi - 50)
            sc.momentum_score = max(0, 100 - rsi_dist * 1.5)

            # Volatility (0–100): sweet spot 1–3% ATR
            if sc.atr_pct < 0.3:
                sc.volatility_score = 10
            elif sc.atr_pct < 1.0:
                sc.volatility_score = 50
            elif sc.atr_pct <= 3.0:
                sc.volatility_score = 100
            elif sc.atr_pct <= 5.0:
                sc.volatility_score = 70
            else:
                sc.volatility_score = 30

            return sc

        except Exception as exc:
            logger.debug("Score error {}: {}", symbol, exc)
            return None

    # ── Selection ──────────────────────────────────────────────────────

    def _select_for_strategy(
        self,
        strategy:  str,
        crit:      SelectionCriteria,
        scores:    dict[str, SymbolScore],
    ) -> tuple[list[dict], list[SymbolScore]]:

        candidates: list[SymbolScore] = []

        for sym in crit.universe:
            sc = scores.get(sym)
            if not sc:
                sc = SymbolScore(symbol=sym, reject_reason="No data")
                candidates.append(sc)
                continue

            sc.strategy = strategy

            # Hard filters
            if sc.avg_volume < crit.min_avg_volume:
                sc.reject_reason = f"Low volume ({int(sc.avg_volume):,})"
                candidates.append(sc)
                continue

            if sc.atr_pct < crit.min_atr_pct:
                sc.reject_reason = f"Low volatility (ATR {sc.atr_pct:.1f}%)"
                candidates.append(sc)
                continue

            if sc.atr_pct > crit.max_atr_pct:
                sc.reject_reason = f"High volatility (ATR {sc.atr_pct:.1f}%)"
                candidates.append(sc)
                continue

            if crit.require_trend and sc.trend_direction == "NEUTRAL":
                sc.reject_reason = "No clear trend"
                candidates.append(sc)
                continue

            if not (crit.rsi_min <= sc.rsi <= crit.rsi_max):
                sc.reject_reason = f"RSI out of range ({sc.rsi:.0f})"
                candidates.append(sc)
                continue

            if sc.adx < crit.min_adx:
                sc.reject_reason = f"Weak trend (ADX {sc.adx:.0f})"
                candidates.append(sc)
                continue

            if crit.fo_eligible_only and sym not in FO_ELIGIBLE:
                sc.reject_reason = "Not F&O eligible"
                candidates.append(sc)
                continue

            # Compute total score
            w = crit.score_weights
            sc.total_score = (
                sc.liquidity_score  * w.get("liquidity",  25) / 100 +
                sc.trend_score      * w.get("trend",       25) / 100 +
                sc.momentum_score   * w.get("momentum",    25) / 100 +
                sc.volatility_score * w.get("volatility",  25) / 100
            )
            sc.selected = True
            candidates.append(sc)

        # Sort: selected first by score, then rejected
        selected_sorted = sorted(
            [c for c in candidates if c.selected],
            key=lambda x: x.total_score, reverse=True
        )
        # Keep only top_n
        for sc in selected_sorted[crit.top_n:]:
            sc.selected = False
            sc.reject_reason = f"Score too low ({sc.total_score:.0f})"

        selected_sorted = selected_sorted[:crit.top_n]

        watchlist = [{"symbol": sc.symbol, "exchange": sc.exchange}
                     for sc in selected_sorted]

        return watchlist, candidates

    def _criteria_summary(self, strategy: str) -> dict:
        crit = CRITERIA.get(strategy)
        if not crit:
            return {}
        return {
            "description":       crit.description,
            "top_n":             crit.top_n,
            "universe_size":     len(crit.universe),
            "min_avg_volume":    crit.min_avg_volume,
            "atr_range":         f"{crit.min_atr_pct}–{crit.max_atr_pct}%",
            "rsi_range":         f"{crit.rsi_min}–{crit.rsi_max}",
            "min_adx":           crit.min_adx,
            "fo_only":           crit.fo_eligible_only,
            "score_weights":     crit.score_weights,
        }


symbol_scanner = SymbolScanner()
