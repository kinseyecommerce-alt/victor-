"""
backtest_engine.py
Runs any strategy on historical OHLCV data and returns pass / fail.
Only symbols that PASS can be traded by agents.

Usage:
    result = backtest_engine.run("RELIANCE", "NSE", "intraday")
    if result["passed"]:
        # allow this symbol for intraday trading
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta
from loguru import logger

from config import settings
from market_data import yf_client


# ── Backtest result ─────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    passed: bool
    total_trades: int     = 0
    wins: int             = 0
    losses: int           = 0
    win_rate: float       = 0.0
    total_pnl: float      = 0.0
    avg_win: float        = 0.0
    avg_loss: float       = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float   = 0.0
    profit_factor: float  = 0.0
    best_trade: float     = 0.0
    worst_trade: float    = 0.0
    fail_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol":           self.symbol,
            "strategy":         self.strategy,
            "passed":           self.passed,
            "total_trades":     self.total_trades,
            "wins":             self.wins,
            "losses":           self.losses,
            "win_rate":         round(self.win_rate, 1),
            "total_pnl":        round(self.total_pnl, 0),
            "avg_win":          round(self.avg_win, 0),
            "avg_loss":         round(self.avg_loss, 0),
            "max_drawdown_pct": round(self.max_drawdown_pct, 1),
            "sharpe_ratio":     round(self.sharpe_ratio, 2),
            "profit_factor":    round(self.profit_factor, 2),
            "best_trade":       round(self.best_trade, 0),
            "worst_trade":      round(self.worst_trade, 0),
            "fail_reasons":     self.fail_reasons,
        }


# ── Strategy parameters for backtesting ────────────────────────────────────────

STRATEGY_PARAMS = {
    "intraday": {
        "interval": "15minute",
        "sl_pct": 1.5,
        "target_pct": 3.0,
        "max_hold_bars": 20,       # ~5 hours on 15min
    },
    "fno": {
        "interval": "60minute",
        "sl_pct": 5.0,
        "target_pct": 10.0,
        "max_hold_bars": 40,
    },
    "swing": {
        "interval": "day",
        "sl_pct": 3.0,
        "target_pct": 7.0,
        "max_hold_bars": 15,       # ~15 trading days
    },
    "scalping": {
        "interval": "5minute",
        "sl_pct": 0.3,
        "target_pct": 0.6,
        "max_hold_bars": 12,       # ~1 hour on 5min
    },
}


# ── Core backtest engine ───────────────────────────────────────────────────────

class BacktestEngine:

    def __init__(self) -> None:
        # Cache: (symbol, strategy) → BacktestResult
        self._cache: dict[tuple[str, str], BacktestResult] = {}

    def run(
        self,
        symbol: str,
        exchange: str = "NSE",
        strategy: str = "intraday",
        lookback_days: int | None = None,
        force: bool = False,
    ) -> BacktestResult:
        key = (symbol, strategy)
        if not force and key in self._cache:
            return self._cache[key]

        days = lookback_days or settings.bt_lookback_days
        params = STRATEGY_PARAMS.get(strategy, STRATEGY_PARAMS["intraday"])

        df = self._fetch_data(symbol, exchange, params["interval"], days)
        if df is None or len(df) < 60:
            result = BacktestResult(
                symbol=symbol, strategy=strategy, passed=False,
                fail_reasons=[f"Insufficient data ({len(df) if df is not None else 0} bars)"]
            )
            self._cache[key] = result
            return result

        signals = self._generate_signals(df, strategy)
        trades = self._simulate_trades(df, signals, params)
        result = self._compute_metrics(symbol, strategy, trades)
        result = self._apply_gate(result)

        self._cache[key] = result
        logger.info(
            "Backtest {} {} → {} | trades={} win_rate={:.0f}% sharpe={:.2f} dd={:.1f}%",
            symbol, strategy, "PASS" if result.passed else "FAIL",
            result.total_trades, result.win_rate, result.sharpe_ratio, result.max_drawdown_pct,
        )
        return result

    def run_batch(self, symbols: list[dict], strategy: str) -> dict[str, BacktestResult]:
        results = {}
        for item in symbols:
            sym = item["symbol"]
            exch = item.get("exchange", "NSE")
            results[sym] = self.run(sym, exch, strategy)
        return results

    def get_approved_symbols(self, strategy: str) -> list[str]:
        return [
            sym for (sym, strat), res in self._cache.items()
            if strat == strategy and res.passed
        ]

    def is_approved(self, symbol: str, strategy: str) -> bool:
        key = (symbol, strategy)
        return key in self._cache and self._cache[key].passed

    _YF_INTERVAL = {
        "15minute": "15m", "5minute": "5m", "60minute": "60m",
        "day": "1d", "minute": "1m",
    }
    _YF_PERIOD = {
        10: "5d", 30: "1mo", 90: "3mo", 180: "6mo", 365: "1y",
    }

    def _fetch_data(self, symbol, exchange, interval, days):
        yf_interval = self._YF_INTERVAL.get(interval, "15m")
        period = "6mo"
        for d, p in sorted(self._YF_PERIOD.items()):
            if days <= d:
                period = p
                break
        df = yf_client.historical(symbol, exchange, yf_interval, period)
        if df.empty:
            return None
        return df

    def _generate_signals(self, df, strategy):
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        signals = pd.Series(0, index=df.index)

        if strategy == "intraday":
            ema9  = ta.trend.EMAIndicator(close, 9).ema_indicator()
            ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
            rsi   = ta.momentum.RSIIndicator(close, 14).rsi()
            vwap  = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
            for i in range(2, len(df)):
                vwap_cross = close.iloc[i-1] < vwap.iloc[i-1] and close.iloc[i] > vwap.iloc[i]
                ema_bull   = ema9.iloc[i] > ema21.iloc[i]
                rsi_ok     = 45 < rsi.iloc[i] < 65
                if vwap_cross and ema_bull and rsi_ok:
                    signals.iloc[i] = 1

        elif strategy == "fno":
            rsi = ta.momentum.RSIIndicator(close, 14).rsi()
            atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
            atr_ma = atr.rolling(30).mean()
            for i in range(30, len(df)):
                iv_proxy = (atr.iloc[i] / atr_ma.iloc[i] * 50) if atr_ma.iloc[i] else 50
                if iv_proxy < 40 and rsi.iloc[i] < 40:
                    signals.iloc[i] = 1
                elif iv_proxy < 40 and rsi.iloc[i] > 60:
                    signals.iloc[i] = 1

        elif strategy == "swing":
            ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator()
            ema20 = ta.trend.EMAIndicator(close, 20).ema_indicator()
            rsi   = ta.momentum.RSIIndicator(close, 14).rsi()
            for i in range(50, len(df)):
                near = abs(close.iloc[i] - ema50.iloc[i]) / ema50.iloc[i] < 0.015
                ema_up = ema20.iloc[i] > ema50.iloc[i]
                rsi_ok = 40 < rsi.iloc[i] < 60
                if near and ema_up and rsi_ok:
                    signals.iloc[i] = 1

        elif strategy == "scalping":
            ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
            rsi  = ta.momentum.RSIIndicator(close, 7).rsi()
            vol_ma = volume.rolling(10).mean()
            for i in range(10, len(df)):
                cross = close.iloc[i-1] < ema9.iloc[i-1] and close.iloc[i] > ema9.iloc[i]
                spike = volume.iloc[i] > vol_ma.iloc[i] * 1.5
                mom   = 50 < rsi.iloc[i] < 70
                if cross and spike and mom:
                    signals.iloc[i] = 1

        return signals

    def _simulate_trades(self, df, signals, params):
        sl_pct   = params["sl_pct"] / 100
        tgt_pct  = params["target_pct"] / 100
        max_bars = params["max_hold_bars"]

        trades = []
        in_trade = False
        entry_price = 0.0
        entry_idx = 0

        for i in range(len(df)):
            if in_trade:
                ltp = df["close"].iloc[i]
                bars_held = i - entry_idx
                sl    = entry_price * (1 - sl_pct)
                tgt   = entry_price * (1 + tgt_pct)
                low_i = df["low"].iloc[i]
                high_i = df["high"].iloc[i]

                if low_i <= sl:
                    pnl = -entry_price * sl_pct
                    trades.append({"entry": entry_price, "exit": sl, "pnl": pnl, "bars": bars_held, "exit_reason": "SL"})
                    in_trade = False
                elif high_i >= tgt:
                    pnl = entry_price * tgt_pct
                    trades.append({"entry": entry_price, "exit": tgt, "pnl": pnl, "bars": bars_held, "exit_reason": "TGT"})
                    in_trade = False
                elif bars_held >= max_bars:
                    pnl = ltp - entry_price
                    trades.append({"entry": entry_price, "exit": ltp, "pnl": pnl, "bars": bars_held, "exit_reason": "TIMEOUT"})
                    in_trade = False

            elif signals.iloc[i] == 1:
                entry_price = df["close"].iloc[i]
                entry_idx = i
                in_trade = True

        return trades

    def _compute_metrics(self, symbol, strategy, trades):
        if not trades:
            return BacktestResult(
                symbol=symbol, strategy=strategy, passed=False,
                fail_reasons=["Zero trades generated"]
            )

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate  = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win   = np.mean(wins)   if wins   else 0
        avg_loss  = np.mean([abs(l) for l in losses]) if losses else 0

        if len(pnls) > 1:
            pnl_arr = np.array(pnls)
            sharpe = (pnl_arr.mean() / pnl_arr.std()) * math.sqrt(len(pnls)) if pnl_arr.std() > 0 else 0
        else:
            sharpe = 0

        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0
        max_dd_pct = (max_dd / max(abs(peak.max()), 1)) * 100 if peak.max() != 0 else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss   = sum(abs(l) for l in losses) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return BacktestResult(
            symbol=symbol, strategy=strategy, passed=False,
            total_trades=len(trades), wins=len(wins), losses=len(losses),
            win_rate=win_rate, total_pnl=total_pnl, avg_win=avg_win, avg_loss=avg_loss,
            max_drawdown_pct=max_dd_pct, sharpe_ratio=sharpe, profit_factor=profit_factor,
            best_trade=max(pnls) if pnls else 0, worst_trade=min(pnls) if pnls else 0,
        )

    def _apply_gate(self, r):
        reasons = []
        if r.total_trades < settings.bt_min_trades:
            reasons.append(f"Too few trades ({r.total_trades} < {settings.bt_min_trades})")
        if r.win_rate < settings.bt_min_win_rate:
            reasons.append(f"Win rate too low ({r.win_rate:.0f}% < {settings.bt_min_win_rate:.0f}%)")
        if r.sharpe_ratio < settings.bt_min_sharpe:
            reasons.append(f"Sharpe too low ({r.sharpe_ratio:.2f} < {settings.bt_min_sharpe})")
        if r.max_drawdown_pct > settings.bt_max_drawdown_pct:
            reasons.append(f"Drawdown too high ({r.max_drawdown_pct:.1f}% > {settings.bt_max_drawdown_pct:.0f}%)")
        if r.total_pnl <= 0:
            reasons.append(f"Negative total P&L (₹{r.total_pnl:.0f})")
        r.passed = len(reasons) == 0
        r.fail_reasons = reasons
        return r

    def clear_cache(self):
        self._cache.clear()
        logger.info("Backtest cache cleared")


backtest_engine = BacktestEngine()