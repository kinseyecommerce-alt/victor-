"""
backtest_engine.py
Runs any strategy on historical OHLCV data and returns pass / fail.
Only symbols that PASS can be traded by agents.

Features:
  • Walk-forward / out-of-sample validation (avoids overfitting)
  • Trade-level CSV export
  • Equity curve data (for chart generation)
  • Multi-strategy comparison
  • Scheduled weekly auto-backtest (triggered from master_agent_v5)

Usage:
    result = backtest_engine.run("RELIANCE", "NSE", "intraday")
    if result["passed"]:
        # allow this symbol for intraday trading
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
from loguru import logger

from config import settings
from market_data import yf_client


# ── Backtest result ──────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    passed: bool
    total_trades: int       = 0
    wins: int               = 0
    losses: int             = 0
    win_rate: float         = 0.0
    total_pnl: float        = 0.0
    avg_win: float          = 0.0
    avg_loss: float         = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float     = 0.0
    profit_factor: float    = 0.0
    best_trade: float       = 0.0
    worst_trade: float      = 0.0
    fail_reasons: list[str] = field(default_factory=list)
    # Walk-forward OOS metrics
    oos_win_rate: float     = 0.0
    oos_total_pnl: float    = 0.0
    oos_trades: int         = 0
    walk_forward_used: bool = False
    # Raw trade log and equity curve (not serialised by default)
    trades: list[dict]      = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self, include_trades: bool = False) -> dict:
        d = {
            "symbol":            self.symbol,
            "strategy":          self.strategy,
            "passed":            self.passed,
            "total_trades":      self.total_trades,
            "wins":              self.wins,
            "losses":            self.losses,
            "win_rate":          round(self.win_rate, 1),
            "total_pnl":         round(self.total_pnl, 0),
            "avg_win":           round(self.avg_win, 0),
            "avg_loss":          round(self.avg_loss, 0),
            "max_drawdown_pct":  round(self.max_drawdown_pct, 1),
            "sharpe_ratio":      round(self.sharpe_ratio, 2),
            "profit_factor":     round(self.profit_factor, 2),
            "best_trade":        round(self.best_trade, 0),
            "worst_trade":       round(self.worst_trade, 0),
            "fail_reasons":      self.fail_reasons,
            "walk_forward_used": self.walk_forward_used,
            "oos_win_rate":      round(self.oos_win_rate, 1),
            "oos_total_pnl":     round(self.oos_total_pnl, 0),
            "oos_trades":        self.oos_trades,
        }
        if include_trades:
            d["trades"] = self.trades
        return d

    def to_csv(self) -> str:
        """Return the trade log as a CSV string."""
        if not self.trades:
            return "trade_num,entry,exit,pnl,bars,exit_reason\n"
        rows = ["trade_num,entry,exit,pnl,bars,exit_reason"]
        for i, t in enumerate(self.trades, 1):
            rows.append(
                f"{i},{t['entry']:.2f},{t['exit']:.2f},"
                f"{t['pnl']:.2f},{t['bars']},{t['exit_reason']}"
            )
        return "\n".join(rows)

    def equity_chart_png(self) -> bytes:
        """Render equity curve as PNG bytes."""
        curve = self.equity_curve or [0.0]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                                 gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(
            f"{self.symbol} — {self.strategy.upper()} | "
            f"W:{self.win_rate:.0f}%  PnL:₹{self.total_pnl:.0f}  "
            f"Sharpe:{self.sharpe_ratio:.2f}  DD:{self.max_drawdown_pct:.1f}%",
            fontsize=11,
        )

        # Equity curve
        ax = axes[0]
        xs = list(range(len(curve)))
        ax.plot(xs, curve, color="steelblue", linewidth=1.5)
        ax.fill_between(xs, curve, 0, where=[v >= 0 for v in curve],
                        alpha=0.15, color="green")
        ax.fill_between(xs, curve, 0, where=[v < 0 for v in curve],
                        alpha=0.15, color="red")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_ylabel("Cumulative PnL (₹)")
        ax.set_xlabel("Trade #")
        ax.grid(True, alpha=0.3)

        # Drawdown
        arr = np.array(curve)
        peak = np.maximum.accumulate(arr)
        dd   = arr - peak
        axes[1].fill_between(xs, dd, 0, color="red", alpha=0.4)
        axes[1].set_ylabel("Drawdown (₹)")
        axes[1].set_xlabel("Trade #")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()


# ── Strategy parameters ──────────────────────────────────────────────────────────

STRATEGY_PARAMS = {
    "intraday": {
        "interval": "15minute",
        "sl_pct": 1.5,
        "target_pct": 3.0,
        "max_hold_bars": 20,
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
        "max_hold_bars": 15,
    },
    "scalping": {
        "interval": "5minute",
        "sl_pct": 0.3,
        "target_pct": 0.6,
        "max_hold_bars": 12,
    },
}

ALL_STRATEGIES = list(STRATEGY_PARAMS.keys())


# ── Core backtest engine ─────────────────────────────────────────────────────────

class BacktestEngine:

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], BacktestResult] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        symbol: str,
        exchange: str = "NSE",
        strategy: str = "intraday",
        lookback_days: int | None = None,
        force: bool = False,
        walk_forward: bool = True,
    ) -> BacktestResult:
        key = (symbol, strategy)
        if not force and key in self._cache:
            return self._cache[key]

        days   = lookback_days or settings.bt_lookback_days
        params = STRATEGY_PARAMS.get(strategy, STRATEGY_PARAMS["intraday"])

        df = self._fetch_data(symbol, exchange, params["interval"], days)
        if df is None or len(df) < 60:
            result = BacktestResult(
                symbol=symbol, strategy=strategy, passed=False,
                fail_reasons=[f"Insufficient data ({len(df) if df is not None else 0} bars)"]
            )
            self._cache[key] = result
            return result

        if walk_forward and len(df) >= 120:
            result = self._walk_forward_run(symbol, strategy, df, params)
        else:
            signals = self._generate_signals(df, strategy)
            trades  = self._simulate_trades(df, signals, params)
            result  = self._compute_metrics(symbol, strategy, trades)

        result = self._apply_gate(result)
        self._cache[key] = result
        logger.info(
            "Backtest {} {} → {} | trades={} win_rate={:.0f}% sharpe={:.2f} dd={:.1f}%{}",
            symbol, strategy, "PASS" if result.passed else "FAIL",
            result.total_trades, result.win_rate, result.sharpe_ratio,
            result.max_drawdown_pct,
            f" OOS={result.oos_win_rate:.0f}%/{result.oos_trades}T" if result.walk_forward_used else "",
        )
        return result

    def run_batch(
        self,
        symbols: list[dict],
        strategy: str,
        walk_forward: bool = True,
    ) -> dict[str, BacktestResult]:
        results = {}
        for item in symbols:
            sym  = item["symbol"]
            exch = item.get("exchange", "NSE")
            results[sym] = self.run(sym, exch, strategy, walk_forward=walk_forward)
        return results

    def compare_strategies(
        self,
        symbol: str,
        exchange: str = "NSE",
        lookback_days: int | None = None,
        walk_forward: bool = True,
    ) -> dict:
        """Run all 4 strategies on the same symbol and rank by Sharpe ratio."""
        results: dict[str, BacktestResult] = {}
        for strat in ALL_STRATEGIES:
            results[strat] = self.run(
                symbol, exchange, strat,
                lookback_days=lookback_days,
                force=True,
                walk_forward=walk_forward,
            )

        ranked = sorted(
            results.items(),
            key=lambda kv: (kv[1].passed, kv[1].sharpe_ratio),
            reverse=True,
        )
        best = ranked[0][0] if ranked else None

        return {
            "symbol":  symbol,
            "best_strategy": best,
            "ranking": [
                {
                    "rank":     i + 1,
                    "strategy": strat,
                    "passed":   r.passed,
                    "sharpe":   round(r.sharpe_ratio, 2),
                    "win_rate": round(r.win_rate, 1),
                    "total_pnl": round(r.total_pnl, 0),
                    "oos_win_rate": round(r.oos_win_rate, 1),
                }
                for i, (strat, r) in enumerate(ranked)
            ],
            "details": {s: r.to_dict() for s, r in results.items()},
        }

    def weekly_auto_backtest(self, universe: list[dict] | None = None) -> dict:
        """
        Run full batch across all strategies for the given universe.
        Called by the master agent every Sunday night.
        Refreshes the approved-symbols cache.
        """
        from symbol_scanner import FULL_UNIVERSE
        syms = universe or [{"symbol": s, "exchange": "NSE"} for s in FULL_UNIVERSE]
        summary: dict[str, dict] = {}
        for strat in ALL_STRATEGIES:
            res = self.run_batch(syms, strat, walk_forward=True)
            passed = [s for s, r in res.items() if r.passed]
            failed = [s for s, r in res.items() if not r.passed]
            summary[strat] = {
                "passed": passed,
                "failed": failed,
                "pass_count": len(passed),
                "fail_count": len(failed),
            }
            logger.info(
                "[weekly_bt] {} → {}/{} symbols approved",
                strat, len(passed), len(syms),
            )
        return summary

    def get_approved_symbols(self, strategy: str) -> list[str]:
        return [
            sym for (sym, strat), res in self._cache.items()
            if strat == strategy and res.passed
        ]

    def is_approved(self, symbol: str, strategy: str) -> bool:
        key = (symbol, strategy)
        return key in self._cache and self._cache[key].passed

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Backtest cache cleared")

    # ── Walk-forward ───────────────────────────────────────────────────────────

    def _walk_forward_run(
        self,
        symbol: str,
        strategy: str,
        df: pd.DataFrame,
        params: dict,
        n_splits: int = 5,
        train_frac: float = 0.70,
    ) -> BacktestResult:
        """
        Split data into n_splits windows.
        Each window: train on first train_frac, validate on remainder.
        IS result comes from the full dataset; OOS metrics come from OOS folds.
        """
        n = len(df)
        window = n // n_splits

        oos_trades_all: list[dict] = []

        for i in range(n_splits):
            start = i * window
            end   = start + window if i < n_splits - 1 else n
            fold  = df.iloc[start:end]
            split = int(len(fold) * train_frac)
            oos_df = fold.iloc[split:]
            if len(oos_df) < 20:
                continue
            sigs = self._generate_signals(oos_df.copy(), strategy)
            oos_trades_all.extend(self._simulate_trades(oos_df, sigs, params))

        # Full-data IS result (for primary metrics)
        signals = self._generate_signals(df, strategy)
        trades  = self._simulate_trades(df, signals, params)
        result  = self._compute_metrics(symbol, strategy, trades)

        # Attach OOS metrics
        if oos_trades_all:
            oos_pnls = [t["pnl"] for t in oos_trades_all]
            oos_wins = [p for p in oos_pnls if p > 0]
            result.oos_win_rate  = len(oos_wins) / len(oos_pnls) * 100
            result.oos_total_pnl = sum(oos_pnls)
            result.oos_trades    = len(oos_pnls)
        result.walk_forward_used = True
        return result

    # ── Data fetching ──────────────────────────────────────────────────────────

    _YF_INTERVAL = {
        "15minute": "15m", "5minute": "5m", "60minute": "60m",
        "day": "1d", "minute": "1m",
    }
    _YF_PERIOD = {
        10: "5d", 30: "1mo", 90: "3mo", 180: "6mo", 365: "1y",
    }

    def _fetch_data(self, symbol: str, exchange: str, interval: str, days: int):
        yf_interval = self._YF_INTERVAL.get(interval, "15m")
        period = "6mo"
        for d, p in sorted(self._YF_PERIOD.items()):
            if days <= d:
                period = p
                break
        df = yf_client.historical(symbol, exchange, yf_interval, period)
        return None if df.empty else df

    # ── Signal generation ──────────────────────────────────────────────────────

    def _generate_signals(self, df: pd.DataFrame, strategy: str) -> pd.Series:
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
            rsi    = ta.momentum.RSIIndicator(close, 14).rsi()
            atr    = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
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
                near   = abs(close.iloc[i] - ema50.iloc[i]) / ema50.iloc[i] < 0.015
                ema_up = ema20.iloc[i] > ema50.iloc[i]
                rsi_ok = 40 < rsi.iloc[i] < 60
                if near and ema_up and rsi_ok:
                    signals.iloc[i] = 1

        elif strategy == "scalping":
            ema9   = ta.trend.EMAIndicator(close, 9).ema_indicator()
            rsi    = ta.momentum.RSIIndicator(close, 7).rsi()
            vol_ma = volume.rolling(10).mean()
            for i in range(10, len(df)):
                cross = close.iloc[i-1] < ema9.iloc[i-1] and close.iloc[i] > ema9.iloc[i]
                spike = volume.iloc[i] > vol_ma.iloc[i] * 1.5
                mom   = 50 < rsi.iloc[i] < 70
                if cross and spike and mom:
                    signals.iloc[i] = 1

        return signals

    # ── Trade simulation ───────────────────────────────────────────────────────

    def _simulate_trades(
        self, df: pd.DataFrame, signals: pd.Series, params: dict
    ) -> list[dict]:
        sl_pct   = params["sl_pct"] / 100
        tgt_pct  = params["target_pct"] / 100
        max_bars = params["max_hold_bars"]

        trades: list[dict] = []
        in_trade    = False
        entry_price = 0.0
        entry_idx   = 0

        for i in range(len(df)):
            if in_trade:
                ltp       = df["close"].iloc[i]
                bars_held = i - entry_idx
                sl        = entry_price * (1 - sl_pct)
                tgt       = entry_price * (1 + tgt_pct)
                low_i     = df["low"].iloc[i]
                high_i    = df["high"].iloc[i]

                if low_i <= sl:
                    pnl = -entry_price * sl_pct
                    trades.append({"entry": entry_price, "exit": sl,
                                   "pnl": pnl, "bars": bars_held, "exit_reason": "SL"})
                    in_trade = False
                elif high_i >= tgt:
                    pnl = entry_price * tgt_pct
                    trades.append({"entry": entry_price, "exit": tgt,
                                   "pnl": pnl, "bars": bars_held, "exit_reason": "TGT"})
                    in_trade = False
                elif bars_held >= max_bars:
                    pnl = ltp - entry_price
                    trades.append({"entry": entry_price, "exit": ltp,
                                   "pnl": pnl, "bars": bars_held, "exit_reason": "TIMEOUT"})
                    in_trade = False

            elif signals.iloc[i] == 1:
                entry_price = df["close"].iloc[i]
                entry_idx   = i
                in_trade    = True

        return trades

    # ── Metrics ────────────────────────────────────────────────────────────────

    def _compute_metrics(
        self, symbol: str, strategy: str, trades: list[dict]
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(
                symbol=symbol, strategy=strategy, passed=False,
                fail_reasons=["Zero trades generated"],
            )

        pnls   = [t["pnl"] for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate  = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win   = float(np.mean(wins))              if wins   else 0.0
        avg_loss  = float(np.mean([abs(l) for l in losses])) if losses else 0.0

        pnl_arr = np.array(pnls)
        if len(pnl_arr) > 1 and pnl_arr.std() > 0:
            sharpe = float((pnl_arr.mean() / pnl_arr.std()) * math.sqrt(len(pnls)))
        else:
            sharpe = 0.0

        cumulative = np.cumsum(pnls).tolist()
        cum_arr    = np.array(cumulative)
        peak       = np.maximum.accumulate(cum_arr)
        drawdown   = peak - cum_arr
        max_dd     = float(np.max(drawdown)) if len(drawdown) else 0.0
        max_dd_pct = (max_dd / max(abs(peak.max()), 1)) * 100 if peak.max() != 0 else 0.0

        gross_profit = sum(wins)              if wins   else 0.0
        gross_loss   = sum(abs(l) for l in losses) if losses else 1.0
        pf           = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return BacktestResult(
            symbol=symbol, strategy=strategy, passed=False,
            total_trades=len(trades), wins=len(wins), losses=len(losses),
            win_rate=win_rate, total_pnl=total_pnl, avg_win=avg_win, avg_loss=avg_loss,
            max_drawdown_pct=max_dd_pct, sharpe_ratio=sharpe, profit_factor=pf,
            best_trade=max(pnls) if pnls else 0.0,
            worst_trade=min(pnls) if pnls else 0.0,
            trades=trades,
            equity_curve=cumulative,
        )

    # ── Pass/fail gate ─────────────────────────────────────────────────────────

    def _apply_gate(self, r: BacktestResult) -> BacktestResult:
        reasons: list[str] = []
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
        # Walk-forward OOS gate: OOS win rate must not be below 80% of IS win rate
        if r.walk_forward_used and r.oos_trades >= 5:
            oos_floor = r.win_rate * 0.80
            if r.oos_win_rate < oos_floor:
                reasons.append(
                    f"OOS win rate degraded ({r.oos_win_rate:.0f}% < {oos_floor:.0f}% floor)"
                )
        r.passed      = len(reasons) == 0
        r.fail_reasons = reasons
        return r


backtest_engine = BacktestEngine()
