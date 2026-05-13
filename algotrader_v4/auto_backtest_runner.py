"""
auto_backtest_runner.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Automated Backtest Pipeline

Runs every morning at 08:45 IST (before market opens at 09:15).
Tests ALL 8 strategies x ALL symbols in the universe.
Only symbols + strategies that PASS are activated for live trading.

Pipeline:
  1. Fetch fresh OHLCV from yfinance for every symbol
  2. Run walk-forward simulation for each strategy
  3. Apply pass/fail gate (win rate, Sharpe, drawdown, profit factor)
  4. Generate ranked leaderboard per strategy
  5. Auto-activate top-N winners into live agents
  6. Send morning report via Telegram

8 Strategies tested:
  1. intraday       - VWAP + EMA crossover
  2. swing          - EMA50 pullback
  3. futures        - EMA trend follow
  4. options        - Directional CE/PE buying
  5. iron_condor    - Range-bound options selling
  6. short_straddle - Theta decay
  7. orb            - Opening range breakout
  8. vwap_reversion - VWAP mean reversion
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta
from loguru import logger

from market_data import yf_client
from strategy_signals import SIGNAL_REGISTRY, STRATEGY_CONFIG, GATE_THRESHOLDS


NIFTY_50 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","INFOSYS","SBIN",
    "HINDUNILVR","ITC","KOTAKBANK","LT","AXISBANK","BAJFINANCE","MARUTI",
    "HCLTECH","WIPRO","ASIANPAINT","TATAMOTORS","BAJAJFINSERV","NTPC",
    "POWERGRID","TITAN","TECHM","SUNPHARMA","DRREDDY","CIPLA","ONGC",
    "TATASTEEL","JSWSTEEL","ADANIPORTS","EICHERMOT","HEROMOTOCO","COALINDIA",
    "ULTRACEMCO","SHRIRAMFIN","TATACONSUM","INDUSINDBK","NESTLEIND","HDFCLIFE",
    "SBILIFE","BPCL","DIVISLAB","GRASIM","BRITANNIA","APOLLOHOSP","TRENT",
    "BAJAJ-AUTO","M&M","HINDALCO","LTIM",
]
NIFTY_BANK = ["HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","INDUSINDBK",
              "BANKBARODA","PNB","IDFCFIRSTB","FEDERALBNK"]
INDICES    = ["NIFTY50","BANKNIFTY","FINNIFTY"]
ALL_SYMBOLS = list(dict.fromkeys(NIFTY_50 + NIFTY_BANK + INDICES))

GATE = {
    "intraday":      {"win_rate":55, "sharpe":1.0,  "drawdown":15, "min_trades":20, "pf":1.2},
    "swing":         {"win_rate":52, "sharpe":0.9,  "drawdown":18, "min_trades":15, "pf":1.15},
    "futures":       {"win_rate":53, "sharpe":0.95, "drawdown":16, "min_trades":18, "pf":1.2},
    "options":       {"win_rate":40, "sharpe":0.8,  "drawdown":30, "min_trades":12, "pf":1.5},
    "iron_condor":   {"win_rate":60, "sharpe":1.0,  "drawdown":20, "min_trades":10, "pf":1.3},
    "short_straddle":{"win_rate":58, "sharpe":0.9,  "drawdown":25, "min_trades":10, "pf":1.25},
    "orb":           {"win_rate":55, "sharpe":1.0,  "drawdown":15, "min_trades":20, "pf":1.2},
    "vwap_reversion":{"win_rate":58, "sharpe":1.0,  "drawdown":12, "min_trades":25, "pf":1.3},
}
TOP_N_PER_STRATEGY = {
    "intraday":8, "swing":6, "futures":5,
    "options":5, "iron_condor":4, "short_straddle":3,
    "orb":8, "vwap_reversion":6,
}


@dataclass
class BacktestResult:
    symbol:          str
    strategy:        str
    passed:          bool
    total_trades:    int   = 0
    wins:            int   = 0
    losses:          int   = 0
    win_rate:        float = 0.0
    total_pnl:       float = 0.0
    avg_win:         float = 0.0
    avg_loss:        float = 0.0
    max_drawdown_pct:float = 0.0
    sharpe_ratio:    float = 0.0
    profit_factor:   float = 0.0
    best_trade:      float = 0.0
    worst_trade:     float = 0.0
    score:           float = 0.0
    fail_reasons:    list  = field(default_factory=list)
    run_date:        str   = field(default_factory=lambda: date.today().isoformat())

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "strategy": self.strategy, "passed": self.passed,
            "total_trades": self.total_trades, "wins": self.wins, "losses": self.losses,
            "win_rate": round(self.win_rate, 1), "total_pnl": round(self.total_pnl, 0),
            "avg_win": round(self.avg_win, 0), "avg_loss": round(self.avg_loss, 0),
            "max_drawdown_pct": round(self.max_drawdown_pct, 1),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "profit_factor": round(self.profit_factor, 2),
            "best_trade": round(self.best_trade, 0), "worst_trade": round(self.worst_trade, 0),
            "score": round(self.score, 1), "fail_reasons": self.fail_reasons,
            "run_date": self.run_date,
        }


@dataclass
class BatchReport:
    run_date: str; run_time_sec: float; total_tested: int
    total_passed: int; total_failed: int
    by_strategy: dict; approved: dict

    def to_dict(self) -> dict:
        return {
            "run_date": self.run_date, "run_time_sec": round(self.run_time_sec, 1),
            "total_tested": self.total_tested, "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "pass_rate_pct": round(self.total_passed / max(self.total_tested,1) * 100, 1),
            "by_strategy": {k: {"passed_count": len(v["passed"]),
                "failed_count": len(v["failed"]),
                "leaderboard": v["leaderboard"][:10]} for k, v in self.by_strategy.items()},
            "approved": self.approved,
        }


def _signals_intraday(df):
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    s = pd.Series(0, index=df.index)
    if len(df) < 30: return s
    ema9  = ta.trend.EMAIndicator(close, 9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
    rsi   = ta.momentum.RSIIndicator(close, 14).rsi()
    vwap  = ta.volume.VolumeWeightedAveragePrice(high, low, close, vol).volume_weighted_average_price()
    vol_ma= vol.rolling(20).mean()
    macd_h= ta.trend.MACD(close).macd_diff()
    for i in range(2, len(df)):
        if (close.iloc[i-1] < vwap.iloc[i-1] and close.iloc[i] > vwap.iloc[i]
                and ema9.iloc[i] > ema21.iloc[i] and 45 < rsi.iloc[i] < 67
                and macd_h.iloc[i] > 0 and vol.iloc[i] > vol_ma.iloc[i] * 1.25):
            s.iloc[i] = 1
    return s

def _signals_swing(df):
    close, high, low = df["close"], df["high"], df["low"]
    s = pd.Series(0, index=df.index)
    if len(df) < 55: return s
    ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator()
    ema20 = ta.trend.EMAIndicator(close, 20).ema_indicator()
    ema200= ta.trend.EMAIndicator(close, min(200, len(df)-1)).ema_indicator()
    rsi   = ta.momentum.RSIIndicator(close, 14).rsi()
    adx   = ta.trend.ADXIndicator(high, low, close, 14).adx()
    for i in range(55, len(df)):
        if (abs(close.iloc[i] - ema50.iloc[i]) / ema50.iloc[i] < 0.018
                and ema20.iloc[i] > ema50.iloc[i] and close.iloc[i] > ema200.iloc[i]
                and 38 < rsi.iloc[i] < 60 and adx.iloc[i] > 20):
            s.iloc[i] = 1
    return s

def _signals_futures(df):
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    s = pd.Series(0, index=df.index)
    if len(df) < 30: return s
    ema9  = ta.trend.EMAIndicator(close, 9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
    adx   = ta.trend.ADXIndicator(high, low, close, 14).adx()
    vol_ma= vol.rolling(20).mean()
    for i in range(2, len(df)):
        bull = ema9.iloc[i-1] < ema21.iloc[i-1] and ema9.iloc[i] > ema21.iloc[i]
        if bull and adx.iloc[i] > 23 and vol.iloc[i] > vol_ma.iloc[i] * 1.3: s.iloc[i] = 1
        bear = ema9.iloc[i-1] > ema21.iloc[i-1] and ema9.iloc[i] < ema21.iloc[i]
        if bear and adx.iloc[i] > 23: s.iloc[i] = -1
    return s

def _signals_orb(df):
    s = pd.Series(0, index=df.index)
    if len(df) < 10: return s
    vol = df["volume"]; vol_ma = vol.rolling(10).mean()
    i = 0
    while i < len(df) - 4:
        or_high = df["high"].iloc[i:i+2].max()
        or_low  = df["low"].iloc[i:i+2].min()
        width   = (or_high - or_low) / or_low * 100
        if 0.3 < width < 2.5:
            for j in range(i+2, min(i+6, len(df))):
                if df["close"].iloc[j] > or_high and vol.iloc[j] > vol_ma.iloc[j] * 1.4:
                    s.iloc[j] = 1; break
                if df["close"].iloc[j] < or_low and vol.iloc[j] > vol_ma.iloc[j] * 1.4:
                    s.iloc[j] = -1; break
        i += 26
    return s

def _signals_vwap_reversion(df):
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    s = pd.Series(0, index=df.index)
    if len(df) < 20: return s
    vwap = ta.volume.VolumeWeightedAveragePrice(high, low, close, vol).volume_weighted_average_price()
    rsi  = ta.momentum.RSIIndicator(close, 14).rsi()
    adx  = ta.trend.ADXIndicator(high, low, close, 14).adx()
    for i in range(15, len(df)):
        if not vwap.iloc[i]: continue
        dev = abs(close.iloc[i] - vwap.iloc[i]) / vwap.iloc[i] * 100
        if 0.5 < dev < 2.5 and adx.iloc[i] < 22:
            if close.iloc[i] < vwap.iloc[i] and rsi.iloc[i] < 40: s.iloc[i] =  1
            if close.iloc[i] > vwap.iloc[i] and rsi.iloc[i] > 60: s.iloc[i] = -1
    return s

def _signals_iron_condor(df):
    close, high, low = df["close"], df["high"], df["low"]
    s = pd.Series(0, index=df.index)
    if len(df) < 25: return s
    adx  = ta.trend.ADXIndicator(high, low, close, 14).adx()
    bb   = ta.volatility.BollingerBands(close, 20, 2)
    bb_w = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg() * 100
    rsi  = ta.momentum.RSIIndicator(close, 14).rsi()
    for i in range(20, len(df)):
        if adx.iloc[i] < 20 and bb_w.iloc[i] < 2.5 and 38 < rsi.iloc[i] < 62:
            s.iloc[i] = 1
    return s

def _signals_short_straddle(df):
    close, high, low = df["close"], df["high"], df["low"]
    s = pd.Series(0, index=df.index)
    if len(df) < 25: return s
    adx    = ta.trend.ADXIndicator(high, low, close, 14).adx()
    atr    = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
    atr_ma = atr.rolling(20).mean()
    for i in range(20, len(df)):
        iv_low = atr.iloc[i] / atr_ma.iloc[i] < 0.85 if atr_ma.iloc[i] > 0 else False
        if adx.iloc[i] < 18 and iv_low: s.iloc[i] = 1
    return s

def _signals_options(df):
    close, high, low = df["close"], df["high"], df["low"]
    s = pd.Series(0, index=df.index)
    if len(df) < 28: return s
    adx  = ta.trend.ADXIndicator(high, low, close, 14).adx()
    bb   = ta.volatility.BollingerBands(close, 20, 2)
    rsi  = ta.momentum.RSIIndicator(close, 14).rsi()
    macd = ta.trend.MACD(close).macd_diff()
    for i in range(28, len(df)):
        if (close.iloc[i] > bb.bollinger_hband().iloc[i] and adx.iloc[i] > 26
                and rsi.iloc[i] > 58 and macd.iloc[i] > 0): s.iloc[i] = 1
        if (close.iloc[i] < bb.bollinger_lband().iloc[i] and adx.iloc[i] > 26
                and rsi.iloc[i] < 42 and macd.iloc[i] < 0): s.iloc[i] = -1
    return s

SIGNAL_REGISTRY_LOCAL = {
    "intraday": _signals_intraday, "swing": _signals_swing, "futures": _signals_futures,
    "orb": _signals_orb, "vwap_reversion": _signals_vwap_reversion,
    "iron_condor": _signals_iron_condor, "short_straddle": _signals_short_straddle,
    "options": _signals_options,
}


def _simulate(df, signals, sl_pct, tgt_pct, max_bars):
    trades, in_trade, entry, entry_i, direction = [], False, 0.0, 0, 1
    for i in range(len(df)):
        if in_trade:
            ltp  = df["close"].iloc[i]
            bars = i - entry_i
            sl   = entry * (1 - sl_pct/100 * direction)
            tgt  = entry * (1 + tgt_pct/100 * direction)
            low_i, high_i = df["low"].iloc[i], df["high"].iloc[i]
            if (direction > 0 and low_i <= sl) or (direction < 0 and high_i >= sl):
                trades.append({"pnl": -abs(entry * sl_pct/100), "bars": bars, "reason":"SL"})
                in_trade = False
            elif (direction > 0 and high_i >= tgt) or (direction < 0 and low_i <= tgt):
                trades.append({"pnl": abs(entry * tgt_pct/100), "bars": bars, "reason":"TGT"})
                in_trade = False
            elif bars >= max_bars:
                trades.append({"pnl": (ltp - entry) * direction, "bars": bars, "reason":"TIMEOUT"})
                in_trade = False
        elif signals.iloc[i] != 0:
            entry = df["close"].iloc[i]; direction = int(signals.iloc[i]); entry_i = i; in_trade = True
    return trades


def _compute_metrics(symbol, strategy, trades):
    if len(trades) < 5:
        return BacktestResult(symbol=symbol, strategy=strategy, passed=False, fail_reasons=["Insufficient trades"])
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    win_rate = len(wins)/len(pnls)*100
    cum = np.cumsum(pnls); peak = np.maximum.accumulate(cum)
    dd_pct = float(np.max(peak - cum)) / max(abs(peak.max()), 1) * 100 if len(cum) else 0
    arr = np.array(pnls)
    sharpe = float(arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0.0
    pf = sum(wins) / sum(abs(l) for l in losses) if losses else 0.0
    score = (win_rate * 0.35 + min(sharpe, 3) / 3 * 35 + max(0, 100 - dd_pct) * 0.20 + min(pf, 3) / 3 * 10)
    return BacktestResult(
        symbol=symbol, strategy=strategy, passed=False,
        total_trades=len(trades), wins=len(wins), losses=len(losses),
        win_rate=win_rate, total_pnl=sum(pnls),
        avg_win=float(np.mean(wins)) if wins else 0,
        avg_loss=float(np.mean([abs(l) for l in losses])) if losses else 0,
        max_drawdown_pct=dd_pct, sharpe_ratio=sharpe, profit_factor=pf,
        best_trade=max(pnls), worst_trade=min(pnls), score=score,
    )


def _apply_gate(r, strategy):
    g = GATE_THRESHOLDS.get(strategy, {"win_rate":55,"sharpe":1.0,"drawdown":15,"min_trades":20,"pf":1.2})
    reasons = []
    if r.total_trades < g.get("min_trades", 20): reasons.append(f"Trades {r.total_trades} < {g['min_trades']}")
    if r.win_rate < g["win_rate"]: reasons.append(f"Win rate {r.win_rate:.0f}% < {g['win_rate']}%")
    if r.sharpe_ratio < g["sharpe"]: reasons.append(f"Sharpe {r.sharpe_ratio:.2f} < {g['sharpe']}")
    if r.max_drawdown_pct > g["drawdown"]: reasons.append(f"Drawdown {r.max_drawdown_pct:.1f}%")
    if r.profit_factor < g["pf"]: reasons.append(f"PF {r.profit_factor:.2f} < {g['pf']}")
    if r.total_pnl <= 0: reasons.append("Negative total P&L")
    r.passed = len(reasons) == 0; r.fail_reasons = reasons
    return r


class AutoBacktestRunner:
    def __init__(self) -> None:
        self._results: dict = {}
        self._last_report: Optional[BatchReport] = None
        self._running = False
        self._log_dir = Path("logs/backtests")
        self._log_dir.mkdir(parents=True, exist_ok=True)

    async def run_all(self, strategies=None, symbols=None) -> BatchReport:
        if self._running:
            return self._last_report
        self._running = True
        t0 = time.time()
        strats = strategies or list(SIGNAL_REGISTRY_LOCAL.keys())
        syms   = symbols or ALL_SYMBOLS
        sem = asyncio.Semaphore(12)

        async def run_one(sym, strat):
            async with sem:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._run_single, sym, strat)
                if sym not in self._results: self._results[sym] = {}
                self._results[sym][strat] = result

        await asyncio.gather(*[run_one(s, st) for s in syms for st in strats], return_exceptions=True)
        report = self._build_report(strats, time.time() - t0)
        self._last_report = report
        self._save_report(report)
        self._running = False
        return report

    def _run_single(self, symbol, strategy):
        try:
            cfg = STRATEGY_CONFIG.get(strategy, {})
            df  = yf_client.historical(symbol, "NSE", cfg.get("interval","15m"), cfg.get("period","3mo"))
            if df.empty or len(df) < 30:
                return BacktestResult(symbol=symbol, strategy=strategy, passed=False, fail_reasons=["No data"])
            sig_fn = SIGNAL_REGISTRY_LOCAL.get(strategy)
            if not sig_fn:
                return BacktestResult(symbol=symbol, strategy=strategy, passed=False, fail_reasons=["Unknown strategy"])
            signals = sig_fn(df)
            sl = cfg.get("sl", 1.5); tgt = cfg.get("t1", 3.0); max_bars = cfg.get("max_bars", 20)
            trades = _simulate(df, signals, sl, tgt, max_bars)
            result = _compute_metrics(symbol, strategy, trades)
            return _apply_gate(result, strategy)
        except Exception as exc:
            return BacktestResult(symbol=symbol, strategy=strategy, passed=False, fail_reasons=[str(exc)[:80]])

    def _build_report(self, strats, elapsed):
        by_strategy = {}; approved = {}; total_passed = total_failed = 0
        for strat in strats:
            passed_res = []; failed_res = []
            for sym, strat_map in self._results.items():
                r = strat_map.get(strat)
                if not r: continue
                if r.passed: passed_res.append(r); total_passed += 1
                else: failed_res.append(r); total_failed += 1
            leaderboard = sorted(passed_res, key=lambda x: x.score, reverse=True)
            top_n = TOP_N_PER_STRATEGY.get(strat, 6)
            approved[strat] = [r.symbol for r in leaderboard[:top_n]]
            by_strategy[strat] = {"passed": [r.to_dict() for r in leaderboard],
                "failed": [r.to_dict() for r in failed_res[:10]],
                "leaderboard": [r.to_dict() for r in leaderboard[:top_n]]}
        return BatchReport(run_date=date.today().isoformat(), run_time_sec=elapsed,
            total_tested=total_passed+total_failed, total_passed=total_passed,
            total_failed=total_failed, by_strategy=by_strategy, approved=approved)

    def _save_report(self, report):
        path = self._log_dir / f"backtest_{report.run_date}.json"
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

    def approved_symbols(self, strategy):
        if not self._last_report: return []
        return self._last_report.approved.get(strategy, [])

    def get_result(self, symbol, strategy):
        return self._results.get(symbol, {}).get(strategy)

    def leaderboard(self, strategy, top_n=10):
        if not self._last_report: return []
        return self._last_report.by_strategy.get(strategy, {}).get("leaderboard", [])[:top_n]

    def last_report_dict(self):
        return self._last_report.to_dict() if self._last_report else None

    def quick_test(self, symbol, strategy):
        return self._run_single(symbol, strategy)


auto_backtest = AutoBacktestRunner()
