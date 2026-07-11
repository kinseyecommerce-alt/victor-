"""
strategy_backtest.py — Walk-forward, cost-adjusted backtest of all 80 strategies.

Validates alpha: replays historical bars through the exact live indicator stack
(IndicatorCalc) and each strategy, simulates entries/exits with realistic
transaction costs and slippage (cost_model), and reports per-strategy
expectancy / Sharpe / profit-factor / max-drawdown across walk-forward folds.

The point is to turn "20 strategies per agent" into "the N per agent that
actually have a positive, out-of-sample, cost-adjusted edge" — everything else
should be switched off. `approved_strategies()` returns that filtered set and
persists it to logs/approved_strategies.json for the live agents to consult.

Data source: the broker (Kite historical) when connected; otherwise a
deterministic synthetic OHLCV series so the harness is runnable/testable
offline (clearly labelled — synthetic results carry no alpha meaning).

Run:
  python strategy_backtest.py                 # all agents, default universe
  python strategy_backtest.py --symbol CRUDEOIL --days 30
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

import mcx_universe
import cost_model
from tick_engine import Tick, Candle, IndicatorCalc
from agents.mcx_strategies import SCtx, update_prev
from agents.mcx_agents import MCX_AGENTS

_APPROVED_PATH = Path("logs/approved_strategies.json")

WARMUP        = 205     # bars before strategies may fire (EMA200 warmup)
MAX_HOLD      = 40      # bars to hold before force-exit
MIN_TRADES    = 8       # minimum trades for a strategy to be judged
FOLDS         = 4       # walk-forward folds
LOOKBACK_BARS = 60      # window fed to IndicatorCalc / SCtx.candles each bar


# ── Data loading ────────────────────────────────────────────────────────────────
def load_bars(symbol: str, interval: str = "5m", days: int = 30) -> tuple[pd.DataFrame, str]:
    """Return (OHLCV DataFrame, source). Broker when connected, else synthetic."""
    try:
        from kite_client import kite_client
        if kite_client.is_connected():
            from tick_engine import tick_engine
            df = tick_engine.get_historical(symbol, "MCX", interval, f"{days}d")
            if df is not None and len(df) >= WARMUP + 50:
                return df, "BROKER"
    except Exception as exc:
        logger.warning("[backtest] broker history failed for {}: {}", symbol, exc)
    return _synthetic_bars(symbol, days, interval), "SYNTHETIC"


def _synthetic_bars(symbol: str, days: int, interval: str) -> pd.DataFrame:
    """Deterministic GBM-ish OHLCV — reproducible per symbol (offline/testing only)."""
    c = mcx_universe.contract(symbol)
    base = c.base_price if c else 1000.0
    per_day = {"1m": 375, "5m": 75, "15m": 25}.get(interval, 75)
    n = max(WARMUP + 120, days * per_day)
    rng = random.Random(hash(symbol) & 0xFFFFFFFF)
    price = base
    rows = []
    ts = datetime(2025, 1, 1, 9, 0, 0)
    drift = rng.choice([1, -1]) * 0.00003
    for k in range(n):
        # regime shifts every ~200 bars to create trends + ranges
        if k % 200 == 0:
            drift = rng.choice([1, -1, 0]) * rng.uniform(0.00002, 0.00009)
        shock = rng.gauss(drift, 0.0016)
        o = price
        price = max(price * math.exp(shock), base * 0.3)
        hi = max(o, price) * (1 + abs(rng.gauss(0, 0.0008)))
        lo = min(o, price) * (1 - abs(rng.gauss(0, 0.0008)))
        vol = int(abs(rng.gauss(1000, 400))) + 100
        rows.append({"timestamp": ts, "open": round(o, 2), "high": round(hi, 2),
                     "low": round(lo, 2), "close": round(price, 2), "volume": vol})
        ts += timedelta(minutes=5)
    return pd.DataFrame(rows)


# ── Trade simulation ────────────────────────────────────────────────────────────
@dataclass
class _Open:
    side:  str
    entry: float
    sl:    float
    tgt:   float
    bar:   int


@dataclass
class StratResult:
    agent:    str
    strategy: str
    symbol:   str
    source:   str
    trades:   int = 0
    wins:     int = 0
    net_total:   float = 0.0
    gross_total: float = 0.0
    cost_total:  float = 0.0
    returns:  list = field(default_factory=list)   # per-trade net return (₹/margin)
    fold_pnl: list = field(default_factory=list)   # net pnl per fold

    def metrics(self, folds: int) -> dict:
        n = self.trades
        win_rate = round(100 * self.wins / n, 1) if n else 0.0
        net_exp  = round(self.net_total / n, 2) if n else 0.0
        mean = sum(self.returns) / n if n else 0.0
        if n > 1:
            var = sum((r - mean) ** 2 for r in self.returns) / (n - 1)
            sd  = math.sqrt(var)
            sharpe = round(mean / sd * math.sqrt(n), 2) if sd > 0 else 0.0
        else:
            sharpe = 0.0
        gains  = sum(r for r in self.returns if r > 0)
        losses = -sum(r for r in self.returns if r < 0)
        pf = round(gains / losses, 2) if losses > 0 else (float("inf") if gains > 0 else 0.0)
        pos_folds = sum(1 for p in self.fold_pnl if p > 0)
        consistency = round(pos_folds / folds, 2) if folds else 0.0
        approved = (self.source == "BROKER" and n >= MIN_TRADES
                    and net_exp > 0 and consistency >= 0.5)
        return {
            "agent": self.agent, "strategy": self.strategy, "symbol": self.symbol,
            "source": self.source, "trades": n, "win_rate": win_rate,
            "net_expectancy": net_exp, "net_total": round(self.net_total, 2),
            "gross_total": round(self.gross_total, 2), "cost_total": round(self.cost_total, 2),
            "sharpe": sharpe, "profit_factor": pf, "fold_consistency": consistency,
            "max_drawdown": self._max_dd(), "approved": approved,
        }

    def _max_dd(self) -> float:
        # not tracked per-trade cumulative here; approximate from returns order
        cum, peak, dd = 0.0, 0.0, 0.0
        for r in self.returns:
            cum += r
            peak = max(peak, cum)
            dd = min(dd, cum - peak)
        return round(dd, 3)


def backtest_agent_symbol(agent_name: str, symbol: str,
                          df: pd.DataFrame, source: str, folds: int = FOLDS) -> list[dict]:
    """Backtest every strategy of one agent on one symbol's bars."""
    agent = MCX_AGENTS[agent_name]
    strategies = agent.strategies
    sl_atr, tgt_atr, min_score = agent.SL_ATR, agent.TGT_ATR, agent.MIN_SCORE
    lot_size = mcx_universe.lot_size(symbol)
    c = mcx_universe.contract(symbol)
    margin = c.margin_per_lot if c else 100000.0

    results = {name: StratResult(agent_name, name, symbol, source) for name, _ in strategies}
    open_trades: dict[str, _Open] = {}
    prev: dict = {}
    n = len(df)
    fold_size = max(1, n // folds)

    opens  = df["open"].tolist();  highs = df["high"].tolist()
    lows   = df["low"].tolist();   closes = df["close"].tolist()
    vols   = df["volume"].tolist()

    for i in range(WARMUP, n - 1):
        window = df.iloc[max(0, i - 210):i + 1]
        close_i = closes[i]
        prev_close = closes[i - 1] if i > 0 else close_i
        tick = Tick(symbol=symbol, ltp=close_i, bid=close_i, ask=close_i,
                    volume=int(vols[i]), change=close_i - prev_close,
                    change_pct=(close_i / prev_close - 1) * 100 if prev_close else 0.0,
                    high=highs[i], low=lows[i], open=opens[i], timestamp=datetime.now())
        ind = IndicatorCalc.compute(symbol, tick, window)
        cand_win = df.iloc[max(0, i - LOOKBACK_BARS):i + 1]
        candles = [Candle(r.open, r.high, r.low, r.close, int(r.volume), datetime.now())
                   for r in cand_win.itertuples()]
        ctx = SCtx(sym=symbol, ltp=close_i, ind=ind, prev=prev,
                   candles=candles, t=datetime.now().time())

        fold = min(folds - 1, i // fold_size)

        # ── manage open trades (exit checks on this bar) ─────────────────
        for name in list(open_trades):
            o = open_trades[name]
            exit_px = None
            if o.side == "BUY":
                if lows[i] <= o.sl:   exit_px = o.sl
                elif highs[i] >= o.tgt: exit_px = o.tgt
            else:
                if highs[i] >= o.sl:  exit_px = o.sl
                elif lows[i] <= o.tgt: exit_px = o.tgt
            if exit_px is None and (i - o.bar) >= MAX_HOLD:
                exit_px = close_i
            if exit_px is not None:
                _close_trade(results[name], o, exit_px, symbol, lot_size, margin, fold)
                del open_trades[name]

        # ── entry checks for idle strategies ─────────────────────────────
        atr = ind.atr_14 or close_i * 0.005
        for name, fn in strategies:
            if name in open_trades:
                continue
            try:
                res = fn(ctx)
            except Exception:
                res = None
            if not res:
                continue
            action, score = res
            if action not in ("BUY", "SELL") or score < min_score:
                continue
            # enter next bar at open + slippage
            slip = cost_model.slippage_cost(symbol, 1) / lot_size
            entry = opens[i + 1] + (slip if action == "BUY" else -slip)
            if action == "BUY":
                sl, tgt = entry - atr * sl_atr, entry + atr * tgt_atr
            else:
                sl, tgt = entry + atr * sl_atr, entry - atr * tgt_atr
            open_trades[name] = _Open(action, entry, sl, tgt, i)

        update_prev(prev, ctx)

    return [results[name].metrics(folds) for name, _ in strategies]


def _close_trade(r: StratResult, o: _Open, exit_px: float, symbol: str,
                 lot_size: int, margin: float, fold: int) -> None:
    direction = 1 if o.side == "BUY" else -1
    gross = (exit_px - o.entry) * direction * lot_size
    cost  = cost_model.round_trip_cost(symbol, 1, o.entry)
    net   = gross - cost
    r.trades += 1
    r.wins   += 1 if net > 0 else 0
    r.gross_total += gross
    r.cost_total  += cost
    r.net_total   += net
    r.returns.append(net / margin if margin else net)
    while len(r.fold_pnl) <= fold:
        r.fold_pnl.append(0.0)
    r.fold_pnl[fold] += net


# ── Orchestration ───────────────────────────────────────────────────────────────
def run(symbols: list[str] | None = None, agents: list[str] | None = None,
        interval: str = "5m", days: int = 30, folds: int = FOLDS) -> dict:
    agents = agents or list(MCX_AGENTS.keys())
    results: list[dict] = []
    source_seen = set()
    for agent_name in agents:
        syms = symbols or [i["symbol"] for i in mcx_universe.get_strategy_watchlist(agent_name)]
        for sym in syms:
            df, source = load_bars(sym, interval, days)
            source_seen.add(source)
            results.extend(backtest_agent_symbol(agent_name, sym, df, source, folds))
    return {"source": "MIXED" if len(source_seen) > 1 else source_seen.pop() if source_seen else "NONE",
            "results": results}


def rank(results: list[dict], by: str = "sharpe") -> list[dict]:
    return sorted(results, key=lambda r: (r.get(by, 0), r.get("net_expectancy", 0)), reverse=True)


def approved_strategies(report: dict) -> dict[str, list[str]]:
    """Per-agent list of strategy names with a positive OOS cost-adjusted edge."""
    out: dict[str, list[str]] = {a: [] for a in MCX_AGENTS}
    for r in report["results"]:
        if r["approved"] and r["strategy"] not in out[r["agent"]]:
            out[r["agent"]].append(r["strategy"])
    return out


def save_approved(approved: dict[str, list[str]]) -> None:
    _APPROVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    _APPROVED_PATH.write_text(json.dumps(approved, indent=2))
    logger.info("[backtest] approved strategies written to {}", _APPROVED_PATH)


def load_approved() -> dict[str, list[str]] | None:
    if _APPROVED_PATH.exists():
        try:
            return json.loads(_APPROVED_PATH.read_text())
        except Exception:
            return None
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol")
    ap.add_argument("--agent")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--interval", default="5m")
    args = ap.parse_args()

    rep = run(symbols=[args.symbol] if args.symbol else None,
              agents=[args.agent] if args.agent else None,
              interval=args.interval, days=args.days)
    ranked = rank(rep["results"])
    print(f"\n{'='*92}\n  STRATEGY BACKTEST — data source: {rep['source']}"
          f"{'  (synthetic — no alpha meaning)' if rep['source']=='SYNTHETIC' else ''}\n{'='*92}")
    print(f"  {'agent':9} {'strategy':22} {'sym':10} {'trades':>6} {'win%':>5} "
          f"{'net/trade':>10} {'sharpe':>7} {'PF':>5} {'cons':>5} {'ok':>3}")
    for r in ranked:
        print(f"  {r['agent']:9} {r['strategy']:22} {r['symbol']:10} {r['trades']:6d} "
              f"{r['win_rate']:5.0f} {r['net_expectancy']:10.0f} {r['sharpe']:7.2f} "
              f"{str(r['profit_factor']):>5} {r['fold_consistency']:5.2f} "
              f"{'Y' if r['approved'] else '-':>3}")
    approved = approved_strategies(rep)
    total = sum(len(v) for v in approved.values())
    print(f"\n  Approved (positive OOS cost-adjusted edge): {total} strategies")
    for a, names in approved.items():
        print(f"    {a}: {len(names)} — {', '.join(names) or '(none)'}")
    if rep["source"] == "BROKER":
        save_approved(approved)
    else:
        print("\n  NOTE: synthetic data — approvals are NOT persisted. Connect the broker"
              "\n        for real historical bars, then re-run to generate approved_strategies.json.")
