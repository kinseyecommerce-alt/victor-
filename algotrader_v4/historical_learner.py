"""
historical_learner.py
One-time pre-learning script. Run ONCE before going live.

Backtests all Nifty 100 × 4 strategies using historical data,
then pre-populates the adaptive engine so the bot starts with
2 years of learned knowledge instead of cold defaults.

Usage:
    cd algotrader_v4
    python3 historical_learner.py

Or with strategy filter:
    python3 historical_learner.py --strategies intraday,scalping
    python3 historical_learner.py --resume       # skip already-done symbols
    python3 historical_learner.py --symbols RELIANCE,TCS,HDFCBANK
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

# Configure clean console output for this script
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> {message}", level="INFO")

PROGRESS_FILE = Path("logs/learning_progress.json")
APPROVED_FILE = Path("logs/approved_symbols.json")
Path("logs").mkdir(exist_ok=True)

ALL_STRATEGIES = ["intraday", "scalping", "fno", "swing"]


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _load_approved() -> dict:
    if APPROVED_FILE.exists():
        return json.loads(APPROVED_FILE.read_text())
    return {s: [] for s in ALL_STRATEGIES}


def _save_approved(approved: dict) -> None:
    APPROVED_FILE.write_text(json.dumps(approved, indent=2, default=str))


async def _run_one(symbol: str, strategy: str, sem: asyncio.Semaphore,
                   approved: dict, progress: dict) -> tuple[str, str, bool]:
    key = f"{strategy}::{symbol}"
    async with sem:
        try:
            from backtest_engine import backtest_engine
            from adaptive_engine import adaptive_engine, AdaptiveParams

            result = await asyncio.to_thread(
                backtest_engine.run, symbol, "NSE", strategy
            )

            # Pre-populate adaptive engine with learned params
            params_key = f"{strategy}::{symbol}"
            existing = adaptive_engine._params.get(params_key)
            if existing is None or existing.win_rate_20 == 0:
                # Seed with backtest results
                from adaptive_engine import STRATEGY_CONFIG, GATE_THRESHOLDS
                cfg  = STRATEGY_CONFIG.get(strategy, {})
                gate = GATE_THRESHOLDS.get(strategy, {})
                new_params = AdaptiveParams(
                    strategy=strategy, symbol=symbol,
                    sl_pct=result.optimal_sl if hasattr(result, "optimal_sl") else cfg.get("sl", 1.5),
                    target_pct=result.optimal_target if hasattr(result, "optimal_target") else cfg.get("t1", 3.0),
                    trail_pct=(result.optimal_sl if hasattr(result, "optimal_sl") else cfg.get("sl", 1.5)) * 0.33,
                    win_rate_20=result.win_rate / 100.0,
                    sharpe_20=result.sharpe_ratio,
                    avg_win_pct=result.avg_win_pct if hasattr(result, "avg_win_pct") else cfg.get("t1", 3.0) * 0.6,
                    avg_loss_pct=result.avg_loss_pct if hasattr(result, "avg_loss_pct") else cfg.get("sl", 1.5) * 0.7,
                    min_rsi=45.0 if result.win_rate >= 55 else 48.0,
                    max_rsi=67.0 if result.win_rate >= 55 else 63.0,
                    min_adx=float(gate.get("win_rate", 55)) / 5,
                    status="ACTIVE" if result.passed else "CAUTIOUS",
                )
                adaptive_engine._params[params_key] = new_params

            if result.passed:
                if symbol not in approved[strategy]:
                    approved[strategy].append(symbol)
                status = "✅ PASS"
            else:
                status = f"⚠  FAIL ({', '.join(result.fail_reasons[:1])})"

            progress[key] = {
                "done": True, "passed": result.passed,
                "win_rate": round(result.win_rate, 1),
                "sharpe": round(result.sharpe_ratio, 2),
                "trades": result.total_trades,
                "ts": datetime.now().isoformat(),
            }
            return symbol, strategy, result.passed, status

        except Exception as exc:
            logger.warning("  {} {} error: {}", symbol, strategy, exc)
            progress[key] = {"done": True, "passed": False, "error": str(exc),
                             "ts": datetime.now().isoformat()}
            return symbol, strategy, False, f"❌ ERROR"


async def learn(symbols: list[str], strategies: list[str],
                resume: bool, concurrency: int) -> None:
    from adaptive_engine import adaptive_engine
    from agents.base_agent import send_telegram

    progress = _load_progress() if resume else {}
    approved = _load_approved() if resume else {s: [] for s in ALL_STRATEGIES}

    total   = len(symbols) * len(strategies)
    done    = 0
    sem     = asyncio.Semaphore(concurrency)
    t_start = time.time()

    print(f"\n{'═'*60}")
    print(f"  AlgoTrader Pro — Historical Learner")
    print(f"  {len(symbols)} symbols × {len(strategies)} strategies = {total} backtests")
    print(f"  Concurrency: {concurrency} | Resume: {resume}")
    print(f"{'═'*60}\n")

    # Notify start
    asyncio.create_task(send_telegram(
        f"🧠 <b>Historical Learning Started</b>\n"
        f"{len(symbols)} Nifty-100 symbols × {len(strategies)} strategies\n"
        f"Expected time: ~{total//concurrency//6} min"
    ))

    tasks = []
    for symbol in symbols:
        for strategy in strategies:
            key = f"{strategy}::{symbol}"
            if resume and progress.get(key, {}).get("done"):
                done += 1
                # Re-add to approved if was passing
                if progress[key].get("passed") and symbol not in approved.get(strategy, []):
                    approved.setdefault(strategy, []).append(symbol)
                continue
            tasks.append(_run_one(symbol, strategy, sem, approved, progress))

    skipped = total - len(tasks)
    if skipped:
        print(f"  Skipping {skipped} already-completed backtests (--resume)\n")

    milestone_pct = 0
    for coro in asyncio.as_completed(tasks):
        sym, strat, passed, status = await coro
        done += 1
        elapsed = time.time() - t_start
        rate    = done / elapsed if elapsed > 0 else 1
        eta_s   = int((total - done) / rate) if rate > 0 else 0
        eta     = f"{eta_s//60}m{eta_s%60:02d}s"

        print(f"  [{done:3d}/{total}] {status:30s} {sym:15s} {strat:10s}  ETA {eta}")

        # Save progress every 10 completions
        if done % 10 == 0:
            _save_progress(progress)
            _save_approved(approved)
            adaptive_engine._save_state()

        # Telegram milestone every 25%
        pct = done * 100 // total
        if pct >= milestone_pct + 25:
            milestone_pct = (pct // 25) * 25
            pass_counts = {s: len(v) for s, v in approved.items()}
            asyncio.create_task(send_telegram(
                f"🧠 Learning {milestone_pct}% done ({done}/{total})\n"
                + "\n".join(f"  {s}: {n} approved" for s, n in pass_counts.items())
            ))

    # Final save
    _save_progress(progress)
    _save_approved(approved)
    adaptive_engine._save_state()

    elapsed = int(time.time() - t_start)
    pass_counts = {s: len(v) for s, v in approved.items()}
    total_approved = sum(pass_counts.values())

    print(f"\n{'═'*60}")
    print(f"  ✅ Learning complete in {elapsed//60}m{elapsed%60:02d}s")
    print(f"  Approved symbols per strategy:")
    for strat, syms in approved.items():
        print(f"    {strat:12s}: {len(syms):3d} / {len(symbols)}")
    print(f"\n  Adaptive params saved → logs/adaptive/adaptive_params.json")
    print(f"  Approved list saved  → logs/approved_symbols.json")
    print(f"\n  Now set in .env:")
    print(f"    SKIP_STARTUP_BACKTEST=true")
    print(f"    USE_NIFTY100_WATCHLIST=true")
    print(f"{'═'*60}\n")

    asyncio.create_task(send_telegram(
        f"✅ <b>Historical Learning Complete</b>\n"
        f"Time: {elapsed//60}m{elapsed%60:02d}s\n"
        + "\n".join(f"  {s}: {n} symbols approved" for s, n in pass_counts.items()) +
        f"\n\nSet SKIP_STARTUP_BACKTEST=true to use pre-learned params."
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="AlgoTrader Historical Learner")
    parser.add_argument("--strategies", default=",".join(ALL_STRATEGIES),
                        help="Comma-separated strategies (default: all)")
    parser.add_argument("--symbols", default="",
                        help="Comma-separated symbols (default: full Nifty 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed backtests")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Parallel backtests (default: 5)")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        from nifty100 import NIFTY_100
        symbols = NIFTY_100

    asyncio.run(learn(symbols, strategies, args.resume, args.concurrency))


if __name__ == "__main__":
    main()
