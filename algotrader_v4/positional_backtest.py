"""
positional_backtest.py — Phase-1 validation CLI for the positional
strategies (spec section G, "Phased Rollout" gate 1).

Data sources:
  --kite            continuous daily futures bars via Kite historical API
                    for every universe root (needs LIVE mode + valid
                    access token; ~2000 days per request, auto-chunked)
  --csv PATH        offline OHLC csv: date,open,high,low,close[,volume]
                    (also accepts FRED fredgraph.csv close-only exports)

Examples:
  python positional_backtest.py --kite
  python positional_backtest.py --kite --symbol CRUDEOILM --years 8
  python positional_backtest.py --csv data/crude.csv --symbol CRUDEOILM
  python positional_backtest.py --csv fredgraph.csv --strategy tsmom

Go/no-go per spec: DSR > 0, OOS Sharpe > 50% of in-sample, ≥30 trades,
walk-forward majority positive. Paper trading (Phase 2) only on GO.
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import sys
from datetime import datetime, timedelta

from strategies import (DailyBar, DonchianBreakoutStrategy,
                        MACrossoverStrategy, TSMOMStrategy)
from strategies.phase1_backtest import phase1_report

FACTORIES = {
    "donchian":     DonchianBreakoutStrategy,
    "ma_crossover": MACrossoverStrategy,
    "tsmom":        lambda: TSMOMStrategy(long_only=True),
}


def load_csv(path: str) -> list[DailyBar]:
    bars: list[DailyBar] = []
    with open(path, newline="") as f:
        reader = csv_mod.DictReader(f)
        cols = [c.lower() for c in reader.fieldnames or []]
        date_col = reader.fieldnames[0]
        is_ohlc = all(c in cols for c in ("open", "high", "low", "close"))
        value_col = None if is_ohlc else reader.fieldnames[1]
        for row in reader:
            try:
                d = datetime.strptime(row[date_col][:10], "%Y-%m-%d").date()
                if is_ohlc:
                    o, h = float(row.get("open") or row["Open"]), float(row.get("high") or row["High"])
                    l, c = float(row.get("low") or row["Low"]), float(row.get("close") or row["Close"])
                else:
                    v = row[value_col]
                    if not v or v == ".":
                        continue
                    o = h = l = c = float(v)
                if c > 0:
                    bars.append(DailyBar(date=d, open=o, high=h, low=l, close=c))
            except (ValueError, KeyError):
                continue
    return bars


def load_kite(root: str, years: int) -> list[DailyBar] | None:
    from config import settings
    from ist_clock import now_ist
    from kite_client import kite_client
    from positional_runner import pick_near_month, sane_bars
    from strategies import get_contract

    if settings.trading_mode != "LIVE":
        print(f"  {root}: TRADING_MODE must be LIVE with a valid Kite token "
              f"for historical data — skipping")
        return None
    contract = get_contract(root)
    instruments = kite_client.get_instruments(contract.exchange)
    inst = pick_near_month(instruments, root, now_ist().date())
    if not inst:
        print(f"  {root}: no tradable contract found — skipping")
        return None
    records = kite_client.historical_data(
        instrument_token=inst["instrument_token"],
        from_date=datetime.now() - timedelta(days=int(years * 365.25)),
        to_date=datetime.now(),
        interval="day", continuous=True)
    bars = sane_bars(records)
    if not bars:
        print(f"  {root}: bar sanity check failed — skipping")
    return bars


def print_report(rep: dict, symbol: str) -> None:
    f, g = rep["full"], rep["gates"]
    print(f"\n─── {rep['strategy']} on {symbol} "
          f"{'─' * max(1, 40 - len(rep['strategy']) - len(symbol))}")
    print(f"  Full sample : CAGR {f['cagr']:+7.1%} | Sharpe {f['sharpe']:5.2f} | "
          f"MaxDD {f['max_dd']:5.1%} | {f['trades']} trades | "
          f"win {f['win_rate']:.0%} | expectancy {f['expectancy']:+.2%}/trade")
    print(f"  In-sample   : Sharpe {rep['in_sample']['sharpe']:5.2f}   "
          f"Hold-out OOS: Sharpe {rep['oos_sharpe']:5.2f}")
    print(f"  DSR         : {rep['dsr']['dsr']:.3f}  "
          f"(needs > 0.5; > 0.95 = strong)")
    mc = rep["monte_carlo"]
    print(f"  Monte Carlo : DD p50 {mc['dd_p50']:.1%} | p95 {mc['dd_p95']:.1%} | "
          f"worst {mc['dd_worst']:.1%} | terminal p5 {mc['ret_p5']:+.1%}")
    wf_pos = sum(1 for b in rep["walk_forward"] if b["sharpe"] > 0)
    print(f"  Walk-forward: {wf_pos}/{len(rep['walk_forward'])} blocks "
          f"Sharpe-positive")
    print(f"  Kill limit  : live max DD auto-halt at {rep['kill_dd_threshold']:.1%} "
          f"(backtest DD × 1.5)")
    checks = " ".join(f"{'✅' if v else '❌'}{k}" for k, v in g.items())
    print(f"  Gates       : {checks}")
    print(f"  VERDICT     : {'🟢 GO — proceed to paper trading' if rep['go'] else '🔴 NO-GO — do not allocate'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-1 positional strategy validation")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--kite", action="store_true",
                     help="fetch continuous daily futures bars via Kite")
    src.add_argument("--csv", help="offline OHLC/FRED csv file")
    ap.add_argument("--symbol", action="append",
                    help="universe root(s); default = positional_universe "
                         "(--kite) or csv basename (--csv)")
    ap.add_argument("--strategy", default="all",
                    choices=["all", *FACTORIES.keys()])
    ap.add_argument("--years", type=int, default=10,
                    help="lookback years for --kite (default 10)")
    ap.add_argument("--trials", type=int, default=9,
                    help="strategy variants tried, for the Deflated Sharpe "
                         "(count every configuration you tested; default 9)")
    args = ap.parse_args()

    if args.csv:
        bars = load_csv(args.csv)
        if len(bars) < 400:
            print(f"Not enough data in {args.csv} ({len(bars)} bars, need ≥400)")
            return 1
        name = args.symbol[0] if args.symbol else args.csv.rsplit("/", 1)[-1]
        datasets = {name: bars}
    else:
        from config import settings
        roots = args.symbol or [s.strip().upper() for s in
                                settings.positional_universe.split(",") if s.strip()]
        datasets = {}
        print(f"Fetching {args.years}y continuous daily bars via Kite…")
        for root in roots:
            bars = load_kite(root, args.years)
            if bars and len(bars) >= 400:
                datasets[root] = bars
        if not datasets:
            print("No usable data — check TRADING_MODE=LIVE and the access token.")
            return 1

    strategies = FACTORIES if args.strategy == "all" else \
        {args.strategy: FACTORIES[args.strategy]}

    any_go = False
    for symbol, bars in datasets.items():
        print(f"\n════ {symbol}: {len(bars)} bars "
              f"({bars[0].date} → {bars[-1].date}) ════")
        for sname, factory in strategies.items():
            rep = phase1_report(factory, sname, bars, n_trials=args.trials)
            print_report(rep, symbol)
            any_go = any_go or rep["go"]
    print("\nSpec reminder: only GO strategies proceed to Phase 2 (paper, "
          "8–12 weeks, ≥20–30 signals); record every additional variant you "
          "test and raise --trials accordingly.")
    return 0 if any_go else 2


if __name__ == "__main__":
    sys.exit(main())
