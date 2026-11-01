"""
momentum_rotation.py — monthly cross-sectional momentum (Jegadeesh & Titman 1993; the
"12-1" variant), the best-documented systematic equity strategy there is.

Rule: once a month, rank the universe (Nifty 100) by past-year return skipping the most
recent month, hold the top N equal-weighted, sell what dropped out. Only buy while the
NIFTY 50 is above its 200-day EMA (market_filter) — the combination cuts the deep
drawdowns momentum is known for. Positions are CNC with a wide GTT stop.

Also here: a monthly-rebalanced backtest of the same rule against buy-and-hold NIFTY 50
on the same dates. The paper simulator uses synthetic histories, so a backtest there only
proves the machinery works; judge the strategy on real data (Kite / yfinance).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from loguru import logger

from config import settings

LOOKBACK = 252          # ~12 months of sessions
SKIP = 21               # skip the latest month (short-term reversal)
HALF = 126              # 6-month leg


def universe() -> list[str]:
    from symbol_scanner import NIFTY_50, NSE_TOP_100
    return list(NIFTY_50 if settings.momentum_universe == "NIFTY50" else NSE_TOP_100)


def _score(closes: list[float]) -> Optional[dict]:
    if len(closes) < LOOKBACK + 1:
        return None
    end = closes[-SKIP - 1]                       # price one month ago
    r12 = end / closes[-LOOKBACK - 1] - 1
    r6 = end / closes[-HALF - 1] - 1
    ema200 = sum(closes[-200:]) / 200
    return {"ret_12_1": round(r12 * 100, 2), "ret_6_1": round(r6 * 100, 2),
            "score": round((0.5 * r12 + 0.5 * r6) * 100, 2), "above_200dma": closes[-1] > ema200,
            "last": round(closes[-1], 2)}


def rankings(symbols: Optional[list[str]] = None) -> list[dict]:
    """Every symbol with enough history, best momentum first."""
    from daily_data import daily_closes
    out = []
    for sym in symbols or universe():
        try:
            closes = [c for _, c in daily_closes(sym)]
        except Exception as exc:
            logger.debug("[momentum] {} skipped: {}", sym, exc)
            continue
        sc = _score(closes)
        if sc:
            out.append({"symbol": sym, **sc})
    out.sort(key=lambda r: -r["score"])
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def selection(top_n: Optional[int] = None) -> list[dict]:
    """Top N that are in an uptrend themselves (positive momentum, above their 200-DMA)."""
    n = top_n or settings.momentum_top_n
    return [r for r in rankings() if r["score"] > 0 and r["above_200dma"]][:n]


def rebalance_plan(held: dict[str, float], top_n: Optional[int] = None) -> dict:
    """held: symbol → entry price of current momentum positions."""
    from market_filter import state as trend_state
    sel = selection(top_n)
    want = [r["symbol"] for r in sel]
    trend = trend_state()
    sells = [s for s in held if s not in want]
    buys = [] if trend["state"] == "RISK_OFF" and settings.trend_filter_block_swing_longs else \
        [r for r in sel if r["symbol"] not in held]
    n = max(top_n or settings.momentum_top_n, 1)
    return {"trend": trend["state"], "selection": sel, "keep": [s for s in held if s in want],
            "sell": sells, "buy": buys, "capital_per_name": round(settings.momentum_capital / n, 2)}


# ── Backtest ─────────────────────────────────────────────────────────────────

def _month_ends(dates: list[date]) -> list[int]:
    idx = []
    for i in range(len(dates) - 1):
        if dates[i].month != dates[i + 1].month:
            idx.append(i)
    return idx


def backtest(months: int = 12, top_n: Optional[int] = None, symbols: Optional[list[str]] = None,
             cost_pct: float = 0.25) -> dict:
    """Monthly rotation on real (or synthetic) daily closes vs buy-and-hold NIFTY 50.
    cost_pct = round-trip cost per rotation leg (delivery STT + slippage ≈ 0.25%)."""
    from daily_data import daily_closes
    n = top_n or settings.momentum_top_n
    series: dict[str, dict[date, float]] = {}
    for sym in symbols or universe():
        try:
            rows = daily_closes(sym)
        except Exception:
            continue
        if len(rows) >= LOOKBACK + 22:
            series[sym] = dict(rows)
    nifty = dict(daily_closes("NIFTY 50", fallback_price=24000.0))
    if not series or not nifty:
        return {"error": "not enough daily history"}
    all_dates = sorted(set.intersection(*(set(s) for s in series.values())) & set(nifty))
    rebal = [i for i in _month_ends(all_dates) if i >= LOOKBACK]
    rebal = rebal[-(months + 1):]            # months periods need months+1 rebalance dates
    if len(rebal) < 2:
        return {"error": f"need {months} month-ends after a {LOOKBACK}-session warm-up; have {len(rebal)}"}

    equity, curve, turnover, held = 1.0, [], 0, []
    nifty_ema: dict[date, float] = {}
    nifty_closes = [nifty[d] for d in all_dates]
    for i, d in enumerate(all_dates):
        if i >= 199:
            nifty_ema[d] = sum(nifty_closes[i - 199:i + 1]) / 200
    for k in range(len(rebal) - 1):
        i0, i1 = rebal[k], rebal[k + 1]
        d0, d1 = all_dates[i0], all_dates[i1]
        scored = []
        for sym, px in series.items():
            hist = [px[x] for x in all_dates[:i0 + 1]]
            sc = _score(hist)
            if sc and sc["score"] > 0 and sc["above_200dma"]:
                scored.append((sc["score"], sym))
        scored.sort(reverse=True)
        risk_on = nifty[d0] >= nifty_ema.get(d0, nifty[d0])
        pick = [s for _, s in scored[:n]] if risk_on else []
        turnover += len(set(pick) ^ set(held))
        held = pick
        if pick:
            period = sum(series[s][d1] / series[s][d0] - 1 for s in pick) / len(pick)
            period -= cost_pct / 100 * len(set(pick) ^ set(held)) / max(len(pick), 1)
        else:
            period = 0.0
        equity *= 1 + period
        curve.append({"date": d1.isoformat(), "equity": round(equity, 4), "held": pick,
                      "nifty": round(nifty[d1] / nifty[all_dates[rebal[0]]], 4), "risk_on": risk_on})
    start, end = all_dates[rebal[0]], all_dates[rebal[-1]]
    yrs = max((end - start).days / 365.25, 1 / 12)
    peak = dd = 0.0
    for c in curve:
        peak = max(peak, c["equity"])
        dd = max(dd, (peak - c["equity"]) / peak)
    strat_ret = (equity - 1) * 100
    idx_ret = (nifty[end] / nifty[start] - 1) * 100
    return {"from": start.isoformat(), "to": end.isoformat(), "months": len(curve), "top_n": n,
            "universe_size": len(series),
            "strategy_return_pct": round(strat_ret, 2), "nifty_return_pct": round(idx_ret, 2),
            "excess_pct": round(strat_ret - idx_ret, 2),
            "strategy_cagr_pct": round(((equity) ** (1 / yrs) - 1) * 100, 2),
            "nifty_cagr_pct": round(((nifty[end] / nifty[start]) ** (1 / yrs) - 1) * 100, 2),
            "max_drawdown_pct": round(dd * 100, 2), "rotations": turnover,
            "beats_index": strat_ret > idx_ret,
            "data": "synthetic (paper simulator)" if not settings.uses_kite_market_data else "real",
            "curve": curve}
