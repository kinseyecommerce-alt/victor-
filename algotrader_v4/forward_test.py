"""
forward_test.py — live-market paper test of the opening-auction fade (stratlab/auction.py).

The 3-year lab test of this strategy used the final 09:15 auction price for both the signal and the fill.
Live, an order has to go in during the 09:00-09:08 pre-open, when only the indicative price is known.
So this module records, every trading day:

  09:07:30  "indicative": quotes of the 97 lab symbols while orders can still be placed
  09:08:45  "auction":    quotes just after the call auction has matched
  09:15:05  "open":       quotes at the start of continuous trading (ohlc.open = official open)
  09:20:00  "exit":       quotes at the 5-minute exit

and builds paper trades for three variants (gap > 3 %, > 2 %, 1-3 %) under two rules:

  final  the signal from the final auction price (exactly what the lab tested)
  live   the signal from the 09:07:30 indicative price (what a real pre-open order can use)

Both fill at the auction price ± slippage, exit at the 09:20 price ± slippage, ₹1 lakh notional each, at most
10 trades per day (largest |gap| first), |gap| > 20 % excluded, 2026 Zerodha MIS costs. Nothing here places an
order with the paper broker or touches the agents' book: it is a separate ledger (data/forward_test.json),
kept across restarts, read through GET /forward-test.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from costs import trade_cost
from ist_clock import now_ist

FILE = Path(os.getenv("FORWARD_TEST_FILE") or Path(os.getenv("PERFORMANCE_FILE") or
            Path(__file__).resolve().parent / "data" / "performance.json").parent / "forward_test.json")

# the 97 NIFTY-100 symbols of the 3-year lab data set (bars3y)
UNIVERSE = (
    "ABB ADANIENT ADANIGREEN ADANIPORTS ADANIPOWER AMBUJACEM APOLLOHOSP ASIANPAINT AUROPHARMA AXISBANK BAJAJ-AUTO "
    "BAJAJFINSV BAJFINANCE BANKBARODA BERGEPAINT BHARTIARTL BOSCHLTD BPCL BRITANNIA CANBK CHOLAFIN CIPLA COALINDIA "
    "COLPAL DABUR DIVISLAB DLF DMART DRREDDY EICHERMOT ETERNAL GAIL GODREJCP GRASIM HAL HAVELLS HCLTECH HDFCBANK "
    "HDFCLIFE HEROMOTOCO HINDALCO HINDPETRO HINDUNILVR ICICIBANK ICICIGI INDHOTEL INDUSINDBK INFY IOC IRCTC ITC "
    "JIOFIN JSWSTEEL JUBLFOOD KOTAKBANK LODHA LT LUPIN M&M MARUTI MOTHERSON MPHASIS NAUKRI NESTLEIND NHPC NMDC NTPC "
    "OFSS ONGC PAGEIND PAYTM PETRONET PIDILITIND PIIND POWERGRID RECLTD RELIANCE SBILIFE SBIN SHRIRAMFIN SIEMENS "
    "SUNPHARMA TATACONSUM TATAPOWER TATASTEEL TCS TECHM TITAN TORNTPHARM TRENT TVSMOTOR ULTRACEMCO UNITDSPR VBL VEDL "
    "WIPRO ZYDUSLIFE"
).split()

STAGES = ("indicative", "auction", "open", "exit")
VARIANTS = {"AUC-FADE-3PCT": (0.03, None), "AUC-FADE-2PCT": (0.02, None), "AUC-FADE-1TO3PCT": (0.01, 0.03)}
RULES = ("final", "live")
NOTIONAL = 100_000
MAX_PER_DAY = 10
MAX_GAP = 0.20
SLIPPAGE_BPS = 2.1                # as in stratlab/auction.py
_EPS = 1e-9


# ── storage ──────────────────────────────────────────────────────────────────
def _load() -> dict:
    try:
        if FILE.exists():
            return json.loads(FILE.read_text())
    except Exception as exc:
        logger.warning("[forward] cannot read {}: {}", FILE, exc)
    return {"days": {}}


def _save(state: dict) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, separators=(",", ":")))
    tmp.replace(FILE)


def _compact(q: dict) -> dict:
    """The fields we keep from one Kite quote."""
    ohlc = q.get("ohlc") or {}
    depth = q.get("depth") or {}
    bid = (depth.get("buy") or [{}])[0].get("price") or 0.0
    ask = (depth.get("sell") or [{}])[0].get("price") or 0.0
    return {"ltp": q.get("last_price") or 0.0, "open": ohlc.get("open") or 0.0, "prev_close": ohlc.get("close") or 0.0,
            "bid": bid, "ask": ask, "volume": q.get("volume") or 0,
            "buy_qty": q.get("buy_quantity") or 0, "sell_qty": q.get("sell_quantity") or 0,
            "ts": str(q.get("timestamp") or "")}


# ── pure trade construction (tested in test_pipeline.py) ─────────────────────
def gap(px: float, prev_close: float) -> Optional[float]:
    if not px or not prev_close or px <= 0 or prev_close <= 0:
        return None
    return px / prev_close - 1


def eligible(g: Optional[float], lo: float, hi: Optional[float]) -> bool:
    if g is None or abs(g) > MAX_GAP:
        return False
    a = abs(g)
    return a > lo + _EPS and (hi is None or a <= hi + _EPS)


def auction_price(stages: dict, sym: str) -> float:
    """The final auction price: the official open once continuous trading starts, else the post-match quote."""
    o = (stages.get("open") or {}).get(sym) or {}
    a = (stages.get("auction") or {}).get(sym) or {}
    return o.get("open") or a.get("open") or a.get("ltp") or 0.0


def prev_close(stages: dict, sym: str) -> float:
    for st in ("open", "auction", "indicative", "exit"):
        pc = ((stages.get(st) or {}).get(sym) or {}).get("prev_close")
        if pc:
            return pc
    return 0.0


def build_trades(stages: dict, slippage_bps: float = SLIPPAGE_BPS) -> dict:
    """{variant: {rule: [trade, ...]}} from one day's stage quotes."""
    s = slippage_bps / 1e4
    out: dict = {v: {r: [] for r in RULES} for v in VARIANTS}
    exits = stages.get("exit") or {}
    for v, (lo, hi) in VARIANTS.items():
        for rule in RULES:
            cands = []
            for sym in UNIVERSE:
                pc, ap = prev_close(stages, sym), auction_price(stages, sym)
                xp = (exits.get(sym) or {}).get("ltp") or 0.0
                if not (pc and ap and xp):
                    continue
                if rule == "final":
                    g_sig = gap(ap, pc)
                else:
                    ind = (stages.get("indicative") or {}).get(sym) or {}
                    g_sig = gap(ind.get("ltp") or 0.0, pc)
                if not eligible(g_sig, lo, hi):
                    continue
                cands.append((abs(g_sig), sym, g_sig, pc, ap, xp))
            cands.sort(key=lambda c: (-c[0], c[1]))
            for _, sym, g_sig, pc, ap, xp in cands[:MAX_PER_DAY]:
                qty = int(NOTIONAL // ap)
                if qty <= 0:
                    continue
                side = "SELL" if g_sig > 0 else "BUY"            # fade the gap
                if side == "SELL":
                    entry, exit_ = ap * (1 - s), xp * (1 + s)
                    gross = (entry - exit_) * qty
                else:
                    entry, exit_ = ap * (1 + s), xp * (1 - s)
                    gross = (exit_ - entry) * qty
                cost = trade_cost(entry, exit_, qty, side, "NSE", "MIS", sym)
                out[v][rule].append({
                    "symbol": sym, "side": side, "qty": qty, "signal_gap_pct": round(g_sig * 100, 3),
                    "auction_gap_pct": round((gap(ap, pc) or 0) * 100, 3), "prev_close": pc,
                    "auction_px": ap, "exit_px": xp, "entry_fill": round(entry, 2), "exit_fill": round(exit_, 2),
                    "gross": round(gross, 2), "cost": cost, "net": round(gross - cost, 2),
                    "net_bps": round((gross - cost) / (ap * qty) * 1e4, 2)})
    return out


def indicative_quality(stages: dict) -> dict:
    """How well the 09:07:30 indicative price predicted the final auction price (the live strategy's key unknown)."""
    diffs, agree, n_big = [], 0, 0
    for sym in UNIVERSE:
        pc, ap = prev_close(stages, sym), auction_price(stages, sym)
        ind = ((stages.get("indicative") or {}).get(sym) or {}).get("ltp") or 0.0
        gi, ga = gap(ind, pc), gap(ap, pc)
        if gi is None or ga is None:
            continue
        diffs.append((ind / ap - 1) * 1e4)
        if abs(ga) > 0.02:
            n_big += 1
            agree += eligible(gi, 0.02, None) and math.copysign(1, gi) == math.copysign(1, ga)
    if not diffs:
        return {"symbols": 0}
    diffs.sort()
    return {"symbols": len(diffs), "median_abs_bps": round(sorted(abs(d) for d in diffs)[len(diffs) // 2], 2),
            "mean_bps": round(sum(diffs) / len(diffs), 2), "same_as_prev_close": sum(
                1 for sym in UNIVERSE if ((stages.get("indicative") or {}).get(sym) or {}).get("ltp")
                and ((stages.get("indicative") or {}).get(sym) or {}).get("ltp") == prev_close(stages, sym)),
            "gaps_over_2pct": n_big, "caught_by_indicative": agree}


def summarize(trades_by_day: dict) -> dict:
    out = {}
    for v in VARIANTS:
        for rule in RULES:
            ts = [t for d in sorted(trades_by_day) for t in (trades_by_day[d].get(v) or {}).get(rule, [])]
            wins = sum(t["net"] for t in ts if t["net"] > 0)
            losses = -sum(t["net"] for t in ts if t["net"] < 0)
            out[f"{v}/{rule}"] = {
                "days": sum(1 for d in trades_by_day if (trades_by_day[d].get(v) or {}).get(rule)),
                "trades": len(ts), "net": round(sum(t["net"] for t in ts), 2),
                "gross": round(sum(t["gross"] for t in ts), 2), "costs": round(sum(t["cost"] for t in ts), 2),
                "win_rate": round(sum(t["net"] > 0 for t in ts) / len(ts), 3) if ts else None,
                "profit_factor": round(wins / losses, 2) if losses else (None if not ts else float("inf")),
                "avg_net_bps": round(sum(t["net_bps"] for t in ts) / len(ts), 2) if ts else None}
    return out


# ── live jobs ────────────────────────────────────────────────────────────────
def record(stage: str) -> dict:
    """Fetch quotes for the universe and store them under today's `stage`; at "exit" also build the trades."""
    from config import settings
    from kite_client import kite_client
    if stage not in STAGES:
        raise ValueError(stage)
    today = now_ist().date().isoformat()
    if not settings.uses_kite_market_data:
        logger.warning("[forward] {}: no Kite market data (PAPER_DATA_SOURCE is not KITE)", stage)
        return {"stage": stage, "error": "no Kite market data"}
    quotes = kite_client.quote_kite([f"NSE:{s}" for s in UNIVERSE]) or {}
    snap = {k.split(":", 1)[1]: _compact(q) for k, q in quotes.items()}
    state = _load()
    day = state["days"].setdefault(today, {"stages": {}, "at": {}})
    day["stages"][stage] = snap
    day["at"][stage] = now_ist().strftime("%H:%M:%S")
    out = {"stage": stage, "symbols": len(snap)}
    if stage == "exit":
        day["trades"] = build_trades(day["stages"])
        day["indicative_quality"] = indicative_quality(day["stages"])
        out["trades"] = {f"{v}/{r}": len(day["trades"][v][r]) for v in VARIANTS for r in RULES}
        out["net"] = {f"{v}/{r}": round(sum(t["net"] for t in day["trades"][v][r]), 2) for v in VARIANTS for r in RULES}
    _save(state)
    logger.info("[forward] {} recorded: {}", stage, out)
    return out


def status(include_quotes: bool = False) -> dict:
    state = _load()
    days = state.get("days", {})
    trades = {d: v["trades"] for d, v in days.items() if v.get("trades")}
    today = now_ist().date().isoformat()
    t = days.get(today, {})
    return {
        "strategy": "opening-auction fade (stratlab/auction.py), paper forward test; separate ledger, no broker orders",
        "rules": {"final": "signal from the final auction price (as tested)",
                  "live": "signal from the 09:07:30 indicative price (what a real pre-open order can use)"},
        "summary": summarize(trades),
        "today": {"date": today, "stages_recorded": t.get("at", {}),
                  "trades": t.get("trades"), "indicative_quality": t.get("indicative_quality"),
                  **({"stages": t.get("stages")} if include_quotes else {})},
        "days": {d: {"indicative_quality": v.get("indicative_quality"),
                     "net": {f"{k}/{r}": round(sum(x["net"] for x in v["trades"][k][r]), 2)
                             for k in VARIANTS for r in RULES} if v.get("trades") else None}
                 for d, v in sorted(days.items())},
    }
