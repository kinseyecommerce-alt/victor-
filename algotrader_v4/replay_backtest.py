"""
replay_backtest.py — replay real minute bars through the SAME code that trades live.

The older backtest_engine.py uses simplified 15-minute signal rules that only approximate
the agents. This one feeds each bar into the live agent's evaluate_tick() (identical
patterns, scoring, cooldowns), sizes like risk_manager, fills at the next bar with slippage,
charges costs.py, and manages the open trade with the real TrailingSLEngine (breakeven,
trail, targets) plus a pessimistic intrabar rule: if a bar touches both the stop and the
target, the stop fills first.

Data, in order: Kite minute history (needs a Kite session), yfinance (1-minute bars for the
last 7 days, 5-minute bars up to 60 days — flagged), or, in the paper simulator, a synthetic
intraday series derived from the symbol's synthetic daily history (flagged "synthetic",
proves the machinery only).

Agents keep per-symbol state in class-level dicts, so the replay runs every symbol under a
"#BT" alias — a backtest never touches the live agents' state.
"""
from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from config import settings
from costs import trade_cost
from ist_clock import now_ist
from tick_engine import Candle, IndicatorCalc, MarketSnapshot, Tick, TickBuffer
from trailing_sl_engine import TrailingSLEngine, TRAIL_CONFIGS

SQUARE_OFF = time(15, 15)
MAX_CNC_SESSIONS = 15
_LAST: dict[str, dict] = {}                   # last result per agent (for the dashboard / research loop)


# ── Data ─────────────────────────────────────────────────────────────────────

def _from_kite(symbol: str, days: int) -> list[dict]:
    from kite_client import kite_client
    tok = kite_client.get_instrument_tokens([symbol], "NSE").get(symbol)
    if not tok:
        return []
    end = now_ist().replace(tzinfo=None)
    rows = kite_client.historical_data(tok, end - timedelta(days=days), end, "minute")
    return [{"ts": r["date"].replace(tzinfo=None) if getattr(r["date"], "tzinfo", None) else r["date"],
             "open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]),
             "close": float(r["close"]), "volume": int(r.get("volume") or 0)} for r in rows]


def _from_yfinance(symbol: str, days: int) -> tuple[list[dict], str]:
    from market_data import yf_client
    interval, period = ("1m", "7d") if days <= 7 else ("5m", "60d")
    df = yf_client.historical(symbol, "NSE", interval=interval, period=period)
    if df.empty:
        return [], interval
    cutoff = now_ist().replace(tzinfo=None) - timedelta(days=days)
    out = []
    for r in df.itertuples():
        ts = r.date.to_pydatetime() if hasattr(r.date, "to_pydatetime") else r.date
        if ts.tzinfo:
            ts = ts.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
        if ts >= cutoff:
            out.append({"ts": ts, "open": float(r.open), "high": float(r.high), "low": float(r.low),
                        "close": float(r.close), "volume": int(r.volume)})
    return out, interval


def _synthetic(symbol: str, days: int) -> list[dict]:
    """Intraday bars inside each synthetic daily bar (paper simulator only): a random walk
    from open to close with the day's high/low respected, repeatable per symbol."""
    from daily_data import daily_closes, _completed_days
    daily = _completed_days(symbol, 1000.0)[-days:]
    rng = random.Random(symbol)
    out = []
    for d in daily:
        o, c, hi, lo = d["open"], d["close"], d["high"], d["low"]
        n = 375
        path = [o]
        for i in range(1, n):
            drift = (c - path[-1]) / (n - i)
            path.append(max(lo, min(hi, path[-1] + drift + rng.gauss(0, (hi - lo) / 60))))
        path[-1] = c
        ts0 = datetime.combine(d["date"], time(9, 15))
        for i in range(n):
            px = path[i]
            prev = path[i - 1] if i else o
            out.append({"ts": ts0 + timedelta(minutes=i), "open": prev, "high": max(prev, px) * (1 + abs(rng.gauss(0, 0.0003))),
                        "low": min(prev, px) * (1 - abs(rng.gauss(0, 0.0003))), "close": px,
                        "volume": int(d["volume"] / n * rng.uniform(0.3, 2.5))})
    return out


def load_bars(symbol: str, days: int) -> tuple[list[dict], str]:
    """(bars, source) — source names the feed and interval so results can be judged."""
    if settings.uses_kite_market_data:
        try:
            rows = _from_kite(symbol, days)
            if rows:
                return rows, "kite:1m"
        except Exception as exc:
            logger.warning("[replay] Kite history {}: {}", symbol, exc)
        try:
            rows, iv = _from_yfinance(symbol, days)
            if rows:
                return rows, f"yfinance:{iv}"
        except Exception as exc:
            logger.warning("[replay] yfinance {}: {}", symbol, exc)
        return [], "none"
    return _synthetic(symbol, days), "synthetic:1m"


# ── Replay ───────────────────────────────────────────────────────────────────

@dataclass
class ReplayTrade:
    symbol: str
    side: str
    entry_ts: datetime
    exit_ts: datetime
    entry: float
    exit: float
    qty: int
    gross: float
    cost: float
    net: float
    exit_reason: str
    bars_held: int
    signal: str = ""


@dataclass
class ReplayResult:
    agent: str
    symbols: list[str]
    days: int
    source: str
    bars: int = 0
    trades: list[ReplayTrade] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    period: tuple = ("", "")

    def metrics(self) -> dict:
        nets = [t.net for t in self.trades]
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n <= 0]
        gp, gl = sum(wins), -sum(losses)
        cum = peak = dd = 0.0
        for n in nets:
            cum += n; peak = max(peak, cum); dd = max(dd, peak - cum)
        mean = sum(nets) / len(nets) if nets else 0.0
        sd = math.sqrt(sum((n - mean) ** 2 for n in nets) / (len(nets) - 1)) if len(nets) > 1 else 0.0
        capital = sum(t.entry * t.qty for t in self.trades) / len(self.trades) if self.trades else 0.0
        return {
            "trades": len(nets), "wins": len(wins), "losses": len(losses),
            "win_rate": round(len(wins) / len(nets) * 100, 1) if nets else None,
            "gross_pnl": round(sum(t.gross for t in self.trades), 2), "costs": round(sum(t.cost for t in self.trades), 2),
            "net_pnl": round(sum(nets), 2), "avg_net": round(mean, 2) if nets else None,
            "profit_factor": round(gp / gl, 2) if gl > 0 else (None if not wins else 99.0),
            "max_drawdown": round(dd, 2), "sharpe_per_trade": round(mean / sd * math.sqrt(len(nets)), 2) if sd > 0 else None,
            "avg_bars_held": round(sum(t.bars_held for t in self.trades) / len(nets), 1) if nets else None,
            "avg_position_value": round(capital, 0),
            "return_on_avg_position_pct": round(sum(nets) / capital * 100, 2) if capital else None,
            "exit_reasons": {r: sum(1 for t in self.trades if t.exit_reason == r) for r in sorted({t.exit_reason for t in self.trades})},
        }

    def to_dict(self, include_trades: bool = True) -> dict:
        d = {"agent": self.agent, "symbols": self.symbols, "days": self.days, "source": self.source, "bars": self.bars,
             "period": {"from": self.period[0], "to": self.period[1]}, "params": self.params, **self.metrics()}
        if include_trades:
            d["trades"] = [{**vars(t), "entry_ts": t.entry_ts.isoformat(), "exit_ts": t.exit_ts.isoformat()} for t in self.trades]
        return d


def _agent(name: str, params: Optional[dict]):
    from agents.strategy_agents import IntradayAgent, ScalpingAgent, SwingAgent
    cls = {"intraday": IntradayAgent, "scalping": ScalpingAgent, "swing": SwingAgent}.get(name)
    if not cls:
        raise ValueError(f"replay supports intraday, scalping, swing (not {name})")
    a = cls()
    for k, v in (params or {}).items():       # instance attrs shadow the class defaults
        if not hasattr(a, k):
            raise ValueError(f"{name} has no parameter {k}")
        setattr(a, k, v)
    return a


_FRAMES: dict[tuple, pd.DataFrame] = {}


def _frame_for(symbol: str, bars: list[dict]) -> pd.DataFrame:
    """Vectorised indicator frame for a bar series (same formulas as the live path, ~100× faster),
    cached so an optimiser sweeping parameters over the same bars computes it once."""
    key = (symbol, len(bars), bars[0]["ts"], bars[-1]["ts"], round(bars[-1]["close"], 4))
    f = _FRAMES.get(key)
    if f is None:
        if len(_FRAMES) > 64:
            _FRAMES.clear()
        full = pd.DataFrame([{"open": x["open"], "high": x["high"], "low": x["low"], "close": x["close"],
                              "volume": x["volume"]} for x in bars])
        f = _FRAMES[key] = IndicatorCalc.compute_frame(full)
    return f


class _Clock:
    """Lets the agents' now_ist() follow the replayed bar."""
    def __init__(self):
        self.now = datetime.now()
    def __call__(self):
        return self.now


async def _replay_symbol(agent, name: str, symbol: str, bars: list[dict], slippage_bps: float,
                         product: str) -> list[ReplayTrade]:
    import agents.strategy_agents as sa
    alias = f"{symbol}#BT"
    clock = _Clock()
    real_now = sa.now_ist
    sa.now_ist = clock
    try:
        buf1, buf5 = TickBuffer(60, maxlen=500), TickBuffer(300, maxlen=200)
        frame = _frame_for(symbol, bars)       # all indicators at once; cached across parameter sweeps
        eng = TrailingSLEngine()
        exits: list[tuple[str, float]] = []
        async def on_hit(pos, ltp, pnl): exits.append(("TSL_STOP", ltp))
        async def on_tgt(pos, ltp, level):
            if level == 2: exits.append(("TARGET2", ltp))
        eng.on_sl_hit, eng.on_target_hit = on_hit, on_tgt

        trades: list[ReplayTrade] = []
        open_t: Optional[dict] = None
        pending: Optional[dict] = None
        day_open = day_high = day_low = None
        cur_day = None
        slip = slippage_bps / 10000

        for i, b in enumerate(bars):
            ts = b["ts"]
            clock.now = ts
            if ts.date() != cur_day:
                cur_day, day_open, day_high, day_low = ts.date(), b["open"], b["high"], b["low"]
                if name == "swing" and hasattr(agent, "_last_eval"):
                    agent._last_eval.pop(alias, None)
            day_high, day_low = max(day_high, b["high"]), min(day_low, b["low"])

            # 1. fill a pending entry at this bar's open
            if pending:
                px = b["open"] * (1 + slip if pending["side"] == "BUY" else 1 - slip)
                qty = pending["qty"]
                oid = f"BT-{len(trades)}"
                pos = eng.register(symbol=alias, strategy=name, side=pending["side"], entry_price=px, quantity=qty,
                                   order_id=oid, atr=pending["atr"])
                pos.current_sl = pending["stop"] if pending["side"] == "BUY" and pending["stop"] < px or \
                    pending["side"] == "SELL" and pending["stop"] > px else pos.current_sl
                open_t = {"side": pending["side"], "entry": px, "qty": qty, "ts": ts, "i": i, "oid": oid,
                          "target": pending["target"], "signal": pending["signal"], "day": ts.date(),
                          "days_held": 0}
                pending = None

            # 2. manage the open trade on this bar
            if open_t:
                pos = eng._positions.get(open_t["oid"])
                side, reason, xpx = open_t["side"], None, None
                stop = pos.current_sl if pos else None
                if ts.date() != open_t["day"]:
                    open_t["day"], open_t["days_held"] = ts.date(), open_t["days_held"] + 1
                hit_stop = stop is not None and ((side == "BUY" and b["low"] <= stop) or (side == "SELL" and b["high"] >= stop))
                tgt = open_t["target"]
                hit_tgt = tgt and ((side == "BUY" and b["high"] >= tgt) or (side == "SELL" and b["low"] <= tgt))
                if hit_stop:                       # pessimistic: stop before target inside one bar
                    reason, xpx = "STOP", stop * (1 - slip if side == "BUY" else 1 + slip)
                elif hit_tgt:
                    reason, xpx = "TARGET", tgt
                else:
                    exits.clear()
                    await eng.on_tick(alias, b["close"], 0.0)
                    if exits:
                        reason, xpx = exits[0][0], b["close"]
                    elif product == "MIS" and ts.time() >= SQUARE_OFF:
                        reason, xpx = "SQUARE_OFF", b["close"]
                    elif product == "CNC" and open_t["days_held"] >= MAX_CNC_SESSIONS:
                        reason, xpx = "TIME", b["close"]
                if reason:
                    e, q = open_t["entry"], open_t["qty"]
                    gross = (xpx - e) * q * (1 if side == "BUY" else -1)
                    cost = trade_cost(e, xpx, q, side, "NSE", product, symbol)
                    trades.append(ReplayTrade(symbol, side, open_t["ts"], ts, round(e, 2), round(xpx, 2), q,
                                              round(gross, 2), cost, round(gross - cost, 2), reason,
                                              i - open_t["i"], open_t["signal"]))
                    eng.deregister(open_t["oid"])
                    open_t = None

            # 3. feed the bar and ask the agent
            tick = Tick(symbol=alias, ltp=b["close"], bid=b["close"] * 0.9998, ask=b["close"] * 1.0002,   # ~0.04% spread, liquid large-cap
                        volume=b["volume"], change=b["close"] - day_open, change_pct=(b["close"] / day_open - 1) * 100,
                        high=day_high, low=day_low, open=day_open, timestamp=ts)
            buf1.push(b["open"], 0, ts); buf1.push(b["high"], 0, ts); buf1.push(b["low"], 0, ts); buf1.push(b["close"], b["volume"], ts)
            buf5.push(b["close"], b["volume"], ts)
            ind = IndicatorCalc.from_row(alias, tick, frame.iloc[i], min(i + 1, 500))
            snap = MarketSnapshot(symbol=alias, tick=tick, indicators=ind, candles_1min=buf1.candles()[-60:],
                                  candles_5min=buf5.candles()[-30:])
            if open_t or pending or len(snap.candles_1min) < 30:
                continue
            if product == "MIS" and ts.time() >= time(15, 0):
                continue
            try:
                action, sig = agent.evaluate_tick(snap)
            except Exception as exc:
                logger.debug("[replay] {} {}: {}", name, symbol, exc)
                continue
            if action in ("BUY", "SELL") and sig:
                if name == "swing" and action != "BUY":
                    continue
                from risk_manager import risk_manager
                qty = risk_manager.calculate_quantity(b["close"], agent=name)
                cap_q = int(settings.total_capital * settings.risk_per_trade_pct / 100 // max(abs(b["close"] - sig.get("stop_loss", b["close"] * 0.99)), 0.01))
                qty = max(1, min(qty, cap_q))
                pending = {"side": action, "qty": qty, "stop": float(sig.get("stop_loss") or 0) or
                           (b["close"] * (1 - TRAIL_CONFIGS[name].initial_sl_pct / 100) if action == "BUY" else
                            b["close"] * (1 + TRAIL_CONFIGS[name].initial_sl_pct / 100)),
                           "target": float(sig.get("target") or 0) or None, "atr": ind.atr_14,
                           "signal": str(sig.get("trigger") or sig.get("pattern") or "")[:60]}
        if open_t:                                  # close whatever is left at the last bar
            b = bars[-1]; e, q, side = open_t["entry"], open_t["qty"], open_t["side"]
            gross = (b["close"] - e) * q * (1 if side == "BUY" else -1)
            cost = trade_cost(e, b["close"], q, side, "NSE", product, symbol)
            trades.append(ReplayTrade(symbol, side, open_t["ts"], b["ts"], round(e, 2), round(b["close"], 2), q,
                                      round(gross, 2), cost, round(gross - cost, 2), "END", len(bars) - open_t["i"], open_t["signal"]))
        return trades
    finally:
        sa.now_ist = real_now
        for attr in vars(type(agent)).values():     # drop the alias keys from the shared class dicts
            if isinstance(attr, dict):
                attr.pop(alias, None)


async def replay_async(agent_name: str, symbols: list[str], days: int = 30, slippage_bps: float = 5.0,
                       params: Optional[dict] = None, bars_by_symbol: Optional[dict[str, list[dict]]] = None) -> ReplayResult:
    agent = _agent(agent_name, params)
    product = "CNC" if agent_name == "swing" else "MIS"
    res = ReplayResult(agent=agent_name, symbols=list(symbols), days=days, source="", params=params or {})
    sources, first, last = set(), None, None
    for sym in symbols:
        if bars_by_symbol and sym in bars_by_symbol:
            bars, src = bars_by_symbol[sym], "provided"
        else:
            bars, src = await asyncio.to_thread(load_bars, sym, days)
        if not bars:
            continue
        sources.add(src)
        res.bars += len(bars)
        first = min(first, bars[0]["ts"]) if first else bars[0]["ts"]
        last = max(last, bars[-1]["ts"]) if last else bars[-1]["ts"]
        res.trades.extend(await _replay_symbol(agent, agent_name, sym, bars, slippage_bps, product))
    res.trades.sort(key=lambda t: t.entry_ts)
    res.source = ",".join(sorted(sources)) or "none"
    res.period = (first.isoformat() if first else "", last.isoformat() if last else "")
    _LAST[agent_name] = res.to_dict(include_trades=False)
    return res


def replay(agent_name: str, symbols: list[str], days: int = 30, slippage_bps: float = 5.0,
           params: Optional[dict] = None, bars_by_symbol: Optional[dict] = None) -> ReplayResult:
    return asyncio.run(replay_async(agent_name, symbols, days, slippage_bps, params, bars_by_symbol))


def benchmark_for(res: ReplayResult) -> dict:
    """Buy-and-hold of the symbols and of NIFTY over the replay period, for the same money."""
    from performance import performance
    out = {"nifty50_pct": None, "symbols_avg_pct": None}
    if not res.period[0]:
        return out
    d0, d1 = res.period[0][:10], res.period[1][:10]
    out["nifty50_pct"] = performance.benchmark_return("NIFTY50", d0, d1)
    try:
        from daily_data import daily_closes
        rets = []
        for s in res.symbols:
            closes = [(d.isoformat(), c) for d, c in daily_closes(s)]
            a = [c for d, c in closes if d <= d0]; b = [c for d, c in closes if d <= d1]
            if a and b:
                rets.append((b[-1] / a[-1] - 1) * 100)
        out["symbols_avg_pct"] = round(sum(rets) / len(rets), 2) if rets else None
    except Exception:
        pass
    m = res.metrics()
    out["strategy_pct"] = m.get("return_on_avg_position_pct")
    out["beats_nifty"] = (out["strategy_pct"] is not None and out["nifty50_pct"] is not None
                          and out["strategy_pct"] > out["nifty50_pct"])
    return out
