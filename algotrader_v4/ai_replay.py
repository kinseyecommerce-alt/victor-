"""
ai_replay.py — replay the AI trader's decision-making over real past sessions.

The live AI trader (ai_trader.py) reads live state: the tick engine, Kite positions, option chains,
levels and so on, so it cannot be pointed at the past directly. This module rebuilds the part of
that loop that can be judged historically — the model's intraday equity decisions — on real
1-minute bars:

  for each session day, at fixed checkpoints (09:45 … 14:45):
      1. bars since the last checkpoint are walked for stops / targets (stop first when both hit)
      2. the model sees a compact market table as of that minute (price, day range, opening range,
         VWAP side, EMA stack, RSI, ADX, volume vs 20-day, 20-day trend, NIFTY change) plus its
         open positions and today's trades, and acts through three tools:
         place_trade / close_position / done
      3. fills are at the next 1-minute bar's open with slippage; costs use costs.trade_cost
  15:00 → everything is squared off, like the live MIS rule.

Hard limits mirror the live tool checks (stop 0.2–2 % on the correct side, target ≥ 1.5R, max
open positions, max trades per day, per-trade value cap, risk-based quantity). The model is the
one configured for the AI trader, through llm_provider.llm_router (Anthropic → OpenAI failover).
Token cost is accounted with the same price table as the live trader.

What it does NOT replay: options, futures, swing trades, the real-time alert path, the bear-case
debate gate, levels/flow/event context. It answers "how does the model's intraday stock judgement
do on real days after costs", nothing more.
"""
from __future__ import annotations

import asyncio
import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from config import settings
from costs import trade_cost
from ist_clock import now_ist

CHECKPOINTS = ["09:45", "10:15", "10:45", "11:15", "11:45", "12:15", "12:45", "13:15", "13:45", "14:15", "14:45"]
SQUARE_OFF = time(15, 0)
INDEX = "NIFTY 50"

DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "KOTAKBANK", "BHARTIARTL", "ITC",
                   "HINDUNILVR", "ASIANPAINT", "SBIN", "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
                   "SUNPHARMA", "ULTRACEMCO", "NTPC", "TATASTEEL"]

SYSTEM_PROMPT = """You are the autonomous trader inside AlgoTrader Pro, replaying past NSE sessions \
on real data so the account owner can judge your intraday stock trading after costs. Treat it exactly \
like a live session: capital preservation first; a skipped trade costs nothing, a bad one costs money. \
Being in a trade is never the goal; being right about the ones you take is.

At every checkpoint you get a table of the watchlist as of that minute and your open positions. Decide \
whether to open, close or do nothing, using the tools. Rules:
- Only trade a clear setup: trend and momentum agreeing (VWAP side, EMA9/21/50 stack, ADX with the \
matching DI, RSI not exhausted, volume well above normal), ideally in the direction of the 20-day trend \
and of NIFTY's day; or a clean mean reversion at a defined level. Prefer no trade over a marginal one. \
Most checkpoints should end with no new trade.
- Every trade needs a stop on the correct side of price, 0.2–2 % away, based on structure (opening range, \
day low/high, VWAP) or about 1–1.5x ATR, and a target of at least 1.5x the risk. Round trip costs are \
about ₹40–50 per trade, so targets under ~0.5 % rarely pay.
- Short selling (SELL first) is allowed intraday. Everything is squared off at 15:00 by code, so after \
14:00 only take trades that can reach the target quickly.
- Close a position whose reason is gone (trend broken, momentum reversed), not on noise. Stops and \
targets are enforced between checkpoints.
- The tools enforce hard limits (open positions, trades per day, trade value). If a tool rejects an \
action, accept it; do not retry with tweaked numbers.
Finish each checkpoint with done(summary) — one or two sentences on what you saw and did."""

TOOLS = [
    {"name": "place_trade",
     "description": "Open an intraday (MIS) NSE equity position at the current price. BUY = long, SELL = short. "
                    "The code sizes the quantity from the stop distance and the risk limits.",
     "input_schema": {"type": "object",
                      "properties": {"symbol": {"type": "string"}, "side": {"type": "string", "enum": ["BUY", "SELL"]},
                                     "stop_loss": {"type": "number", "description": "stop price"},
                                     "target": {"type": "number", "description": "target price, at least 1.5x the risk"},
                                     "reasoning": {"type": "string", "description": "one line: the setup and why now"}},
                      "required": ["symbol", "side", "stop_loss", "target", "reasoning"], "additionalProperties": False}},
    {"name": "close_position",
     "description": "Close an open position at the current price.",
     "input_schema": {"type": "object", "properties": {"symbol": {"type": "string"}, "reasoning": {"type": "string"}},
                      "required": ["symbol", "reasoning"], "additionalProperties": False}},
    {"name": "done",
     "description": "End this checkpoint. Always call it last.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                      "required": ["summary"], "additionalProperties": False}},
]

_LAST: dict = {"status": "idle"}          # dashboard / API state of the most recent run
_FILE = Path(__file__).resolve().parent / "data" / "ai_replay_last.json"


def _sym_bars(symbol: str, days: int) -> list[dict]:
    from replay_backtest import load_bars
    bars, _ = load_bars(symbol, days)
    return bars


def _daily(bars: list[dict]) -> list[tuple[date, float]]:
    per: dict[date, float] = {}
    for b in bars:
        per[b["ts"].date()] = b["close"]
    return sorted(per.items())


class _Sym:
    """One symbol's bars + vectorised indicators, indexed by bar timestamp."""
    def __init__(self, symbol: str, bars: list[dict]):
        from replay_backtest import _frame_for
        self.symbol, self.bars = symbol, bars
        self.frame = _frame_for(symbol, bars) if len(bars) >= 60 else None
        self.idx = {b["ts"]: i for i, b in enumerate(bars)}
        self.daily = _daily(bars)

    def at(self, ts: datetime) -> Optional[int]:
        """Index of the last bar at or before ts (same day), else None."""
        i = self.idx.get(ts)
        if i is not None:
            return i
        lo, hi = 0, len(self.bars) - 1
        while lo < hi:                                      # bisect on ts
            mid = (lo + hi + 1) // 2
            if self.bars[mid]["ts"] <= ts:
                lo = mid
            else:
                hi = mid - 1
        return lo if self.bars and self.bars[lo]["ts"] <= ts and self.bars[lo]["ts"].date() == ts.date() else None

    def prev_close(self, d: date) -> Optional[float]:
        prior = [c for dd, c in self.daily if dd < d]
        return prior[-1] if prior else None

    def trend20(self, d: date) -> str:
        prior = [c for dd, c in self.daily if dd < d][-20:]
        if len(prior) < 10:
            return "n/a"
        return "UP" if prior[-1] > sum(prior) / len(prior) else "DOWN"

    def day_slice(self, d: date, upto: int) -> list[dict]:
        j = upto
        while j > 0 and self.bars[j - 1]["ts"].date() == d:
            j -= 1
        return self.bars[j:upto + 1]

    def row(self, ts: datetime) -> Optional[dict]:
        i = self.at(ts)
        if i is None or self.frame is None:
            return None
        b = self.bars[i]
        d = ts.date()
        day = self.day_slice(d, i)
        f = self.frame.iloc[i]
        orb = [x for x in day if x["ts"].time() < time(9, 30)]
        pc = self.prev_close(d)
        def g(k):
            v = f.get(k)
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        last5 = [x["close"] for x in day[-25::5]][-5:]
        return {"symbol": self.symbol, "ltp": b["close"], "i": i,
                "chg_pct": round((b["close"] / pc - 1) * 100, 2) if pc else None,
                "day_open": day[0]["open"], "day_high": max(x["high"] for x in day), "day_low": min(x["low"] for x in day),
                "orb_high": max(x["high"] for x in orb) if orb else None, "orb_low": min(x["low"] for x in orb) if orb else None,
                "vwap": g("vwap"), "ema9": g("ema9"), "ema21": g("ema21"), "ema50": g("ema50"), "rsi14": g("rsi_14"),
                "adx": g("adx_14"), "plus_di": g("plus_di"), "minus_di": g("minus_di"), "atr": g("atr_14"),
                "vol_ratio": g("volume_ratio"), "trend20": self.trend20(d), "last5": last5}


def _fmt(r: dict) -> str:
    def p(v, nd=1):
        return "-" if v is None else f"{v:.{nd}f}"
    ema = "-"
    if r["ema9"] and r["ema21"] and r["ema50"]:
        ema = "9>21>50" if r["ema9"] > r["ema21"] > r["ema50"] else "9<21<50" if r["ema9"] < r["ema21"] < r["ema50"] else "mixed"
    vw = "-" if not r["vwap"] else ("above" if r["ltp"] > r["vwap"] else "below")
    di = "-"
    if r["adx"] is not None and r["plus_di"] is not None and r["minus_di"] is not None:
        di = f"{r['adx']:.0f}{'+' if r['plus_di'] > r['minus_di'] else '-'}"
    return (f"{r['symbol']:<11}{r['ltp']:>9.2f} {p(r['chg_pct'],2):>6}% | day {r['day_low']:.1f}-{r['day_high']:.1f} "
            f"open {r['day_open']:.1f} | ORB {p(r['orb_low'])}-{p(r['orb_high'])} | VWAP {vw} {p(r['vwap'])} | EMA {ema} | "
            f"RSI {p(r['rsi14'],0)} | ADX {di} | ATR {p(r['atr'])} | vol {p(r['vol_ratio'])}x | 20d {r['trend20']} | "
            f"5m closes {' '.join(f'{x:.1f}' for x in r['last5'])}")


class AIReplay:
    def __init__(self, days: int = 5, symbols: Optional[list[str]] = None, checkpoints: Optional[list[str]] = None,
                 slippage_bps: float = 5.0, create: Optional[Callable] = None, capital: Optional[float] = None,
                 max_positions: Optional[int] = None, max_trades_per_day: Optional[int] = None,
                 max_position_value: Optional[float] = None, risk_pct: Optional[float] = None,
                 bars_by_symbol: Optional[dict[str, list[dict]]] = None, progress: Optional[Callable] = None):
        self.days = max(1, int(days))
        self.symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        self.checkpoints = [time(*map(int, c.split(":"))) for c in (checkpoints or CHECKPOINTS)]
        self.slip = slippage_bps / 10_000
        self.create = create                       # test hook: async (messages, system, tools) -> LLMResponse
        self.capital = float(capital or settings.total_capital)
        self.max_positions = int(max_positions or settings.ai_agent_max_open_positions)
        self.max_trades_per_day = int(max_trades_per_day or settings.ai_agent_max_trades_per_day)
        self.max_value = float(max_position_value or settings.ai_agent_max_position_value)
        self.risk_pct = float(risk_pct or settings.risk_per_trade_pct)
        self.bars_by_symbol = bars_by_symbol
        self.progress = progress
        self.syms: dict[str, _Sym] = {}
        self.index: Optional[_Sym] = None
        self.positions: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.decisions: list[dict] = []
        self.rejections: list[dict] = []
        self.llm = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cost_usd": 0.0, "models": {}, "errors": 0}
        self.day_trades = 0
        self.model_used: Optional[str] = None

    # ── data ─────────────────────────────────────────────────────────────
    def load(self) -> None:
        need = self.days + 40                       # 20 trading days of history for the trend + buffer
        src = self.bars_by_symbol or {}
        for s in self.symbols + [INDEX]:
            bars = src.get(s) if src else None
            if bars is None and not src:
                try:
                    bars = _sym_bars(s, need)
                except Exception as exc:
                    logger.warning("[ai-replay] {}: {}", s, exc)
                    bars = []
            if bars:
                obj = _Sym(s, bars)
                if s == INDEX:
                    self.index = obj
                elif obj.frame is not None:
                    self.syms[s] = obj
        if not self.syms:
            raise RuntimeError("no bars for any symbol")

    def session_days(self) -> list[date]:
        days = sorted({b["ts"].date() for s in self.syms.values() for b in s.bars})
        return days[-self.days:]

    # ── simulation ───────────────────────────────────────────────────────
    def _qty(self, price: float, stop: float) -> int:
        risk_cap = self.capital * self.risk_pct / 100
        by_risk = int(risk_cap // max(abs(price - stop), 0.01))
        by_value = int(self.max_value // price)
        return max(0, min(by_risk, by_value))

    def _fill(self, sym: str, i: int, side: str) -> tuple[float, datetime]:
        bars = self.syms[sym].bars
        j = min(i + 1, len(bars) - 1)
        if bars[j]["ts"].date() != bars[i]["ts"].date():
            j = i
        px = bars[j]["open"]
        px *= (1 + self.slip) if side == "BUY" else (1 - self.slip)
        return round(px, 2), bars[j]["ts"]

    def _close(self, sym: str, px: float, ts: datetime, reason: str) -> dict:
        p = self.positions.pop(sym)
        side = p["side"]
        px = round(px * ((1 - self.slip) if side == "BUY" else (1 + self.slip)), 2)
        gross = (px - p["entry"]) * p["qty"] * (1 if side == "BUY" else -1)
        cost = trade_cost(p["entry"], px, p["qty"], side, "NSE", "MIS", sym)
        t = {"symbol": sym, "side": side, "qty": p["qty"], "entry": p["entry"], "exit": px,
             "entry_ts": p["entry_ts"].isoformat(), "exit_ts": ts.isoformat(), "stop": p["stop"], "target": p["target"],
             "gross": round(gross, 2), "cost": round(cost, 2), "net": round(gross - cost, 2), "exit_reason": reason,
             "reasoning": p["reasoning"], "day": p["entry_ts"].date().isoformat()}
        self.trades.append(t)
        return t

    def _walk(self, d: date, start: time, end: time) -> None:
        """Advance every open position through the bars in (start, end], honouring stop/target."""
        for sym in list(self.positions):
            s = self.syms[sym]
            p = self.positions[sym]
            for b in s.bars[p["i"] + 1:]:
                if b["ts"].date() != d or b["ts"].time() > end:
                    break
                if b["ts"].time() <= start:
                    continue
                p["i"] = s.idx[b["ts"]]
                if p["side"] == "BUY":
                    if b["low"] <= p["stop"]:
                        self._close(sym, p["stop"], b["ts"], "STOP"); break
                    if b["high"] >= p["target"]:
                        self._close(sym, p["target"], b["ts"], "TARGET"); break
                else:
                    if b["high"] >= p["stop"]:
                        self._close(sym, p["stop"], b["ts"], "STOP"); break
                    if b["low"] <= p["target"]:
                        self._close(sym, p["target"], b["ts"], "TARGET"); break

    def _square_off(self, d: date) -> None:
        for sym in list(self.positions):
            s = self.syms[sym]
            i = s.at(datetime.combine(d, SQUARE_OFF))
            if i is None:
                i = self.positions[sym]["i"]
            self._close(sym, s.bars[i]["close"], s.bars[i]["ts"], "SQUARE_OFF")

    # ── tools ────────────────────────────────────────────────────────────
    def _tool(self, name: str, args: dict, ts: datetime) -> dict:
        if name == "done":
            return {"ok": True}
        sym = str(args.get("symbol", "")).upper().strip()
        if name == "close_position":
            if sym not in self.positions:
                return {"ok": False, "error": f"no open position in {sym}"}
            s = self.syms[sym]; i = s.at(ts)
            px, fts = self._fill(sym, i, "SELL" if self.positions[sym]["side"] == "BUY" else "BUY")
            t = self._close(sym, px / ((1 - self.slip) if self.positions[sym]["side"] == "BUY" else (1 + self.slip)), fts,
                            "AI_CLOSE")
            return {"ok": True, "closed": t}
        if name != "place_trade":
            return {"ok": False, "error": f"unknown tool {name}"}
        if sym not in self.syms:
            return {"ok": False, "error": f"{sym} is not in the watchlist"}
        if sym in self.positions:
            return {"ok": False, "error": f"already in a {self.positions[sym]['side']} position in {sym}"}
        if len(self.positions) >= self.max_positions:
            return {"ok": False, "error": f"max open positions ({self.max_positions}) reached"}
        if self.day_trades >= self.max_trades_per_day:
            return {"ok": False, "error": f"max trades per day ({self.max_trades_per_day}) reached"}
        if ts.time() >= time(14, 45):
            return {"ok": False, "error": "no new entries after 14:45 (square-off at 15:00)"}
        side = str(args.get("side", "")).upper()
        if side not in ("BUY", "SELL"):
            return {"ok": False, "error": "side must be BUY or SELL"}
        s = self.syms[sym]; i = s.at(ts)
        if i is None:
            return {"ok": False, "error": f"no price for {sym} now"}
        ltp = s.bars[i]["close"]
        try:
            stop, target = float(args.get("stop_loss")), float(args.get("target"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "stop_loss and target must be numbers"}
        if (side == "BUY" and not (stop < ltp < target)) or (side == "SELL" and not (target < ltp < stop)):
            return {"ok": False, "error": f"stop/target on the wrong side of price {ltp:.2f}"}
        sl_pct = abs(ltp - stop) / ltp * 100
        if not (settings.ai_agent_min_sl_pct <= sl_pct <= settings.ai_agent_max_sl_pct):
            return {"ok": False, "error": f"stop must be {settings.ai_agent_min_sl_pct}-{settings.ai_agent_max_sl_pct}% "
                                          f"from price (yours {sl_pct:.2f}%)"}
        rr = abs(target - ltp) / abs(ltp - stop)
        if rr < 1.5:
            return {"ok": False, "error": f"target gives {rr:.2f}R; at least 1.5R required"}
        qty = self._qty(ltp, stop)
        if qty < 1:
            return {"ok": False, "error": f"price {ltp:.2f} exceeds the per-trade value limit ₹{self.max_value:,.0f}"}
        px, fts = self._fill(sym, i, side)
        self.positions[sym] = {"side": side, "qty": qty, "entry": px, "stop": round(stop, 2), "target": round(target, 2),
                               "entry_ts": fts, "i": s.idx[fts], "reasoning": str(args.get("reasoning", ""))[:200]}
        self.day_trades += 1
        return {"ok": True, "filled": {"symbol": sym, "side": side, "qty": qty, "price": px,
                                       "value": round(px * qty), "stop": stop, "target": target, "rr": round(rr, 2)}}

    # ── model ────────────────────────────────────────────────────────────
    def _market_text(self, ts: datetime) -> str:
        lines = []
        idx = self.index.row(ts) if self.index else None
        if idx and idx["chg_pct"] is not None:
            lines.append(f"NIFTY 50 {idx['ltp']:.1f} ({idx['chg_pct']:+.2f}%) day {idx['day_low']:.0f}-{idx['day_high']:.0f} "
                         f"VWAP {'above' if idx['vwap'] and idx['ltp'] > idx['vwap'] else 'below'} 20d {idx['trend20']}")
        for s in self.syms.values():
            r = s.row(ts)
            if r and r["chg_pct"] is not None:
                lines.append(_fmt(r))
        return "\n".join(lines)

    def _positions_text(self, ts: datetime) -> str:
        if not self.positions:
            return "none"
        out = []
        for sym, p in self.positions.items():
            s = self.syms[sym]; i = s.at(ts); ltp = s.bars[i]["close"] if i is not None else p["entry"]
            pnl = (ltp - p["entry"]) * p["qty"] * (1 if p["side"] == "BUY" else -1)
            out.append(f"{sym} {p['side']} {p['qty']} @ {p['entry']:.2f} now {ltp:.2f} P&L ₹{pnl:+.0f} "
                       f"stop {p['stop']:.2f} target {p['target']:.2f} since {p['entry_ts']:%H:%M}")
        return "\n".join(out)

    async def _call(self, messages: list):
        if self.create is not None:
            return await self.create(messages=messages, system=SYSTEM_PROMPT, tools=TOOLS)
        from llm_provider import llm_router
        return await llm_router.create(messages=messages, system=SYSTEM_PROMPT, tools=TOOLS, max_tokens=4000,
                                       effort=settings.ai_agent_effort)

    def _account(self, resp) -> None:
        from llm_provider import price_for
        u = resp.usage
        inp, out = getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        p = price_for(getattr(resp, "model", None) or "")
        self.llm["calls"] += 1
        self.llm["input"] += inp; self.llm["output"] += out; self.llm["cache_read"] += cr
        self.llm["cost_usd"] += (inp * p["in"] + out * p["out"] + cr * p["cache_read"]) / 1_000_000
        m = getattr(resp, "model", None) or "?"
        self.llm["models"][m] = self.llm["models"].get(m, 0) + 1
        self.model_used = m

    async def _checkpoint(self, d: date, ts: datetime) -> None:
        today = [t for t in self.trades if t["day"] == d.isoformat()]
        tt = "\n".join(f"{t['symbol']} {t['side']} {t['qty']} {t['entry']:.2f}→{t['exit']:.2f} {t['exit_reason']} net ₹{t['net']:+.0f}"
                       for t in today) or "none"
        recent = [t for t in self.trades if t["day"] != d.isoformat()][-5:]
        lessons = "\n".join(f"{t['day']} {t['symbol']} {t['side']} {t['exit_reason']} net ₹{t['net']:+.0f} — {t['reasoning'][:80]}"
                            for t in recent) or "none yet"
        user = (f"Checkpoint {ts:%A %d %b %Y %H:%M} IST. Watchlist as of now:\n{self._market_text(ts)}\n\n"
                f"Open positions:\n{self._positions_text(ts)}\n\nToday's closed trades:\n{tt}\n\n"
                f"Recent trades on earlier days:\n{lessons}\n\n"
                f"Limits: max {self.max_positions} open, {self.max_trades_per_day} trades/day "
                f"({self.day_trades} used), ₹{self.max_value:,.0f} per trade, stop {settings.ai_agent_min_sl_pct}-"
                f"{settings.ai_agent_max_sl_pct}%. Decide, then call done(summary).")
        messages = [{"role": "user", "content": user}]
        summary, actions = "", []
        for _ in range(4):
            try:
                resp = await self._call(messages)
            except Exception as exc:
                self.llm["errors"] += 1
                summary = f"model error: {str(exc)[:160]}"
                logger.warning("[ai-replay] {} {}: {}", d, ts.time(), exc)
                break
            self._account(resp)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                summary = summary or "".join(getattr(b, "text", "") or "" for b in resp.content if b.get("type") == "text").strip()
                break
            results, finished = [], False
            for b in resp.content:
                if b.get("type") != "tool_use":
                    continue
                res = self._tool(b.name, dict(b.input or {}), ts)
                if b.name == "done":
                    summary = str((b.input or {}).get("summary", ""))[:300]; finished = True
                elif res.get("ok"):
                    actions.append({"tool": b.name, **{k: v for k, v in (b.input or {}).items() if k != "reasoning"}})
                else:
                    self.rejections.append({"ts": ts.isoformat(), "tool": b.name, "args": b.input, "error": res["error"]})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(res)})
            messages.append({"role": "user", "content": results})
            if finished:
                break
        self.decisions.append({"ts": ts.isoformat(), "summary": summary, "actions": actions,
                               "open": list(self.positions)})
        if self.progress:
            self.progress(self)

    async def run(self) -> dict:
        if not self.syms:
            await asyncio.to_thread(self.load)
        days = self.session_days()
        for d in days:
            self.day_trades = 0
            prev = time(9, 15)
            for cp in self.checkpoints:
                ts = datetime.combine(d, cp)
                self._walk(d, prev, cp)
                if any(s.at(ts) is not None for s in self.syms.values()):
                    await self._checkpoint(d, ts)
                prev = cp
            self._walk(d, prev, SQUARE_OFF)
            self._square_off(d)
        return self.result(days)

    # ── report ───────────────────────────────────────────────────────────
    def result(self, days: Optional[list[date]] = None) -> dict:
        nets = [t["net"] for t in self.trades]
        wins = [n for n in nets if n > 0]; losses = [n for n in nets if n <= 0]
        peak = run = 0.0; dd = 0.0
        for n in nets:
            run += n; peak = max(peak, run); dd = max(dd, peak - run)
        per_day: dict[str, float] = {}
        for t in self.trades:
            per_day[t["day"]] = round(per_day.get(t["day"], 0.0) + t["net"], 2)
        days = days or self.session_days() if self.syms else []
        for d in days:
            per_day.setdefault(d.isoformat(), 0.0)
        return {"status": "done", "finished_at": now_ist().isoformat(timespec="seconds"),
                "days": [d.isoformat() for d in days], "symbols": list(self.syms), "model": self.model_used,
                "capital": self.capital, "limits": {"max_positions": self.max_positions, "max_trades_per_day": self.max_trades_per_day,
                                                    "max_position_value": self.max_value, "risk_pct": self.risk_pct},
                "trades": self.trades, "decisions": self.decisions, "rejections": self.rejections, "llm": self.llm,
                "metrics": {"trades": len(nets), "wins": len(wins), "win_rate": round(len(wins) / len(nets) * 100, 1) if nets else 0.0,
                            "gross": round(sum(t["gross"] for t in self.trades), 2), "costs": round(sum(t["cost"] for t in self.trades), 2),
                            "net": round(sum(nets), 2), "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) else None,
                            "max_drawdown": round(dd, 2), "avg_trade": round(sum(nets) / len(nets), 2) if nets else 0.0,
                            "per_day": dict(sorted(per_day.items())), "return_pct": round(sum(nets) / self.capital * 100, 3),
                            "exits": {k: sum(1 for t in self.trades if t["exit_reason"] == k)
                                      for k in sorted({t["exit_reason"] for t in self.trades})}}}


# ── API-facing runner ────────────────────────────────────────────────────
def _save(state: dict) -> None:
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        _FILE.write_text(json.dumps(state, default=str))
    except Exception as exc:
        logger.debug("[ai-replay] not saved: {}", exc)


def last() -> dict:
    if _LAST.get("status") == "idle" and _FILE.exists():
        try:
            _LAST.update(json.loads(_FILE.read_text()))
        except Exception:
            pass
    return _LAST


async def run_and_record(days: int = 5, symbols: Optional[list[str]] = None, **kw) -> dict:
    """Run a replay and keep its state in _LAST for the API (status running → done/error)."""
    if _LAST.get("status") == "running":
        return _LAST
    rp = AIReplay(days=days, symbols=symbols, **kw)
    _LAST.clear()
    _LAST.update({"status": "running", "started_at": now_ist().isoformat(timespec="seconds"), "days": days,
                  "checkpoints_done": 0, "trades_so_far": 0, "cost_usd": 0.0})
    def prog(r: AIReplay):
        _LAST.update({"checkpoints_done": len(r.decisions), "trades_so_far": len(r.trades),
                      "cost_usd": round(r.llm["cost_usd"], 4), "model": r.model_used,
                      "last_decision": r.decisions[-1] if r.decisions else None})
    rp.progress = prog
    try:
        res = await rp.run()
        _LAST.clear(); _LAST.update(res)
    except Exception as exc:
        logger.exception("[ai-replay] failed")
        _LAST.clear(); _LAST.update({"status": "error", "error": str(exc)[:300], **rp.result()} if rp.syms
                                    else {"status": "error", "error": str(exc)[:300]})
    _save(_LAST)
    return _LAST
