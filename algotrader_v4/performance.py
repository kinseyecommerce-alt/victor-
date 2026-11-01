"""
performance.py — the honest scoreboard, per-strategy statistics and automatic pruning.

* Every closed trade is recorded with an estimated transaction cost (costs.py), so the
  numbers here are after brokerage and taxes. The Claude API spend is charged too.
* A daily snapshot (after the close) stores the day's net P&L next to the NIFTY 50 and
  Nasdaq-100 closes, so returns can be compared over any period.
* The scoreboard answers one question: is this account beating the index it could have
  simply bought? NIFTY 50 is the real yardstick (it is what a Zerodha account could hold
  instead); Nasdaq-100 is shown for reference only.
* Pruning: once a strategy has PRUNE_MIN_TRADES closed trades, if its last
  PRUNE_LOOKBACK_TRADES are net losers (net P&L < 0 and profit factor < 1) it is disabled
  automatically and the owner is told. Re-enable it from the dashboard / API; the
  judgement then restarts from that point.

State lives in data/performance.json (atomic writes).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from costs import trade_cost
from ist_clock import now_ist

FILE = Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json")
MAX_TRADES = 5000
BENCHMARKS = {"NIFTY50": "^NSEI", "NASDAQ100": "^NDX"}
STRATEGY_LABELS = {
    "intraday": "Intraday agent", "scalping": "Scalping agent", "swing": "Swing agent", "fno": "F&O agent",
    "ai": "AI intraday", "ai_opt": "AI option buy", "ai_fut": "AI future", "ai_swing": "AI swing",
    "ai_swing_opt": "AI swing put", "ai_spread": "AI credit spread", "momentum": "Momentum rotation",
}


class Performance:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.trades: list[dict] = []
        self.days: dict[str, dict] = {}            # date → snapshot
        self.pruned: dict[str, dict] = {}          # strategy → {since, reason}
        self.enabled_since: dict[str, str] = {}    # strategy → ISO time of the last manual re-enable
        self._bench_cache: dict[str, list[tuple[str, float]]] = {}
        self._bench_day: Optional[date] = None
        self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            d = json.loads(FILE.read_text())
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.error("[perf] state unreadable: {}", exc)
            return
        self.trades = d.get("trades", [])[-MAX_TRADES:]
        self.days = d.get("days", {})
        self.pruned = d.get("pruned", {})
        self.enabled_since = d.get("enabled_since", {})

    def _save(self) -> None:
        try:
            FILE.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=FILE.parent, prefix=".perf.")
            with os.fdopen(fd, "w") as f:
                json.dump({"trades": self.trades[-MAX_TRADES:], "days": self.days, "pruned": self.pruned,
                           "enabled_since": self.enabled_since}, f)
            os.replace(tmp, FILE)
        except Exception as exc:
            logger.error("[perf] could not save: {}", exc)

    # ── recording ────────────────────────────────────────────────────────
    def record_trade(self, strategy: str, symbol: str, side: str, quantity: int, entry_price: float,
                     pnl: float, exchange: str = "NSE", product: str = "MIS",
                     exit_price: Optional[float] = None, note: str = "",
                     signal_price: Optional[float] = None) -> dict:
        """Called once per closed trade with its gross P&L. Returns the stored row (net of costs)."""
        if exit_price is None and quantity and entry_price:
            exit_price = entry_price + (pnl / quantity) * (1 if side == "BUY" else -1)
        cost = trade_cost(entry_price, exit_price or entry_price, quantity, side, exchange, product, symbol)
        row = {"t": now_ist().isoformat(timespec="seconds"), "strategy": strategy, "symbol": symbol,
               "side": side, "qty": quantity, "entry": round(entry_price, 2),
               "exit": round(exit_price or entry_price, 2), "gross": round(pnl, 2), "cost": cost,
               "net": round(pnl - cost, 2), "product": product, "exchange": exchange, "note": note[:80]}
        if signal_price:
            from execution import slippage_bps
            row["signal"] = round(signal_price, 2)
            row["slip_bps"] = slippage_bps(signal_price, entry_price, side)
        with self._lock:
            self.trades.append(row)
            self.trades = self.trades[-MAX_TRADES:]
            newly = self._evaluate_pruning(strategy)
            self._save()
        if newly:
            self._announce_prune(strategy, newly)
        return row

    # ── pruning ──────────────────────────────────────────────────────────
    def _strategy_trades(self, strategy: str, since: Optional[str] = None) -> list[dict]:
        rows = [t for t in self.trades if t["strategy"] == strategy]
        since = since or self.enabled_since.get(strategy)
        if since:
            rows = [t for t in rows if t["t"] >= since]
        return rows

    def _evaluate_pruning(self, strategy: str) -> Optional[str]:
        if not settings.prune_enabled or strategy in self.pruned:
            return None
        rows = self._strategy_trades(strategy)[-settings.prune_lookback_trades:]
        if len(rows) < settings.prune_min_trades:
            return None
        net = sum(t["net"] for t in rows)
        gp = sum(t["net"] for t in rows if t["net"] > 0)
        gl = -sum(t["net"] for t in rows if t["net"] <= 0)
        pf = gp / gl if gl > 0 else float("inf")
        if net < 0 and pf < 1.0:
            reason = (f"last {len(rows)} trades: net ₹{net:,.0f} after costs, profit factor {pf:.2f}")
            self.pruned[strategy] = {"since": now_ist().isoformat(timespec="seconds"), "reason": reason}
            self._apply_toggle(strategy, False)
            return reason
        return None

    def _apply_toggle(self, strategy: str, enabled: bool) -> None:
        try:
            import bot_state
            if strategy in ("intraday", "scalping", "swing", "fno"):
                bot_state.set_agent_enabled(strategy, enabled)
        except Exception:
            pass

    def _announce_prune(self, strategy: str, reason: str) -> None:
        msg = (f"⛔ Strategy {STRATEGY_LABELS.get(strategy, strategy)} disabled automatically — {reason}. "
               "Re-enable it from the dashboard if you believe conditions changed.")
        logger.warning("[perf] {}", msg)
        try:
            import asyncio
            from agents.base_agent import send_telegram
            asyncio.get_running_loop().create_task(send_telegram(msg))
        except Exception:
            pass

    def is_allowed(self, strategy: str) -> tuple[bool, str]:
        p = self.pruned.get(strategy)
        if p:
            return False, f"{STRATEGY_LABELS.get(strategy, strategy)} was disabled automatically ({p['reason']})"
        return True, ""

    def set_enabled(self, strategy: str, enabled: bool, reason: str = "manual") -> dict:
        with self._lock:
            if enabled:
                self.pruned.pop(strategy, None)
                self.enabled_since[strategy] = now_ist().isoformat()     # full precision: trades of the same second are excluded
            else:
                self.pruned[strategy] = {"since": now_ist().isoformat(timespec="seconds"), "reason": reason}
            self._apply_toggle(strategy, enabled)
            self._save()
        return self.strategy_stats().get(strategy, {"strategy": strategy, "enabled": enabled})

    # ── statistics ───────────────────────────────────────────────────────
    @staticmethod
    def _stats(rows: list[dict]) -> dict:
        nets = [t["net"] for t in rows]
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n <= 0]
        gp, gl = sum(wins), -sum(losses)
        peak = cum = dd = 0.0
        for n in nets:
            cum += n
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        return {"trades": len(rows), "wins": len(wins), "losses": len(losses),
                "win_rate": round(len(wins) / len(rows) * 100, 1) if rows else None,
                "gross_pnl": round(sum(t["gross"] for t in rows), 2), "costs": round(sum(t["cost"] for t in rows), 2),
                "net_pnl": round(sum(nets), 2),
                "avg_net_per_trade": round(sum(nets) / len(rows), 2) if rows else None,
                "profit_factor": round(gp / gl, 2) if gl > 0 else (None if not wins else 99.0),
                "max_drawdown": round(dd, 2)}

    def execution_stats(self) -> dict:
        """Entry slippage (fill vs signal price) per strategy, in basis points. Positive = paid more."""
        by: dict[str, list[float]] = defaultdict(list)
        for t in self.trades:
            if t.get("slip_bps") is not None:
                by[t["strategy"]].append(t["slip_bps"])
        out = {}
        for s, v in by.items():
            v_sorted = sorted(v)
            out[s] = {"trades": len(v), "avg_bps": round(sum(v) / len(v), 1), "median_bps": v_sorted[len(v) // 2],
                      "worst_bps": max(v), "best_bps": min(v),
                      "cost_inr_per_lakh": round(sum(v) / len(v) / 10000 * 100000, 0)}
        allv = [x for v in by.values() for x in v]
        return {"per_strategy": out,
                "overall": {"trades": len(allv), "avg_bps": round(sum(allv) / len(allv), 1) if allv else None},
                "note": "entry fills only; exits are at market/marketable limit and not yet measured"}

    def strategy_stats(self) -> dict[str, dict]:
        import bot_state
        out = {}
        names = set(STRATEGY_LABELS) | {t["strategy"] for t in self.trades}
        for s in sorted(names):
            rows = self._strategy_trades(s)
            st = self._stats(rows)
            st.update(strategy=s, label=STRATEGY_LABELS.get(s, s),
                      recent=self._stats(rows[-settings.prune_lookback_trades:]) if rows else None,
                      pruned=self.pruned.get(s), enabled=s not in self.pruned and
                      (bot_state.is_agent_enabled(s) if s in ("intraday", "scalping", "swing", "fno") else True),
                      trades_until_judgement=max(0, settings.prune_min_trades - len(rows[-settings.prune_lookback_trades:])))
            out[s] = st
        return out

    # ── benchmarks ───────────────────────────────────────────────────────
    def _benchmark_closes(self, name: str) -> list[tuple[str, float]]:
        """Daily closes [(YYYY-MM-DD, close)] for a benchmark, cached for the day.
        NIFTY50 in the paper simulator = the streamed synthetic index."""
        today = now_ist().date()
        if self._bench_day != today:
            self._bench_cache, self._bench_day = {}, today
        if name in self._bench_cache:
            return self._bench_cache[name]
        rows: list[tuple[str, float]] = []
        try:
            if name == "NIFTY50" and not settings.uses_kite_market_data:
                from daily_data import daily_closes
                rows = [(d.isoformat(), c) for d, c in daily_closes("NIFTY 50", fallback_price=24000.0)]
            else:
                import yfinance as yf
                df = yf.download(BENCHMARKS[name], period="2y", interval="1d", progress=False, auto_adjust=True)
                if not df.empty:
                    close = df["Close"]
                    if hasattr(close, "columns"):
                        close = close.iloc[:, 0]
                    rows = [(d.strftime("%Y-%m-%d"), float(c)) for d, c in close.dropna().items()]
        except Exception as exc:
            logger.warning("[perf] benchmark {} unavailable: {}", name, exc)
        self._bench_cache[name] = rows
        return rows

    def benchmark_return(self, name: str, start: str, end: str) -> Optional[float]:
        """% return of a benchmark from the last close on/before `start` to the last close on/before `end`."""
        rows = self._benchmark_closes(name)
        if not rows:
            return None
        def close_at(day: str) -> Optional[float]:
            prior = [c for d, c in rows if d <= day]
            return prior[-1] if prior else None
        a, b = close_at(start), close_at(end)
        return round((b / a - 1) * 100, 2) if a and b else None

    # ── daily snapshot ───────────────────────────────────────────────────
    def snapshot_today(self, ai_cost_usd: float = 0.0) -> dict:
        """Store today's net P&L (and the API spend) with the benchmark closes. Idempotent."""
        today = now_ist().date().isoformat()
        rows = [t for t in self.trades if t["t"][:10] == today]
        snap = {"date": today, "gross": round(sum(t["gross"] for t in rows), 2),
                "costs": round(sum(t["cost"] for t in rows), 2),
                "ai_cost_inr": round(ai_cost_usd * settings.usd_inr, 2), "trades": len(rows)}
        snap["net"] = round(snap["gross"] - snap["costs"] - snap["ai_cost_inr"], 2)
        for name in BENCHMARKS:
            closes = self._benchmark_closes(name)
            prior = [c for d, c in closes if d <= today]
            snap[name.lower()] = prior[-1] if prior else None
        with self._lock:
            self.days[today] = snap
            self._save()
        return snap

    # ── scoreboard ───────────────────────────────────────────────────────
    def scoreboard(self, live_day_pnl: Optional[float] = None, ai_cost_today_usd: float = 0.0) -> dict:
        today = now_ist().date()
        cap = settings.total_capital or 1.0
        days = sorted(self.days)
        first = days[0] if days else today.isoformat()
        periods = {
            "today": today.isoformat(),
            "week":  (today - timedelta(days=today.weekday())).isoformat(),
            "month": today.replace(day=1).isoformat(),
            "year":  today.replace(month=1, day=1).isoformat(),
            "since_start": first,
        }
        today_net = None
        if live_day_pnl is not None:
            today_rows = [t for t in self.trades if t["t"][:10] == today.isoformat()]
            today_net = round(live_day_pnl - sum(t["cost"] for t in today_rows)
                              - ai_cost_today_usd * settings.usd_inr, 2)
        out = {"capital": cap, "as_of": now_ist().strftime("%Y-%m-%d %H:%M"), "periods": {}}
        for label, start in periods.items():
            if label == "today":
                net = today_net if today_net is not None else self.days.get(start, {}).get("net", 0.0)
            else:
                net = sum(v["net"] for d, v in self.days.items() if start <= d < today.isoformat())
                net += today_net if today_net is not None else self.days.get(today.isoformat(), {}).get("net", 0.0)
            prev_day = (date.fromisoformat(start) - timedelta(days=1)).isoformat()
            acct = round(net / cap * 100, 3)
            row = {"net_pnl": round(net, 2), "account_pct": acct}
            for name in BENCHMARKS:
                b = self.benchmark_return(name, prev_day, today.isoformat())
                if label == "today" and name == "NIFTY50":
                    b = self._nifty_today_pct() if self._nifty_today_pct() is not None else b
                row[name.lower() + "_pct"] = b
                row["vs_" + name.lower()] = round(acct - b, 3) if b is not None else None
            out["periods"][label] = row
        out["beating_nifty_since_start"] = out["periods"]["since_start"]["vs_nifty50"]
        out["days_tracked"] = len(days)
        out["verdict"] = self._verdict(out)
        return out

    def _nifty_today_pct(self) -> Optional[float]:
        try:
            from tick_engine import tick_engine
            tick, _ = tick_engine.latest("NIFTY 50")
            if tick and tick.ltp > 0 and tick.change_pct:
                return round(float(tick.change_pct), 2)
        except Exception:
            pass
        return None

    @staticmethod
    def _verdict(sb: dict) -> str:
        p = sb["periods"]["since_start"]
        if sb["days_tracked"] < 10:
            return f"Too early to judge ({sb['days_tracked']} trading days tracked; wait for 20+)."
        if p["vs_nifty50"] is None:
            return "NIFTY benchmark unavailable — cannot compare."
        if p["vs_nifty50"] > 0:
            return (f"Ahead of NIFTY 50 by {p['vs_nifty50']:+.2f} pts since tracking began "
                    f"({p['account_pct']:+.2f}% vs {p['nifty50_pct']:+.2f}%), after costs.")
        return (f"Behind NIFTY 50 by {abs(p['vs_nifty50']):.2f} pts since tracking began "
                f"({p['account_pct']:+.2f}% vs {p['nifty50_pct']:+.2f}%). An index fund would have done better.")

    def equity_curve(self) -> list[dict]:
        cum = 0.0
        out = []
        for d in sorted(self.days):
            cum += self.days[d]["net"]
            out.append({"date": d, "cum_net": round(cum, 2), "nifty50": self.days[d].get("nifty50"),
                        "nasdaq100": self.days[d].get("nasdaq100")})
        return out


performance = Performance()
