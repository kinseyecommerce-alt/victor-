"""
performance.py — the trade ledger, per-strategy statistics and the daily scoreboard.

* Every closed trade is recorded once with its transaction cost (costs.py), so every number here is after
  brokerage and taxes. The validated agents book the lab's cost assumption (cost_kind) and their lab id.
* A daily snapshot stores the day's net P&L next to the NIFTY 50 close (Kite day candles), so the account can
  be compared with the index it could simply have held.
* No automatic pruning: the validated agents carry their own drawdown kill and the shadow registry's
  pre-registered demotion rule. The ledger file keeps the "pruned" / "enabled_since" keys of older builds
  unchanged (round-tripped, never interpreted).

State lives in PERFORMANCE_FILE (default data/performance.json; atomic writes).
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
BENCHMARKS = {"NIFTY50": "NIFTY 50"}    # Kite index name; day closes from Kite historical candles
STRATEGY_LABELS = {
    "intraday": "Intraday agent", "scalping": "Scalping agent", "swing": "Swing agent", "fno": "F&O agent",
    "ai": "AI intraday", "ai_opt": "AI option buy", "ai_fut": "AI future", "ai_swing": "AI swing",
    "ai_swing_opt": "AI swing put", "ai_spread": "AI credit spread", "momentum": "Momentum rotation",
    "auc_fade": "Auction fade (validated)",
    "attn_on": "Attention overnight (validated)",
    "vrp_winged": "NIFTY winged strangle (validated)",
    "vrp_naked": "NIFTY naked strangle (candidate)",
    "opt_ic2510": "NIFTY iron condor 0.25/0.10 (candidate)",
    "opt_wcfri": "NIFTY Friday condor (candidate)",
    "pead_h60": "Post-earnings drift 60d (candidate)",
}


def slippage_bps(signal_price: float, fill_price: float, side: str) -> Optional[float]:
    """How much worse (positive) or better the fill was than the price the rule saw, in basis points."""
    if not signal_price or not fill_price:
        return None
    adverse = (fill_price - signal_price) if side == "BUY" else (signal_price - fill_price)
    return round(adverse / signal_price * 10000, 1)


def _version(strategy: str) -> str:
    """The live agent's VERSION ("v1" for agents that never declared one)."""
    try:
        from agents.strategy_agents import ALL_AGENTS
        return str(getattr(ALL_AGENTS.get(strategy), "VERSION", "v1"))
    except Exception:
        return "v1"


class Performance:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.trades: list[dict] = []
        self.days: dict[str, dict] = {}            # date → snapshot
        self.pruned: dict[str, dict] = {}          # strategy → {since, reason}
        self.enabled_since: dict[str, str] = {}    # strategy → ISO time of the last manual re-enable
        self._extra: dict = {}                     # unknown top-level keys of the file, written back unchanged
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
        self._extra = {k: v for k, v in d.items() if k not in ("trades", "days", "pruned", "enabled_since")}

    def _save(self) -> None:
        try:
            FILE.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=FILE.parent, prefix=".perf.")
            with os.fdopen(fd, "w") as f:
                json.dump({**self._extra, "trades": self.trades[-MAX_TRADES:], "days": self.days,
                           "pruned": self.pruned, "enabled_since": self.enabled_since}, f)
            os.replace(tmp, FILE)
        except Exception as exc:
            logger.error("[perf] could not save: {}", exc)

    # ── recording ────────────────────────────────────────────────────────
    def record_trade(self, strategy: str, symbol: str, side: str, quantity: int, entry_price: float,
                     pnl: float, exchange: str = "NSE", product: str = "MIS",
                     exit_price: Optional[float] = None, note: str = "",
                     signal_price: Optional[float] = None, order_id: Optional[str] = None,
                     cost_kind: Optional[str] = None, charge_risk: bool = True,
                     extra: Optional[dict] = None) -> dict:
        """Called once per closed trade with its gross P&L. Returns the stored row (net of costs).
        order_id (the entry order) lets a restored paper book tell a trade that already closed from one
        still open (paper_book.py). cost_kind: cost the trade as this costs.py instrument class (the validated
        agents book the lab's cost assumption); charge_risk: accepted for compatibility (there is no risk manager
        to charge any more); extra: fields merged into the stored row."""
        if exit_price is None and quantity and entry_price:
            exit_price = entry_price + (pnl / quantity) * (1 if side == "BUY" else -1)
        # an option settled worthless exits at exactly ₹0 (not "unknown"): keep it, never fall back to the entry
        settled_zero = exit_price is not None and exit_price == 0 and exchange in ("NFO", "BFO")
        if cost_kind or settled_zero:
            from costs import round_trip_cost, instrument_class
            ev, xv = entry_price * quantity, (0.0 if settled_zero else (exit_price or entry_price)) * quantity
            buy_v, sell_v = (ev, xv) if side == "BUY" else (xv, ev)
            cost = round_trip_cost(buy_v, sell_v, cost_kind or instrument_class(exchange, product, symbol))
        else:
            cost = trade_cost(entry_price, exit_price or entry_price, quantity, side, exchange, product, symbol)
        row = {"t": now_ist().isoformat(timespec="seconds"), "strategy": strategy, "version": _version(strategy),
               "symbol": symbol,
               "side": side, "qty": quantity, "entry": round(entry_price, 2),
               "exit": round(0.0 if settled_zero else (exit_price or entry_price), 2), "gross": round(pnl, 2), "cost": cost,
               "net": round(pnl - cost, 2), "product": product, "exchange": exchange, "note": note[:80]}
        if order_id:
            row["oid"] = str(order_id)
        if signal_price:
            row["signal"] = round(signal_price, 2)
            row["slip_bps"] = slippage_bps(signal_price, entry_price, side)
        if cost_kind:
            row["cost_kind"] = cost_kind
        if extra:
            row.update({k: v for k, v in extra.items() if k not in row})
        with self._lock:
            self.trades.append(row)
            self.trades = self.trades[-MAX_TRADES:]
            self._save()
        return row

    def _strategy_trades(self, strategy: str, since: Optional[str] = None, all_versions: bool = False) -> list[dict]:
        """Trades of the strategy's CURRENT version (an agent rebuilt under the same name must not inherit its
        predecessor's record)."""
        v = _version(strategy)
        rows = [t for t in self.trades if t["strategy"] == strategy and (all_versions or t.get("version", "v1") == v)]
        if since:
            rows = [t for t in rows if t["t"] >= since]
        return rows

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
        """Per-strategy statistics of the kept strategies (and of any other name still in the ledger)."""
        import bot_state
        out = {}
        names = set(bot_state.VALIDATED_AGENTS) | {t["strategy"] for t in self.trades}
        for s in sorted(names):
            rows = self._strategy_trades(s)
            st = self._stats(rows)
            st.update(strategy=s, label=STRATEGY_LABELS.get(s, s), version=_version(s),
                      kept=s in bot_state.VALIDATED_AGENTS,
                      enabled=bot_state.is_agent_enabled(s) if s in bot_state.VALIDATED_AGENTS else False)
            out[s] = st
        return out

    # ── benchmarks ───────────────────────────────────────────────────────
    def _benchmark_closes(self, name: str) -> list[tuple[str, float]]:
        """Daily closes [(YYYY-MM-DD, close)] of a benchmark from Kite day candles (2 years), cached for the day.
        Empty without Kite market data / a session."""
        today = now_ist().date()
        if self._bench_day != today:
            self._bench_cache, self._bench_day = {}, today
        if name in self._bench_cache:
            return self._bench_cache[name]
        rows: list[tuple[str, float]] = []
        try:
            from kite_client import kite_client
            if settings.uses_kite_market_data and kite_client.has_session():
                tok = kite_client.get_instrument_tokens([BENCHMARKS[name]], "NSE").get(BENCHMARKS[name])
                if tok:
                    start = datetime.combine(today - timedelta(days=730), datetime.min.time())
                    for r in kite_client.historical_data(tok, start, datetime.combine(today, datetime.max.time()),
                                                         "day"):
                        d = r["date"]
                        rows.append(((d.date() if hasattr(d, "date") else d).isoformat()[:10], float(r["close"])))
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
        """Store today's net P&L with the benchmark closes. Idempotent."""
        today = now_ist().date().isoformat()
        snap = self._day_row(today, ai_cost_usd)
        # earlier days with trades but no snapshot (the 15:35 job did not run: restart, outage) — fill them in
        # from the ledger, or their P&L is missing from every week/month/year total
        missed = sorted({t["t"][:10] for t in self.trades} - set(self.days) - {today})
        if missed and len(self.trades) >= MAX_TRADES and missed[0] == self.trades[0]["t"][:10]:
            missed = missed[1:]              # the ledger was trimmed: its oldest day may be incomplete
        rows = {d: self._day_row(d, 0.0) for d in missed if d < today}
        with self._lock:
            self.days.update(rows)
            self.days[today] = snap
            self._save()
        return snap

    def _day_row(self, day: str, ai_cost_usd: float) -> dict:
        rows = [t for t in self.trades if t["t"][:10] == day]
        snap = {"date": day, "gross": round(sum(t["gross"] for t in rows), 2),
                "costs": round(sum(t["cost"] for t in rows), 2),
                "ai_cost_inr": 0.0, "trades": len(rows)}
        snap["net"] = round(snap["gross"] - snap["costs"] - snap["ai_cost_inr"], 2)
        for name in BENCHMARKS:
            closes = self._benchmark_closes(name)
            prior = [c for d, c in closes if d <= day]
            snap[name.lower()] = prior[-1] if prior else None
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
            today_net = round(live_day_pnl - sum(t["cost"] for t in today_rows), 2)
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
            out.append({"date": d, "cum_net": round(cum, 2), "nifty50": self.days[d].get("nifty50")})
        return out


performance = Performance()
