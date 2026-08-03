"""
positional_runner.py
Live wiring for the positional trend-following system (strategies/).

Two scheduled jobs (see platform_scheduler.py, enabled by
POSITIONAL_ENABLED=true):

  EOD (23:45 IST, after the MCX session close)
    kill-switch check → fetch continuous daily bars per universe root →
    bar sanity checks → PositionalEngine.run_eod() → supersede stale
    pending plans → queue fresh plans (idempotent) → persist state.

  Morning (09:16 IST, after both MCX and NSE are open)
    kill-switch check → start-of-day reconciliation (broker vs internal;
    mismatch = halt + alert, per spec) → contract rollover check
    (≤3 days to expiry → square-off near month + re-enter next month,
    move GTT stop) → place pending plans as NRML market orders → place
    GTT stops for entries / delete GTTs on exits.

State lives in SQLite (logs/positional_state.db) so restarts recover
cleanly. All order placement is idempotent via the pending_plans table.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist
from kite_client import kite_client
from strategies import DailyBar, PositionalEngine, get_contract
from strategies.positional_engine import OrderPlan
from strategies.state_store import PositionalStateStore

LOOKBACK_DAYS   = 400        # enough history for 252-day TSMOM + warm-up
OUTLIER_RET     = 0.25       # |1-day return| above this = bad data, skip symbol


# ── Pure helpers (unit-tested in test_strategies.py) ───────────────────────

def pick_near_month(instruments: list[dict], root: str, today: date,
                    buffer_days: int = 3) -> Optional[dict]:
    """Nearest-expiry FUT contract for `root` expiring more than
    `buffer_days` away — the auto-rollover rule: within the buffer we are
    already trading the next month."""
    def _expiry(inst) -> Optional[date]:
        e = inst.get("expiry")
        if isinstance(e, datetime):
            return e.date()
        if isinstance(e, date):
            return e
        try:
            return datetime.strptime(str(e)[:10], "%Y-%m-%d").date()
        except Exception:
            return None

    cutoff = today + timedelta(days=buffer_days)
    candidates = []
    for inst in instruments:
        if inst.get("name") != root or inst.get("instrument_type") != "FUT":
            continue
        exp = _expiry(inst)
        if exp and exp > cutoff:
            candidates.append((exp, inst))
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[0])[1]


def reconcile(broker_net: dict[str, int], expected_net: dict[str, int],
              universe_roots: list[str]) -> tuple[bool, list[str]]:
    """Compare broker net quantities (by tradingsymbol, which carries the
    expiry suffix) against the engine's expected net per root symbol.
    Only symbols in the positional universe are considered — the intraday
    agents' positions are out of scope."""
    roots = sorted(universe_roots, key=len, reverse=True)

    def _root_of(tradingsymbol: str) -> Optional[str]:
        for r in roots:
            if tradingsymbol.startswith(r):
                return r
        return None

    broker_by_root: dict[str, int] = {}
    for tsym, qty in broker_net.items():
        r = _root_of(tsym)
        if r and qty != 0:
            broker_by_root[r] = broker_by_root.get(r, 0) + qty

    mismatches = []
    for r in set(broker_by_root) | set(expected_net):
        b, e = broker_by_root.get(r, 0), expected_net.get(r, 0)
        if b != e:
            mismatches.append(f"{r}: broker={b} internal={e}")
    return not mismatches, mismatches


def sane_bars(records: list[dict]) -> Optional[list[DailyBar]]:
    """Convert Kite historical records to DailyBars with the spec's EOD
    sanity checks: zero/NaN/outlier rejection → None means skip + alert."""
    bars: list[DailyBar] = []
    prev_close = None
    for r in records:
        o, h, l, c = (r.get("open"), r.get("high"), r.get("low"), r.get("close"))
        if not all(isinstance(v, (int, float)) and v == v and v > 0
                   for v in (o, h, l, c)):
            return None
        if prev_close and abs(c / prev_close - 1.0) > OUTLIER_RET:
            return None
        prev_close = c
        d = r.get("date")
        d = d.date() if hasattr(d, "date") else d
        bars.append(DailyBar(date=d, open=float(o), high=float(h),
                             low=float(l), close=float(c),
                             volume=float(r.get("volume") or 0)))
    return bars if bars else None


# ── Runner ─────────────────────────────────────────────────────────────────

class PositionalRunner:

    def __init__(self) -> None:
        self._engine: Optional[PositionalEngine] = None
        self._store:  Optional[PositionalStateStore] = None
        self._halted_reason: str = ""
        self._last_eod:     str = ""
        self._last_morning: str = ""

    # Lazy init so importing this module costs nothing when disabled
    def _ensure(self) -> None:
        if self._engine is not None:
            return
        self._store = PositionalStateStore()
        equity = settings.total_capital * settings.futures_capital_pct / 100
        self._engine = PositionalEngine(
            equity=equity, risk_fraction=settings.positional_risk_fraction)
        saved = self._store.load_engine_state()
        if saved:
            self._engine.load_state(saved)
            logger.info("[positional] state restored ({} open books)",
                        len(self._engine.book.positions))

    @property
    def universe(self) -> list[str]:
        return [s.strip().upper() for s in
                settings.positional_universe.split(",") if s.strip()]

    def _kill_switch_ok(self) -> bool:
        from sebi_compliance import sebi_compliance, KillSwitchState
        state = sebi_compliance.status().get("state", "ACTIVE")
        return state == KillSwitchState.ACTIVE.value

    # ── EOD job ────────────────────────────────────────────────────────────

    async def eod_job(self) -> dict:
        from agents.base_agent import send_telegram
        self._ensure()
        today = now_ist().date().isoformat()
        if not self._kill_switch_ok():
            logger.warning("[positional] EOD skipped — kill switch not ACTIVE")
            return {"status": "skipped", "reason": "kill_switch"}

        bars_by_symbol: dict[str, list[DailyBar]] = {}
        skipped: list[str] = []
        for root in self.universe:
            bars = self._fetch_daily_bars(root)
            if bars is None:
                skipped.append(root)
                continue
            bars_by_symbol[root] = bars

        if skipped:
            await send_telegram(
                f"⚠️ <b>[POSITIONAL]</b> EOD data skipped (bad/missing bars): "
                f"{', '.join(skipped)}")
        if not bars_by_symbol:
            self._last_eod = today
            return {"status": "no_data", "skipped": skipped}

        plans = self._engine.run_eod(bars_by_symbol)
        self._store.cancel_stale_pending(today)      # supersede unexecuted plans
        queued = self._store.queue_plans(today, plans)
        self._store.save_engine_state(self._engine.to_state())
        self._last_eod = today

        if plans:
            lines = [f"{p.action} {p.symbol} ×{p.lots} lot "
                     f"[{p.strategy}/{p.system}] {p.reason}" for p in plans]
            await send_telegram(
                "<b>[POSITIONAL]</b> EOD signals (execute next open):\n"
                + "\n".join(lines))
        logger.info("[positional] EOD run: {} signals, {} queued, {} skipped",
                    len(plans), queued, len(skipped))
        return {"status": "ok", "signals": len(plans),
                "queued": queued, "skipped": skipped}

    def _fetch_daily_bars(self, root: str) -> Optional[list[DailyBar]]:
        contract = get_contract(root)
        if settings.trading_mode != "LIVE":
            logger.info("[positional] PAPER mode — no Kite historical for {}", root)
            return None
        try:
            instruments = kite_client.get_instruments(contract.exchange)
            inst = pick_near_month(instruments, root, now_ist().date(),
                                   settings.positional_rollover_days)
            if not inst:
                return None
            records = kite_client.historical_data(
                instrument_token=inst["instrument_token"],
                from_date=datetime.now() - timedelta(days=LOOKBACK_DAYS),
                to_date=datetime.now(),
                interval="day",
                continuous=True,       # continuous futures across expiries
            )
            return sane_bars(records)
        except Exception as exc:
            logger.warning("[positional] bar fetch failed for {}: {}", root, exc)
            return None

    # ── Morning job ────────────────────────────────────────────────────────

    async def morning_job(self) -> dict:
        from agents.base_agent import send_telegram
        self._ensure()
        today = now_ist().date().isoformat()
        if not self._kill_switch_ok():
            return {"status": "skipped", "reason": "kill_switch"}

        # 1. Start-of-day reconciliation — mismatch halts placement (spec)
        broker_net = {
            p["tradingsymbol"]: p.get("quantity", 0)
            for p in kite_client.positions().get("net", [])
            if p.get("product") == "NRML"
        }
        ok, mismatches = reconcile(broker_net, self._engine.net_quantities(),
                                   self.universe)
        if not ok:
            self._halted_reason = f"reconcile mismatch: {'; '.join(mismatches)}"
            await send_telegram(
                "🔴 <b>[POSITIONAL] RECONCILE MISMATCH — trading halted, "
                "manual review needed</b>\n" + "\n".join(mismatches))
            logger.error("[positional] {}", self._halted_reason)
            return {"status": "halted", "mismatches": mismatches}
        self._halted_reason = ""

        # 2. Order placement (idempotent via pending_plans)
        placed, failed = 0, 0
        for plan_id, plan in self._store.pending_plans():
            try:
                self._place_plan(plan_id, plan)
                placed += 1
            except Exception as exc:
                failed += 1
                self._store.mark_plan(plan_id, "failed")
                logger.error("[positional] plan {} failed: {}", plan_id, exc)
        self._last_morning = today

        if placed or failed:
            await send_telegram(
                f"<b>[POSITIONAL]</b> Morning execution: "
                f"{placed} placed, {failed} failed")
        return {"status": "ok", "placed": placed, "failed": failed}

    def _place_plan(self, plan_id: int, plan: OrderPlan) -> None:
        contract = get_contract(plan.symbol)
        inst = None
        if settings.trading_mode == "LIVE":
            instruments = kite_client.get_instruments(contract.exchange)
            inst = pick_near_month(instruments, plan.symbol, now_ist().date(),
                                   settings.positional_rollover_days)
            if not inst:
                raise RuntimeError(f"no tradable contract for {plan.symbol}")
        tradingsymbol = inst["tradingsymbol"] if inst else plan.symbol

        from sebi_compliance import sebi_compliance
        sebi_ok, algo_id, reason = sebi_compliance.pre_order_check(
            strategy=f"positional_{plan.strategy}", symbol=tradingsymbol,
            exchange=plan.exchange, transaction_type=plan.action,
            quantity=plan.quantity, order_type="MARKET",
            price_at_signal=0.0, signal_source="positional_eod",
            regime="positional")
        if not sebi_ok:
            raise RuntimeError(f"SEBI blocked: {reason}")

        order_id = kite_client.place_order(
            tradingsymbol=tradingsymbol, exchange=plan.exchange,
            transaction_type=plan.action, quantity=plan.quantity,
            order_type="MARKET", product=plan.product,
            tag=plan.tag[:20])
        sebi_compliance.record_order_id(f"positional_{plan.strategy}",
                                        tradingsymbol, order_id)

        gtt_id = ""
        if plan.stop and plan.action in ("BUY", "SELL"):
            if plan.tag.endswith("-exit"):
                pass                       # exits carry no stop
            else:
                # Protective GTT stop for the new position
                stop_side = "SELL" if plan.action == "BUY" else "BUY"
                last_price = self._last_price(plan.exchange, tradingsymbol)
                gtt_id = kite_client.place_gtt_stop(
                    tradingsymbol=tradingsymbol, exchange=plan.exchange,
                    transaction_type=stop_side, quantity=plan.quantity,
                    trigger_price=round(plan.stop, 2),
                    last_price=last_price or plan.stop * 1.05,
                    product=plan.product)
        self._store.mark_plan(plan_id, "placed", order_id, gtt_id)

    def _last_price(self, exchange: str, tradingsymbol: str) -> float:
        try:
            q = kite_client.quote_kite([f"{exchange}:{tradingsymbol}"])
            return float(q.get(f"{exchange}:{tradingsymbol}", {})
                          .get("last_price", 0.0))
        except Exception:
            return 0.0

    # ── Status (REST) ──────────────────────────────────────────────────────

    def status(self) -> dict:
        self._ensure()
        return {
            "enabled":        settings.positional_enabled,
            "equity":         self._engine.equity,
            "universe":       self.universe,
            "open_positions": {
                s.name: {sym: {"side": p.side, "units": p.units,
                               "stop": p.stop, "system": p.system}
                         for sym, p in getattr(s, "_positions", {}).items()}
                for s in self._engine.strategies
            },
            "portfolio_heat": round(self._engine.book.heat(), 4),
            "expected_net":   self._engine.net_quantities(),
            "pending_plans":  len(self._store.pending_plans()),
            "halted_reason":  self._halted_reason,
            "last_eod":       self._last_eod,
            "last_morning":   self._last_morning,
        }


positional_runner = PositionalRunner()
