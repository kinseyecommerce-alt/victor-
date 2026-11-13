"""
agents/validated_base.py — base for the validated (lab-protocol) agents: PAPER only, scheduled, exact lab rules.

A validated agent trades a rule that passed stratlab's pre-registered protocol (DISCOVERY H1/H2 screen + one FINAL
look) and nothing else: its rule functions and constants are imported from the lab module (never copied, pinned by
LAB_FINGERPRINT), it is driven by platform_scheduler jobs at the tested times (never by the tick engine), and it
bypasses every guard that would resize, delay or skip a tested trade (order_guard, risk_manager counters, allocator,
Claude gate, regime pauses, pruning). Only the SEBI kill switch / pause applies. Its own overlays (settings) sit
outside the lab's observed extremes:

  * kill switch      settings.validated_<x>_enabled / the dashboard toggle (bot_state) block ENTRIES only
  * daily loss stop  realised net today <= -limit → no new entry batch today (exits always run)
  * drawdown kill    live net drawdown of the current VERSION >= limit → disabled (bot_state) until a manual
                     re-enable, which restarts the drawdown measurement

Every early return in an entry path goes through _note(): BaseAgent._skip (log + count) and the per-day decision
log (day_log, exposed in get_status). P&L goes to the performance ledger (charge_risk=False: it does not feed
risk_manager's account-level daily loss), the positions to paper_book via export_book / import_book.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from agents.base_agent import BaseAgent
from config import settings
from ist_clock import now_ist

DAY_LOG_MAX = 500


@dataclass
class ValidatedPosition:
    oid: str                 # entry order id (ledger order_id → paper_book reconciliation)
    agent: str
    symbol: str
    exchange: str
    product: str             # "MIS" | "CNC" | "NRML"
    side: str                # "BUY" | "SELL" (opening side)
    qty: int
    entry_px: float          # paper fill
    entry_ref: float         # the price the rule saw
    opened: str              # ISO IST datetime
    overnight: bool = False  # True: attention CNC, strangle legs
    group: str = ""          # strangle structure id
    meta: dict = field(default_factory=dict)

    @property
    def sign(self) -> int:
        return 1 if self.side == "BUY" else -1


@dataclass(frozen=True)
class PlannedTrade:
    symbol: str
    side: int                # +1 long, -1 short
    qty: int
    entry_ref: float
    entry_t: str
    exit_t: str
    key: float               # ranking key (|gap| / ratio)


def _kite():
    from kite_client import kite_client
    return kite_client


def paper_row(symbol: str, product: str) -> Optional[dict]:
    """The paper broker's net row of (symbol, product), or None."""
    return next((p for p in _kite()._paper_positions if p.get("tradingsymbol") == symbol
                 and (p.get("product") or "MIS") == product), None)


def paper_marketable(symbol: str, exchange: str, side: str, qty: int, product: str, quote: dict,
                     tag: str) -> tuple[str, float]:
    """A marketable LIMIT at the quote touch (BUY → ask, SELL → bid; the LTP when the touch is 0). With no tick
    for the symbol the paper broker fills a LIMIT at its price, i.e. at the quote touch. PAPER only."""
    if settings.trading_mode != "PAPER":
        raise RuntimeError("validated agents place PAPER orders only")
    q = quote or {}
    px = float((q.get("ask") if side == "BUY" else q.get("bid")) or 0.0) or float(q.get("ltp") or 0.0)
    if px <= 0:
        raise ValueError(f"no usable quote for {symbol}")
    kc = _kite()
    oid = kc.place_order(tradingsymbol=symbol, exchange=exchange, transaction_type=side, quantity=int(qty),
                         order_type="LIMIT", product=product, price=round(px, 2), tag=tag)
    return oid, float(kc.fill_price(oid) or px)


class ValidatedAgent(BaseAgent):
    scheduled = True              # driven by platform_scheduler jobs, never by tick_engine
    validated_by_replay = True
    strategy_rule_exits = False
    COST_KIND: Optional[str] = None
    ENABLE_SETTING: str = ""      # settings attribute of the kill switch
    LAB_EXPECTATION: dict = {}

    VERSION: str = "validated/v1"
    LAB_ID: str = ""
    LAB_FINGERPRINT: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._pos: dict[str, ValidatedPosition] = {}
        self.day_log: list[dict] = []
        self._day: dict = {}
        self._kill: Optional[dict] = None          # {"at", "dd", "version"} once the drawdown kill fired
        self._dd_since: Optional[str] = None       # drawdown measured from here after a manual re-enable
        self.deviations: dict[str, int] = {}

    # ── clock (tests patch the agent module's now_ist) ─────────────────────
    def _now(self) -> datetime:
        fn = getattr(sys.modules.get(type(self).__module__), "now_ist", None) or now_ist
        return fn()

    def _today(self):
        return self._now().date()

    # ── BaseAgent interface: no tick trading ────────────────────────────────
    def evaluate_tick(self, snap):
        return "HOLD", None

    def should_exit_position(self, position, ind):
        return False, ""

    def start(self, queue=None) -> None:
        self.state.running = True
        logger.info("[{}] ready (scheduled, {})", self.name, self.VERSION)

    def stop(self) -> None:
        self.state.running = False
        logger.info("[{}] stopped: no new entries (open positions still exit by their rule)", self.name)

    def filter_watchlist(self, watchlist):
        return []                 # trades its own lab universe, never the tick watchlist

    # ── decision log ─────────────────────────────────────────────────────────
    def day_state(self) -> dict:
        today = self._today().isoformat()
        if self._day.get("date") != today:
            self._day = {"date": today}
            self.day_log = []
        return self._day

    def _note(self, stage: str, symbol: str, outcome: str, reason: str, log: bool = True) -> None:
        """outcome "skip": counted (and logged unless log=False) through BaseAgent._skip; always in day_log."""
        self.day_state()
        if outcome == "skip":
            if log:
                self._skip(symbol, reason)
            else:
                key = reason.split(":")[0][:40]
                self.state.skip_counts[key] = self.state.skip_counts.get(key, 0) + 1
        elif log:
            logger.info("[{}] {} {}: {} — {}", self.name, stage, symbol, outcome, reason)
        self.day_log.append({"t": self._now().strftime("%H:%M:%S"), "stage": stage, "symbol": symbol,
                             "outcome": outcome, "reason": reason})
        del self.day_log[:-DAY_LOG_MAX]

    def _deviation(self, kind: str, symbol: str, detail: str = "") -> None:
        self.deviations[kind] = self.deviations.get(kind, 0) + 1
        self._note("deviation", symbol, "deviation", f"{kind}{': ' + detail if detail else ''}")

    # ── ledger-derived stats ────────────────────────────────────────────────
    def ledger_rows(self) -> list[dict]:
        from performance import performance
        v = self.VERSION
        return [t for t in performance.trades if t.get("strategy") == self.name and t.get("version") == v]

    def realised_today(self) -> float:
        today = self._today().isoformat()
        return round(sum(t["net"] for t in self.ledger_rows() if str(t.get("t", ""))[:10] == today), 2)

    def drawdown(self) -> float:
        rows = sorted(self.ledger_rows(), key=lambda t: t.get("t", ""))
        if self._dd_since:
            rows = [t for t in rows if t.get("t", "") >= self._dd_since]
        cum = peak = 0.0
        for t in rows:
            cum += t["net"]
            peak = max(peak, cum)
        return round(peak - cum, 2)          # the current drawdown from the best cumulative net

    def daily_loss_limit(self) -> float:
        return float("inf")

    def dd_limit(self) -> float:
        return float("inf")

    def enabled_by_setting(self) -> bool:
        return bool(getattr(settings, self.ENABLE_SETTING, True)) if self.ENABLE_SETTING else True

    def entry_block_reason(self) -> Optional[str]:
        import bot_state
        if settings.trading_mode != "PAPER":
            return "live_mode_refused"
        if not settings.uses_kite_market_data:
            return "no_kite_data"
        if not getattr(settings, "validated_only", True):
            return "validated_only_off"
        if not self.enabled_by_setting():
            return "kill_switch"
        enabled = bot_state.is_agent_enabled(self.name)
        if self._kill:
            if not enabled:
                return "drawdown_kill"
            logger.warning("[{}] re-enabled after the drawdown kill — drawdown measured afresh", self.name)
            self._kill = None
            self._dd_since = self._now().isoformat(timespec="seconds")
        if not enabled:
            return "disabled"
        if not self.state.running:
            return "stopped"
        dd = self.drawdown()
        if dd >= self.dd_limit():
            self._kill = {"at": self._now().isoformat(timespec="seconds"), "dd": dd, "version": self.VERSION}
            bot_state.set_agent_enabled(self.name, False)
            logger.error("[{}] DRAWDOWN KILL: live drawdown ₹{:,.0f} ≥ ₹{:,.0f} — disabled until re-enabled",
                         self.name, dd, self.dd_limit())
            return "drawdown_kill"
        if self.realised_today() <= -self.daily_loss_limit():
            return "daily_loss_stop"
        return None

    # ── positions ────────────────────────────────────────────────────────────
    def positions(self) -> list[ValidatedPosition]:
        return list(self._pos.values())

    def overnight_holdings(self) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        for p in self._pos.values():
            if p.overnight:
                out[(p.symbol, p.product)] = out.get((p.symbol, p.product), 0) + p.sign * p.qty
        return out

    def _open(self, pt: PlannedTrade, product: str, exchange: str, fill_px: float, oid: str,
              overnight: bool = False, meta: Optional[dict] = None, group: str = "") -> ValidatedPosition:
        pos = ValidatedPosition(oid=str(oid), agent=self.name, symbol=pt.symbol, exchange=exchange, product=product,
                                side="BUY" if pt.side > 0 else "SELL", qty=int(pt.qty), entry_px=float(fill_px),
                                entry_ref=float(pt.entry_ref), opened=self._now().isoformat(timespec="seconds"),
                                overnight=overnight, group=group, meta=dict(meta or {}))
        self._pos[pos.oid] = pos
        self.state.trades_today += 1
        self.state.signals_fired += 1
        self.state.last_signal = {"symbol": pos.symbol, "side": pos.side, "qty": pos.qty, "px": pos.entry_px,
                                  "t": pos.opened}
        self._save_book()
        return pos

    def _close(self, pos: ValidatedPosition, exit_px: float, note: str, kind: str = "normal") -> float:
        """The paper exit fill is already done: book the ledger once, update today's P&L, persist."""
        from performance import performance
        pnl = round(pos.sign * (float(exit_px) - pos.entry_px) * pos.qty, 2)
        extra = {"lab_id": self.LAB_ID, "exit_kind": kind}
        row = performance.record_trade(self.name, pos.symbol, pos.side, pos.qty, pos.entry_px, pnl, pos.exchange,
                                       pos.product, exit_price=float(exit_px), note=note, signal_price=pos.entry_ref,
                                       order_id=pos.oid, cost_kind=self.COST_KIND, charge_risk=False, extra=extra)
        net = float(row["net"])
        self.state.pnl_today += net
        self._pos.pop(pos.oid, None)
        logger.info("[{}] closed {} {} x{} {:.2f} → {:.2f} net ₹{:,.2f} ({})", self.name, pos.side, pos.symbol,
                    pos.qty, pos.entry_px, exit_px, net, note)
        self._save_book()
        return net

    @staticmethod
    def _save_book() -> None:
        try:
            import paper_book
            paper_book.save(force=True)
        except Exception as exc:
            logger.debug("[validated] paper book save failed: {}", exc)

    # ── persistence (paper_book) ────────────────────────────────────────────
    def export_book(self) -> dict:
        return {"version": self.VERSION, "positions": [asdict(p) for p in self._pos.values()],
                "day": dict(self._day), "day_log": list(self.day_log), "kill": self._kill,
                "dd_since": self._dd_since, "deviations": dict(self.deviations)}

    def import_book(self, d: dict, closed: set, stale: bool) -> None:
        """Restore positions (those the ledger already closed are dropped, and their paper row flattened at the
        ledger's exit price if the snapshot still shows it open) and, for a same-day snapshot, today's state."""
        d = d or {}
        self._kill = d.get("kill") or self._kill
        self._dd_since = d.get("dd_since") or self._dd_since
        for k, v in (d.get("deviations") or {}).items():
            self.deviations[k] = max(self.deviations.get(k, 0), int(v))
        if not stale and (d.get("day") or {}).get("date") == self._today().isoformat():
            self._day = dict(d["day"])
            self.day_log = list(d.get("day_log") or [])[-DAY_LOG_MAX:]
        restored = []
        for raw in d.get("positions") or []:
            try:
                pos = ValidatedPosition(**raw)
            except TypeError as exc:
                logger.error("[{}] malformed position in the snapshot ignored: {}", self.name, exc)
                continue
            if pos.oid in self._pos:
                continue
            if pos.oid in closed:
                self._flatten_closed_row(pos)
                continue
            self._pos[pos.oid] = pos
            restored.append(pos)
        if restored:
            logger.warning("[{}] restored {} open position(s): {}", self.name, len(restored),
                           ", ".join(f"{p.side} {p.symbol} x{p.qty}" for p in restored))
        if stale and restored:
            self._on_stale_restore(restored)

    def _flatten_closed_row(self, pos: ValidatedPosition) -> None:
        """The ledger holds the exit but the snapshot (written before it) still has the paper row open."""
        row = paper_row(pos.symbol, pos.product)
        q = int((row or {}).get("quantity") or 0)
        if q * pos.sign < pos.qty:
            return
        from performance import performance
        led = next((t for t in reversed(performance.trades) if t.get("oid") == pos.oid), {})
        px = float(led.get("exit") or row.get("last_price") or pos.entry_px)
        _kite().paper_fill_at(pos.symbol, pos.exchange, "SELL" if pos.sign > 0 else "BUY", pos.qty, pos.product,
                              px, tag="VAL-RECON", fill_kind="RECONCILE")
        logger.warning("[{}] {} already in the ledger: paper row flattened at ₹{:.2f}", self.name, pos.symbol, px)

    def _on_stale_restore(self, positions: list[ValidatedPosition]) -> None:
        """Positions restored from an earlier session's snapshot (agent-specific)."""
        return None

    # ── status ───────────────────────────────────────────────────────────────
    def live_stats(self) -> dict:
        rows = self.ledger_rows()
        n = len(rows)
        net = sum(t["net"] for t in rows)
        gp = sum(t["net"] for t in rows if t["net"] > 0)
        gl = -sum(t["net"] for t in rows if t["net"] <= 0)
        bps = [t["net"] / (t["entry"] * t["qty"]) * 1e4 for t in rows if t.get("entry") and t.get("qty")]
        return {"trades": n, "net": round(net, 2), "pf": round(gp / gl, 2) if gl > 0 else None,
                "win_pct": round(100 * sum(t["net"] > 0 for t in rows) / n, 1) if n else None,
                "net_bps": round(sum(bps) / len(bps), 2) if bps else None}

    def live_readiness(self) -> dict:
        return {}

    def _mark(self, p: ValidatedPosition) -> Optional[float]:
        row = paper_row(p.symbol, p.product)
        px = float((row or {}).get("last_price") or 0.0)
        return round(p.sign * (px - p.entry_px) * p.qty, 2) if px > 0 else None

    def get_status(self) -> dict:
        st = super().get_status()
        st.update({
            "validated": True, "lab_id": self.LAB_ID, "version": self.VERSION, "fingerprint": self.LAB_FINGERPRINT,
            "enabled": self.enabled_by_setting(), "entry_block": self.entry_block_reason_readonly(),
            "positions": [{**asdict(p), "unrealised": self._mark(p)} for p in self._pos.values()],
            "day": self._day, "day_log": self.day_log[-100:], "live": self.live_stats(),
            "lab_expectation": self.LAB_EXPECTATION, "deviations": dict(self.deviations),
            "drawdown": self.drawdown(), "drawdown_kill": self._kill, "realised_today": self.realised_today(),
            "live_readiness": self.live_readiness(),
        })
        return st

    def entry_block_reason_readonly(self) -> Optional[str]:
        """entry_block_reason without side effects (status must never fire the drawdown kill)."""
        import bot_state
        if settings.trading_mode != "PAPER":
            return "live_mode_refused"
        if not settings.uses_kite_market_data:
            return "no_kite_data"
        if not getattr(settings, "validated_only", True):
            return "validated_only_off"
        if not self.enabled_by_setting():
            return "kill_switch"
        if self._kill and not bot_state.is_agent_enabled(self.name):
            return "drawdown_kill"
        if not bot_state.is_agent_enabled(self.name):
            return "disabled"
        if not self.state.running:
            return "stopped"
        if self.drawdown() >= self.dd_limit():
            return "drawdown_kill"
        if self.realised_today() <= -self.daily_loss_limit():
            return "daily_loss_stop"
        return None
