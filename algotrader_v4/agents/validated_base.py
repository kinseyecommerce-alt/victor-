"""
agents/validated_base.py — base for the agents that trade a lab-validated rule (pre-registered stratlab protocol:
DISCOVERY H1/H2 screen + one FINAL look) as PAPER agents.

Parity first: the agent applies exactly the lab rule and nothing else. Its rule functions and constants are imported
from the lab module (never copied); LAB_FINGERPRINT pins them, so any change to the lab code forces a VERSION bump.
The legacy guards (trade gate, debate, MTF, Kelly, allocator, risk_manager limits, order_guard, pruning, trailing
stops) are not applied: none of them was part of the tested rule. SEBI pre-order checks are. Per-agent overlays
(daily loss stop, drawdown kill, kill switch) only ever block ENTRIES; positions always exit by their rule.

Driven by platform_scheduler jobs (scheduled = True), never by the tick engine. PAPER only: in LIVE every job is
refused and kite_client.paper_fill_at raises.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
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
    entry_px: float          # fill
    entry_ref: float         # the price the rule saw
    opened: str              # ISO IST datetime
    overnight: bool          # True: attention CNC, strangle legs
    group: str = ""          # strangle structure id
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedTrade:
    symbol: str
    side: int                # +1 BUY, -1 SELL
    qty: int
    entry_ref: float
    entry_t: str
    exit_t: str
    key: float               # ranking key (|gap| / ratio)


def fingerprint(*objs, consts: Optional[dict] = None) -> str:
    """sha1 of the lab functions' source plus the constants: a lab change shows up as a new fingerprint."""
    h = hashlib.sha1()
    for o in objs:
        h.update(inspect.getsource(o).encode())
    for k in sorted(consts or {}):
        h.update(f"{k}={consts[k]!r}".encode())
    return h.hexdigest()[:16]


# ── order helpers (PAPER) ─────────────────────────────────────────────────────
def paper_marketable(symbol: str, exchange: str, side: str, qty: int, product: str, quote: dict,
                     tag: str) -> tuple[str, float]:
    """A marketable LIMIT at the quote touch (BUY → ask, SELL → bid; the LTP when the touch is 0). The paper broker
    fills it at its own tick touch when it has one, else at this price. Returns (order id, fill price)."""
    from kite_client import kite_client
    if settings.trading_mode != "PAPER":
        raise RuntimeError("validated agents are PAPER only")
    q = quote or {}
    px = float((q.get("ask") if side == "BUY" else q.get("bid")) or 0) or float(q.get("ltp") or 0)
    if px <= 0:
        raise ValueError(f"{symbol}: no price in the quote")
    oid = kite_client.place_order(tradingsymbol=symbol, exchange=exchange, transaction_type=side, quantity=qty,
                                  order_type="LIMIT", product=product, price=round(px, 2), tag=tag)
    fill = kite_client.fill_price(oid)
    return oid, float(fill if fill else px)


def paper_fill_at(symbol: str, exchange: str, side: str, qty: int, product: str, price: float, tag: str,
                  fill_kind: str) -> tuple[str, float]:
    from kite_client import kite_client
    oid = kite_client.paper_fill_at(symbol, exchange, side, qty, product, price, tag, fill_kind)
    return oid, float(price)


# ── the agent base ────────────────────────────────────────────────────────────
class ValidatedAgent(BaseAgent):
    scheduled = True                  # driven by platform_scheduler jobs, never by tick_engine
    validated_by_replay = True
    strategy_rule_exits = False
    VERSION: str = "validated/v1"
    LAB_ID: str = ""
    LAB_EXPECTATION: dict = {}
    LAB_FINGERPRINT: str = ""
    COST_KIND: Optional[str] = None   # None: performance.trade_cost by exchange/product
    SETTING: str = ""                 # settings.validated_<SETTING>_enabled / _daily_loss_stop / _dd_kill

    def __init__(self) -> None:
        super().__init__()
        self._positions: dict[str, ValidatedPosition] = {}
        self.day_log: list[dict] = []
        self._day_log_day: Optional[str] = None
        self.days_done: dict[str, str] = {}       # stage → ISO date it last ran to completion (entry once a day)
        self.flags: dict[str, str] = {}           # day flags (ca_calendar_unavailable ...) keyed "flag" → date
        self.deviations: dict[str, int] = {}
        self._dd_killed: Optional[dict] = None    # {"at": iso, "dd": x} while the drawdown kill holds
        self._dd_since: str = ""                  # ledger rows before this (a manual re-enable) are not counted

    # ── clock ──
    def _now(self) -> datetime:
        mod = sys.modules.get(type(self).__module__)
        fn = getattr(mod, "now_ist", None) or now_ist
        return fn()

    # ── BaseAgent hooks that do not apply ──
    def evaluate_tick(self, snap):
        return "HOLD", None

    def should_exit_position(self, position, ind):
        return False, ""

    def start(self, queue=None) -> None:
        self.state.running = True
        logger.info("[{}] ready (scheduled: {})", self.name, self.VERSION)

    def stop(self) -> None:
        self.state.running = False               # no new entries; open positions still exit by their rule
        logger.info("[{}] stopped — no new entries (open positions still exit by rule)", self.name)

    # ── limits ──
    def _setting(self, suffix: str, default=None):
        return getattr(settings, f"validated_{self.SETTING}_{suffix}", default)

    def _scale(self) -> float:
        return float(getattr(settings, "validated_notional_scale", 1.0) or 1.0)

    def daily_loss_limit(self) -> float:
        return float(self._setting("daily_loss_stop", 0.0) or 0.0) * self._scale()

    def dd_limit(self) -> float:
        return float(self._setting("dd_kill", 0.0) or 0.0) * self._scale()

    def entry_block_reason(self) -> Optional[str]:
        import bot_state
        if settings.trading_mode != "PAPER":
            return "live_mode_refused"
        if not settings.uses_kite_market_data:
            return "no_kite_data"
        if not getattr(settings, "validated_only", False):
            return "validated_only_off"
        if not self._setting("enabled", True):
            return "kill_switch"
        enabled = bot_state.is_agent_enabled(self.name)
        if self._dd_killed is not None:
            if not enabled:
                return "drawdown_kill"
            # switched back on by hand after a kill: measure the drawdown from here
            logger.warning("[{}] re-enabled after the drawdown kill ({}) — drawdown measured afresh",
                           self.name, self._dd_killed)
            self._dd_killed = None
            self._dd_since = self._now().isoformat(timespec="seconds")
        if not enabled:
            return "disabled"
        if not self.state.running:
            return "stopped"
        lim = self.dd_limit()
        if lim > 0 and self.drawdown() >= lim:
            self._dd_killed = {"at": self._now().isoformat(timespec="seconds"), "dd": round(self.drawdown(), 2)}
            bot_state.set_agent_enabled(self.name, False)
            logger.error("[{}] DRAWDOWN KILL: live drawdown ₹{:,.0f} ≥ ₹{:,.0f} — entries off until re-enabled",
                         self.name, self.drawdown(), lim)
            return "drawdown_kill"
        lim = self.daily_loss_limit()
        if lim > 0 and self.realised_today() <= -lim:
            return "daily_loss_stop"
        return None

    # ── logging ──
    def _log(self, stage: str, symbol: str, outcome: str, reason: str = "") -> None:
        now = self._now()
        d = now.date().isoformat()
        if self._day_log_day != d:
            self.day_log, self._day_log_day = [], d
        if len(self.day_log) < DAY_LOG_MAX:
            self.day_log.append({"t": now.strftime("%H:%M:%S"), "stage": stage, "symbol": symbol,
                                 "outcome": outcome, "reason": reason})

    def _skip(self, sym: str, reason: str, stage: str = "entry", log: bool = True) -> None:
        """Every candidate / batch that does not become an order says why (counted; logged unless log=False)."""
        key = reason[:40] if reason.startswith("no_data:") else reason.split(":")[0][:40]
        self.state.skip_counts[key] = self.state.skip_counts.get(key, 0) + 1
        if log:
            self.state.last_skip = f"{sym}: {reason}"[:160]
            logger.info("[{}] {} {} skipped — {}", self.name, stage, sym, reason)
        self._log(stage, sym, "skip", reason)

    def _deviation(self, kind: str, symbol: str, detail: str = "") -> None:
        self.deviations[kind] = self.deviations.get(kind, 0) + 1
        logger.warning("[{}] deviation {} {} {}", self.name, kind, symbol, detail)
        self._log("deviation", symbol, kind, detail)

    def _error(self, where: str, exc: Exception) -> None:
        msg = f"{where}: {str(exc)[:200]}"
        self.state.errors.append(msg)
        self.state.errors[:] = self.state.errors[-20:]
        logger.exception("[{}] {}", self.name, msg)

    # ── positions ──
    def positions(self) -> list[ValidatedPosition]:
        return list(self._positions.values())

    def overnight_holdings(self) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        for p in self._positions.values():
            if p.overnight:
                k = (p.symbol, p.product)
                out[k] = out.get(k, 0) + p.qty * (1 if p.side == "BUY" else -1)
        return out

    def _open(self, pt: PlannedTrade, product: str, exchange: str, fill_px: float, oid: str, overnight: bool,
              meta: Optional[dict] = None) -> ValidatedPosition:
        pos = ValidatedPosition(oid=oid, agent=self.name, symbol=pt.symbol, exchange=exchange, product=product,
                                side="BUY" if pt.side > 0 else "SELL", qty=int(pt.qty), entry_px=float(fill_px),
                                entry_ref=float(pt.entry_ref), opened=self._now().isoformat(timespec="seconds"),
                                overnight=overnight, meta=dict(meta or {}))
        self._positions[oid] = pos
        self.state.trades_today += 1
        self.state.signals_fired += 1
        self._log("entry", pt.symbol, "open", f"{pos.side} {pos.qty} @ {fill_px:.2f}")
        self._save_book()
        return pos

    def _ledger_extra(self, pos: ValidatedPosition, exit_px: float) -> dict:
        return {}

    def _close(self, pos: ValidatedPosition, exit_px: float, note: str, kind: str = "normal") -> float:
        """The paper exit fill is already done: book the ledger once (with the entry order id), the agent's day P&L,
        and write the paper book now."""
        from performance import performance
        if pos.oid not in self._positions:
            return 0.0
        sign = 1 if pos.side == "BUY" else -1
        pnl = round((float(exit_px) - pos.entry_px) * pos.qty * sign, 2)
        extra = {"lab_id": self.LAB_ID, **self._ledger_extra(pos, exit_px)}
        if kind != "normal":
            extra["deviation"] = kind
        row = performance.record_trade(self.name, pos.symbol, pos.side, pos.qty, pos.entry_px, pnl, pos.exchange,
                                       pos.product, exit_price=float(exit_px), note=note,
                                       signal_price=pos.entry_ref, order_id=pos.oid, cost_kind=self.COST_KIND,
                                       charge_risk=False, extra=extra)
        net = float(row.get("net", pnl))
        self._positions.pop(pos.oid, None)
        self.state.pnl_today += net
        self._log("exit", pos.symbol, "closed", f"{note} @ {float(exit_px):.2f} net {net:.2f}")
        self._save_book(force=True)
        return net

    @staticmethod
    def _save_book(force: bool = False) -> None:
        try:
            import paper_book
            paper_book.save(force=force)
        except Exception as exc:
            logger.debug("[validated] paper book save failed: {}", exc)

    # ── ledger views ──
    def ledger_rows(self) -> list[dict]:
        from performance import performance
        return [t for t in list(performance.trades) if t.get("strategy") == self.name
                and t.get("version") == self.VERSION and str(t.get("t", "")) >= self._dd_since]

    def realised_today(self) -> float:
        d = self._now().date().isoformat()
        return round(sum(float(t.get("net", 0)) for t in self.ledger_rows() if str(t.get("t", ""))[:10] == d), 2)

    def drawdown(self) -> float:
        """Current drawdown of the live net P&L (this VERSION) from its running peak (start = 0)."""
        cum = peak = 0.0
        for t in sorted(self.ledger_rows(), key=lambda r: str(r.get("t", ""))):
            cum += float(t.get("net", 0))
            peak = max(peak, cum)
        return round(peak - cum, 2)

    def live_stats(self, rows=None, net_key: str = "net") -> dict:
        rows = self.ledger_rows() if rows is None else rows
        n = len(rows)
        if not n:
            return {"n": 0}
        nets = [float(r.get(net_key, 0)) for r in rows]
        gp, gl = sum(x for x in nets if x > 0), -sum(x for x in nets if x < 0)
        bps = [x / (float(r.get("entry", 0)) * float(r.get("qty", 0))) * 1e4 for x, r in zip(nets, rows)
               if float(r.get("entry", 0)) * float(r.get("qty", 0)) > 0]
        return {"n": n, "net": round(sum(nets), 2), "pf": round(gp / gl, 2) if gl else None,
                "win_pct": round(100 * sum(x > 0 for x in nets) / n, 1),
                "net_bps": round(sum(bps) / len(bps), 2) if bps else None}

    # ── persistence (paper_book "validated" block) ──
    def export_book(self) -> dict:
        return {"version": self.VERSION, "positions": [asdict(p) for p in self._positions.values()],
                "days_done": dict(self.days_done), "flags": dict(self.flags), "deviations": dict(self.deviations),
                "day_log_day": self._day_log_day, "day_log": list(self.day_log[-DAY_LOG_MAX:]),
                "dd_killed": self._dd_killed, "dd_since": self._dd_since}

    def import_book(self, d: dict, closed: set, stale: bool) -> None:
        d = d or {}
        for row in d.get("positions") or []:
            pos = ValidatedPosition(**row)
            if pos.oid in closed:                # already in the ledger (closed after the snapshot)
                logger.info("[{}] {} already booked — not restored", self.name, pos.symbol)
                continue
            self._positions.setdefault(pos.oid, pos)
        self.days_done.update(d.get("days_done") or {})
        self.flags.update(d.get("flags") or {})
        for k, v in (d.get("deviations") or {}).items():
            self.deviations[k] = max(self.deviations.get(k, 0), int(v))
        if d.get("day_log_day") and not stale:
            self._day_log_day, self.day_log = d["day_log_day"], list(d.get("day_log") or [])
        self._dd_killed = d.get("dd_killed") or self._dd_killed
        self._dd_since = d.get("dd_since") or self._dd_since
        if self._positions:
            logger.warning("[{}] restored {} validated position(s): {}", self.name, len(self._positions),
                           ", ".join(p.symbol for p in self._positions.values()))

    def done_today(self, stage: str) -> bool:
        return self.days_done.get(stage) == self._now().date().isoformat()

    def mark_done(self, stage: str) -> None:
        self.days_done[stage] = self._now().date().isoformat()
        self._save_book(force=True)

    # ── status ──
    def live_readiness(self) -> dict:
        return {}

    def get_status(self) -> dict:
        st = super().get_status()
        block = None
        try:
            block = self.entry_block_reason()
        except Exception as exc:
            block = f"error: {exc}"
        st.update({
            "lab_id": self.LAB_ID, "version": self.VERSION, "fingerprint": self.LAB_FINGERPRINT,
            "scheduled": True, "entries_blocked": block,
            "positions": [asdict(p) for p in self._positions.values()],
            "day_log": self.day_log[-100:], "deviations": dict(self.deviations), "flags": dict(self.flags),
            "live": self.live_stats(), "lab_expectation": self.LAB_EXPECTATION,
            "realised_today": self.realised_today(), "drawdown": self.drawdown(),
            "limits": {"daily_loss_stop": self.daily_loss_limit(), "dd_kill": self.dd_limit()},
            "dd_killed": self._dd_killed, "live_readiness": self.live_readiness(),
        })
        return st
